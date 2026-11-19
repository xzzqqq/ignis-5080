#!/usr/bin/env python3
"""Pinned, single-owner lab control. Unix socket transport, no task queue."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import select
import socket
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_support.common import atomic, read, digest, key, lock, status, identity, alive, source_identity, host_changes, GPU_LOCK, validate_config, deadline, LabTimeout
from kernel_lab import Session
from lab_support.iteration import Iteration
from build_trellis_candidate import build as build_candidate


def run(argv, log, env=None, cwd=None):
    began=time.monotonic()
    print('lab: '+Path(log).stem, file=sys.stderr, flush=True)
    with Path(log).open('w') as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    print(f'lab: {Path(log).stem} {time.monotonic()-began:.3f}s', file=sys.stderr, flush=True)
    if p.returncode:
        raise ValueError(f'command failed ({p.returncode}); see {log}')


def checked(reply):
    if not reply.get('ok'):
        raise RuntimeError(reply.get('error', 'engine operation failed'))
    return reply['result']


def corpus_verify(directory, expected):
    manifest = read(directory / 'manifest.json')
    if not manifest or manifest.get('identity') != expected or not manifest.get('complete'):
        raise ValueError('incomplete/incompatible corpus: ' + str(directory))
    for name in ('data.bin', 'records.jsonl'):
        if digest(directory / name) != manifest['files'][name]:
            raise ValueError('corrupt corpus: ' + name)
    return manifest


def case_agrees(a, b):
    return all(a.get(k) == b.get(k) for k in ('tokens','input_tokens','finish_reason','stop_token','logits_sha256')) and [
        (r['emitted'],r['extent']) for r in a['rounds']] == [(r['emitted'],r['extent']) for r in b['rounds']]


def filesystem_type(path):
    """Read the mount covering the resolved path, including nested bind mounts."""
    if sys.platform != 'linux':
        return None
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        mount, filesystem = line.split(' - ', 1)
        name = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), mount.split()[4])
        point = Path(name)
        if path.is_relative_to(point):
            matches.append((len(point.parts), filesystem.split()[0]))
    if not matches:
        raise ValueError('cannot determine build filesystem: ' + str(path))
    return max(matches, key=lambda match: match[0])[1]


def require_disk(path):
    resolved = Path(path).resolve()
    filesystem = filesystem_type(resolved)
    if filesystem in ('tmpfs', 'ramfs'):
        raise ValueError(f'Lab build path {path} resolves to {resolved} on {filesystem}; '
                         'set build_root to a disk-backed directory (prefer a local SSD). '
                         'Existing kernel/build links and cache subdirectories must also be disk-backed; '
                         'migrate inactive caches and update links before retrying. No files were moved.')


def prepare_build(root, build_root):
    """Validate all compiler destinations before creating directories or building."""
    native = root/'kernel/build'
    paths = [build_root, *(build_root/name for name in ('target', 'tmp', 'kernel', 'candidates'))]
    existing_native = native.exists() or native.is_symlink()
    for path in [*paths, *([native] if existing_native else [])]:
        require_disk(path)
    if native.is_symlink() and not native.exists():
        raise ValueError(f'kernel/build has a broken link: {native}; repair it to a disk-backed cache before retrying')
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    if not existing_native:
        native.symlink_to(build_root/'kernel', target_is_directory=True)
    return dict(os.environ, CARGO_TARGET_DIR=str(build_root/'target'), TMPDIR=str(build_root/'tmp'))


class Owner(Iteration):
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.config = state['config']
        self.directory = Path(state['directory'])
        self.engine = None
        self.state_file = root / '.scratch/lab/state.json'

    @property
    def build_root(self):
        return Path(self.config.get('build_root', str(Path(self.config['data_dir'])/'build'))).resolve()

    def publish(self, **fields):
        current=read(self.state_file)
        if current and current['session']!=self.state['session']:
            raise RuntimeError('session was replaced; refusing stale state update')
        self.state.update(fields)
        atomic(self.directory/'session.json', self.state)
        atomic(self.state_file, self.state)

    def setup(self):
        cfg = self.config
        validate_config(cfg)
        if self.state.get('profiling_enabled',False) and not shutil.which('nsys'):
            raise ValueError('--enable-profiling requires Nsight Systems (nsys) in the remote PATH')
        if int(cfg.get('warmup',1))<1 or int(cfg.get('samples',3))<1:
            raise ValueError('warmup/samples must be positive')
        for k in ('artifact', 'data_dir'):
            if not Path(cfg[k]).is_absolute():
                raise ValueError(k + ' must be an absolute remote path')
        build_root = self.build_root
        env = prepare_build(self.root, build_root)
        env['IGNIS_KERNEL_BUILD_JOBS'] = str(cfg.get('build_jobs',4))
        cargo = ['cargo']
        if (self.root/'.remote.cargo.toml').is_file():
            cargo += ['--config',str(self.root/'.remote.cargo.toml')]
        baseline = Path(cfg['data_dir'])/'baselines'/key(self.state['source'])
        if self.state.get('baseline'):
            baseline = Path(self.state['baseline'])
        else:
            baseline.mkdir(parents=True,exist_ok=True)
            if not (baseline/'complete.json').is_file():
                run(cargo+['build','--locked','--release','-p','ignis-core','--features','cuda','--example','kernel_lab'],
                    self.directory/'build-engine.log',env,self.root)
                run(cargo+['build','--locked','--release','-p','ignis-server','--example','lab_prepare'],
                    self.directory/'build-prepare.log',env,self.root)
                if source_identity(self.root) != self.state['source']:
                    raise ValueError('source changed during baseline build')
                binaries=build_root/'target/x86_64-unknown-linux-gnu/release/examples'
                for name in ('kernel_lab','lab_prepare'):
                    shutil.copy2(binaries/name,baseline/name)
                revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
                patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=self.root)
                (baseline/'source.patch').write_bytes(patch)
                atomic(baseline/'complete.json',dict(source=self.state['source'],revision=revision,
                    binaries={n:digest(baseline/n) for n in ('kernel_lab','lab_prepare')}))
        base = read(baseline/'complete.json')
        if not base or any(digest(baseline/n)!=sha for n,sha in base['binaries'].items()):
            raise ValueError('pinned baseline is absent or changed')
        self.publish(baseline=str(baseline), source=base['source'])
        prepared=self.directory/'inputs.json'
        run([str(baseline/'lab_prepare'),cfg['artifact'],str(prepared)],self.directory/'prepare.log')
        self.cases=read(prepared)['cases']
        for i,c in enumerate(self.cases):
            c['request_index']=i
            if 'diagnostic_output_tokens' in cfg:
                c['output_tokens']=int(cfg['diagnostic_output_tokens']);c['diagnostic']=True
                if not 0<c['output_tokens']<=8192:raise ValueError('invalid diagnostic budget')
        selected=cfg.get('benchmark','astropy__astropy-12907')
        self.quick_case=next(c for c in self.cases if c['id']==selected)
        atomic(prepared,dict(version=1,cases=self.cases))
        settings=dict(max_context=65536,prefill_chunk=1024,lanes=2,kv_format='hq-e8-2b',speculation='dflash2')
        settings.update(cfg.get('settings',{}))
        atomic(self.directory/'settings.json',settings)
        run(['make','gpu-status'],self.directory/'gpu-before.log',cwd=self.root)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if apps:raise ValueError('GPU already has compute processes')
        machine=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
        if len(machine.splitlines())!=1 or 'RTX 5080' not in machine:
            raise ValueError('lab requires exactly one RTX 5080')
        self.engine=Session(baseline/'kernel_lab',Path(cfg['artifact']),self.directory/'settings.json',
                            self.directory,float(cfg.get('setup_timeout_s',3600)),
                            profiler_session='ignis-lab-'+self.state['session'] if self.state.get('profiling_enabled',False) else None)
        atomic(self.directory/'engine.identity.json',identity(getattr(self.engine,'engine_pid',self.engine.process.pid)))
        self.corpus_identity=dict(version=3,benchmark=selected,engine=base['binaries']['kernel_lab'],artifact=digest(cfg['artifact']),
            prepared=digest(prepared),settings=settings,machine=machine,
            environment={k:os.environ.get(k) for k in ('IGNIS_TRELLIS_SM_CAP','CUDA_VISIBLE_DEVICES')})
        self.corpus=Path(cfg['data_dir'])/'corpora'/key(self.corpus_identity)
        self.corpus.mkdir(parents=True,exist_ok=True)
        self.references={}
        self.publish(status='starting',benchmark=selected,engine_pid=getattr(self.engine,'engine_pid',self.engine.process.pid),
            corpus=str(self.corpus),settings=settings,ready=self.engine.ready,
            profiler_session=getattr(self.engine,'profiler_session',None))


def relay_progress(path, offset):
    try:
        with Path(path).open() as stream:
            stream.seek(offset)
            for line in stream:
                if line.startswith('lab:'):print(line,end='',file=sys.stderr,flush=True)
            return stream.tell()
    except FileNotFoundError:return offset


def start(root,owner,payload):
    cfg=payload['config'];validate_config(cfg)
    profiling_enabled=payload.get('profiling_enabled',False)
    if type(profiling_enabled) is not bool:raise ValueError('profiling_enabled must be a boolean')
    if profiling_enabled and not shutil.which('nsys'):
        raise ValueError('--enable-profiling requires Nsight Systems (nsys) in the remote PATH')
    linger=subprocess.check_output(['loginctl','show-user',str(os.getuid()),'--property=Linger','--value'],text=True).strip()
    if linger!='yes':raise ValueError('systemd user lingering is required; configure loginctl enable-linger before setup')
    directory=Path(cfg['data_dir'])/'sessions'/uuid.uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    with lock(root/'.scratch/lab/start.lock'):
        previous=status(root)
        if previous and previous['status']!='stopped':raise ValueError('lab already active')
        runtime=directory/'runtime';(runtime/'lab_support').mkdir(parents=True)
        for name in ('common.py','host.py','iteration.py','profiling.py','forward_timeline.py','__init__.py'):
            shutil.copy2(Path(__file__).parent/name,runtime/'lab_support'/name)
        for name in ('kernel_lab.py','build_trellis_candidate.py'):
            shutil.copy2(root/'scripts'/name,runtime/name)
        env=runtime/'.remote.env'
        if (root/'.remote.env').exists():shutil.copy2(root/'.remote.env',env)
        else:env.write_text('')
        session=directory.name
        unit='ignis-lab-'+session
        socket_dir=Path(os.environ.get('XDG_RUNTIME_DIR',f'/tmp/ignis-lab-{os.getuid()}'))
        socket_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
        state=dict(root=str(root),owner=owner,session=session,status='starting',config=cfg,config_hash=key(cfg),
            source=payload['source'],directory=str(directory),control=str(runtime/'lab_support/host.py'),
            socket=str(socket_dir/('ignis-lab-'+session[:16]+'.sock')),unit=unit,process=None,
            profiling_enabled=profiling_enabled)
        if previous and not payload['new_baseline']:
            if previous['config_hash']!=key(cfg):raise ValueError('configuration change requires --new-baseline')
            if previous.get('baseline'):state['baseline']=previous['baseline'];state['source']=previous['source']
        if source_identity(root)!=payload['source']:raise ValueError('initial sync/source mismatch')
        atomic(directory/'session.json',state)
        atomic(root/'.scratch/lab/state.json',state)
        activate='set -e; source "$1"; exec /usr/bin/python3 "$2" --root "$3" serve --owner "$4"'
        argv=['systemd-run','--user','--quiet','--collect','--unit='+unit,
              '--working-directory='+str(root),'--property=Type=exec','--property=KillMode=control-group',
              '--property=TimeoutStopSec=10','--property=StandardOutput=append:'+str(directory/'owner.log'),
              '--property=StandardError=append:'+str(directory/'owner.log'),
              '/bin/bash','--noprofile','--norc','-c',activate,'ignis-lab',str(env),state['control'],str(root),owner]
        subprocess.run(argv,check=True,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    deadline=time.monotonic()+float(cfg.get('setup_timeout_s',3600))
    log_offset=0
    while time.monotonic()<deadline:
        log_offset=relay_progress(directory/'owner.log',log_offset)
        state=read(root/'.scratch/lab/state.json')
        if not state or state.get('session')!=session:
            raise ValueError('setup session replaced or missing')
        if state['status']=='stopped':raise ValueError('setup cancelled: session stopped')
        if state['status']=='ready':return state
        if state['status']=='failed':raise ValueError(state.get('error','setup failed'))
        time.sleep(.2)
    subprocess.run(['systemctl','--user','stop',unit],check=False)
    raise ValueError('setup timed out; unit stopped')


def serve(root,owner):
    state=read(root/'.scratch/lab/state.json')
    if state['owner']!=owner:raise ValueError('owner mismatch')
    service=Owner(root,state)
    service.publish(process=identity(os.getpid()))
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        with lock(GPU_LOCK) as lease:
            os.environ['IGNIS_GPU_LEASE_FD']=str(lease.fileno())
            with deadline(service.config.get('setup_timeout_s',3600)):
                service.setup()
            service.engine.timeout=float(service.config.get('timeout_s',1800))
            listener.bind(state['socket']);os.chmod(state['socket'],0o600);listener.listen(1)
            # Publish ready only once the control socket also exists.
            service.publish(status='ready')
            while True:
                connection,_=listener.accept()
                with connection:
                    request=json.loads(connection.makefile('r').readline())
                    if request.get('owner')!=owner or request.get('session')!=state['session']:
                        response=dict(passed=False,error='foreign session')
                    else:
                        try:
                            with deadline(service.config.get('timeout_s',1800)):
                                response=service.evaluate(request,request['action'])
                        except ValueError as e:
                            response=dict(passed=False,error=str(e))
                    try:connection.sendall((json.dumps(response)+'\n').encode())
                    except (BrokenPipeError,ConnectionResetError):pass
    except BaseException as e:
        service.publish(status='failed',error=str(e),active=None)
        raise
    finally:
        listener.close();Path(state['socket']).unlink(missing_ok=True)
        if service.engine:service.engine.close()


def control(root,owner,action,payload):
    state=read(root/'.scratch/lab/state.json')
    if not state:return dict(status='stopped')
    if state['owner']!=owner:raise ValueError('foreign owner')
    if action=='stop':
        # systemd owns the cgroup even if the owner is blocked or has died.
        load=subprocess.run(['systemctl','--user','show',state['unit'],'--property=LoadState','--value'],capture_output=True,text=True)
        if load.stdout.strip() not in ('','not-found'):
            subprocess.run(['systemctl','--user','stop',state['unit']],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        engine=read(Path(state['directory'])/'engine.identity.json')
        if alive(state.get('process')) or alive(engine):raise ValueError('managed processes still alive')
        if engine:
            deadline=time.monotonic()+10
            while True:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
                if str(engine['pid']) not in [x.strip() for x in apps]:break
                if time.monotonic()>=deadline:raise ValueError('owned GPU context has not been released')
                time.sleep(.2)
        stopped=dict(state,status='stopped',active=None)
        sock=Path(state['socket'])
        if sock.is_socket():sock.unlink()
        atomic(Path(state['directory'])/'session.json',stopped)
        atomic(root/'.scratch/lab/state.json',stopped)
        return dict(status='stopped',session=state['session'])
    if not alive(state.get('process')) or state['status']!='ready':raise ValueError('busy/not ready')
    if payload['config']!=state['config']:raise ValueError('config mismatch')
    payload=dict(payload,action=action,session=state['session'])
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    sock.settimeout(float(state['config'].get('timeout_s',1800))+60)
    with sock:
        log_path=Path(state['directory'])/'owner.log'
        log_offset=log_path.stat().st_size if log_path.exists() else 0
        sock.connect(state['socket']);sock.sendall((json.dumps(payload)+'\n').encode())
        response_deadline=time.monotonic()+float(state['config'].get('timeout_s',1800))+60
        while not select.select([sock],[],[],.2)[0]:
            log_offset=relay_progress(log_path,log_offset)
            if time.monotonic()>=response_deadline:
                raise ValueError('owner response deadline; inspect session logs; not retried')
        relay_progress(log_path,log_offset)
        line=sock.makefile('r').readline()
        if not line:
            result=read(Path(state['directory'])/'results'/payload['operation']/'result.json')
            if result and result.get('finished'):return result
            raise ValueError('owner exited; operation incomplete, inspect session logs; not retried')
        return json.loads(line)


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('action',choices=['start','serve','stop','replay','forward','compare','candidates','profile','export'])
    p.add_argument('--owner',required=True)
    args=p.parse_args()
    try:
        payload=json.load(sys.stdin) if args.action in ('start','replay','forward','compare','candidates','profile','export') else None
        if args.action=='serve':serve(args.root,args.owner);return
        result=start(args.root,args.owner,payload) if args.action=='start' else control(args.root,args.owner,args.action,payload)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error':str(e)}))
        raise SystemExit(2)

if __name__=='__main__':main()
