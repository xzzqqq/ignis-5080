#!/usr/bin/env python3
"""Pinned, single-owner lab control. Unix socket transport, no task queue."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_support.common import atomic, read, digest, key, lock, status, identity, alive, source_identity, host_changes, GPU_LOCK, validate_config, deadline, LabTimeout
from kernel_lab import Session
from build_trellis_candidate import build as build_candidate


def run(argv, log, env=None, cwd=None):
    with Path(log).open('w') as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
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


class Owner:
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.config = state['config']
        self.directory = Path(state['directory'])
        self.engine = None
        self.state_file = root / '.scratch/lab/state.json'

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
        if int(cfg.get('warmup',1))<1 or int(cfg.get('samples',3))<1:
            raise ValueError('warmup/samples must be positive')
        for k in ('artifact', 'data_dir'):
            if not Path(cfg[k]).is_absolute():
                raise ValueError(k + ' must be an absolute remote path')
        build_root = Path(cfg.get('build_root', str(Path(cfg['data_dir'])/'build')))
        build_root.mkdir(parents=True, exist_ok=True)
        (build_root/'tmp').mkdir(exist_ok=True)
        native = self.root/'kernel/build'
        (build_root/'kernel').mkdir(exist_ok=True)
        if not native.exists():
            native.symlink_to(build_root/'kernel', target_is_directory=True)
        env = dict(os.environ, CARGO_TARGET_DIR=str(build_root/'target'),TMPDIR=str(build_root/'tmp'),
                   IGNIS_KERNEL_BUILD_JOBS=str(cfg.get('build_jobs',4)))
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
                            self.directory,float(cfg.get('setup_timeout_s',3600)))
        atomic(self.directory/'engine.identity.json',identity(self.engine.process.pid))
        corpus_identity=dict(version=1,engine=base['binaries']['kernel_lab'],artifact=digest(cfg['artifact']),
            prepared=digest(prepared),settings=settings,machine=machine,
            capture_names=cfg.get('capture_names',[]),
            environment={k:os.environ.get(k) for k in ('IGNIS_TRELLIS_SM_CAP','CUDA_VISIBLE_DEVICES')})
        corpus=Path(cfg['data_dir'])/'corpora'/key(corpus_identity)
        if corpus.exists():
            manifest=corpus_verify(corpus,corpus_identity)
        else:
            corpus.parent.mkdir(parents=True,exist_ok=True)
            staging=corpus.with_name(corpus.name+'.partial-'+uuid.uuid4().hex)
            staging.mkdir()
            pin=int(cfg.get('capture_pin_bytes',1<<30));disk=int(cfg.get('capture_disk_bytes',512<<30))
            if pin<=0 or disk<=0:raise ValueError('capture budgets must be positive')
            minimum=checked(self.engine.call(dict(op='capture_minimum',names=','.join(cfg.get('capture_names',[])),
                tokens=sum(len(c['tokens']) for c in self.cases),requests=len(self.cases))))['prefill_bytes_lower_bound']
            atomic(self.directory/'capture-storage.json',dict(prefill_bytes_lower_bound=minimum,limit=disk,
                available=shutil.disk_usage(staging).free))
            if minimum>disk or minimum>shutil.disk_usage(staging).free:
                raise ValueError(f'capture needs at least {minimum} bytes for prefill alone; choose explicit scope/storage budget')
            checked(self.engine.call(dict(op='capture_start',path=str(staging),pin_bytes=pin,disk_bytes=disk,
                                          names=','.join(cfg.get('capture_names',[])))))
            references=[]
            for c in self.cases:
                references.append(checked(self.engine.call(dict(op='benchmark',case=c))))
            checked(self.engine.call(dict(op='capture_stop')))
            records=sum(1 for _ in (staging/'records.jsonl').open())
            if not records:raise ValueError('capture produced no calls')
            # Prove recording did not change the trajectory, with clean graphs.
            clean=[checked(self.engine.call(dict(op='benchmark',case=c))) for c in self.cases]
            if not all(case_agrees(a,b) for a,b in zip(references,clean)):
                raise ValueError('recording changed baseline trajectory')
            manifest=dict(identity=corpus_identity,complete=True,records=records,references=clean,
                files={n:digest(staging/n) for n in ('data.bin','records.jsonl')})
            atomic(staging/'manifest.json',manifest);staging.rename(corpus)
        self.corpus,self.manifest=corpus,manifest
        # References are persisted numerics only; no timing result is reused.
        current=[checked(self.engine.call(dict(op='benchmark',case=c))) for c in self.cases]
        if not all(case_agrees(a,b) for a,b in zip(manifest['references'],current)):
            raise ValueError('loaded corpus baseline differs from current engine')
        self.publish(status='starting',engine_pid=self.engine.process.pid,corpus=str(corpus),
            corpus_records=manifest['records'],settings=settings,ready=self.engine.ready)

    def records(self):
        view=self.config.get('view',{})
        for line in (self.corpus/'records.jsonl').open():
            r=json.loads(line)
            names=[r['weight']['name']]+([r['up']['name']] if r.get('up') else [])
            if view.get('names') and not any(n in view['names'] for n in names):continue
            if view.get('tokens') and r['tokens'] not in view['tokens']:continue
            if view.get('ids') and r['id'] not in view['ids']:continue
            yield r

    def local(self, directory, label):
        total=0.0;count=0;passed=True
        with (directory/(label+'-replay.jsonl')).open('w') as log:
            for r in self.records():
                result=checked(self.engine.call(dict(op='replay',data=str(self.corpus/'data.bin'),record=r,
                    warmup=int(self.config.get('warmup',1)),samples=int(self.config.get('samples',3)))))
                log.write(json.dumps(result)+'\n');log.flush();count+=1
                if not result['passed']:
                    passed=False;break
                total+=statistics.median(result['samples_ms'])
        if not count:raise ValueError('measurement view selects no records')
        return dict(passed=passed,records=count,sum_local_medians_ms=total,
                    interpretation='local replay aggregate only; not predicted full-forward time')

    def forward(self,directory,label):
        output=[]
        for reference,c in zip(self.manifest['references'],self.cases):
            rows=[]
            for i in range(int(self.config.get('warmup',1))+int(self.config.get('samples',3))):
                value=checked(self.engine.call(dict(op='benchmark',case=c)))
                if not case_agrees(reference,value):
                    return dict(passed=False,case=c['id'],mismatch=value)
                if i>=int(self.config.get('warmup',1)):rows.append(value)
            output.append(dict(id=c['id'],samples=rows,median_decode_ms=statistics.median(x['decode_ms'] for x in rows)))
        atomic(directory/(label+'-forward.json'),output)
        return dict(passed=True,cases=output)

    def evaluate(self,payload,action):
        actual=source_identity(self.root)
        if actual!=payload['source']:raise ValueError('sync/source identity mismatch; nothing measured')
        changes=host_changes(self.state['source'],actual)
        if changes:raise ValueError('host changes require new baseline: '+', '.join(changes))
        d=self.directory/'results'/payload['operation'];d.mkdir(parents=True,exist_ok=False)
        report=dict(passed=False,operation=payload['operation'],action=action,engine_pid=self.engine.process.pid,
                    source=actual,corpus=str(self.corpus),config_hash=self.state['config_hash'],finished=False,
                    settings=self.state.get('settings'),view=self.config.get('view',{}),
                    warmup=self.config.get('warmup',1),samples=self.config.get('samples',3),
                    diagnostic_output_tokens=self.config.get('diagnostic_output_tokens'))
        atomic(d/'result.json',report)
        self.publish(status='busy',active=payload['operation'],result=str(d/'result.json'))
        began=time.monotonic()
        try:
            try:
                run(['make','gpu-status'],d/'gpu-status.log',cwd=self.root)
            except Exception as e:
                raise RuntimeError('GPU preflight failed: '+str(e)) from e
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
            if any(x.strip()!=str(self.engine.process.pid) for x in apps if x.strip()):
                raise RuntimeError('foreign GPU process; refusing measurement')
            # Copy exactly the candidate compilation inputs; later sync cannot change them.
            import build_trellis_candidate as builder
            frozen=d/'source'
            files=set(self.root.glob('kernel/src/**/*.h'))|set(self.root.glob('kernel/src/**/*.cuh'))|set(self.root.glob('kernel/include/*.h'))
            files|={self.root/'kernel/src'/n for n in builder.SOURCES}
            for src in files:
                dst=frozen/src.relative_to(self.root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                if digest(dst)!=actual[str(src.relative_to(self.root))]:raise ValueError('source changed while copying candidate')
            try:
                candidate=build_candidate(frozen,Path(self.config['data_dir'])/'candidates',str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'))
            except LabTimeout:
                raise
            except Exception as e:
                raise ValueError('candidate build failed: '+str(e)) from e
            report['candidate']=candidate
            before=self.local(d,'before')
            if not before['passed']:raise RuntimeError('baseline replay failed')
            report['before']=before
            if action=='forward':
                report['before_forward']=self.forward(d,'before')
                if not report['before_forward']['passed']:raise RuntimeError('baseline forward failed')
            report['reload']=self.engine.reload(candidate['path'])
            try:
                report['replay']=self.local(d,'candidate')
                report['passed']=report['replay']['passed']
                if action=='forward' and report['passed']:
                    report['forward']=self.forward(d,'candidate');report['passed']=report['forward']['passed']
            finally:
                try:
                    report['restore']=self.engine.reload('builtin')
                except Exception as e:
                    raise RuntimeError('baseline restoration failed: '+str(e)) from e
            report['after']=self.local(d,'after')
            if not report['after']['passed']:raise RuntimeError('baseline restoration failed')
            if action=='forward' and report['passed']:
                report['after_forward']=self.forward(d,'after')
                if not report['after_forward']['passed']:raise RuntimeError('baseline forward changed')
        except ValueError as e:
            report.update(passed=False,error=str(e))
            try:
                self.engine.reload('builtin')
            except Exception as failure:
                report['error'] += '; baseline recovery failed: '+str(failure)
                raise RuntimeError(report['error']) from failure
        except Exception as e:
            report.update(passed=False,error=str(e))
            raise
        finally:
            report['finished']=True
            report['wall_s']=time.monotonic()-began
            atomic(d/'result.json',report)
        self.publish(status='ready',active=None,last_result=str(d/'result.json'))
        return report


def start(root,owner,payload):
    cfg=payload['config'];validate_config(cfg)
    linger=subprocess.check_output(['loginctl','show-user',str(os.getuid()),'--property=Linger','--value'],text=True).strip()
    if linger!='yes':raise ValueError('systemd user lingering is required; configure loginctl enable-linger before setup')
    directory=Path(cfg['data_dir'])/'sessions'/uuid.uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    with lock(root/'.scratch/lab/start.lock'):
        previous=status(root)
        if previous and previous['status']!='stopped':raise ValueError('lab already active')
        runtime=directory/'runtime';(runtime/'lab_support').mkdir(parents=True)
        for name in ('common.py','host.py','__init__.py'):
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
            socket=str(socket_dir/('ignis-lab-'+session[:16]+'.sock')),unit=unit,process=None)
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
    while time.monotonic()<deadline:
        state=read(root/'.scratch/lab/state.json')
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
        sock.connect(state['socket']);sock.sendall((json.dumps(payload)+'\n').encode())
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
    p.add_argument('action',choices=['start','serve','stop','replay','forward'])
    p.add_argument('--owner',required=True)
    args=p.parse_args()
    try:
        payload=json.load(sys.stdin) if args.action in ('start','replay','forward') else None
        if args.action=='serve':serve(args.root,args.owner);return
        result=start(args.root,args.owner,payload) if args.action=='start' else control(args.root,args.owner,args.action,payload)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error':str(e)}))
        raise SystemExit(2)

if __name__=='__main__':main()
