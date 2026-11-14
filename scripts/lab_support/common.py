import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

HOT = {'kernel/src/' + x for x in ('trellis_linear.cu', 'trellis_linear_decode.cu',
    'trellis_linear_prefill.cu', 'trellis_linear_internal.cuh', 'trellis_decode.cuh', 'trellis_linear_mlp_t8.cu', 'trellis_decode_helpers.cuh')}
GPU_LOCK = Path('/tmp/ignis-gpu-profile.lock')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_time=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record and identity(record['pid']) == record)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('busy: ' + str(path)) from None
        yield f


def source_identity(root):
    root = Path(root)
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--cached',
                                     '--others', '--exclude-standard']).split(b'\0')
    files = {}
    for raw in sorted(set(filter(None, names))):
        name = os.fsdecode(raw)
        if not (name.startswith(('crates/', 'kernel/', 'scripts/', 'mk/')) or name in ('Cargo.toml', 'Cargo.lock', 'Makefile', '.cargo/config.toml', 'rust-toolchain.toml')):
            continue
        if '/tests/' in name or name.endswith(('.md', '.png')):
            continue
        p = root / name
        if p.is_file():
            files[name] = digest(p)
    return files


def host_changes(before, after):
    return sorted(n for n in before.keys() | after.keys() if n not in HOT and before.get(n) != after.get(n))


def terminate(record, timeout=10):
    if not alive(record):
        return
    pid = record['pid']
    if os.getpgid(pid) != pid:
        raise ValueError('refusing to stop an unowned process group')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while alive(record) and time.monotonic() < deadline:
        time.sleep(.05)
    if alive(record):
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + timeout
        while alive(record) and time.monotonic() < deadline:
            time.sleep(.05)
    if alive(record):
        raise ValueError('process still alive; GPU ownership retained')


def status(root):
    state = read(Path(root) / '.scratch/lab/state.json')
    if state and not alive(state.get('process')):
        unit=subprocess.run(['systemctl','--user','show',state['unit'],'--property=ActiveState','--value'],capture_output=True,text=True)
        if unit.stdout.strip() not in ('active','activating','deactivating'):
            state = dict(state, status='stopped')
    return state


def validate_config(cfg):
    allowed={'artifact','data_dir','build_root','build_jobs','settings','capture_names',
             'capture_pin_bytes','capture_disk_bytes','view','warmup','samples',
             'timeout_s','setup_timeout_s','diagnostic_output_tokens','benchmark'}
    if not isinstance(cfg,dict) or set(cfg)-allowed:
        raise ValueError('unknown or malformed lab settings')
    for name in ('artifact','data_dir','build_root'):
        value=cfg.get(name)
        if value is None and name=='build_root':continue
        if not isinstance(value,str) or not value.startswith('/') or any(c in value for c in '\0\n\r'):
            raise ValueError(name+' must be an absolute remote path')
    for name in ('build_jobs','capture_pin_bytes','capture_disk_bytes','warmup','samples','diagnostic_output_tokens'):
        if name in cfg and (type(cfg[name]) is not int or cfg[name]<1):
            raise ValueError(name+' must be a positive integer')
    if cfg.get('samples',3)>10000 or cfg.get('warmup',1)>10000 or cfg.get('diagnostic_output_tokens',8192)>8192:
        raise ValueError('sample or diagnostic budget exceeds supported bound')
    for name in ('timeout_s','setup_timeout_s'):
        if name in cfg and (type(cfg[name]) not in (int,float) or not math.isfinite(cfg[name]) or cfg[name]<=0):
            raise ValueError(name+' must be finite and positive')
    if cfg.get('benchmark','astropy__astropy-12907') not in ('astropy__astropy-12907','django__django-10097','sympy__sympy-11618'):
        raise ValueError('unknown benchmark')
    settings=cfg.get('settings',{})
    if not isinstance(settings,dict) or set(settings)-{'max_context','prefill_chunk','lanes','kv_format','speculation'}:
        raise ValueError('unknown engine setting')
    for name in ('max_context','prefill_chunk','lanes'):
        if name in settings and (type(settings[name]) is not int or settings[name]<1):
            raise ValueError('invalid '+name)
    context=settings.get('max_context',65536);chunk=settings.get('prefill_chunk',1024)
    if context>131072 or chunk%128 or chunk>context or settings.get('lanes',2)>2:
        raise ValueError('unsupported engine geometry')
    if settings.get('kv_format','hq-e8-2b') not in ('bf16','hq-e8-2b') or settings.get('speculation','dflash2') not in ('off','dflash2'):
        raise ValueError('unsupported KV format or speculation')
    view=cfg.get('view',{})
    if not isinstance(view,dict) or set(view)-{'names','tokens','ids'}:
        raise ValueError('unknown replay view field')
    for values in (cfg.get('capture_names',[]),view.get('names',[])):
        if not isinstance(values,list) or any(not isinstance(v,str) or not v or ',' in v for v in values):
            raise ValueError('weight names must be a list of nonempty names')
    for name in ('tokens','ids'):
        values=view.get(name,[])
        if not isinstance(values,list) or any(type(v) is not int or v<(1 if name=='tokens' else 0) for v in values):
            raise ValueError('invalid view '+name)


class LabTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise LabTimeout('lab operation deadline exceeded')
    previous=signal.signal(signal.SIGALRM,expired)
    timer=signal.setitimer(signal.ITIMER_REAL,float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if timer[0]:signal.setitimer(signal.ITIMER_REAL,*timer)
