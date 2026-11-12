import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .common import atomic, key, lock, read, source_identity, host_changes, validate_config
import remote

ROOT = Path(__file__).resolve().parents[2]


def call(config_file, argv, payload=None, no_env=True):
    cmd = [sys.executable, str(ROOT / 'scripts/remote.py'), '--config', str(config_file), 'exec']
    if no_env:
        cmd.append('--no-env')
    result = subprocess.run(cmd + ['--', *argv], input=None if payload is None else json.dumps(payload),
                            stdout=subprocess.PIPE, text=True)
    value = json.loads(result.stdout)
    if result.returncode:
        raise ValueError(value.get('error', 'remote lab failed'))
    return value


def probe(config_file):
    # The receiver need not exist yet, and cleanup must not source a broken env.
    common = (ROOT / 'scripts/lab_support/common.py').read_text()
    return call(config_file, ['/usr/bin/python3', '-c', common + '\nprint(json.dumps(status(Path.cwd())))'])


def rpc(config_file, state, action, payload=None):
    return call(config_file, ['/usr/bin/python3', state['control'], '--root', state['root'],
        action, '--owner', state['owner']], payload)


def summary(value, local_result=None):
    if 'operation' not in value:
        fields=('status','session','engine_pid','corpus','corpus_records','settings','last_result','error')
        return {k:value[k] for k in fields if k in value}
    fields=('passed','operation','action','engine_pid','wall_s','error','candidate','before','replay','after',
            'diagnostic_output_tokens','warmup','samples','view','client_timings_s')
    result={k:value[k] for k in fields if k in value}
    result['local_result']=str(local_result) if local_result else None
    if value.get('forward',{}).get('cases'):
        result['forward']=[dict(id=c['id'],median_decode_ms=c['median_decode_ms'],
            output_tokens=[r['output_tokens'] for r in c['samples']],
            finish_reasons=[r['finish_reason'] for r in c['samples']]) for c in value['forward']['cases']]
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['setup', 'replay', 'forward', 'stop'])
    p.add_argument('--config', type=Path, help='Lab settings JSON (first setup only)')
    p.add_argument('--new-baseline', action='store_true')
    args = p.parse_args(argv)
    began=time.monotonic()
    local = ROOT / '.scratch/lab'
    local.mkdir(parents=True, exist_ok=True)
    receipt_file = local / 'receipt.json'
    receipt = read(receipt_file)
    try:
        if args.action == 'stop':
            if not receipt:
                print(json.dumps({'status':'stopped'})); return
            state = probe(local / 'remote.json')
            if state and state['owner'] != receipt['owner']:
                raise ValueError('remote session belongs to another worktree')
            if state:
                print(json.dumps(rpc(local / 'remote.json',state,'stop')))
            else:
                print(json.dumps({'status':'stopped'}))
            return
        with lock(local / 'command.lock'):
            if receipt is None:
                if args.action != 'setup':
                    raise ValueError('run setup first')
                if not args.config:
                    raise ValueError('first setup requires --config LAB_SETTINGS_JSON')
                config = json.loads(args.config.read_text())
                validate_config(config)
                receipt = dict(owner=uuid.uuid4().hex, config=config, local_worktree=str(ROOT))
                atomic(local / 'remote.json', remote.load_config(remote.config_path(ROOT)))
                atomic(receipt_file, receipt)
            elif args.config:
                config = json.loads(args.config.read_text())
                validate_config(config)
                if config != receipt['config']:
                    if not args.new_baseline:
                        raise ValueError('changed configuration requires stopped session and --new-baseline')
                    receipt = dict(receipt, config=config)
            validate_config(receipt['config'])
            target = local / 'remote.json'
            current = remote.load_config(remote.config_path(ROOT))
            if current != read(target):
                raise ValueError('remote target changed; stop uses the recorded target')
            state = probe(target)
            if state and state['owner'] != receipt['owner']:
                raise ValueError('remote lab belongs to another worktree; refusing sync')
            if state and state['status'] != 'stopped':
                if args.action == 'setup':
                    if args.new_baseline or state['config_hash'] != key(receipt['config']):
                        raise ValueError('stop the existing session before changing baseline/config')
                    print(json.dumps(summary(state))); return
                if state['status'] != 'ready':
                    raise ValueError('busy/not ready: ' + state['status'])
            elif args.action != 'setup':
                raise ValueError('engine stopped; run setup')
            source = source_identity(ROOT)
            if state and not args.new_baseline:
                changes = host_changes(state['source'], source)
                if changes:
                    raise ValueError('host source changed; stop and setup --new-baseline: ' + ', '.join(changes))
            atomic(receipt_file, receipt)
            sync_start=time.monotonic()
            subprocess.run([sys.executable,str(ROOT/'scripts/remote.py'),'--config',str(target),'sync'],
                           stdout=sys.stderr,check=True)
            synced=time.monotonic()
            payload = dict(owner=receipt['owner'],config=receipt['config'],source=source,
                           new_baseline=args.new_baseline,operation=uuid.uuid4().hex)
            if args.action == 'setup':
                state = call(target, ['/usr/bin/python3','scripts/lab_support/host.py','--root',
                    current['repo'],'start','--owner',receipt['owner']], payload, no_env=False)
                atomic(local/'session.json',state)
                print(json.dumps(summary(state)))
            else:
                atomic(local/'pending.json',dict(operation=payload['operation'],action=args.action))
                reply = rpc(target,state,args.action,payload)
                reply['client_timings_s']=dict(preflight=sync_start-began,sync=synced-sync_start,
                    remote_request=time.monotonic()-synced,total=time.monotonic()-began)
                atomic(local/(payload['operation']+'.json'),reply)
                print(json.dumps(summary(reply,local/(payload['operation']+'.json'))))
                if not reply.get('passed'):
                    raise SystemExit(1)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print('lab: '+str(e),file=sys.stderr)
        raise SystemExit(2)
