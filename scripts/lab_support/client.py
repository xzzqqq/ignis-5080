import argparse
from collections import Counter, defaultdict
import statistics
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


def candidate_summary(value):
    result={k:value[k] for k in ('identity','path','sha256','cache_hit','seconds','link_seconds') if k in value}
    if value.get('objects'):
        result['object_cache_hits']=sum(bool(o['cache_hit']) for o in value['objects'])
        result['objects']=len(value['objects'])
    return result


def numerical_summary(rows):
    counts=('mismatches','total_elements','finite_elements','reference_nan','reference_inf','actual_nan','actual_inf')
    result={field:sum(r.get('diagnostics',{}).get(field,r.get(field,0)) for r in rows) for field in counts}
    result.update(records=len(rows),mismatch_records=sum(r.get('diagnostics',{}).get('mismatches',r.get('mismatches',0))>0 for r in rows))
    for source,target in (('max_abs_error','max_abs_error'),('relative_l2','max_record_relative_l2')):
        values=[r.get('diagnostics',{}).get(source) for r in rows]
        finite=[v for v in values if v is not None]
        result[target]=max(finite) if finite else None
        result['undefined_'+source+'_records']=sum(v is None for v in values)
    result['zero_reference_records']=sum(bool(r.get('diagnostics',{}).get('relative_l2_zero_reference')) for r in rows)
    state=[r['state_diagnostics'] for r in rows if 'state_diagnostics' in r]
    if state:
        result['state']=dict(records=len(state),mismatched_bytes=sum(r['mismatched_bytes'] for r in state),
            checked_bytes=sum(r['checked_bytes'] for r in state),exact_match=all(r['exact_match'] for r in state))
        implementations=sorted({r.get('implementation','unknown') for r in rows if 'state_diagnostics' in r})
        component='native_segments' if any(r.get('kind')=='segment' for r in rows) else 'native_attention'
        result[component]=dict(implementations=implementations,
            candidate_applied=any(r.get('candidate_applied',False) for r in rows if 'state_diagnostics' in r),
            scope='three finite C1/T8 blocks' if component=='native_segments' else 'C1/T8/HQ',unsupported_shapes='builtin')
        if component=='native_segments':
            preservation=defaultdict(list)
            for row in rows:
                for binding in row.get('state_diagnostics',{}).get('bindings',[]):
                    preservation[binding['binding']].append(binding)
            result['state']['bindings']={name:dict(mismatched_bytes=sum(b['mismatched_bytes'] for b in values),
                checked_bytes=sum(b['checked_bytes'] for b in values),readonly=all(b['readonly'] for b in values))
                for name,values in preservation.items()}
            result['state']['readonly_mismatched_bytes']=sum(b['mismatched_bytes'] for b in result['state']['bindings'].values() if b['readonly'])
            result['state']['mutable_mismatched_bytes']=sum(b['mismatched_bytes'] for b in result['state']['bindings'].values() if not b['readonly'])
            outputs=defaultdict(list)
            for row in rows:
                for output in row.get('outputs',[]):outputs[output['binding']].append(output)
            result['outputs']={name:numerical_summary(values) for name,values in outputs.items()}
    return result


def timing_summary(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row.get('operator'),row.get('up'),row.get('tokens'))].extend(row.get('samples_ms',[]))
    values=[dict(operator=name,up=up,tokens_per_call=tokens,measurements=len(times),
                 median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times))
            for (name,up,tokens),times in grouped.items() if times]
    values.sort(key=lambda r:r['median_ms'],reverse=True)
    return dict(groups=len(values),shown=min(len(values),20),slowest=values[:20])


def difference_summary(value):
    result={}
    for name,pair in value.items():
        a,b=pair.get('baseline'),pair.get('candidate')
        if isinstance(a,list) and isinstance(b,list):
            first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            result[name]=dict(baseline_count=len(a),candidate_count=len(b),first_difference=first,
                baseline=a[first] if first<len(a) else None,candidate=b[first] if first<len(b) else None)
        elif name=='logits_sha256':result[name]=True
        else:result[name]=pair
    return result


def forward_summary(value):
    cases=[]
    for case in value.get('cases',[]):
        rows=case['samples'];differences=[r['differences'] for r in rows if r.get('differences')]
        rounds=[r for row in rows for r in row.get('rounds',[])]
        fields=Counter(k for d in differences for k in d)
        cases.append(dict(id=case['id'],samples=len(rows),median_prefill_ms=case['median_prefill_ms'],
            median_decode_ms=case['median_decode_ms'],
            output_tokens=dict(min=min(r['output_tokens'] for r in rows),max=max(r['output_tokens'] for r in rows)),
            round_count=dict(min=min(r['round_count'] for r in rows),max=max(r['round_count'] for r in rows)),
            median_tokens_per_round=statistics.median(r['tokens_per_round'] for r in rows if r.get('tokens_per_round') is not None) if any(r.get('tokens_per_round') is not None for r in rows) else None,
            median_round_ms=statistics.median(r['ms'] for r in rounds) if rounds else None,
            finish_reasons=dict(Counter(r.get('finish_reason') for r in rows)),
            stop_tokens=dict(Counter(str(r.get('stop_token')) for r in rows)),
            differing_samples=len(differences),difference_counts=dict(fields),
            difference_examples=[difference_summary(d) for d in differences[:3]]))
    return dict(passed=value.get('passed'),full=value.get('full'),numerical_match=value.get('numerical_match'),
        reference_available=value.get('reference_available',True),cases=cases)


def coverage_summary(value):
    return dict(mode=value['mode'],records=value['records'],shards=len(value['shards']),
        selection=[dict(shard=g['shard'],records=g['records'],available_records=g.get('available_records'),
            record_ids=g.get('record_ids',[])[:20],record_ids_shown=min(len(g.get('record_ids',[])),20),
            capture_scope=g.get('capture_scope'),integrity=g.get('integrity'),
            source_records=g.get('source_records',[])[:20]) for g in value['shards'][:20]])


def replay_summary(value):
    result={k:value[k] for k in ('passed','records','performance_records','warmup','samples','numerical_match','state_match','capture_trajectory_match') if k in value}
    if 'coverage' in value:result['coverage']=coverage_summary(value['coverage'])
    result['numerical']=numerical_summary(value.get('diagnostics',[]))
    result['timing_numerical']=numerical_summary(value.get('timings',[]))
    result['timing']=timing_summary(value.get('timings',[]))
    capture=value.get('capture_differences',[])
    result['capture_differing_shards']=len(capture)
    result['capture_difference_examples']=[dict(shard=r['shard'],differences=difference_summary(r['differences'])) for r in capture[:3]]
    return result


def compare_summary(value):
    result={k:value[k] for k in ('passed','forward') if k in value}
    if 'coverage' in value:result['coverage']=coverage_summary(value['coverage'])
    result.update(baseline=candidate_summary(value['baseline']),candidate=candidate_summary(value['candidate']),
                  pairs=len(value.get('order',[]))//2)
    if value.get('forward'):
        result['legs']={}
        for name,legs in value['legs'].items():
            cases=defaultdict(list)
            for leg in legs:
                for case in leg.get('cases',[]):cases[case['id']].extend(case['samples'])
            merged=dict(passed=all(leg.get('passed',True) for leg in legs),
                numerical_match=all(leg.get('numerical_match',True) for leg in legs),
                full=legs[0].get('full') if legs else False,cases=[dict(id=case,samples=rows,
                    median_prefill_ms=statistics.median(r['prefill_ms'] for r in rows),
                    median_decode_ms=statistics.median(r['decode_ms'] for r in rows)) for case,rows in cases.items()])
            result['legs'][name]=forward_summary(merged)
        grouped=defaultdict(list)
        for row in value.get('performance',[]):grouped[row['case']].append(row)
        result['performance']=[]
        for name,rows in grouped.items():
            entry=dict(case=name,pairs=len(rows))
            for field in ('baseline_prefill_ms','candidate_prefill_ms','prefill_speedup','baseline_decode_ms','candidate_decode_ms','decode_speedup'):
                values=[r[field] for r in rows if r.get(field) is not None]
                entry['median_'+field]=statistics.median(values) if values else None
            diffs=[r['differences'] for r in rows if r.get('differences')]
            entry.update(differing_pairs=len(diffs),difference_counts=dict(Counter(k for d in diffs for k in d)),
                difference_examples=[difference_summary(d) for d in diffs[:3]])
            result['performance'].append(entry)
    else:
        result['checks']={}
        for name,check in value.get('checks',{}).items():
            capture=check.get('capture_differences',[])
            result['checks'][name]=dict(candidate=check.get('candidate'),records=check.get('records'),
                numerical_match=check.get('numerical_match'),numerical=numerical_summary(check.get('diagnostics',[])),
                capture_trajectory_match=check.get('capture_trajectory_match',True),capture_differing_shards=len(capture),
                capture_difference_examples=[dict(shard=r['shard'],differences=difference_summary(r['differences'])) for r in capture[:3]])
        result['legs']={}
        for name,legs in value['legs'].items():
            result['legs'][name]=dict(pairs=len(legs),
                representatives_per_pair=legs[0].get('performance_records') if legs else 0,
                numerical_match=all(leg.get('numerical_match',True) for leg in legs),
                timing_numerical=numerical_summary([r for leg in legs for r in leg.get('timings',[])]))
        labels={(r['shard'],r['record']):(r.get('operator'),r.get('up'),r.get('tokens'))
                for leg in value['legs'].get('candidate',[]) for r in leg.get('timings',[])}
        grouped=defaultdict(list)
        for row in value.get('performance',[]):grouped[labels.get((row['shard'],row['record']),(None,None,None))].append(row)
        timings=[]
        for (name,up,tokens),rows in grouped.items():
            entry=dict(operator=name,up=up,tokens_per_call=tokens,measurements=len(rows))
            for field in ('baseline_ms','candidate_ms','speedup'):
                values=[r[field] for r in rows if r.get(field) is not None]
                entry['median_'+field]=statistics.median(values) if values else None
            timings.append(entry)
        timings.sort(key=lambda r:r['median_candidate_ms'],reverse=True)
        ratios=[r['speedup'] for r in value.get('performance',[]) if r.get('speedup') is not None]
        result['performance']=dict(groups=len(timings),shown=min(len(timings),20),slowest=timings[:20],
            regressions=sum(r<1 for r in ratios),min_speedup=min(ratios) if ratios else None,
            max_speedup=max(ratios) if ratios else None,median_speedup=statistics.median(ratios) if ratios else None)
    return result


def summary(value, local_result=None):
    if 'operation' not in value:
        fields=('status','session','engine_pid','corpus','corpus_records','settings','profiling_enabled','profiler_session','last_result','error')
        return {k:value[k] for k in fields if k in value}
    fields=('passed','operation','action','engine_pid','wall_s','error','finished','scope','profiler_session','profiling_enabled','client_timings_s')
    result={k:value[k] for k in fields if k in value}
    result['status']='completed' if value.get('passed') else 'failed'
    result['local_result']=str(local_result) if local_result else None
    if 'candidate' in value:result['candidate']=candidate_summary(value['candidate'])
    if 'candidates' in value:result['candidates']={name:candidate_summary(c) for name,c in value['candidates'].items()}
    if 'replay' in value:result['replay']=replay_summary(value['replay'])
    for field in ('forward','before_forward','after_forward'):
        if field in value:result[field]=forward_summary(value[field])
    if 'compare' in value:result['compare']=compare_summary(value['compare'])
    if 'export' in value:
        result['export']={k:v for k,v in value['export'].items() if k in ('passed','path','manifest','record','shard')}
        result['export']['coverage']=coverage_summary(value['export']['coverage'])
    if 'profile' in value:
        result['profile']={k:v for k,v in value['profile'].items() if k not in ('kernels','candidate','forward','attribution')}
        if 'coverage' in result['profile']:result['profile']['coverage']=coverage_summary(result['profile']['coverage'])
        result['profile']['kernels']=value['profile'].get('kernels',[])[:20]
        result['profile']['kernels_shown']=len(result['profile']['kernels'])
        if 'forward' in value['profile']:
            result['profile']['forward']=forward_summary(value['profile']['forward'])
        if 'attribution' in value['profile']:
            attribution=value['profile']['attribution']
            result['profile']['attribution']={k:v for k,v in attribution.items() if k not in ('timeline','instances','aggregates')}
            for field,limit in (('instances',5),('aggregates',20)):
                if field not in attribution:continue
                result['profile']['attribution'][field+'_count']=len(attribution.get(field,[]))
                result['profile']['attribution'][field]=attribution.get(field,[])[:limit]
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description='Resident Kernel Lab: setup fixes the model/settings; replay explores numerical differences and timing; forward measures model execution; compare alternates paired measurements; candidates lists saved snapshots; profile reports operator or full-model CUDA execution; stop releases the GPU.')
    p.add_argument('action', choices=['setup', 'replay', 'forward', 'compare', 'candidates', 'profile', 'export', 'stop'], help='replay is quick exploration, not strict numerical acceptance; forward needs no replay or capture')
    p.add_argument('--config', type=Path, help='Fixed artifact/settings/resources JSON, setup only (required on first setup)')
    p.add_argument('--new-baseline', action='store_true', help='After stopping, rebuild the fixed host baseline (setup only)')
    p.add_argument('--enable-profiling', action='store_true', help='Launch this session under Nsight Systems; disabled by default on new setup (setup only)')
    p.add_argument('--full',action='store_true',help='Evaluate all benchmark requests; forward also runs fresh builtin controls before and after')
    p.add_argument('--operator', action='append', default=[], dest='operators', help='Exact weight/operator name; repeat for several; omitted means all supported calls (replay/profile/compare without --forward)')
    p.add_argument('--tokens-per-call', type=int, action='append', default=[], help='Operator input token count, not generation length; repeat for several; omitted means every shape')
    p.add_argument('--record-start',type=int,default=0,help='Skip this many matching calls before capture, or matching records in execution order in --corpus')
    p.add_argument('--record-limit',type=int,default=0,help='Select at most this many matching records (0 means all); capture selection happens before device copies')
    p.add_argument('--corpus',help='Existing remote corpus shard directory; select immutable record IDs without recording a new request')
    p.add_argument('--warmup', type=int, help='Warmup executions (0..10000); profile --forward defaults to 0, other modes use their usual defaults')
    p.add_argument('--samples', type=int, help='Timed samples; profile --forward defaults to 1; compare alternates this many pairs')
    p.add_argument('--candidate', default='current', help='current freezes worktree kernels; otherwise a saved immutable name or builtin')
    p.add_argument('--baseline', default='builtin', help='Baseline snapshot name for paired compare (default builtin)')
    p.add_argument('--save-as', help='Save this replay candidate under an immutable name for this model/settings baseline')
    p.add_argument('--forward', action='store_true', dest='compare_forward', help='compare: pair full-model requests; profile: capture the fixed request prefill and natural generation without an operator recording')
    arguments=list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(arguments)
    supplied={a.split('=',1)[0] for a in arguments if a.startswith('--')}
    allowed={
        '--config':{'setup'}, '--new-baseline':{'setup'}, '--enable-profiling':{'setup'}, '--full':{'forward','compare'},
        '--operator':{'replay','profile','compare','export'}, '--tokens-per-call':{'replay','profile','compare','export'},
        '--record-start':{'replay','profile','compare','export'}, '--record-limit':{'replay','profile','compare','export'},
        '--corpus':{'replay','profile','compare','export'},
        '--warmup':{'replay','forward','compare','profile'}, '--samples':{'replay','forward','compare','profile'},
        '--candidate':{'replay','forward','compare','profile','export'}, '--baseline':{'compare'},
        '--save-as':{'replay'}, '--forward':{'compare','profile'}}
    for option in supplied:
        if option in allowed and args.action not in allowed[option]:p.error(option+' does not apply to '+args.action)
    if args.action=='compare' and args.full and not args.compare_forward:p.error('--full requires compare --forward')
    if not 0<=args.record_start<(1<<64) or not 0<=args.record_limit<(1<<64):p.error('--record-start/--record-limit must be unsigned 64-bit integers')
    if args.corpus and (not args.corpus.startswith('/') or any(c in args.corpus for c in '\0\n\r')):p.error('--corpus must be an absolute remote shard path')
    if args.action=='export' and args.record_limit!=1:p.error('export requires --record-limit 1')
    if args.compare_forward and (args.operators or args.tokens_per_call or args.record_start or args.record_limit or args.corpus or supplied & {'--record-start','--record-limit','--corpus'}):p.error('operator/record selectors do not apply to '+args.action+' --forward')
    if args.warmup is not None and not 0 <= args.warmup <= 10000:p.error('--warmup must be 0..10000')
    if args.samples is not None and not 1 <= args.samples <= 10000:p.error('--samples must be 1..10000')
    if any(not n or ',' in n for n in args.operators):p.error('--operator requires one exact nonempty weight name per flag')
    if any(t < 1 for t in args.tokens_per_call):p.error('--tokens-per-call must be positive')
    if args.save_as and args.action != 'replay':p.error('--save-as requires replay')
    if args.compare_forward and args.action not in ('compare','profile'):p.error('--forward requires compare or profile')
    if args.full and args.action not in ('forward','compare'):p.error('--full requires forward')
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
                if state['status']=='stopped':
                    print(json.dumps({'status':'stopped'}));return
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
                if state['status']!='stopped':
                    raise ValueError('remote lab belongs to another worktree; refusing sync')
                state=None
            if state and state['status'] != 'stopped':
                if args.action == 'setup':
                    if args.new_baseline or state['config_hash'] != key(receipt['config']):
                        raise ValueError('stop the existing session before changing baseline/config')
                    if args.enable_profiling and not state.get('profiling_enabled',False):
                        raise ValueError('profiling is disabled in the active session; stop and setup --enable-profiling')
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
            print('lab: source synced; '+args.action, file=sys.stderr, flush=True)
            payload = dict(owner=receipt['owner'],config=receipt['config'],source=source,
                           new_baseline=args.new_baseline,operation=uuid.uuid4().hex,full=args.full,
                           operators=args.operators,tokens_per_call=args.tokens_per_call,
                           record_start=args.record_start,record_limit=args.record_limit,corpus=args.corpus,
                           candidate=args.candidate,baseline=args.baseline,save_as=args.save_as,
                           compare_forward=args.compare_forward and args.action=='compare',
                           profile_forward=args.compare_forward and args.action=='profile')
            for name in ('warmup','samples'):
                if getattr(args,name) is not None:payload[name]=getattr(args,name)
            if args.action == 'setup':
                payload['profiling_enabled']=args.enable_profiling
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
