"""Resident exploration, immutable recordings and named candidate snapshots."""
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid

from .common import atomic, digest, key, read, source_identity, host_changes


def checked(reply):
    if not reply.get('ok'):
        raise RuntimeError(reply.get('error', 'engine operation failed'))
    return reply['result']


def selected(record, operators, tokens):
    names=[record['weight']['name']]+([record['up']['name']] if record.get('up') else [])
    return (not operators or any(n in operators for n in names)) and (not tokens or record['tokens'] in tokens)


def covers(scope, operators, tokens):
    return all(not scope.get(k) or bool(want) and set(want)<=set(scope[k])
               for k,want in (('operators',operators),('tokens_per_call',tokens)))


def forward_difference(reference, value):
    fields=('tokens','input_tokens','finish_reason','stop_token','logits_sha256')
    result={k:dict(baseline=reference.get(k),candidate=value.get(k))
            for k in fields if reference.get(k)!=value.get(k)}
    old=[(r['emitted'],r['extent']) for r in reference['rounds']]
    new=[(r['emitted'],r['extent']) for r in value['rounds']]
    if old!=new:result['rounds']=dict(baseline=old,candidate=new)
    return result


class Iteration:
    @contextlib.contextmanager
    def stage(self, name):
        began=time.monotonic()
        print('lab: '+name, file=sys.stderr, flush=True)
        try:yield
        finally:
            seconds=time.monotonic()-began
            if hasattr(self,'stage_timings'):self.stage_timings.append(dict(stage=name,seconds=seconds))
            print(f'lab: {name} {seconds:.3f}s', file=sys.stderr, flush=True)

    def capture(self, operators, tokens):
        scope=dict(operators=operators,tokens_per_call=tokens)
        destination=self.corpus/key(scope)
        if destination.exists():
            return self.verify_shard(destination)
        staging=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
        staging.mkdir(parents=True)
        pin=int(self.config.get('capture_pin_bytes',1<<30))
        disk=int(self.config.get('capture_disk_bytes',512<<30))
        with self.stage('capture '+json.dumps(scope,sort_keys=True)):
            minimum=checked(self.engine.call(dict(op='capture_minimum',names=','.join(operators),
                tokens=len(self.quick_case['tokens']),requests=1,tokens_per_call=tokens)))['prefill_bytes_lower_bound']
            if minimum>min(disk,shutil.disk_usage(staging).free):
                raise ValueError(f'capture needs at least {minimum} bytes; narrow --operator/--tokens-per-call or increase storage')
            clean=self.reference_forward()[self.quick_case['id']]
            checked(self.engine.call(dict(op='capture_start',path=str(staging),pin_bytes=pin,
                disk_bytes=disk,names=','.join(operators),tokens_per_call=tokens)))
            try:
                reference=checked(self.engine.call(dict(op='benchmark',case=self.quick_case)))
            finally:
                checked(self.engine.call(dict(op='capture_stop')))
            with (staging/'records.jsonl').open() as stream:
                count=sum(1 for _ in stream)
            manifest=dict(identity=self.corpus_identity,scope=scope,complete=True,records=count,
                reference=reference,clean_reference=clean,recording_differences=forward_difference(clean,reference),
                files={n:digest(staging/n) for n in ('data.bin','records.jsonl')})
            atomic(staging/'manifest.json',manifest)
            staging.rename(destination)
        return destination,manifest

    def verify_shard(self, directory):
        from .host import corpus_verify
        if not hasattr(self,'verified_shards'):self.verified_shards={}
        signature=tuple((n,(directory/n).stat().st_size,(directory/n).stat().st_mtime_ns)
                        for n in ('manifest.json','data.bin','records.jsonl'))
        cached=self.verified_shards.get(str(directory))
        if cached and cached[0]==signature:return directory,cached[1]
        manifest=corpus_verify(directory,self.corpus_identity)
        self.verified_shards[str(directory)]=(signature,manifest)
        return directory,manifest

    def select_corpus(self, payload):
        operators=payload.get('operators',[]);tokens=payload.get('tokens_per_call',[])
        cached=[]
        for path in sorted(self.corpus.glob('*/manifest.json')):
            if not re.fullmatch(r'[0-9a-f]{64}',path.parent.name):continue
            manifest=read(path)
            if not manifest or manifest.get('identity')!=self.corpus_identity or not manifest.get('complete'):
                raise ValueError('incomplete/incompatible corpus: '+str(path.parent))
            cached.append((path.parent,manifest))
        cells=[([operator] if operator else [],[token] if token else [])
               for operator in operators or [None] for token in tokens or [None]]
        # Missing cells share one benchmark traversal, even for many operators/shapes.
        missing=[(names,sizes) for names,sizes in cells
                 if not any(covers(m['scope'],names,sizes) for _,m in cached)]
        if missing:
            names=sorted({n for ns,_ in missing for n in ns}) if all(ns for ns,_ in missing) else []
            sizes=sorted({t for _,ts in missing for t in ts}) if all(ts for _,ts in missing) else []
            cached.append(self.capture(names,sizes))
        groups={}
        for names,sizes in cells:
            p,_=next((p,m) for p,m in cached if covers(m['scope'],names,sizes))
            if str(p) not in groups:
                p,m=self.verify_shard(p)
                groups[str(p)]=dict(data=str(p/'data.bin'),shard=str(p),records=[],manifest=m,selectors=[])
            groups[str(p)]['selectors'].append((names,sizes))
        output=[]
        for group in groups.values():
            selectors=group.pop('selectors')
            with (Path(group['shard'])/'records.jsonl').open() as stream:
                rows=(json.loads(line) for line in stream)
                group['records']=[r for r in rows if any(selected(r,n,t) for n,t in selectors)]
            if group['records']:output.append(group)
        if not output:raise ValueError('requested scope selects no supported operator calls')
        return output

    def replay_rows(self, groups, warmup, samples):
        output=[]
        for group in groups:
            rows=checked(self.engine.call(dict(op='replay_batch',data=group['data'],records=group['records'],
                warmup=warmup,samples=samples,exploratory=True)))
            labels={r['id']:dict(operator=r['weight']['name'],up=r.get('up',{}).get('name') if r.get('up') else None,
                tokens=r['tokens']) for r in group['records']}
            output.extend(dict(r,shard=group['shard'],**labels[r['record']]) for r in rows)
        return output

    def representative_scope(self, groups):
        selected=[]
        for group in groups:
            representatives={}
            for record in group['records']:
                shape=key({k:record.get(k) for k in ('weight','up','tokens','ldy','epilogue')})
                representatives.setdefault(shape,record)
            selected.append(dict(group,records=list(representatives.values())))
        return selected

    def check_scope(self, directory, label, groups=None):
        groups=self.selection if groups is None else groups
        with self.stage(label+' input diagnostics'):
            rows=self.replay_rows(groups,warmup=0,samples=1)
        atomic(directory/(label+'-correctness.json'),rows)
        return dict(passed=True,records=len(rows),numerical_match=all(r.get('mismatches',0)==0 for r in rows),
            diagnostics=rows,capture_trajectory_match=all(not g.get('manifest',{}).get('recording_differences') for g in groups),
            capture_differences=[dict(shard=g['shard'],differences=g['manifest']['recording_differences'])
                for g in groups if g.get('manifest',{}).get('recording_differences')])

    def measure_scope(self, directory, label, groups, warmup, samples):
        with self.stage(label+' representative timing'):
            rows=self.replay_rows(groups,warmup=warmup,samples=samples)
        atomic(directory/(label+'-timing.json'),rows)
        return dict(passed=True,performance_records=len(rows),warmup=warmup,samples=samples,
            numerical_match=all(r.get('mismatches',0)==0 for r in rows),timings=rows)

    def local(self, directory, label, groups=None, warmup=None, samples=None):
        groups=self.selection if groups is None else groups
        warmup=self.config.get('warmup',1) if warmup is None else warmup
        samples=self.config.get('samples',3) if samples is None else samples
        diagnostics=self.check_scope(directory,label,groups)
        measured=self.measure_scope(directory,label,self.representative_scope(groups),warmup,samples)
        return dict(diagnostics,**{k:v for k,v in measured.items() if k!='numerical_match'},
            numerical_match=diagnostics['numerical_match'] and measured['numerical_match'],
            interpretation='representative operator calls; exploratory numerical diagnostics')

    def reference_forward(self, full=False):
        if not hasattr(self,'references'):self.references={}
        for c in self.cases if full else [self.quick_case]:
            if c['id'] not in self.references:
                with self.stage('builtin forward reference '+c['id']):
                    self.references[c['id']]=checked(self.engine.call(dict(op='benchmark',case=c)))
        return self.references

    def forward(self,directory,label,full=False,warmup=None,samples=None):
        output=[]
        for c in self.cases if full else [self.quick_case]:
            rows=[]
            warm=self.config.get('warmup',1) if full else 0
            count=self.config.get('samples',3) if full else 1
            if warmup is not None:warm=warmup
            if samples is not None:count=samples
            with self.stage(label+' forward '+c['id']):
                for i in range(warm+count):
                    value=checked(self.engine.call(dict(op='benchmark',case=c)))
                    rounds=value['rounds'];emitted=sum(r['emitted'] for r in rounds)
                    value=dict(value,round_count=len(rounds),tokens_per_round=emitted/len(rounds) if rounds else None,
                        differences=forward_difference(self.references[c['id']],value))
                    if i>=warm:rows.append(value)
            output.append(dict(id=c['id'],samples=rows,median_prefill_ms=statistics.median(x['prefill_ms'] for x in rows),
                median_decode_ms=statistics.median(x['decode_ms'] for x in rows)))
        atomic(directory/(label+'-forward.json'),output)
        return dict(passed=True,cases=output,full=full,numerical_match=all(not r['differences'] for c in output for r in c['samples']))

    def names_path(self):
        return Path(self.config['data_dir'])/'candidate-names'/key(self.corpus_identity)/'names.json'

    def candidates(self):
        return read(self.names_path()) or {}

    def resolve_candidate(self, selector, directory):
        if selector=='builtin':return dict(identity='builtin',path='builtin')
        if selector!='current':
            candidate=self.candidates().get(selector)
            if candidate is None:raise ValueError('unknown candidate for this baseline: '+selector)
            if digest(candidate['path'])!=candidate['sha256']:raise ValueError('saved candidate checksum mismatch')
            return candidate
        import build_trellis_candidate as builder
        from .host import build_candidate
        actual=getattr(self,'evaluation_source',None) or source_identity(self.root)
        frozen=directory/'source';frozen.mkdir(parents=True)
        files=set(self.root.glob('kernel/src/**/*.h'))|set(self.root.glob('kernel/src/**/*.cuh'))|set(self.root.glob('kernel/include/*.h'))
        files|={self.root/'kernel/src'/n for n in builder.SOURCES}
        for src in files:
            dst=frozen/src.relative_to(self.root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            if digest(dst)!=actual[str(src.relative_to(self.root))]:raise ValueError('source changed while copying candidate')
        with self.stage('compile candidate'):
            try:
                candidate=build_candidate(frozen,Path(self.config['data_dir'])/'candidates',str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'))
            except Exception as e:
                raise ValueError('candidate build failed: '+str(e)) from e
        return dict(candidate,sha256=digest(candidate['path']),source={n:actual[n] for n in actual if n.startswith('kernel/')})

    def save_candidate(self, name, candidate):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or name in ('current','builtin'):
            raise ValueError('candidate name must be a simple name other than current/builtin')
        names=self.candidates()
        if name in names and names[name]['identity']!=candidate['identity']:
            raise ValueError('candidate name already identifies another immutable candidate: '+name)
        names[name]=candidate;atomic(self.names_path(),names)

    def compare(self,directory,payload,baseline,candidate):
        full=bool(payload.get('full'));forward=bool(payload.get('compare_forward'))
        count=payload.get('samples',self.config.get('samples',3));warmup=payload.get('warmup',self.config.get('warmup',1))
        legs={'baseline':[],'candidate':[]};order=[];checks={}
        if not forward:
            representatives=self.representative_scope(self.selection)
            checked_identities={}
            for label,item in (('baseline',baseline),('candidate',candidate)):
                identity=item.get('identity',item['path'])
                if identity not in checked_identities:
                    self.engine.reload(item['path'])
                    checked_identities[identity]=dict(self.check_scope(directory,label),candidate=identity)
                checks[label]=checked_identities[identity]
        # Alternating order gives both candidates the same capture and paired sample conditions.
        for i in range(count):
            pair=[('baseline',baseline),('candidate',candidate)]
            if i%2:pair.reverse()
            for label,item in pair:
                self.engine.reload(item['path']);order.append(label)
                leg=self.forward(directory,f'{label}-{i}',full,warmup,1) if forward else self.measure_scope(directory,f'{label}-{i}',representatives,warmup,1)
                legs[label].append(leg)
        performance=[]
        if not forward:
            for a,b in zip(legs['baseline'],legs['candidate']):
                refs={(r['shard'],r['record']):statistics.median(r['samples_ms']) for r in a['timings']}
                for r in b['timings']:
                    base=refs[(r['shard'],r['record'])];value=statistics.median(r['samples_ms'])
                    performance.append(dict(shard=r['shard'],record=r['record'],baseline_ms=base,candidate_ms=value,speedup=base/value if value else None))
        else:
            for pair,(a,b) in enumerate(zip(legs['baseline'],legs['candidate'])):
                references={c['id']:c for c in a['cases']}
                for case in b['cases']:
                    reference=references[case['id']]
                    before=reference['samples'][0];after=case['samples'][0]
                    performance.append(dict(pair=pair,case=case['id'],
                        baseline_prefill_ms=reference['median_prefill_ms'],candidate_prefill_ms=case['median_prefill_ms'],
                        prefill_speedup=reference['median_prefill_ms']/case['median_prefill_ms'] if case['median_prefill_ms'] else None,
                        baseline_decode_ms=reference['median_decode_ms'],candidate_decode_ms=case['median_decode_ms'],
                        decode_speedup=reference['median_decode_ms']/case['median_decode_ms'] if case['median_decode_ms'] else None,
                        baseline_output_tokens=before.get('output_tokens'),candidate_output_tokens=after.get('output_tokens'),
                        baseline_tokens_per_round=before['tokens_per_round'],candidate_tokens_per_round=after['tokens_per_round'],
                        differences=forward_difference(before,after)))
        return dict(passed=True,baseline=baseline,candidate=candidate,order=order,checks=checks,legs=legs,performance=performance,forward=forward)

    def evaluate(self,payload,action):
        actual=source_identity(self.root)
        if actual!=payload['source']:raise ValueError('sync/source identity mismatch; nothing measured')
        self.evaluation_source=actual
        changes=host_changes(self.state['source'],actual)
        if changes:raise ValueError('host changes require new baseline: '+', '.join(changes))
        d=self.directory/'results'/payload['operation'];d.mkdir(parents=True,exist_ok=False)
        report=dict(passed=False,operation=payload['operation'],action=action,engine_pid=getattr(self.engine,'engine_pid',self.engine.process.pid),
            source=actual,config_hash=self.state['config_hash'],finished=False,settings=self.state.get('settings'),
            scope={k:payload.get(k,[]) for k in ('operators','tokens_per_call')},
            profiler_session=getattr(self.engine,'profiler_session',None),
            profiling_enabled=self.state.get('profiling_enabled',False))
        self.publish(status='busy',active=payload['operation'],result=str(d/'result.json'))
        began=time.monotonic();self.stage_timings=[]
        fatal=None
        try:
            from .host import run
            run(['make','gpu-status'],d/'gpu-status.log',cwd=self.root)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
            if any(x.strip()!=str(getattr(self.engine,'engine_pid',self.engine.process.pid)) for x in apps if x.strip()):raise RuntimeError('foreign GPU process; refusing measurement')
            if action=='candidates':
                report.update(passed=True,candidates=self.candidates())
            else:
                if action=='profile' and not self.state.get('profiling_enabled',False):
                    raise ValueError('profiling is disabled; stop and setup --enable-profiling')
                if action=='profile' and (not shutil.which('nsys') or not getattr(self.engine,'profiler_session',None)):
                    raise ValueError('operator profiling requires nsys available at setup; stop and setup after configuring Nsight Systems')
                candidate=self.resolve_candidate(payload.get('candidate','current'),d)
                report['candidate']=candidate
                if payload.get('save_as'):self.save_candidate(payload['save_as'],candidate)
                need_forward=action=='forward' or action=='compare' and payload.get('compare_forward')
                if need_forward:
                    self.reference_forward(bool(payload.get('full')))
                    if action=='forward' and payload.get('full'):
                        report['before_forward']=self.forward(d,'before',full=True)
                elif not (action=='profile' and payload.get('profile_forward')):
                    self.selection=self.select_corpus(payload)
                if action=='compare':
                    baseline=self.resolve_candidate(payload.get('baseline','builtin'),d/'baseline')
                    report['compare']=self.compare(d,payload,baseline,candidate)
                else:
                    report['reload']=self.engine.reload(candidate['path'])
                    if action=='forward':report['forward']=self.forward(d,'candidate',bool(payload.get('full')),payload.get('warmup'),payload.get('samples'))
                    elif action=='profile':
                        from .profiling import profile
                        report['profile']=profile(self,d,payload,candidate)
                    else:report['replay']=self.local(d,'candidate',warmup=payload.get('warmup'),samples=payload.get('samples'))
                report['passed']=all(report[k].get('passed',True) for k in ('replay','forward','compare','profile') if k in report)
        except Exception as e:
            report['error']=str(e)
            if isinstance(e,(RuntimeError,TimeoutError)):fatal=e
        finally:
            try:
                with self.stage('restore builtin'):report['restore']=self.engine.reload('builtin')
                if 'before_forward' in report:
                    report['after_forward']=self.forward(d,'after',full=True)
            except Exception as e:
                report['passed']=False;report['error']='baseline restoration failed: '+str(e);fatal=RuntimeError(report['error'])
            report.update(finished=True,wall_s=time.monotonic()-began,stages=self.stage_timings)
            atomic(d/'result.json',report)
        if fatal:raise fatal
        self.publish(status='ready',active=None,last_result=str(d/'result.json'))
        return report
