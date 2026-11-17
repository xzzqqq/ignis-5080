"""Opt-in end-to-end test of the Laptop entry point and real 5080 session.

IGNIS_LAB_E2E_CONFIG=/absolute/lab-settings.json python3 -m unittest discover -s tests -p test_lab_gpu.py
The test derives a configuration with four output tokens and one timing sample.
Set IGNIS_LAB_E2E_PROFILE=1 as well to require a real Nsight report. These are
functional regressions, never a production benchmark verdict.
"""
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import sys
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('IGNIS_LAB_E2E_CONFIG'), 'opt-in exclusive RTX 5080 integration')
class LabGpuTests(unittest.TestCase):
    def command(self,*args,ok=True):
        p=subprocess.run([sys.executable,str(ROOT/'scripts/lab.py'),*args],cwd=ROOT,
                         capture_output=True,text=True,timeout=5400)
        self.logs.append(dict(args=args,code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        (self.output/'commands.json').write_text(json.dumps(self.logs,indent=2))
        self.assertEqual(p.returncode==0,ok,p.stderr+'\n'+p.stdout[-2000:])
        summary=json.loads(p.stdout) if p.stdout.strip() else {}
        self.assertNotIn('source',summary,'stdout includes the full source inventory')
        if summary.get('local_result'):
            full=json.loads(Path(summary['local_result']).read_text())
            self.assertEqual(summary['operation'],full['operation'])
            self.assertEqual(summary['passed'],full['passed'])
            return full
        return summary

    def prepare(self, label):
        self.output=ROOT/'.scratch'/('lab-e2e-'+label+'-'+uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True);self.logs=[]
        config=json.loads(Path(os.environ['IGNIS_LAB_E2E_CONFIG']).read_text())
        config.setdefault('build_root',str(Path(config['data_dir'])/'build'))
        config['data_dir']=str(Path(config['data_dir'])/'e2e'/self.output.name)
        config.update(diagnostic_output_tokens=4,warmup=1,samples=1)
        path=self.output/'settings.json';path.write_text(json.dumps(config,indent=2))
        return str(path),config

    def remote(self, code, *args):
        return json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/remote.py'),
            'exec','--no-env','--','/usr/bin/python3','-c',code,*map(str,args)],cwd=ROOT,text=True))

    def corpus_inventory(self, data_dir):
        return self.remote("import json,sys,hashlib;from pathlib import Path; p=Path(sys.argv[1])/'corpora';print(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('**/manifest.json')}))",data_dir)

    def assert_replay(self, report, pid, samples, numerical_match=True):
        self.assertTrue(report['passed'],report)
        self.assertTrue(report['finished']);self.assertEqual(report['engine_pid'],pid)
        result=report['replay'];self.assertEqual(result['numerical_match'],numerical_match)
        self.assertGreater(result['records'],0);self.assertTrue(result['timings'])
        for row in result['timings']:
            self.assertEqual(len(row['samples_ms']),samples)
            self.assertTrue(all(ms>0 for ms in row['samples_ms']),row)
            self.assertEqual(row['check'],'Exploration')
            d=row['diagnostics'];self.assertGreater(d['total_elements'],0)
            self.assertEqual(d['mismatches'],row['mismatches'])
            self.assertLessEqual(d['finite_elements'],d['total_elements'])
            for field in ('max_abs_error','relative_l2','reference_nan','reference_inf','actual_nan','actual_inf'):
                self.assertIn(field,d)
        return result

    def assert_captured_scope(self, report, operator, tokens):
        shards=sorted({row['shard'] for row in report['replay']['timings']})
        records=self.remote("import json,sys;from pathlib import Path; rows=[json.loads(line) for p in json.loads(sys.argv[1]) for line in (Path(p)/'records.jsonl').read_text().splitlines()];print(json.dumps([dict(weight=r['weight']['name'],up=r['up']['name'] if r.get('up') else None,tokens=r['tokens']) for r in rows]))",json.dumps(shards))
        self.assertTrue(records)
        self.assertTrue(all(row['tokens']==tokens and operator in (row['weight'],row['up']) for row in records),records)

    def assert_forward(self, report, pid, full=False):
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['engine_pid'],pid)
        result=report['forward'];self.assertEqual(result['full'],full)
        self.assertEqual(len(result['cases']),3 if full else 1)
        for case in result['cases']:
            self.assertEqual(len(case['samples']),1)
            sample=case['samples'][0]
            self.assertGreater(sample['input_tokens'],0)
            self.assertLessEqual(sample['output_tokens'],4)
            self.assertGreater(sample['round_count'],0)
            self.assertEqual(sample['round_count'],len(sample['rounds']))
            self.assertIn('tokens_per_round',sample);self.assertIn('differences',sample)
            self.assertEqual(len(sample['tokens']),sample['output_tokens'])
        return result

    def test_capture_reload_compile_failure_and_persistent_reuse(self):
        config,cfg=self.prepare('iteration')
        source=ROOT/'kernel/src/trellis_linear.cu';original=source.read_text()
        suffix=uuid.uuid4().hex[:10];exact='e2e-exact-'+suffix;approx='e2e-approx-'+suffix
        try:
            self.command('stop')
            before_setup=self.corpus_inventory(cfg['data_dir'])
            started=self.command('setup','--config',config,'--new-baseline')
            pid=started['engine_pid'];corpus=started['corpus']
            self.assertEqual(self.corpus_inventory(cfg['data_dir']),before_setup,'setup eagerly recorded operator inputs')
            # Quick forward works before any replay or recorded scope exists.
            self.assert_forward(self.command('forward','--candidate','builtin','--warmup','0','--samples','1'),pid)
            decode=8 if started['settings']['speculation']=='dflash2' else 1
            scope=('--operator','layer.0.gate','--tokens-per-call',str(decode))
            good=self.command('replay',*scope,'--warmup','0','--samples','2','--save-as',exact)
            self.assert_replay(good,pid,2)
            self.assert_captured_scope(good,'layer.0.gate',decode)
            first=self.corpus_inventory(cfg['data_dir'])
            self.assertGreater(len(first),len(before_setup))
            down=self.command('replay','--operator','layer.0.down','--tokens-per-call',str(decode),
                '--warmup','0','--samples','1')
            self.assert_replay(down,pid,1)
            self.assert_captured_scope(down,'layer.0.down',decode)
            second=self.corpus_inventory(cfg['data_dir'])
            self.assertGreater(len(second),len(first));self.assertTrue(first.items()<=second.items())
            prompt=self.remote("import json,sys;from pathlib import Path; s=json.loads(Path('.scratch/lab/state.json').read_text()); c=json.loads((Path(s['directory'])/'inputs.json').read_text())['cases'];print(json.dumps(len(next(x for x in c if x['id']==sys.argv[1])['tokens'])))",cfg.get('benchmark','astropy__astropy-12907'))
            prefill=min(started['settings']['prefill_chunk'],prompt)
            self.assertNotEqual(prefill,decode,'fixture needs distinct prefill and decode widths')
            token_scope=self.command('replay','--operator','layer.0.gate','--tokens-per-call',str(prefill),
                '--warmup','0','--samples','1')
            self.assert_replay(token_scope,pid,1)
            self.assert_captured_scope(token_scope,'layer.0.gate',prefill)
            third=self.corpus_inventory(cfg['data_dir'])
            self.assertGreater(len(third),len(second));self.assertTrue(second.items()<=third.items())
            # Real edits produce a different immutable library while reusing the model and recordings.
            source.write_text(original+'\n// resident-lab candidate identity regression\n')
            changed=self.command('replay',*scope,'--warmup','0','--samples','1')
            self.assert_replay(changed,pid,1)
            self.assertEqual(self.corpus_inventory(cfg['data_dir']),third,'covered scope was recorded again')
            self.assertNotEqual(good['candidate']['identity'],changed['candidate']['identity'])
            source.write_text(original+'\nTHIS_IS_AN_INTENTIONAL_COMPILE_FAILURE\n')
            failed=self.command('replay',*scope,ok=False)
            self.assertFalse(failed['passed']);self.assertIn('build failed',failed['error'])
            self.assertEqual(self.command('setup')['engine_pid'],pid)
            # Flip one finite BF16 output's low bit after real SwiGLU work. This
            # introduces an ordinary numerical difference without invalid execution.
            marker='extern "C" int32_t ignis_trellis_linear_swiglu('
            kernel='\n__global__ void lab_e2e_perturb(uint16_t *h) { if ((h[0]&0x7f80)!=0x7f80) h[0]^=1; }\n'
            run='return run(op, st, gate, up, x, tokens, h, ldh, kSwiGlu, workspace, stream);'
            self.assertEqual(original.count(run),1)
            altered=original.replace(marker,kernel+marker,1).replace(run,
                'int rc=run(op, st, gate, up, x, tokens, h, ldh, kSwiGlu, workspace, stream);\n'
                '  if (rc) return rc;\n'
                '  lab_e2e_perturb<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint16_t *>(h));\n'
                '  return cudaGetLastError()==cudaSuccess?0:-1;',1)
            source.write_text(altered)
            approximate=self.command('replay',*scope,'--warmup','1','--samples','3','--save-as',approx)
            replay=self.assert_replay(approximate,pid,3,numerical_match=False)
            self.assertTrue(any(row['mismatches']>0 for row in replay['diagnostics']))
            self.assertTrue(all(row['executions_checked']==4 for row in replay['timings']))
            names=self.command('candidates')
            self.assertEqual(names['candidates'][exact]['identity'],good['candidate']['identity'])
            self.assertEqual(names['candidates'][approx]['identity'],approximate['candidate']['identity'])
            # Named snapshots remain usable even while current edits do not compile.
            source.write_text(original+'\nTHIS_IS_AN_INTENTIONAL_COMPILE_FAILURE\n')
            compared=self.command('compare',*scope,'--baseline',exact,'--candidate',approx,'--warmup','0','--samples','2')
            self.assertTrue(compared['passed']);self.assertEqual(compared['engine_pid'],pid)
            comparison=compared['compare']
            self.assertEqual(comparison['baseline']['identity'],good['candidate']['identity'])
            self.assertEqual(comparison['candidate']['identity'],approximate['candidate']['identity'])
            self.assertEqual(comparison['order'],['baseline','candidate','candidate','baseline'])
            self.assertTrue(all(leg['numerical_match'] for leg in comparison['legs']['baseline']))
            self.assertTrue(all(not leg['numerical_match'] for leg in comparison['legs']['candidate']))
            self.assertTrue(comparison['performance'])
            self.assertTrue(all(r['baseline_ms']>0 and r['candidate_ms']>0 for r in comparison['performance']))
            forward_pair=self.command('compare','--forward','--baseline',exact,'--candidate',approx,
                '--warmup','0','--samples','1')
            self.assertTrue(forward_pair['passed']);self.assertEqual(forward_pair['engine_pid'],pid)
            pair=forward_pair['compare'];self.assertTrue(pair['forward'])
            self.assertEqual(pair['baseline']['identity'],good['candidate']['identity'])
            self.assertEqual(pair['candidate']['identity'],approximate['candidate']['identity'])
            self.assertEqual(pair['order'],['baseline','candidate'])
            for label in ('baseline','candidate'):
                self.assertEqual(len(pair['legs'][label]),1)
                self.assert_forward(dict(passed=True,engine_pid=pid,forward=pair['legs'][label][0]),pid)
            self.assert_forward(self.command('forward','--candidate',approx,'--warmup','0','--samples','1'),pid)
            full=self.command('forward','--candidate',exact,'--full','--warmup','0','--samples','1')
            self.assert_forward(full,pid,full=True)
            self.assertTrue(full['before_forward']['numerical_match']);self.assertTrue(full['after_forward']['numerical_match'])
            source.write_text(original)
            inventory=self.corpus_inventory(cfg['data_dir'])
            self.command('stop');self.command('stop')
            restarted=self.command('setup')
            self.assertNotEqual(restarted['engine_pid'],pid)
            self.assertEqual(restarted['corpus'],corpus)
            self.assertEqual(self.corpus_inventory(cfg['data_dir']),inventory,'restart changed immutable recordings')
            self.assert_replay(self.command('replay',*scope,'--candidate',exact,'--warmup','0','--samples','1'),restarted['engine_pid'],1)
        finally:
            source.write_text(original)
            self.command('stop')

    @unittest.skipUnless(os.environ.get('IGNIS_LAB_E2E_PROFILE'),'opt-in real Nsight capture')
    def test_real_profile_preserves_resident_engine(self):
        config,_=self.prepare('profile')
        try:
            self.command('stop')
            ready=self.command('setup','--config',config,'--new-baseline','--enable-profiling')
            self.assertTrue(ready['profiling_enabled'])
            decode=8 if ready['settings']['speculation']=='dflash2' else 1
            reports=[]
            for _ in range(2):
                result=self.command('profile','--candidate','builtin','--operator','layer.0.gate',
                    '--tokens-per-call',str(decode),'--warmup','1','--samples','2')
                self.assertTrue(result['passed']);self.assertEqual(result['engine_pid'],ready['engine_pid'])
                profile=result['profile'];self.assertEqual(profile['profiler'],'nsys')
                self.assertGreater(profile['kernel_count'],0);self.assertTrue(profile['kernels'])
                self.assertTrue(profile['devices']);self.assertGreater(profile['representatives'],0)
                self.assertTrue(self.remote("import json,sys;from pathlib import Path;print(json.dumps(all(Path(p).stat().st_size>0 for p in json.loads(sys.argv[1]).values())))",json.dumps(profile['artifacts'])))
                reports.append(profile['artifacts']['report'])
                self.assertEqual(self.command('setup')['engine_pid'],ready['engine_pid'])
                # The resident JSON protocol must remain usable after every
                # stopped capture, including when Nsight emits progress output.
                replay=self.command('replay','--candidate','builtin','--operator','layer.0.gate',
                    '--tokens-per-call',str(decode),'--warmup','0','--samples','1')
                self.assert_replay(replay,ready['engine_pid'],1)
            self.assertNotEqual(reports[0],reports[1],'repeated profiles overwrote the same report')
            self.assertTrue(self.remote("import json,sys;from pathlib import Path;print(json.dumps(all(Path(p).stat().st_size>0 for p in json.loads(sys.argv[1]))))",json.dumps(reports)))
        finally:
            self.command('stop')

    def test_disconnect_busy_stop_and_owner_death(self):
        config,_=self.prepare('lifecycle')
        def remote(code,*args):
            return json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/remote.py'),
                'exec','--no-env','--','/usr/bin/python3','-c',code,*args],cwd=ROOT,text=True))
        def state():
            return remote("import sys,json;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import status;print(json.dumps(status(Path.cwd())))")
        def wait(predicate,limit=180):
            deadline=time.monotonic()+limit
            while time.monotonic()<deadline:
                current=state()
                if predicate(current):return current
                time.sleep(.5)
            self.fail('lifecycle deadline: '+str(current))
        def start(action,label,*args):
            out=(self.output/(label+'.out')).open('w');err=(self.output/(label+'.err')).open('w')
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/lab.py'),action,*args],cwd=ROOT,
                stdout=out,stderr=err,start_new_session=True)
            out.close();err.close();return p
        process=None
        try:
            self.command('stop')
            ready=self.command('setup','--config',config,'--new-baseline')
            pid=ready['engine_pid']
            decode=8 if ready['settings']['speculation']=='dflash2' else 1
            process=start('replay','disconnected','--candidate','builtin','--operator','layer.0.gate',
                '--tokens-per-call',str(decode),'--warmup','0','--samples','2048')
            active=wait(lambda s:s['status']=='busy');operation=active['active']
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20);process=None
            done=wait(lambda s:s['status']=='ready')
            self.assertEqual(done['engine_pid'],pid)
            result=remote("import json,sys;from pathlib import Path;print(Path(sys.argv[1]).read_text())",done['last_result'])
            self.assertTrue(result['finished']);self.assertTrue(result['passed'])
            self.assertEqual(result['operation'],operation)
            process=start('forward','stopped-busy','--candidate','builtin','--samples','64')
            wait(lambda s:s['status']=='busy');time.sleep(1)
            self.command('stop')
            self.assertNotEqual(process.wait(timeout=30),0);process=None
            self.assertEqual(state()['status'],'stopped')
            ready=self.command('setup')
            current=state()
            remote("import json,sys,os,signal;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import identity; s=json.loads(Path('.scratch/lab/state.json').read_text()); assert identity(s['process']['pid'])==s['process'];os.kill(s['process']['pid'],signal.SIGKILL);print('{}')")
            wait(lambda s:s['status']=='stopped',60)
            gone=remote("import json,sys;from pathlib import Path;sys.path.insert(0,'scripts');from lab_support.common import read,alive;print(json.dumps(not alive(read(Path(sys.argv[1])/'engine.identity.json'))))",current['directory'])
            self.assertTrue(gone,'owner death leaked the GPU child')
            self.command('stop')
            (self.output/'completed.json').write_text(json.dumps(dict(passed=True,disconnected_operation=operation)))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20)
            self.command('stop')
