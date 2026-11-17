"""CPU contracts for four-command ownership, source freshness and corpus reuse."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import client, common, host, iteration


class LabTests(unittest.TestCase):
    def test_deadline_interrupts_blocking_work_and_disarms(self):
        import time
        with self.assertRaises(common.LabTimeout):
            with common.deadline(.02):
                time.sleep(.2)
        time.sleep(.03)

    def test_only_admitted_changes_keep_the_host(self):
        old={'kernel/src/trellis_linear.cu':'old','crates/core/src/step.rs':'host'}
        new=dict(old,**{'kernel/src/trellis_linear.cu':'new'})
        self.assertEqual(common.host_changes(old,new),[])
        new['crates/core/src/step.rs']='modified'
        self.assertEqual(common.host_changes(old,new),['crates/core/src/step.rs'])
        del new['crates/core/src/step.rs']
        self.assertEqual(common.host_changes(old,new),['crates/core/src/step.rs'])

    def test_config_rejects_ignored_or_invalid_controls(self):
        valid={'artifact':'/model','data_dir':'/data'}
        common.validate_config(valid)
        for bad in [dict(valid,unused_option=1),dict(valid,warmup=0),dict(valid,samples=True),
                    dict(valid,timeout_s=float('inf')),dict(valid,diagnostic_output_tokens=8193),
                    dict(valid,settings={'prefill_chunk':127}),dict(valid,view={'unknown':1})]:
            with self.subTest(config=bad),self.assertRaises(ValueError):
                common.validate_config(bad)

    def test_session_provenance_survives_next_session(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);directory=root/'session-a';directory.mkdir()
            state=dict(session='a',config={'artifact':'/model','data_dir':t},directory=str(directory))
            service=host.Owner(root,state)
            service.publish(status='ready')
            common.atomic(root/'.scratch/lab/state.json',dict(session='b'))
            self.assertEqual(common.read(directory/'session.json')['status'],'ready')
            with self.assertRaisesRegex(RuntimeError,'replaced'):
                service.publish(status='busy')
            self.assertEqual(common.read(root/'.scratch/lab/state.json')['session'],'b')

    def test_incomplete_and_corrupt_corpora_never_pass(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);identity={'model':'frozen'}
            (d/'data.bin').write_bytes(b'captured')
            (d/'records.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                host.corpus_verify(d,identity)
            manifest={'identity':identity,'complete':True,'files':{n:common.digest(d/n) for n in ('data.bin','records.jsonl')}}
            common.atomic(d/'manifest.json',manifest)
            self.assertEqual(host.corpus_verify(d,identity),manifest)
            (d/'data.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                host.corpus_verify(d,identity)

    def fixture(self,d):
        local=d/'.scratch/lab';local.mkdir(parents=True)
        target={'host':'gpu-test','repo':'/test/lab'}
        receipt={'owner':'owner-a','config':{'artifact':'/model','data_dir':'/data'},'local_worktree':str(d)}
        common.atomic(local/'receipt.json',receipt);common.atomic(local/'remote.json',target)
        common.atomic(d/'.remote.json',target)
        return receipt,target

    def test_stopped_foreign_session_allows_setup_and_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            state=dict(owner='another-worktree',status='stopped')
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client,'source_identity',return_value={}),patch.object(client.subprocess,'run'),patch.object(client,'call',return_value={'status':'ready'}) as call,patch.object(client,'rpc') as rpc:
                client.main(['setup','--new-baseline'])
                call.assert_called_once();rpc.assert_not_called()
                client.main(['stop']);rpc.assert_not_called()

    def test_active_foreign_session_still_refuses_setup_before_sync(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            state=dict(owner='another-worktree',status='ready')
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client.subprocess,'run') as run:
                with self.assertRaises(SystemExit):client.main(['setup','--new-baseline'])
                run.assert_not_called()

    def test_busy_is_refused_before_sync(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value={'owner':'owner-a','status':'busy'}),patch.object(client.subprocess,'run') as run:
                with self.assertRaises(SystemExit) as e:client.main(['replay'])
                self.assertEqual(e.exception.code,2);run.assert_not_called()

    def test_failed_sync_cannot_measure_old_code(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            state={'owner':'owner-a','status':'ready','source':{}}
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client,'source_identity',return_value={}),patch.object(client.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['sync'])),patch.object(client,'rpc') as rpc:
                with self.assertRaises(SystemExit):client.main(['replay'])
                rpc.assert_not_called()

    def test_stop_uses_recorded_target_without_source_or_sync(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            (root/'.remote.json').write_text('broken configuration')
            state={'owner':'owner-a','status':'busy'}
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client,'source_identity',side_effect=AssertionError('must not inspect source')),patch.object(client,'rpc',return_value={'status':'stopped'}) as rpc,patch.object(client.subprocess,'run') as run:
                client.main(['stop'])
                run.assert_not_called()
                self.assertEqual(rpc.call_args.args[2],'stop')
                self.assertEqual(common.read(rpc.call_args.args[0]),target)

    def test_existing_setup_cannot_reset_baseline(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            state={'owner':'owner-a','status':'ready','config_hash':common.key(receipt['config'])}
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client.subprocess,'run') as run:
                client.main(['setup']);run.assert_not_called()
                with self.assertRaises(SystemExit):client.main(['setup','--new-baseline'])
                run.assert_not_called()

    def test_profiling_flag_only_applies_to_setup(self):
        for action in ('replay','forward','compare','profile','candidates','stop'):
            with self.subTest(action=action),patch.object(client,'probe') as probe:
                with self.assertRaises(SystemExit) as error:client.main([action,'--enable-profiling'])
                self.assertEqual(error.exception.code,2);probe.assert_not_called()

    def test_live_setup_reports_current_mode_and_explicit_enable_needs_restart(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,_=self.fixture(root)
            state=dict(owner='owner-a',status='ready',config_hash=common.key(receipt['config']),profiling_enabled=False)
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client.subprocess,'run') as run:
                with self.assertRaises(SystemExit):client.main(['setup','--enable-profiling'])
                state['profiling_enabled']=True
                client.main(['setup'])
                client.main(['setup','--enable-profiling'])
                run.assert_not_called()

    def test_stopped_setup_mode_is_explicit_and_does_not_change_config(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,_=self.fixture(root)
            state=dict(owner='owner-a',status='stopped',source={},profiling_enabled=True)
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=state),patch.object(client,'source_identity',return_value={}),patch.object(client.subprocess,'run'),patch.object(client,'call',return_value={'status':'ready'}) as call:
                for flags,enabled in ((['--enable-profiling'],True),([],False)):
                    client.main(['setup',*flags])
                    payload=call.call_args.args[2]
                    self.assertIs(payload['profiling_enabled'],enabled)
                    self.assertFalse(payload['new_baseline'])
                    self.assertEqual(payload['config'],receipt['config'])
                    self.assertEqual(common.read(root/'.scratch/lab/receipt.json')['config'],receipt['config'])

    def test_missing_explicit_profiler_fails_before_heavy_setup(self):
        payload=dict(config={'artifact':'/model','data_dir':'/data'},profiling_enabled=True)
        with patch.object(host.shutil,'which',return_value=None),patch.object(host.subprocess,'check_output') as check:
            with self.assertRaisesRegex(ValueError,'--enable-profiling requires'):
                host.start(Path('/unused'),'owner',payload)
            check.assert_not_called()

    def test_setup_mode_controls_launch_without_invalidating_baseline_or_corpus(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'kernel').mkdir()
            baseline=root/'baseline';baseline.mkdir()
            for name in ('kernel_lab','lab_prepare'):(baseline/name).write_text('pinned '+name)
            common.atomic(baseline/'complete.json',dict(source={},binaries={n:common.digest(baseline/n) for n in ('kernel_lab','lab_prepare')}))
            artifact=root/'model';artifact.write_bytes(b'fixed model')
            cfg=dict(artifact=str(artifact),data_dir=str(root/'data'))
            corpora=[];names=[];launched=[]
            def run(argv,log,*args,**kwargs):
                self.assertNotIn('cargo',argv)
                if argv[0]==str(baseline/'lab_prepare'):
                    common.atomic(argv[2],dict(cases=[dict(id='astropy__astropy-12907',tokens=[1])]))
            def session(*args,**kwargs):
                launched.append(kwargs['profiler_session'])
                return SimpleNamespace(process=SimpleNamespace(pid=123),engine_pid=123,ready={'pid':123},profiler_session=kwargs['profiler_session'])
            for enabled in (False,True,False):
                directory=root/('run-'+str(len(corpora)));directory.mkdir()
                state=dict(config=cfg,directory=str(directory),baseline=str(baseline),source={},session='test',profiling_enabled=enabled)
                service=host.Owner(root,state);service.publish=Mock()
                with patch.object(host.shutil,'which',return_value='/bin/nsys'),patch.object(host,'run',side_effect=run),patch.object(host.subprocess,'check_output',side_effect=['','RTX 5080, uuid, driver']),patch.object(host,'Session',side_effect=session):
                    service.setup()
                corpora.append((service.corpus,service.corpus_identity));names.append(service.names_path())
            self.assertEqual(launched,[None,'ignis-lab-test',None])
            self.assertEqual(corpora[0],corpora[1]);self.assertEqual(corpora[1],corpora[2])
            self.assertEqual(names[0],names[1]);self.assertEqual(names[1],names[2])

    def test_failed_restore_never_publishes_a_pass(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import build_trellis_candidate
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run_dir=root/'run';run_dir.mkdir()
            files={}
            for n in build_trellis_candidate.SOURCES:
                p=root/'kernel/src'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test')
                files[str(p.relative_to(root))]=common.digest(p)
            state=dict(config={'data_dir':str(root/'data')},directory=str(run_dir),source=files,config_hash='cfg')
            service=host.Owner(root,state)
            service.corpus=root/'corpus'
            service.engine=SimpleNamespace(process=SimpleNamespace(pid=123),
                reload=Mock(side_effect=[{'ok':True},ValueError('capture restore failure')]))
            service.publish=Mock()
            service.local=Mock(return_value={'passed':True})
            service.select_corpus=Mock(return_value=[])
            service.resolve_candidate=Mock(return_value={'path':'/candidate','identity':'candidate'})
            with patch.dict(host.os.environ,{'CUDA_HOME':'/test/cuda'}),patch.object(iteration,'source_identity',return_value=files),patch.object(host,'run'),patch.object(host.subprocess,'check_output',return_value='123\n'),patch.object(host,'build_candidate',return_value={'path':'/candidate','identity':'candidate'}):
                with self.assertRaises(RuntimeError):
                    service.evaluate({'source':files,'operation':'failure'},'replay')
            result=common.read(run_dir/'results/failure/result.json')
            self.assertFalse(result['passed'])
            self.assertIn('error',result)

    def test_start_exits_when_stopped_or_replaced(self):
        for terminal in ('stopped','replaced','missing'):
            with self.subTest(terminal=terminal),tempfile.TemporaryDirectory() as t:
                root=Path(t)
                payload=dict(config={'artifact':'/model','data_dir':str(root/'data')},
                    source={},new_baseline=True)
                state_file=root/'.scratch/lab/state.json'
                def launch(*args,**kwargs):
                    state=common.read(state_file)
                    if terminal=='missing':state_file.unlink()
                    else:
                        if terminal=='stopped':state['status']='stopped'
                        else:state.update(session='replacement',status='ready')
                        common.atomic(state_file,state)
                expected='cancelled' if terminal=='stopped' else 'replaced|missing'
                with patch.object(host,'status',return_value=None),patch.object(host,'source_identity',return_value={}),patch.object(host.shutil,'copy2'),patch.object(host.subprocess,'check_output',return_value='yes'),patch.object(host.subprocess,'run',side_effect=launch) as run,patch.object(host.time,'sleep',side_effect=AssertionError('must not keep waiting')):
                    with self.assertRaisesRegex(ValueError,expected):
                        host.start(root,'owner-a',payload)
                    self.assertEqual(run.call_count,1)

    def test_cancelled_setup_releases_command_lock_for_restart(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);receipt,target=self.fixture(root)
            with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=None),patch.object(client,'source_identity',return_value={}),patch.object(client.subprocess,'run'),patch.object(client,'call',side_effect=[ValueError('setup cancelled'),{'status':'ready'}]) as call:
                with self.assertRaises(SystemExit) as error:client.main(['setup'])
                self.assertEqual(error.exception.code,2)
                client.main(['setup'])
                self.assertEqual(call.call_count,2)


    def test_forward_comparison_checks_speculative_progress(self):
        base=dict(tokens=[1,2],input_tokens=4,finish_reason='length',stop_token=None,
                  logits_sha256='hash',rounds=[dict(emitted=2,extent=7,ms=10)])
        other=dict(base,rounds=[dict(emitted=2,extent=7,ms=9)])
        self.assertTrue(host.case_agrees(base,other))
        other['rounds'][0]['extent']=3
        self.assertFalse(host.case_agrees(base,other))

    def test_replay_checks_all_inputs_once_and_times_fixed_representatives(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            service=host.Owner(root,dict(config={'warmup':1,'samples':3},directory=t))
            service.corpus=root
            service.record_cache=[dict(id=i,weight={'name':'gate'},tokens=8,ldy=128,epilogue=0) for i in range(5)]
            calls=[]
            def reply(request):
                calls.append(request)
                return dict(ok=True,result=[dict(record=r['id'],passed=True,samples_ms=[1.0]) for r in request['records']])
            service.engine=SimpleNamespace(call=Mock(side_effect=reply))
            service.selection=[dict(data=str(root/'data.bin'),shard=str(root),records=service.record_cache)]
            result=service.local(root,'candidate')
            self.assertTrue(result['passed'])
            self.assertEqual((len(calls[0]['records']),calls[0]['warmup'],calls[0]['samples']),(5,0,1))
            self.assertEqual((len(calls[1]['records']),calls[1]['warmup'],calls[1]['samples']),(1,1,3))


    def test_fast_forward_uses_selected_request_once_full_uses_all(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            service=host.Owner(root,dict(config={'warmup':1,'samples':2},directory=t))
            service.cases=[dict(id=n) for n in ('a','b','c')];service.quick_case=service.cases[1]
            references=[dict(c,tokens=[1],rounds=[],prefill_ms=2,decode_ms=3) for c in service.cases]
            service.references={r['id']:r for r in references}
            service.engine=SimpleNamespace(call=Mock(side_effect=lambda req:dict(ok=True,result=next(r for r in references if r['id']==req['case']['id']))))
            self.assertTrue(service.forward(root,'quick')['passed'])
            self.assertEqual([c.args[0]['case']['id'] for c in service.engine.call.call_args_list],['b'])
            service.engine.call.reset_mock()
            self.assertTrue(service.forward(root,'full',full=True)['passed'])
            self.assertEqual(service.engine.call.call_count,9)

if __name__=='__main__':unittest.main()
