"""CPU contracts for four-command ownership, source freshness and corpus reuse."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import client, common, host


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
            with patch.dict(host.os.environ,{'CUDA_HOME':'/test/cuda'}),patch.object(host,'source_identity',return_value=files),patch.object(host,'run'),patch.object(host.subprocess,'check_output',return_value='123\n'),patch.object(host,'build_candidate',return_value={'path':'/candidate'}):
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

    def test_failed_candidate_forward_still_checks_restored_full_model(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import build_trellis_candidate
        for restored in (True,False):
            with self.subTest(restored=restored),tempfile.TemporaryDirectory() as t:
                root=Path(t);run_dir=root/'run';run_dir.mkdir()
                files={}
                for n in build_trellis_candidate.SOURCES:
                    p=root/'kernel/src'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test')
                    files[str(p.relative_to(root))]=common.digest(p)
                state=dict(config={'data_dir':str(root/'data')},directory=str(run_dir),source=files,config_hash='cfg')
                service=host.Owner(root,state);service.corpus=root/'corpus'
                service.engine=SimpleNamespace(process=SimpleNamespace(pid=123),reload=Mock(return_value={'ok':True}))
                service.publish=Mock();service.local=Mock(return_value={'passed':True})
                service.forward=Mock(side_effect=[{'passed':True},{'passed':False},{'passed':restored}])
                with patch.dict(host.os.environ,{'CUDA_HOME':'/test/cuda'}),patch.object(host,'source_identity',return_value=files),patch.object(host,'run'),patch.object(host.subprocess,'check_output',return_value='123\n'),patch.object(host,'build_candidate',return_value={'path':'/candidate'}):
                    if restored:service.evaluate({'source':files,'operation':'failure'},'forward')
                    else:
                        with self.assertRaisesRegex(RuntimeError,'baseline forward changed'):
                            service.evaluate({'source':files,'operation':'failure'},'forward')
                self.assertEqual([c.args[1] for c in service.forward.call_args_list],['before','candidate','after'])
                self.assertEqual(service.engine.reload.call_args_list[-1].args,('builtin',))
                result=common.read(run_dir/'results/failure/result.json')
                self.assertTrue(result['finished']);self.assertFalse(result['passed'])
                self.assertEqual(result['after_forward']['passed'],restored)
                self.assertEqual(any(c.kwargs.get('status')=='ready' for c in service.publish.call_args_list),restored)

    def test_forward_comparison_checks_speculative_progress(self):
        base=dict(tokens=[1,2],input_tokens=4,finish_reason='length',stop_token=None,
                  logits_sha256='hash',rounds=[dict(emitted=2,extent=7,ms=10)])
        other=dict(base,rounds=[dict(emitted=2,extent=7,ms=9)])
        self.assertTrue(host.case_agrees(base,other))
        other['rounds'][0]['extent']=3
        self.assertFalse(host.case_agrees(base,other))

if __name__=='__main__':unittest.main()
