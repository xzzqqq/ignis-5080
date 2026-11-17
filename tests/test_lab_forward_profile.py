"""Forward profiling captures actual requests without replay corpora or hidden runs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import client, common, host, iteration, profiling


class ForwardProfileTests(unittest.TestCase):
    def owner(self, fail=False):
        calls=[]
        output=dict(id='one',tokens=[1,2],input_tokens=8,output_tokens=2,finish_reason='length',
            stop_token=None,logits_sha256='baseline',rounds=[dict(emitted=2,extent=8,ms=3)],prefill_ms=2,decode_ms=3)
        def call(request):
            calls.append(request)
            if request['op']=='benchmark':
                if fail:return dict(ok=False,error='benchmark execution failed')
                return dict(ok=True,result=dict(output))
            return dict(ok=True,result={})
        def reload(path,profile_recapture=False):
            return call(dict(op='reload',path=path,profile_recapture=profile_recapture))
        return SimpleNamespace(engine=SimpleNamespace(profiler_session='session',call=call,reload=reload),
            config=dict(warmup=7,samples=9,timeout_s=1),quick_case=dict(id='one',tokens=[4]*8,output_tokens=2),
            calls=calls,references={}),output

    @contextlib.contextmanager
    def collector(self, directory):
        commands=[]
        def command(argv, log, required=True, timeout=None):
            commands.append(argv)
            if argv[1]=='stop':(directory/'forward-profile.nsys-rep').touch()
            return 0
        with patch.object(profiling.shutil,'which',return_value='/bin/nsys'),patch.object(profiling,'command',side_effect=command),\
                patch.object(profiling,'summarize',return_value=dict(kernel_count=999,timeline=[],kernels=[],devices=[])),\
                patch('lab_support.forward_timeline.attribute_forward',return_value=dict(available=True,coverage={},instances=[],aggregates=[],timeline=[],
                    request_summary=dict(kernel_count=1,kernels=[],devices=[],timeline=[]))):
            yield commands

    def test_default_captures_exactly_one_natural_request_without_reference_run(self):
        owner,_=self.owner()
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            with self.collector(directory) as commands:
                result=profiling.profile(owner,directory,dict(profile_forward=True),dict(identity='saved',path='/saved'))
            self.assertEqual([r['op'] for r in owner.calls],['profiler_start','reload','benchmark','profiler_stop'])
            self.assertEqual(owner.calls[2]['profile_sample'],0)
            self.assertEqual(owner.calls[2]['case'],owner.quick_case)
            self.assertEqual((result['warmup'],result['samples']),(0,1))
            self.assertEqual(result['kernel_count'],1,'setup kernels leaked into request summary')
            self.assertEqual(result['mode'],'forward');self.assertEqual(result['timing_kind'],'instrumented')
            self.assertFalse(result['reference_available']);self.assertIsNone(result['forward']['numerical_match'])
            row=result['forward']['cases'][0]['samples'][0]
            self.assertIsNone(row['differences']);self.assertEqual(row['tokens_per_round'],2)
            self.assertEqual(row['finish_reason'],'length');self.assertEqual(row['tokens'],[1,2])
            self.assertEqual([r[1] for r in commands],['start','stop','export'])
            self.assertTrue((directory/'profile-forward.json').is_file())
            self.assertIn('attribution',common.read(directory/'profile-timeline.json'))

    def test_explicit_warmups_outside_capture_samples_labeled_and_cached_reference_reused(self):
        owner,reference=self.owner();owner.references={'one':dict(reference,tokens=[9,9])}
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            with self.collector(directory):
                result=profiling.profile(owner,directory,dict(profile_forward=True,warmup=2,samples=3),dict(identity='current',path='/current'))
            self.assertEqual([r['op'] for r in owner.calls],['benchmark','benchmark','profiler_start',
                'reload','benchmark','benchmark','benchmark','profiler_stop'])
            captured=[r for r in owner.calls if 'profile_sample' in r]
            self.assertEqual([r['profile_sample'] for r in captured],[0,1,2])
            self.assertEqual((result['warmup'],result['samples']),(2,3))
            self.assertTrue(result['reference_available']);self.assertFalse(result['forward']['numerical_match'])
            self.assertTrue(all('tokens' in r['differences'] for r in result['forward']['cases'][0]['samples']))

    def test_benchmark_failure_stops_native_and_nsight_ranges(self):
        owner,_=self.owner(fail=True)
        with tempfile.TemporaryDirectory() as t:
            with self.collector(Path(t)) as commands:
                with self.assertRaisesRegex(RuntimeError,'benchmark execution failed'):
                    profiling.profile(owner,Path(t),dict(profile_forward=True),dict(identity='saved',path='/saved'))
            self.assertEqual([r['op'] for r in owner.calls],['profiler_start','reload','benchmark','profiler_stop'])
            self.assertEqual(commands[-1][1],'stop')

    def test_missing_sample_attribution_fails_with_raw_outputs_preserved(self):
        owner,_=self.owner()
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            with self.collector(directory),patch('lab_support.forward_timeline.attribute_forward',return_value=dict(
                    available=False,coverage={},timeline=[],request_summary=dict(kernel_count=0,kernels=[],devices=[]))):
                with self.assertRaisesRegex(ValueError,'no sample-attributed CUDA kernels'):
                    profiling.profile(owner,directory,dict(profile_forward=True),dict(identity='saved',path='/saved'))
            self.assertTrue((directory/'profile-forward.json').is_file())
            self.assertTrue((directory/'profile-timeline.json').is_file())
            self.assertEqual(owner.calls[-1]['op'],'profiler_stop')

    def test_recapture_failure_closes_range_without_running_the_request(self):
        owner,_=self.owner();owner.engine.reload=Mock(side_effect=RuntimeError('graph recapture failed'))
        with tempfile.TemporaryDirectory() as t:
            with self.collector(Path(t)) as commands:
                with self.assertRaisesRegex(RuntimeError,'graph recapture failed'):
                    profiling.profile(owner,Path(t),dict(profile_forward=True),dict(identity='saved',path='/saved'))
            self.assertEqual([r['op'] for r in owner.calls],['profiler_start','profiler_stop'])
            self.assertEqual(commands[-1][1],'stop')

    def test_nvtx_enabled_only_for_optional_profiled_session(self):
        import kernel_lab
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as t:
                root=Path(t);process=SimpleNamespace(pid=100,stdin=io.StringIO(),stdout=io.StringIO(),poll=lambda:0)
                with patch.object(kernel_lab.subprocess,'Popen',return_value=process) as launch,                        patch.object(kernel_lab.Session,'read',return_value={'event':'ready','pid':200}):
                    session=kernel_lab.Session(Path('/engine'),Path('/model'),Path('/settings'),root,10,
                        profiler_session='session' if enabled else None)
                    try:
                        self.assertEqual(launch.call_args.kwargs['env']['IGNIS_LAB_PROFILE'],'1' if enabled else '0')
                        if enabled:
                            self.assertIn('--trace=cuda,nvtx',launch.call_args.args[0])
                            self.assertIn('--cuda-graph-trace=node',launch.call_args.args[0])
                        else:self.assertEqual(launch.call_args.args[0][0],'/engine')
                    finally:session.close()

    def test_owner_dispatch_skips_corpus_and_baseline_and_restores_after_profile(self):
        for selector in ('current','saved','builtin'):
            with self.subTest(selector=selector),tempfile.TemporaryDirectory() as t:
                root=Path(t);owner=host.Owner(root,dict(config={},directory=t,source={},config_hash='fixed',profiling_enabled=True))
                owner.publish=Mock();owner.engine=SimpleNamespace(process=SimpleNamespace(pid=12),engine_pid=12,
                    profiler_session='session',reload=Mock(return_value={}))
                owner.resolve_candidate=Mock(return_value=dict(identity=selector,path=selector))
                owner.select_corpus=Mock(side_effect=AssertionError('must not record operators'))
                owner.reference_forward=Mock(side_effect=AssertionError('must not run hidden baseline'))
                with patch.object(iteration,'source_identity',return_value={}),patch.object(host,'run'),\
                        patch.object(iteration.subprocess,'check_output',return_value='12\n'),\
                        patch.object(iteration.shutil,'which',return_value='/bin/nsys'),\
                        patch.object(profiling,'profile',return_value={'passed':True,'mode':'forward'}) as profile:
                    result=owner.evaluate(dict(source={},operation='forward-profile',candidate=selector,profile_forward=True),'profile')
                self.assertTrue(result['passed']);self.assertEqual(result['engine_pid'],12)
                owner.resolve_candidate.assert_called_once_with(selector,root/'results/forward-profile')
                self.assertTrue(profile.call_args.args[2]['profile_forward'])
                self.assertEqual([c.args[0] for c in owner.engine.reload.call_args_list],[selector,'builtin'])

    def test_owner_restores_builtin_when_profile_execution_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);owner=host.Owner(root,dict(config={},directory=t,source={},config_hash='fixed',profiling_enabled=True))
            owner.publish=Mock();owner.engine=SimpleNamespace(process=SimpleNamespace(pid=12),engine_pid=12,
                profiler_session='session',reload=Mock(return_value={}))
            owner.resolve_candidate=Mock(return_value=dict(identity='saved',path='/saved'))
            with patch.object(iteration,'source_identity',return_value={}),patch.object(host,'run'),\
                    patch.object(iteration.subprocess,'check_output',return_value='12\n'),\
                    patch.object(iteration.shutil,'which',return_value='/bin/nsys'),\
                    patch.object(profiling,'profile',side_effect=RuntimeError('capture execution failed')):
                with self.assertRaisesRegex(RuntimeError,'capture execution failed'):
                    owner.evaluate(dict(source={},operation='failed-profile',profile_forward=True),'profile')
            result=common.read(root/'results/failed-profile/result.json')
            self.assertFalse(result['passed']);self.assertTrue(result['finished'])
            self.assertEqual(owner.engine.reload.call_args_list[-1].args,('builtin',))

    def test_profile_rpc_returns_aggregates_and_keeps_full_timeline_only_in_artifact(self):
        owner,_=self.owner()
        count=10000
        attribution=dict(schema_version=1,available=True,coverage=dict(kernels=count),
            capture_coverage=dict(request=count,setup=1,unplaced=0),limitations=['unattributed work stays visible'],
            instances=[dict(sample=0,phase='decode_round',layer=i) for i in range(count)],
            aggregates=[dict(sample=0,phase='decode_round',layer=0)],
            timeline=[dict(kernel=i,duration_us=1) for i in range(count)],
            request_summary=dict(kernel_count=count,kernels=[],devices=[]))
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            with self.collector(directory),patch('lab_support.forward_timeline.attribute_forward',return_value=attribution):
                result=profiling.profile(owner,directory,dict(profile_forward=True),dict(identity='saved',path='/saved'))
            returned=result['attribution']
            self.assertNotIn('instances',returned);self.assertNotIn('timeline',returned)
            self.assertEqual((returned['instances_count'],returned['timeline_count']),(count,count))
            self.assertEqual(returned['coverage']['kernels'],count)
            self.assertEqual(len(returned['aggregates']),1)
            self.assertLess(len(json.dumps(result)),4000)
            stored=common.read(directory/'profile-timeline.json')['attribution']
            self.assertEqual(len(stored['instances']),count);self.assertEqual(len(stored['timeline']),count)
            compact=client.summary(dict(operation='one',passed=True,profile=result),'/local/full.json')
            self.assertEqual(compact['profile']['attribution']['instances_count'],count)
            self.assertNotIn('instances',compact['profile']['attribution'])

    def test_compact_summary_bounds_attribution_and_natural_outputs(self):
        owner,_=self.owner()
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            with self.collector(directory):
                profile=profiling.profile(owner,directory,dict(profile_forward=True),dict(identity='saved',path='/saved'))
            profile['attribution']=dict(available=True,coverage={'kernels':10000},limitations=['unattributed work remains visible'],
                instances=[dict(sample=i,phase='decode_round',layer=0) for i in range(10000)],
                aggregates=[dict(phase='decode_round',layer=i) for i in range(10000)],timeline=[dict(kernel=i) for i in range(10000)])
            result=client.summary(dict(operation='one',passed=True,profile=profile),'/full.json')
            self.assertLess(len(json.dumps(result)),5000)
            self.assertNotIn('timeline',result['profile']['attribution'])
            self.assertEqual(result['profile']['attribution']['instances_count'],10000)
            self.assertEqual(len(result['profile']['attribution']['instances']),5)
            self.assertEqual(len(result['profile']['attribution']['aggregates']),20)
            self.assertFalse(result['profile']['forward']['reference_available'])
            self.assertNotIn('tokens',result['profile']['forward']['cases'][0])

    def test_cli_rejects_operator_controls_and_full_for_forward_profile(self):
        for args in (['profile','--forward','--operator','gate'],['profile','--forward','--tokens-per-call','1'],
                     ['profile','--forward','--full']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),patch.object(client,'probe') as probe:
                with self.assertRaises(SystemExit) as error:client.main(args)
                self.assertEqual(error.exception.code,2);probe.assert_not_called()

    def test_cli_dispatches_forward_profile_and_preserves_explicit_budgets(self):
        for options in ([],['--warmup','2','--samples','3']):
            with self.subTest(options=options),tempfile.TemporaryDirectory() as t:
                root=Path(t);local=root/'.scratch/lab';local.mkdir(parents=True)
                target=dict(host='gpu-test',repo='/test/lab');config=dict(artifact='/model',data_dir='/data')
                common.atomic(local/'receipt.json',dict(owner='owner',config=config))
                common.atomic(local/'remote.json',target);common.atomic(root/'.remote.json',target)
                def rpc(target,state,action,payload):return dict(passed=True,operation=payload['operation'])
                with patch.object(client,'ROOT',root),patch.object(client,'probe',return_value=dict(owner='owner',status='ready',source={})),\
                        patch.object(client,'source_identity',return_value={}),patch.object(client.subprocess,'run'),\
                        patch.object(client,'rpc',side_effect=rpc) as call,contextlib.redirect_stdout(io.StringIO()):
                    client.main(['profile','--forward','--candidate','saved',*options])
                payload=call.call_args.args[3]
                self.assertTrue(payload['profile_forward']);self.assertFalse(payload['compare_forward'])
                self.assertEqual(payload['candidate'],'saved')
                if options:self.assertEqual((payload['warmup'],payload['samples']),(2,3))
                else:self.assertNotIn('warmup',payload);self.assertNotIn('samples',payload)


if __name__=='__main__':unittest.main()
