"""CPU coverage for exploratory replay, scope caching and immutable snapshots."""
import json
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import common, host, iteration, client


class IterationTests(unittest.TestCase):
    def owner(self, root):
        value=host.Owner(root,dict(config={'data_dir':str(root/'data')},directory=str(root),source={},config_hash='cfg'))
        value.corpus=root/'corpora';value.corpus.mkdir()
        value.corpus_identity={'baseline':'fixed'}
        return value

    def shard(self, owner, operators, tokens):
        scope=dict(operators=operators,tokens_per_call=tokens)
        path=owner.corpus/common.key(scope);path.mkdir()
        records=[dict(id=i,weight={'name':name},tokens=token,ldy=128,epilogue=0)
                 for i,(name,token) in enumerate((n,t) for n in operators or ['gate','down'] for t in tokens or [1,8])]
        (path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (path/'data.bin').write_bytes(b'frozen')
        common.atomic(path/'manifest.json',dict(identity=owner.corpus_identity,scope=scope,complete=True,
            files={n:common.digest(path/n) for n in ('records.jsonl','data.bin')}))
        return path,common.read(path/'manifest.json')

    def test_many_missing_operators_share_one_capture_and_reuse_shards(self):
        with tempfile.TemporaryDirectory() as t:
            value=self.owner(Path(t));self.shard(value,['gate'],[1])
            value.capture=Mock(side_effect=lambda names,tokens:self.shard(value,names,tokens))
            selected=value.select_corpus(dict(operators=['gate','down','up'],tokens_per_call=[1,8]))
            value.capture.assert_called_once_with(['down','gate','up'],[1,8])
            self.assertEqual(sum(len(g['records']) for g in selected),6)
            value.capture.reset_mock()
            selected=value.select_corpus(dict(operators=['gate','down'],tokens_per_call=[1]))
            value.capture.assert_not_called()
            self.assertEqual(sum(len(g['records']) for g in selected),2)

    def test_omitted_selectors_expand_beyond_cached_narrow_scope(self):
        with tempfile.TemporaryDirectory() as t:
            value=self.owner(Path(t));self.shard(value,['gate'],[1])
            value.capture=Mock(side_effect=lambda names,tokens:self.shard(value,names,tokens))
            groups=value.select_corpus({})
            value.capture.assert_called_once_with([],[])
            self.assertEqual(sum(len(g['records']) for g in groups),4)

    def test_cached_replay_does_not_rehash_capture_data(self):
        with tempfile.TemporaryDirectory() as t:
            value=self.owner(Path(t));self.shard(value,['gate'],[1])
            with patch.object(host,'digest',wraps=common.digest) as digest:
                value.select_corpus(dict(operators=['gate'],tokens_per_call=[1]))
                count=digest.call_count
                self.assertEqual(count,2)
                value.select_corpus(dict(operators=['gate'],tokens_per_call=[1]))
                self.assertEqual(digest.call_count,count)
                (next(value.corpus.glob('*/data.bin'))).write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'corrupt'):
                    value.select_corpus(dict(operators=['gate'],tokens_per_call=[1]))

    def test_fresh_capture_is_verified_without_rehash_until_file_changes(self):
        with tempfile.TemporaryDirectory() as t:
            value=self.owner(Path(t))
            value.quick_case=dict(id='quick',tokens=[1])
            reference=dict(tokens=[2],input_tokens=1,finish_reason='length',stop_token=None,
                           logits_sha256='fixed',rounds=[])
            value.reference_forward=Mock(return_value={'quick':reference})
            record=dict(id=0,weight={'name':'gate'},tokens=1)
            def engine_call(request):
                result={}
                if request['op']=='capture_minimum':result=dict(prefill_bytes_lower_bound=6)
                elif request['op']=='capture_start':
                    path=Path(request['path'])
                    (path/'records.jsonl').write_text(json.dumps(record)+'\n')
                    (path/'data.bin').write_bytes(b'frozen')
                elif request['op']=='benchmark':result=reference
                return dict(ok=True,result=result)
            value.engine=SimpleNamespace(call=Mock(side_effect=engine_call))
            payload=dict(operators=['gate'],tokens_per_call=[1])
            with patch.object(iteration,'digest',wraps=common.digest) as capture_digest, \
                 patch.object(host,'digest',wraps=common.digest) as verify_digest:
                groups=value.select_corpus(payload)
                path=Path(groups[0]['shard'])
                self.assertEqual(groups[0]['records'],[record])
                self.assertEqual(capture_digest.call_count,2)
                self.assertEqual({call.args[0].name for call in capture_digest.call_args_list},
                                 {'data.bin','records.jsonl'})
                value.verify_shard(path)
                value.select_corpus(payload)
                verify_digest.assert_not_called()
                self.assertEqual(capture_digest.call_count,2)
                (path/'data.bin').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'corrupt corpus: data.bin'):
                    value.select_corpus(payload)
                verify_digest.assert_called_once_with(path/'data.bin')

    def hashed_shard(self, owner, records=4):
        path,manifest=self.shard(owner,['gate'],[1])
        rows=[dict(id=i,source_record=i+40,weight={'name':'gate'},tokens=1,
                   offset=i*4,bytes=4) for i in range(records)]
        (path/'data.bin').write_bytes(b''.join(bytes([i])*4 for i in range(records)))
        for row in rows:row['data_sha256']=iteration.range_digest(path/'data.bin',row['offset'],row['bytes'])
        (path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest.update(records=len(rows),files={n:common.digest(path/n) for n in ('records.jsonl','data.bin')})
        common.atomic(path/'manifest.json',manifest)
        return path,rows

    def test_publication_hashes_full_stream_and_records_in_one_bounded_pass(self):
        data=b'a'*(2<<20)+b'bc'
        rows=[dict(id=0,offset=0,bytes=2<<20),dict(id=1,offset=2<<20,bytes=2)]
        sizes=[]
        class Stream(io.BytesIO):
            def read(self,size=-1):
                sizes.append(size)
                return super().read(size)
        with patch.object(Path,'open',return_value=Stream(data)) as opened, \
             patch.object(iteration,'digest',side_effect=AssertionError('must not rehash whole file')):
            actual=iteration.publish_record_integrity(Path('/data.bin'),rows)
        opened.assert_called_once_with('rb')
        self.assertEqual(actual,hashlib.sha256(data).hexdigest())
        self.assertEqual([r['data_sha256'] for r in rows],[hashlib.sha256(data[:-2]).hexdigest(),hashlib.sha256(b'bc').hexdigest()])
        self.assertTrue(all(0<size<=1<<20 for size in sizes))

    def test_selected_existing_record_hashes_only_selected_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,rows=self.hashed_shard(owner)
            payload=dict(corpus=str(path),record_start=2,record_limit=1)
            owner.capture=Mock(side_effect=AssertionError('existing corpus must not capture'))
            with patch.object(iteration,'digest',wraps=common.digest) as digest, \
                 patch.object(iteration,'range_digest',wraps=iteration.range_digest) as ranges:
                groups=owner.select_corpus(payload)
            self.assertEqual(groups[0]['records'],[rows[2]])
            self.assertEqual([c.args[0].name for c in digest.call_args_list],['records.jsonl'])
            ranges.assert_called_once_with(path/'data.bin',8,4)
            self.assertEqual(iteration.coverage(groups)['mode'],'sample')
            # Changes outside the explicitly selected record do not read or invalidate it.
            with (path/'data.bin').open('r+b') as data:data.write(b'XXXX')
            self.assertEqual(owner.select_corpus(payload)[0]['records'],[rows[2]])
            with (path/'data.bin').open('r+b') as data:data.seek(8);data.write(b'XXXX')
            with self.assertRaisesRegex(ValueError,'corrupt corpus record: 2'):owner.select_corpus(payload)

    def test_existing_sample_orders_source_calls_before_slicing(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,rows=self.hashed_shard(owner)
            rows[0]['source_record']=43;rows[3]['source_record']=40
            (path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest=common.read(path/'manifest.json')
            manifest['files']['records.jsonl']=common.digest(path/'records.jsonl')
            common.atomic(path/'manifest.json',manifest)
            groups=owner.select_corpus(dict(corpus=str(path),record_start=0,record_limit=1))
            self.assertEqual(groups[0]['records'],[rows[3]])
            self.assertEqual(iteration.coverage(groups)['shards'][0]['source_records'],[40])

    def test_record_selectors_reject_invalid_or_overflow_values_before_capture(self):
        owner=iteration.Iteration();owner.capture=Mock(side_effect=AssertionError('must not capture'))
        for name in ('record_start','record_limit'):
            for invalid in (None,True,False,-1,1.0,'1',1<<64):
                with self.subTest(name=name,invalid=invalid),self.assertRaisesRegex(ValueError,'unsigned 64-bit'):
                    owner.select_corpus({name:invalid})
        owner.capture.assert_not_called()
        self.assertEqual(iteration.record_window(dict(record_start=(1<<64)-1,record_limit=(1<<64)-1)),((1<<64)-1,(1<<64)-1))

    def test_existing_sample_rejects_invalid_record_ordinals_before_engine(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,rows=self.hashed_shard(owner)
            owner.engine=SimpleNamespace(call=Mock(side_effect=AssertionError('must not replay')))
            for field in ('id','source_record'):
                for invalid in (None,True,-1,'2',1.5,1<<64):
                    malformed=[dict(r) for r in rows];malformed[0][field]=invalid
                    (path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in malformed))
                    with self.subTest(field=field,invalid=invalid),self.assertRaisesRegex(ValueError,'invalid corpus record '+field):
                        owner.select_corpus(dict(corpus=str(path),record_limit=1))
            owner.engine.call.assert_not_called()

    def test_selected_metadata_and_legacy_integrity_remain_checked(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,rows=self.hashed_shard(owner)
            with (path/'records.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'corrupt corpus: records.jsonl'):
                owner.select_corpus(dict(corpus=str(path),record_limit=1))
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,_=self.shard(owner,['gate'],[1])
            with patch.object(host,'digest',wraps=common.digest) as digest:
                owner.select_corpus(dict(corpus=str(path),record_limit=1))
            self.assertEqual({c.args[0].name for c in digest.call_args_list},{'data.bin','records.jsonl'})

    def test_limited_scope_never_satisfies_full_capture(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));path,manifest=self.shard(owner,['gate'],[1])
            limited=dict(operators=['gate'],tokens_per_call=[1],record_start=5,record_limit=1)
            manifest['scope']=limited
            common.atomic(path/'manifest.json',manifest)
            new_path=path.with_name(common.key(limited));path.rename(new_path)
            self.assertFalse(iteration.covers(limited,['gate'],[1]))
            owner.capture=Mock(side_effect=lambda names,tokens:self.shard(owner,names,tokens))
            owner.select_corpus(dict(operators=['gate'],tokens_per_call=[1]))
            owner.capture.assert_called_once_with(['gate'],[1])

    def test_capture_window_reaches_engine_before_publication_and_keeps_source_id(self):
        with tempfile.TemporaryDirectory() as t:
            owner=self.owner(Path(t));owner.quick_case=dict(id='quick',tokens=[1])
            reference=dict(tokens=[2],input_tokens=1,finish_reason='length',stop_token=None,logits_sha256='x',rounds=[])
            owner.reference_forward=Mock(return_value={'quick':reference})
            def engine(request):
                if request['op']=='capture_minimum':return dict(ok=True,result=dict(prefill_bytes_lower_bound=10**30))
                if request['op']=='capture_start':
                    self.assertEqual((request['record_start'],request['record_limit']),(42,1))
                    path=Path(request['path']);(path/'data.bin').write_bytes(b'data')
                    (path/'records.jsonl').write_text(json.dumps(dict(id=0,source_record=42,weight={'name':'gate'},tokens=1,offset=0,bytes=4))+'\n')
                return dict(ok=True,result=reference if request['op']=='benchmark' else {})
            owner.engine=SimpleNamespace(call=Mock(side_effect=engine))
            with patch.object(iteration,'range_digest',side_effect=AssertionError('fresh capture must not reread data')):
                groups=owner.select_corpus(dict(operators=['gate'],tokens_per_call=[1],record_start=42,record_limit=1))
            self.assertEqual(groups[0]['records'][0]['source_record'],42)
            self.assertEqual(groups[0]['records'][0]['data_sha256'],common.digest(Path(groups[0]['data'])))
            self.assertEqual(groups[0]['manifest']['scope']['record_limit'],1)
            self.assertEqual(iteration.coverage(groups)['mode'],'sample')

    def test_export_replays_selected_candidate_and_enriches_native_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);owner=self.owner(root);path,rows=self.hashed_shard(owner)
            owner.selection=owner.select_corpus(dict(corpus=str(path),record_start=2,record_limit=1))
            exported=root/'tensors'/'record-2';exported.mkdir(parents=True)
            (exported/'candidate.bin').write_bytes(b'actual')
            manifest=exported/'manifest.json'
            common.atomic(manifest,dict(tensors=[dict(name='output',purpose='candidate_output',dtype='bf16',shape=[3],file='candidate.bin',offset=0,bytes=6)]))
            owner.engine=SimpleNamespace(call=Mock(return_value=dict(ok=True,result=[dict(record=2,export=dict(path=str(exported),manifest=str(manifest)))])))
            owner.evaluation_source={'kernel/source':'sha'}
            result=owner.export_record(root,dict(identity='candidate',path='/candidate.so'))
            request=owner.engine.call.call_args.args[0]
            self.assertEqual(request['records'],[rows[2]])
            self.assertEqual((request['warmup'],request['samples']),(0,1))
            final=common.read(manifest)
            self.assertEqual(final['tensors'][0]['sha256'],common.digest(exported/'candidate.bin'))
            self.assertEqual(final['source'],owner.evaluation_source)
            self.assertEqual(final['candidate']['identity'],'candidate')
            self.assertEqual(final['corpus']['record']['id'],2)
            self.assertEqual(result['coverage']['mode'],'sample')

    def test_explicit_legacy_sample_times_every_selected_record(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);owner=self.owner(root);path,rows=self.hashed_shard(owner)
            owner.selection=owner.select_corpus(dict(corpus=str(path),record_start=1,record_limit=3))
            def replay(request):
                return dict(ok=True,result=[dict(record=r['id'],mismatches=0,samples_ms=[1.0]) for r in request['records']])
            owner.engine=SimpleNamespace(call=Mock(side_effect=replay))
            result=owner.local(root,'sample',warmup=0,samples=1)
            self.assertEqual(result['records'],3)
            self.assertEqual(result['performance_records'],3)
            self.assertEqual([r['record'] for r in result['timings']],[1,2,3])

    def test_sample_compare_reuses_exact_record_ids_for_all_legs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);owner=self.owner(root);path,rows=self.hashed_shard(owner)
            owner.selection=owner.select_corpus(dict(corpus=str(path),record_start=2,record_limit=1))
            def replay(request):
                return dict(ok=True,result=[dict(record=r['id'],mismatches=0,samples_ms=[1.0]) for r in request['records']])
            owner.engine=SimpleNamespace(call=Mock(side_effect=replay),reload=Mock())
            result=owner.compare(root,dict(samples=2,warmup=0),dict(identity='before',path='/before'),dict(identity='after',path='/after'))
            self.assertTrue(all(c.args[0]['records']==[rows[2]] for c in owner.engine.call.call_args_list))
            self.assertEqual(result['coverage']['mode'],'sample')
            self.assertEqual(result['coverage']['shards'][0]['record_ids'],[2])
            self.assertEqual(result['coverage']['shards'][0]['source_records'],[42])

    def test_cli_rejects_inapplicable_record_controls_before_remote_calls(self):
        invalid=[['export'],['export','--record-limit','2'],['replay','--record-start','-1'],
            ['forward','--record-limit','1'],['setup','--corpus','/remote/shard'],
            ['replay','--record-start',str(1<<64)],['replay','--record-limit',str(1<<64)],
            ['profile','--forward','--record-limit','0'],['replay','--corpus','relative']]
        with patch.object(client,'probe') as probe:
            for args in invalid:
                with self.subTest(args=args),self.assertRaises(SystemExit) as error:client.main(args)
                self.assertEqual(error.exception.code,2)
            probe.assert_not_called()

    def test_numerical_mismatch_still_times_representatives(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);calls=[]
            rows=[dict(id=i,weight={'name':'gate'},tokens=1,ldy=128,epilogue=0) for i in range(3)]
            value.selection=[dict(data='/data',shard='scope',records=rows)]
            def replay(request):
                calls.append(request)
                return dict(ok=True,result=[dict(record=r['id'],passed=False,mismatches=2,samples_ms=[1.0],
                    diagnostics=dict(max_abs_error=0.5,relative_l2=0.1,actual_nan=0,actual_inf=0)) for r in request['records']])
            value.engine=SimpleNamespace(call=Mock(side_effect=replay))
            result=value.local(root,'candidate',warmup=2,samples=4)
            self.assertTrue(result['passed']);self.assertFalse(result['numerical_match'])
            self.assertEqual(len(result['diagnostics']),3);self.assertEqual(len(result['timings']),1)
            self.assertTrue(all(c['exploratory'] for c in calls))
            self.assertEqual((calls[1]['warmup'],calls[1]['samples']),(2,4))

    def test_gqa_representatives_preserve_each_recorded_window_and_mask(self):
        value=iteration.Iteration()
        records=[dict(id=i,kind='gqa_attention',weight={'name':'layer.3.attention'},tokens=8,
                      position=64+i,max_keys=1024,tensors=[{'shape':[8,1,1,1]}]) for i in range(3)]
        groups=[dict(shard='gqa',records=records)]
        self.assertEqual(value.representative_scope(groups)[0]['records'],records)
        self.assertTrue(iteration.selected(records[1],['layer.3.attention'],[8]))
        self.assertFalse(iteration.selected(records[1],['layer.3.output'],[8]))

    def test_mixed_block_and_legacy_selectors_are_rejected_before_capture(self):
        value=iteration.Iteration();value.capture=Mock(side_effect=AssertionError('must not capture'))
        with self.assertRaisesRegex(ValueError,'cannot be mixed'):
            value.select_corpus(dict(operators=['layer.0.mlp_block','layer.0.down'],tokens_per_call=[8]))
        value.capture.assert_not_called()

    def test_finite_segment_records_keep_distinct_state_and_boundaries(self):
        value=iteration.Iteration()
        records=[dict(id=i,kind='segment',weight={'name':name},tokens=8,profile={'kind':kind})
            for i,(name,kind) in enumerate([('layer.0.mlp_block',1),('layer.3.gqa_block',2),('layer.0.gdn_block',3),('layer.0.gdn_block',3)])]
        groups=[dict(records=records)]
        self.assertEqual(value.representative_scope(groups)[0]['records'],records)
        self.assertTrue(iteration.selected(records[2],['layer.0.gdn_block'],[8]))
        self.assertFalse(iteration.selected(records[2],['layer.0.mlp_block'],[8]))

    def test_gqa_state_differences_are_separate_exploratory_diagnostics(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root)
            record=dict(id=0,kind='gqa_attention',weight={'name':'layer.3.attention'},tokens=8)
            value.selection=[dict(data='/data',shard='gqa',records=[record])]
            value.engine=SimpleNamespace(call=Mock(return_value=dict(ok=True,result=[dict(record=0,
                mismatches=0,passed=False,state_diagnostics=dict(exact_match=False,mismatched_bytes=1),samples_ms=[1.0])])) )
            result=value.local(root,'gqa',warmup=0,samples=1)
            self.assertTrue(result['passed']);self.assertTrue(result['numerical_match'])
            self.assertFalse(result['state_match']);self.assertEqual(len(result['timings']),1)

    def test_saved_names_are_immutable_and_check_library_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);library=root/'candidate.so';library.write_bytes(b'original')
            candidate=dict(identity='one',path=str(library),sha256=common.digest(library))
            value.save_candidate('baseline-1',candidate)
            self.assertEqual(value.resolve_candidate('baseline-1',root),candidate)
            with self.assertRaisesRegex(ValueError,'already identifies'):
                value.save_candidate('baseline-1',dict(candidate,identity='two'))
            library.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                value.resolve_candidate('baseline-1',root)
            value.corpus_identity={'baseline':'different'}
            self.assertEqual(value.candidates(),{})

    def test_compare_checks_fixed_inputs_once_and_warms_every_timing_leg(self):
        for pairs in (3,30):
            with self.subTest(pairs=pairs),tempfile.TemporaryDirectory() as t:
                root=Path(t);value=self.owner(root);active=None;calls=[]
                records=[dict(id=i,weight={'name':'gate'},tokens=1,ldy=128,epilogue=0) for i in range(5)]
                value.selection=[dict(data='/data',shard='scope',records=records)]
                def reload(path):
                    nonlocal active
                    active=path
                def replay(request):
                    calls.append((active,request))
                    return dict(ok=True,result=[dict(record=r['id'],mismatches=int(active=='b' and r['id']==4),
                        samples_ms=[2.0],executions_checked=request['warmup']+request['samples']) for r in request['records']])
                value.engine=SimpleNamespace(reload=Mock(side_effect=reload),call=Mock(side_effect=replay))
                with patch.object(value,'representative_scope',wraps=value.representative_scope) as representatives:
                    result=value.compare(root,dict(samples=pairs,warmup=2),dict(identity='a',path='a'),dict(identity='b',path='b'))
                    representatives.assert_called_once()
                full=[(candidate,request) for candidate,request in calls if len(request['records'])==5]
                timed=[(candidate,request) for candidate,request in calls if len(request['records'])==1]
                self.assertEqual([candidate for candidate,_ in full],['a','b'])
                self.assertTrue(all((r['warmup'],r['samples'])==(0,1) for _,r in full))
                self.assertEqual(len(timed),2*pairs)
                self.assertTrue(all((r['warmup'],r['samples'])==(2,1) for _,r in timed))
                self.assertEqual(result['order'][:6],['baseline','candidate','candidate','baseline','baseline','candidate'])
                self.assertTrue(result['checks']['baseline']['numerical_match'])
                self.assertFalse(result['checks']['candidate']['numerical_match'])
                self.assertEqual(len(result['checks']['candidate']['diagnostics']),5)
                self.assertTrue(all(leg['numerical_match'] for leg in result['legs']['candidate']))
                self.assertTrue(all(not leg.get('diagnostics') for leg in result['legs']['candidate']))
                self.assertTrue(all(leg['timings'][0]['executions_checked']==3 for leg in result['legs']['candidate']))
                self.assertNotIn('builtin',[c.args[0] for c in value.engine.reload.call_args_list])

    def test_compare_aliases_check_the_same_immutable_identity_once(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root)
            value.selection=[dict(data='/data',shard='scope',records=[dict(id=i,weight={'name':'gate'},tokens=1,ldy=128,epilogue=0) for i in range(3)])]
            calls=[]
            def replay(request):
                calls.append(request)
                return dict(ok=True,result=[dict(record=r['id'],mismatches=0,samples_ms=[1]) for r in request['records']])
            value.engine=SimpleNamespace(reload=Mock(),call=Mock(side_effect=replay))
            candidate=dict(identity='same',path='/saved')
            result=value.compare(root,dict(samples=3,warmup=1),candidate,candidate)
            self.assertEqual(sum(len(r['records'])==3 for r in calls),1)
            self.assertIs(result['checks']['baseline'],result['checks']['candidate'])
            self.assertEqual(len(calls),7)

    def test_forward_compare_reports_direct_pair_differences_and_speedups(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);value.engine=SimpleNamespace(reload=Mock())
            baseline=dict(tokens=[1],input_tokens=4,finish_reason='length',stop_token=None,logits_sha256='a',rounds=[],tokens_per_round=1,output_tokens=1)
            candidate=dict(baseline,tokens=[2],finish_reason='stop',tokens_per_round=2)
            value.forward=Mock(side_effect=[dict(cases=[dict(id='one',samples=[baseline],median_prefill_ms=4,median_decode_ms=6)]),
                dict(cases=[dict(id='one',samples=[candidate],median_prefill_ms=2,median_decode_ms=3)])]*3)
            result=value.compare(root,dict(compare_forward=True,samples=3,warmup=2),dict(path='a'),dict(path='b'))
            self.assertEqual([c.args[3] for c in value.forward.call_args_list],[2]*6)
            row=result['performance'][0]
            self.assertEqual((row['prefill_speedup'],row['decode_speedup']),(2,2))
            self.assertEqual(set(row['differences']),{'tokens','finish_reason'})
            self.assertEqual(row['baseline_tokens_per_round'],1)

    def test_capture_reuses_clean_reference_and_exposes_trajectory_changes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);value.quick_case=dict(id='one',tokens=[1]);value.cases=[value.quick_case]
            reference=dict(tokens=[1],input_tokens=1,finish_reason='length',stop_token=None,logits_sha256='a',rounds=[])
            active=None;benchmarks=0
            def reply(request):
                nonlocal active,benchmarks
                if request['op']=='capture_minimum':return dict(ok=True,result={'prefill_bytes_lower_bound':0})
                if request['op']=='capture_start':
                    active=Path(request['path']);(active/'data.bin').write_bytes(b'frozen')
                    (active/'records.jsonl').write_text(json.dumps(dict(id=1,weight={'name':'gate'},tokens=1))+'\n')
                if request['op']=='benchmark':
                    benchmarks+=1
                    return dict(ok=True,result=dict(reference,tokens=[2]) if active else reference)
                if request['op']=='capture_stop':active=None
                return dict(ok=True,result={})
            value.engine=SimpleNamespace(call=Mock(side_effect=reply))
            first=value.capture(['gate'],[1]);second=value.capture(['down'],[1])
            self.assertEqual(benchmarks,3)
            self.assertIn('tokens',first[1]['recording_differences'])
            self.assertEqual(first[1]['clean_reference'],second[1]['clean_reference'])

    def test_forward_reports_output_round_and_termination_differences(self):
        reference=dict(tokens=[1],input_tokens=4,finish_reason='length',stop_token=None,logits_sha256='a',rounds=[dict(emitted=1,extent=2)])
        candidate=dict(reference,tokens=[2],finish_reason='stop',rounds=[dict(emitted=2,extent=3)],prefill_ms=1,decode_ms=2)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);value.cases=[dict(id='case')];value.quick_case=value.cases[0]
            value.references={'case':reference}
            value.engine=SimpleNamespace(call=Mock(return_value=dict(ok=True,result=candidate)))
            result=value.forward(root,'candidate')
            self.assertTrue(result['passed']);self.assertFalse(result['numerical_match'])
            row=result['cases'][0]['samples'][0]
            self.assertEqual(row['tokens_per_round'],2)
            self.assertEqual(set(row['differences']),{'tokens','finish_reason','rounds'})

    def test_setup_loads_model_without_recording_or_benchmark(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'kernel').mkdir();artifact=root/'model';artifact.write_bytes(b'model')
            baseline=root/'baseline';baseline.mkdir()
            for name in ('kernel_lab','lab_prepare'):(baseline/name).write_bytes(name.encode())
            common.atomic(baseline/'complete.json',dict(source={},binaries={n:common.digest(baseline/n) for n in ('kernel_lab','lab_prepare')}))
            state=dict(config=dict(artifact=str(artifact),data_dir=str(root/'data')),directory=str(root),
                source={},baseline=str(baseline),session='test')
            value=host.Owner(root,state);value.publish=Mock()
            engine=SimpleNamespace(process=SimpleNamespace(pid=12),engine_pid=12,ready={},call=Mock())
            def run(argv,*args,**kwargs):
                if str(argv[0]).endswith('lab_prepare'):
                    common.atomic(Path(argv[2]),dict(cases=[dict(id='astropy__astropy-12907',tokens=[1])]))
            def output(argv,**kwargs):
                return 'NVIDIA GeForce RTX 5080, UUID, DRIVER' if '--query-gpu=name,uuid,driver_version' in argv else ''
            with patch.object(host,'run',side_effect=run),patch.object(host.subprocess,'check_output',side_effect=output),patch.object(host,'Session',return_value=engine),patch.object(host.shutil,'which',return_value=None),patch.object(host,'build_candidate') as build:
                value.setup()
            engine.call.assert_not_called();build.assert_not_called()
            self.assertEqual(list(value.corpus.glob('*/manifest.json')),[])
            self.assertEqual(value.references,{})

    def test_partial_scope_is_not_selected_after_interrupted_publish(self):
        with tempfile.TemporaryDirectory() as t:
            value=self.owner(Path(t));path,_=self.shard(value,['gate'],[1])
            path.rename(path.with_name(path.name+'.partial-interrupted'))
            value.capture=Mock(side_effect=lambda names,tokens:self.shard(value,names,tokens))
            value.select_corpus(dict(operators=['gate'],tokens_per_call=[1]))
            value.capture.assert_called_once_with(['gate'],[1])

    def test_forward_never_requires_replay_or_corpus_and_restores_once(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);value.publish=Mock()
            value.engine=SimpleNamespace(process=SimpleNamespace(pid=12),reload=Mock(return_value={}))
            value.resolve_candidate=Mock(return_value=dict(identity='saved',path='/saved'))
            value.reference_forward=Mock();value.forward=Mock(return_value=dict(passed=True,numerical_match=False))
            value.select_corpus=Mock(side_effect=AssertionError('forward must not capture'))
            value.local=Mock(side_effect=AssertionError('forward must not replay'))
            with patch.object(iteration,'source_identity',return_value={}),patch.object(host,'run'),patch.object(iteration.subprocess,'check_output',return_value='12\n'):
                result=value.evaluate(dict(source={},operation='forward',candidate='saved'),'forward')
            self.assertTrue(result['passed']);self.assertFalse(result['forward']['numerical_match'])
            self.assertEqual([c.args[0] for c in value.engine.reload.call_args_list],['/saved','builtin'])

    def test_missing_profiler_fails_before_compile_or_capture(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);value=self.owner(root);value.publish=Mock()
            value.engine=SimpleNamespace(process=SimpleNamespace(pid=12),reload=Mock(return_value={}))
            value.state['profiling_enabled']=True
            value.resolve_candidate=Mock(side_effect=AssertionError('do not compile'))
            value.select_corpus=Mock(side_effect=AssertionError('do not capture'))
            with patch.object(iteration,'source_identity',return_value={}),patch.object(host,'run'),patch.object(iteration.subprocess,'check_output',return_value='12\n'),patch.object(iteration.shutil,'which',return_value=None):
                result=value.evaluate(dict(source={},operation='profile'),'profile')
            self.assertFalse(result['passed']);self.assertIn('requires nsys',result['error'])
            self.assertEqual([c.args[0] for c in value.engine.reload.call_args_list],['builtin'])

    def test_profile_launch_keeps_actual_engine_pid_and_protocol_stream(self):
        import io
        import kernel_lab
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            process=SimpleNamespace(pid=100,stdin=io.StringIO(),stdout=io.StringIO(),poll=lambda:0)
            with patch.object(kernel_lab.subprocess,'Popen',return_value=process) as launch,patch.object(kernel_lab.Session,'read',return_value={'event':'ready','pid':200}):
                value=kernel_lab.Session(Path('/engine'),Path('/model'),Path('/settings'),root,10,profiler_session='lab-test')
                self.assertEqual(value.engine_pid,200)
                self.assertEqual(launch.call_args.args[0][:3],['nsys','launch','--session-new=lab-test'])
                self.assertEqual(value.profiler_session,'lab-test')
                value.close()

    def test_irrelevant_cli_options_are_rejected_before_any_remote_call(self):
        with patch.object(client,'probe',side_effect=AssertionError('no remote')):
            for args in (['stop','--samples','3'],['setup','--operator','gate'],['forward','--tokens-per-call','1'],['compare','--full'],['compare','--forward','--operator','gate']):
                with self.subTest(args=args),self.assertRaises(SystemExit) as error:client.main(args)
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
