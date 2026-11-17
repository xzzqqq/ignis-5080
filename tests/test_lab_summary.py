"""Compact CLI output keeps aggregate anomalies and links complete result files."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support import client


class SummaryTests(unittest.TestCase):
    def row(self, i):
        return dict(record=i,shard='/scope',operator='layer.0.gate',up='layer.0.up',tokens=8,
            mismatches=1,samples_ms=[1,3],diagnostics=dict(mismatches=1,total_elements=10,finite_elements=8,
                reference_nan=1,reference_inf=0,actual_nan=0,actual_inf=1,max_abs_error=i/10,
                relative_l2=i/100,relative_l2_zero_reference=False))

    def test_thousands_of_rows_and_source_hashes_do_not_reach_stdout(self):
        rows=[self.row(i) for i in range(2000)]
        report=dict(operation='one',action='replay',passed=True,source={'huge-source':'sha'},
            candidate=dict(identity='candidate',path='/candidate.so',source={'private-source':'sha'},objects=[dict(cache_hit=True)]),
            replay=dict(records=2000,performance_records=2000,numerical_match=False,
                diagnostics=rows,timings=rows,capture_trajectory_match=True))
        result=client.summary(report,'/local/full.json');encoded=json.dumps(result)
        self.assertLess(len(encoded),4000)
        self.assertNotIn('huge-source',encoded);self.assertNotIn('private-source',encoded)
        self.assertNotIn('diagnostics',result['replay']);self.assertNotIn('timings',result['replay'])
        self.assertEqual(result['local_result'],'/local/full.json')
        stats=result['replay']['numerical']
        self.assertEqual(stats['mismatches'],2000);self.assertEqual(stats['actual_inf'],2000)
        self.assertEqual(stats['max_abs_error'],199.9)
        self.assertEqual(stats['max_record_relative_l2'],19.99)
        timing=result['replay']['timing']['slowest'][0]
        self.assertEqual((timing['operator'],timing['tokens_per_call'],timing['median_ms']),('layer.0.gate',8,2))

    def test_undefined_norm_and_nonfinite_anomalies_remain_explicit(self):
        row=self.row(1);row['diagnostics'].update(relative_l2=None,relative_l2_zero_reference=True,max_abs_error=None)
        result=client.numerical_summary([row])
        self.assertIsNone(result['max_record_relative_l2'])
        self.assertEqual(result['undefined_relative_l2_records'],1)
        self.assertEqual(result['zero_reference_records'],1)
        self.assertEqual(result['reference_nan'],1)
        self.assertEqual(result['actual_inf'],1)

    def test_huge_output_differences_report_counts_and_first_change(self):
        result=client.difference_summary(dict(tokens=dict(baseline=list(range(8192)),candidate=[-1]*8192),
            rounds=dict(baseline=[[1,2]]*8192,candidate=[[2,3]]*8192),
            finish_reason=dict(baseline='length',candidate='stop'),
            logits_sha256=dict(baseline='a'*64,candidate='b'*64)))
        self.assertLess(len(json.dumps(result)),600)
        self.assertEqual(result['tokens']['first_difference'],0)
        self.assertEqual(result['rounds']['candidate_count'],8192)
        self.assertEqual(result['finish_reason']['candidate'],'stop')

    def test_compare_replay_aggregates_all_legs_without_per_record_dumps(self):
        rows=[self.row(i) for i in range(100)]
        leg=dict(numerical_match=False,diagnostics=rows,timings=rows)
        candidate=dict(identity='saved',path='/saved',source={'hidden':'sha'})
        comparison=dict(passed=True,forward=False,baseline=candidate,candidate=candidate,
            order=['baseline','candidate']*50,checks=dict(baseline=dict(candidate='saved',records=100,numerical_match=False,diagnostics=rows),
                candidate=dict(candidate='saved',records=100,numerical_match=False,diagnostics=rows)),
            legs=dict(baseline=[dict(leg,diagnostics=[])]*50,candidate=[dict(leg,diagnostics=[])]*50),
            performance=[dict(shard='/scope',record=r['record'],baseline_ms=4,candidate_ms=2,speedup=2) for r in rows]*50)
        result=client.compare_summary(comparison)
        self.assertLess(len(json.dumps(result)),4000)
        self.assertEqual(result['pairs'],50)
        self.assertEqual(result['checks']['candidate']['numerical']['mismatches'],100)
        self.assertEqual(result['legs']['candidate']['timing_numerical']['mismatches'],5000)
        self.assertNotIn('numerical',result['legs']['candidate'])
        self.assertEqual(result['performance']['median_speedup'],2)
        self.assertEqual(result['performance']['slowest'][0]['operator'],'layer.0.gate')
    def test_forward_compare_aggregates_later_anomalies_and_bounds_output(self):
        first=dict(tokens=list(range(8192)),output_tokens=8192,round_count=1,tokens_per_round=8192,
            prefill_ms=2,decode_ms=3,rounds=[dict(ms=3,emitted=8192,extent=8192)],finish_reason='length',stop_token=None,differences={})
        later=dict(first,differences=dict(tokens=dict(baseline=first['tokens'],candidate=[-1]*8192)))
        def leg(row):return dict(passed=True,numerical_match=not row['differences'],full=False,
            cases=[dict(id='one',samples=[row],median_prefill_ms=2,median_decode_ms=3)])
        candidate=dict(identity='saved',path='/saved')
        value=dict(passed=True,forward=True,baseline=candidate,candidate=candidate,
            order=['baseline','candidate']*2,legs=dict(baseline=[leg(first),leg(first)],candidate=[leg(first),leg(later)]),
            performance=[dict(case='one',baseline_prefill_ms=2,candidate_prefill_ms=2,prefill_speedup=1,
                baseline_decode_ms=3,candidate_decode_ms=3,decode_speedup=1,differences={}),
                dict(case='one',baseline_prefill_ms=2,candidate_prefill_ms=2,prefill_speedup=1,
                baseline_decode_ms=3,candidate_decode_ms=3,decode_speedup=1,differences=later['differences'])])
        result=client.compare_summary(value)
        self.assertLess(len(json.dumps(result)),3000)
        self.assertFalse(result['legs']['candidate']['numerical_match'])
        self.assertEqual(result['legs']['candidate']['cases'][0]['differing_samples'],1)
        self.assertEqual(result['performance'][0]['differing_pairs'],1)



if __name__=='__main__':unittest.main()
