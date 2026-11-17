import contextlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lab_support.forward_timeline import fields, project, attribute_forward


PID = 11 << 24
TID = PID | 7


def scope(id, start, end, label, thread=TID):
    return dict(id=id,start=start,end=end,globalTid=thread,label='ignis.profile.v1/'+label)


def api(id, start, end, correlation, context=1, thread=TID):
    return dict(id=id,start=start,end=end,correlationId=correlation,contextId=context,globalTid=thread)


def kernel(start, end, correlation, context=1, node=0):
    return dict(start=start,end=end,correlationId=correlation,contextId=context,deviceId=0,
                streamId=1,globalPid=PID,graphNodeId=node,name='kernel')


class ForwardAttributionTests(unittest.TestCase):
    def test_eager_nested_scopes_and_overlapping_kernel_metrics(self):
        ranges=[scope(1,0,10000,'sample=0'),scope(2,100,9000,'phase=prefill'),
                scope(3,200,5000,'layer=0:gqa'),scope(4,300,4000,'op=gqa_attention')]
        result=project([kernel(1000,4000,1),kernel(2000,5000,2)],ranges,
                       [api(1,500,600,1),api(2,700,800,2)],[])
        self.assertEqual(result['coverage']['fully_attributed'],2)
        layer=next(row for row in result['instances'] if row['level']=='layer')
        self.assertEqual((layer['span_us'],layer['kernel_total_us'],layer['busy_us'],layer['overlap_excess_us']),(4,6,4,2))
        self.assertEqual(layer['sample'],'0')
        self.assertEqual(layer['layer_kind'],'gqa')

    def test_graph_source_annotations_follow_original_node_and_runtime_round(self):
        ranges=[scope(1,0,1000,'layer=2:gdn'),scope(2,10,900,'op=linear.qkv'),
                scope(3,10000,19000,'phase=decode_round/round=0/position=4'),
                scope(4,30000,39000,'phase=decode_round/round=1/position=12')]
        nodes=[dict(start=100,end=200,globalTid=TID,graphNodeId=10,originalGraphNodeId=0),
               dict(start=2000,end=2100,globalTid=TID,graphNodeId=20,originalGraphNodeId=10)]
        result=project([kernel(12000,14000,5,node=20),kernel(32000,35000,6,node=20)],ranges,
                       [api(5,11000,11500,5),api(6,31000,31500,6)],nodes)
        self.assertEqual(result['coverage']['fully_attributed'],2)
        layers=[row for row in result['instances'] if row['level']=='layer']
        self.assertEqual([row['round'] for row in layers],['0','1'])
        self.assertEqual([row['span_us'] for row in layers],[2,3])
        aggregate=next(row for row in result['aggregates'] if row['level']=='layer')
        self.assertEqual((aggregate['instances'],aggregate['sum_span_us'],aggregate['max_span_us']),(2,5,3))
        self.assertTrue(all(row['method']=='graph_node_nvtx' for row in result['timeline']))

    def test_context_and_process_qualify_reused_correlation_ids(self):
        ranges=[scope(1,0,9000,'phase=prefill'),scope(2,0,9000,'op=output_head')]
        apis=[api(1,100,200,9,context=1),api(2,300,400,9,context=2),
              api(3,500,600,9,context=1,thread=(12<<24)|7)]
        result=project([kernel(1000,2000,9,context=1)],ranges,apis,[])
        self.assertEqual(result['coverage']['fully_attributed'],1)
        self.assertEqual(result['timeline'][0]['bin'],'nonlayer')
        # Without a context-bearing API schema, conflicting correlation links
        # cannot safely establish which thread's NVTX scope launched the work.
        del apis[0]['contextId'];del apis[1]['contextId']
        result=project([kernel(1000,2000,9)],ranges,apis,[])
        self.assertEqual(result['coverage']['fully_attributed'],0)
        self.assertEqual(result['timeline'][0]['unattributed_reason'],'missing_or_ambiguous_api_correlation')

    def test_missing_graph_provenance_remains_unclassified(self):
        result=project([kernel(1000,2000,1,node=42)],
                       [scope(1,0,9000,'phase=decode_round/round=0')],[api(1,100,200,1)],[])
        self.assertEqual(result['coverage']['phase_attributed'],1)
        self.assertEqual(result['coverage']['operator_attributed'],0)
        self.assertEqual(result['timeline'][0]['bin'],'unclassified')
        self.assertEqual(result['timeline'][0]['unattributed_reason'],'missing_or_ambiguous_graph_node_scope')

    def test_samples_and_layer_occurrences_are_distinct(self):
        ranges=[scope(1,0,4000,'sample=0'),scope(2,0,4000,'phase=prefill'),
                scope(3,0,4000,'layer=0:gdn'),scope(4,0,4000,'op=rmsnorm'),
                scope(5,10000,14000,'sample=1'),scope(6,10000,14000,'phase=prefill'),
                scope(7,10000,14000,'layer=0:gdn'),scope(8,10000,14000,'op=rmsnorm')]
        result=project([kernel(1000,2000,1),kernel(11000,13000,2)],ranges,
                       [api(1,100,200,1),api(2,10100,10200,2)],[])
        layers=[row for row in result['aggregates'] if row['level']=='layer']
        self.assertEqual(len(layers),2)
        self.assertEqual({row['sample'] for row in layers},{'0','1'})

    def test_unknown_labels_do_not_invent_annotations(self):
        self.assertEqual(fields('gqa_attention_layer_0'),{})
        result=project([kernel(1,2,1)],[],[],[])
        self.assertFalse(result['available'])
        self.assertEqual(result['coverage']['unattributed'],1)

    def test_repeated_prefill_chunks_keep_separate_layer_spans(self):
        ranges=[scope(1,0,20000,'sample=0/phase=prefill'),
                scope(2,100,3000,'layer=0:gdn/op=mlp.down'),
                scope(3,10000,14000,'layer=0:gdn/op=mlp.down')]
        result=project([kernel(1000,2000,1),kernel(12000,13000,2)],ranges,
                       [api(1,200,300,1),api(2,11000,11500,2)],[],request_only=True)
        layers=[row for row in result['instances'] if row['level']=='layer']
        self.assertEqual([row['span_us'] for row in layers],[1,1])
        summary=next(row for row in result['aggregates'] if row['level']=='layer')
        self.assertEqual((summary['instances'],summary['sum_span_us']),(2,2))

    def test_graph_without_launch_link_cannot_merge_unknown_rounds(self):
        ranges=[scope(1,0,100,'layer=3:gqa/op=mlp.down')]
        nodes=[dict(start=10,end=10,globalTid=TID,graphNodeId=42,originalGraphNodeId=None)]
        result=project([kernel(1000,2000,1,node=42),kernel(10000,12000,2,node=42)],ranges,[],nodes)
        self.assertEqual(result['instances'],[])
        self.assertEqual(result['coverage']['unattributed'],2)


class SQLiteAttributionTests(unittest.TestCase):
    def write(self,path,include_nvtx=True):
        # Column identities and point graph creation/clone events follow the
        # inspected Nsight Systems 2026.2 export; values are a CPU fixture.
        with contextlib.closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
            db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'kernel'),(2,'ignis.profile.v1/sample=0')])
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,correlationId INTEGER,globalPid INTEGER,demangledName INTEGER,graphNodeId INTEGER)')
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?)',
                [(2100,2200,0,1,1,1,PID,1,20),(4000,4500,0,1,1,2,PID,1,None),
                 (6000,6100,0,1,1,3,PID,1,20),(8000,8200,0,1,1,4,PID,1,20)])
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER)')
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)',
                [(2000,2050,TID,1),(3300,3400,TID,2),(5100,5200,TID,3),(7100,7200,TID,4)])
            db.execute('CREATE TABLE CUDA_GRAPH_NODE_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER)')
            db.executemany('INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?,?,?,?,?)',
                [(300,300,TID,10,None),(1000,1000,TID,20,None),(1010,1010,TID,20,10)])
            if include_nvtx:
                db.execute('CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,endGlobalTid INTEGER,text TEXT,textId INTEGER)')
                ranges=[(0,2500,TID,None,'ignis.profile.v1/phase=setup/stage=recapture',None),
                    (100,600,TID,None,'ignis.profile.v1/layer=3:gqa',None),
                    (200,500,TID,None,'ignis.profile.v1/op=mlp.down',None),
                    (3000,9000,TID,None,None,2),
                    (3100,5000,TID,None,'ignis.profile.v1/phase=prefill',None),
                    (3200,3600,TID,None,'ignis.profile.v1/layer=0:gdn/op=linear.qkv',None),
                    (5000,6500,TID,None,'ignis.profile.v1/phase=decode_round/round=0',None),
                    (7000,8500,TID,None,'ignis.profile.v1/phase=decode_round/round=1',None)]
                db.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?)',ranges)

    def test_reader_excludes_recapture_and_projects_graph_original_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.sqlite';self.write(path)
            result=attribute_forward(path)
        self.assertEqual(result['capture_coverage'],dict(kernels=4,request=3,setup=1,unplaced=0))
        self.assertEqual(result['coverage']['fully_attributed'],3)
        self.assertEqual(result['request_summary']['kernel_count'],3)
        self.assertTrue(all(row['phase']!='setup' for row in result['instances']))
        actual=[row for row in result['timeline'] if row['capture_role']=='request']
        self.assertEqual([(row['layer'],row['operator']) for row in actual],
                         [('0','linear.qkv'),('3','mlp.down'),('3','mlp.down')])
        self.assertTrue(all(row['stage']!='recapture' for row in actual))
        self.assertEqual([row['round'] for row in actual[1:]],['0','1'])

    def test_missing_nvtx_cannot_invent_request_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.sqlite';self.write(path,include_nvtx=False)
            result=attribute_forward(path)
        self.assertFalse(result['available'])
        self.assertEqual(result['capture_coverage']['unplaced'],4)
        self.assertEqual(result['request_summary']['kernel_count'],0)
        self.assertTrue(any('Missing NVTX_EVENTS' in message for message in result['limitations']))


if __name__=='__main__':unittest.main()
