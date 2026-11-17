"""Project explicit Ignis NVTX scopes onto CUDA work in Nsight SQLite exports."""
import bisect
import collections
import contextlib
import sqlite3
from pathlib import Path


PREFIX = 'ignis.profile.v1/'


def fields(label):
    if not isinstance(label, str) or not label.startswith(PREFIX):
        return {}
    result = {}
    for field in label[len(PREFIX):].split('/'):
        key, separator, value = field.partition('=')
        if separator and key in ('sample', 'phase', 'round', 'position', 'stage', 'layer', 'op') and value:
            result[key] = value
    return result


def interval_metrics(kernels):
    intervals = sorted((row['start'], row['end']) for row in kernels)
    merged = []
    for start, end in intervals:
        if end < start:
            raise ValueError('Nsight export contains a negative kernel duration')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span = max(end for _, end in intervals) - intervals[0][0]
    busy = sum(end-start for start, end in merged)
    total = sum(end-start for start, end in intervals)
    return dict(kernel_count=len(intervals), span_us=span/1000, kernel_total_us=total/1000,
                busy_us=busy/1000, gap_us=(span-busy)/1000, overlap_excess_us=(total-busy)/1000)


def process(global_id):
    # Nsight's global PID/TID stores the process in the upper bits, and a
    # 24-bit thread identifier in the lower bits of globalTid.
    return None if global_id is None else int(global_id) >> 24


class Scopes:
    def __init__(self, ranges):
        events = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in ranges:
            if row.get('end') is None or row['end'] <= row['start']:
                continue
            annotation = fields(row.get('label'))
            if annotation:
                row = dict(row, annotation=annotation)
                events[row['globalTid']][row['start']].append((True,row))
                events[row['globalTid']][row['end']].append((False,row))
        self.threads = {}
        for thread, boundaries in events.items():
            active = {}
            times = []; snapshots = []
            for timestamp, changes in sorted(boundaries.items()):
                for entering, row in sorted(changes, key=lambda value:value[0]):
                    if entering:active[row['id']] = row
                    else:active.pop(row['id'],None)
                times.append(timestamp)
                snapshots.append(tuple(sorted(active.values(), key=lambda row:(row['start'],-row['end']))))
            self.threads[thread] = (times,snapshots)

    def enclosing(self, thread, start, end):
        times, snapshots = self.threads.get(thread, ([],[]))
        index = bisect.bisect_right(times,start)-1
        choices = {}
        for scope in snapshots[index] if index >= 0 else ():
            if end <= scope['end']:
                specificity = (scope['start'],-scope['end'])
                for name, value in scope['annotation'].items():
                    previous = choices.get(name)
                    if previous is None or specificity > previous[0]:
                        choices[name] = (specificity,[(value,scope['id'])])
                    elif specificity == previous[0]:
                        previous[1].append((value,scope['id']))
        # Equally specific conflicting scopes have no defensible attribution.
        return {name:min(values,key=lambda value:value[1]) for name,(_,values) in choices.items()
                if len({value for value,_ in values}) == 1}



def project(kernels, ranges, apis, graph_nodes, request_only=False):
    """Use process-qualified API correlations and explicit graph-node ancestry."""
    scopes = Scopes(ranges)
    api_index = collections.defaultdict(list)
    for api in apis:
        api_index[(process(api['globalTid']), api['correlationId'])].append(api)
    nodes = collections.defaultdict(list)
    for node in graph_nodes:
        nodes[(process(node['globalTid']), node['graphNodeId'])].append(node)

    graph_cache = {}

    def graph_scope(pid, node_id, seen=None):
        if (pid,node_id) in graph_cache:return graph_cache[(pid,node_id)]
        seen = set() if seen is None else seen
        if node_id in seen:
            return None
        seen = seen | {node_id}
        candidates = []
        for node in nodes.get((pid, node_id), []):
            source = scopes.enclosing(node['globalTid'], node['start'], node['end'])
            if source.get('layer') or source.get('op'):
                candidates.append(source)
            original = node.get('originalGraphNodeId')
            if original and original != node_id:
                inherited = graph_scope(pid, original, seen)
                if inherited:
                    candidates.append(inherited)
        # Repeated events may describe the same node. Distinct source scopes
        # are ambiguous and must not be selected by row order.
        unique = {tuple(sorted(value.items())):value for value in candidates}
        result = next(iter(unique.values())) if len(unique) == 1 else None
        graph_cache[(pid,node_id)] = result
        return result

    timeline = []
    origin = min((row['start'] for row in kernels), default=0)
    counts = collections.Counter()
    capture_counts = collections.Counter()
    request_kernels = []
    buckets = collections.defaultdict(list)
    metadata = {}
    for kernel in kernels:
        if kernel['end'] < kernel['start']:
            raise ValueError('Nsight export contains a negative kernel duration')
        pid = process(kernel['globalPid'])
        candidates = api_index.get((pid, kernel.get('correlationId')), []) if pid is not None else []
        if 'contextId' in kernel:
            candidates = [api for api in candidates if api.get('contextId',kernel['contextId']) == kernel['contextId']]
        # Prefer an API whose CPU invocation has begun by the kernel start.
        # Do not join reused correlations merely by the nearest timestamp.
        candidates = [api for api in candidates if api['start'] <= kernel['start']]
        unique = {tuple(sorted(api.items())):api for api in candidates}
        api = next(iter(unique.values())) if len(unique) == 1 else None
        annotation = scopes.enclosing(api['globalTid'], api['start'], api['end']) if api else {}
        graph = bool(kernel.get('graphNodeId'))
        reason = None
        if graph:
            source = graph_scope(pid, kernel['graphNodeId'])
            if source:
                annotation.update({key:value for key,value in source.items() if key in ('layer','op')})
                if 'stage' not in annotation and source.get('stage',(None,None))[0] in ('decode','draft','verify','commit'):
                    annotation['stage'] = source['stage']
            else:
                reason = 'missing_or_ambiguous_graph_node_scope'
        if not api:
            reason = 'missing_or_ambiguous_api_correlation'
        if 'phase' not in annotation:
            reason = reason or 'missing_phase_scope'
        labels = {name:value[0] for name,value in annotation.items()}
        layer, separator, kind = labels.get('layer','').partition(':')
        common = dict(sample=labels.get('sample'), phase=labels.get('phase'), round=labels.get('round'), position=labels.get('position'),
                      stage=labels.get('stage'), layer=layer or None, layer_kind=kind if separator else None,
                      operator=labels.get('op'), bin='layer' if layer else 'nonlayer' if labels.get('op') else 'unclassified',
                      device=kernel['deviceId'])
        if 'phase' not in annotation or 'op' not in annotation:
            reason = reason or 'missing_operator_scope'
        role = 'setup' if common['phase'] == 'setup' or common['stage'] == 'recapture' else 'request' if common['sample'] is not None else 'unplaced'
        capture_counts[role] += 1
        if role == 'request':request_kernels.append(kernel)
        timeline.append(dict(common, capture_role=role, name=kernel.get('name'), start_us=(kernel['start']-origin)/1000,
            duration_us=(kernel['end']-kernel['start'])/1000, stream=kernel.get('streamId'),
            method='graph_node_nvtx' if graph and 'op' in annotation else 'api_nvtx' if api else None,
            unattributed_reason=reason, correlation_id=kernel.get('correlationId'), graph_node_id=kernel.get('graphNodeId')))
        if request_only and role != 'request':continue
        counts['kernels'] += 1
        for label, counter in (('phase','phase_attributed'),('layer','layer_attributed'),('op','operator_attributed')):
            counts[counter] += int(label in annotation)
        complete = 'phase' in annotation and 'op' in annotation
        counts['fully_attributed'] += int(complete)
        if not complete:
            counts['unattributed'] += 1
            reason = reason or 'missing_operator_scope'
        # Each source layer/op range belongs to one eager occurrence or one
        # graph launch. Repeated rounds never become one large GPU span.
        for level, field in (('phase','phase'),('layer','layer'),('operator','op')):
            if field not in annotation or graph and not api and level != 'phase':
                continue
            occurrence = (annotation.get('phase',(None,None))[1], annotation[field][1],
                          api['id'] if graph and level != 'phase' and api else None)
            identity = (level, kernel['deviceId'], occurrence)
            row = dict(common, level=level)
            if level == 'phase':
                row.update(layer=None, layer_kind=None, operator=None, stage=None, bin='phase')
            elif level == 'layer':
                row.update(operator=None)
            buckets[identity].append(kernel)
            metadata[identity] = row
    instances = [dict(metadata[key], instance=index, **interval_metrics(rows))
                 for index,(key,rows) in enumerate(buckets.items())]
    aggregates = collections.defaultdict(list)
    for instance in instances:
        identity = tuple((name,instance[name]) for name in ('sample','level','phase','stage','layer','layer_kind','operator','bin','device'))
        aggregates[identity].append(instance)
    summaries = []
    for identity, rows in aggregates.items():
        value = dict(identity, instances=len(rows), kernel_count=sum(row['kernel_count'] for row in rows))
        for metric in ('span_us','kernel_total_us','busy_us','gap_us','overlap_excess_us'):
            values = [row[metric] for row in rows]
            value['sum_'+metric] = sum(values)
            value['mean_'+metric] = sum(values)/len(values)
            value['max_'+metric] = max(values)
        summaries.append(value)
    reasons = collections.Counter(row['unattributed_reason'] for row in timeline
        if row['unattributed_reason'] and (not request_only or row['capture_role'] == 'request'))
    coverage = {name:counts[name] for name in ('kernels','phase_attributed','layer_attributed','operator_attributed','fully_attributed','unattributed')}
    coverage['unattributed_reasons'] = dict(reasons)
    return dict(schema_version=1, available=any(fields(row.get('label')) for row in ranges), coverage=coverage,
                capture_coverage=dict(kernels=len(kernels), **{role:capture_counts[role] for role in ('request','setup','unplaced')}),
                request_summary=request_summary(request_kernels) if request_only else None, instances=instances,
                aggregates=summaries, timeline=timeline,
                limitations=['Only explicit Ignis NVTX scopes and process-qualified CUDA API/graph-node links establish attribution.',
                    'Kernel duration sums and summed per-instance spans are not request wall time. Phase, layer and operator levels overlap and must not be added together.',
                    'Unattributed work remains visible; graph kernels require captured node provenance.'])


def request_summary(kernels):
    devices = collections.defaultdict(list)
    symbols = collections.defaultdict(list)
    for kernel in kernels:
        devices[kernel['deviceId']].append(kernel)
        symbols[kernel.get('name')].append((kernel['end']-kernel['start'])/1000)
    intervals = []
    for device,rows in sorted(devices.items()):
        metrics = interval_metrics(rows)
        intervals.append(dict(device=device, kernel_span_us=metrics['span_us'],
            kernel_busy_us=metrics['busy_us'],kernel_gap_us=metrics['gap_us'],
            overlap_excess_us=metrics['overlap_excess_us']))
    grouped = [dict(name=name,count=len(values),total_us=sum(values),mean_us=sum(values)/len(values),
                    min_us=min(values),max_us=max(values)) for name,values in symbols.items()]
    grouped.sort(key=lambda row:-row['total_us'])
    return dict(kernel_count=len(kernels),kernels=grouped,devices=intervals,
                interpretation='Only kernels explicitly linked to a captured request sample are summarized. Recapture/setup and unplaced work remain in attribution.timeline and capture_coverage. Global kernel span can include gaps between rounds/samples; use attribution instances for phase/layer spans. Summed kernel durations are not request wall time.')


def attribute_forward(database):
    """Read the observed Nsight Systems 2026.2 SQLite relational trace schema."""
    limitations = []
    with contextlib.closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(connection.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}

        def rows(table, required, optional=()):
            if table not in tables:
                limitations.append('Missing '+table+'; associated attribution remains unavailable.')
                return []
            columns = {row[1] for row in connection.execute('PRAGMA table_info('+table+')')}
            if not set(required) <= columns:
                limitations.append('Unsupported '+table+' schema; associated attribution remains unavailable.')
                return []
            selected = list(required)+[field for field in optional if field in columns and field not in required]
            return [dict(row) for row in connection.execute('SELECT rowid AS id,'+','.join(selected)+' FROM '+table)]

        kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL',('start','end','deviceId','globalPid','correlationId'),
                       ('contextId','streamId','demangledName','shortName','name','graphNodeId'))
        if not kernels:
            raise ValueError('Nsight export contains no CUDA kernel timeline for forward attribution')
        for row in kernels:
            name = next((row.get(field) for field in ('demangledName','shortName','name') if row.get(field) is not None),None)
            row['name'] = strings.get(name,str(name)) if name is not None else None
        ranges = rows('NVTX_EVENTS',('start','end','globalTid'),('text','textId','endGlobalTid'))
        ranges = [dict(row,label=row.get('text') or strings.get(row.get('textId')))
                  for row in ranges if row.get('globalTid') is not None and
                  row.get('endGlobalTid') in (None,row['globalTid'])]
        apis = rows('CUPTI_ACTIVITY_KIND_RUNTIME',('start','end','globalTid','correlationId'),('contextId',))
        apis = [row for row in apis if row['globalTid'] is not None and row['correlationId'] is not None]
        nodes = rows('CUDA_GRAPH_NODE_EVENTS',('start','end','globalTid','graphNodeId','originalGraphNodeId')) if any(row.get('graphNodeId') for row in kernels) else []
        nodes = [row for row in nodes if row['globalTid'] is not None]
    result = project(kernels,ranges,apis,nodes,request_only=True)
    result['limitations'].extend(limitations)
    return result
