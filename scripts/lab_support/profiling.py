"""Bounded Nsight Systems collection and its exported CUDA kernel timeline."""
import collections
import contextlib
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from .common import atomic, key


def checked(reply):
    if not reply.get('ok'):
        raise RuntimeError(reply.get('error', 'profile replay failed'))
    return reply['result']


def command(argv, log, required=True, timeout=None):
    with Path(log).open('w') as stream:
        try:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('Nsight command timed out; see ' + str(log)) from error
    if required and result.returncode:
        raise ValueError('Nsight command failed; see ' + str(log))
    return result.returncode


def profile(owner, directory, payload, candidate):
    """Profile the already-loaded candidate without restarting the resident engine."""
    nsys = shutil.which('nsys')
    session = getattr(owner.engine, 'profiler_session', None)
    if not nsys or not session:
        raise ValueError('operator profiling requires nsys available when setup launches the engine; stop and setup after configuring Nsight Systems')
    groups = []
    for group in owner.selection:
        representatives = {}
        for record in group['records']:
            identity = key({field:record.get(field) for field in ('weight', 'up', 'tokens', 'ldy', 'epilogue')})
            representatives.setdefault(identity, record)
        if representatives:
            groups.append(dict(data=group['data'], records=list(representatives.values())))
    if not groups:
        raise ValueError('profile scope selects no recorded inputs')
    warmup = int(payload.get('warmup') if payload.get('warmup') is not None else owner.config.get('warmup', 1))
    samples = int(payload.get('samples') if payload.get('samples') is not None else owner.config.get('samples', 3))
    if warmup < 0 or samples < 1:
        raise ValueError('profile requires nonnegative warmup and positive samples')
    # Warmup never contributes launches to the exported profile.
    if warmup:
        for group in groups:
            checked(owner.engine.call(dict(op='replay_batch', **group, warmup=0, samples=warmup, exploratory=True)))
    prefix = directory / 'operator-profile'
    report = prefix.with_suffix('.nsys-rep')
    database = prefix.with_suffix('.sqlite')
    began = time.monotonic()
    rows = []
    try:
        command([nsys, 'start', '--session=' + session, '--capture-range=cudaProfilerApi',
                 '--capture-range-end=stop', '--sample=none', '--cpuctxsw=none',
                 '--output=' + str(prefix)], directory/'nsys-start.log')
        checked(owner.engine.call(dict(op='profiler_start')))
        try:
            for group in groups:
                rows.extend(checked(owner.engine.call(dict(op='replay_batch', **group, warmup=0,
                                                         samples=samples, exploratory=True))))
        finally:
            checked(owner.engine.call(dict(op='profiler_stop')))
    finally:
        # ProfilerStop normally ends collection itself. Explicit stop also
        # handles failures before the range starts and waits for report export.
        stop_status = command([nsys, 'stop', '--session=' + session], directory/'nsys-stop.log',
                              required=False, timeout=10)
    # Some versions return from stop while the auto-ended range is exporting.
    timeout = float(owner.config.get('timeout_s', 1800))
    while not report.is_file():
        if time.monotonic() - began >= timeout:
            raise ValueError('Nsight produced no report; see nsys-start.log and nsys-stop.log')
        time.sleep(0.1)
    command([nsys, 'export', '--type=sqlite', '--output=' + str(database), str(report)],
            directory/'nsys-export.log')
    summary = summarize(database)
    atomic(directory/'profile-timeline.json', summary)
    atomic(directory/'profile-replay.json', rows)
    return dict(passed=True, candidate=candidate['identity'], profiler='nsys',
                representatives=sum(len(group['records']) for group in groups),
                warmup=warmup, samples=samples, profile_wall_s=time.monotonic()-began,
                stop_status=stop_status, artifacts=dict(report=str(report), sqlite=str(database),
                    timeline=str(directory/'profile-timeline.json'), replay=str(directory/'profile-replay.json')),
                **{field:value for field,value in summary.items() if field != 'timeline'})


def summarize(database):
    """Read Nsight's SQLite export; durations are deliberately not wall time."""
    with contextlib.closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = 'CUPTI_ACTIVITY_KIND_KERNEL'
        if table not in tables:
            raise ValueError('Nsight export contains no CUDA kernel timeline')
        columns = {row[1] for row in connection.execute('PRAGMA table_info(' + table + ')')}
        name = next((field for field in ('demangledName', 'shortName', 'name') if field in columns), None)
        if name is None or not {'start', 'end', 'deviceId', 'contextId', 'streamId'} <= columns:
            raise ValueError('unsupported Nsight CUDA kernel export schema')
        strings = dict(connection.execute('SELECT id, value FROM StringIds')) if 'StringIds' in tables else {}
        rows = list(connection.execute('SELECT start, end, ' + name + ', deviceId, contextId, streamId FROM ' + table + ' ORDER BY start, end'))
    if not rows:
        raise ValueError('Nsight capture contains no CUDA kernel executions')
    grouped = collections.defaultdict(list)
    devices = collections.defaultdict(list)
    timeline = []
    origin = min(row[0] for row in rows)
    for start, end, symbol, device, context, stream in rows:
        if end < start:
            raise ValueError('Nsight export contains a negative kernel duration')
        symbol = strings.get(symbol, str(symbol))
        grouped[symbol].append(end - start)
        devices[device].append((start, end))
        timeline.append(dict(name=symbol, start_us=(start-origin)/1000, duration_us=(end-start)/1000,
                             device=device, context=context, stream=stream))
    device_spans = []
    for device, intervals in sorted(devices.items()):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        span = max(end for _, end in intervals) - min(start for start, _ in intervals)
        busy = sum(end-start for start, end in merged)
        duration = sum(end-start for start, end in intervals)
        device_spans.append(dict(device=device, kernel_span_us=span/1000, kernel_busy_us=busy/1000,
                                 kernel_gap_us=(span-busy)/1000, overlap_excess_us=(duration-busy)/1000))
    kernels = [dict(name=name, count=len(durations), total_us=sum(durations)/1000,
                    mean_us=sum(durations)/len(durations)/1000, min_us=min(durations)/1000,
                    max_us=max(durations)/1000) for name, durations in sorted(grouped.items(), key=lambda item: -sum(item[1]))]
    return dict(kernel_count=len(rows), kernels=kernels, devices=device_spans, timeline=timeline,
                interpretation='Instrumented replay kernel timeline. Span is first kernel start to last kernel end; gaps can contain memcpy and host work. Summed durations are not operator wall time; overlap excess counts concurrent kernel time with multiplicity.')
