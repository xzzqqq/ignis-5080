#!/usr/bin/env python3
"""Run ordered layer/forward gates in one resident kernel_lab process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import time


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        return digest.hexdigest()


def validate_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get('cases'), list) or not plan['cases']:
        raise ValueError('plan requires nonempty cases')
    ids, previous = set(), -1
    for case in plan['cases']:
        if not isinstance(case, dict) or not isinstance(case.get('command'), dict):
            raise ValueError('case and command must be objects')
        name, tier, command = case.get('id'), case.get('tier'), case.get('command', {})
        rank = {'layer': 0, 'forward': 1}.get(tier)
        if not isinstance(name, str) or not name or name in ids or rank is None or rank < previous:
            raise ValueError('cases need unique ids and layer-before-forward ordering')
        if command.get('op') != {'layer': 'layer', 'forward': 'measure'}[tier]:
            raise ValueError('case command does not match tier')
        ids.add(name)
        previous = rank
    return plan


def agrees(reference, candidate, tier):
    # Deliberately strict initial gate. Relax only with a separately justified
    # numerical oracle; matching token prefixes alone is not a correctness test.
    keys = ('input_sha256', 'output_sha256') if tier == 'layer' else ('input_tokens', 'logits_sha256', 'tokens')
    return all(reference.get(k) is not None and reference.get(k) == candidate.get(k) for k in keys)


def latency(result, tier):
    return result['wall_ms'] if tier == 'layer' else result['decode_ms']


class Session:
    def __init__(self, engine, artifact, settings, output, timeout):
        self.timeout = timeout
        self.log = (output / 'engine.jsonl').open('w')
        self.stderr = (output / 'engine.stderr').open('w')
        lease = os.environ.get('IGNIS_GPU_LEASE_FD')
        self.process = subprocess.Popen([str(engine), str(artifact), str(settings)], text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            pass_fds=(() if lease is None else (int(lease),)))
        try:
            self.ready = self.read()
            if self.ready.get('event') != 'ready':
                raise ValueError('engine did not become ready')
        except BaseException:
            self.close()
            raise

    def read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError('engine response deadline; restart required')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('engine exited; inspect engine.stderr')
        self.log.write(line)
        self.log.flush()
        return json.loads(line)

    def call(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()
        return self.read()

    def reload(self, path):
        request = {'op': 'reload', 'path': str(path)}
        if path != 'builtin':
            request['sha256'] = sha256(path)
        reply = self.call(request)
        if not reply.get('ok'):
            raise ValueError(reply.get('error', 'reload failed'))
        return reply

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.stderr.close()
        self.log.close()


def sample(session, case, count):
    results = []
    for _ in range(count):
        reply = session.call(case['command'])
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', 'measurement failed'))
        results.append(reply['result'])
    return results


def gate(session, case, reference, warmup, samples):
    for observed in sample(session, case, warmup):
        if not agrees(reference, observed, case['tier']):
            return {'passed': False, 'reason': 'warmup numerical mismatch', 'samples': [observed]}
    results = sample(session, case, samples)
    passed = all(agrees(reference, observed, case['tier']) for observed in results)
    return {'passed': passed, 'reason': None if passed else 'numerical mismatch', 'samples': results,
            'median_ms': statistics.median(latency(r, case['tier']) for r in results)}


def run(args):
    plan = validate_plan(json.loads(args.plan.read_text()))
    if args.samples < 1 or args.warmup < 1:
        raise ValueError('samples and warmup must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    # Plans and settings are copied exactly. No mutable-file cache of verdicts.
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (args.out / 'settings.json').write_bytes(args.settings.read_bytes())
    identity_start = time.monotonic()
    report = {'version': 1, 'passed': False, 'engine_sha256': sha256(args.engine),
              'artifact': str(args.artifact), 'artifact_sha256': sha256(args.artifact),
              'source_snapshot': os.environ.get('IGNIS_EXPERIMENT_SOURCE'),
              'settings_sha256': sha256(args.settings),
              'warmup': args.warmup, 'samples': args.samples, 'candidates': []}
    report['identity_s'] = time.monotonic() - identity_start
    def save():
        (args.out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    session = Session(args.engine, args.artifact, args.out / 'settings.json', args.out, args.timeout)
    try:
        report['ready'] = session.ready
        reference = {}
        for case in plan['cases']:
            reference[case['id']] = sample(session, case, 1)[0]
        for path in args.candidate:
            record = {'path': str(path), 'sha256': sha256(path), 'passed': False, 'gates': []}
            report['candidates'].append(record)
            save()
            session.reload('builtin')
            before = {c['id']: gate(session, c, reference[c['id']], args.warmup, args.samples) for c in plan['cases']}
            if not all(g['passed'] for g in before.values()):
                raise RuntimeError('built-in reference changed before candidate')
            try:
                record['reload'] = session.reload(path)
            except ValueError as error:
                record['error'] = str(error)
                save()
                continue
            try:
                for case in plan['cases']:
                    g = gate(session, case, reference[case['id']], args.warmup, args.samples)
                    record['gates'].append(dict(id=case['id'], tier=case['tier'], **g))
                    save()
                    if not g['passed']:
                        break
                record['passed'] = len(record['gates']) == len(plan['cases']) and all(g['passed'] for g in record['gates'])
            finally:
                record['restore'] = session.reload('builtin')
            # A second control exposes thermal/clock drift and verifies rollback.
            after = {c['id']: gate(session, c, reference[c['id']], args.warmup, args.samples) for c in plan['cases']}
            record.update(before=before, after=after)
            if not all(g['passed'] for g in after.values()):
                raise RuntimeError('built-in reference changed after candidate')
            for g in record['gates']:
                if g['passed']:
                    a, b = before[g['id']]['median_ms'], after[g['id']]['median_ms']
                    g['baseline_before_after_ratio'] = a / b
                    g['speedup_vs_bracket_mean'] = (a + b) / (2 * g['median_ms'])
            save()
        report['passed'] = all(c['passed'] for c in report['candidates'])
        save()
    finally:
        session.close()
    return 0 if report['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'artifact', 'settings', 'plan', 'out'):
        p.add_argument('--' + name, type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--candidate', type=lambda s: Path(s).resolve(), action='append', required=True)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--timeout', type=float, default=1800)
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
