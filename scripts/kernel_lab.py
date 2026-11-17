#!/usr/bin/env python3
"""Private JSONL engine client shared by the resident lab owner."""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        return digest.hexdigest()


class Session:
    def __init__(self, engine, artifact, settings, output, timeout, profiler_session=None):
        self.timeout = timeout
        self.profiler_session = profiler_session
        self.next_id = 0
        self.log = (output / 'engine.jsonl').open('w')
        self.profiler_log = (output / 'engine.profiler.log').open('w')
        self.stdout_buffer = bytearray()
        self.stderr = (output / 'engine.stderr').open('w')
        lease = os.environ.get('IGNIS_GPU_LEASE_FD')
        self.transport_path = None
        self.output_stream = None
        argv = [str(engine), str(artifact), str(settings)]
        if profiler_session:
            self.transport_path = output / 'engine.stdout.fifo'
            os.mkfifo(self.transport_path, 0o600)
            descriptor = os.open(self.transport_path, os.O_RDWR | os.O_NONBLOCK)
            self.output_stream = os.fdopen(descriptor, 'rb', buffering=0)
            wrapper = 'exec "$1" "$2" "$3" > "$4"'
            argv = ['nsys', 'launch', '--session-new='+profiler_session,
                    '--trace=cuda,nvtx', '--cuda-graph-trace=node',
                    '/bin/bash', '--noprofile', '--norc', '-c', wrapper,
                    'ignis-lab-engine', *argv, str(self.transport_path)]
        environment = dict(os.environ, IGNIS_LAB_PROFILE='1' if profiler_session else '0')
        self.process = subprocess.Popen(argv, text=True, stdin=subprocess.PIPE,
            stdout=self.profiler_log if profiler_session else subprocess.PIPE, stderr=self.stderr, env=environment,
            pass_fds=(() if lease is None else (int(lease),)))
        if self.output_stream is None:
            self.output_stream = self.process.stdout
        try:
            self.ready = self.read()
            if self.ready.get('event') != 'ready':
                raise ValueError('engine did not become ready')
            self.engine_pid = self.ready.get('pid',self.process.pid)
            if profiler_session and (type(self.ready.get('pid')) is not int or self.engine_pid<1):
                raise ValueError('profiled engine ready response lacks its process ID')
        except BaseException:
            self.close()
            raise

    def read(self):
        response_deadline = time.monotonic() + self.timeout
        while True:
            remaining = response_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('engine response deadline; restart required')
            if b'\n' not in self.stdout_buffer:
                if not select.select([self.output_stream], [], [], min(remaining, 0.1))[0]:
                    if self.process.poll() is not None:
                        raise RuntimeError('engine exited; inspect engine.stderr and engine.profiler.log')
                    continue
                block = os.read(self.output_stream.fileno(), 65536)
                if not block:
                    raise RuntimeError('engine exited; inspect engine.stderr')
                self.stdout_buffer.extend(block)
                continue
            line, _, rest = self.stdout_buffer.partition(b'\n')
            self.stdout_buffer = bytearray(rest)
            self.log.write(line.decode('utf-8', errors='backslashreplace') + '\n')
            self.log.flush()
            try:
                return json.loads(line)
            except ValueError as e:
                raise RuntimeError('invalid engine JSON; session must restart') from e

    def call(self, value):
        self.next_id += 1
        self.process.stdin.write(json.dumps(dict(value,id=self.next_id)) + '\n')
        self.process.stdin.flush()
        reply = self.read()
        if reply.get('id') != self.next_id:
            raise RuntimeError('engine reply id mismatch; session must restart')
        return reply

    def reload(self, path, profile_recapture=False):
        request = {'op': 'reload', 'path': str(path)}
        if profile_recapture:
            request['profile_recapture'] = True
        if path != 'builtin':
            request['sha256'] = sha256(path)
        reply = self.call(request)
        if not reply.get('ok'):
            raise ValueError(reply.get('error', 'reload failed'))
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin,self.output_stream):
            try:
                stream.close()
            except BrokenPipeError:
                pass
        self.stderr.close()
        self.log.close()
        self.profiler_log.close()
        if self.transport_path is not None:
            self.transport_path.unlink(missing_ok=True)



if __name__ == '__main__':
    raise SystemExit('Use scripts/lab.py setup, replay, forward or stop from the Laptop.')
