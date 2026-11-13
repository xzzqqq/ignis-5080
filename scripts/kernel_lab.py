#!/usr/bin/env python3
"""Private JSONL engine client shared by the resident lab owner."""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        return digest.hexdigest()


class Session:
    def __init__(self, engine, artifact, settings, output, timeout):
        self.timeout = timeout
        self.next_id = 0
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
        for stream in (self.process.stdin,self.process.stdout):
            try:
                stream.close()
            except BrokenPipeError:
                pass
        self.stderr.close()
        self.log.close()



if __name__ == '__main__':
    raise SystemExit('Use scripts/lab.py setup, replay, forward or stop from the Laptop.')
