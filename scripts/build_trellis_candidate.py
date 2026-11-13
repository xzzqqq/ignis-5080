#!/usr/bin/env python3
"""Build just the three Trellis translation units into an immutable lab plugin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['trellis_linear.cu', 'trellis_linear_decode.cu', 'trellis_linear_prefill.cu']
ADAPTER = '''#include "ignis_trellis_candidate.h"
extern "C" __attribute__((visibility("default")))
const ignis_trellis_candidate_api *ignis_trellis_candidate_v1() {
  static const ignis_trellis_candidate_api api = {
    IGNIS_TRELLIS_CANDIDATE_ABI, sizeof(ignis_trellis_candidate_api),
    sizeof(ignis_trellis_weight), sizeof(ignis_trellis_workspace), "%s",
    ignis_trellis_prepare, ignis_trellis_workspace_bytes, ignis_trellis_linear,
    ignis_trellis_linear_swiglu, ignis_trellis_last_error};
  return &api;
}
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output, nvcc):
    began = time.monotonic()
    # Hash all local headers, including transitive CUDA includes. Toolchain and
    # flags are part of identity; cached files are verified before reuse.
    files = sorted(set(source.glob('kernel/src/**/*.h')) | set(source.glob('kernel/src/**/*.cuh')) |
                   set(source.glob('kernel/include/*.h')) |
                   {source / 'kernel/src' / name for name in SOURCES})
    flags = ['-std=c++20', '-O3', '-arch=sm_120a', '--shared', '--cudart=shared',
             '-Xcompiler=-fPIC,-fvisibility=hidden', '-Xlinker=-Bsymbolic']
    identity = dict(version=1, files={str(p.relative_to(source)): digest(p) for p in files},
                    nvcc=subprocess.check_output([nvcc, '--version'], text=True),
                    flags=flags, adapter=ADAPTER)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = output / key
    if target.exists():
        manifest = json.loads((target / 'manifest.json').read_text())
        if (manifest['identity'] != key or manifest['inputs'] != identity
                or manifest['library_sha256'] != digest(target / 'candidate.so')):
            raise ValueError('cached candidate checksum mismatch')
        return dict(path=str(target / 'candidate.so'), identity=key, cache_hit=True,
                    seconds=time.monotonic() - began)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.candidate-', dir=output) as scratch:
        tmp = Path(scratch)
        (tmp / 'adapter.cpp').write_text(ADAPTER % key)
        argv = [nvcc, *flags, '-I' + str(source / 'kernel/include'),
                '-I' + str(source / 'kernel/src'),
                *[str(source / 'kernel/src' / name) for name in SOURCES],
                str(tmp / 'adapter.cpp'), '-o', str(tmp / 'candidate.so')]
        with (tmp / 'build.log').open('w') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((tmp / 'build.log').read_text())
        if any(digest(p) != identity['files'][str(p.relative_to(source))] for p in files):
            raise ValueError('candidate source changed during compilation')
        manifest = dict(identity=key, inputs=identity, library_sha256=digest(tmp / 'candidate.so'),
                        build_seconds=time.monotonic() - began)
        (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Never overwrite a loaded library; each source/flags digest is a new path.
        tmp.rename(target)
    return dict(path=str(target / 'candidate.so'), identity=key, cache_hit=False,
                seconds=time.monotonic() - began)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nvcc', default=str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc'))
    args = p.parse_args()
    print(json.dumps(build(args.source.resolve(), args.out.resolve(), args.nvcc)))


if __name__ == '__main__':
    main()
