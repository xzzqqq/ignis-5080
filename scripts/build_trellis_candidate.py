#!/usr/bin/env python3
"""Build just the Trellis translation units into an immutable lab plugin."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['trellis_linear.cu', 'trellis_linear_decode.cu', 'trellis_linear_prefill.cu', 'trellis_linear_mlp_t8.cu']
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


def dependencies(source, name, nvcc, flags):
    raw = subprocess.check_output([nvcc, *flags, '-M', str(source/'kernel/src'/name)], text=True)
    names = shlex.split(raw.replace('\\\n', ' ').split(':', 1)[1])
    return {str(Path(n).resolve()): digest(Path(n)) for n in names}


def build(source, output, nvcc):
    began = time.monotonic()
    source = source.resolve()
    flags = ['-std=c++20', '-O3', '-arch=sm_120a', '-Xcompiler=-fPIC,-fvisibility=hidden']
    includes = ['-I'+str(source/'kernel/include'), '-I'+str(source/'kernel/src')]
    toolchain = dict(nvcc=subprocess.check_output([nvcc, '--version'], text=True),
                     host=subprocess.check_output(['g++', '--version'], text=True),
                     environment={k:os.environ.get(k) for k in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CPATH','CPLUS_INCLUDE_PATH')})
    output.mkdir(parents=True, exist_ok=True)
    objects = []; timings = []; identities = []
    for name in SOURCES:
        start = time.monotonic()
        deps = dependencies(source, name, nvcc, flags+includes)
        # Frozen source directories differ per operation; relative dependency names do not.
        normalized = {str(Path(n).relative_to(source)) if Path(n).is_relative_to(source) else n: sha
                      for n, sha in deps.items()}
        inputs = dict(version=2, source=name, dependencies=normalized, flags=flags, toolchain=toolchain)
        identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        objdir = output/'objects'/identity
        cached = objdir.exists()
        if cached:
            manifest = json.loads((objdir/'manifest.json').read_text())
            if manifest['inputs'] != inputs or manifest['sha256'] != digest(objdir/'code.o'):
                raise ValueError('object cache checksum mismatch')
        else:
            objdir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.object-', dir=objdir.parent) as scratch:
                tmp = Path(scratch)
                with (tmp/'build.log').open('w') as log:
                    result = subprocess.run([nvcc, *flags, *includes, '-c', str(source/'kernel/src'/name),
                                             '-o', str(tmp/'code.o')], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError((tmp/'build.log').read_text())
                if any(digest(Path(n)) != sha for n,sha in deps.items()):
                    raise ValueError('candidate source changed during compilation')
                (tmp/'manifest.json').write_text(json.dumps(dict(inputs=inputs,sha256=digest(tmp/'code.o'))))
                tmp.rename(objdir)
        objects.append(objdir/'code.o'); identities.append(identity)
        timings.append(dict(source=name,cache_hit=cached,seconds=time.monotonic()-start))
    inputs = dict(version=2,objects=identities,adapter=ADAPTER,toolchain=toolchain,
                  abi=digest(source/'kernel/include/ignis_trellis_candidate.h'),
                  linear_abi=digest(source/'kernel/include/ignis_trellis_linear.h'))
    key = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    target = output/key
    cached = target.exists(); linked = time.monotonic()
    if cached:
        manifest=json.loads((target/'manifest.json').read_text())
        if manifest['inputs'] != inputs or manifest['library_sha256'] != digest(target/'candidate.so'):
            raise ValueError('cached candidate checksum mismatch')
    else:
        with tempfile.TemporaryDirectory(prefix='.candidate-', dir=output) as scratch:
            tmp=Path(scratch);(tmp/'adapter.cpp').write_text(ADAPTER % key)
            with (tmp/'build.log').open('w') as log:
                result=subprocess.run([nvcc,*flags,*includes,'--shared','--cudart=shared','-Xlinker=-Bsymbolic',
                    *map(str,objects),str(tmp/'adapter.cpp'),'-o',str(tmp/'candidate.so')],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError((tmp/'build.log').read_text())
            (tmp/'manifest.json').write_text(json.dumps(dict(identity=key,inputs=inputs,
                library_sha256=digest(tmp/'candidate.so'))))
            tmp.rename(target)
    return dict(path=str(target/'candidate.so'),identity=key,cache_hit=cached,objects=timings,
                link_seconds=time.monotonic()-linked,seconds=time.monotonic()-began)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nvcc', default=str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc'))
    args = p.parse_args()
    print(json.dumps(build(args.source.resolve(), args.out.resolve(), args.nvcc)))


if __name__ == '__main__':
    main()
