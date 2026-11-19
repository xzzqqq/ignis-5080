#!/usr/bin/env python3
"""Build incremental Trellis and full GQA launchers into an immutable lab plugin."""
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
SOURCES = ['trellis_linear.cu', 'trellis_linear_decode.cu', 'trellis_linear_prefill.cu', 'trellis_linear_mlp_t8.cu', 'lab_gqa_candidate.cu', 'lab_mlp_segment.cu', 'lab_gqa_segment.cu', 'lab_gdn_segment.cu']
ADAPTER = '''#include "ignis_trellis_candidate.h"
#include "ignis_gqa_candidate.h"
#include "ignis_segment_candidate.h"
#include "ignis_mlp_segment.h"
#include "ignis_gqa_segment.h"
#include "ignis_gdn_segment.h"
extern "C" int32_t ignis_lab_gqa_prepare();
extern "C" uint64_t ignis_lab_gqa_workspace(uint32_t);
extern "C" int32_t ignis_lab_gqa_run(const ignis_gqa_candidate_call *,void *,uint64_t,void *);
extern "C" const char *ignis_lab_gqa_error();
extern "C" __attribute__((visibility("default")))
const ignis_trellis_candidate_api *ignis_trellis_candidate_v1() {
  static const ignis_trellis_candidate_api api = {
    IGNIS_TRELLIS_CANDIDATE_ABI, sizeof(ignis_trellis_candidate_api),
    sizeof(ignis_trellis_weight), sizeof(ignis_trellis_workspace), "%s",
    ignis_trellis_prepare, ignis_trellis_workspace_bytes, ignis_trellis_linear,
    ignis_trellis_linear_swiglu, ignis_trellis_last_error};
  return &api;
}
extern "C" __attribute__((visibility("default")))
const ignis_gqa_candidate_api *ignis_gqa_candidate_v1() {
  static const ignis_gqa_candidate_api api = {
    IGNIS_GQA_CANDIDATE_ABI,sizeof(ignis_gqa_candidate_api),sizeof(ignis_gqa_candidate_call),0,"%s",
    ignis_lab_gqa_prepare,ignis_lab_gqa_workspace,ignis_lab_gqa_run,ignis_lab_gqa_error};
  return &api;
}
namespace {
thread_local const char *segment_error="";
int32_t segment_prepare(){
  if(ignis_lab_mlp_segment_prepare()){segment_error=ignis_lab_mlp_segment_error();return -1;}
  if(ignis_lab_gqa_segment_prepare()){segment_error=ignis_lab_gqa_segment_error();return -1;}
  if(ignis_lab_gdn_segment_prepare()){segment_error=ignis_lab_gdn_segment_error();return -1;}
  return 0;
}
uint64_t segment_workspace(const ignis_segment_profile *p){
  if(!p)return 0;
  switch(p->kind){case IGNIS_SEGMENT_MLP:return ignis_lab_mlp_segment_workspace(p);
    case IGNIS_SEGMENT_GQA:return ignis_lab_gqa_segment_workspace(p);
    case IGNIS_SEGMENT_GDN:return ignis_lab_gdn_segment_workspace(p);default:return 0;}
}
int32_t run_mlp(const ignis_mlp_segment_call *c,void *w,uint64_t n,void *s){
  int rc=ignis_lab_mlp_segment_run(c,w,n,s);if(rc)segment_error=ignis_lab_mlp_segment_error();return rc;
}
int32_t run_gqa(const ignis_gqa_segment_call *c,void *w,uint64_t n,void *s){
  int rc=ignis_lab_gqa_segment_run(c,w,n,s);if(rc)segment_error=ignis_lab_gqa_segment_error();return rc;
}
int32_t run_gdn(const ignis_gdn_segment_call *c,void *w,uint64_t n,void *s){
  int rc=ignis_lab_gdn_segment_run(c,w,n,s);if(rc)segment_error=ignis_lab_gdn_segment_error();return rc;
}
const char *segment_last_error(){return segment_error;}
}
extern "C" __attribute__((visibility("default")))
const ignis_segment_candidate_api *ignis_segment_candidate_v1(){
  static const ignis_segment_candidate_api api{IGNIS_SEGMENT_CANDIDATE_ABI,sizeof(ignis_segment_candidate_api),
    sizeof(ignis_segment_profile),sizeof(ignis_mlp_segment_call),sizeof(ignis_gqa_segment_call),sizeof(ignis_gdn_segment_call),
    "%s",segment_prepare,segment_workspace,run_mlp,run_gqa,run_gdn,segment_last_error};return &api;
}
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies(source, name, nvcc, flags, env=None):
    raw = subprocess.check_output([nvcc, *flags, '-M', str(source/'kernel/src'/name)], text=True, env=env)
    names = shlex.split(raw.replace('\\\n', ' ').split(':', 1)[1])
    return {str(Path(n).resolve()): digest(Path(n)) for n in names}


def compile_flags():
    return ['-std=c++20', '-O3', '-arch=sm_120a', '-Xcompiler=-fPIC,-fvisibility=hidden']


def include_flags(source):
    return ['-I'+str(source/p) for p in ('kernel/include','kernel/src','kernel/vendor/include','kernel/vendor/src')]


def freeze_inputs(source, nvcc, env=None):
    """Exact local transitive compiler dependencies, discovered before freezing.

    External toolkit/system dependencies remain tracked by the object builder.
    This is not a hot-source whitelist: admission separately limits editable files.
    """
    source=source.resolve()
    files={source/'kernel/include'/name for name in
           ('ignis_trellis_candidate.h','ignis_trellis_linear.h','ignis_gqa_candidate.h','ignis_segment_candidate.h','ignis_mlp_segment.h','ignis_gqa_segment.h','ignis_gdn_segment.h')}
    for name in SOURCES:
        files.add(source/'kernel/src'/name)
        for dependency in dependencies(source,name,nvcc,compile_flags()+include_flags(source),env):
            path=Path(dependency).resolve()
            if path.is_relative_to(source):files.add(path)
    return files


def build(source, output, nvcc, env=None):
    began = time.monotonic()
    source = source.resolve()
    flags = compile_flags()
    includes = include_flags(source)
    environment = os.environ if env is None else env
    toolchain = dict(nvcc=subprocess.check_output([nvcc, '--version'], text=True, env=env),
                     host=subprocess.check_output(['g++', '--version'], text=True, env=env),
                     environment={k:environment.get(k) for k in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CPATH','CPLUS_INCLUDE_PATH')})
    output.mkdir(parents=True, exist_ok=True)
    objects = []; timings = []; identities = []
    for name in SOURCES:
        start = time.monotonic()
        deps = dependencies(source, name, nvcc, flags+includes, env)
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
                                             '-o', str(tmp/'code.o')], stdout=log, stderr=subprocess.STDOUT, env=env)
                if result.returncode: raise RuntimeError((tmp/'build.log').read_text())
                if any(digest(Path(n)) != sha for n,sha in deps.items()):
                    raise ValueError('candidate source changed during compilation')
                (tmp/'manifest.json').write_text(json.dumps(dict(inputs=inputs,sha256=digest(tmp/'code.o'))))
                tmp.rename(objdir)
        objects.append(objdir/'code.o'); identities.append(identity)
        timings.append(dict(source=name,cache_hit=cached,seconds=time.monotonic()-start))
    inputs = dict(version=2,objects=identities,adapter=ADAPTER,toolchain=toolchain,
                  abi=digest(source/'kernel/include/ignis_trellis_candidate.h'),
                  linear_abi=digest(source/'kernel/include/ignis_trellis_linear.h'),
                  gqa_abi=digest(source/'kernel/include/ignis_gqa_candidate.h'),
                  segment_abi={n:digest(source/'kernel/include'/n) for n in
                    ('ignis_segment_candidate.h','ignis_mlp_segment.h','ignis_gqa_segment.h','ignis_gdn_segment.h')})
    key = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    target = output/key
    cached = target.exists(); linked = time.monotonic()
    if cached:
        manifest=json.loads((target/'manifest.json').read_text())
        if manifest['inputs'] != inputs or manifest['library_sha256'] != digest(target/'candidate.so'):
            raise ValueError('cached candidate checksum mismatch')
    else:
        with tempfile.TemporaryDirectory(prefix='.candidate-', dir=output) as scratch:
            tmp=Path(scratch);(tmp/'adapter.cpp').write_text(ADAPTER % (key,key,key))
            with (tmp/'build.log').open('w') as log:
                result=subprocess.run([nvcc,*flags,*includes,'--shared','--cudart=shared','-Xlinker=-Bsymbolic',
                    *map(str,objects),str(tmp/'adapter.cpp'),'-o',str(tmp/'candidate.so')],stdout=log,stderr=subprocess.STDOUT,env=env)
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
