#!/usr/bin/env python3
"""CPU-only builder for selected-replay provenance/opaque-candidate GPU tests.

First build an ordinary candidate with scripts/build_trellis_candidate.py. This
reuses its cached objects; only the diagnostic fault GQA translation unit is
recompiled. No CUDA entry point or GPU workload executes during this build.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_trellis_candidate as builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_result.read_text())
    library = Path(candidate['path'])
    manifest = json.loads((library.parent / 'manifest.json').read_text())
    cache = library.parent.parent
    objects = [cache / 'objects' / identity / 'code.o' for identity in manifest['inputs']['objects']]
    nvcc = str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc')
    flags = builder.compile_flags() + builder.include_flags(ROOT)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fault = output / 'tap-fault.o'
    subprocess.run([nvcc, *flags, '-c', str(ROOT / 'tests/fixtures/segment_gqa_tap_fault.cu'), '-o', str(fault)], check=True)
    version = output / 'opaque.map'
    version.write_text('{ global: ignis_trellis_candidate_v1; ignis_gqa_candidate_v1; ignis_segment_candidate_v1; local: *; };\n')
    result = {}
    for kind in ('tap-fault', 'opaque'):
        adapter = output / (kind + '.cpp')
        adapter.write_text(builder.ADAPTER % (kind, kind, kind))
        selected = list(objects)
        if kind == 'tap-fault':
            selected[builder.SOURCES.index('lab_gqa_segment.cu')] = fault
        target = output / (kind + '.so')
        extra = ['-Xlinker=--version-script=' + str(version)] if kind == 'opaque' else []
        subprocess.run([nvcc, *flags, '--shared', '--cudart=shared', '-Xlinker=-Bsymbolic',
                        '-Xlinker=--no-undefined', *extra, *map(str, selected), str(adapter), '-o', str(target)], check=True)
        symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(target)], text=True)
        for name in ('mlp', 'gqa', 'gdn'):
            symbol = f'ignis_lab_{name}_segment_tensors_v1'
            if (symbol in symbols) != (kind == 'tap-fault'):
                raise RuntimeError('unexpected diagnostic visibility: ' + symbol)
        result[kind] = str(target)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
