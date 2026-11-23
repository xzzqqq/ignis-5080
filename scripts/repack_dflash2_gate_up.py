"""Losslessly tile selected 27B DFlash2 NVFP4 projections for warp loads."""
import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


def permute(data, rows, cols, inverse=False):
    # Each 16x16 tile stores four consecutive code bytes per MMA lane.
    from_idx, to_idx = (1, 0) if inverse else (0, 1)
    groups = cols // 16
    codes = rows * cols // 2
    result = bytearray(data)
    for base in range(0, rows, 16):
        for r in range(16):
            row = base + r
            for byte in range(8):
                src = row * cols // 2 + byte
                dst = base * cols // 2 + (r % 8 * 4 + byte % 4) * 4 + byte // 4 * 2 + r // 8
                spans = (slice(src, src + (groups - 1) * 8 + 1, 8), slice(dst, dst + (groups - 1) * 128 + 1, 128))
                result[spans[to_idx]] = data[spans[from_idx]]
            for q in range(4):
                src = codes + row // 128 * (cols // 64) * 512 + row % 32 * 16 + row % 128 // 32 * 4 + q
                dst = codes + base * groups + r + 16 * q
                spans = (slice(src, src + (groups // 4 - 1) * 512 + 1, 512),
                         slice(dst, dst + (groups // 4 - 1) * 64 + 1, 64))
                result[spans[to_idx]] = data[spans[from_idx]]
    return result


SHAPES = {
    "mlp/gate_up": (34816, 5120),
    "mlp/down": (5120, 17408),
    "feature_projection": (5120, 25600),
    "attention/conv_proj": (1280, 5120),
    "mlp/conv_proj": (1280, 5120),
    "attention/query_key_value": (6144, 5120),
    "attention/output": (5120, 4096),
}


def convert(source, dest, projections=("mlp/gate_up",)):
    targets = {}
    for slot in projections:
        if slot == "feature_projection":
            targets["dflash2/feature_projection"] = SHAPES[slot]
        else:
            for layer in range(5):
                targets[f"dflash2/layers/{layer}/{slot}"] = SHAPES[slot]
    source_meta = Path(str(source) + '.conversion.json')
    dest_meta = Path(str(dest) + '.conversion.json')
    metadata = json.loads(source_meta.read_text())
    if dest.exists() or dest_meta.exists():
        raise FileExistsError(dest)
    if set(metadata["artifact"]) != {"bytes"} or metadata["artifact"]["bytes"] != source.stat().st_size:
        raise ValueError("expected a size-only artifact identity matching the source file")
    metadata.update(recipe_id=dest.stem, layout_conversion={
        'source_artifact': str(source),
        'source_recipe_id': metadata['recipe_id'],
        'tensors': sorted(targets),
        'layout': 'nvfp4-mma16-v1',
        'verification': 'per-tensor inverse permutation byte comparison',
    })
    with source.open('rb') as src, tempfile.NamedTemporaryFile(dir=dest.parent) as out:
        prefix = src.read(16)
        if prefix[:8] != b'NINFER\x00\x02':
            raise ValueError('expected NINFER v2')
        size, = struct.unpack('<Q', prefix[8:])
        header = json.loads(src.read(size))
        out.write(prefix)
        out.seek(16 + size)
        shutil.copyfileobj(src, out, length=8 << 20)
        for obj in header['objects']:
            if obj['name'] not in targets:
                continue
            rows, cols = targets[obj['name']]
            if (obj['format'], obj['layout'], obj['shape'], obj['bytes']) != (
                'NVFP4', 'blockscale-k16-m128x4-v1', [rows, cols], rows * cols * 9 // 16 + 4
            ):
                raise ValueError(obj)
            offset = 16 + size + obj['offset']
            src.seek(offset)
            data = src.read(obj['bytes'])
            packed = permute(data, *obj['shape'])
            if permute(packed, *obj['shape'], inverse=True) != data:
                raise ValueError('round trip failed for ' + obj['name'])
            out.seek(offset)
            out.write(packed)
            obj['layout'] = 'nvfp4-mma16-v1'
            del targets[obj['name']]
        if targets:
            raise ValueError(f'missing layers: {targets}')
        encoded = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode()
        if len(encoded) > size:
            raise ValueError('directory exceeds reserved space')
        out.seek(16)
        out.write(encoded.ljust(size, b' '))
        out.flush()
        os.fsync(out.fileno())
        with tempfile.NamedTemporaryFile(dir=dest.parent, mode='w') as sidecar:
            json.dump(metadata, sidecar, indent=2)
            sidecar.flush()
            os.fsync(sidecar.fileno())
            os.link(sidecar.name, dest_meta)
            try:
                os.link(out.name, dest)
            except OSError:
                dest_meta.unlink()
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('dest', type=Path)
    parser.add_argument('--projection', action='append', choices=SHAPES,
                        help='Projection to convert (repeatable, default: mlp/gate_up; feature_projection is shared across layers)')
    args = parser.parse_args()
    convert(args.source, args.dest, args.projection or ('mlp/gate_up',))
