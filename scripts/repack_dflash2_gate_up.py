"""Losslessly tile the five 27B DFlash2 NVFP4 gate/up tensors for warp loads."""
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


def convert(source, dest):
    source_meta = Path(str(source) + '.conversion.json')
    dest_meta = Path(str(dest) + '.conversion.json')
    metadata = json.loads(source_meta.read_text())
    if dest.exists() or dest_meta.exists():
        raise FileExistsError(dest)
    metadata.update(recipe_id=dest.stem, layout_conversion={
        'source_artifact': str(source),
        'source_recipe_id': metadata['recipe_id'],
        'tensors': [f'dflash2/layers/{i}/mlp/gate_up' for i in range(5)],
        'layout': 'nvfp4-mma16-v1',
        'verification': 'per-tensor inverse permutation byte comparison',
    })
    with source.open('rb') as src, tempfile.NamedTemporaryFile(dir=dest.parent) as out:
        prefix = src.read(16)
        if prefix[:8] != b'NINFER\x00\x02':
            raise ValueError('expected NINFER v2')
        size, = struct.unpack('<Q', prefix[8:])
        header = json.loads(src.read(size))
        targets = {f'dflash2/layers/{i}/mlp/gate_up' for i in range(5)}
        out.write(prefix)
        out.seek(16 + size)
        shutil.copyfileobj(src, out, length=8 << 20)
        for obj in header['objects']:
            if obj['name'] not in targets:
                continue
            if (obj['format'], obj['layout'], obj['shape'], obj['bytes']) != (
                'NVFP4', 'blockscale-k16-m128x4-v1', [34816, 5120], 100270084
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
            targets.remove(obj['name'])
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
    args = parser.parse_args()
    convert(args.source, args.dest)
