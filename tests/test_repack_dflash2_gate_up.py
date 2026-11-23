import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.repack_dflash2_gate_up import SHAPES, convert, permute


class GateUpLayoutTest(unittest.TestCase):
    def test_second_tile_lane_and_divisor(self):
        rows, cols = 256, 128
        codes = rows * cols // 2
        src = bytearray(codes + rows * cols // 16 + 4)
        # Row 137, group 5, packed pair byte 6: tile 69, lane 6, byte 3.
        src[137 * 64 + 5 * 8 + 6] = 0xD2
        src[codes + (2 + 1) * 512 + 9 * 16 + 1] = 0x38
        src[-4:] = bytes.fromhex('0000803f')
        dst = permute(src, rows, cols)
        self.assertEqual(dst[69 * 128 + 6 * 4 + 3], 0xD2)
        self.assertEqual(dst[codes + 69 * 16 + 9], 0x38)
        self.assertEqual(dst[-4:], src[-4:])
        self.assertEqual(permute(dst, rows, cols, inverse=True), src)

    def test_convert_shared_and_per_layer_projections(self):
        rows, cols = 128, 64
        selected = ["dflash2/feature_projection"] + [
            f"dflash2/layers/{layer}/mlp/down" for layer in range(5)
        ]
        untouched = "dflash2/layers/0/mlp/gate_up"
        names = [untouched, *selected]
        size = rows * cols * 9 // 16 + 4
        objects = []
        payload = bytearray()
        expected = bytearray()
        for i, name in enumerate(names):
            data = bytes((j + i) % 251 for j in range(size))
            offset = len(payload)
            layout = "nvfp4-mma16-v1" if name == untouched else "blockscale-k16-m128x4-v1"
            objects.append(dict(name=name, format="NVFP4", shape=[rows, cols],
                                layout=layout, offset=offset, bytes=size))
            padding = b"\xab" * (-size % 256)
            payload.extend(data + padding)
            expected.extend((data if name == untouched else permute(data, rows, cols)) + padding)

        header = json.dumps({"objects": objects}).encode().ljust(4080, b" ")
        prefix = b"NINFER\0\2" + struct.pack("<Q", len(header))
        with tempfile.TemporaryDirectory() as directory:
            src, dst = (Path(directory) / filename for filename in ("prior.ninfer", "next.ninfer"))
            src.write_bytes(prefix + header + payload)
            sidecar = {"recipe_id": src.stem, "artifact": {"bytes": src.stat().st_size},
                       "layout_conversion": {"tensors": [untouched]}}
            Path(str(src) + ".conversion.json").write_text(json.dumps(sidecar))
            with patch.dict(SHAPES, {"mlp/down": (rows, cols), "feature_projection": (rows, cols)}):
                convert(src, dst, projections=["mlp/down", "feature_projection"])

            result = dst.read_bytes()
            end = len(prefix) + len(header)
            self.assertEqual(result[:len(prefix)], prefix)
            self.assertEqual(json.loads(result[len(prefix):end])["objects"],
                             [dict(obj, layout="nvfp4-mma16-v1") for obj in objects])
            self.assertEqual(result[end:], expected)
            record = json.loads(Path(str(dst) + ".conversion.json").read_text())["layout_conversion"]
            self.assertEqual(record["source_recipe_id"], src.stem)
            self.assertEqual(record["tensors"], sorted(selected))
