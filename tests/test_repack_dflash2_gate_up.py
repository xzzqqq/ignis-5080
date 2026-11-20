import unittest

from scripts.repack_dflash2_gate_up import permute


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
