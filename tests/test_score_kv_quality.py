import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_kv_quality.py"
spec = importlib.util.spec_from_file_location("score_kv_quality", SCRIPT)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class KvQualityTests(unittest.TestCase):
    def test_identical_and_shift_invariant_rows(self):
        for shift in (0, 1000, -1000):
            row = score.row_scores([1, 2, -3], [1 + shift, 2 + shift, -3 + shift], 2)
            self.assertAlmostEqual(row["kl_ref_to_arm"], 0, places=12)
            self.assertEqual(row["top1_agreement"], 1)

    def test_known_kl_nll_and_lowest_id_tie(self):
        for backend in (None, score.np):
            with patch.object(score, "np", backend):
                row = score.row_scores([math.log(.25), math.log(.75)], [math.log(.5)] * 2, 1)
                self.assertAlmostEqual(row["kl_ref_to_arm"], .25 * math.log(.5) + .75 * math.log(1.5))
                self.assertAlmostEqual(row["nll"], math.log(2))
                self.assertEqual(row["top1_agreement"], 0)
                self.assertAlmostEqual(row["reference_pick_gap"], math.log(3))

    def test_next_token_accuracy_and_entropy_are_distinct_from_reference_agreement(self):
        for backend in (None, score.np):
            with patch.object(score, "np", backend):
                row = score.row_scores([math.log(.9), math.log(.1)], [math.log(.4), math.log(.6)], 1)
                self.assertEqual(row["top1_agreement"], 0)
                self.assertEqual(row["next_token_top1_correct"], 1)
                self.assertEqual(row["reference_next_token_top1_correct"], 0)
                self.assertAlmostEqual(row["entropy"], -.4 * math.log(.4) - .6 * math.log(.6))
                self.assertAlmostEqual(row["max_probability"], .6)
                self.assertNotIn("entropy_on_wrong_next_token", row)
                wrong = score.row_scores([0, 0], [math.log(.4), math.log(.6)], 0)
                self.assertAlmostEqual(wrong["max_probability_on_wrong_next_token"], .6)
                self.assertAlmostEqual(wrong["entropy_on_wrong_next_token"], wrong["entropy"])

    def test_extreme_logits_stable_and_invalid_rows_refused(self):
        self.assertTrue(math.isfinite(score.row_scores([10000, -10000], [-10000, 10000], 1)["nll"]))
        for bad in ([math.nan, 0], [math.inf, 0]):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                score.row_scores([0, 1], bad)
        with self.assertRaises(ValueError):
            score.row_scores([0, 1], [1])
        with self.assertRaises(ValueError):
            score.row_scores([0, 1], [1, 0], 2)

    def fixture(self, directory):
        root = Path(directory)
        # Exact BF16 rows [0,1], [2,0]. Row0 predicts tokens[2]=1;
        # row1's next token is unknown and is omitted from NLL.
        (root / "x.bin").write_bytes(struct.pack("<4H", 0, 0x3F80, 0x4000, 0))
        manifest = {"vocab": 2, "windows": [{"id": "tail", "tokens": [1, 0, 1],
                    "start": 1, "rows": 2, "logits": {arm: "x.bin" for arm in ("bf16", "hq", "nvfp4")}}]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root, manifest

    def test_dump_alignment_unknown_last_and_token_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            for external in (False, True):
                if external:
                    window = manifest["windows"][0]
                    (root / "tokens.json").write_text(json.dumps([window.pop("tokens")]))
                    window.update(tokens_file="tokens.json", window_index=0)
                    (root / "manifest.json").write_text(json.dumps(manifest))
                result = score.run(root / "manifest.json")
                self.assertEqual(result["windows"][0]["nll_rows"], 1)
                self.assertAlmostEqual(result["pooled_by_position"]["hq"]["nll"]["mean"], math.log1p(math.exp(-1)))
                self.assertIsNone(result["paired_NV4_minus_HQ"]["nll"]["window_bootstrap_95pct"])

    def test_wrong_dump_size_and_nonfinite_dump_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            (root / "x.bin").write_bytes(b"\0\0")
            with self.assertRaisesRegex(ValueError, "dump size"):
                score.run(root / "manifest.json")
            (root / "x.bin").write_bytes(struct.pack("<4H", 0x7F80, 0, 0, 0))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                score.run(root / "manifest.json")

    def test_window_bootstrap_preserves_pairing(self):
        rows = [{"arms": {"nvfp4": {"nll": {"mean": a}}, "hq": {"nll": {"mean": b}}}}
                for a, b in ((1, 2), (4, 5), (8, 9))]
        self.assertEqual(score.paired_windows(rows, "nvfp4", "hq")["nll"]["window_bootstrap_95pct"], [-1, -1])

    def test_cli_refuses_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            output = root / "result.json"
            output.write_text("keep")
            run = subprocess.run([sys.executable, str(SCRIPT), str(root / "manifest.json"), "--out", str(output)],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
