import contextlib
import importlib.util
import fcntl
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("gpu_profile", Path(__file__).parents[1] / "scripts/gpu_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_inherited_queue_lease_is_reentrant_and_keeps_host_lock_after_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "ignis-gpu-profile.lock").open("a") as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.dict(os.environ, {"IGNIS_GPU_LEASE_FD": str(lease.fileno()),
                                             "IGNIS_EXPERIMENT_GPU_DIR": str(root)}):
                    with profile.exclusive_marker(root):
                        self.assertTrue((root / "ignis-gpu-preflight.ok").exists())
                with (root / "ignis-gpu-profile.lock").open("a") as competitor:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse((root / "ignis-gpu-preflight.ok").exists())

    def test_busy_card_is_refused_even_with_little_memory_used(self):
        replies = iter(["NVIDIA GeForce RTX 5080\n", "123, another-agent\n"])
        with self.assertRaisesRegex(RuntimeError, "held"):
            profile.check_card(lambda *a, **kw: next(replies))

    def test_wrong_card_and_multiple_cards_are_refused(self):
        for cards in ["RTX 5090\n", "RTX 5080\nRTX 5080\n", ""]:
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                profile.check_card(lambda *a, **kw: cards)

    def test_failed_run_removes_only_its_own_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with profile.exclusive_marker(root) as marker:
                    self.assertTrue(marker.exists())
                    raise RuntimeError("failure")
            self.assertFalse((root / "ignis-gpu-preflight.ok").exists())
            marker.write_text("another profile")
            with self.assertRaises(FileExistsError):
                with profile.exclusive_marker(root):
                    self.fail("must refuse another profile")
            self.assertEqual(marker.read_text(), "another profile")

    def test_profile_keeps_a_symlink_artifacts_adjacent_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "source" / "model.ninfer"
            actual.parent.mkdir()
            actual.write_bytes(b"artifact")
            alias = root / "model.ninfer"
            alias.symlink_to(actual)
            sidecar = root / "model.ninfer.conversion.json"
            sidecar.write_text("{}")
            captured = []
            with patch.object(profile.subprocess, "run"), patch.object(profile, "check_card"), \
                 patch.object(profile, "exclusive_marker", return_value=contextlib.nullcontext(root / "marker")), \
                 patch.object(profile, "stage", side_effect=lambda cmd, log, env, marker: captured.append(env)), \
                 patch("sys.argv", ["gpu_profile.py", "--artifact", str(alias), "--out", str(root / "logs")]):
                profile.main()
            self.assertTrue(captured)
            for env in captured:
                supplied = Path(env["IGNIS_ARTIFACT_27B"])
                self.assertEqual(supplied.read_bytes(), b"artifact")
                self.assertEqual(Path(str(supplied) + ".conversion.json").read_text(), "{}")
