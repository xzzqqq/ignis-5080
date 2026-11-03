"""Real Git snapshots and bundles, with SSH replaced by the receiver locally."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import remote_sync


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name) / "laptop"
        self.other = Path(self.temp.name) / "gpu ' $(no)"
        self.local.mkdir()
        self.git(self.local, "init", "-q")
        self.git(self.local, "config", "user.name", "Test")
        self.git(self.local, "config", "user.email", "test@example.invalid")
        (self.local / ".gitignore").write_text(".scratch/\nmodels/\ntarget/\n.remote.env\n.remote.json\nlocal.mk\n")
        (self.local / "source.txt").write_text("original\n")
        (self.local / "remove.txt").write_text("remove later\n")
        self.git(self.local, "add", ".")
        self.git(self.local, "commit", "-qm", "base")
        self.git(self.local, "clone", "-q", str(self.local), str(self.other))
        for directory in [self.local, self.other]:
            (directory / "target").mkdir()
            (directory / "target/keep").write_text("build cache")
            (directory / "models").mkdir()
            (directory / "models/keep").write_text("model")
            (directory / ".remote.env").write_text("private")

    def git(self, root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()

    def transfer(self):
        state = remote_sync.probe(self.other)
        with remote_sync.prepare(self.local, state) as prepared:
            if prepared["bundle"] is not None:
                with Path(prepared["bundle"]).open("rb") as stream:
                    remote_sync.receive(self.other, state, prepared["ref"], prepared["snapshot"], stream)
            return prepared

    def test_commits_dirty_new_binary_delete_and_noop_preserve_private_files_and_index(self):
        (self.local / "committed.txt").write_text("new commit\n")
        self.git(self.local, "add", ".")
        self.git(self.local, "commit", "-qm", "next")
        (self.local / "source.txt").write_text("staged\n")
        self.git(self.local, "add", "source.txt")
        (self.local / "source.txt").write_text("unstaged final\n")
        (self.local / "new ' file.bin").write_bytes(bytes(range(256)))
        (self.local / "remove.txt").unlink()
        before = self.git(self.local, "status", "--porcelain")
        result = self.transfer()
        self.assertGreater(result["bytes"], 0)
        self.assertEqual((self.other / "source.txt").read_text(), "unstaged final\n")
        self.assertEqual((self.other / "new ' file.bin").read_bytes(), bytes(range(256)))
        self.assertFalse((self.other / "remove.txt").exists())
        self.assertEqual(self.git(self.other, "rev-parse", "HEAD"), self.git(self.local, "rev-parse", "HEAD"))
        self.assertEqual(self.git(self.local, "status", "--porcelain"), before)
        for name in ["target/keep", "models/keep", ".remote.env"]:
            self.assertEqual((self.other / name).read_bytes(), (self.local / name).read_bytes())
        self.assertEqual(self.transfer()["bytes"], 0)
        (self.local / "source.txt").write_text("next edit\n")
        self.transfer()
        self.assertEqual((self.other / "source.txt").read_text(), "next edit\n")

    def test_remote_edits_and_index_changes_are_refused(self):
        self.transfer()
        (self.other / "source.txt").write_text("remote work\n")
        with self.assertRaisesRegex(ValueError, "remote.*changes"):
            remote_sync.probe(self.other)
        self.git(self.other, "add", "source.txt")
        (self.other / "source.txt").write_text("original\n")
        with self.assertRaisesRegex(ValueError, "remote.*changes"):
            remote_sync.probe(self.other)

    def test_receiver_rechecks_state_before_writing(self):
        state = remote_sync.probe(self.other)
        (self.local / "source.txt").write_text("local edit\n")
        with remote_sync.prepare(self.local, state) as prepared:
            (self.other / "source.txt").write_text("remote edit during transfer\n")
            with Path(prepared["bundle"]).open("rb") as stream:
                with self.assertRaises(ValueError):
                    remote_sync.receive(self.other, state, prepared["ref"], prepared["snapshot"], stream)
        self.assertEqual((self.other / "source.txt").read_text(), "remote edit during transfer\n")

    def test_remote_untracked_files_and_ignored_collisions_are_preserved(self):
        (self.other / "untracked.txt").write_text("remote only")
        with self.assertRaisesRegex(ValueError, "untracked"):
            remote_sync.probe(self.other)
        (self.other / "untracked.txt").unlink()
        exclude = self.other / ".git/info/exclude"
        with exclude.open("a") as f:
            f.write("\nlocal-only.txt\n")
        (self.other / "local-only.txt").write_text("remote only")
        (self.local / "local-only.txt").write_text("laptop")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.transfer()
        self.assertEqual((self.other / "local-only.txt").read_text(), "remote only")

    def test_private_paths_cannot_be_force_added_to_a_snapshot(self):
        self.git(self.local, "add", "-f", ".remote.env")
        with self.assertRaisesRegex(ValueError, "private"):
            self.transfer()

    def test_rename_destination_cannot_overwrite_an_ignored_remote_file(self):
        with (self.other / ".git/info/exclude").open("a") as f:
            f.write("\nrenamed.txt\n")
        (self.other / "renamed.txt").write_text("remote only")
        (self.local / "source.txt").rename(self.local / "renamed.txt")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.transfer()
        self.assertEqual((self.other / "renamed.txt").read_text(), "remote only")

    def test_new_file_cannot_replace_an_ignored_parent_symlink(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        with (self.other / ".git/info/exclude").open("a") as f:
            f.write("\nlinked\n")
        (self.other / "linked").symlink_to(outside, target_is_directory=True)
        (self.local / "linked").mkdir()
        (self.local / "linked/new.txt").write_text("laptop")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.transfer()
        self.assertTrue((self.other / "linked").is_symlink())
        self.assertFalse((outside / "new.txt").exists())

    def test_dry_preparation_does_not_mutate_either_worktree(self):
        (self.local / "source.txt").write_text("new\n")
        state = remote_sync.probe(self.other)
        before = self.git(self.local, "status", "--porcelain")
        with remote_sync.prepare(self.local, state) as prepared:
            self.assertGreater(prepared["bytes"], 0)
        self.assertEqual((self.other / "source.txt").read_text(), "original\n")
        self.assertEqual(self.git(self.local, "status", "--porcelain"), before)
        self.assertEqual(self.git(self.local, "for-each-ref", "refs/ignis-sync/export"), "")

    def test_next_sync_does_not_resend_an_unchanged_uncommitted_binary(self):
        (self.local / "fixture.bin").write_bytes(os.urandom(1024 * 1024))
        first = self.transfer()
        (self.local / "source.txt").write_text("one small edit\n")
        second = self.transfer()
        self.assertGreater(first["bytes"], 1024 * 1024)
        self.assertLess(second["bytes"], 4096)
        self.assertEqual(self.transfer()["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
