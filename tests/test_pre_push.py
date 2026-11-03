"""Exercise the installed hook with real Git pushes to disposable local remotes."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pre_push.py"
FAKE_TOKEN = "ghp_" + hashlib.sha256(b"synthetic pre-push fixture; never a credential").hexdigest()[:36]


class PrePushTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "work ' tree"
        self.remote = Path(self.temp.name) / "remote.git"
        self.root.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Hook test")
        self.git("config", "user.email", "hook@example.invalid")
        (self.root / ".gitignore").write_text(".remote.env\n.local-hooks/\n")
        self.commit("base.txt", "base\n")
        self.git("init", "-q", "--bare", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", "main")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        (self.root / "scripts").mkdir()
        shutil.copyfile(SCRIPT, self.root / "scripts/pre_push.py")
        self.install()

    def git(self, *args, check=True, env=None):
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                                text=True, env=env, timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def install(self, check=True):
        result = subprocess.run([sys.executable, str(self.root / "scripts/pre_push.py"), "install"],
                                cwd=self.root, capture_output=True, text=True, timeout=10)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def commit(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git("add", "-f", name)
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD").stdout.strip()

    def remote_head(self, ref="refs/heads/main"):
        return self.git("--git-dir=" + str(self.remote), "rev-parse", "--verify", ref, check=False).stdout.strip()

    def assert_blocked(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(reason, result.stderr)
        self.assertNotIn(FAKE_TOKEN, result.stdout + result.stderr)
        self.assertEqual(self.remote_head(), self.base)

    def test_clean_push_ignores_private_worktree_files_and_install_is_idempotent(self):
        self.install()
        self.assertNotIn(str(self.root), (self.root / ".git/hooks/pre-push").read_text())
        self.assertNotIn(sys.executable, (self.root / ".git/hooks/pre-push").read_text())
        (self.root / ".remote.env").write_text("TOKEN=" + FAKE_TOKEN)
        head = self.commit(".remote.example.json", '{"host":"gpu-dev","repo":"/path/to/repo"}\n')
        result = self.git("push", "origin", "main")
        self.assertEqual(self.remote_head(), head)
        self.assertIn("passed", result.stderr)
        self.assertEqual(self.git("config", "--get", "core.hooksPath", check=False).returncode, 1)

    def test_secret_then_delete_is_blocked_despite_stale_tracking_ref_and_allow_comments(self):
        bad = self.commit("config.txt", 'api_key="' + FAKE_TOKEN + '" # gitleaks:allow\n')
        self.git("rm", "config.txt")
        self.git("commit", "-qm", "remove fixture")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        result = self.git("push", "origin", "main", check=False,
                          env=dict(os.environ, GITLEAKS_CONFIG_TOML='title="disabled"'))
        self.assert_blocked(result, "github-pat")
        self.assertIn(bad[:12], result.stderr)

    def test_private_path_is_blocked_even_when_added_then_deleted(self):
        self.commit(".remote.env", "PATH=/local/toolchain\n")
        self.git("rm", ".remote.env")
        self.git("commit", "-qm", "remove private file")
        self.assert_blocked(self.git("push", "origin", "main", check=False), "private path")

    def test_new_branch_uses_actual_remote_refs_and_first_push_scans_history(self):
        self.git("checkout", "-qb", "topic")
        self.commit("config.txt", 'api_key="' + FAKE_TOKEN + '"\n')
        self.git("update-ref", "refs/remotes/origin/topic", "HEAD")
        self.assert_blocked(self.git("push", "origin", "topic", check=False), "github-pat")
        self.assertEqual(self.remote_head("refs/heads/topic"), "")
        empty = Path(self.temp.name) / "empty.git"
        self.git("init", "-q", "--bare", str(empty))
        self.assert_blocked(self.git("push", str(empty), "topic", check=False), "github-pat")

    def test_force_push_and_merge_resolution_are_scanned(self):
        self.git("checkout", "--orphan", "replacement")
        self.git("rm", "-rf", ".")
        self.commit("config.txt", 'api_key="' + FAKE_TOKEN + '"\n')
        self.assert_blocked(self.git("push", "--force", "origin", "HEAD:main", check=False), "github-pat")
        self.git("checkout", "main")
        self.git("checkout", "-qb", "side")
        self.commit("side.txt", "side\n")
        self.git("checkout", "main")
        self.commit("main.txt", "main\n")
        self.git("merge", "--no-ff", "--no-commit", "side")
        self.commit("merge.txt", 'api_key="' + FAKE_TOKEN + '"\n')
        self.assert_blocked(self.git("push", "origin", "main", check=False), "github-pat")

    def test_commit_and_annotated_tag_messages_are_scanned(self):
        self.git("commit", "--allow-empty", "-qm", "api_key=" + FAKE_TOKEN)
        self.assert_blocked(self.git("push", "origin", "main", check=False), "github-pat")
        self.git("tag", "-a", "bad-tag", self.base, "-m", "api_key=" + FAKE_TOKEN)
        self.assert_blocked(self.git("push", "origin", "bad-tag", check=False), "github-pat")

    def test_credentials_in_a_new_archive_are_scanned(self):
        with zipfile.ZipFile(self.root / "fixture.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("config.txt", 'api_key="' + FAKE_TOKEN + '"\n')
        self.git("add", "fixture.zip")
        self.git("commit", "-qm", "archive fixture")
        self.assert_blocked(self.git("push", "origin", "main", check=False), "github-pat")

    def test_missing_or_broken_scanner_fails_closed(self):
        self.commit("safe.txt", "safe\n")
        missing = self.root / "missing-scanner"
        self.git("config", "ignis.gitleaksBinary", str(missing))
        self.assert_blocked(self.git("push", "origin", "main", check=False), "scanner")
        missing.write_text("#!/bin/sh\nexit 2\n")
        missing.chmod(0o755)
        self.assert_blocked(self.git("push", "origin", "main", check=False), "scanner")

    def test_existing_hook_is_preserved_and_internal_refs_are_blocked(self):
        custom = self.root / ".local-hooks"
        custom.mkdir()
        hook = custom / "pre-push"
        hook.write_text("#!/bin/sh\nexit 0\n")
        self.git("config", "core.hooksPath", str(custom))
        result = self.install(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing", result.stderr)
        self.assertEqual(hook.read_text(), "#!/bin/sh\nexit 0\n")
        self.git("config", "--unset", "core.hooksPath")
        self.git("update-ref", "refs/ignis-sync/test", "HEAD")
        self.assert_blocked(self.git("push", "origin", "refs/ignis-sync/test", check=False), "internal ref")

    def test_multiple_refs_are_blocked_together_and_unknown_baseline_is_refused(self):
        self.commit("safe.txt", "safe\n")
        self.git("checkout", "-qb", "bad")
        self.commit("config.txt", 'api_key="' + FAKE_TOKEN + '"\n')
        self.assert_blocked(self.git("push", "origin", "main", "bad", check=False), "github-pat")
        self.assertEqual(self.remote_head("refs/heads/bad"), "")
        entry = f"refs/heads/main {self.base} refs/heads/main {'1' * 40}\n"
        result = subprocess.run([sys.executable, str(self.root / "scripts/pre_push.py"), "check", "origin", str(self.remote)],
                                cwd=self.root, input=entry, capture_output=True, text=True)
        self.assert_blocked(result, "fetch destination history")


if __name__ == "__main__":
    unittest.main()
