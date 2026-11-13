"""Exercise remote shell quoting and CLI failures without a GPU or network."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import remote


class RemoteTest(unittest.TestCase):
    def test_experiment_help_lists_actions_without_connecting(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.object(remote, "load_config") as config, self.assertRaises(SystemExit) as result:
            remote.main(["experiment", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("submit, retry, status", output.getvalue())
        config.assert_not_called()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / "gpu ' $(touch injected) repo"
        self.repo.mkdir()
        self.config = {"host": "gpu-test", "repo": str(self.repo)}
        self.path = self.root / ".remote.json"
        self.path.write_text(json.dumps(self.config))

    def test_config_accepts_example_and_rejects_bad_values_without_echoing_them(self):
        self.assertEqual(remote.load_config(self.path), self.config)
        example = remote.load_config(remote.ROOT / ".remote.example.json")
        self.assertEqual(example["host"], "gpu-dev")
        for config in [[], {**self.config, "password": "private-value"},
                       {**self.config, "host": "-oProxyCommand=private-value"},
                       {**self.config, "host": "user@private-value"},
                       {**self.config, "repo": "relative/private-value"},
                       {**self.config, "repo": "/private-value\n"}]:
            with self.subTest(config_type=type(config).__name__):
                self.path.write_text(json.dumps(config))
                with self.assertRaises(ValueError) as error:
                    remote.load_config(self.path)
                self.assertNotIn("private-value", str(error.exception))

    def test_ssh_shell_preserves_cwd_arguments_streams_and_exit_status(self):
        arguments = ["space ' ; $(touch injected)", "", "line\nbreak", "--flag", "`touch injected`"]
        script = ("import json, os, sys; "
                  "print(json.dumps([os.getcwd(), sys.argv[1:], sys.stdin.read()])); "
                  "print('diagnostic', file=sys.stderr); sys.exit(7)")
        ssh = remote.ssh_command(self.config, [sys.executable, "-c", script, *arguments])
        # Exercise the actual remote shell boundary, with only the SSH transport omitted.
        result = subprocess.run(["sh", "-c", ssh[-1]], cwd=self.root, input="input",
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 7)
        cwd, received, stdin = json.loads(result.stdout)
        self.assertEqual(Path(cwd).resolve(), self.repo.resolve())
        self.assertEqual(received, arguments)
        self.assertEqual(stdin, "input")
        self.assertEqual(result.stderr, "diagnostic\n")
        self.assertFalse((self.root / "injected").exists())
        self.assertFalse((self.repo / "injected").exists())

    def test_missing_checkout_does_not_execute_command(self):
        config = {**self.config, "repo": str(self.root / "missing")}
        ssh = remote.ssh_command(config, ["touch", "executed"])
        result = subprocess.run(["sh", "-c", ssh[-1]], cwd=self.root, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "executed").exists())

    def test_remote_private_environment_is_loaded_and_errors_stop_execution(self):
        env = self.repo / ".remote.env"
        env.write_text("export IGNIS_REMOTE_TEST='private env value'\n")
        command = remote.ssh_command(self.config, [sys.executable, "-c",
            "import os; print(os.environ['IGNIS_REMOTE_TEST'])"])
        result = subprocess.run(["sh", "-c", command[-1]], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "private env value\n")
        env.write_text("false\n")
        command = remote.ssh_command(self.config, ["touch", "executed"])
        result = subprocess.run(["sh", "-c", command[-1]], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "executed").exists())

    def test_control_can_skip_broken_private_environment(self):
        (self.repo / ".remote.env").write_text("false\n")
        command = remote.ssh_command(self.config, [sys.executable, "-c", "print('cleanup')"], False)
        result = subprocess.run(["sh", "-c", command[-1]], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "cleanup\n")

    def test_explicit_recorded_config_does_not_use_current_checkout_config(self):
        other = self.root / "recorded.json"
        other.write_text(json.dumps(self.config))
        self.path.write_text("broken")
        with patch.object(remote, "ROOT", self.root), patch.object(remote.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(remote.main(["--config", str(other), "exec", "--no-env", "--", "true"]), 0)
            run.assert_called_once_with(remote.ssh_command(self.config, ["true"], False))

    def test_cli_preserves_failures_without_retry(self):
        for code in [0, 7, 255, -15]:
            with self.subTest(code=code), patch.object(remote, "ROOT", self.root), \
                    patch.object(remote.subprocess, "run") as run:
                run.return_value.returncode = code
                self.assertEqual(remote.main(["exec", "--", "false"]), code if code >= 0 else 143)
                run.assert_called_once_with(remote.ssh_command(self.config, ["false"]))

    def test_missing_or_invalid_config_never_connects(self):
        for content in [None, "private-value: invalid json"]:
            if content is None:
                self.path.unlink()
            else:
                self.path.write_text(content)
            with patch.object(remote, "ROOT", self.root), \
                    patch.object(remote.subprocess, "run") as run, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(remote.main(["exec", "--", "true"]), 2)
                run.assert_not_called()
                self.assertNotIn("private-value", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
