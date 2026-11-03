"""Exercise the Linux daemon hooks with a real, CPU-only HTTP child."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "mk/linux/server.py"
CHILD = '''import http.server,sys,time
started=time.monotonic()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if time.monotonic()-started >= float(sys.argv[2]) else 503)
        self.end_headers()
    def log_message(self,*args): pass
print("child stdout",flush=True)
print("child stderr",file=sys.stderr,flush=True)
http.server.HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
'''


@unittest.skipUnless(sys.platform == "linux", "daemon identity uses Linux /proc")
class LinuxServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pid = self.root / "server.pid"
        self.log = self.root / "server.log"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.port = port
        self.options = ["--pid", str(self.pid), "--log", str(self.log),
                        "--url", f"http://127.0.0.1:{port}", "--timeout", "3"]
        self.addCleanup(lambda: self.call("stop"))

    def call(self, action, command=(), extra=()):
        args = [sys.executable, str(SCRIPT), action, *self.options, *extra]
        if command:
            args += ["--", *command]
        return subprocess.run(args, capture_output=True, text=True, timeout=10)

    def start(self, delay="0"):
        return self.call("start", [sys.executable, "-c", CHILD, str(self.port), delay])

    def test_detaches_waits_for_ready_reports_status_and_stops_idempotently(self):
        started = time.monotonic()
        result = self.start("0.3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertTrue(self.pid.exists())
        self.assertIn("running", self.call("status").stdout)
        duplicate = self.start()
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("already running", duplicate.stderr)
        self.assertEqual(self.call("stop").returncode, 0)
        self.assertFalse(self.pid.exists())
        self.assertEqual(self.call("stop").returncode, 0)
        self.assertIn("not running", self.call("status").stdout)

    def test_timeout_stops_child_and_keeps_failure_logs(self):
        result = self.call("start", [sys.executable, "-c", CHILD, str(self.port), "20"],
                           ["--timeout", "0.2"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ready", result.stderr)
        self.assertFalse(self.pid.exists())
        self.assertIn("child stderr", Path(str(self.log) + ".err").read_text())

    def test_early_exit_is_reported_and_restart_archives_existing_logs(self):
        result = self.call("start", [sys.executable, "-c", "import sys; print('failure detail',file=sys.stderr); sys.exit(7)"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("7", result.stderr)
        self.assertFalse(self.pid.exists())
        self.assertEqual(self.start().returncode, 0)
        archived = list(self.root.glob("server.log.err.*"))
        self.assertEqual(len(archived), 1)
        self.assertIn("failure detail", archived[0].read_text())

    def test_stale_identity_does_not_kill_an_unrelated_process(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: (victim.terminate(), victim.wait()))
        self.pid.write_text(json.dumps({"pid": victim.pid, "start_time": "wrong", "boot_id": "wrong"}))
        result = self.call("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(victim.poll())
        self.assertFalse(self.pid.exists())

    def test_occupied_port_cannot_make_an_unready_child_look_ready(self):
        self.assertEqual(self.start().returncode, 0)
        other_pid = self.root / "other.pid"
        result = self.call("start", [sys.executable, "-c", "import time; time.sleep(30)"],
                           ["--pid", str(other_pid)])
        # Clean up the second child even when this regression test fails.
        self.call("stop", extra=["--pid", str(other_pid)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already listening", result.stderr)
        self.assertTrue(self.pid.exists())

    def test_gpu_guard_ignores_an_exited_unreaped_server_but_refuses_a_live_one(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: (child.terminate(), child.wait()))
        for name, body in [("pgrep", f"echo '{child.pid} ignis-server'"), ("nvidia-smi", "echo 0")]:
            path = self.root / name
            path.write_text("#!/bin/sh\n" + body + "\n")
            path.chmod(0o755)
        environment = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"], TMPDIR=str(self.root))
        command = ["bash", str(SCRIPT.with_name("gpu.sh")), "guard"]
        self.assertNotEqual(subprocess.run(command, env=environment, capture_output=True).returncode, 0)
        child.terminate()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.01)
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
