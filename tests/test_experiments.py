"""CPU contract/transaction/transport/recovery tests; no SSH, systemd or GPU."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiments import client, host, runner, schema, source, store, systemd, worker
import remote


def request(job_id="candidate"):
    return dict(version=1, id=job_id, task="attention", question="Does the candidate work?",
                timeout_s=30, steps=[dict(name="measure", argv=["true"], artifacts=[])])


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo ' space"
        self.root.mkdir()
        source.git(self.root, "init", "-q")
        source.git(self.root, "config", "user.email", "test@localhost")
        source.git(self.root, "config", "user.name", "Test")
        (self.root / ".gitignore").write_text(".scratch/\ntarget/\n.remote.env\n.remote.json\n.remote.cargo.toml\n")
        (self.root / "candidate.txt").write_text("base")
        source.git(self.root, "add", ".")
        source.git(self.root, "commit", "-qm", "fixture")
        self.head = source.git(self.root, "rev-parse", "HEAD")
        self.store = store.Store(self.root)

    def payload(self, job_id="candidate"):
        return dict(request=request(job_id), source=dict(head=self.head, snapshot=self.head,
                    tree=source.git(self.root, "rev-parse", "HEAD^{tree}"), dirty=False, ref="unused"),
                    runtime_sha256="test")

    def running(self, payload=None):
        payload = payload or self.payload()
        job_id = payload["request"]["id"]
        directory = self.store.directory(job_id)
        (directory / "config").mkdir(parents=True)
        self.store.submit(payload)
        return self.store.claim(job_id, "test.service")

    def run_candidate(self, payload):
        record = self.running(payload)
        lease = os.open(self.root / ".scratch/lease", os.O_CREAT | os.O_RDWR, 0o600)
        with patch.object(runner, "receive_lease", return_value=lease):
            code = runner.run(self.root, record["id"])
        return code, json.loads((self.store.directory(record["id"]) / "done.json").read_text())

    def test_snapshot_preserves_index_and_import_never_changes_remote_checkout(self):
        other = Path(self.temporary.name) / "remote"
        subprocess.run(["git", "clone", "-q", str(self.root), str(other)], check=True)
        (self.root / "candidate.txt").write_text("staged")
        source.git(self.root, "add", "candidate.txt")
        (self.root / "candidate.txt").write_text("actual working bytes")
        (self.root / "new.txt").write_text("new candidate")
        (self.root / ".remote.env").write_text("private")
        before = source.git(self.root, "status", "--porcelain")
        with source.freeze(self.root, self.head) as (frozen, bundle):
            source.receive(other, frozen, bundle)
            self.assertTrue(frozen["dirty"])
            self.assertEqual(source.git(other, "show", frozen["snapshot"] + ":candidate.txt"), "actual working bytes")
            self.assertEqual(source.git(other, "rev-parse", "HEAD"), self.head)
            self.assertEqual((other / "candidate.txt").read_text(), "base")
            self.assertNotIn(".remote.env", source.git(other, "ls-tree", "-r", frozen["snapshot"]))
        self.assertEqual(source.git(self.root, "status", "--porcelain"), before)
        self.assertEqual(source.git(self.root, "show", ":candidate.txt"), "staged")

    def test_older_explicit_commit_needs_no_bundle_when_remote_has_descendant(self):
        (self.root / "new.txt").write_text("new")
        source.git(self.root, "add", ".")
        source.git(self.root, "commit", "-qm", "next")
        base = source.git(self.root, "rev-parse", "HEAD")
        with source.freeze(self.root, base, self.head) as (frozen, bundle):
            self.assertIsNone(bundle)
            source.receive(self.root, frozen, bundle)

    def test_worktree_uses_main_private_config_without_changing_source_root(self):
        (self.root / ".remote.json").write_text('{"host":"gpu","repo":"/remote"}')
        worktree = Path(self.temporary.name) / "candidate"
        source.git(self.root, "worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertEqual(remote.config_path(worktree), self.root.resolve() / ".remote.json")
        with source.freeze(worktree, self.head) as (frozen, _):
            self.assertEqual(frozen["head"], self.head)

    def test_concurrent_fifo_and_idempotent_lost_receipt(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(lambda i: store.Store(self.root).submit(self.payload(f"candidate-{i}")), range(20)))
        self.assertEqual(len(records), 20)
        jobs = self.store.jobs()
        self.assertEqual([job["sequence"] for job in jobs], list(range(1, 21)))
        first = self.store.submit(jobs[0]["payload"])
        self.assertEqual(first["id"], jobs[0]["id"])
        self.assertEqual(len(self.store.jobs()), 20)
        changed = json.loads(json.dumps(jobs[0]["payload"]))
        changed["request"]["config"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "different"):
            self.store.submit(changed)

    def test_queued_cancel_is_terminal_and_running_cancel_wins_finish_race(self):
        self.store.submit(self.payload())
        cancelled = self.store.cancel("candidate")
        self.assertEqual(cancelled["result"]["state"], "cancelled")
        with self.assertRaises(ValueError):
            self.store.claim("candidate", "no.service")
        self.running(self.payload("running"))
        self.store.cancel("running")
        self.store.finish("running", "succeeded")
        self.assertEqual(self.store.get("running")["state"], "cancelled")

    def test_corrupt_database_is_not_reset(self):
        (self.store.path / "queue.sqlite3").write_bytes(b"corrupt")
        with self.assertRaises(Exception):
            store.Store(self.root)
        self.assertEqual((self.store.path / "queue.sqlite3").read_bytes(), b"corrupt")

    def test_runner_builds_in_candidate_and_binds_artifact_to_source(self):
        payload = self.payload()
        payload["request"]["steps"] = [dict(name="check", argv=[sys.executable, "-c",
            "import os,pathlib; assert pathlib.Path('candidate.txt').read_text() == 'base'; "
            "assert os.environ['CARGO_TARGET_DIR'] == str(pathlib.Path.cwd()/'target'); "
            "pathlib.Path(os.environ['IGNIS_EXPERIMENT_OUTPUT'],'result.txt').write_text(os.environ['IGNIS_EXPERIMENT_SOURCE'])"],
            artifacts=["result.txt"])]
        code, done = self.run_candidate(payload)
        self.assertEqual(code, 0, done)
        self.assertEqual(done["state"], "succeeded")
        artifact = done["execution"]["steps"][0]["artifacts"][0]
        self.assertEqual(artifact["sha256"], hashlib.sha256(self.head.encode()).hexdigest())
        self.assertEqual((self.root / "candidate.txt").read_text(), "base")

    def test_missing_artifact_and_nonzero_exit_short_circuit(self):
        for name, argv, artifacts in [("missing", [sys.executable, "-c", "pass"], ["missing.json"]),
                                      ("nonzero", [sys.executable, "-c", "raise SystemExit(9)"], [])]:
            payload = self.payload(name)
            payload["request"]["steps"] = [dict(name="first", argv=argv, artifacts=artifacts),
                                             dict(name="must-not-run", argv=["false"])]
            code, done = self.run_candidate(payload)
            self.assertEqual(code, 1)
            self.assertEqual(len(done["execution"]["steps"]), 1)

    def test_source_mutation_and_wrong_input_hash_fail(self):
        payload = self.payload("mutate")
        payload["request"]["steps"] = [dict(name="edit", argv=[sys.executable, "-c",
            "from pathlib import Path; Path('candidate.txt').write_text('changed')"])]
        _, done = self.run_candidate(payload)
        self.assertIn("modified tracked", done["error"])
        payload = self.payload("input")
        payload["request"]["inputs"] = [dict(path=str(self.root / "candidate.txt"), sha256="0" * 64)]
        _, done = self.run_candidate(payload)
        self.assertIn("input hash mismatch", done["error"])
        self.assertFalse(done["execution"]["steps"])

    def test_output_directory_symlink_cannot_escape_artifact_or_fetch_boundary(self):
        directory = self.store.directory("candidate")
        directory.mkdir(parents=True)
        (directory / "output").symlink_to(self.root, target_is_directory=True)
        self.store.submit(self.payload())
        self.store.cancel("candidate")
        with self.assertRaisesRegex(ValueError, "escaping"):
            runner.artifact(directory / "output", "candidate.txt")
        with self.assertRaisesRegex(ValueError, "symlinked"):
            host.export(self.store, "candidate")

    def test_worker_version_mismatch_rejects_admission_before_source_import(self):
        runtime = self.root / ".scratch/tool.pyz"
        runtime.write_bytes(b"new runtime")
        payload = self.payload()
        payload["runtime_sha256"] = source.sha256(runtime)
        health = dict(service=dict(ActiveState="active"), worker=dict(runtime_sha256="old runtime",
                      at=time.time(), stopping=False))
        with patch.object(host, "status", return_value=health), patch.object(host, "receive") as receive:
            with self.assertRaisesRegex(ValueError, "matching worker"):
                host.submit(self.store, payload, None, runtime)
            receive.assert_not_called()
        self.assertFalse(self.store.jobs())

    def test_recovery_attaches_live_unit_and_never_replays_missing_unit(self):
        record = self.running()
        with patch.object(systemd, "inspect", return_value=dict(ActiveState="active", SubState="running")), \
                patch.object(systemd, "stop") as stop:
            self.assertFalse(worker.reconcile(self.store, record, 256))
            stop.assert_not_called()
        with patch.object(systemd, "inspect", return_value=dict(LoadState="not-found")), \
                patch.object(systemd, "stop"), patch.object(worker, "gpu_status", return_value={}), \
                patch.object(worker, "GPU_LOCK", self.root / ".scratch/lease"):
            self.assertTrue(worker.reconcile(self.store, record, 256))
        self.assertEqual(self.store.get("candidate")["state"], "interrupted")

    def test_timeout_diagnosis_survives_cleanup_retry_and_worker_restart(self):
        record = self.running()
        unit = dict(ActiveState="failed", Result="timeout")
        with patch.object(systemd, "inspect", return_value=unit), patch.object(systemd, "stop"), \
                patch.object(worker, "gpu_status", side_effect=ValueError("busy")), \
                patch.object(worker, "GPU_LOCK", self.root / ".scratch/lease"):
            with self.assertRaisesRegex(ValueError, "busy"):
                worker.reconcile(self.store, record, 256)
        self.assertEqual(self.store.get("candidate")["state"], "running")
        with patch.object(systemd, "inspect", return_value=dict(LoadState="not-found")), \
                patch.object(systemd, "stop"), patch.object(worker, "gpu_status", return_value={}), \
                patch.object(worker, "GPU_LOCK", self.root / ".scratch/lease"):
            self.assertTrue(worker.reconcile(self.store, self.store.get("candidate"), 256))
        self.assertEqual(self.store.get("candidate")["state"], "timed_out")

    def test_recovery_publishes_completed_result_only_after_cleanup(self):
        payload = self.payload()
        _, done = self.run_candidate(payload)
        with patch.object(systemd, "inspect", return_value=dict(ActiveState="active", SubState="exited", Result="success")), \
                patch.object(systemd, "stop", side_effect=ValueError("cgroup still populated")), \
                patch.object(worker, "gpu_status") as gpu:
            with self.assertRaises(ValueError):
                worker.reconcile(self.store, self.store.get("candidate"), 256)
            gpu.assert_not_called()
        self.assertEqual(self.store.get("candidate")["state"], "running")
        with patch.object(systemd, "inspect", return_value=dict(ActiveState="inactive")), \
                patch.object(systemd, "stop"), patch.object(worker, "gpu_status", return_value={}), \
                patch.object(worker, "GPU_LOCK", self.root / ".scratch/lease"):
            worker.reconcile(self.store, self.store.get("candidate"), 256)
        self.assertEqual(self.store.get("candidate")["state"], "succeeded")

    def test_submit_captures_config_once_and_retries_after_worker_stops(self):
        runtime = self.root / ".scratch/tool.pyz"
        runtime.write_bytes(b"tool")
        payload = self.payload()
        payload["runtime_sha256"] = source.sha256(runtime)
        (self.root / ".remote.env").write_text("export FLAG=old\n")
        health = dict(service=dict(ActiveState="active"), worker=dict(runtime_sha256=source.sha256(runtime),
                      at=time.time(), stopping=False))
        with patch.object(host, "status", return_value=health):
            first = host.submit(self.store, payload, None, runtime)
        (self.root / ".remote.env").write_text("export FLAG=new\n")
        with patch.object(host, "status", side_effect=AssertionError("must not need worker")):
            second = host.submit(self.store, payload, None, runtime)
        self.assertEqual(first, second)
        self.assertIn("old", (self.store.directory("candidate") / "config/.remote.env").read_text())

    def test_lost_ssh_reply_retry_uses_saved_source_after_more_local_edits(self):
        path = self.root / ".scratch/request.json"
        store.atomic(path, request())
        seen = []
        def transport(config, ssh_command, runtime_hash, archive, arguments, output=None):
            if arguments == ["probe"]:
                output.write(schema.canonical(dict(head=self.head)))
                return subprocess.CompletedProcess([], 0)
            with tarfile.open(archive) as tar:
                seen.append(json.load(tar.extractfile("payload.json")))
            output.write(schema.canonical(dict(id="candidate")))
            return subprocess.CompletedProcess([], 255 if len(seen) == 1 else 0)
        with patch.object(client, "call", side_effect=transport), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(client.main(self.root, {}, None, ["submit", str(path)]), 255)
            (self.root / "candidate.txt").write_text("a later candidate")
            self.assertEqual(client.main(self.root, {}, None, ["retry", "candidate"]), 0)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(source.git(self.root, "show", seen[1]["source"]["snapshot"] + ":candidate.txt"), "base")

    @unittest.skipUnless(sys.platform == "linux", "lease transport uses Linux abstract Unix sockets")
    def test_transferred_lease_survives_sender_closing_its_descriptor(self):
        lock = self.root / ".scratch/lease"
        with systemd.listener(self.root, "transfer") as channel, ThreadPoolExecutor(max_workers=1) as pool:
            with lock.open("a") as sender:
                fcntl.flock(sender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receiving = pool.submit(systemd.receive_lease, self.root, "transfer")
                systemd.send_lease(channel, sender.fileno())
                receiver = receiving.result()
            try:
                with lock.open("a") as competitor:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(receiver)
        with lock.open("a") as competitor:
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


class ContractTests(unittest.TestCase):
    def test_schema_rejects_typos_nonfinite_timeout_and_isolation_overrides(self):
        for changes in [dict(version=2), dict(id="../bad"), dict(timeout_s=float("nan")),
                        dict(timeout_s=True), dict(extra=True), dict(env={"CARGO_TARGET_DIR": "/shared"}),
                        dict(steps=[dict(name="check", argv=["true"], artifacts=["../old.json"])])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                schema.validate(dict(request(), **changes))
        self.assertEqual(schema.validate(request()), request())

    def test_package_is_deterministic_and_bootstrap_preserves_payload_arguments(self):
        self.assertEqual(client.package(), client.package())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = client.package()
            archive = root / "upload.tar.gz"
            client.upload(archive, runtime, {})
            # A non-git root must fail cleanly, not execute shell-looking path text.
            with archive.open("rb") as stream:
                result = subprocess.run([sys.executable, "-c", client.BOOTSTRAP,
                    str(root), hashlib.sha256(runtime).hexdigest(), "probe"], stdin=stream, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertTrue(list((root / ".scratch/experiment-tools").glob("*.pyz")))

    def test_fetch_rejects_paths_links_and_changed_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, name in enumerate(["../escape", "/escape", "valid"]):
                archive = root / f"{i}.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    entry = tarfile.TarInfo(name)
                    entry.type = tarfile.SYMTYPE if name == "valid" else tarfile.REGTYPE
                    tar.addfile(entry, io.BytesIO(b""))
                with self.assertRaises(ValueError):
                    client.unpack(archive, root / f"out{i}")
            archive = root / "bad-hash.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in [("manifest.json", schema.canonical({"result.txt": dict(bytes=1, sha256="0" * 64)})),
                                   ("result.txt", b"x")]:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    tar.addfile(entry, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                client.unpack(archive, root / "out")

    def test_gpu_guard_rejects_external_compute_even_with_low_memory(self):
        with patch.object(worker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "status", "")), \
                patch.object(worker.subprocess, "check_output", side_effect=["NVIDIA RTX 5080, 0, uuid, driver", "123, external, 4"]):
            with self.assertRaisesRegex(ValueError, "GPU busy"):
                worker.gpu_status(Path("."), 256)


if __name__ == "__main__":
    unittest.main()
