#!/usr/bin/env python3
"""Explicit remote GPU acceptance test; creates only its own worktrees/jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.source import freeze, git
from experiments.store import atomic

ROOT = Path(__file__).resolve().parents[1]


def cli(root, *arguments, codes=(0,)):
    process = subprocess.run([sys.executable, str(root / "scripts/remote.py"), *arguments],
                             text=True, capture_output=True, timeout=90)
    if process.returncode not in codes:
        raise RuntimeError(f"{arguments}: exit {process.returncode}\n{process.stderr}\n{process.stdout}")
    return json.loads(process.stdout) if process.stdout.strip() else None


def wait_for(root, job_id, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = cli(root, "experiment", "result", job_id, codes=(0, 1, 3))
        if predicate(record):
            return record
        if record["state"] in {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}:
            raise RuntimeError(f"{job_id} ended before the expected observation: {record.get('error')}")
        time.sleep(.5)
    raise RuntimeError("timed out observing " + job_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="accept-" + time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    installed = cli(ROOT, "experiment", "install")
    initial = cli(ROOT, "experiment", "status")
    if any(job["state"] not in {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}
           for job in initial["jobs"]):
        raise RuntimeError("acceptance requires an idle queue; existing jobs were left untouched")
    subprocess.run([sys.executable, str(ROOT / "scripts/remote.py"), "exec", "--", "make", "gpu-status"], check=True)
    worktrees = []
    with freeze(ROOT, git(ROOT, "rev-parse", "HEAD")) as (frozen, _):
        for name, value in [("a", 11), ("b", 22)]:
            path = args.out / ("worktree-" + name)
            git(ROOT, "worktree", "add", "--quiet", "--detach", str(path), frozen["snapshot"])
            fixture = path / "tests/fixtures/experiment_probe.cu"
            fixture.write_text(fixture.read_text().replace("#define PROBE_VALUE 11", f"#define PROBE_VALUE {value}"))
            worktrees.append(path)
    a, b = worktrees
    submitted = {}
    def submit(root, suffix, hold=1, fail=False, daemon=False, timeout=90, leave_child=False):
        job_id = args.prefix + "-" + suffix
        steps = [dict(name="build", argv=["bash", "-c", 'exec "$CUDA_HOME/bin/nvcc" "$@"', "nvcc", "-arch=sm_120a", "-O2",
            "tests/fixtures/experiment_probe.cu", "-o", "{output}/probe"], artifacts=["probe"])]
        if daemon:
            code = ("import os,pathlib,subprocess,time; "
                    "p=pathlib.Path(os.environ['IGNIS_EXPERIMENT_OUTPUT']); "
                    f"subprocess.Popen([str(p/'probe'),str(p/'probe.json'),'{hold}'], start_new_session=True); "
                    "time.sleep(120)")
            if leave_child:
                code = code.replace("time.sleep(120)", "\nwhile not (p/'probe.json').exists(): time.sleep(.1)")
            argv = ["python3", "-c", code]
        else:
            argv = ["{output}/probe", "{output}/probe.json", str(hold)]
        steps.append(dict(name="measure", argv=argv, artifacts=["probe.json"]))
        if fail:
            steps += [dict(name="failure", argv=["python3", "-c", "raise SystemExit(23)"]),
                      dict(name="unreachable", argv=["touch", "{output}/incorrect"])]
        req = dict(version=1, id=job_id, task=suffix, question="Validate isolated serial GPU execution",
                   timeout_s=timeout, steps=steps, config=dict(candidate=11 if root == a else 22, hold_s=hold))
        path = root / ".scratch/request.json"
        atomic(path, req)
        record = cli(root, "experiment", "submit", str(path))
        atomic(args.out / (suffix + "-submission.json"), record)
        submitted[suffix] = (root, job_id)
        return record
    # Two independent worktrees issue SSH submissions at the same time.
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit, a, "a", 4)
        second = pool.submit(submit, b, "b", 2)
        first.result()
        second.result()
    # Replaying a lost reply must return the same record, not enqueue a second job.
    replay = cli(a, "experiment", "retry", submitted["a"][1])
    assert replay["id"] == submitted["a"][1]
    submit(a, "daemon-success", hold=120, daemon=True, leave_child=True)
    submit(a, "failure", fail=True)
    submit(b, "cancel", hold=120, daemon=True)
    submit(a, "after-cancel")
    cancel_id = submitted["cancel"][1]
    current = wait_for(b, cancel_id, lambda row: row["state"] == "running" and
                       any(s["name"] == "measure" and "pid" in s for s in row.get("execution", {}).get("steps", [])))
    # Wait for the detached CUDA child to write its report, proving cancellation
    # happens after GPU execution, not merely during compilation.
    remote_output = current["execution"]["output"]
    check = "import pathlib,time; p=pathlib.Path(__import__('sys').argv[1]); deadline=time.monotonic()+15\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.1)\nassert p.exists()\nprint('{}')"
    cli(ROOT, "exec", "--", "python3", "-c", check, remote_output + "/probe.json")
    cli(b, "experiment", "cancel", cancel_id)
    wait_for(a, submitted["after-cancel"][1], lambda row: row["state"] in {"succeeded", "failed", "interrupted"})
    submit(a, "timeout", hold=120, daemon=True, timeout=10)
    submit(b, "after-timeout")
    wait_for(b, submitted["after-timeout"][1], lambda row: row["state"] in {"succeeded", "failed", "interrupted"})
    submit(a, "recovery", hold=10)
    submit(b, "after-recovery")
    wait_for(a, submitted["recovery"][1], lambda row: row["state"] == "running" and
             any(s["name"] == "measure" and "pid" in s for s in row.get("execution", {}).get("steps", [])))
    cli(ROOT, "exec", "--", "systemctl", "--user", "kill", "--signal=KILL", "--kill-who=main", installed["service"])
    wait_for(b, submitted["after-recovery"][1], lambda row: row["state"] in {"succeeded", "failed", "interrupted"})
    results = {}
    for suffix, (root, job_id) in submitted.items():
        result = cli(root, "experiment", "result", job_id, codes=(0, 1))
        expected = {"failure": "failed", "cancel": "cancelled", "timeout": "timed_out"}.get(suffix, "succeeded")
        assert result["state"] == expected, (suffix, result)
        assert result["payload"]["source"] == json.loads((args.out / (suffix + "-submission.json")).read_text())["payload"]["source"]
        destination = args.out / suffix
        cli(root, "experiment", "fetch", job_id, "--out", str(destination))
        report = json.loads((destination / "output/probe.json").read_text())
        assert report["cuda_passed"] and report["candidate_value"] == (11 if root == a else 22)
        assert not (destination / "output/incorrect").exists()
        results[suffix] = result
    ordered = sorted(results.values(), key=lambda result: result["started_at"])
    for previous, following in zip(ordered, ordered[1:]):
        assert previous["finished_at"] <= following["started_at"], (previous["id"], following["id"])
    assert results["a"]["payload"]["source"]["tree"] != results["b"]["payload"]["source"]["tree"]
    final = cli(ROOT, "experiment", "status")
    assert len([job for job in final["jobs"] if job["id"].startswith(args.prefix + "-")]) == len(submitted)
    atomic(args.out / "validation.json", dict(installed=installed, results=results, status=final))
    subprocess.run([sys.executable, str(ROOT / "scripts/remote.py"), "exec", "--", "make", "gpu-status"], check=True)
    # Only these terminal candidates' build/cache directories are removed.
    for root, job_id in submitted.values():
        cli(root, "experiment", "prune", job_id)
    for path in worktrees:
        git(ROOT, "worktree", "remove", "--force", str(path))
    print(json.dumps(dict(verified=len(results), output=str(args.out))))


if __name__ == "__main__":
    main()
