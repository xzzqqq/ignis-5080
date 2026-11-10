"""One candidate, inside a systemd cgroup. The worker publishes the final result."""
import os
from pathlib import Path
import subprocess
import time

from .source import git, sha256
from .store import Store, atomic
from .systemd import receive_lease


def artifact(output, name):
    path = output / name
    if (output.is_symlink() or not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(output.resolve())):
        raise ValueError("missing or escaping artifact: " + name)
    return dict(path=name, bytes=path.stat().st_size, sha256=sha256(path))


def run(root, job_id):
    store = Store(root)
    record = store.get(job_id)
    directory = store.directory(job_id)
    request, source = record["payload"]["request"], record["payload"]["source"]
    worktree = directory / "worktree"
    output = directory / "output"
    output.mkdir(exist_ok=True)
    execution = dict(id=job_id, source=source, steps=[], started_at=time.time(),
                     worktree=str(worktree), output=str(output), inputs=request.get("inputs", []))
    lease = None
    state, error = "failed", None
    try:
        lease = receive_lease(root, job_id)
        git(root, "worktree", "add", "--quiet", "--detach", str(worktree), source["snapshot"])
        if git(worktree, "rev-parse", "HEAD^{tree}") != source["tree"]:
            raise ValueError("checked-out source does not match request")
        config = directory / "config/.remote.cargo.toml"
        cargo_config = worktree / ".cargo/config.toml"
        if config.exists():
            cargo_config.parent.mkdir(exist_ok=True)
            if cargo_config.is_symlink():
                raise ValueError("candidate cargo config must not be a symlink")
            with cargo_config.open("ab") as handle:
                handle.write(b"\n" + config.read_bytes())
        expected_diff = git(worktree, "diff", "HEAD", "--")
        env = dict(os.environ)
        env.update(request.get("env", {}))
        env.update(CARGO_TARGET_DIR=str(worktree / "target"), TMPDIR=str(directory / "tmp"),
                   XDG_CACHE_HOME=str(directory / "cache"), CUDA_CACHE_PATH=str(directory / "cache/cuda"),
                   IGNIS_EXPERIMENT_ID=job_id, IGNIS_EXPERIMENT_OUTPUT=str(output),
                   IGNIS_EXPERIMENT_SOURCE=source["snapshot"], IGNIS_GPU_LEASE_FD=str(lease))
        for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        # The shared profile marker/lock must use /tmp even though build temp
        # files are private. gpu_profile recognizes the inherited lease fd.
        env["IGNIS_EXPERIMENT_GPU_DIR"] = "/tmp"
        execution["environment"] = {key: env[key] for key in (
            "RUSTUP_TOOLCHAIN", "CARGO_BUILD_TARGET", "CUDA_HOME", "CUDAARCHS",
            "IGNIS_KERNEL_BUILD_JOBS", "CARGO_BUILD_JOBS", "CARGO_TARGET_DIR",
            "TMPDIR", "CUDA_CACHE_PATH") if key in env}
        execution["configuration_sha256"] = {
            p.name: sha256(p) for p in (directory / "config").iterdir() if p.is_file()}
        execution["tools"] = {}
        nvcc = str(Path(env["CUDA_HOME"]) / "bin/nvcc") if env.get("CUDA_HOME") else "nvcc"
        for name, argv in {"rustc": ["rustc", "-Vv"], "nvcc": [nvcc, "--version"],
                           "python": ["python3", "--version"]}.items():
            try:
                result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=10)
                execution["tools"][name] = dict(code=result.returncode, output=result.stdout.strip())
            except (OSError, subprocess.TimeoutExpired) as exception:
                execution["tools"][name] = dict(error=str(exception))
        for item in request.get("inputs", []):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("input hash mismatch: " + item["path"])
        atomic(directory / "execution.json", execution)
        for step in request["steps"]:
            argv = [arg.replace("{output}", str(output)).replace("{worktree}", str(worktree))
                    for arg in step["argv"]]
            log = directory / (step["name"] + ".log")
            detail = dict(name=step["name"], argv=argv, started_at=time.time(), log=log.name, artifacts=[])
            execution["steps"].append(detail)
            atomic(directory / "execution.json", execution)
            with log.open("wb") as handle:
                process = subprocess.Popen(argv, cwd=worktree, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, pass_fds=(lease,))
                detail["pid"] = process.pid
                atomic(directory / "execution.json", execution)
                detail["returncode"] = process.wait()
            detail["finished_at"] = time.time()
            atomic(directory / "execution.json", execution)
            if detail["returncode"]:
                raise ValueError(f"step {step['name']} exited {detail['returncode']}")
            detail["artifacts"] = [artifact(output, path) for path in step.get("artifacts", [])]
            atomic(directory / "execution.json", execution)
        if git(worktree, "rev-parse", "HEAD") != source["snapshot"] or git(worktree, "diff", "HEAD", "--") != expected_diff:
            raise ValueError("experiment modified tracked source/configuration during execution")
        for item in request.get("inputs", []):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("input changed during execution: " + item["path"])
        state = "succeeded"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        execution["finished_at"] = time.time()
        atomic(directory / "execution.json", execution)
        atomic(directory / "done.json", dict(state=state, error=error, execution=execution))
        if lease is not None:
            os.close(lease)
    return 0 if state == "succeeded" else 1
