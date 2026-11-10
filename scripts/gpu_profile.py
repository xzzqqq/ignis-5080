#!/usr/bin/env python3
"""Serialized RTX 5080 checks. Missing hardware/fixtures and skipped legs fail."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CORE_TESTS = ("trellis_gpu", "trellis_projections_gpu", "trellis_layers_gpu", "trellis_lanes_gpu")


def check_card(run=subprocess.check_output):
    cards = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(cards) != 1 or "RTX 5080" not in cards[0]:
        raise RuntimeError("profile requires exactly one RTX 5080")
    apps = run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if apps:
        raise RuntimeError("GPU is held by compute processes:\n" + apps)


@contextlib.contextmanager
def exclusive_marker(directory):
    lock = Path(os.environ.get("IGNIS_EXPERIMENT_GPU_DIR", directory)) / "ignis-gpu-profile.lock"
    marker = directory / "ignis-gpu-preflight.ok"
    inherited = os.environ.get("IGNIS_GPU_LEASE_FD")
    if inherited is not None:
        fd = int(inherited)
        held, expected = os.fstat(fd), lock.stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("inherited GPU lease does not match the host lock")
        handle = os.fdopen(os.dup(fd), "a")
    else:
        handle = lock.open("a")
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Never replace another harness's pass, even if it does not use our lock.
        with marker.open("x") as stamp:
            stamp.write(str(time.time()))
        try:
            yield marker
        finally:
            marker.unlink()


def stage(command, log, env, marker):
    subprocess.run(["make", "gpu-status"], cwd=ROOT, check=True)
    check_card()
    marker.touch()
    print("GPU profile:", " ".join(command), flush=True)
    skipped = False
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                output.write(line)
                print(line, end="", flush=True)
                skipped |= "SKIP:" in line
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if code or skipped:
        raise RuntimeError(f"stage failed: exit={code}, skipped={skipped}; {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=os.environ.get("IGNIS_MODEL_ARTIFACT"))
    parser.add_argument("--reference-artifact", type=Path, default=os.environ.get("IGNIS_REFERENCE_ARTIFACT"),
                        help="NVFP4 27B control for the trellis projection oracle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact is None or not args.artifact.is_file():
        parser.error("--artifact (or IGNIS_MODEL_ARTIFACT) must name the trellis artifact")
    args.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(["make", "gpu-status"], cwd=ROOT, check=True)
    subprocess.run(["make", "gpu-guard"], cwd=ROOT, check=True)
    check_card()
    env = dict(os.environ, IGNIS_GPU_PROFILE="1", IGNIS_MODEL_ARTIFACT=str(args.artifact.absolute()))
    if args.reference_artifact is not None:
        env["IGNIS_REFERENCE_ARTIFACT"] = str(args.reference_artifact.absolute())
    commands = [("kernel", ["bash", "kernel/build.sh", "build", "--test"])]
    for name in CORE_TESTS:
        commands.append((name, ["cargo", "test", "--locked", "-p", "ignis-core", "--features", "cuda",
                                "--target", "x86_64-unknown-linux-gnu", "--test", name, "--",
                                "--ignored", "--test-threads=1", "--nocapture"]))
    commands.append(("vram_reload_gpu", ["cargo", "test", "--locked", "-p", "ignis-server", "--features", "cuda",
                                       "--target", "x86_64-unknown-linux-gnu", "--test", "vram_reload_gpu", "--",
                                       "--ignored", "--test-threads=1", "--nocapture"]))
    env["IGNIS_ARTIFACT_27B"] = str(args.artifact.absolute())
    failures = []
    with exclusive_marker(Path(tempfile.gettempdir())) as marker:
        for name, command in commands:
            try:
                stage(command, args.out / f"{name}.log", env, marker)
            except RuntimeError as error:
                failures.append(str(error))
    if failures:
        raise RuntimeError("GPU profile failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
