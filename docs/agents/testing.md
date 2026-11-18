# Testing

## When to validate

Run final acceptance at the end of an optimization round on the accumulated
changes intended for delivery or merge. During exploration, use replay and
targeted checks; do not repeat the full acceptance suite for each candidate.
Choose regression coverage from the accumulated changes and their affected
paths, not from the number of experiments. Documentation-only edits need
content and link checks, not CPU or GPU suites.

## Regression coverage

### CPU and build checks

Run the tests for changed components and affected dependents. Use the full
Rust workspace suite for cross-cutting Rust changes or when narrower coverage
cannot establish the impact. A kernel-only change does not by itself require
Python or Web suites; run those when their code or behavior is affected.

Local commands, selected by scope:

```sh
cargo test -p <affected-crate>
cargo test --workspace
python3 -m unittest discover -s tests -p 'test_*.py'
npm --prefix web test
npm --prefix web run build
```

Cargo defaults to the local host target. Checks requiring Linux or CUDA run
remotely. When executing changed source outside Kernel Lab, sync it first:

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- cargo test --locked -p <affected-crate> --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- cargo test --locked --workspace --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- make check CUDA=1
```

For kernel or CUDA-path changes, include CUDA `--all-targets` compilation and
relevant operator and integration tests: CPU tests do not compile GPU branches.
For vendored kernel changes, include the manifest and recorded-patch checks.
For generated user-reference changes, regenerate with
`IGNIS_BLESS_USER_DOCS=1 cargo test -p ignis-server --test user_docs_reference`
and run the reference test.

On Linux, keep `TMPDIR` outside any Git checkout, on a filesystem that
supports `O_DIRECT` and has more than 2 GiB free for the packer's safety margin.
Artifact tests and the layer oracle use the normal reader for temporary
containers; `/dev/shm` cannot serve those reads on this host. Build caches may
use RAM.

### GPU regression checks

Before GPU work, check `make gpu-status` through the remote entry point and
coordinate exclusive use of the card. A resident Lab or service may own the
GPU; do not launch a competing process. Check memory usage and process identity.

`scripts/gpu_profile.py --artifact <trellis.ninfer> --out <new-directory>`
is the GPU regression runner. It requires one RTX 5080 and no compute process,
holds a host-wide lock, writes the Rust preflight marker, and fails on any
failed or skipped stage. It builds/runs CTest, the four Trellis GPU suites
and the VRAM reload test. It does not run service performance, capacity or
canary checks. The projection oracle needs the NVFP4 27B control via
`--reference-artifact`; its absence is a failure. For a symlinked artifact,
put the sidecar beside the supplied path. Write logs to a new directory
under `.scratch/`.

## Optimization acceptance

Validate the final combined version's service performance using the fixed
[coding-agent benchmark](../user/contributing/coding-agent-benchmark.md),
which defines the artifact, fixtures, settings, clocks, warmup and samples.
Replay speedups alone do not establish a service performance improvement.
Run the benchmark's capacity-only and canary checks separately. Add Vision,
tools, cancellation and reload checks when the accumulated changes affect
those paths.

Follow [AGENTS.md](../../AGENTS.md#numerical-correctness) for numerical
correctness. Choose checks appropriate to the transformation, using independent
references and eager execution or speculation-off controls where useful.
Do not weaken checks to hide errors; a bitwise mismatch is not automatically
a failure of numerical equivalence.

Record the coverage and results, including failed or skipped checks, missing
fixtures and unavailable hardware. For a pre-existing failure, retain the
reproducer and before/after results, and report the check as failed.
