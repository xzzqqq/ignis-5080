# Testing

Add a meaningful test for every code change and pass the workspace CPU
suite before reporting completion. Record GPU results separately, including
failed checks, missing fixtures, skipped tests and unavailable hardware.

## CPU

On the Laptop, use the host Rust target:

```sh
cargo test --workspace --target aarch64-apple-darwin
python3 -m unittest discover -s tests -p 'test_*.py'
npm --prefix web test
npm --prefix web run build
```

On remote Linux, after `python3 scripts/remote.py sync`:

```sh
python3 scripts/remote.py exec -- cargo test --locked --workspace --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- make check CUDA=1
```

CUDA `--all-targets` compilation is required: CPU tests do not compile GPU
branches. The workspace vendor tests verify the manifest and recorded
patches. The reference test checks that generated pages match the code;
regenerate them with `IGNIS_BLESS_USER_DOCS=1 cargo test -p ignis-server --test user_docs_reference`.
When removing a feature, delete its exclusive tests. Keep tests for shared
behavior and update their fixtures when defaults change.

## GPU

Before any GPU command, run `make gpu-status` through the remote entry point
and ensure no other worktree or process is using the card. GPU runs are
serial. Check both memory usage and the process list.

`scripts/gpu_profile.py --artifact <trellis.ninfer> --out <new-directory>`
requires one RTX 5080 and no compute process, holds a host-wide lock, writes
the Rust preflight marker, and fails on any failed or skipped stage. It
builds/runs CTest and the named trellis GPU suites. The projection oracle also
needs the NVFP4 27B control via `--reference-artifact`; its absence is a failure.
For a symlinked artifact, put the sidecar beside the supplied path.
Write logs to a new directory under `.scratch/`.

The [coding-agent benchmark](../user/contributing/coding-agent-benchmark.md)
fixes the artifact, fixtures, settings, clocks, warmup and samples used for
performance comparisons. Run its capacity-only and canary checks separately.
Test Vision, tools, cancellation and reload when a change affects those paths.
Use eager execution, BF16 references and speculation-off runs as numerical
controls.

Keep numerical assertions and tests for operators used by the service.
For a pre-existing failure, record the reproducer and before/after results,
and report the check as failed.
