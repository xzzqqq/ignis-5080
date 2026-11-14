---
title: "Kernel Lab reference"
description: "Configuration, recovery, corpus semantics and development checks."
---

See the [Kernel Lab guide](kernel-lab.md) for the four-command workflow.
This page covers configuration, recovery and measurement details.

## Configure once

Run from the Laptop worktree. CUDA builds and all GPU work run in its configured
remote Linux checkout. The four commands call `scripts/remote.py sync/exec`
internally; do not manually synchronize while a lab command is running.

Use a dedicated remote development checkout and the existing private
`.remote.json` / validated `.remote.env` described in
[remote development](remote-dev.md). The remote needs systemd user services
and user lingering for a resident process to survive an SSH command ending.
Each session uses a transient systemd user unit to clean up child processes.
One RTX 5080 supports one resident session.

Create an ignored local configuration, for example `.scratch/lab-settings.json`:

```json
{
  "artifact": "/path/to/qwen3_8_27b_trellis_sc30-v2.ninfer",
  "data_dir": "/large-disk/ignis-lab",
  "build_root": "/build-cache/ignis-lab",
  "build_jobs": 4,
  "benchmark": "astropy__astropy-12907",
  "settings": {
    "max_context": 65536,
    "prefill_chunk": 1024,
    "lanes": 2,
    "kv_format": "hq-e8-2b",
    "speculation": "dflash2"
  },
  "capture_names": [
    "layer.0.qkv", "layer.0.z", "layer.0.output",
    "layer.0.gate", "layer.0.down",
    "layer.3.q", "layer.3.k", "layer.3.v", "layer.3.output", "head"
  ],
  "capture_pin_bytes": 1073741824,
  "capture_disk_bytes": 137438953472,
  "view": {"tokens": [8]},
  "warmup": 1,
  "samples": 3,
  "timeout_s": 1800,
  "setup_timeout_s": 3600
}
```

Paths in this JSON refer to the remote host. `data_dir` holds pinned baselines,
source copies, corpora and results. `build_root` is a reusable Cargo/CMake
cache. Both paths need enough free space. The lab loads the fixed text model
and allocates memory from these settings. The context must fit each prepared
prompt plus its output budget.

`capture_names` selects which Trellis calls to record from the benchmark run.
GDN layers have `qkv`, `z`,
`output`, `gate`, `up`, `down`; GQA layers (3, 7, …, 63) have `q`, `k`, `v`,
`output`, `gate`, `up`, `down`; `head` is the shared output head. Gate/up is
captured as one SwiGLU call. Omitting the selection captures all Trellis calls.

A small number of benchmark requests can still produce a very large corpus.
The owner checks a prefill-only byte lower bound before recording; decode adds
more. Setup fails if recording exceeds a budget; it does not sample to fit.
The pinned-memory budget must cover the selected graphs plus the largest eager
call; choose a smaller explicit scope or larger budget if it does not fit.

`view` selects the records to measure from that stored corpus. Supported
selectors are `names`, `tokens` and `ids`; omitted/empty selectors mean all.
The capture scope stays fixed for the session and appears in results. The example
stores both prefill and round calls, while measuring T=8 calls. Local aggregate
times are not a prediction of full-model latency.

## Execution and timing

`setup` returns when the owner, model, clean graphs and references are ready.
First setup includes building the baseline, preparing inputs and recording;
these costs are reported separately from inference. Later setup reuses
compatible data.

`benchmark` fixes the request used by replay and quick forward. The default is
`astropy__astropy-12907`; alternatives are `django__django-10097` and
`sympy__sympy-11618`. Setup records only this request's selected operators,
while preparing full-generation references for all three requests.

Replay checks every record in the configured view once against its frozen
output, in one batch request. It then measures the first record for each distinct
weight/view, token count, stride and epilogue combination with `warmup` and
`samples`. These representatives are fixed for the session. Repeated timing
runs keep inputs on the GPU, restore mutable outputs/workspace outside timing,
and compare outputs on the first execution. Raw timings and per-record speedups
against setup's baseline are in the operation result. Baseline timings can drift;
use `forward --full` when you need a fresh comparison, especially for small differences.

Quick `forward` executes the chosen request once, without a separate warmup,
through natural EOS or the original output cap. It reuses a passing replay only
for the identical candidate in this session. Otherwise it runs replay first.
`forward --full` runs all requests using configured warmups/samples and fresh
baseline runs before and after the candidate. Its final baseline run also runs
after candidate numerical rejection; a restoration check failure ends the session.
Neither mode injects recorded activations or forces baseline output tokens.

Replay and quick forward restore builtin code and clean graphs without rerunning
the baseline workload. Successful reload confirms code/graph restoration, not
full-model numerical revalidation. Fatal execution or restoration errors still
end the session. Full mode retains full-model restoration validation.

The builder caches each CUDA object by its source, compiler-reported transitive
headers, toolchain and flags. T=8 MLP is a separate compilation unit; editing it
reuses generic decode and prefill objects. Setup primes this cache. Candidate
libraries remain immutable; results report per-object hits/times and link time.

A command exiting normally leaves the engine resident; no open terminal is
required. Concurrent evaluations are refused rather than queued. `stop` uses
the recorded target and pinned control code without syncing or rebuilding,
and confirms that the owned GPU process has disappeared.
Stopping during setup cancels the waiting command and releases its command lock,
so another setup can start without waiting for the setup timeout.

## Source, baseline and recovery

Hot loading supports `trellis_linear.cu`, `trellis_linear_decode.cu`,
`trellis_linear_prefill.cu`, `trellis_linear_mlp_t8.cu`,
`trellis_linear_internal.cuh`, `trellis_decode.cuh` and
`trellis_decode_helpers.cuh` in `kernel/src/`.
Other runtime/build/ABI changes require a new baseline:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --new-baseline
```

Repeating setup for a matching live session reports its state and keeps its
baseline. Recovery also preserves the baseline. To change the
configuration, stop first and provide the new `--config` with `--new-baseline`.

Sync failure stops the command before measurement. Compilation or candidate
compatibility checks can fail without invalidating the baseline. Numerical
failure restores baseline and records the rejected case. `timeout_s` bounds
the entire accepted evaluation, including compilation and all its samples;
`setup_timeout_s` bounds setup. CUDA failure, timeout, malformed or out-of-order
engine replies, or failed restoration end the session. A lost reply does not trigger a retry; inspect
the saved result and logs.

Local control receipts and full returned results live under `.scratch/lab/`.
Remote session directories retain their own `session.json`, settings, prepared
inputs, engine JSONL/stderr, source copies, build logs and per-operation results,
after the session ends. Baseline metadata keeps the source revision and patch. Reports separate client preflight/sync/remote
round-trip time from owner build/reload/measurement time. CLI output is a
compact summary; retain the linked full result for analysis. Do not delete the
control receipts while a session is running.

## Corpus and measurement semantics

The inputs come from the frozen Astropy, Django and SymPy conversations and
tools in `crates/bench/fixtures/coding_agent/`. The offline preparer reuses
instruction/history and tool-default normalization, the artifact chat template
and tokenizer. Generation uses the primary EOS, reasoning-close behavior,
greedy sampling and the benchmark's 8,192-token maximum; no tools execute.

Use `diagnostic_output_tokens` to shorten functional checks. These truncated
runs cannot establish full-workload performance.

A corpus consists of `manifest.json`, `records.jsonl` and `data.bin`. It stores
BF16 call inputs, initial targets and reference outputs plus weight/view,
shape/stride, request/phase/position metadata. Weights are rebound by stable
logical name and view offsets in the current model; GPU pointers are never
persisted. Model/baseline/prepared-input/settings/hardware/scope identities and
file hashes are checked at setup. Missing, incompatible or corrupted data are
not reused. Partial recording directories are not complete corpora.

Recording graphs copy actual executed tensors; host calls at graph-construction
time are not counted as executions. After recording, the engine destroys those
graphs and captures clean ones before measuring. A separate clean run checks
that recording did not change the generated trajectory.

The default gate requires exact outputs and finite values. Arithmetic
changes that alter output bits require a separately justified correctness check.
Local replay has different cache conditions from complete forward; report its coverage and use full forward for whole-model conclusions.

These are engine timings, excluding HTTP and scheduler work. Do not compare
them directly with historical service TPS. Use the service workflow for
acceptance measurements.

## Development checks

The workspace and Python suites check CPU behavior. The opt-in
end-to-end regression uses the real four-command interface and the single GPU:

```sh
IGNIS_LAB_E2E_CONFIG="$PWD/.scratch/lab-settings.json" \
  python3 -m unittest discover -s tests -p test_lab_gpu.py
```

The test temporarily edits a supported CUDA file to check candidate identity,
compilation failure and incorrect outputs. It restores the file in `finally`
and stops the session. Run with no other writer or GPU workload.
A deliberately short diagnostic configuration is suitable for that functional
check and must not be reported as performance evidence.

The native `lab_capture_gpu` and `trellis_reload_gpu` ignored tests cover
recording/clean-graph parity, padded output, repeated ADD restoration, invalid
views, actual plugin use and rollback. Run them on an exclusive card with the
artifact and the test-only fault libraries described by their test headers.
The normal GPU profile covers the unchanged serving path; see
[testing](../../agents/testing.md).
