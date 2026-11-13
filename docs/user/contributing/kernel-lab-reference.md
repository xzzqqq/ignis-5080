---
title: "Kernel Lab reference"
description: "Configuration, recovery, corpus semantics and development checks."
---

For the daily workflow and four commands, start with the
[Kernel Lab guide](kernel-lab.md). Consult this page when configuring a lab,
changing its baseline, diagnosing failures or interpreting measurements.

## Configure once

Run from the Laptop worktree. CUDA builds and all GPU work run in its configured
remote Linux checkout. The four commands call `scripts/remote.py sync/exec`
internally; do not manually synchronize while a lab command is running.

Use a dedicated remote development checkout and the existing private
`.remote.json` / validated `.remote.env` described in
[remote development](remote-dev.md). The remote needs systemd user services
and user lingering for a resident process to survive an SSH command ending.
The owner uses one transient user unit per session for child-process cleanup;
it does not use the experiment queue. One 5080 admits one resident session.

Create an ignored local configuration, for example `.scratch/lab-settings.json`:

```json
{
  "artifact": "/path/to/qwen3_8_27b_trellis_sc30-v2.ninfer",
  "data_dir": "/large-disk/ignis-lab",
  "build_root": "/build-cache/ignis-lab",
  "build_jobs": 4,
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
cache; use a sufficiently large filesystem (or a suitably sized RAM disk).
Only the fixed text model is loaded; this is not the service's adaptive memory
planner. The context must fit each prepared prompt plus its output budget.

`capture_names` is an explicit scope, not a set of synthetic inputs. It can
select any supported layer's Trellis weights. GDN layers have `qkv`, `z`,
`output`, `gate`, `up`, `down`; GQA layers (3, 7, …, 63) have `q`, `k`, `v`,
`output`, `gate`, `up`, `down`; `head` is the shared output head. Gate/up is
captured as one SwiGLU call. Omitting the selection captures all Trellis calls.

A small number of benchmark requests can still produce a very large corpus.
The owner checks a prefill-only byte lower bound before recording; decode adds
more. Budgets are explicit and exhaustion fails setup, never silently samples.
The pinned-memory budget must cover the selected graphs plus the largest eager
call; choose a smaller explicit scope or larger budget if it does not fit.

`view` selects the records to measure from that stored corpus. Supported
selectors are `names`, `tokens` and `ids`; omitted/empty selectors mean all.
Coverage remains fixed for a session and is visible in results. The example
stores both prefill and round calls, while measuring T=8 calls. Local aggregate
times are not a prediction of full-model latency.

## Execution and timing

`setup` returns when the owner, model, clean graphs and references are ready.
First setup includes building the baseline, preparing inputs and recording;
these costs are reported separately from inference. Later setup reuses
compatible data.

`replay` checks the baseline before loading the candidate and again after
restoring it. Each sample restores mutable buffers and workspace outside
CUDA-event timing. All comparisons use the same view, warmup and sample counts.
`forward` also measures baseline full runs before and after the candidate.
It starts a new sequence for each request and never injects recorded activations
or forces baseline output tokens. C1 means sequential requests; two configured
lanes remain available in the model.

A command exiting normally leaves the engine resident; no open terminal is
required. Concurrent evaluations are refused rather than queued. `stop` uses
the recorded target and pinned control code without syncing or rebuilding,
and confirms that the owned GPU process has disappeared.

## Source, baseline and recovery

The admitted hot files are the three `trellis_linear*.cu` units plus
`trellis_linear_internal.cuh` and `trellis_decode.cuh` in `kernel/src/`.
Other runtime/build/ABI changes require a new baseline:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --new-baseline
```

This choice is explicit: recovery must not silently promote the most recently
edited candidate to baseline. Repeating setup for a matching live session
reports the existing state; it does not change that baseline. To change the
configuration, stop first and provide the new `--config` with `--new-baseline`.

Sync failure cannot fall through to a measurement of old code. Compile and
admission errors return failure while preserving a usable baseline. Numerical
failure restores baseline and records the rejected case. `timeout_s` bounds
the entire accepted evaluation, including compilation and all its samples;
`setup_timeout_s` bounds setup. CUDA failure, timeout, malformed/out-of-order
engine replies or failed restoration ends the session. Lost replies are not automatically
replayed; the attempt and its logs remain inspectable.

Local control receipts and full returned results live under `.scratch/lab/`.
Remote session directories retain their own `session.json`, settings, prepared
inputs, engine JSONL/stderr, source copies, build logs and per-operation results,
even after another session replaces the active pointer. Baseline metadata keeps
the source revision and patch. Reports separate client preflight/sync/remote
round-trip time from owner build/reload/measurement time. CLI output is a
compact summary; retain the linked full result for analysis. Do not delete the
control receipts while a session is running.

## Corpus and measurement semantics

The inputs come from the frozen Astropy, Django and SymPy conversations and
tools in `crates/bench/fixtures/coding_agent/`. The offline preparer reuses
instruction/history and tool-default normalization, the artifact chat template
and tokenizer. Generation uses the primary EOS, reasoning-close behavior,
greedy sampling and the benchmark's 8,192-token maximum; no tools execute.

For explicit short functional checks, `diagnostic_output_tokens` can reduce the
output budget. Such a run is a truncated diagnostic, not the full workload.

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

The default gate requires exact outputs and finite values. Valid arithmetic
changes that alter bits need an independently justified oracle, not a disabled
check. Local replay has different cache conditions from complete forward;
report its coverage and use full forward for whole-model conclusions.

These are engine timings, excluding HTTP and scheduler work. Do not compare
them directly with historical service TPS. Service acceptance is a separate
workflow outside this feature.

## Development checks

CPU contracts run with the ordinary workspace and Python suites. The opt-in
end-to-end regression uses the real four-command interface and the single GPU:

```sh
IGNIS_LAB_E2E_CONFIG="$PWD/.scratch/lab-settings.json" \
  python3 -m unittest discover -s tests -p test_lab_gpu.py
```

It temporarily edits an admitted CUDA file to exercise changed candidate
identity, compiler failure and deliberately wrong computation, restores it in
`finally`, and stops its session. Run with no other writer or GPU workload.
A deliberately short diagnostic configuration is suitable for that functional
check and must not be reported as performance evidence.

The native `lab_capture_gpu` and `trellis_reload_gpu` ignored tests cover
recording/clean-graph parity, padded output, repeated ADD restoration, invalid
views, actual plugin use and rollback. Run them on an exclusive card with the
artifact and the test-only fault libraries described by their test headers.
The normal GPU profile covers the unchanged serving path; see
[testing](../../agents/testing.md).
