# Kernel Lab separates incremental compilation, correctness and timing

- Kind: experiment
- Status: current
- Observed: 2026-11-14
- Last verified: 2026-11-14
- Scope: RTX 5080, single-request Kernel Lab iteration
- Related: [initial replay measurements](2026-11-12-benchmark-kernel-replay.md), [guide](../user/contributing/kernel-lab.md)
- Superseded by: none

## Change

CUDA objects are cached independently using compiler-reported transitive
headers, toolchain identity and flags. T=8 MLP has its own compilation unit;
its arithmetic and dispatch conditions are unchanged. Shared decode helpers
remain shared, so edits to those helpers invalidate both dependent objects.

Setup selects a fixed benchmark request (Astropy by default), records its
selected operators, establishes baseline timings, and primes the object cache.
Replay checks each selected real input once against frozen outputs, then times
a fixed representative per weight/view, token count, stride and epilogue.
Repeated timing keeps inputs on device and restores mutable output/workspace
outside the timed interval. Calls are sent to the engine in batches.

Replay and quick forward no longer run full baseline workloads before and
after candidates. Both still restore builtin code and graphs. Quick forward
runs the selected request once and reuses a passing replay for the identical
candidate within the session. `forward --full` retains all requests, configured
repetitions and fresh baseline runs before/after the candidate, including after
numerical rejection. Numerical criteria are unchanged.

## Measurement

Exactly one replay and one quick forward were timed after implementation; no
old-implementation timing run or full-mode performance run was requested.
The candidate changed only a comment in the T=8 MLP source, to exercise an
uncached incremental build without changing computation. This measures workflow
latency, not kernel speedup or compiler latency for every possible edit.

Settings: the existing Qwen3.8-27B Trellis artifact on one RTX 5080, context
65,536, prefill chunk 1,024, hq-e8-2b, DFlash2, two lanes with sequential requests.
The selected request is `astropy__astropy-12907`: 19,954 prompt tokens and 913
output tokens, ending at primary EOS. Capture names are `layer.0.gate/down/qkv`,
`layer.3.q/k/output` and `head`; replay selects T=8. Performance representatives
use one warmup and three measured samples. Quick forward uses one complete run.

| Measurement | Seconds |
| --- | ---: |
| Replay client total | 9.684 |
| Replay preflight / sync / remote request | 1.274 / 1.881 / 6.529 |
| Candidate build, including dependency checks and link | 3.888 |
| Changed T=8 MLP object | 3.065 |
| Candidate link | 0.126 |
| Quick forward client total | 17.818 |
| Quick forward prefill / decode | 8.608 / 4.512 |

Linear, generic decode and prefill objects hit cache. Replay checked 1,309
records and timed seven representatives; every numerical check passed.
Quick forward reused that replay result, matched frozen outputs and termination,
and performed no full baseline reruns. The session was stopped afterward.
These single observations establish the feedback time scale, not a statistical
latency distribution. Earlier three-request/bracketed measurements use a
different scope and are not a controlled speedup comparison.

## Validation and limits

- Local Python suite: 83 tests, nine skipped (seven Linux-only, two opt-in GPU).
  Tests cover per-object source/dependency invalidation, unrelated headers,
  immutable library checks, selected-request execution, representative timing,
  same-candidate replay reuse and full-mode recovery after rejection.
- Local workspace Rust: 3,068 passed, six existing ignores.
- CUDA workspace/all-target compilation passed.
- Native GPU capture/replay regression passed, including padded destinations,
  ADD resets, rejection of a faulty plugin and baseline restoration. The first
  attempt failed because the runner had not built `fault-4.so`; it passed after
  that fixture was supplied. No numerical criterion was changed.
- Full-mode behavior has CPU control-flow coverage; `forward --full` and the
  complete GPU profile were not rerun in this iteration.
- Historical baseline timings may drift. Quick forward accepts only the selected
  workload; full mode adds coverage across all requests and fresh baseline comparisons.
- Correctness records still allocate their local buffers/graphs and read tensor
  data per record. Batch IPC and the smaller measurement set remove repeated
  work; this implementation does not cache the entire corpus in GPU memory.

Evidence: `.scratch/fast-{python-final,cargo,cuda-check,native-recheck,replay,forward}.log`;
operation JSON files under `.scratch/lab/` with IDs
`2e05775ab3ee44ec934773a22ae4c575` (replay) and
`1b5a47d2cea34c699cc83219bc085019` (forward). Remote pinned baselines, corpora and
operation results remain in the configured data directory.
