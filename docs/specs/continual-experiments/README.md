# Resident experiments for the fixed 27B model

## Problem and source

An experiment currently freezes/transfers source, waits for the GPU lease,
configures/builds the native leaf and Rust binary, starts a new CUDA context,
loads weights, allocates state, captures graphs, warms up, then measures.
Queue time, compilation, readiness, warmup and measurement must be reported
separately. A total job duration is not an engine-startup measurement.

[Together Compiler](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference)
(2026-11-04) retains engine state while replacing candidate computation and
uses progressively more representative measurements. Those two ideas fit this
fork. Its fleet-wide parallelism search is outside our single-GPU problem.

## Boundary

Keep the model, weights, stream, KV/GDN pool, scheduler and memory ownership in
Ignis. Replace the two Trellis entry points used by `trellis_body`: linear and
SwiGLU. A library contains just the three existing Trellis translation units
and a versioned function table. This covers the current decode MLP search and
other Trellis projections without exposing the model's C++ layout to plugins.

Replacing the whole Rust engine would require a new state-transfer and ABI
contract. Replacing the whole C++ leaf would also couple old model handles to
new internal layouts. Loading individual cubins would require a second launch
path for the existing CUDA-runtime wrappers, including PDL. The two existing
Trellis functions already take device views and caller-owned workspace, so
this boundary needs no model-state migration.

The experiment engine is a Rust example using the normal model/step ABI. It
has a serial JSONL control channel, no HTTP route and no production deployment
action. A replacement is model-local. The default service uses built-in code.
Changing Rust scheduling, attention, KV format, quantization, model topology,
weights, or the workspace budget still requires a cold engine. This is an
explicitly bounded implementation, not arbitrary forward-pass replacement.

Reload requires an empty sequence pool on its owning model thread. It checks
the ABI and every actual model shape/width against the reserved workspace,
synchronizes the stream, loads and prepares the candidate, and captures new
decode and verify graphs. Old graphs and their library stay alive until the
replacement succeeds. Missing symbols, ABI/workspace refusals or incomplete
capture retain the old generation. Resetting all workspace bytes prevents
candidate scratch from contaminating the next arm. CUDA execution failures
can poison the context; terminate that session and start a fresh one.

A native plugin is trusted executable code. ABI validation catches accidental
incompatibilities; it is not a sandbox or proof of numerical correctness.
The caller never overwrites a loaded library. Content-addressed builds record
source/header hashes, compiler version, flags and library SHA-256. Source and
artifact identity remain the remote experiment queue's responsibility.

## Measurement levels

| Level | Measurement | Purpose and limit |
|---|---|---|
| 0: contract and operator | CPU request/ABI checks, candidate compilation, existing Trellis numerical GPU tests, workspace admission | Reject malformed and numerically wrong kernels before expensive runs. Use BF16/f64 oracles; random/synthetic inputs alone do not establish model behavior. |
| 1: layer | Resident GDN/GQA layer with supplied BF16 residual, fresh sequence state, widths 1/8/16 and prefill widths | Cheap integration feedback for projections, activation layout and residuals. Record input hash/provenance. A position-zero replay does not reproduce a long-context recurrent state or cache pressure. |
| 2: forward | Resident full 64-layer prefill/decode, actual token arrays, speculation off and DFlash2, one/two lanes | Detect graph, state, speculative and full-model regressions. Record logits hashes, output tokens, per-round widths/times and acceptance effects. Fixed-output-budget timing is diagnostic, not the service benchmark. |
| 3: service | Rebuilt production binary, frozen coding-agent requests and settings, canary/capacity and lifecycle checks | Confirm actual latency and correctness before integration. Include Vision/tools/cancellation/reload when affected. Only this level can support a production performance claim. |

The layer and forward levels share one resident full-model load. This fixed
artifact fits the card, so a separate few-layer loader would add ownership and
binding complexity without improving per-candidate startup. The layer command
executes only its selected layer; it does not reduce the initial weight load.

The supplied driver runs level 1 before level 2, stops a candidate at the first
numerical mismatch and restores built-in code. One warmup and three measured
samples are defaults, configurable in the experiment CLI. Built-in controls
bracket each candidate. Raw samples and drift ratios are retained; the driver
does not automatically choose or deploy a winner.

The initial gate requires exact layer output, prefill-logit hashes and decoded
tokens. This intentionally rejects some numerically valid arithmetic changes.
Such changes need a named numerical oracle/tolerance and the existing full GPU
profile, not a switch that disables checks. An exact token match by itself is
insufficient because it can hide large logit changes.

## What fixing the model removes

The artifact fixes tensor shapes, K distributions, layer types (48 GDN and 16
GQA), hidden width, vocabulary and weight layout. The device fixes SM120a.
These become the compatibility envelope and replay corpus, eliminating model
architecture discovery, graph partition search, distributed parallelism and
multi-platform dispatch from this implementation.

Traffic remains variable: context length, prefill chunk tails, lane count,
DFlash2 acceptance, prompt reuse and Vision change the work. The fixed model
therefore reduces the search space; it does not justify benchmarking only one
synthetic T=8 matrix or treating a layer speedup as an end-to-end speedup.

## Resource and promotion policy

One 5080 means one active engine. Run a candidate campaign inside one frozen
`remote.py experiment` job and its inherited GPU lease; do not run a persistent
engine beside serving or escape the queue's cgroup. A campaign can test several
libraries while amortizing the load. Different campaigns still start cold.
Reusing a resident engine across separately submitted queue jobs is deferred:
it would change source isolation, failure recovery and lease ownership.

Use the existing fixed coding-agent baseline as the level-3 denominator.
Measure experiment-throughput gains separately from inference gains. This
change adds infrastructure and makes no model-throughput improvement claim.

## Acceptance

- Workspace CPU tests, Python tests and CUDA all-targets compilation pass.
- On a single idle RTX 5080, one process loads the real artifact and runs
  built-in → candidate → built-in with identical outputs and unchanged model
  and pool allocations, with speculation off and on.
- ABI, checksum, workspace, active-sequence and capture failures do not advance
  generation or corrupt the old engine.
- A changed computation is observed after replacement; stale CUDA graphs cannot
  silently make the same-code parity test pass.
- Startup and each replacement are timed separately, and all failed attempts
  remain in the evidence record.

See the [usage guide](../../user/contributing/kernel-lab.md) for commands.
