# Resident Trellis experiments remove repeated model loading

- Kind: experiment
- Status: current
- Observed: 2026-11-12
- Last verified: 2026-11-12
- Scope: experiment workflow / RTX 5080, candidate compilation, model startup
- Related: [resident experiment design](../specs/continual-experiments/README.md), [coding-agent baseline](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

Where does starting an optimization experiment spend time, and what is the
smallest useful resident-engine boundary for our fixed 27B model?

## Evidence

The host is one RTX 5080 16 GB, CUDA 13.0.88, driver 580.105.08, Rust 1.99.1.
GPU jobs ran serially under the existing experiment queue. Sources were frozen
from `89361ce47b87a9c4ac1d3a102a4f96afab19a123` plus the resident-engine changes.
The Trellis artifact SHA-256 is
`2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c`.

Historical service evidence is the existing `mlp2-static-service-003` job.
Its six starts use the fixed service configuration: two lanes, DFlash2 window
7, full draft head, 1024 prefill chunk, hq-e8-2b KV, host Vision, prefix reuse
off. That experiment's candidate Trellis implementation is the one integrated
into the starting revision. Timestamp differences in its structured logs give:

| Phase | Six-launch range |
|---|---:|
| `model.loading` → `process.ready` | 6.230–6.271 s |
| Artifact verification + frontend | 0.227–0.240 s |
| VRAM plan → KV pool ready | 5.808–5.860 s |
| CUDA graph capture, reported internally | 28.001–28.774 ms |
| First traversal warmup, reported internally | 60–61 ms |

These are log boundaries, not an OS process-launch stopwatch. The plan-to-pool
interval combines weight materialization and allocation; the old log does not
separate them. The candidate's release-build log reports 111 s; the whole
six-start experiment takes 2079.42 s. That total includes measurements and
checks, and must not be presented as startup time.

Fresh job `continual-lab-1112-002` uses the new resident Rust example with the
full real 64-layer model, BF16 KV, 256 context, 128 prefill chunk, two lanes and
no Vision. Its measured phases are:

| Phase | Speculation off | DFlash2 window 7 |
|---|---:|---:|
| Open, bind, frontend | 0.219 s | 0.219 s |
| CUDA context | 1.232 s | 1.237 s |
| Weight materialization | 4.020 s | 4.378 s |
| Model/workspace | 3.49 ms | 3.63 ms |
| Sequence pool | 0.10 ms | 0.12 ms |
| Initial graph capture | 12.01 ms | 32.67 ms |
| Total startup inside the example | 5.487 s | 5.870 s |
| First candidate reload, including SHA-256 and graph capture | 13.91 ms | 23.11 ms |
| Restore built-in code | 6.94 ms | 13.22 ms |

The frozen job's cold CUDA workspace/all-targets check takes 198.16 s; the
subsequent release example build takes 11.08 s. Compiling the three candidate
translation units into a library takes 31.14 s. Source checkout, tool records
and artifact hashing before its first step take another 5.87 s. Uncontended
queue dispatch in the preceding jobs takes approximately 6 s. These phases
are distinct, and existing incremental caches will change build cost.

A second fresh job, `continual-lab-1112-003` (snapshot
`d1895e06838be4ddea1cbb169c32834d18eb10f3`), reproduces the lab result:
5.472/5.906 s cold startup, 13.52/22.73 ms candidate reload, and 31.10 s candidate
compilation. Three standard production `make start` launches take
8.829/8.868/8.881 s wall time including the GPU guard; their internal
load-to-ready intervals are 6.243/6.304/6.279 s. The server release build in
this already-populated workspace takes 20.34 s. That warm build is not
comparable to either the historical fresh 111-second build or the 198-second
all-targets check; cache state must accompany build numbers.

The job's same-code built-in → library → built-in comparisons preserve
prefill-logit hashes and generated tokens with speculation off and on. The
ordered campaign checks GDN layer 0, GQA layer 3 and full DFlash2 decode. Its
same-code speed ratios are 0.9961, 1.0084 and 0.9990, respectively, against
bracketing controls: no inference improvement is claimed. Layer residuals in
this validation are synthetic, explicitly a wiring/numerical check. The
forward case tokenizes an actual short chat prompt; it is not the coding-agent
benchmark.

Raw evidence is retained under
`.scratch/experiment-results/mlp2-static-service-003/` and
`.scratch/experiment-results/continual-lab-1112-002/`: structured service logs,
`output/steps.json`, `output/{off,dflash2}-session.json`,
`output/campaign/run.json`, and `output/reload-tests.log`. The repeated run's
evidence is in `.scratch/experiment-results/continual-lab-1112-003/`, including
three new daemon logs and the complete profile logs. The first attempt,
`continual-lab-1112-001`, failed CUDA compilation because the example accessed
private `Speculation` fields; using the existing accessors fixed it. That
failed attempt is retained and is not counted as a successful measurement.

The latest reload regression passes both speculation modes, repeated
candidate/built-in cycles, and failures for ABI, workspace, preparation,
missing symbol, active sequence and graph capture. A graph-safe deliberately
wrong plugin produces zero-valued later decode/verify tokens, proving that
replacement does not silently replay the old graph; restoring built-in code
recovers the original observation. The full GPU profile passes all 130 CTest
checks and all five named Rust suites (six tests), with no skips, in 521.93 s.

Local `cargo test --workspace` passes: 3,066 tests, six pre-existing ignored
checks. Python passes 65 checks with seven Linux-only checks skipped on macOS.
Job `continual-lab-cpu-1112-004` also passes the Linux workspace suite and all
65 Python checks with no Python skips. Its logs are retained in
`.scratch/experiment-results/continual-lab-cpu-1112-004/output/`.

## Finding

Observed: the current service starts in about six seconds, rather than the
minutes described by the motivating article. Weight loading dominates that
startup. Keeping the model resident makes a same-code Trellis replacement a
14–23 ms operation in the tested lab settings, including fresh graphs.
Candidate compilation remains a roughly 31-second cost in this implementation.

Inference: for this repository, experiment throughput benefits from combining
three changes: smaller candidate builds, model residency, and early rejection
with layer/forward measurements. Hot loading alone does not remove compilation
or the duration of a complete coding-agent benchmark.

The least disruptive boundary observed in code is `trellis_body`'s linear and
SwiGLU calls. Model-local function tables replace their code while preserving
all owned state. The plugin ABI uses stable tensor/workspace descriptors, not
the internal C++ model structure or a dynamically loaded Rust engine.

## Implications

Use a four-level funnel: operator contract/numerical checks; resident layer
replay; resident full forward; rebuilt production service. Keep real-input
provenance and cache/state effects visible. Preserve the existing coding-agent
benchmark as the production performance denominator.

A fixed model makes the compatibility envelope small: enumerate actual tensor
shapes, quantization widths, layer kinds and all served token widths on one
GPU architecture. It removes the need for generic graph partitioning or
multi-GPU strategy selection. Context, lanes, chunk tails, speculative
acceptance and Vision still vary and require separate measurements.

## Limits and unknowns

- Lab startup and production startup use different settings; their absolute
  numbers are not a controlled speed comparison.
- Two observed lab cold starts and reloads per mode are latency measurements,
  not a statistical distribution. Compilation, input hashing and thermal warmup
  remain outside reload time.
- The library only replaces Trellis projections. Topology, quantization,
  workspace capacity, attention and Rust scheduler changes still start cold.
- Layer replay starts with empty state at position zero. Real residual capture
  plus long-context state replay is a future extension; synthetic layer inputs
  cannot establish data-dependent performance.
- The exact-output gate may reject valid arithmetic changes. Those require
  independent numerical tolerances/oracles and complete service validation.
- A campaign owns one queue job. Cross-job engine reuse, online service
  replacement and automatic deployment are not implemented.

## Follow-ups

The [specification](../specs/continual-experiments/README.md) defines the
replacement and measurement contracts. The [guide](../user/contributing/kernel-lab.md)
contains repeatable build, replay and fault-injection commands.

The [subsequent four-command and benchmark-replay measurement](2026-11-12-benchmark-kernel-replay.md)
replaces the original zero-state layer workflow with persisted operator inputs.
The startup figures above describe the original harness, not corpus preparation.
