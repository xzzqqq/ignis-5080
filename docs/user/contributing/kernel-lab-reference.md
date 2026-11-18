---
title: "Kernel Lab reference"
description: "Configuration, candidate comparisons, profiling and recovery."
---

See the [Kernel Lab guide](kernel-lab.md) for the edit/replay workflow.

## Configure once

Run from the Laptop worktree. CUDA builds and GPU execution use the configured
remote Linux checkout through `scripts/remote.py`. Use a dedicated checkout and
the private `.remote.json` / validated `.remote.env` described in
[remote development](remote-dev.md). The remote needs systemd user services and
user lingering. One transient systemd user unit owns the session's processes;
one RTX 5080 supports one session. Do not manually sync during a lab command.

Create an ignored local file, for example `.scratch/lab-settings.json`:

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
  "capture_pin_bytes": 1073741824,
  "capture_disk_bytes": 137438953472,
  "warmup": 1,
  "samples": 3,
  "timeout_s": 1800,
  "setup_timeout_s": 3600
}
```

Paths refer to the remote host. `data_dir` holds pinned baselines, source copies,
candidates, recordings and results. `build_root` holds the Cargo/CMake cache.
The context must fit each prepared prompt and its output budget. `benchmark`
fixes the request used by recording and quick forward. The default is
`astropy__astropy-12907`; alternatives are `django__django-10097` and
`sympy__sympy-11618`. `diagnostic_output_tokens` explicitly truncates generation
for functional checks; these runs cannot establish full-workload performance.

## Select inputs per command

Setup fixes model, runtime settings and request, without requiring one permanent
operator scope. Use repeatable `--operator` and `--tokens-per-call` selectors on
replay, compare and profile. Token count means rows in an individual operator
call, rather than the request's output budget. Selectors and actual coverage are
persisted with each result. Omitted selectors select all supported operators and
all token shapes; choose an explicit scope for quick feedback.

GDN layers expose `qkv`, `z`, `output`, `gate`, `up`, `down`. GQA layers
(3, 7, …, 63) expose `q`, `k`, `v`, `output`, `gate`, `up`, `down`.
`head` is the shared output head. Gate/up is one SwiGLU call. GQA layers also
expose `attention` through the explicit C1/T8 HQ scope described below.

The first command selecting an uncached scope records real calls from the fixed
request using builtin code. Compatible later commands reuse the recording.
Recording obeys `capture_pin_bytes` and `capture_disk_bytes` and available disk
space. It fails when the selected scope cannot fit; it does not silently sample
to fit. Narrow the scope or increase the budgets. Legacy `capture_names` and
`view` settings are rejected; replace them with command selectors. Changing scope
does not reload the model.

Recordings contain BF16 inputs, initial mutable targets and frozen outputs,
weight/view and shape/stride metadata, request and execution positions. Weights
are rebound in the resident model by stable identity; GPU pointers are never
persisted. Graph capture copies tensors at actual execution, then removes
recording nodes and rebuilds clean graphs. Incompatible, corrupted or partial
recordings are not reused.

## Replay and candidates

Replay is quick exploration: errors are reported, not strict acceptance gates.
It checks selected real inputs against their frozen outputs and times fixed
representatives of each distinct weight/view, token count, stride and epilogue.
It reports output error statistics even when numerical values differ and still
runs timing. Execution failures remain failures. Raw samples and numerical
results are retained. A completed exploratory replay does not approve numerical
changes for shipping.

Inputs stay on the GPU across timing repetitions. Mutable outputs and workspace
are restored outside CUDA event timing. Transfer, file I/O, recording and reload
costs are separate from operator samples. Representatives are selected by input
identity, never by a candidate's performance. Local cache conditions differ from
full forward; aggregate replay times are not full-model latency estimates.

`--candidate current` builds the synchronized source. `builtin` means the pinned
session baseline. `replay --save-as NAME` keeps the immutable compiled candidate
and its identity under a name. `candidates` lists saved candidates.
`compare --baseline NAME --candidate current` runs both candidates against the
same selected records and reports timing ratios and errors. Compare uses fresh
measurements; old setup timing is not its control. Keep small speedup claims
proportional to sample variability.

Compare alternates baseline/candidate order across pairs. `--samples` selects
the number of pairs; `--warmup` selects warmup executions for every leg.
Operator comparison checks all selected inputs once per immutable candidate
per invocation, then times fixed representatives in alternating legs.
`compare.checks` retains the complete diagnostics; every warmup/timing execution
also retains output diagnostics. No diagnostic result is cached across commands. Both flags also
override configured repetition counts on replay, forward and profile; warmup
may be zero and samples must be positive. Use
`compare --baseline before --candidate current --forward` for paired full-model
feedback on the fixed request, or add `--full` for all requests. Operator/token
selectors apply to operator comparisons and cannot be combined with `--forward`.

The builder caches CUDA objects by source, transitive headers, toolchain and
flags. T=8 MLP is a separate compilation unit, so editing it reuses generic
decode/prefill objects. Candidate libraries stay immutable. Results include
candidate identity, cache hits and compile/link costs.

## Stateful GQA attention replay

```sh
python3 scripts/lab.py replay --candidate builtin --operator layer.3.attention --tokens-per-call 8
```

`layer.N.attention` records the native append-and-attend operator after Q/K
normalization and RoPE, before output projection. It currently requires an
explicit token filter containing only `8` and an `hq-e8-2b` pool. Only actual C1
calls are instrumented; unused C2 graph variants are excluded even when the
pool has two lanes. The fixed benchmark executes C1. Prompt attention, C2
attention, other widths and cache formats are unsupported; omitted operator
selectors retain their existing Trellis scope. A request producing no C1/T8
calls fails selection rather than inventing inputs.

Version-1 records contain real contiguous Q, K, V and gate tensors; device
positions, valid-column mask and slot selectors; full block tables; and the
selected layer's complete physical K/V code and metadata planes, BF16 residual
K/V side planes and shared ring-valid words. Inputs and pre-state are copied at
actual graph execution. Output and post-state are copied immediately after the
attention call, before verify rejection processing can invalidate ring entries.
Graph execution more than once between flushes fails recording explicitly
instead of silently retaining only its final call. Capture nodes are removed
and clean graphs rebuilt when capture stops.

Every replay binds separate GPU allocations and keeps immutable inputs and a
pristine state image resident. Before every warmup and sample it restores all
mutable state, clears output/workspace and synchronizes outside CUDA event
timing. It never appends into the resident pool. Timed work is one captured
native attention invocation. Every recorded GQA window remains a timing
representative, so different device positions, masks and slot/page mappings are
never merged by tensor shape alone.

`diagnostics` describes BF16 attention output errors. `state_diagnostics`
separately reports exact differing bytes and total checked bytes over all
physical code/meta planes, residual planes and ring words, including untouched
pool slots. `state_match` summarizes this separately from `numerical_match`.
Exploratory mode retains every timing sample despite either kind of difference.
Native replay validates shape, byte bounds, positions, masks, slot indices and
referenced page ids before executing file-provided metadata.

This first implementation stores full physical state rather than compacting
active pages. At context 65,536 with two pool lanes, one layer's pre/post image
plus inputs/output is about 153 MiB per call, or roughly 15 GiB for 100 rounds.
Graph host buffers count against `capture_pin_bytes`; every record counts
against `capture_disk_bytes`. Budgets are strict and recording does not sample
to fit. The prefill lower-bound query can return zero for an attention-only
scope; actual graph allocation and disk writes enforce the complete budgets.
Narrow selected layers or shorten generation explicitly for diagnostic runs.

### GQA candidates and real forward

The immutable candidate library can export optional `ignis_gqa_candidate_v1`
alongside the unchanged Trellis ABI. It owns the complete C1/T8/HQ invocation:
append/attention launches, split policy, private intermediate layout and final
reduction. The host passes typed tensor/cache pointers and allocates the
workspace size reported by that candidate, outside graph capture. Candidate
workspace does not have to fit the builtin partial buffers. The host keeps it
alive with the library and its graphs. Query/admission/capture failure restores
the previous code, workspace and graphs; successful switching destroys old
graphs before unloading old code. Legacy Trellis-only libraries use builtin
attention.

```sh
python3 scripts/lab.py replay --candidate current --operator layer.3.attention --tokens-per-call 8 --save-as gqa-before
# Edit a supported GQA candidate source or policy.
python3 scripts/lab.py compare --baseline gqa-before --candidate current --operator layer.3.attention --tokens-per-call 8
python3 scripts/lab.py forward --candidate current
```

The same model-local dispatch selects code in isolated replay and real forward
graphs. Unsupported profiles (including C2, T1 and other cache formats) explicitly
use builtin attention. Replay rows identify the actual `implementation` and
`candidate_applied`; reload/forward metadata identifies `extension_loaded` and
`supported_scope`, which describe availability rather than counting runtime
invocations. Candidate GPU kernels are named `ignis_lab_gqa_partial_kernel` and
`ignis_lab_gqa_reduce_kernel` for direct profiling confirmation.

`lab_gqa_candidate_policy.cuh` defaults to the resident split policy.
Setting `IGNIS_LAB_GQA_FORCED_SPLITS` to `42` explores the earlier 42-split
repartition for actual windows greater than 20,160 and at most 70,000 keys.
Both producer and reducer use that policy. It can change BF16 output through
rounding even though the mathematical attention operation is unchanged.
Exploratory replay reports those errors and still times every execution; exact
KV append/state bytes are reported separately. This knob is a candidate for
measurement, not an approved quality/performance improvement.

The builder discovers exact transitive local dependencies with `nvcc -M`,
freezes their bytes, verifies the frozen source against the submitted source
identity, and caches each CUDA object separately. Editing only the GQA policy
reuses all four Trellis objects and relinks a new immutable library. The hot
source whitelist remains explicit; unrelated vendor/core headers, ABI and
runtime changes require a new baseline. Existing artifact/vendor manifests are
not rewritten by candidate building.

Recordings remain tied to the pinned resident engine, model and configuration.
Candidate switching reuses the same immutable corpus, without recapturing it
from candidate activations. Changing the native runtime/ABI still requires
`setup --new-baseline`; recordings from a different engine baseline are not
silently admitted.

## Finite layer segments

Three typed block selectors support recording, isolated replay, candidate
switching and the same dispatch in real forward:

| Selector | Boundary | Initial profile |
| --- | --- | --- |
| `layer.N.mlp_block` | post-attention RMSNorm → gate/up + SwiGLU → down projection + residual | Trellis, eight flattened columns |
| `layer.N.gqa_block` | input RMSNorm → Q/K/V + query gate → Q/K norm + RoPE → paged append/attention → output projection + residual | Trellis, C1/T8 verify, HQ |
| `layer.N.gdn_block` | input RMSNorm → QKV/Z + gating → convolution → GDN record/readout → gated norm → output projection + residual | Trellis, C1/T8 verify; independent of KV format |

The two attention blocks stop before the MLP. The MLP operation is independent
per column and its eight-column profile may also match an eager prefill tail;
actual phase/position coverage is retained in each record. Other profiles use
the original layer path. Segments are explicit opt-in capture scopes. Unfiltered
legacy operator capture keeps its existing meaning. A command mixing block and
legacy operator selectors is rejected, so an intercepted block cannot silently
hide requested inner calls.

```sh
python3 scripts/lab.py replay --candidate current --operator layer.0.mlp_block --tokens-per-call 8 --save-as before-blocks
python3 scripts/lab.py replay --candidate current --operator layer.3.gqa_block --operator layer.0.gdn_block --tokens-per-call 8
python3 scripts/lab.py compare --baseline before-blocks --candidate current --operator layer.0.mlp_block --tokens-per-call 8
```

The optional `ignis_segment_candidate_v1` extension describes exactly these
three kinds, with separate typed calls. It exposes boundary tensors, read-only
model weights and externally consumed state/liveouts. It does not prescribe
normalized inputs, QKV, convolution, gate/up, partials or other internal buffers.
Each candidate owns ordering, fusion, private temporary layout and all launches
inside the block, and supplies its own workspace query. Workspace and code stay
alive with their graph generation; admission/capture failures restore the
previous generation. Legacy libraries without the segment extension retain
original layer execution. A library without the segment extension may still
replace inner Trellis or low-level GQA operators during real forward. Its block
replay is an explicitly labeled builtin control (`candidate_applied=false`),
not a prediction of that legacy library's forward behavior. Block comparisons
of candidate behavior require builtin or segment-aware libraries; use legacy
operator replay/forward to evaluate older libraries.

Recording wraps the original resident layer chain with before/after copies;
it never substitutes the reconstructed standalone segment implementation to
produce its own expected output. Records identify this reference as
`original_layer_chain`. Reconstructed builtin replay and every candidate are
therefore checked against independently captured original outputs and state.

Version-1 records persist a stable layer identity and boundary tensor inventory,
not GPU addresses or copies of model weights. Replay rebinds the pinned model's
weights and RoPE configuration by layer. RoPE positions are independent signed
I32 analytic angles and need not lie in the KV context range; KV positions,
slot selectors and referenced page ids still require their actual storage bounds.
Inputs, masks, positions, full HQ state
and GDN state/record preimages are copied at actual execution. Final residuals
and externally consumed records are captured before later MLP/commit work can
modify them. Replays use isolated allocations and restore preimages and private
workspace outside CUDA event timing. Every distinct recorded block call remains
a representative; different windows and masks are never collapsed by shape.

GDN verify reads persistent convolution and recurrent state without modifying
it. Its `conv_record`, `key_record`, `value_record` and `gate_record` are liveouts
used by the existing acceptance/commit path. The convolution projection writes
all columns; key/value/gate records write only the valid prefix. Replay preserves
and checks the invalid suffix byte-for-byte, including pre-existing NaN bits,
and excludes that undefined suffix from numeric error norms. Per-binding
`outputs` diagnostics separate residual error from BF16 records and FP32 gate
records. Readonly inputs/state and raw mutable cache bytes have separate exact
preservation diagnostics. Numerical differences retain exploratory timing;
changed encoded HQ cache bytes are evidence to investigate rather than an
automatic rejection of an upstream mathematical transformation.

`kernel/src/lab_{mlp,gqa,gdn}_segment.cu` are the editable complete launchers.
The MLP policy `lab_mlp_segment_policy.cuh` defaults to fused SwiGLU. Setting
`IGNIS_LAB_MLP_SPLIT_SWIGLU` to `1` explores separate gate/up projections, stored
BF16 intermediates and an independent SiLU-multiply kernel. This changes both
workspace layout and launch sequence, and introduces an extra rounding boundary;
report its errors and forward results before considering adoption. The GQA
segment composes the candidate's own attention policy in a DSO, while the pinned
builtin segment calls the original resident attention implementation. Existing
candidate-only GQA policy edits do not redefine the builtin recording reference.

Changes to these three launcher files, the owned MLP policy, and the exact
norm/RoPE/projected-conv/gating/recurrent kernel dependencies named in
`scripts/lab_support/common.py` are admitted by the explicit hot-source whitelist
and exact dependency-frozen builder. ABI,
shared dispatch, recording and model-binding changes require a new baseline.
The focused GPU test
`finite_segments_capture_replay_and_preserve_gdn_commit_liveouts` covers all
three boundaries, old HQ history, nonzero slots, masked tails, repeated replay
and preserved commit liveouts. Set `IGNIS_SEGMENT_CANDIDATE` to a default-policy
library for candidate parity and real-forward coverage. Re-run with
`IGNIS_SEGMENT_TEST_KV_FORMAT=bf16` to check the GDN/MLP blocks independently of
HQ caches; that run omits the HQ-only GQA block. The HQ run also checks signed
RoPE values below zero and beyond the KV context range while retaining strict
KV position bounds. These modified inputs are a validator-domain regression,
not model-quality or performance evidence. CUDA-path delivery also
requires the `--all-targets` compilation described in [testing](../../agents/testing.md).

## Full-model feedback

`forward` runs the fixed request from original prepared inputs, through natural
EOS or the original cap, using candidate-generated activations, state and tokens.
It reports engine prefill/decode timing and output differences. A quick run uses
one request and one sample. `forward --full` uses all three requests, configured
warmups/samples and fresh baseline controls. The fixtures retain tools and
history, normalization, artifact template/tokenizer, primary EOS, reasoning-close
behavior, greedy sampling and the 8,192-token cap; generated tools do not execute.

These are engine measurements, excluding HTTP and scheduler work. Do not compare
them directly with historical service TPS. Use the service workflow for service
acceptance and independent references for arithmetic changes.

## Operator profiling

Normal `setup` starts the engine directly, with profiling disabled even if
Nsight Systems is installed. Enable injection explicitly for a new session:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --enable-profiling
python3 scripts/lab.py profile --candidate current --operator layer.0.down --tokens-per-call 8
```

The remote `nsys` executable must be in PATH when enabling profiling. If missing,
setup fails before building or loading the model. Disabled sessions reject
profile before compilation or recording, with the restart command. Missing tool
support, collection/export failure and empty kernel traces are failures.

If Nsight reports insufficient temporary disk space, check the filesystem used
by `TMPDIR`, even when the final result directory has room. Set `TMPDIR` in the
private remote `.remote.env` to an adequately sized writable filesystem. This
is host filesystem space, separate from GPU memory. Stop and restart Lab after
changing the environment so its resident processes inherit the new value.

The flag is a session option, separate from model configuration and baseline,
recording and candidate identities. Stop and run `setup` without the flag to
start a normal session; this preserves the pinned baseline and caches and does
not require `--new-baseline`. A bare `setup` on an existing live session reports
its current mode. Explicit `--enable-profiling` on a live disabled session asks
you to stop first. Setup/evaluation results report `profiling_enabled`.

A profile-capable engine starts under Nsight injection. Collection starts only
for the requested workload; model loading and lazy recording are excluded.
The same resident weights, selected recorded inputs and candidate function table
are used. Warmup is outside collection. No second inference engine or model copy
is started. Injection can itself affect ordinary timing; compare like sessions
and report whether Nsight was active.

Results retain the `.nsys-rep`, SQLite export and machine-readable timeline.
The summary groups kernel names with counts and duration statistics. Per-device
kernel span runs from the first captured kernel start to the last kernel end;
busy time is the union of kernel intervals, gap time is span minus that union,
and overlap excess is summed kernel durations minus the union. Overlap excess
counts concurrency with multiplicity. Gaps can include copies, synchronization
and host work; they do not identify a cause. CUDA event replay samples provide
operator timings separately. Summing kernel durations is never operator wall
time. The captured envelope includes replay setup/restoration kernels as well
as the measured launches; inspect the timeline and counts when interpreting it.

### Full-forward profiling

Use `profile --forward --candidate current` for the fixed request through the
complete model. It requires a profiling-enabled session and accepts `--warmup`
and `--samples`, defaulting to zero and one respectively. Operator/token selectors
and `--full` do not apply. The request uses its own activations, recurrent/KV
state and generated tokens; profiling does not record or consume an operator
corpus. It does not run an extra baseline request. When an existing builtin
reference is available, results include output differences; otherwise
`reference_available` is false and differences are unavailable. A missing
reference does not mean outputs match.

Forward capture includes one preparation step: recapture the selected
candidate's graphs under explicit setup annotations. This makes graph source
layer/operator ranges available to Nsight before actual samples run. It retains
the same resident model and PID. Preparation kernels remain in the raw trace
as setup work and are excluded from request summaries. Instrumented benchmark
prefill/decode timing is reported separately from recapture and collection costs.

GPU attribution uses explicit versioned Ignis NVTX annotations. Eager kernels
link to their issuing CUDA runtime API and enclosing sample/phase/layer/operator
ranges. Graph kernels link through recorded graph-node creation/clone ancestry
to source layer/operator ranges, and through their actual launch to the current
sample, phase and round. Correlations are qualified by process; context is used
where available. Ambiguous/missing links stay unclassified. Kernel names and
execution order do not establish layer identity.

In the saved `profile-timeline.json` artifact, `attribution.instances` contains
phase, layer and operator occurrences with
sample, phase, decode round/position, stage, layer index/type and operator when
known. Draft layers use their own type; output-head work has an explicit
nonlayer bin. Other unannotated work is visible as unclassified. A repeated
prefill chunk or decode graph launch creates a separate layer occurrence.
`aggregates` reports counts and sum/mean/max of those occurrence metrics; it
does not take one layer's first kernel in the first round and last kernel in
the final round as a single layer latency. Samples remain separate.

For each occurrence, `span_us` is first kernel start to last kernel end,
`kernel_total_us` sums durations, `busy_us` unions intervals, `gap_us` is span
minus busy time, and `overlap_excess_us` is total minus busy time. These are GPU
kernel intervals, rather than CPU NVTX range durations. Phase, layer and operator
levels overlap and must not be added together. Sum of occurrence spans and sum
of kernel durations are not request wall time.

`coverage` counts request kernels attributed at each level.
`capture_coverage` separately counts request, setup and unplaced kernels across
the whole capture. Kernels without an explicit request-sample link remain
unplaced rather than being guessed into a sample. Top-level forward kernel
summaries cover the explicitly linked request subset; check these counts before
claiming complete coverage. The machine-readable timeline retains every captured
kernel and its role/link status. Command results return aggregates, coverage and
instance/timeline counts; the full instance and per-kernel arrays stay in the
artifact so long requests do not transfer the entire trace over the control
channel. CLI lists are capped; the artifact retains all occurrences and aggregates.

Nsight Compute memory/occupancy counters are not collected by this command.
Profiler timings include instrumentation overhead and are diagnostic evidence;
use replay/compare for latency measurements.

## Source, baseline and recovery

Hot loading supports `trellis_linear.cu`, `trellis_linear_decode.cu`,
`trellis_linear_prefill.cu`, `trellis_linear_mlp_t8.cu`,
`trellis_linear_internal.cuh`, `trellis_decode.cuh` and
`trellis_decode_helpers.cuh` in `kernel/src/`. GQA hot edits also admit
`kernel/src/lab_gqa_candidate.cu`, `kernel/src/lab_gqa_candidate_policy.cuh`,
`kernel/include/ignis_gqa_hq_reduce.cuh`, and these exact files under
`kernel/vendor/src/ops/kernel/`: `gqa_attention_decode.cuh`,
`gqa_attention_decode_bf16.cuh`, `gqa_attention_decode_hq.cuh`, `hq_codec.cuh`.
Other runtime/build/ABI changes require a new baseline:

```sh
python3 scripts/lab.py stop
python3 scripts/lab.py setup --new-baseline
```

Repeated setup for a matching live session reports its state and preserves its
baseline. Recovery also preserves the pinned baseline. To change configuration,
stop first and provide the new `--config` with `--new-baseline`.

Ordinary commands restore builtin code and clean graphs. Numerical differences
are recorded. Sync/build/admission failures do not measure stale code. CUDA
failure, timeout, invalid engine replies or failed restoration end the session.
`timeout_s` bounds an accepted evaluation and `setup_timeout_s` bounds setup.
Lost replies are not retried; inspect saved results and logs.

A command returning leaves the engine resident. Concurrent evaluations are
refused. `stop` uses recorded ownership and pinned control code, without sync or
rebuild, and confirms the GPU process has disappeared. It can interrupt setup or
evaluation. Results, recordings and caches survive stopping.

Local receipts and returned results live under `.scratch/lab/`. Remote session
directories retain settings, inputs, engine logs, source/build identity and
operation results. Reports separate client preflight/sync/round-trip from owner
build/reload/recording/measurement. Keep the linked full results and do not delete
control receipts while the session runs.

## Development checks

CPU suites check orchestration and exported-timeline arithmetic. The opt-in real
GPU regression runs through the public commands:

```sh
IGNIS_LAB_E2E_CONFIG="$PWD/.scratch/lab-settings.json" \
  python3 -m unittest discover -s tests -p test_lab_gpu.py
```

The regression edits supported CUDA sources, restores them in `finally` and
stops its session. Run without another writer or GPU workload. A short diagnostic
configuration is suitable for functionality, not performance evidence. Before
launching any GPU check, run `make gpu-status` through the remote entry point.

Native `lab_capture_gpu::hq_attention_recording_restores_pages_residual_ring_and_masks`
additionally covers C1 HQ attention in a two-lane pool, nonzero slots, page
boundaries, histories beyond the 544-row residual window, masked tails, repeated
isolated replay, corrupt state/metadata and unchanged forward outputs. It only
requires `IGNIS_MODEL_ARTIFACT`; it does not require fault libraries.
Optionally set `IGNIS_GQA_CANDIDATE` to a default-policy candidate and
`IGNIS_GQA_FAULT_DIR` to libraries `gqa-fault-{1,2,3}.so` compiled from
`tests/fixtures/gqa_candidate_fault.cu` with corresponding `GQA_TEST_MODE`.
This additionally checks identical-corpus candidate replay/forward and rollback
after bad GQA ABI, invalid workspace query and graph-capture failure.

Native `lab_capture_gpu` and `trellis_reload_gpu` ignored tests cover
recording/clean-graph parity, padded output, repeated ADD restoration, invalid
views, plugin use and rollback. Use the artifact and fault libraries documented
in their headers. See [testing](../../agents/testing.md) for required checks.
