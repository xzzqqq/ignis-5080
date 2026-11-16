# Trellis static decode and selective down reduction

- Kind: experiment
- Status: current
- Observed: 2026-11-16
- Last verified: 2026-11-16
- Scope: kernel / RTX 5080 Trellis decode scheduling and T=8 down reduction
- Related: [T=8 MLP scheduling and PDL](2026-11-11-rtx5080-t8-mlp-scheduling.md), [Kernel Lab](../user/contributing/kernel-lab.md)
- Superseded by: none

## Question

Does removing persistent scheduling or separating the light reduction tail improve Trellis decode on the RTX 5080 without changing numerical grouping or the correctness gate?

## Evidence

All measurements use one RTX 5080 16 GB and Qwen3.8-27B Trellis. Kernel Lab uses the frozen Astropy request, context 65,536, prefill chunk 1,024, two configured lanes, hq-e8-2b KV and DFlash2. The recorded selection covers GDN layers 0/57, GQA layer 3 and the head. Its T=8 view checks 3,179 real-input records exactly and times 17 fixed representatives with one warmup and three samples. These timings exclude HTTP and scheduler overhead.

The uncapped static CTA mapping retains eight warps, one 16-output tile per warp, the existing K splits and accumulation order. Replay passes all selected records. One quick forward measured decode at 4,171.713 ms against the original dynamic setup baseline's 4,509.050 ms, a 7.48% reduction. It generated the same 913 tokens and passed the frozen logit, termination and round-trajectory checks. This was one comparison against the earlier setup baseline, without fresh bracketing controls; it is limited evidence rather than a stable whole-model performance estimate.

A separate nine-evaluation ABC/CAB/BCA comparison used A = static fused, B = generic separate reduction and C = dedicated eight-warp down with four K splits and separate reduction. Each arm occupied each order position once. All nine exact replays passed. The table gives medians of three evaluation medians, and C versus A reductions paired within each cycle.

| T=8 down | A fused, µs | B generic separate, µs | C dedicated separate, µs | C latency reduction, three pairs |
|---|---:|---:|---:|---|
| 2-bit, layer 0 | 38.400 | 39.616 | 36.320 | 5.42%, 6.43%, 5.57% |
| 3-bit, layer 3 | 44.320 | 43.552 | 41.344 | 4.30%, 4.33%, 6.71% |
| 4-bit, layer 57 | 58.816 | 58.048 | 58.112 | 1.14%, 1.36%, -0.33% |

An A/C/C/A quick forward comparison measured A at 4,175.849/4,174.778 ms and C at 4,160.551/4,159.740 ms: a 0.363% median reduction relative to the static control. All four runs retained the same 913-token trajectory. C in this historical comparison included the dedicated 4-bit arm; it is evidence about that candidate, not a separate measurement of the final 2/3-bit-only combination. The 7.48% and 0.363% figures have different controls and must not be added as a measured total gain.

Broader generic separation passed exact replay, but an A/B/B/A/A/B repeat did not establish consistent QKV/output/down improvements. For example, paired 2-bit down reductions were 1.21%, -1.56%, 1.96%; 3-bit was 0.44%, 0.66%, -2.50%. No layer whitelist was selected from those noisy results.

Synthetic FP64 probes used independently decoded weights, unchanged bounds and padding checks on every graph execution, including warmups. The dedicated eight-warp/four-split separate arm agreed bitwise with the original fused arm in its tested cases. Four-warp down candidates with split counts 5/6/7/8/12/16 passed those synthetic numerical bounds but failed the unchanged real-input exact gate. They have no accepted real-input performance or quality result. The split8 reducer warp sweep also failed that gate. Trace diagnostics put the reduction envelope near 0.7–1.0 µs and the final-partial-to-reducer handoff near 0.5 µs; instrumentation changed timing and register allocation, so these are diagnostic envelopes rather than achievable overlap gains.

The final combination, candidate `a7d2044a6f34ce2742e6e9498ddfe40daab207895db08aec70f8f3014d4e3615`, passed all 3,179 selected real-input records and 17 representatives in replay operation `57c508ec8e8245498930ff97cd07f1ef`. A single final quick forward (`4158edddeffe4cf7900c7f0ddb06d7ef`) passed with the same 913-token trajectory and 4,161.313 ms decode time; it had no fresh paired static-control run. The compiled final partial/reducer kernels use 90/47 registers without spills, versus 112/47 in the earlier untraced dedicated probe. The historical percentages above therefore describe their measured candidates and cannot be assigned to this final binary.

Final validation passed `cargo test --workspace --target aarch64-apple-darwin`, `make check CUDA=1` (workspace CUDA/all-targets compilation), and the complete native `ignis_kernel_trellis_linear_test` (9.04 s). The native test includes the new real-shape 2/3-bit T=8 ADD FP64 checks, padded same-split fused controls, five shared-workspace graph replays and bit-width/token/epilogue/SM-cap fallbacks. These checks retain the existing tolerance and exact comparisons. The Lab session was stopped afterward and released the GPU; recordings and receipts were retained.

Reproduction and raw evidence remain in the ignored scratchpad:

- [Static mapping report](../../.scratch/static-projections-2026-11-15/report.md), including baseline/candidate identities and Lab result IDs.
- [Real-input comparison](../../.scratch/down-real-compare-2026-11-16/report.md), [generic repeat](../../.scratch/trellis-select-2026-11-16/report.md), with raw Lab results and paired timings.
- [Split/reducer sweep](../../.scratch/down-reduce-sweep-2026-11-16/report.md), [near-six and trace probes](../../.scratch/down-near6-2026-11-16/report.md), including probes, source snapshots, raw timings and numerical results.
- [Final replay](../../.scratch/trellis-final-2026-11-16/replay.log), [single quick forward](../../.scratch/trellis-final-2026-11-16/forward.log), [native test](../../.scratch/trellis-final-2026-11-16/native-test.log), [CUDA compile check](../../.scratch/trellis-final-2026-11-16/cuda-check.log) and [workspace CPU suite](../../.scratch/trellis-separate-2026-11-16/final-cargo.log). The [Kernel Lab guide](../user/contributing/kernel-lab.md) describes `replay` and `forward`; the final source can be checked with `ignis_kernel_trellis_linear_test` before repeating model measurements.

## Finding

Retain the uncapped generic static CTA mapping and the existing static T=8 gate/up path. Adopt dedicated down partial/reduce kernels only for uncapped 84-SM execution, T=8, 17,408 inputs, 5,120 outputs, one weight, residual ADD, existing split count exactly four, and 2/3-bit weights. The dedicated partial kernel still uses eight warps and the same K grouping. Leave 4-bit down and all other calls on their existing fused routes; capped calls retain persistent scheduling.

Do not adopt broad generic separation, down four-warp geometry, any changed split count, the reducer warp sweep or fine-grained reduction overlap from these experiments. These exclusions reflect inconsistent gains or failed exact checks, rather than a relaxed numerical contract. Dedicated specialization, compiler scheduling and register allocation covary, so the observed improvement cannot be attributed solely to removing arrival atomics.

## Implications

Keep rotation-to-partial PDL and the activation dependency wait. Ordinary same-stream ordering publishes all partials before the reducer and prevents a subsequent call from reusing the shared activation/partial workspace too early. The existing workspace already reserves the four splits; no forced split policy or workspace expansion is necessary. The specialized path does not touch arrival counters. The fused capped path retains its reset protocol.

Preserve the scale multiplication's round boundary before residual ADD with `__fmul_rn`. Earlier contraction into an FFMA changed one BF16 output in an exact comparison; changing the gate or tolerance would hide that arithmetic change. The real-shape regression checks FP64, identical-split capped fused output, padded output and shared-workspace graph replay across optimized and fallback calls.

## Limits and unknowns

Selected T=8 records do not cover every layer, token width or hardware platform. Timing runs do not lock clocks or provide confidence intervals. Synthetic numerical passes do not supersede real-input exact failures. The historical quick forwards and final replay do not establish the final combination's full three-request or service-level gain. Full GPU profile, capacity/canary, Vision, tools and service lifecycle acceptance are separate evidence requirements.

## Follow-ups

Use fresh baseline controls and the fixed coding-agent benchmark before making a whole-model or service performance claim. Investigate real-weight numerical changes independently before reconsidering changed K splits; keep the existing exact gate and numerical bounds fixed.
