# NVFP4 KV storage with BF16 attention on RTX 5080

- Kind: experiment
- Status: current
- Observed: 2026-11-20
- Last verified: 2026-11-20
- Scope: kernel / Qwen3.8-27B NVFP4 KV storage and BF16 attention on RTX 5080
- Related: [prior HQ block integration](2026-11-19-gqa-uniform-fp32.md), [adapted kernels](../adr/0051-kernel-code-is-ours-adapted-or-vendored.md)
- Superseded by: none

## Question

Can NVFP4 storage reduce attention latency while retaining a BF16 mathematical path, and which parts of the temporary kernel evidence transfer to a formal paged-cache implementation?

## Evidence

### Storage and mathematical paths

The formal implementation uses adjacent E2M1 nibbles with one positive E4M3 scale per 16 coordinates. The restoration global is fixed at **1**, with no per-request calibration or mutable global state. Each GQA layer stores four paged U8 planes: K codes, K scales, V codes, V scales. At D256 their leading extents are 128/16/128/16 bytes. The 27B's 16 GQA layers and 4 KV heads therefore cost 18,432 bytes per token: 4.5 bits per coordinate including block scales. HQ remains 9,216 bytes/token; BF16 remains 65,536. NVFP4 does not provide HQ's 2-bit storage density.

The selected decoder derives from [FlashInfer v0.7.2](https://github.com/flashinfer-ai/flashinfer/tree/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94), commit `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`. The independent Apache-2.0 subtree, original hashes, license and local patch are in `kernel/third_party/flashinfer_xqa/`; attribution is in `kernel/NOTICE`.

FP4 values and E4M3 scales are converted to BF16 before multiplication. Both QK and PV use BF16-input Tensor Core MMA with FP32 accumulators; probabilities and multiblock partial output tiles pass through BF16. The final attention value is BF16 before the FP32 sigmoid/product and final BF16 gate rounding. This is **NVFP4 storage with BF16 computation**, not two native-FP4 GEMMs. Official default fast-math compilation is retained only for the XQA translation unit and recorded separately from strict-math controls.

### Selected M48/K32/V32 decoder

C1/T8 has 48 useful query/head rows per KV head. M48 handles those rows in one CTA group, avoiding the two 32-row groups' padding. K32 is the logical K head-part size before unpacking; V32 is the V staging sequence tile. The K-scale addressing uses aligned 4-byte pairs with explicit low/high-half selection. A 512-case CPU ownership audit checks that resource change's scale addresses.

The graph captures a **capacity** of 21 splits and 4 KV heads, not the instantaneous useful split count. Device metadata selects `activeS=min(21,ceil(actual_window/256))`; inactive CTAs exit before collective work. Runtime sequence length, masks and slot/page selectors remain device inputs, so graph replay can change the visible window and valid prefix. A graph prepared for a long envelope must not force short requests to execute 21 useful splits or size scratch for a smaller unadvertised capacity.

The frozen K32/V32 candidate uses 242 registers, 88,192 bytes dynamic shared memory, zero compiler spill and zero stack, versus the M48 default control's 248 registers, 95,872 bytes shared and 64-byte stack. Both remain one CTA per SM by their resource limits. Static resource differences alone are not elapsed-time attribution.

### Local matched kernel measurements

The following are direct same-input comparisons with the corresponding frozen M48 default control. Values are separate latency medians in microseconds, not ratios multiplied across optimization steps.

| Input and scope | M48 control | M48/K32/V32 |
|---|---:|---:|
| Actual W=19,962, full chain, warm |69.696|65.856|
| Actual W=19,962, full chain, eviction-conditioned |94.272|89.600|
| Actual W=19,962, attention-only, warm |66.752|62.720|
| Actual W=19,962, attention-only, eviction-conditioned |90.880|82.880|
| Synthetic W=65,536, full chain, warm |190.144|177.952|
| Synthetic W=65,536, full chain, eviction-conditioned |212.896|200.832|

The actual case is one layer 3 C1/T8 call: 192 unique query rows, 49,152 scalar coordinates per endpoint. Pre-gate and gated outputs are two views of those same rows, not 384 independent rows. It has 9 alternating/rotated samples per arm and condition. The synthetic 64K case has **3** samples per arm/condition, according to its sample arrays. It repeats original BF16 historical KV with actual new eight rows/Q/gate and consistent causal positions; it is not a real 64K model trajectory or quality test.

Full-chain timing includes quantizing the eight appended rows, a 16-byte counter clear, attention/merge and gate. Attention-only starts from a real postappend cache and still includes counter clear/gate. Prefix preparation, restoration, a warmup or 256 MiB read/write eviction are outside CUDA event timing. Eviction-conditioned measurements are not certified all-cache-cold measurements. Full and attention-only medians cannot be subtracted to infer encoder cost.

The candidate and its corresponding M48 control have matching observed pre-gate/gated output bits on these cases, with repeat/full-versus-cached/counter diagnostics passing. Evidence is retained under `.scratch/kv-format-2026-11-19/m48-k32-v32/`: `results-real/report.json`, `results-65536/report.json`, `quality.json`, `build-receipt.json`, `source-hashes.json` and `scale-address-audit.json`.

A later source review found that the earlier private XQA workspace query did not reproduce the kernel segmenter's type-size alignment. At 21 splits it reported 2,107,392 bytes instead of 2,113,536, a 6,144-byte shortfall; at 18 splits the shortfall is 12,288 bytes. The formal public 64 MiB arena masked this query defect, while older scratch wrappers allocated exactly the queried size. Therefore the earlier scratch results are historical optimization signals, **not a memory-safety validation**. The corrected query matches the kernel's three type-aligned segment allocations and also reserves 24,575 bytes for worst-case base padding; run aligns the raw base to the 24,576-byte scratch-block size by integer division, not a power-of-two bitmask. Thus adding only the 6,144-byte relative shortfall would not be a complete fix. The corrected operator regression passed in 2.068 seconds, covering exact queried allocation sizes with prefix/tail redzones at maximum-key envelopes 20,000 and 32,769, workspace base offsets 0 and 256, same-graph active splits 8 and 21, and valid prefixes 8, 7 and 0. Math checks passed and semaphores returned to zero; the same target also passed the original operator, band and COW cases. This validates the corrected private workspace contract, not older exact-query scratch allocations. The earlier S18 variations are confined to the final KV head and high dimensions, spatially consistent with underallocation, but a repaired S18 rerun has not established every earlier difference's cause.

### Calibration boundary

Those kernel screens used the earlier **historical-prefix-calibrated** K/V globals. They must not be relabeled as formal fixed global 1 measurements. Earlier `screen-first` BF16 data affected by a cross-stream initialization race are excluded; subsequent driver checks verify the complete BF16 prefix and appended history.

A separate CPU scan over the original pre-HQ BF16 W=19,962 call compares fixed globals 1, 1/16 and 1/128. All three yield the same decoded-BF16 attention on that input: pre-gate relative L2 approximately 0.04079885 and continuously gated relative L2 approximately 0.04339147 against the original-BF16 FP64 target. No scale clipping or nonzero-group scale underflow is observed; global 1 has one subnormal V scale. Global 1's maximum finite group amplitude 2688 gives more range than the two smaller globals. Evidence: `.scratch/nvfp4-integration-2026-11-20/accuracy/global-policy-scan.json`.

This target includes codec loss relative to original BF16 K/V. The prior HQ finding's FP32-partial error analysis used a fixed decoded-HQ mathematical target and isolated kernel rounding. Their error magnitudes do not describe the same question.

### Bounded fixed-history model evidence

Three supplied histories contain 19,954 Astro, 22,420 Django and 16,717 Sympy tokens. Each format starts from a fresh sequence, prefills the head in 1,024-token chunks, then evaluates the final 1,024-token tail with the same chunking. The Sympy tail crosses the 16K boundary. These runs provide 3,072 logit distributions and 3,069 known next-token targets. All arms use the same Trellis SC30 artifact. The quality configuration names `exception_weights=fp8`, whereas the server default is `bf16`; this artifact has no BF16 exception linear to re-encode, so the flag is a physical no-op. The existing `fp8_exceptions_never_reencode_a_trellis_matrix` inventory test keeps every Trellis entry as `DeviceEncoding::Stored`, and old/default and new service plans both report 10,907,958,020 weight bytes. Thus these runs do not establish a different FP8-weight baseline. “BF16 reference” means BF16 **KV storage**, not a BF16-weight checkpoint. The 32-column output-head readout blocks do not change the backbone's 1,024-token attention calls.

| KV format | Mean KL from BF16 KV, nats | BF16 argmax agreement | Known next tokens correct | Next-token NLL, nats |
|---|---:|---:|---:|---:|
| BF16 | 0 | 100% | 1,147 / 3,069 | 10.0891 |
| HQ | 4.5943 | 58.9193% | 1,177 / 3,069 | 9.6930 |
| NVFP4, global 1 | 2.4565 | 69.5638% | 1,172 / 3,069 | 9.9564 |

NVFP4 improves KL and BF16-argmax agreement over HQ in each window. Its observed next-token NLL is **0.2634 nat higher** than HQ, with five fewer correct known next tokens overall. Of those targets, 2,471 are inside supplied tool responses; this is not assistant-gold completion quality or a task-solving score. The full entropy results do not support a general claim that HQ merely softens the distribution. A separate 32-token tail processed as four eager C1/T8 calls gives KL 3.4404 versus HQ 4.2787, BF16-argmax agreement 68.75% versus 62.5%, and 17 / 31 correct known next tokens for all three formats. This exercises the formal M48 path but is not verify-graph quality coverage.

These bounded results establish improved fidelity to BF16 KV on the supplied histories; they do not establish service-level quality or justify an unconditional adoption decision. The service failure below remains material despite these distribution metrics. The format-route comparison includes each route's prefill arithmetic and encoding, rather than isolating its codebook alone. The [sanitized quality summary](artifacts/2026-11-20-nvfp4/quality-summary.json) records per-window values, the adverse NLL direction, role counts, exact source/binary/scorer identities and row-layout checks without raw tokens or logits.

### Bounded fallback and service integration

In R2, actual widths T1 through T8 reuse the same nominal-T8 M48 consumer per lane, including C2; device actual-Q and validity metadata mask the unused rows. Widths above T8 use decoded BF16 bands, not an entire-context BF16 shadow cache. The band cap is 16,384 keys: K+V scratch occupies 64 MiB, independent of full context length and reused across lanes. Unnormalized FP32 numerator/max/sum carry persists across bands; normalization/BF16 output rounding occurs only at the last band. At a 1,024-token prefill chunk, the carry adds 24 MiB plus 192 KiB of scalar statistics. Query positions, valid prefixes and causal limits come from actual call inputs.

The first formal scope is Qwen3.8-27B, 24Q/4KV/D256, with a 524,288-key envelope bound. Unsupported families explicitly reject NVFP4 rather than treating any U8 cache as HQ. BF16/HQ format codes 0/1 remain; NVFP4 uses 2, and snapshots/COW retain all four code/scale planes with no calibration state outside the format identity.

The current implementation sets the RTX 5080 Qwen3.8-27B profile to NVFP4 with 128 Ki input plus an 8 Ki output allowance (`max_context=139264`) and 8 Ki adaptation spare. Context limits count input and output together. An explicit NVFP4 service loaded that full context without adaptation down, with two lanes, a 1,024-token prefill chunk, DFlash2/7 and 4,096 host vision tokens: allocated memory was 15,348,177,196 bytes and free memory 1,497,235,456 bytes, with the existing 1,200 MiB safety headroom unchanged. This establishes the loaded memory plan, not the separate 128 Ki-input capacity test. The global format default remains HQ for profile-free and unsupported-family configurations; explicit HQ and BF16 remain available.

Lab settings accept NVFP4 for builtin full-model forward and forward profiling.
The existing HQ attention/GQA-block replay rejects G16 NVFP4 cache views;
it does not acquire an NVFP4 native hot-reload ABI. See the
[Lab scope](../user/contributing/kernel-lab.md) for the supported commands.

### CPU and documentation checks

The final local `cargo test --locked --workspace` run passed: **3,079 passed, 0 failed and 6 ignored** across 253 test targets, with `CARGO_INCREMENTAL=0`. Generated configuration/profile documentation passed both regeneration and a subsequent check without blessing. The first workspace attempt ran below the artifact packer's existing 2 GiB disk-safety margin; after clearing this project's rebuildable Cargo incremental cache, a rerun exposed one stale profile expectation in a server test. That test now checks the selected 139,264-token context and 8,192 spare-token profile, and the complete rerun passed. Neither the disk margin nor a production behavior was weakened to obtain the result. A subsequent targeted `ignis-vendor` repository-manifest test passed all three cases, including the independent XQA copy/patch hashes without a machine-local reference checkout. These are CPU results, separate from cumulative CUDA/service acceptance. The [acceptance summary](artifacts/2026-11-20-nvfp4/acceptance-summary.json) records counts, commands and evidence hashes; unfinished checks remain explicitly pending.

### Cumulative CUDA acceptance

For the R1 integration, CUDA `--all-targets` checking passed in 100.2 seconds. The initial official runner returned **1**, not a clean pass: CTest passed 131 / 132 cases, with the remaining old host fixture treating the newly assigned NVFP4 ABI value 2 as unknown; the final reload stage was also interrupted during an NFS debug build. The host fixture now uses 3 for unknown format and supported geometry for known NVFP4, and its targeted rerun passed 1 / 1 in 0.0014 seconds. All four Trellis GPU suites passed with zero ignored tests. Reload was rerun in release mode and passed 1 / 1 with zero ignored tests (23.88 seconds actual GPU work, 93.67 seconds total); its cold/reload page counts, 2,486 / 2,479, satisfy the existing test. These results complete cumulative coverage after the test-only and build-environment corrections; they are not a claim that the original runner passed in one attempt.

### Service R1: mixed throughput and a Django failure

The R1 service benchmark uses loop guard **off**, as required by its fixed protocol. Its `nvfp4-default` label identifies the KV profile, not every product default: the product's default loop-guard mode is Rewind. The matched run's per-task own-decode throughput medians are:

| Task | HQ, tokens/s | NVFP4 R1, tokens/s |
|---|---:|---:|
| Astro | 238.779 | 284.676 |
| Django | 223.148 | 329.096 |
| Sympy | 264.853 | 233.975 |

The fixed [v1 reference](../user/contributing/coding-agent-benchmark.md) remains `rtx5080-coding-agent-v1-2026-11-03`; the same-build HQ control above does not replace it. Using the exact recorded medians gives:

| Task | v1 tokens/s | NVFP4 R1 / v1 rate | Output tokens, v1 → NVFP4 R1 |
|---|---:|---:|---:|
| Astro | 207.657750 | 1.37089 (+37.09%) | 876 → 770, both tool calls |
| Django | 184.816402 | 1.78067 (+78.07%), throughput only | 954 tool call → 8,192 length cap, no tool |
| Sympy | 189.941801 | 1.23183 (+23.18%) | 1,268 → 1,245, both tool calls |

The [exact-rate comparison](artifacts/2026-11-20-nvfp4/v1-rate-comparison.json) uses full-precision service medians and the fixed reference values. V1 used HQ with context 262,144; this run uses NVFP4 with context 139,264 and different generated histories. These retained numerical rate ratios are not controlled algorithmic gains, and the Django value is not a useful-completion benefit. The matched same-build HQ comparison remains the main control.

These are mixed results, not an overall speedup. In particular, all three NVFP4 Django samples reach the 8,192-token limit while repeating the same reasoning block and producing no content or tool call. The HQ comparison completes with an effective tool call at 1,760 tokens; a separate BF16 diagnostic does so at 792 tokens. The higher NVFP4 Django token rate therefore does **not** represent a faster useful completion. Sympy is slower and draft acceptance falls from approximately 62.9% under HQ to 51.9% under NVFP4. A no-speculation diagnostic capped at 2,048 tokens has not reached the observed stable-loop onset near token 2,537, so it cannot establish the absence of that failure.

A matched first-frame projection diagnostic across all 16 GQA layers found zero bit differences in Q, gate, K and V between the compared projection routes, excluding that specific wiring suspicion. It does not establish that quantization or the later trajectory is harmless. The independent R2 same-request product-default Rewind check also reaches 8,192 tokens with the identical canonical message and no tool call, in 35.210 seconds total. No selected loop-guard event was observed and no actual rewind was demonstrated. The unchanged detector's CPU replay also remains above its trigger thresholds. This run does not establish NVFP4 rollback coverage and does not replace or erase the loop-guard-off benchmark failure.

R2 addresses the small-token-width performance fallback by routing each lane at T ≤ 8 through the same M48 consumer. Its targeted strong-causal/redzone regression passed in 2.118 seconds. This routing change does not alter the mathematical kernel used by R1's principal C1/T8 benchmark. R1 results remain tied to their original binary. A short R2 T1 signal is 59.615 tokens/s at a 128-token cap, compared with the historical slow path's 21.129 tokens/s at a 2,048-token cap; these unequal caps and trajectories do not support a strict speedup ratio.

### R2 post checks and remaining quality limits

The capacity fixture passed with **131,072 input tokens actually computed during prefill**, zero cached tokens and one prefill pass: 97.9916 seconds prefill, followed by a nonempty one-token response. This establishes the requested input-capacity operation within the 139,264-token total envelope; it is not a long-context answer-quality or multi-sequence-capacity result. The existing canary's four cases passed its bounded sanity and determinism checks. Consumer cancellation was observed after eight tokens and a following request generated a nonempty response.

The Vision request completed the image/encoding/generation smoke path, but did **not** pass semantic review: the input is a red horizontal rectangle, while the answer calls it a perfect square. Color/background match and the 96-token response is length-truncated. No matched HQ semantic comparison was run, so this error is not attributed specifically to NVFP4. Neither this smoke nor canary criteria resolve the Django failure.

A final targeted scale-layout test passed in 2.151 seconds. It covers all 16 E4M3 scale positions, odd-half selection and HND addressing, using the same nonuniform Q for XQA and the band path; the checked maximum absolute difference was 0.00311851 with zero wrong cases. This is operator/layout validation, not a model-quality pass. The [sanitized service summary](artifacts/2026-11-20-nvfp4/service-summary.json) retains exact R1/R2 binary/source identities, request/response hashes, capacity/canary results and the failures above without raw private traces.

### Final evidence identities and Lab state

R1 service measurements retain runtime-source digest `94486531733c99a00f02dc49d68d7fdb63b1cf8d1ec53c460ef6624eb107f145` and server digest `965cd7f801a9e46ad053904dd2998ad4bffb4e11555207bd7628b595f7620260`. R2's small-width route uses source digest `a59dd6176e8124c2b7853529d7418533ce2854269d484de2d670f9bb7993adcb` and server digest `f225e193a235164fbaa51e33d06eb2a226b45c9fffed4349ad338509e7f7ce4d`. The acceptance summary hashes the small stage, targeted and identity receipts. The final SF test's source/executable digests apply only to that test; they are not backfilled onto earlier targeted results that did not record their own binary hash.

The saved final Lab receipt reports ready, no active operation, profiling enabled and NVFP4 at context 65,536 / prefill 1,024 / two lanes / DFlash2, using the R2 source digest and no diagnostic source. This is the saved restoration result, not a new liveness or GPU probe by the document author.

## Finding

Observed: the formal fixed-global-1 storage, graph-safe M48 route, bounded band fallback, workspace boundary checks and cumulative integration checks are validated within the stated scopes. The 128 Ki-input capacity and bounded canary checks passed. The earlier local kernel measurements suggest performance opportunities, but the R1 service behavior is mixed: Django fails to produce a useful tool turn under both loop-guard-off and product-default Rewind checks, and Sympy throughput falls. **Implementation validated; service behavior mixed; default adoption pending.**

Inference: reduced padding and a smaller staging footprint are useful resource opportunities on SM120, but the timing does not isolate one cause. Neither those resources nor the local kernel measurements establish full-service improvement.

## Implications

Keep graph capacity separate from active device work, preserve actual masks/page selectors, and budget both decoded bands and FP32 carry. Compare formats against original pre-HQ inputs and also against their own decoded-storage targets. Keep the existing-library consumer's precision and fast-math choices explicit.

## Limits and unknowns

Fixed global 1 has bounded model evidence on the supplied histories above; task-solving quality beyond those histories is not established. Cumulative CUDA coverage and the stated R2 targeted/capacity/canary checks are complete. The Django service failure remains unresolved; Vision is only a path smoke with a wrong shape description and no HQ semantic control. Default adoption remains pending a user decision; no overall service speedup or comprehensive model-quality pass is claimed here. Earlier prefix-calibrated and synthetic measurements cannot close those gaps. Raw model data, corpora and profiler captures remain outside this finding.

## Follow-ups

The authoritative current acceptance notes and raw evidence remain under `.scratch/nvfp4-integration-2026-11-20/`. The sanitized summaries distinguish formal R1/R2 identities from the historical screens above. Preserve the unresolved service results when deciding whether to adopt the new default. Any later global-scale policy change must preserve or version the cache/snapshot interpretation; the experiment's calibrated globals are not a production option.
