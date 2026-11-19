# C1/T8 HQ GQA: balanced partitions, FP32 partials, and block integration

- Kind: experiment
- Status: current
- Observed: 2026-11-19
- Last verified: 2026-11-19
- Scope: kernel / RTX 5080 C1/T8 Trellis HQ GQA attention and block integration
- Related: [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md), [exact GQA replay](2026-11-18-gqa-exact-replay-and-integer-cache.md)
- Superseded by: none

## Question

Can balanced partitioning and changed intermediate rounding improve C1/T8 HQ GQA on the RTX5080, and which exact block fusions retain the resulting mathematical behavior? What do independent reference checks establish before final service acceptance?

The final accumulated **snapshot B: private Q S3 plus exact shared-L/four-split reducer prefetch** measures **1.389599x** median paired speedup against initial math on222 identical captured inputs, with B faster222/222 points: about28.04% lower paired block latency. Frozen build, official cumulative GPU runner, B service/capacity/canary checks and bounded Vision smoke pass. [The final B service table](#snapshot-b-final-service-and-vision-validation) reports actual own-trajectory behavior, which is not a same-work service speedup. The earlier **snapshot A: native Q S1/cooperative reducer** retains its separate1.390969x/714-point figures and service/profile results; those values are not relabeled or multiplied to produce B.

The initial-math control is the frozen round-start **dirty** algorithm, which already contained the earlier U2 and owned D256 reducer work. This PR also includes those prerequisite changes. The measured 1.389599x is therefore not a comparison against clean `main`; no clean-main total speedup is claimed.

The selected implementation combines balanced short/long split scheduling, FP32 partials, one IFWHT after merge, a cooperative reducer, Q FWHT inside norm/RoPE, inactive-split write elimination, four-lane fast HQ decode with eight-lane general fallback, fixed-S K/V projection pairing, interleaved Q/gate consumption, append-side FWHT reuse, private Q S3 partitioning and exact shared-L/four-split reducer prefetch. It remains bounded to the native C1/T8 Trellis/HQ verify path. Other shapes, eager prefill/readout paths and unsupported profiles keep their established routes. It introduces no new cache quantization or Half PV precision policy.

Balanced uniform42 with the original BF16 partials was already numerically reasonable on independent frozen actual-input checks. FP32 partials and late IFWHT were a separate optimization opportunity, not a condition for accepting mathematical repartitioning. The final FP32 choice improves independent pre-gate accuracy on every checked row at the two full-head calls.

## Evidence

### Workload and measurement boundaries

The early-round immutable corpus has 187 actual layer3 C1/T8 calls at windows 19,962–20,870, all valid8, from the same engine/configuration. Lab records complete inputs and physical pre/post-state. The corpus is `cf48431b…`; original engine baseline is `9ea5ca8e…` and source starts at `8cf6eb8c73d194dc3e0f6a1815258dc0dcffad16`. Complete diagnostic passes inspect 14,951,406,976 state bytes. Helpers restore pristine full mutable state before each timing sample; restoration, output/workspace initialization and synchronization are outside timing. Warm contentmemo experiments explicitly retain only their memo workspace. Profiling-enabled resident Lab and non-injected targeted helper timings are reported separately.

Attention timings include append, partial and reducer. Block timings include input norm, Q/K/V projections, QK norm/RoPE, attention, output projection and residual; the MLP is outside this GQA segment. Individual medians across different operations/sessions must not be chained into a claimed total speedup. A separate final same-input initial-math-control versus native comparison is reported below; its ratio is measured directly, not composed from these steps.

### Measured performance

The first sequential attention screens gave 161.056us for uniform42/BF16 and 163.200us for FP32late. These were exploratory screens, not a paired proof that FP32 is slower. On actual record54, a rotating 31-sample helper measured original builtin176.192us, BF16 uniform148.928us, FP32late warp0 reducer151.616us, and cooperative256 FP32late147.776us. The cooperative reducer preserves the FP32late output bits on this control.

The following block measurements are fresh paired controls on the same 187 recorded inputs. “Same diagnostics” means matching per-record numerical diagnostics, output diagnostic fields and physical state fields between the two arms; it does not mean the mathematical FP32 candidate equals the original BF16 recording.

| Change, with the specified control | Timing points | Median control→candidate us | Median paired ratio | Slower points | Same diagnostics |
|---|---:|---:|---:|---:|---|
| Q FWHT fusion vs cooperative FP32late (`2a9ae105…`) |374|261.648→259.376|1.00917x|35|yes|
| NoInactive/page-safe control vs QFWHT candidate (`e3c0b309…`) |374|256.704→253.600|1.01226x|42|yes|
| Codec4 vs NoInactive/page-safe control (`97dfe902…`) |374|254.240→250.832|1.01349x|18|yes|
| Fixed-S K/V PairStore vs same control (`53e67731…`) |374|255.424→252.048|1.01382x|18|yes|
| Strided Q/gate vs same control (`63104abb…`) |374|253.968→251.840|1.00840x|60|yes|
| Exact combination vs codec4 (`faec5d5f…`) |374|250.656→246.288|1.01798x|12|yes|
| Append FWHT reuse vs exact combination (`c73c1174…`) |374|246.672→241.328|1.02143x|5|yes|

The exact-combination identity is `b6101d1291928cb6d03d7caf6769d620aa04efa3d4979f5f9abfb197d172e75d`. Append reuse is `5553b66e3bb27e7e642fb8ffad1df5bb33c2fe573ca004751cfa2afa4d945f65`. The exact-combination versus earlier QFWHT/cooperative candidate `37afc1c0…` direct evidence currently available here is **full forward**, operation `47ab5114…`, rather than a direct paired block comparison: identical generated tokens/rounds in all three requests, decode ratios 1.005496x/1.006480x/1.006437x for Astro/Django/Sympy. Prefill ratios .996902x/.997460x/.998092x show no claimed prefill gain. These forward ratios compare two already-optimized candidates; the separate final initial-math control below supplies the total fixed-input block comparison.

Raw step summaries are in `accuracy/final-performance-extract.json`; source op JSONs are retained under `.scratch/lab/`.

#### Snapshot A: direct total paired block comparison

Operation `df2b2f4640e44e938d804a6ee8bc565f` compares native builtin with an immutable initial-math control on the **same newly captured native trajectory inputs**. Corpus `1bb9e643…` has357 layer3 GQA-block records,57,148,631,400 recorded bytes, anchor positions19,954–21,788 and actual visible windows19,962–21,796. A separate CPU range-read of all357 records checks the32-byte positions and4-byte valid count (12,852 bytes total): every record is valid8, has eight consecutive positions and satisfies `positions[7]+1=anchor+8`. This is actual recorded window evidence, not an unchecked anchor inference; proof and CSV are in `figures/actual-windows-proof.json` and `figures/actual_windows.csv`. Recording trajectory differences are empty. This is a second corpus; it is distinct from the original187-call corpus used for the step screens.

The CLI direction is baseline=native, candidate=old. The correct native speedup is computed as **old_ms/native_ms for each timing point, then median**, rather than using the CLI's reported reverse-direction ratio or dividing aggregate medians. Across714 paired points, native is faster714/714: median1.390969x, range1.037905–1.524880x. Separate latency medians are347.728us old and250.112us native. This is the directly measured accumulated block gain; no per-step ratios are multiplied.

Native builtin exactly reproduces all357 captured outputs and complete physical state. Initial control preserves all physical state, but its mathematical outputs differ as expected from the arithmetic changes: zero nonfinite outputs, maximum peer-distance absolute error .125 and worst-record relativeL2 .00256977. Those are distances between arms, not errors against mathematical truth. The earlier common-input FP64 studies supply the independent rounding/error analysis; they are not relabeled as FP64 evaluation of every new-corpus input.

The initial control is `1d0c70deb5e70bc9537113e153e7c57926d459b35f6e9055d8829111ab02571f`, built with the current compatible host/ABI and initial algorithm HOT files. All7 ABI headers match initial hashes;61/62 actual compilation dependencies match the initial source map, with only an explicit BALANCED0 policy annotation differing. Ordinary public-Q actual54/179 runs reproduce the original recorded full outputs bitwise and state exactly. This is a validated initial algorithm control, not an unverified old binary. The temporary HOT-source compilation window was restored in a finally path; the receipt confirms all runtime hashes restored. See `candidate-initial-math-control/`, `initial-control-window.json` and `accuracy/final-direct-paired-summary.json` under the evidence root. Formal full-service results are reported separately below with their different trajectories.

#### Short-context policy

The selected simple native policy uses B32, `tiles=ceil(W/32)`, `S=min(capacity,42,max(4,tiles))` for 0<W≤70,000, then the original active-split policy outside this range, including invalid/neutral windows. Capacity remains the original 85-split allocation; reducing the physical grid/capacity alone had no consistent benefit (three pairs .99605x). The min4 keeps the measured tiny-window behavior, where S1 lost at W≤32.

Derived short-window operator scans reuse actual Q/K/V/gate and reconstruct legal prefix/side state; they are scheduling evidence rather than actual short-prompt quality evidence. K1/min4 was stronger at ≤1K and 4K–16K, with median ratios 1.0695x and 1.3061x in the final fixed-grid scan. There is a real local cost around 1.4–2.6K: old policy wins the checked 43–72-tile transition cases, by about5.08% median over K1, and W2048 was .92975x in one final scan. The two-tile ramp did not solve this: 0/11 wins, median1.57% slower than the best measured arm; generic auto formulas also lost at larger windows. The chosen broad K1 rule therefore retains an explicit local limitation, rather than claiming a universal win or introducing an untested per-length dispatch table.

### Independent numerical target

Frozen exports include actual BF16 Qrot and independently slow-HQ decoded selected Krot/Vrot, including sink/recent residual side planes, original positions/masks and scale .0625. Slow serial and actual fast-group BF16 KV agree bytewise. Exported builtin reproduces the recorded public output exactly; all variants preserve physical state. Neither property makes builtin the mathematical truth.

The CPU reference computes unsplit FP64 QK, stable softmax, PV and inverse signed Hadamard over the same selected inputs. It keeps both a continuous pre-gate target and the public rounding target. Manifest SHA checks, axis/head mapping (`Qhead//6`), selected-side semantics and scale are audited. An independent fsum/scalar-Hadamard calculation confirms dominant outliers to 0 or4.44e-16. Actual pregate BF16 multiplied by captured GPU sigmoid reconstructs the public output bitwise.

Full-head calls0/W19962 and179/W20846 each cover24 Q heads,4 KV heads and8 tokens:192 rows/49,152 scalars each. The earlier call54/KVhead0 adds48 distinct rows; total unique evidence is432 rows across three calls in one layer. This is finite actual-input evidence, not a universal numerical bound or model-quality benchmark.

#### Continuous pre-gate error against unsplit FP64

| W | Variant | Max abs | Median abs | RMS | Relative L2 |
|---:|---|---:|---:|---:|---:|
|19962|builtin|.00542727|.000174790|.000511281|.00196656|
|19962|uniform42 BF16|.00542727|.000174380|.000512968|.00197305|
|19962|FP32late|.00508739|.000110179|.000439183|.00168925|
|20846|builtin|.0156740|.000281336|.000754298|.00241629|
|20846|uniform42 BF16|.0156740|.000286985|.000759148|.00243183|
|20846|FP32late|.00775842|.000137009|.000527752|.00169058|

Uniform42 changes pooled error by+.330%/+.643% **of the small existing error**, with lower worst-row relativeL2 at both calls. FP32late reduces it14.10%/30.03%, improves384/384 pre-gate rows and all8 KV groups, and lowers pre-gate p99 ULP from13/21 to2/4. Per-row/per-head metrics, RMS/max/median and ULP tails remain in `accuracy/record{0,179}-allheads-report.json`; maximum ULP near cancellation is interpreted alongside absolute error, not used as an isolated rejection gate.

#### Gate targets and rounding attribution

Continuous gated error compares to `y64 * captured_sigmoid32` in FP64. Public projection follows `RN_FP32(y64) → RN_BF16 attention → FP32 sigmoid multiply → RN_BF16 output`. These are different targets because the public pipeline has an intermediate attention rounding.

| W | Variant | Continuous-gated relativeL2 | Public-projection relativeL2 |
|---:|---|---:|---:|
|19962|builtin|.00266425|.00222754|
|19962|uniform42 BF16|.00267533|.00226076|
|19962|FP32late|.00258382|.000928764|
|20846|builtin|.00256937|.00153893|
|20846|uniform42 BF16|.00266316|.00332882|
|20846|FP32late|.00237318|.000342487|

Two concrete bins explain why a single projection aggregate can move sharply. At call54/t3/head5/d17, y64=-1.3803928673; correctly rounded attention is-1.3828125 (uniform42/FP32late), while builtin gives-1.375. Sigmoid .7892336845 yields continuous target-1.0894525488. The public double-rounded target is-1.09375, but builtin's larger pre-gate error happens to yield-1.0859375, closer to the continuous gated target. At call179/t0/head3/d251, y64=2.3495935456 lies near the BF16 midpoint: builtin/FP32late give2.34375, uniform42 gives2.359375; gate .8006919622 yields public1.875 versus1.890625. These are rounding-boundary effects with exact selected inputs/state, not evidence of missing keys or gate amplification.

#### FP32 partial versus IFWHT placement isolation

At call54 and179/KVhead0, balanced42 BF16-early relativeL2 is .002624172/.002935104; BF16-late is .002251564/.002405149 (14.20%/18.06% error reduction); FP32-early is .00173197111/.00175952156; FP32-late is .00173197115/.00175952166. FP32 early/late differ in only1/12,288 and2/12,288 pregate BF16 components, max4.77e-7/1.91e-6. Thus removing intermediate BF16 rounding supplies the main FP32 improvement, while moving IFWHT after merge enables the performance change with negligible added FP32-order error here. BF16-late also changes rounding locations, so its gain cannot be attributed to transform placement alone.

No prefill-specific tolerance was reused as a long-HQ acceptance threshold. Existing plain BF16 double-reference and codec-vs-BF16 tests retain their scope; exact indices/state/layout checks remain exact. This round adds frozen independent-reference evidence for the long-HQ arithmetic gap rather than inventing a new unsupported threshold.

### Semantics, state and lifecycle

- Repartitioning and FP32 arithmetic preserve masks, scale, causal positions, sink/recent selection and append ownership. Repeated all187 diagnostic runs report exact mutable and read-only physical state. Exact layout/scheduling candidates preserve the prior FP32 per-record output diagnostics.
- Late IFWHT initially turned forced+0 invalid rows into signed zeros. The repair preserves canonical+0 for finite gates, retains0×sigmoid(NaN) where the old path did, and keeps the all-neutral m=-inf gate bypass. Targeted masks0..8, special gates and nullmask pass; allvalid output is unchanged. This cannot explain valid-token forward branching.
- GateStride2 has a separate segment-only prepared entry. Public-frame and older rotated helpers remain GateStride1. Five targeted cases through two same-object adapters verify finite/special gates, invalid forced-zero, all-neutral and nullmask semantics; interleaved Q slots were NaN-poisoned, and all outputs/state/inputs match.
- Codec4 uses warp-uniform eligibility. Any real general-k row forces the whole warp through two complete eight-lane phases; side/outside/inactive rows do not incorrectly trigger general fallback. Independent integer/address/collective audit and focused GPU147,456 BF16 coordinates pass, including actual k1 rows, side and ring-invalid cases. Persistent tests extend existing codec-row and HQ verification targets rather than relying on actual-all-k0 traces alone.
- HQ page addressing was repaired before adoption: a fixed128-entry per-split PageIds cache could exceed bounds (W40000/S4 needs157 pages; large supported windows also exceed128). HQ now looks up the page for each tile directly, while linear cache behavior remains unchanged. Independent CPU boundary checks and legal derived W40000/S4 fast-versus-independent-slow-HQ GPU check cover the repair (the extension contains original zero/unwritten rows, so it is bounds evidence, not real40K performance or FP64 quality evidence); the old out-of-bounds path was not executed as a reference.
- Inactive split writes are skipped only with reducer neutral-metadata handling that never reads their poisoned payloads. 7F/FF poison, all-neutral and masks are explicitly checked.
- Native attention workspace uses raw byte queries: FP32 partial+two stats is16,842,240 bytes at the conservative supported allocation, versus the old BF16 budget8,486,400. Host planner/replay arena take max(old,native), and max the whole native chain+live tail; tensor ne/split count are not inflated to hide the dtype change. Native prepare precedes graph capture. Transient TL synchronization prefix is cleared inside every timed native chain using Layout's shared16,640-byte constant; Lab persistent workspace defaults are retained.
- A new existing-target `test_trellis_body` fixture exercises the **complete** native Q/pairedKV/output chain with real5120/6144/1024 geometry and39.3MB synthetic K6 weights. valid8/7/0 zero-workspace baselines are compared to whole-workspace7F/FF poison through native and builtin entries, all outputs and physical state exact. The actual CMake test target compiled and passed on GPU (rc0,0.1767s,printed `ok`), including the original Trellis body cases; evidence is `native-targeted-evidence/native-workspace-targeted-gpu.{json,log}`. This was the owner's real target, not a reset-helper-only substitute.

### Forward observations and limits

Natural-EOS three-request exploratory forwards used the original8192 output cap. Builtin versus the mathematical FP32 candidate produced Astro913→1839, Django2319→512 and Sympy1379→1155 tokens, all EOS248046 with closed thinking. Initial prefill-logit hashes agree. Uniform42 BF16 alone also branches (Astro365 tokens), so greedy branching is not specific to FP32 or a signed-zero defect.

Lightweight decoded-content review finds valid tool JSON/parameters and complete tool/thinking delimiters. Astro follows the same bug/fix with different inspection tools; Sympy addresses the same distance/padding issue with different investigate/edit balance; Django continues the same URL-regex investigation, but the shorter candidate is not evidence of finishing the SWE task. This is syntax/semantic inspection, not a SWE quality benchmark. Own-trajectory TPS over different token/round workloads is not a fixed-work speedup.

Common identical-work prefixes before the first emitted/shape/position fork are115/267/28 committed tokens. A shared cap24 is a diagnostic service control only; it is short, can change final mask/extent, and must itself be checked for matched workload. Formal8192-cap natural-EOS service results stay separate. Exact-combination versus37af later produced identical complete outputs/rounds in all three requests, supporting that the layout/codec changes preserve the chosen FP32 trajectory.

### Alternatives retained, not adopted

Rejection or deferral below is based on measured performance or incomplete integration/quality evidence, not a bare bit mismatch.

| Alternative | Evidence and disposition |
|---|---|
| Uniform42 BF16 partials | Independent error stays at builtin scale; remains a reasonable mathematically equivalent alternative. Final FP32 selection has stronger accuracy evidence and cooperative performance. |
| Compact64 reducer / B48 | Actual54 compact167.520us and B48 S21/S42231.776/241.952us versus cooperative147.776us; not competitive here. |
| Direct global RQ | Exact187 diagnostics but whole-block .95921x,373/374 points slower; retains lower shared-memory opportunity for different tiling. |
| Natural weights2 K/V split count | Fixed250.912→natural250.368us,1.00251x one pair; sideK27/sideV157 bytes change from mathematical repartitioning, codes/meta/ring/read-only exact. Gain weak and integration evidence insufficient; fixed-S selected. |
| RowScale reassociation | Actual54 all4heads/KV changed2595 codec rows/3289 coordinates, all adjacent BF16 ULP. Independent common-codec F64 changed-subset relativeL2 .001660718020578→.001660718075859; tiny deltaSSE1.086e-7, not a whole-cache pooled norm. Finite legal-domain audit has no FP32 overflow concern (inv_scale .0431–.3448). Retained but not folded into final codec4/native under the source lock. |
| Half PV inputs/accumulator | A1(HalfP/V,F32acc) improves pooled R64 pre-gate error .78%/1.15%; A2(Halfacc) changes+1.28%/-.27% on two calls. m/l/partial remain F32 and state exact. Range/subnormal census and hardware diagnostics exist, but model-quality/precision-contract and final integration evidence are incomplete. Neither is accepted in final native. |
| Y4 bounded content memo | Tiny16phase/131,072 coordinates exact; actual54 base144.960us, cold239.232us,warm208.608us (.6949x), plus832MiB memo (889,257,472 total WS). Stopped after measured warm slowdown. |
| Generic auto / two-tile ramp | Local transition improvements do not generalize; ramp0/11 wins and larger-window regressions. Keep simple measured K1 policy with documented local cost. |

### Native route and host verification

Operation `16fc554aea794698941df5e19eee1fed` profiles the default builtin engine with no GQA extension loaded. Attribution identifies5,712 `gqa.native_block` instances across16 GQA layers and357 decode rounds. Its complete Astro output is1,839 tokens; token sequence, timing-independent round trajectory, stop state and prefill-logit SHA match the pre-port exact-combination forward `47ab5114…`. The profile has no independent builtin numerical reference (`reference_available=false`, `numerical_match=null`); the cross-run trajectory check is separate evidence that the native route is active and preserves the selected candidate trajectory. It is not a controlled speedup against the original BF16 engine. The independent parser result is `accuracy/native-profile-crosscheck.json`.

Local `cargo test --workspace` on aarch64-apple-darwin, without a Linux target override, returns0:252 reported result groups are all ok, with log totals3,068 passed/0 failed/6 ignored. The ignored tests explicitly require real artifacts or machine-local inputs. Current vendor410-file hashes, patched-file hashes and manifest match the official verification receipt; all3 frozen HOT sources match. Relevant implementation diff checks pass. These host checks do not replace Linux CUDA/capture/service acceptance. Logs and hash receipts are in `final-validation/local/`.

### CUDA/GPU acceptance: original failure and targeted repair

The final CUDA workspace/all-target command `make check CUDA=1` passed (rc0,86.79s). The official `scripts/gpu_profile.py` runner then completed with **rc1**, which is preserved in `final-validation/remote/results.json` and `gpu-profile.log`. Its initial CTest stage passed130/131 targets; the sole failure was `ignis_kernel_model_load_trellis_test`, reporting decode-graph actual20,968,208 versus the stale closed-form expectation20,621,328. The original failure log remains `gpu-profile/kernel.log`; it is not relabeled as an all-green original run.

Independent allocation derivation identifies the missing native C1 peak. Fixed-profile FP32 attention storage is16,842,240 bytes, intermediate tensors720,896 and fixed-S paired TL scratch1,559,296, giving native workspace19,122,432. The live post/fused MLP tail adds360,448, yielding a layer peak19,482,880 versus the old decomposed B2 peak19,136,000. With outer buffers1,485,312 and identifiers16, the strict total is20,968,208. The test repair derives pair scratch from the public single-KV query and public maximum-output counter prefix; it calls neither the model nor native-segment capacity query as its reference, uses no observed-total literal, and preserves exact equality. Additional B1/B8 checks cover both the native C1 floor and larger-lane decomposed peak alongside existing B2.

Only this test expectation changed. Its real CMake target was rebuilt and rerun successfully (rc0,5.2935s,printed `ok`), retaining the original130 passing CTest results without repeating them. Current hashes of all24 frozen changed/new runtime implementation/header files still match the pre-repair snapshot; no runtime arithmetic was changed for this repair.

The same official run also passes all four Trellis GPU suites and VRAM reload, with no ignored/skipped tests:

| Suite | Passed | Failed | Ignored |
|---|---:|---:|---:|
| `trellis_gpu` |2|0|0|
| `trellis_lanes_gpu` |1|0|0|
| `trellis_layers_gpu` |1|0|0|
| `trellis_projections_gpu` |1|0|0|
| `vram_reload_gpu` |1|0|0|

The combined evidence is therefore original130 CTest passes plus the repaired strict-budget target pass, and all five Rust GPU stages passed: **no remaining failed GPU item**. This conclusion does not rewrite the original runner's rc1. Source receipts, original logs, `model-budget-rebuild.log` and `model-budget-rerun.{json,log}` are retained under `final-validation/remote/`; the CPU aggregation is `final-validation/local/combined-gpu-evidence-check.json`. Service, capacity, canary and the bounded native Vision smoke follow below.

### Snapshot A: formal initial/native service runs

`final-validation/services-v2/` contains two completed owner-managed launches: initial followed by native, each one warmup and three measured samples per frozen request. This is one launch per arm, not a balanced multi-launch experiment. Both legs pass the benchmark's validation: cold prefix (`cached_tokens=0`), one complete prefill pass, matching tokenize/computed input counts, consistent positive output counts and valid server phase timings. Full effective configuration, benchmark configuration/protocol/fixture manifest/models view and the three saved requests are equal across legs.

Both use the same Trellis artifact path and validated symlink target (13,379,877,124 bytes, conversion sidecar309,495 bytes), context262,144, chunk1,024, two configured lanes/actual C1, HQ-E8-2b, DFlash2 seven draft tokens/full draft head, profile `rtx50-16gb`, prompt reuse disabled, retained slots0, loop guard off, and host-resident Vision enabled. Requests keep greedy sampling, seed0, thinking enabled, original8,192 output cap and `ignore_eos=false`. Both server logs report `artifact.verified`. This round's new Lab setup records full artifact SHA256 `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c` in corpus `1bb9e643…`'s identity; the CPU check reconstructs that corpus identity key. Lab's setup implementation computes `digest(cfg["artifact"])` over the full file. Setup configuration and both service commands name the same artifact path; the recorded symlink target/size agree, with mtime2026-10-31T17:54:15.848059Z preceding both service legs. This reuses the **existing setup digest**, with path/stat evidence under `binary-provenance/`; it does not claim a new13GB scan for each service leg. The same SHA also appears in the [historical coding-agent baseline](2026-11-03-coding-agent-5080-baseline.md).

Immutable initial server SHA256 is `ec17b4e30c5755abf9e144e984b8d68a0fc42101b18eae6c75a495a8905d9012`; its build receipt rechecks all803 initial runtime files against the frozen initial source. Native server SHA256 is `0797d40d022679fffa342d8d7d28b9278293c7d5f5946033a5362afd8339b7d4`; its shadow build matches final main runtime sources. At snapshot A freeze, the CPU analysis independently checks all810 entries in that native runtime source map against the matching main files, with zero differences, and joins both launch-receipt server hashes to their frozen build receipts. Both benchmark invocations use the same frozen native bench executable. The common `code_commit=8cf6…` field alone does not identify the native dirty-source build; these hashes and source maps do. Receipts are under `final-validation/binary-provenance/`.

The table reports the median of each arm's three samples. Ratios are **native/initial own-trajectory TPS**, not fixed-work speedups:

| Request | Input tokens | Initial output / rounds | Native output / rounds | Median decode seconds initial→native | Initial TPS | Native TPS | Own-trajectory TPS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Astro |19,954|913 /187|1,839 /357|4.109→7.433|222.193|247.421|1.11354x|
| Django |22,420|2,319 /472|512 /111|10.750→2.334|215.727|219.323|1.01667x|
| Sympy |16,717|1,379 /255|1,155 /234|5.363→4.808|257.111|240.226|.93433x|

Astro's median decode duration increases from4.109s to7.433s as its output grows from913 to1,839 tokens. Its higher own-trajectory TPS does not mean that the complete request finishes sooner. Django and Sympy generate fewer tokens and use fewer rounds, so their shorter decode durations likewise do not isolate an operator gain.

All18 measured samples and6 warmups finish as HTTP `tool_calls`, with no `length` truncation; exposed response messages are stable within each arm/request. Every tool name is declared in the saved request, all24 argument strings parse as JSON, and required parameters/declared enums checked by the CPU parser are valid. Both reasoning and visible/tool fields are retained. The HTTP response does not expose the raw thinking-open flag or raw completion IDs, so these checks do not claim raw state/token equality. No tool is executed by this benchmark, and valid tool structure is not a SWE task-quality verdict. The different Astro/Django/Sympy next steps remain the content differences described above. **Sympy own-trajectory TPS falls6.57%**, while its output changes1,379→1,155 tokens and255→234 rounds. This regression is retained rather than hidden by aggregating the three tasks; the three service cases do not all accelerate.

#### Cap24 observable-prefix diagnostic

For each request, only `max_tokens` changes from8,192 to24. The capped wire request is identical across arms. All three cases return24 tokens with `finish_reason=length`; exposed message fields and normalized reasoning/content are equal, and aggregate speculative work is equal:

| Request | Rounds / drafted / accepted | Initial decode ms | Native decode ms | Initial/native decode ratio |
|---|---:|---:|---:|---:|
| Astro |9 /55 /15|189.565|187.081|1.01328x|
| Django |6 /37 /18|137.092|126.340|1.08510x|
| Sympy |6 /41 /18|126.559|123.174|1.02748x|

Both full `drafted_at` and `accepted_at` histograms agree. These are single samples per arm, with short decode spans and much longer prefill, so they are diagnostic observations rather than a statistical service benchmark. HTTP exposes neither raw token IDs/logprobs nor chronological per-round positions/emissions/masks. Equal text, counts and histograms therefore establish only **observable text and aggregate-work equality**, not complete trajectory or exact kernel-work equality.

#### Capacity and canary

Both formal capacity invocations pass with exactly131,072 input tokens, zero cached tokens, one full prefill pass, one generated token and `finish_reason=stop`, without truncation. They are capacity checks, not speed cells. Both model startups and graph capture complete; the native log reports two captured graphs and a non-oversubscribed VRAM plan. Native decode/capacity/canary requests subsequently complete using that service instance.

Each arm's four canaries passes the harness's sanity and within-arm determinism checks. Some64-token canaries expose reasoning-only or shortened visible answers; this limited harness is not a complete-answer or cross-arm quality evaluation. Capacity, canary, startup, GPU telemetry and stop evidence are retained separately under each leg. CPU medians/provenance/response checks are reproducible with `accuracy/analyze_final_services.py`; structured results are `accuracy/final-services-summary.json`, and tool parsing is also retained in `accuracy/service-tool-parse-summary.json`.

### Native Vision smoke and its truncation limit

A separate profiled native-server session uses the existing colour-image fixture and the question “Describe everything you can see in this image, in detail.” A completed text warmup precedes exactly one image request. The image-only NVTX interval, rather than whole-process counts, contains416 native partial and416 native reducer executions, exactly16 GQA layers×26 decode rounds. The reducer grid is[24,1,8] with GateStride2; the partial grid is[4,85,1], so native C1/T8 execution is observed during the actual image request, not merely at initialization.

The request reports90 prompt tokens,96 output tokens,26 speculative rounds,177 drafted and70 accepted, including25 rounds with a full seven-token draft. Visible output describes a solid red square. It ends with **`finish_reason=length` at the explicit96-token cap**, in the middle of the detailed description; it is not a natural-EOS or complete-answer quality result. The smoke validates nonempty image-conditioned output, native-route coverage and speculation. It does not replace teacher-forced Vision quality/reference tests or establish multi-image behavior. The profiled590ms server elapsed time is not reused as an unprofiled service benchmark.

The isolated session closes cleanly (server rc0), with one image attempt and warmup completion recorded. Raw response/usage, lifecycle, explicit server argv and image-range kernel counts are retained under `final-validation/vision-native/`; Nsight bulk artifacts stay in scratch. This smoke is separate from both formal service launches.

### Derived65K actual-prefill smoke

A derived synthetic text request uses half-length padding from the existing capacity construction plus a long-reply instruction. Tokenize and actual prefill both report65,574 tokens, with0 cached tokens and one full prefill pass. It generates96 tokens over37 speculative rounds (247 drafted/59 accepted), including34 full-seven-draft rounds. The response is nonempty and ends with `finish_reason=length` at the explicit96-token cap. Recorded prefill/decode times are30.666s/.919s; these are one smoke execution, not a new benchmark contract.

This exercises genuinely populated long-context KV through the same native Trellis/HQ default guard. There is no additional long-request kernel trace: route coverage here combines that guard contract and full-seven-draft counters, rather than claiming directly counted native kernels. This is not a real coding-agent prompt, natural-EOS result or long-context quality benchmark. Exact tokenize and execution summaries are retained under `final-validation/derived64k/{tokenize,summary}.json`.

### Snapshot A restored resident Lab identity

The pre-accumulation status receipt reports Lab `ready`, no active operation, profiling enabled, session `dbcfa8bc3611442f911390911fd693d8`, engine PID2551303, native baseline `bf3d5b466ed6c19150fb1a95bebdd72145a241983055d5c077ad191cd99982ca`, and corpus `1bb9e6434290c337b4b331e03774b6b60753fba4f6a49f03bb9d8c667798ee80`. All810 runtime-file hashes match the native baseline. This records the restored resident state; it does not substitute for the per-operation receipts above. The small status artifact is `final-validation/final-lab-status.json`.

### Snapshot A remaining native GQA stage costs

The existing `16fc554a…` full-forward trace supplies357 rounds×16 GQA layers = **5,712 layer-rounds**, with57,120 attributed kernels. All5,712 instances match the native ten-kernel order by name and graph/NVTX provenance: RMS→Q rotate/decode→joint K/V rotate/decode→norm/RoPE/FWHT→partial→reducer→out rotate/decode. The three identical TL rotate/decode occurrences are assigned Q/pair/out by the frozen body's launch order, rather than pretending their symbols distinguish those roles. The means below are cumulative kernel duration divided by5,712; they exclude prefill, MLP/post-attention norm, GDN, other work, reset and copies. K/V is a joint pair, with no separate K-versus-V timing attribution.

| Current native stage | Kernel µs/layer-round | Share of GQA kernel sum |
|---|---:|---:|
| Input RMSnorm |1.203|.55%|
| Q projection |45.792|21.11%|
| Joint K/V projection |17.322|7.99%|
| QK norm/RoPE/Q FWHT |2.222|1.02%|
| Attention partial / append |122.777|56.61%|
| Reducer / late IFWHT / gate |6.739|3.11%|
| Output projection / residual add |20.838|9.61%|

The kernel sum is216.895µs/layer-round, counting overlap. The corresponding kernel-only span is215.280µs, busy union213.960µs, gap1.320µs and overlap excess2.935µs. These all-layer instrumented profile quantities are not the250.112µs standalone layer3 replay wall; subtracting them would not identify an attributable overhead. Separate prior raw-trace evidence counts5,712 prefix memsets totaling3,443.268µs, about.603µs/call, outside the kernel sum. Reset precedes the first kernel. Residual-copy duration is not available in this kernel-event slice, so it is not inferred from the gap. Stage/projection/per-layer details are in `accuracy/gqa-stage-costs/{summary.json,report.md}`; historical composition remains in scratch only.

### Snapshot A hardware counters and next opportunities

The existing final NCU report profiles actual record54 through the private prepared-Q **Gate1** path; service consumes Gate2. Both intermediate and final reports use `cache-control=all`, `clock-control=none`, kernel replay and six passes, so cold counter/elapsed results are not resident service costs. Final partial executes50.328M warp instructions with63.962MB combined local load/store traffic, issue active43.8% and achieved occupancy16.2%; L2 traffic is66.258MB and DRAM reads14.130MB. This is not established as a simple DRAM-bandwidth bound. The helper's state-equality flag is same-winner repeat consistency, not an independent post-state oracle; independent exact state evidence is the Lab acceptance above.

The partial uses255 reported registers/thread (256 allocated) and48,640 allocated shared bytes/CTA. Registers and shared memory each cap residency at two128-thread CTAs/SM:8 warps against a48-warp hardware maximum, a16.67% static cap already nearly filled. Reducing registers alone cannot admit a third CTA while shared remains this large. No sampled clock/cycle counter was collected, so no strict SOL percentage follows from the nominal clock. The reducer has40 registers and zero spill; its cache-flushed14.24µs NCU duration cannot replace the full-forward6.739µs kernel mean or establish a removable resident cost.

The practical priority is partial-path decode/instruction/live-state pressure, supported by its56.61% stage share and substantial local traffic; Q and output projection decode are next. The three projection decoders together cost80.080µs/layer-round, versus3.873µs for their rotations, so small launch/rotate tweaks target a much smaller pool. Reducer and norm-only changes have limited total-component budgets. This ranks future experiments without promising occupancy gains or adding new production changes. Exact commands/raw counters and interpretation are retained in `ncu-final-native/{receipt.json,summary.json,analysis.md}`.

### Final accumulated selection: Q S3 and exact reducer prefetch

A bounded private Q projection keeps the existing5120→12288/T8/STORE contract, input rotation, FP16 MMA folds every32 inputs and BF16 output boundary, but partitions40 input blocks into S3 ranges13/13/14. Its288 work units use1,279,488 bytes, with a matching private workspace query included in the native max and an independently sufficient decomposed Trellis budget; required bytes are checked explicitly; public/global TL, KV/out and MLP paths remain unchanged. Decomposed C1/T8/HQ recording uses the same scoped Q partition, while eager and legacy-candidate priority remain intact. Generic/public individual `q_proj` replay still uses S1; a separately recorded scoped-S3 Q output is therefore not an automatic bit-exact builtin individual-op oracle. This limitation is distinct from the matched S3 GQA-block route.

Independent357-record matched screens use the same persistent-workspace S1 control. S2 is flat/negative (.995768x,245.056→245.312µs,232 slower points); S3 has a positive signal (1.028265x,244.672→238.176µs,3 slower points). Both are finite and preserve physical KV state on fixed inputs. These are separate comparisons, not components to multiply into the earlier1.390969x total.

For S3, actual corpus1bb9 record0 exports one shared GPU-normalized BF16 input and the original encoded K3 Q weight. An independent CPU trellis decoder and FP64 two-H128 projection compares **both** S1 and S3 to the same mathematical target. Six stored codec fixtures validate196,608 decoded FP16 coordinates bitwise; direct parity-H128/math.fsum checks of all changed actual coordinates agree with the vectorized reference within1.92e-15.

| Actual projection against common FP64 | S1 | S3 |
|---|---:|---:|
| Pooled Q+gate relativeL2 |.001703332247895|.001703332364278|
| Q-only relativeL2 |.001913167160653|.001913173127700|
| Gate-only relativeL2 |.001699391116410|.001699391116410|
| Violations of existing TL error bound |0|0|
| Worst fraction of existing bound |.880784|.880784|

Only4/98,304 BF16 values differ, all adjacent-ULP Q values; all49,152 gate values match. Three token/head/role rows improve, one worsens and380 tie. Pooled true-reference relativeL2 changes by1.16e-10. This is consistent with ordinary FP32 grouping noise, with no new FP16 boundary; one actual tensor is not a whole-model quality benchmark. Three exploratory S3 forwards reach EOS with changed trajectories, so their lengths/TPS are not fixed-work performance evidence. Raw/reference/row metrics are in `accuracy/q-projection-reference/`.

The exact reducer option caches the original `tile_l<=0` skip predicate, including its NaN behavior, and prefetches four partials while retaining increasing-s scalar FMA order and all gate/mask boundaries. On actual54,31 interleaved samples give full-chain medians136.928→135.904µs (ratio-of-medians1.007535x; median paired ratio1.009459x), with18/31 faster points. Warm-reducer medians8.416→6.944µs give1.211982x (paired median1.206422x). Full output SHA90d386… and state match accepted arithmetic exactly;24 mask/special-gate arms and W9 empty-active FF-poison checks also match. This is a small exact control/data-movement opportunity, not a cold-NCU14µs latency claim or a357-record cumulative gain.

A guarded112-CTA Q+K+V S1 joint decoder remains a **strong alternative**: its separate357-record screen measures1.033887x (246.432→237.152µs,3 slower points), with finite output and worst-record peer relativeL2 5.77e-5. It is not a head-to-head winner over S3's1.028265x from a different pair. Independent shape/workspace audit passes; wide/capped inputs retain original Q plus fixed-S KV pair, so observed weight coverage would be14/16 layers. K/V changes from S21 to S1, producing742 changed physical numeric bytes across the corpus (side K/V and six code/meta bytes); read-only indices, tables and ring remain exact. These content changes are **not** a rejection reason: they require common encoded-K/V FP64 attribution and correct original-encoder-on-new-input association, rather than an old-state byte oracle. S3 plus exact reducer is selected for the final integration because it already has the independent actual-Q reference, forward coverage and unchanged KV-update contents; QKV is retained for further numerical/integration evidence within a later bounded evaluation. No small nominal difference between these separate speed ratios is used to claim superiority.

Snapshot B's frozen build succeeds (rc0,153.75s), producing server SHA256 `5bec02e3d27f9658a63eb9024dfa85df134721c9ec9663a1dac943ab15548507`. The source-map delta from A is exactly the six intended files: TL host, shared token mixer, owned reducer, owned attention implementation/policy and scoped decomposed GQA caller. The original initial binary remains unchanged; test-source synchronization is separately recorded and does not alter the810-file runtime identity. Receipts/maps are under `final-validation/s3-binary-provenance/`.

Snapshot B then passes its own CUDA all-target stage (rc0,27.97s) and official GPU-profile runner (rc0,572.81s): CTest131/131, all four Trellis GPU suites and VRAM reload pass, with zero ignored/skipped tests in those GPU suites. This is a new completed B receipt, not snapshot A's repaired runner relabeled. Its810-runtime source-map key is `df10c9f1f90bcc1576321fc635a9412bda0f12c62554080c725aad56db83db49`.

An extra SM-cap1 exploration of the complete old Trellis body fails its unrelated T80 strided-store and residual-add cases identically on prior A and B; both original failure logs remain. A test-only `--native-gqa-only` selector then runs the complete native S3 query/chain and valid8/7/0 zero/7F/FF-workspace cases at SM-cap1: rc0,1.584s. With no selector, the default strict full body still passes at cap0 (rc0,1.522s); runtime810/server identity does not change for this test-only selector. Thus the focused native-cap result passes, while **the entire cap1 body suite is not claimed passed**. Receipts are `final-validation/s3-cap1-native-workspace*`, `prior-A-cap1-body*`, `s3-native-body-default-rerun*` and `s3-cumulative/`.

An additional accumulated-combo versus QKV comparison `035f…` was terminated by systemd-oomd and is incomplete; it contributes no performance or accuracy result. The successful QKV screen above remains evidence for that alternative. Late experiment receipts remain under `q-split-screen/`, `reducer-prefetch4-*` and `accuracy/private-qkv-s1-independent-review.md`. Snapshot B cumulative GPU and service/Vision stages and its new direct initial-versus-B comparison are complete under their own receipts. Snapshot A's1.390969x/714-pair figures and service results retain their original source labels, without multiplying component gains.

### Snapshot B: final service and Vision validation

The final B native service is a new completed launch under `final-validation/services-s3/native-s3/`, using frozen server SHA5bec02e3…548507. Its receipt matches the B build, source-map key df10c9f1…db49 and all six selected runtime files. It reuses the original initial-v2 leg as the comparison, not snapshot A's native measurements. Full effective configuration, request protocol, fixture manifest and saved requests match initial-v2 (and A's configuration); the native bench executable is unchanged. This is one later B launch with one warmup and three samples per request, not a balanced multi-launch same-work experiment.

Each B three-sample median is recomputed from raw report values:

| Request | Initial output / rounds | Final B output / rounds | Median decode seconds initial→B | Initial TPS | Final B TPS | Own-trajectory TPS ratio |
|---|---:|---:|---:|---:|---:|---:|
| Astro |913 /187|1,092 /222|4.109→4.571|222.193|238.891|1.07515x|
| Django |2,319 /472|1,760 /378|10.750→7.884|215.727|223.228|1.03477x|
| Sympy |1,379 /255|1,371 /254|5.363→5.170|257.111|265.209|1.03149x|

All nine measured samples and three warmups are valid, cold-cache full-prefill measurements and finish as HTTP `tool_calls`, without length truncation. Within each B request, exposed reasoning/text/tool messages are stable across warmup and samples. All12 tool argument strings parse as JSON, name declared tools and satisfy required parameters/checked enums. The HTTP thinking-open flag and raw completion-token IDs remain unavailable; no raw-token/chronological equality is inferred. Tool calls are not executed, so this is not a task-quality benchmark.

Final B own-trajectory TPS rises7.52%/3.48%/3.15% against initial, but generated tokens/rounds differ. Astro still takes4.571s decode versus4.109s initially because it generates1,092 rather than913 tokens. Therefore the table does not imply that every complete request finishes sooner or isolate a same-work model speedup. Snapshot A's earlier Sympy −6.57% own-TPS observation remains historical in its table; it is not the current B result.

The single-sample B cap24 diagnostic again matches initial in capped wire request, exposed message fields,24-token count and all aggregate speculative counts/histograms. Decode spans are189.565→185.362ms,137.092→125.376ms and126.559→122.002ms (initial/B ratios1.02267x/1.09344x/1.03736x). They retain the same single-sample/short-span/no chronological-trace limits as A; no raw-ID matched benchmark is claimed.

B separately passes the131,072-input capacity case (zero cached tokens, one full prefill, one output token, `stop`, no truncation) and all four sanity/determinism canaries. The canary's limited64-token content coverage is unchanged. All service files are closed and the controller reports completion; the model payload was not rescanned by this analysis.

A new B Vision session also closes cleanly with one image request after its text warmup. It produces96 tokens over25 rounds (167 drafted/71 accepted;23 full-seven-draft rounds), visible red-square content and explicit `length` truncation. Its image-only interval contains400 native partial and400 reducer calls =16×25; the reducer symbol explicitly shows GateStride2 and Prefetch4=true. An additional read-only extraction from the same trace counts400 decode events with gridX=288, consistent with Q's96 output blocks×S3, and400 zero-memset events of the native prefix's16,640-byte size. Generic decode symbols do not themselves encode Q weight identity, and the SQL does not assign each memset to a source statement; together with the frozen source/binary and native reducer signature, these counts support the accumulated B route without a new trace. It remains a capped route/speculation smoke, not a complete-answer or teacher-forced quality result. Receipts are under `final-validation/vision-s3/`; snapshot A's416-call/26-round Vision values remain separate historical evidence.

CPU aggregation and response/provenance checks are in `accuracy/snapshot-b-evidence.json` and `snapshot-b-service-detail.json`. The final direct comparison below is B's own fixed-input measurement; neither A's1.390969x nor a product of component ratios is substituted for it.

### Snapshot B: final direct comparison and resident state

Closed operation `b70872ee36f6457fb7e8313a34260fe1` uses222 freshly captured B-trajectory layer3/C1/T8/HQ records, one paired measurement each, profiling enabled. CPU range-reads only7,992 payload bytes (32-byte positions plus4-byte valid per record) verify actual KV windows19,962–21,049, all valid8. Corpus identity is `e397fd57bca0ad8ee29e734df8a29504463503dd47ce5cd801af401676275bf1`; it is distinct from A's357-record corpus.

The CLI baseline is builtin B, including its timed TL counter-prefix clear, and the candidate is the same validated initial-math control1d0c70de…. Compute old/B at every point first, then median:

| Final B paired result | Value |
|---|---:|
| Paired points / records |222 /222|
| B faster points |222 /222|
| Median pointwise old/B |1.389599x|
| Pointwise ratio range |1.090656–1.493536x|
| Initial latency median |331.872µs|
| B latency median |238.816µs|

The paired-ratio latency reduction is `1−1/1.389599`≈28.04%; the two latency medians remain separate statistics. B exactly reproduces all captured outputs and complete physical state. Initial control also preserves state, with finite mathematical-output differences (maximum peer absolute distance.125, worst-record relativeL2 .00307615); these are arm distances, not truth errors. Capture trajectory checks pass with empty differences. This total is measured directly, not composed. Different A/B corpora and trajectories mean their two total ratios are not an incremental A-to-B measurement.

The reviewable [B figure](artifacts/2026-11-19-gqa/gqa_block_initial_vs_final_native_b.svg), [222-point CSV](artifacts/2026-11-19-gqa/paired_points.csv) and [compact summary](artifacts/2026-11-19-gqa/summary.json) are included with [SHA-256 hashes](artifacts/2026-11-19-gqa/sha256.json). Full source scripts and actual-window proof remain in the local evidence archive. It uses222 single paired points and no uncertainty bands. A's original figure and714-point provenance are preserved unchanged. Independent parsing is `accuracy/final-b-direct/{summary.json,paired_points.csv}`.

After comparison, all810 runtime hashes are restored to B source key `df10c9f1f90bcc1576321fc635a9412bda0f12c62554080c725aad56db83db49`. Final Lab status is `ready`, no active operation, profiling enabled, session `7a3ae288579143448827096217bb302a`, engine PID2649195, B baseline df10c9f1…db49 and corpus e397fd57…bf1. The B engine/binary pin is separately recorded in `s3-baseline-complete.json`; prior A status is preserved rather than overwritten as if it were B. Final small receipts are `final-validation/final-lab-status-s3.json` and `final-lab-status.json`. The owner leaves that resident Lab ready and performs no further profiling.

## Finding

Observed: uniform42 BF16 retains the original error scale on the frozen actual inputs; FP32 partials improve every pre-gate row at the two full-head calls. Exact block changes preserve the chosen FP32 diagnostics and reduce paired block latency, including append FWHT reuse at1.02143x against the exact-combination control. The final snapshot B direct222-record comparison measures1.389599x native speedup over validated initial math on identical inputs, with B faster222/222 paired points; the earlier A357-record/714-point result remains historical. Snapshot A's native route is exercised in5,712 profiled blocks and preserves that pre-port candidate Astro trajectory. Snapshot B has separate222-record direct/capture evidence and400 native partial/reducer calls in its actual-image interval.

Inference: the repartitioning differences are consistent with expected rounding redistribution, supported by selected-input/state agreement and independent FP64 error. FP32 partials remove a material intermediate rounding source; late IFWHT is a separate performance placement opportunity. Neither statement makes the old implementation mathematical truth or establishes whole-model quality from operator error alone.

## Implications

Evaluate mathematical repartitioning against a common independent target and investigate error rather than requiring old bits. Preserve exact checks for indices, cache state, masks and layout-only changes. Native integration must budget FP32 storage in bytes and reset transient synchronization state inside the measured chain. Reuse the existing encoded FWHT registers only before quantization and preserve default caller boundaries. Keep the short-context transition cost and the final acceptance limits explicit.

## Limits and unknowns

Independent R64 coverage is finite and layer-local; natural-EOS content inspection is not a model-quality benchmark. Derived short windows and the40K bounds fixture do not establish actual short/long-prompt service behavior. The snapshot A same-input block comparison establishes that snapshot's cumulative block speedup. Snapshot A full-service own-trajectory TPS is mixed. Snapshot B increases own-trajectory TPS in all three requests, but generated work differs and Astro decode duration is longer; neither snapshot establishes a uniform fixed-work service speedup or whole-model quality.

Snapshot A native default routing,357-record capture/replay consistency, transient-workspace reuse, GPU checks, service/capacity/canary checks and the capped native Vision smoke are observed within their stated scopes. Snapshot B has its own successful build/GPU and completed service/capacity/canary/Vision receipts. Its final222-point direct comparison and restored resident Lab state are also recorded independently. Earlier A measurements remain attached to A. Unsupported shapes retain fallback and are not claimed to receive native acceleration. The cap24 HTTP limitation cannot be removed by calling aggregate histograms a chronological trace. The96-token Vision cap and lack of a task-quality benchmark remain evidence limits; neither is presented as a complete-quality result.

## Follow-ups

Completed receipts and reproducible analyses are retained under `.scratch/gqa-10h-2026-11-19/`. Future work may evaluate the strong QKV alternative with common encoded-K/V FP64 and original-encoder-on-new-input attribution, then compare it directly with selected B on matched inputs; its incomplete `035f…` run supplies no winner. The1.4–2.6K short-context transition remains another measured opportunity. A fixed-history logits comparison can diagnose near ties if useful, without treating different greedy lengths as a quality verdict. Further changes retain the existing [testing requirements](../agents/testing.md).
