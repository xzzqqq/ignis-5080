# Exact GQA replay, fixed reduction geometry and integer-cache limits

- Kind: experiment
- Status: current
- Observed: 2026-11-18
- Last verified: 2026-11-18
- Scope: kernel / RTX 5080 HQ 27 small-T attention and split reduction
- Related: [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md)
- Superseded by: none

## Question

Which exact changes reduce HQ 27 attention latency on the RTX 5080 without
changing KV format, quantization, softmax/split order or precision? Can a
small replay distinguish codec, occupancy and memory-layout hypotheses?

## Evidence

The original exclusive-card full-forward profile attributes 3.3613 ms/round
(15.21% of GPU kernel time) to main GQA attention. The fixed Astropy request
has 19954 input tokens,913 generated tokens and 187 DFlash 2 rounds under the
original 8192 cap, context 65536, lanes 2, chunk 1024 and HQ-E8-2 b. Trellis candidate
`a7d2044a…` is reloaded in both arms. The pinned native engine is
`9fcaa0753c0e545ea4d285f1e7b66b78f97789f23ef4d7557bdba9cad158e54f`.
Source baseline is `ff0989a0c84cfd54688c729dc52b6e700dbf218b`, CUDA13.0.88,
SM120a. Raw artifacts and frozen requests are in
`.scratch/gqa-10h-2026-11-18/` and `.scratch/experiment-results/`.

The original Nsight Compute sample at round 10/layer 3 is 147.808 us, with DRAM
94.98 GB/s (10.04% of peak),255 registers/thread and 48128 B static shared plus
1024 B driver shared. It is not at its memory roofline. Both registers and
shared limit residency to 2 CTAs/SM. At 19992 keys only 168 of the 340 graph-capacity
CTAs do attention work; the rest write neutral partials. A generic 340-block
tail estimate therefore does not describe 168 equal active blocks.

A read-only issue-budget reanalysis uses the original U1 invocation's
57,825,672 issued warp instructions, 84 SMs, the receipt's four-slot/SM
normalization and measured 2.786305 GHz clock. Its optimistic fixed-stream
issue budget is about **61.77 us**, versus the measured 147.808 us. This is
conditional on that instruction stream and issue model, not a universal SOL
or an attainable latency prediction for U2/cache variants. The DRAM-only
estimate is about **14.62 us**: the export lacks a byte-total counter, so
traffic is inferred from the multi-pass rate times duration before dividing
by nominal 960 GB/s. It is not an exact same-pass byte measurement. The
scheduler's 57.07% no-eligible fraction is not removable time; no complete
dynamic dependency graph was captured to quantify the remaining critical
path. See NVIDIA's [counter hardware model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)
and [RTX Blackwell bandwidth specification](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
Raw inputs, cross-checks and assumptions are preserved in
`.scratch/gqa-10h-2026-11-18/issue-sol-analysis.json`.

The standalone exact replay uses the committed real KV fixture under its true
head/key/role seeds, with deterministic generated Q/new KV. This is not an
Astropy activation capture. It restores code/meta K/V, side K/V, ring-valid and
partial m/l/acc outside timing; a 256 MiB scrub is also outside timing. The
independent partial oracle is `IgnisThroughput=false`. A cold full-partial
replay and warm reducer graph replay have different cache conditions.

| Change | Exact coverage | Measured result |
|---|---|---|
| Rice steady-loop unroll 2 | Original partial/cache byte oracle and codec edge tests | Append 20 K about 145.4 us versus 147.5 us original;70 K about 471 versus 481 us; dynamic loop/total instruction reduction, same resources. |
| Cached reducer weights, original 256 thread/D64 |160 final-output byte gates, including alias/nonfinite/underflow | Warm 20 K 16.24→11.8 us; real Astropy U2+weights versus original gives 0.488%/0.629% decode gains in two pairs. |
| Fixed 256 thread/D256 cached reducer |190 final-output byte gates, original generic reducer unchanged | Warm 20 K 5.68–5.77 us,70 K 8.70–8.75 us,T1 about 4.17 us; same per-output recurrence and 256-leaf tree. |
| VO192 with 2 CTA launch bound | Original exact partial oracle |167 registers and zero spill do not give a robust latency win. More nominal occupancy alone is insufficient. |
| QK column split over 6 warps |15-case stages and enhanced codec gates |20 K improves slightly,70 K regresses to about 485 us versus U2471;168 registers/36 B spill stores and loads. Rejected. |
| Coalesced side-row bit copy |18 partial cases and enhanced codec gates | Some frontier cells improve, but process variation and mixed Y 4 results do not justify adding it to the ordinary candidate. |

The fixed reducer's 256-thread contract is explicit and shared with its
launcher. The original generic reducer stays intact for other geometry/dtypes
and as the independent oracle. The owned adaptation caches exactly the same
FP32 exponential weight once per split, preserves the distinct `l>0` versus
`l<=0` NaN predicates, and retains an original-expression fallback above the
shared weight capacity. It never skips `0*Inf/NaN` based on weight value.

The 190 gates include real original partials plus neutral, nonfinite, underflow
and subnormal cases; aliasing gate/output; masks, offsets and multiple lanes;
and 257 splits, where leaf 0 must visit both 0 and 256. Every final BF16 output
byte, including untouched offset columns, is compared to the original reducer.

The final ordinary candidate is **U2 plus the owned HQ27 256-thread/D256
reducer**. It has no persistent cache, extra global allocation or runtime
experiment switch. The frozen source identity is
`849b51dfb88d260b5816c48ddb02ff0a3da80bd0a23ee2839bbe5feb17f57e8d`;
its validated native engine SHA256 is
`61c300dd2abe1148a75836221d0fd4eb8e2ee9d7a42e8adf4e2b23714bc71e32`.
The unmodified pinned engine remains `9fcaa075…`; both arms load the same
Trellis candidate and use the same prepared requests/settings.

Fresh balanced original/candidate/candidate/original Sessions give these
ordinary-candidate decode reductions:

| Fixed request | Pair 1 | Pair 2 |
|---|---:|---:|
| Astropy | 0.715% | 1.134% |
| Django | 1.189% | 1.430% |
| Sympy | 0.802% | 1.057% |

The comparison contains 48 runs: 36 measured runs and 12 warmups, with six
measurements per arm/request. All runs match emitted tokens, finish/stop and
reasoning state, **prefill** logits hash, and round position/emitted/extent
paths. This ordinary harness does not copy or hash complete decode logits;
that stronger readout belongs to separate cache diagnostics below. The
original 8192 output cap and natural-EOS behavior remain unchanged. These
are direct real-engine Kernel Lab calls, excluding HTTP/server scheduling.
Receipts: job `gqa10h-ordinary-validation-v1`, `comparison.json`,
`measurements.json`, `identity.json` and `validation-summary.json`.

The same frozen ordinary candidate completed final native/CUDA checks in
2800.7 s for the complete validation job, including builds and paired forward:

| Check | Result |
|---|---|
| Laptop workspace Rust | 3068 passed, 0 failed, 6 existing ignored |
| Linux workspace Rust | 3069 passed, 0 failed, 6 ignored; 252 result bins |
| Laptop Python | 146 tests, OK, 11 recorded skips |
| Web | 1147 tests passed; production build passed |
| Vendor manifest/patch verification | passed |
| Linux CUDA workspace all-target compilation | passed |
| Related native GPU gates | passed |
| Standard native CTest | 131/131 passed |
| Standard Rust GPU suites | 6 tests passed across five suites; 0 failed, 0 ignored |

The Python skips are explicit: seven Linux-specific transport/daemon tests
on the Laptop and four opt-in exclusive-GPU Lab integration tests. A separate
Linux-only CPU Python continuation returned exit 0; do not describe the
Laptop run as skip-free or conflate its skips with the standard GPU profile.
The GPU profile used the existing Trellis artifact and NVFP4 projection
control. Compiler caches were RAM-backed; direct-I/O temporary containers
used the separate tested NFS TMPDIR. Local and remote Rust counts are
reported separately rather than normalized to one number.

A second receipt, `gqa10h-ordinary-op-profile-v1`, confirms the mechanism at
one instrumented **late Astropy round 180 (zero-based), input position 20842,
visible window 20850**: 44 active splits/176 active CTAs within grid capacity
340. This is not the replay window 20864 or the earlier round-10 capture.

| Kernel family, 16 calls per arm | Pinned original total | Ordinary total |
|---|---:|---:|
| HQ partial | 3755.209 us | 3659.525 us |
| Split reducer | 256.390 us | 85.413 us |

The partial grid/block remains `(4,85,1)/(128,1,1)`; the reducer changes from
`(24,4,8)/(256,1,1)` to `(24,1,8)/(256,1,1)`. Reducer register/shared allocation
is 40/1024 B versus 44/1364 B. The profiler provides family totals only,
without reliable per-layer ordinal attribution. One instrumented round per
arm supports the timing mechanism; the repeated unprofiled paired requests
above remain the statistical performance evidence. Both profile requests
retain 913 emitted tokens and the 187-round natural-EOS path.

The final restored profiling Lab binds a new ordinary-source baseline
`cf8f1b29ed93e359091fd6d077dacb5ea362aa2f5318b39336053e212737dc02`.
One same-session full Astropy profile reloads the same `a7d2044a…` Trellis
candidate (SHA256 `a8703289…`) and captures all 187 rounds with graph-node
provenance: 2992 HQ partial calls total 568774.572 us and 2992 owned reducers
total 16481.228 us, or 3.129710 ms combined per round. Round 10 partial mean
is 135.069 us and round 180 is 232.330 us. This is another single instrumented
observation, not a fresh paired speed estimate. The new session initially
has no stored forward reference; an offline comparison against the frozen
pinned-original JSON confirms identical tokens, prefill hash and every round
position/emitted/extent/graph-launch path. It does not add full decode-logit
coverage. That session returned ready with profiling enabled. It was subsequently released
for the bounded continuation diagnostic and restored to the same accepted
baseline with profiling enabled; the full-profile receipt retains its original
session identity.

Q-rotation precomputation is another exact but **unadopted replay-only**
experiment (`gqa10h-q-precompute-v1`, binary `16bac634…`). One binary toggles
precomputation at runtime, checks the original Q bytes remain unchanged, and
passes the original 18-case partial/write-set gate. Timed events include the
precompute launch plus the complete partial; both arms reserve the same
96 KiB C1/T8 scratch outside events. Four-round medians change from
145.408 us to 142.352–143.360 at 19992, from 212.992 to 208.896 at 20480,
from 235.520–237.568 to 231.424–231.456 at 20864, and from 471.040 to
464.512–464.896 at 70000. The 53.0 s job measures fixed-fixture precompute plus partial only; it is
not engine integration or a reason to add it to ordinary code.

For the historical cache comparisons, **R64** means the experimental
64-physical-thread reducer retaining 256 virtual leaves; it is not the final
ordinary 256-thread/D256 reducer. **A256/D64** means the earlier cached-weight
256-thread/D64 reducer, and is not the unmodified pinned generic reducer.
A cache+R64 comparison against U2+A256/D64 combines two changes; only the
separate R64-only cache control isolates the cache effect.

An isolated native integer cache stores exact E8 coordinates as
U4(Y-row_min)+raw norm/escalation metadata, without changing the original HQ
stream or FP reconstruction. With 512 MiB persistent budget, capacity 30784 and
Astropy prefix 19936, a native static cache+R64 improves decode 2.715%/2.846%
against U2+A256/D64. Against the same-binary R64-only no-cache control, fresh paired gains are
1.916%/2.008% on Django,0.343%/0.678% on Sympy,2.201%/2.411% on Astropy.
All three preserve actual BF16 decode-logit hashes, tokens and round paths.
The static cache and R64 remain scratch, single-active-owner diagnostics; neither is part of the ordinary candidate.

A lag 128,32-row incremental diagnostic also passes 19 real-request gates,
including 8 allocation/event fault points and slot/prefix reuse. Maintenance is
inside decode round wall time. Its final paired changes versus static cache, both using the new fixed 256-thread/D256 reducer,
are negative on Astropy (−0.921%/−0.533%), negative or negligible on Sympy
(−0.482%/−0.019%), and mixed on Django (−0.052%/+0.190%). It is rejected.
Native maintenance counters exclude small wrapper-return overhead; complete
round wall includes the whole call. No-op updates have no CUDA calls.

A same-binary prefix-lag replay preserves 18 exact cases while changing only
runtime prefix input. Lag 512 does not robustly beat 0/128 at windows 18080,
19992,20480,20864,70000. Even full-prefix cache rises from about 119 us at 19992 to
170–172 us at 20480 and 182 us at 20864. This rejects the claim that an uncached
new-history tail alone explains the frontier growth.

A scheduling-only diagnostic (`gqa10h-cta-order-v1`) then kept the original
logical splits, per-split FP order, grid capacity and full mutable write-set,
while varying identity, reversed-active, head/tail-interleaved and head-swizzled
task mapping. A separate unmodified-U2 mode calibrated observer-kernel costs.
All 110 exact gate executions (22 cases × five modes) and 28 instrumented
trace gates passed. Four alternating timing rounds contain 7200 uninstrumented
samples across nine windows and five modes.

No order gives a stable >5% late-window win. Uninstrumented identity medians
are 145.408 us at 19992, 212.992 at 20480, 243.712 at 20640/20641 and 235.520
at 20864. Reversal improves 20480 to 208.896 and 20640/20641 to 235.520 us,
but worsens 20161 from 147.456 to 151.552 and 70000 from 471.040 to about
483–485 us. No permutation is added to the ordinary production candidate.

The independent instrumented identity traces show a delayed-CTA tail:

| Window | Active CTAs | Late active CTAs / logical task | Late start after first CTA | Diagnostic total span |
|---|---:|---|---:|---:|
| 19992 | 168 | none; 84 SMs each run two active CTAs | — | 151.872 us |
| 20161 | 172 | four / split 42, one tile | 138.720–139.648 us | 159.584 us |
| 20480 | 172 | four / split 42, ten tiles | 138.272–139.904 us | 227.680 us |
| 20640 | 172 | four / split 42, fifteen tiles | 139.072–139.168 us | 261.568 us |
| 20864 | 176 | eight / splits 42–43; critical split 42 has fifteen tiles | 141.024–141.600 us | 255.872 us |

This is direct evidence of delayed active work in the diagnostic binary,
consistent with crossing 84 SM × two resident CTAs and with a growing tail
split. It does not establish exact uninstrumented scheduling or a universal
CUDA launch-order guarantee. At 19992, head-swizzling changes every SM from
one to two logical KV heads in the trace, while all four uninstrumented timing
medians remain 145.408 us. Simple head affinity is therefore not sufficient
to explain the latency cliff on this fixture.

Trace and timing are distinct binaries: the diagnostic adds globaltimer/
clock64/SMID stores and a CTA-exit barrier. Append spill stores/loads rise from
224/240 B to 236/312 B; both report 255 registers, 48128 B static shared and
two resident CTAs. Diagnostic spans above are first-thread entry through the
all-warp exit barrier, not CUDA-event kernel latency and not speedup samples.
Cross-SM ordering uses globaltimer; clock64 is used only within an SM. Raw
64-bit timestamps are differenced as integers before conversion to units.

The final bounded repartition diagnostic (`gqa10h-partition-bound-v1`,
66.2 s) measured normal, declared-42 and declared-84 partitions with the same
85-split grid capacity. The original false partial and accepted reducer
remained the reference. Only candidate active-count metadata changed; the
original partial exact assertions and nonzero failure exits were retained.
All code/meta/side/ring bytes stayed identical, including repeated replay.
Normal mode passed all 18 exact cases. Changed partitions failed the original
partial-byte contract and remain **unaccepted and non-production**, irrespective
of their timing. No new tolerance was set.

| Window / declared→nonempty splits | Partial median range | Partial+reduce median range | Final BF16 elements different /49152 | Max absolute / relative L2 |
|---|---:|---:|---:|---|
|19992 /42→42, all three calibration modes|147.456 us|155.648 us|0|0 /0|
|20864 /normal44→44|239.456–247.488 us|247.648–255.680 us|0|0 /0|
|20864 /42→41|149.536–153.616 us|157.712–161.808 us|17623|0.0625 /0.00156023|
|20864 /84→82|161.776–165.888 us|172.000–176.128 us|17581|0.0625 /0.00151492|
|70000 /normal85→85|473.088–474.752 us|483.328–485.024 us|0|0 /0|
|70000 /42→42|454.272–454.304 us|462.464 us|13848|0.0625 /0.00134273|
|70000 /84→82|470.656–470.784 us|480.896 us|12306|0.125 /0.00130086|

The override policy applies only at20160<window<=70000: labels42/84 at19992
both retain normal42 as a calibration, not a forced84 result. Ranges cover
four alternating rounds, 1440 partial samples and the corresponding1440
partial+reduce event-chain samples. The diagnostic gate is BF16+Inf so
sigmoid(gate)=1: these are normalized-attention BF16 differences against the
unchanged reference, not actual-model gate/logit or model-quality measurements.
All reference/candidate49152 outputs per cell are finite, with zero NaN/Inf.
Declared counts differ from actual nonempty work; the conservative PageIds
bound is at most28 versus allocated128, not an observed28-page split.

A final bounded raw-FP32 continuation diagnostic
(`gqa10h-checkpoint-resume-v1`, 154.0 s) preserves the original logical splits.
The first plain same-stream launch checkpoints only split 0/1 across four
KV heads per batch; the second full-grid launch resumes those jobs. Each
checkpoint contains the original 96 participating threads' 128 accumulator
and four m/l floats, 50688 B/job, stored component-major without arithmetic.
Append runs only in the first launch; Q is reconstructed from the original
read-only BF16 input, K is restaged at the resume tile, and final BF16/IFWHT
runs once. All selected flags are initialized in the first timed launch,
including no-handoff and early-return paths. Resume never consumes flags
initialized only by the timing harness.

All 184 exact executions (23 shapes × eight modes) pass unchanged partial,
code/meta/side/ring and Q-byte gates. Cuts 0/1/8/14 and a terminal boundary are
covered alongside original U2, same-candidate off and off+empty-resume.
Cut 8 triggers eight checkpoints at the three long timed windows and zero
at the 40-key control. The 3840 samples cover four alternating rounds:

| Window | Original U2 | Candidate off | Off+empty B | Cut-8 A+B | Terminal A+B |
|---|---:|---:|---:|---:|---:|
|40|34.816|34.816–35.760|36.864|36.864–37.280|47.104|
|19992|145.408|143.360|145.408|202.752|155.648–155.680|
|20864|237.568|237.568|239.616|249.856|245.760–247.808|
|70000|471.040|470.672–471.040|472.688–472.768|581.216–581.312|480.960–481.280|

Values are ranges of per-round median microseconds, with visibly discrete
measurement plateaus. Terminal minus matched off+empty is 6.144–8.192 us at
20864, 10.240–10.272 us at 19992 and 8.256–8.592 us at 70000. These are costs
of this complete two-launch framework, including Q reconstruction, final
writes and global serialization, not isolated copy times or a measured
persistent-scheduler gain. B still launches the entire `(4,85,C)` grid, with
empty CTAs; the eight selected logical jobs are not guaranteed to be the
first eight hardware workers. Both kernels use 255 registers, 48128 B static
shared and two resident CTAs, but checkpoint control changes stack/spill:
240 B stack and 128/156 B spill stores/loads versus U2's 336 B and 224/240 B.
The same-candidate off arm is therefore essential. No continuation code
enters ordinary production.

A subsequent limited **single-kernel static drain** prototype
(`gqa10h-static-drain-v1`, 81.2 s) tests actual task reassignment without
changing partitions or arithmetic. It enables only runtime C1/T8, full-valid
(null validity pointer or at least eight valid columns), width 8/offset 0,
168 < active jobs <= 176 and more than eight tile units per split. Templates
remain MultiBatch=true/Masked=true. For R=active jobs−168, the first R logical
jobs checkpoint at tile 8, then their physical CTAs execute reserved late jobs
168+slot. The original late physical CTAs skip; truly inactive tasks retain
original neutral writes. Other workers and producers completing their late
job claim published continuations with device-scope acquire/release and CAS.
Every state-writing thread fences before the block barrier and ready release;
every restoring reader acquires. There is no global wait, and each producer
also drains its continuation so early-exiting consumers cannot strand it.

A tiny initialization kernel clears eight ready/claimed slots **inside the
measured chain**. The state pool is 405504 B plus 64 B flags for C1, and shared
allocation is 48144 B including a selector (original 48128 B); registers stay
255 and residency two CTAs. Stack/spill changes to 304 B and 196/256 B
stores/loads, so a same-candidate off+init control is retained. All 81 exact
executions (27 shapes × three modes) pass original partial/cache/ring/Q-byte
checks and expected ready==claimed counts. The extra gates include eligible
Cached/full-valid-pointer inputs, partial-mask fallback and a 2760-key A44
case whose two tile units correctly prevent scheduling. Original references
and coverage are retained, not replaced by the new task body.

Four alternating rounds contain 4800 full-chain samples:

| Window | Original U2 | Same candidate off+init | Schedule on+init |
|---|---:|---:|---:|
|40|34.816|40.960|40.960|
|2760|59.392|63.488|63.488|
|19992|145.408|151.552|153.600|
|20160|141.312|147.456|149.488–149.504|
|20161|145.408–147.456|155.648|165.888|
|20480|210.944–212.992|223.232–223.248|198.656|
|20640|243.712|258.032–258.048|208.896|
|20641|243.712|256.000|208.896–210.944|
|20864|235.520|247.808|204.800|
|70000|468.992–470.608|495.616–497.296|495.456–497.664|

Values are per-round median ranges in microseconds. The exact late-window
mechanism is supported: 20864 improves about 13.0% against original U2,
20640 about 14.3%, and 20480 about 5.8–6.7%. But the small late tail at 20161
regresses, as do early and 70K fallback controls. Off+init already exposes
substantial compilation/control-flow and initialization costs; on/off alone
is not the production gain. This version is **not adopted as a replacement**,
was not migrated to a real forward request, and remains a bounded scratch
prototype. It establishes an exact scheduling opportunity rather than an
accepted general policy or end-to-end speedup. Immutable raw output and
`.scratch/gqa-10h-2026-11-18/phase22-static-drain/audit.json` retain all arms.

The final wrapper experiment (`gqa10h-fallback-specialize-v1`, 134.1 s)
keeps the old phase22 kernel in the same binary and adds an independent
constant-phase fallback call site with the original physical head/split.
It also toggles a geometry-based guard: if the first reserved late job has
at most eight tiles, retain the original task mapping. Shared storage remains
single and the task FP/atomic protocol is unchanged. All 135 exact executions
(27 shapes × five modes), expected ready/claimed counts and 8000 samples pass.
The guard correctly suppresses continuation publication at 20161. All init
kernels are timed; original U2 is still the separate one-launch calibration.

|Window|Original U2|Old phase22 on|New off+init|New guard off|New guard on|
|---|---:|---:|---:|---:|---:|
|40|34.816–34.832|40.960|36.864–38.880|36.864–36.880|36.864–38.720|
|2760|59.392|63.488|63.488|63.488|63.488|
|19992|145.408|153.600|153.600|153.600|153.600|
|20160|141.312|149.504|149.504|149.504|149.504|
|20161|145.408–147.456|165.888–165.920|155.648|172.032–172.064|155.648|
|20480|210.976–212.992|198.656|225.280|204.800|204.800|
|20640|243.712|208.896|258.048–259.072|217.088|217.088–218.128|
|20641|243.712|208.896–210.944|258.048|219.136|219.136|
|20864|235.520|204.800|249.856|210.944|210.944|
|70000|468.992–469.008|495.616–497.664|497.344–497.664|497.344–497.664|497.312–497.664|

These are ranges of four per-round median microseconds. The constant fallback
call site does not solve the large-window overhead: new off+init at 19992 is
153.600 us versus original 145.408, and 70K remains about 497 us versus 469.
The short-tail guard removes part of 20161's scheduling cost (172.032–172.064
to 155.648 us), but still loses to original 145.408–147.456. New scheduled
20864 is 210.944 us, slower than the same-binary old phase22 204.800.
Only the tiny 40-key fallback clearly improves versus the old wrapper.
The new kernel therefore remains **rejected/unadopted**, not a repaired
production policy. Registers/shared/stack stay 255/48144/304 B; compiler
static spill stores/loads rise to 340/368 B versus old 196/256 B. Those counts
combine compiled paths and do not measure dynamic fallback traffic. No new
profile or real-forward claim is inferred. The full five-arm audit is
`.scratch/gqa-10h-2026-11-18/phase23-fallback-specialize/audit.json`.

Separately, the ordinary candidate completed service capability acceptance in
`gqa10h-service-acceptance-v2` (279.883 s): the existing fixed three-request
protocol, exact131072-token capacity leg, canary, one Vision TTFT smoke and
observed tool-response structure checks passed. All12 observed function-call
argument strings parsed as JSON; no tools were executed. This continuation
reused the same source tree/server/bench binaries after correcting only the
controller metadata URL from `/config` to `/v1/config`. Server SHA256 is
`ffb6d870518bdb0e4fe73d5a9ccb9775c425b32ae67623b70525f72d6c4cd00e`;
bench SHA256 is
`d16ab4188c686d51b289556bc8ed799c6d90d5b14382c8188d159b762bda3270`.
The service run used context262144, lanes2, chunk1024 and DFlash2 draft7;
that differs from context65536 in the paired Kernel Lab measurements.
These are candidate-only capability/absolute-metric checks, not a live/live
service speedup comparison. The speed evidence remains the balanced engine
pairs under their stated configuration. The service daemon was stopped and
the GPU released before the owner's final resident-Lab restoration.

## Finding

Reducing repeated work and fixing the actual reducer launch contract yields a
small exact implementation. Register/shared counters and bank-conflict shares
help select hypotheses; they are not elapsed-time fractions. The rejected
within-word layout, seed-cache, VO/QK and lag experiments must not be treated as
proof that every shared layout, cache or pipeline design is ineffective.

The independent original partial and reducer oracles remain essential. Binary,
source, runtime-input and sample identities are recorded when generated, and
comparison code joins those identities rather than inferring arm order.

## Implications

The ordinary candidate is U2 plus the owned HQ 27 fixed 256/D256 reducer, with no
additional global allocation or user configuration. Its native, CUDA and
real-engine paired validation completed separately from memory-consuming
cache prototypes. No experimental cache or R64 dispatch entered ordinary code. Native Kernel Lab still has
no general GQA hot-swap/replay API; the standalone GQA harness must not be
presented as Trellis replay or as actual activation replay.

## Limits and unknowns

The frozen ordinary candidate completed the native/CUDA and direct-engine
checks above, and the separate service capability continuation passed. No
live/live service speedup was measured. The CTA traces support delayed work
in their instrumented binary, not a complete causal attribution for the
uninstrumented kernel. Repartition timing does not override its failed exact
contract, and normalized-attention differences do not establish model quality. Cold
partial and warm reducer times are not directly comparable. The static cache
budget is persistent: its initial full reference buffer adds about 126 MB during
construction, and driver/allocator granularity is not a measured peak. No
cache is enabled by default or integrated into general service page lifetimes.
The incremental GPU event subregion excludes stats reset/publication copies;
the primary decode wall includes all those operations and synchronization.

## Follow-ups

Retain the frozen source/binary audits and final check results with the ordinary
patch. Treat any cache adoption as a separate explicit memory/lifecycle choice.

A possible future exact-preserving direction is to suspend complete logical
splits at tile boundaries, save/restore their FP32 m/l/acc state bitwise, and
let persistent CTAs resume them while retaining each split's original tile
order and only one final BF16/IFWHT rounding. The two-launch diagnostic verifies exact raw-state restoration, and the
limited static-drain prototype verifies one task-reassignment scheme. A
general adaptive scheduler and real-engine adoption remain unimplemented
and unverified.
Each48×256 FP32 accumulator state alone is48 KiB, with extra global/L2 traffic,
Q reload/reconstruction, acquire/release ownership, append visibility and
correct PDL completion requirements. It is not an accepted substitute or an
additional optimization from this task.

A CPU-only continuation cost model makes that follow-up conditional. An
8-tile quantum at window 20864 would suspend 172 long jobs once. Saving
49,152 B accumulator plus 1,536 B conservative per-lane m/l state, restoring
Q and discarding one code prefetch gives about 22.06 MB extra service traffic
per partial, before queues and additional computation. At an assumed
500 GB/s effective service rate that alone costs about 44 us, leaving only
26 us extra overhead against a hypothetical 70 us recovered tail. Neither
bandwidth nor recoverable time is established for a continuation kernel.
The exact tile-boundary save/restore replay above supplies an initial
measurement instead of relying on this traffic model; it does not establish
the benefit of a general scheduler. The later static-drain replay supplies
limited actual scheduling measurements, including its regressions. Full cost assumptions and append/epoch/PDL
obligations remain in `.scratch/gqa-10h-2026-11-18/checkpoint-cost-review.md`.

The targeted eight-checkpoint case transfers only 811008 B of raw state
(write plus read), rather than checkpointing all 172 long jobs. Including
Q reload and one discarded K prefetch gives about 1.026 MB logical service
traffic. The larger 22.06 MB model remains the cost of checkpointing all
172 jobs once, not the cost of the eight-job prototype. Measured terminal
deltas must not be divided into these bytes to infer bandwidth, because they
include other work and different scheduling.
