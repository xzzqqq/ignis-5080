# C1/T8 GDN: joint projections and fused output preparation

- Kind: experiment
- Status: current
- Observed: 2026-11-21
- Last verified: 2026-11-21
- Scope: kernel / RTX 5080 Trellis C1/T8 GDN verify
- Related: [Trellis down reduction](2026-11-16-trellis-static-decode-and-down-reduction.md), [Kernel Lab](../user/contributing/kernel-lab.md)
- Superseded by: none

## Question

Can joint projection scheduling and fused gated-norm/output preparation improve
GDN verify without changing its numerical or persistent-state contract?

## Evidence

### Implementation

The GDN sublayer runs QKV and Z in one decode grid while keeping each weight's
input rotation, original K splits and accumulation order. The output preparation
combines 128-dimensional gated RMSNorm with Trellis input rotation. It retains
the intermediate BF16 rounding, produces the original FP16 fragment layout and
inverse scales, and adds the incoming residual directly in the output projection.

The native route reuses the existing model Trellis workspace. The two input
projections use disjoint arrival slots in its fixed synchronization prefix;
other payload bytes may be overwritten by subsequent MLP/head calls. No extra
persistent allocation or per-layer workspace reset is required. C1/T8 geometry,
narrow QKV/Z weights, no SM cap and sufficient existing workspace guard the
native route. Other shapes and legacy operator recording use the composed path.
The finite GDN Lab launcher and native route share the implementation.

### Candidate screening

Each initial screen used two actual block calls and five alternating comparison
pairs. Promising cases were repeated with fifteen pairs. Isolated block timings
are not model-wide speed estimates. The original GDN combination measured
1.0315x on layer 1 and 1.0331x on one layer-0 4-bit call; outputs were bitwise equal.
Reducing the fused epilogue from 128 to 64 threads gave a further 1.0105x in a
nine-pair comparison against the saved combination. 256 threads were flat.

A three-pair full-forward comparison on the frozen Astropy request measured
median decode 3695.608 -> 3664.313 ms, median paired speedup 1.00759x. Both arms
produced the same 889 output tokens, 180 rounds and full recorded trajectory.
This candidate used a private Lab workspace; the final native shared-workspace
route is validated separately below.

MLP alternatives did not establish useful gains: separate SwiGLU/rotation
finishing fell to about 0.3% on repeat; a 168-CTA mapping, 64-column CTAs and
uint2 tile loading regressed. Parallel gate/up within one CTA was bitwise equal
but measured 0.9926x with eight warps and 0.9058x with sixteen. Those variants
remain experiment artifacts and are excluded from the implementation.

Joint GQA QKV was about 0.8% faster on repeat, with max record relative L2
3.24e-5; a legal 60-CTA norm/gating MMA arm was about 1.2% faster with relative
L2 2.73e-4. Both are retained alternatives, not accepted numerical changes.
An N64 diagnostic of the 120-CTA cooperative gating schedule exceeded residency
on this 84-SM device (one resident CTA/SM). T8 conv specialization and standalone inverse-norm precomputation
were bitwise equal but flat in complete GDN replay.

### Final acceptance

`make check CUDA=1` passed CUDA/all-target compilation. The official GPU runner
passed all 132 CTests (including the new mixed-bit GDN workspace-poison test),
the four Trellis GPU suites and VRAM reload, with no skipped stages. The finite
segment regression passed in HQ, BF16 and NVFP4 configurations. It compares
native execution with independently recorded composed execution, requires exact
GDN output, and covers nonzero slots, masked tails and commit liveouts. HQ/BF16
also exercised the independently built candidate DSO.

Service follows the fixed coding-agent protocol: C1, two configured lanes,
HQ KV, DFlash2 seven drafts, 262,144 requested context, 1,024 prefill chunk,
Vision host, no prefix/retained reuse, loop guard off, natural EOS, one full
warmup and three measured responses per request. Four independent starts run
A/B/A/B. A is the composed-route server built after the shared decoder refactor
but before enabling native GDN dispatch; it is not a pristine historical-HEAD
binary. B enables the final native route. Their immutable binaries and effective
configs are retained. The earlier Lab comparison uses the original pinned
builtin as its separate control.

| Request | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s | Paired gain |
|---|---:|---:|---:|---:|---:|
| astropy | 241.482 | 242.176 | 240.285 | 241.958 | +0.29% / +0.70% |
| django | 224.082 | 225.228 | 223.574 | 225.059 | +0.51% / +0.66% |
| sympy | 229.584 | 231.286 | 229.478 | 231.232 | +0.74% / +0.76% |

All 48 responses, including warmups, agree after removing generated tool-call
IDs; input/output token counts and every speculative counter agree. These are
small, repeated same-direction gains, not a confidence interval or a universal
speedup. The derived [comparison](artifacts/2026-11-21-gdn/service-comparison.json)
retains the numerical values and counts.

The separate capacity pass reports exactly 131,072 input tokens, zero cached
tokens and one generated token before EOS. Canary passes its sanity and repeat
checks. A Vision request produces 96 tokens across 25 speculative rounds; this
is a capped functional smoke, not an image-quality benchmark. A streaming
request is closed early and a subsequent request completes normally. Server
shutdown leaves no GPU compute process. No generated tool calls are executed.

### Review follow-up

The native caller now supplies the `gdn.native_block` NVTX operator scope.
An instrumented builtin forward (`0a0d4fc9e5d84f5a866289a9b583c8a7`)
attributes all 239,620 kernels: 48 GDN layers contribute 8,640 native block
instances and 86,400 kernels, with zero unattributed kernels overall. This is
attribution evidence, not a replacement performance measurement.

The serial wide/capped projection fallback reuses workspace A for both calls;
workspace B's prefix is disposable payload and must never be interpreted as
initialized counters. The dirty-payload regression now covers either projection
at 5-bit, as well as narrow mixed-bit inputs. SM-cap1 exposed a separate capped
scheduling failure during this follow-up; disabling PDL on capped decode launches
removed that failure while leaving uncapped launch policy unchanged. Both cap0
and cap1 tests pass, along with the complete Trellis body test.

Model loading now prepares GDN kernels before capture, including the fused
`gated_rotation` kernel. CUDA/all-target compilation, the cold LAZY-mode finite
segment regression and 12 forward-timeline CPU tests pass. Private GDN entries
validate weights, geometry, alignment, workspace capacity and output parameters.
The earlier service performance numbers were not remeasured for these fixes.

## Finding

The retained exact GDN combination reduces isolated block latency and provides
0.29–0.76% higher decode throughput across the six service pairs. Independent
input transforms and split grouping can coexist with joint scheduling. Matching
the 128-dimensional gated norm and input rotation eliminates a materialization
boundary while retaining BF16 rounding.

## Implications

Use the guarded native route for eligible C1/T8 calls and the same implementation
for finite-block experiments. Keep arrival counters in the shared reserved
prefix; the payload cannot be assumed to survive intervening Trellis operations.
The original composed route remains the recording reference and fallback.

## Limits and unknowns

Raw source snapshots, immutable candidate identities, paired timings and errors
are retained in `.scratch/body-explore-2026-11-21/` and
`.scratch/body-deeper-2026-11-21/`. Remote acceptance receipts are in
`.scratch/body-accept-2026-11-21/`. The initial Lab baseline is rooted at
`a634633b49c1`; its configuration used HQ KV, two configured lanes, actual C1,
65,536 context and the fixed Astropy request. Service measurements use the
separate fixed coding-agent protocol and must not be compared by absolute time
with that Lab request. Normal boost was used; clocks were not locked.

The final service binary SHA-256 is
`0a468ec1f0735d1f4b0b8a54b7c1713e01b5c4ebdf170ef0933c33b3e928aeea`;
the final candidate DSO identity is
`a84e24e9d9dd26ff5141be45cdf0223172067a7946584f6781478184cf93faf1`.
The exploration's original 120-CTA cooperative gating attempt invalidated graph
capture; the later explicit residency check safely rejected the N64/120-CTA diagnostic.
The initial capture failure was not independently attributed to one cause.
Neither attempt is counted as a correctness pass. NFS stalled one initial GQA
recording; recording and compilation costs are excluded from kernel timing.

## Follow-ups

Further MLP/GQA numerical and scheduling alternatives remain in the scratch
reports. This result does not establish gains for C2, general prefill, other
cards or reduced-precision recurrent state.
