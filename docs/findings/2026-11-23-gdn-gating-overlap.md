# GDN verify gating on a side stream

- Kind: experiment
- Status: current
- Observed: 2026-11-23
- Last verified: 2026-11-23
- Scope: kernel / RTX 5080 Trellis C1/T8 GDN verify
- Related: [joint projection and gated rotation](2026-11-21-gdn-joint-projection-and-gated-rotation.md), [coding-agent benchmark](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

How much decode time can the BF16 gating projection overlap with the QKV/Z projections and convolution after input RMSNorm?

## Evidence

The GDN segment has two consumers of the normalized input. The gating projection produces FP32 gates and beta in its own scratch planes. The other branch produces Q, K, V and Z. The model records a fork event after RMSNorm and joins the gating stream before recurrence. All kernels retain their original arithmetic and output layouts.

Kernel Lab used three actual layer-1 calls from the fixed Astropy request, HQ KV and fifteen alternating pairs per call. Starting gating immediately after RMSNorm saved a median 4.54 µs per block (45/45 faster). Starting it after joint projection saved 3.10 µs (44/45 faster). The outputs, four replay record planes and checked state bytes matched the serial reference. Three paired Astropy full forwards saved a median 0.258 ms per round. Each produced 889 tokens in 180 rounds with matching logits hashes and emitted/extent traces.

Production A/B compared a release built from main at `2e7e73ba` with `54456237` on the same RTX 5080. Both used the configured Trellis artifact with packed DFlash2 gate/up, NVFP4 KV, C1, two configured lanes, seven drafts, 139264 requested context, 1024-token prefill chunks and host Vision. Prompt and retained reuse were disabled. Each request ran to its natural finish once for warmup and three times for timing.

| Request | Input / output tokens | Rounds | Control tok/s | Parallel tok/s | Decode gain | Prefill ms, control → parallel |
|---|---:|---:|---:|---:|---:|---:|
| Astropy | 19,954 / 1,108 | 212 | 269.466 | 272.963 | +1.30% | 9,263 → 9,318 |
| Django | 22,420 / 873 | 172 | 259.555 | 262.498 | +1.13% | 10,630 → 10,682 |
| Sympy | 16,717 / 659 | 126 | 270.019 | 273.592 | +1.32% | 7,737 → 7,773 |

All 24 responses across both arms finished with tool calls. Their choices, including tool arguments, matched after removing generated call IDs. Input and output counts and every reported speculative counter matched for each request. Prefill medians were 0.47–0.60% higher on the candidate start.

## Finding

Starting gating after RMSNorm improved production decode throughput by 1.13–1.32% on these requests. The shorter overlap window in block replay recovered less time.

## Implications

Use the model-owned branch for the finite C1/T8 GDN verify call. Join before recurrent record so every layer can reuse its workspace and the same pair of events.

## Limits and unknowns

The production comparison uses one server start per arm with normal GPU boost. Its rates cover these three requests. The experiment does not resolve how much of the overlap occurs inside the projection tail or the convolution.

## Follow-ups

The Kernel Lab screen and production acceptance receipts are retained in the measurement checkout under `.scratch/gdn-quick-20261123/` and `.scratch/gdn-prod-20261123/`.
