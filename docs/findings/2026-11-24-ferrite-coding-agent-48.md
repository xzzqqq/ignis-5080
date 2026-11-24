# Ferrite and Ignis on 48 frozen coding-agent turns

- Kind: experiment
- Status: current
- Observed: 2026-11-24
- Last verified: 2026-11-24
- Scope: serving / RTX 5080 C1 coding-agent decode
- Related: [Ignis results](2026-11-23-coding-agent-48-decode.md), [input manifest](../../crates/bench/fixtures/coding_agent_48/README.md)
- Superseded by: none

## Question

How fast do Ferrite and Ignis decode the next reply to the same long-context coding-agent requests on one RTX 5080?

## Evidence

Ignis `8495f3a0` used its production profile with packed DFlash2 projections and NVFP4 KV. Ferrite 0.9.2 at `e7aefef8b0a6ad59730f3ee1e08b1e3d7322b801` used its RTX 5080 preset, EXL3 3.0 bpw mix, 5-bit output head, FP4 prompt weights, 6-bit KV, one slot, MTP (8 drafts) and up to 47 copy rows. Its two model directories passed the upstream SHA-256 manifest. See the [Ferrite preset](https://github.com/BHCC2025/ferrite/blob/e7aefef8b0a6ad59730f3ee1e08b1e3d7322b801/docs/rtx-5080.md).

The same independent 48 requests and separate Astropy warmup used C1, greedy generation, thinking enabled with xhigh effort, an 8,192-token output cap, no prefix reuse and no loop guard. Ferrite's normalized tool schemas and chat-template options reproduced the ordered Ignis prompt token sequence for every request, checked using 64-bit fingerprints. Every timed request computed a cold prompt. Tools were offered and no generated calls were executed. Each arm ran once per task in a separate launch on the same card.

| Measure | Ferrite | Ignis |
|---|---:|---:|
| Median task decode tok/s | 183.21 | 261.71 |
| Output tokens | 53,630 | 57,357 |
| Output tokens / total decode seconds, tok/s | 199.36 | 280.33 |
| Tool-call finishes | 47 | 45 |
| Output-cap finishes | 1 | 3 |

Ferrite rates use the emitted token count before EOS and its engine's elapsed time from sampling/emitting the first anchor after target prefill to scheduler finish. This includes the final prompt chunk's MTP absorption and draft, all verify rounds, and the terminal round. A logging-only patch records existing scheduler timestamps at full precision. The Ferrite HTTP usage includes a terminal token for native EOS finishes; the output record keeps the HTTP count and adjustment. Ignis rates use its recorded final-prefill-to-scheduler-done window. Neither arm uses SSE arrival timing.

## Finding

Ignis/Ferrite decode speedup, geometric mean of paired task rates, was **1.440×** (95% stratified bootstrap interval 1.370–1.514×; 10,000 draws with replacement within the 12 length/stage cells). Ignis was faster on 47 tasks and Ferrite on 1.

| Group | Tasks | Ferrite median tok/s | Ignis median tok/s | Paired Ignis/Ferrite |
|---|---:|---:|---:|---:|
| explore | 16 | 183.38 | 286.42 | 1.505× |
| edit | 16 | 171.00 | 239.94 | 1.397× |
| debug | 16 | 187.61 | 252.84 | 1.421× |
| Length Q1 | 12 | 194.39 | 283.89 | 1.412× |
| Length Q2 | 12 | 192.99 | 270.30 | 1.457× |
| Length Q3 | 12 | 169.09 | 231.59 | 1.341× |
| Length Q4 | 12 | 157.56 | 255.50 | 1.561× |

The [paired results](artifacts/2026-11-24-ferrite-48/results.json) contain every rate, round count and finish reason. The Ferrite run completed its timed requests in 9.40 minutes; Ignis took 12.96 minutes. These wall times include prefill and naturally different output lengths. Raw responses and engine logs are in `.scratch/ferrite48-20261124-005231/`.

## Implications

The comparison measures the two shipping decode configurations and their realized output trajectories on repository exploration, edits and debugging.

### Public reference points (2026-11-24)

These independent measurements all use Qwen3.8-27B. Each row retains its source workload and hardware. The paired speedups above use only the two engines measured on the frozen 48 requests.

| Engine and configuration | GPU | Published decode, tok/s | Workload |
|---|---|---:|---|
| FLYX, UD-IQ3_XXS + MTP | 1× RTX 5080 16 GB | 225.9 / 161.0 / 105.8 | 1K / 128K / 256K actual input, greedy, thinking off, 512-token Python parser generation. [Benchmark](https://github.com/enwove/flyx/blob/main/docs/benchmarks.md) |
| llama.cpp, UD-IQ3_S + MTP3 | 1× RTX 5080 16 GB | 94.5 cold, 95.7–96.0 warm | 30K-token prompt, 98,304 configured context; short-prompt rate was 106.8–109.0. [Configuration and measurements](https://github.com/BFinn/5080-llm-configs/tree/main/models/qwen3.8-27b) |
| NInfer-5080 v1.5 | 1× RTX 5080 16 GB | 112.2 | Release sustained-decode benchmark, 131,072 configured context and KV capacity. [Release record](https://github.com/toddballinger/ninfer-5080) |
| SGLang + Escha-W2, MTP | 1× RTX 5080 16 GB | 109.2 | Single-user decode on the 2-bit model; 61.7 without speculation. [Model card](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2) |
| ExLlamaV3 1.6.0, EXL3 4.0 bpw + DFlash2 | 1× RTX 5090 32 GB | 274.6 | RigMark code, thinking off, five-run median. [Benchmark](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install#what-was-measured) |
| vLLM 0.28.0, NVFP4 KV + MTP | 1× RTX 5090 32 GB | 175.0 | Single-stream code decode in the project's one-card comparison. [Measurement matrix](https://github.com/adrienbrault/qwen3.8-27b-rtx5090#readme) |

## Limits and unknowns

Different quantizations, KV formats and draft algorithms can change the generated responses. A single natural generation per task gives no estimate of run-to-run timing variance. This workload measures inference speed; patch correctness was not scored.

## Follow-ups

None.
