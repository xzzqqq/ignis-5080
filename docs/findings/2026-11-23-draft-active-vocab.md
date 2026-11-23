# Skip inactive Trellis blocks in the DFlash2 draft head

- Kind: experiment
- Status: current
- Observed: 2026-11-23
- Last verified: 2026-11-23
- Scope: RTX 5080, Qwen3.8-27B, single-request DFlash2 decode in Kernel Lab
- Related: [Static shortlist A/B](2026-09-24-upstream-quick-wins-ab.md), [fork baseline](2026-11-03-coding-agent-5080-baseline.md)
- Superseded by: none

## Question

How much C1 decode time can the drafter save by reading only active vocabulary blocks, after accounting for changes in the number of speculative rounds?

## Evidence

The full draft and verify heads each read roughly 636 MB of Trellis weights per round. This Kernel Lab prototype reduces the draft read by selecting output blocks from the existing head. It enables the first 256 blocks plus every 128-row block represented in the prompt or committed output. Verify still scans all 1,940 blocks.

Each CUDA thread block handles one output block. An inactive block writes negative infinity to its logit positions and exits before loading its weights. Active blocks use the original rotation and token IDs. The existing top-16 selector reads the dense output. The prototype uses a single device-wide 244-byte bitmap, which the host uploads when a newly committed token opens a block. It requires one active request and a single copy of the head weights.

### Decode results

The same release binary ran full / active / active / full. Each engine received a warmup. The three fixed coding requests ran to their natural stop under greedy decoding with NVFP4 KV and seven draft tokens. Clocks were unlocked. Kernel Lab decode time includes draft, verify, commit and bitmap updates. Rates below pool output tokens and decode time over the two runs in each arm.

| Request | Final active blocks | Output tokens | Rounds, full → active | ms/round, full → active | Decode tok/s, full → active | Throughput gain, paired runs |
|---|---:|---:|---:|---:|---:|---:|
| Astropy | 422 | 1,108 | 212 → 215 | 19.377 → 18.861 | 269.73 → 273.24 | +1.19% / +1.41% |
| Django | 544 | 873 | 172 → 173 | 19.519 → 19.039 | 260.03 → 265.04 | +1.88% / +1.97% |
| SymPy | 432 | 659 | 126 → 126 | 19.304 → 18.772 | 270.94 → 278.61 | +2.80% / +2.86% |

The three requests produced 2,640 tokens per run. Full-head decode took 9.890 and 9.905 s, compared with 9.714 and 9.715 s for the active head, a **1.89% combined throughput gain**. Astropy required three additional rounds and Django one. A separate synthetic identifier-rewrite request produced 304 tokens in 39 rounds in both arms, with 271 active blocks and throughput gains of 3.15% and 3.24%. Both pairs matched the full-head prefill logits hash, output token sequence, finish reason and stop token on all four requests.

### Comparison with the fork baseline

| Request | Nov 3 service decode tok/s | Active-head Lab decode tok/s | Change since baseline |
|---|---:|---:|---:|
| Astropy | 207.658 | 273.24 | +31.58% |
| Django | 184.816 | 265.04 | +43.41% |
| SymPy | 189.942 | 278.61 | +46.68% |

The November 3 service measured the same request fixtures with HQ KV and produced 876, 954 and 1,268 tokens respectively. The current measurements use NVFP4 KV and Kernel Lab engine timing. These historical ratios include intervening optimizations, configuration changes and different output trajectories. The full/active pairs above isolate the draft-head change. [Archived service measurements](2026-11-03-coding-agent-5080-baseline.md#decode-results)

### Projection timing and numerical checks

The isolated projection test used real K4 weights and seven columns of deterministic synthetic BF16 activations. For each of seven masks it checked all 1,738,240 output words. Every retained logit matched the full-head BF16 bits and every masked logit was negative infinity. The full projection took about 0.707 ms. The three initial request masks with 421, 541 and 431 active blocks took 0.174, 0.225 and 0.175 ms respectively. At 1,850 blocks it took 0.681 ms. With all blocks enabled it took 0.709 ms against 0.707 ms for full. A separate all-blocks run reproduced the first 128 Astropy output tokens and the first 28 rounds' emission counts.

### Prefix controls

With 192 always-active blocks, the three requests ended at 405, 534 and 412 active blocks. In a 192 / 256 / 256 / 192 comparison, all four requests matched in output tokens, per-round emission counts, prefill logits hash, finish reason and stop token. The three coding requests combined gained 0.141% and 0.002% in the two pairs relative to 256, or 0.072% pooled.

With only prompt and committed-output blocks, Django diverged at output token 380. The full head stopped at 873 tokens, while the context-only run reached its 8,192-token cap. An independent process reproduced all 8,192 tokens and all 1,277 per-round emission counts. The runs already differed in emission count on round one. Token 380 fell at the end of a full-head round and at the start of a context-only round. Both selected IDs (81 and 25,543) belonged to active blocks at the divergence. The target logits and state difference at that position were not measured. The other three requests reproduced their full-head tokens.

## Finding

The 256-block prefix plus context blocks reduced combined C1 decode time from 9.897 s to 9.714 s for the same 2,640 output tokens (mean of the two runs per arm). The gain survived three extra rounds on Astropy and one on Django. The identifier rewrite matched the full head on one synthetic example. The earlier static shortlist used a different GPU and head format, so this experiment does not isolate the effect of adding prompt blocks to that implementation.

## Implications

The C1 gain warrants testing an opt-in service path with per-sequence state and a full-head fallback. The evidence supports keeping 256 as the experimental baseline. Enabling it by default would require evidence that the throughput gain survives representative service traffic and that the mask follows the correct request through slot reuse and prefix restoration.

## Limits and unknowns

All three real prompts were English Python coding tasks with 16.7K–22.4K input tokens. Two paired runs establish repeatability on those inputs, not a population variance or signal-to-noise estimate. Sampling, multilingual requests, short free generation, 128K context and C2 may change both block coverage and tokens committed per round. A shared C2 projection could use the union of both masks; falling back to full for C2 is another option.

The three prompt-derived masks all include blocks 1937 and 1938. Opening a block makes every token in it eligible, even if the prompt never contained that particular token. The Django context-only divergence is consistent with sensitivity to changed round boundaries. Both candidate blocks were open at the divergence; a near tie or a specific numerical cause would require target logits and state measurements at the common prefix.

## Follow-ups

For a service evaluation, freeze the block policy and compare against the full head on held-out multilingual, short-generation and sampled requests. Track output and stop behavior, total decode time, emitted tokens per round and mask density. At 128K and C2, measure the full-head fallback and verify request isolation. Include reasoning and tool closures on prompts that do not already activate their blocks.

[Prototype and reproduction](artifacts/2026-11-23-draft-active-vocab/README.md) · [Per-request measurements](artifacts/2026-11-23-draft-active-vocab/forward.csv) · [Projection measurements](artifacts/2026-11-23-draft-active-vocab/head-results.txt)
