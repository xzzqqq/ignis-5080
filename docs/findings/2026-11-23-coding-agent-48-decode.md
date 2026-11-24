# RTX 5080 decode on 48 frozen coding-agent requests

- Kind: experiment
- Status: current
- Observed: 2026-11-23
- Last verified: 2026-11-23
- Scope: serving / RTX 5080 C1 coding-agent decode
- Related: [request manifest](../../crates/bench/fixtures/coding_agent_48/README.md), [historical control](2026-11-03-coding-agent-5080-baseline.md)
- Superseded by: none

## Question

How fast do the baseline and optimized 5080 service generate the next response to the same 48 requests?

## Evidence

B used main at `8495f3a0`, the 31-tensor packed Trellis artifact from PR #19, NVFP4 KV and 139,264 context. Its artifact SHA-256 was `0295bd728310a651bbae9e741de99dfe2da8814f0daa04056f71663e08b368eb`. It used DFlash2 with seven drafts, two configured lanes and one active request, a 1,024-token prefill chunk, greedy generation with maximum reasoning effort, natural EOS and an 8,192-token output cap. The client disabled prefix reuse, checked each request hash against the manifest and checked the response for a single full prefill and zero cached tokens. One separate historical Astropy request warmed the engine. Each frozen request then ran once in pilot-first order.

The [B per-request measurements](artifacts/2026-11-23-coding-agent-48/B.json) pin the commit, binary, artifact and manifest hashes. B produced 57,357 output tokens in 204.608 decode seconds. The 48-request wall time was 777.87 seconds. All 48 samples passed the checks. 45 ended with a tool call and three reached the output cap. Raw responses, counters, service settings and timing records are in `.scratch/decode48-B-8495f3a0-20261123-200314/`.

A ran from `e25780e2` in an isolated Linux checkout with its CUDA release binary, original Trellis artifact, HQ KV and 262,144 context. Both arms used one RTX 5080, the same request order and generation options. The same historical Astropy warmup ran once before each arm. B ran first, then A in a separate service launch. The [A per-request measurements](artifacts/2026-11-23-coding-agent-48/A.json) record provenance and all 48 rows. A's raw archive is `.scratch/decode48-A-e25780e2-20261123-202238/`.

| Measure | A, baseline | B, optimized |
|---|---:|---:|
| Median task decode, tok/s | 191.34 | 261.71 |
| Combined output / decode seconds, tok/s | 203.24 | 280.33 |
| Output tokens | 72,371 | 57,357 |
| Total decode seconds | 356.094 | 204.608 |
| Decode ms per speculative round, pooled | 25.983 | 19.245 |
| Accepted / drafted, pooled | 61.20% | 62.85% |
| Finished with `length` | 3 | 3 |
| Finished with `tool_calls` | 45 | 45 |

All 48 A requests passed the input, cache, timing and output checks. A's measured wall time was 14.65 minutes, excluding build, loading and warmup. Timed length finishes remain in the results.

## Finding

The geometric mean of the 48 per-task B/A decode-rate ratios was **1.3500× (+35.00%)**. B was faster on 47 tasks and slower on 1. The median ratio was 1.351× and P10–P90 was 1.141–1.637×. The stratified bootstrap 95% interval for the geometric mean was 1.301–1.399× (10,000 draws, seed 5080, resampling four tasks with replacement within each of the 12 manifest cells).

| Task group | Tasks | A median tok/s | B median tok/s | Geometric mean of paired speedups |
|---|---:|---:|---:|---:|
| Length Q1 | 12 | 210.76 | 283.89 | 1.300× |
| Length Q2 | 12 | 205.25 | 270.30 | 1.285× |
| Length Q3 | 12 | 173.73 | 231.59 | 1.355× |
| Length Q4 | 12 | 158.63 | 255.50 | 1.466× |
| explore | 16 | 203.09 | 286.42 | 1.393× |
| edit | 16 | 187.32 | 239.94 | 1.323× |
| debug | 16 | 183.12 | 252.84 | 1.335× |

The three smallest B/A ratios were `matplotlib__matplotlib-24627` (0.882×), `pytest-dev__pytest-10051` (1.088×), `matplotlib__matplotlib-25332` (1.119×). The [paired record](artifacts/2026-11-23-coding-agent-48/paired.json) includes every ratio.

On the only slower task, `matplotlib__matplotlib-24627`, B shortened the mean round from 24.34 to 18.65 ms, while draft acceptance fell from 65.01% to 39.43%. A generated 987 tokens and B generated 281. The resulting decode rates were 227.79 and 200.92 tok/s.

### Estimated model bandwidth utilization

Following the [model-traffic accounting](2026-11-17-decode-mbu.md), divide summed logical draft, verify and commit bytes by the measured server decode seconds and the RTX 5080's nominal 960 GB/s (decimal). A round shares its weight reads across the tokens it emits.

| Across all 48 requests | A | B |
|---|---:|---:|
| Speculative rounds | 13,705 | 10,632 |
| Output tokens per round | 5.281 | 5.395 |
| Estimated logical GB per round | 12.367 | 12.587 |
| Effective logical GB/s | 475.96 | 654.07 |
| Estimated model bandwidth utilization | 49.6% | 68.1% |

The logical weight payload per round is 11,560,273,812 bytes on A, including two full-head projections. B uses 11,501,291,412 bytes after omitting the 4,096 unused Q rows of each of the five commit projections. The remaining terms follow the earlier account: target KV, 544 BF16 sink/recent rows, draft KV, persistent GDN state and selector gathers. Compressed target history costs 9,216 bytes per position on A and 18,432 on B. For each task the measured round, draft and accepted counts give the total verify width and committed length. The midpoint estimate distributes committed positions evenly across rounds. Constraining each round to advance by one to eight tokens gives 49.56–49.60% for A and 68.08–68.18% for B when varying only the unknown order of those advances. No DRAM hardware counters were captured during these service runs.

## Implications

The measured change covers the optimized runtime, packed DFlash2 weights and NVFP4 KV together. The full-window rate includes drafting, verification and completion scheduling.

## Limits and unknowns

The two versions produced identical returned choices on 0 of 48 tasks after dropping generated tool-call IDs. This comparison includes reasoning, visible text, tool arguments and finish reason. The endpoint returned no raw output-token ID sequence. Natural output length and draft acceptance can change the measured decode rate. Each task was measured once per arm on one card; the interval describes variation across sampled tasks, with no repeated-launch timing estimate. The 12 cells have equal weight and the benchmark does not execute generated tool calls.

## Follow-ups

None.
