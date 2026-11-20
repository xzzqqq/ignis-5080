# DFlash2 context append and NVFP4 gate/up on the RTX 5080

- Kind: experiment
- Status: current
- Observed: 2026-11-21
- Last verified: 2026-11-21
- Scope: Qwen3.8-27B DFlash2 on one RTX 5080 16 GB
- Related: [A16 NVFP4 MMA](2026-09-29-a16-nvfp4-on-the-tensor-cores.md), [coding-agent benchmark](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

Can the drafter reduce commit work and accelerate its fused gate/up projection at eight tokens while keeping one resident copy of the weights?

## Evidence

Context append now projects the K and V rows directly. It omits the 4096 unused Q rows per layer and the ten row-extraction copies across five layers. Decode append projects through the longest committed prefix in the batch. Each lane writes its own valid prefix. A single active column takes the existing GEMV route.

The small-T kernel decodes NVFP4 weights to BF16 and accumulates on tensor cores in FP32. In a first T=8 gate/up screen, moving partial reductions to a separate launch took at least 330 µs versus 172 µs for fused SIMT. A second screen used actual layer-0 weights, synthetic BF16 activations and the original MMA reduction. Reordering each 16×16 weight tile for warp loads gave the following medians across three measurement orders, each with 50 timed launches:

| T=8 fused gate/up + SwiGLU | µs/call |
|---|---:|
| SIMT, original layout | 175.7–176.1 |
| MMA, original layout | 231.4–233.4 |
| MMA, packed layout | 124.8–125.0 |

Both MMA layouts produced identical output bits. Global load instructions fell from 4.87M to 2.79M and L1/TEX global-load sectors from 33.42M to 9.75M. Profiled DRAM reads rose from 91.0 MB to 100.4 MB. The packed artifact stores the same codes, group scales and FP32 divisor. It places 128 code bytes per 16×16 tile in warp-lane order. Only the five DFlash2 gate/up tensors carry the new layout tag.

At T=16 the original-layout small MMA microbenchmark measured the following
same-input times (µs, SIMT → MMA): QKV 72.1 → 51.5, attention output
41.4 → 31.3, fused gate/up 309.3 → 278.7, MLP down 143.8 → 130.7.
These operator measurements selected the original-layout T=16 routes.

For append, a separate RTX 5080 screen timed both 1024×5120 KV halves
using 32-column SIMT chunks versus one MMA launch per half. Median µs
were 116.0 / 180.7 at T=64, 173.4 / 180.8 at T=96 and 230.8 / 180.7 at
T=128. The reviewed implementation starts narrow KV MMA at T=128.
The service A/B below predates this threshold adjustment; it does not
measure the standalone speedup of context append.

### Service comparison

Both artifacts ran on the same release binary (SHA-256 `159baa5a3244e198c70d258a190a440c262f93e289d45db6d6aba4eac1b124eb`), with NVFP4 KV, two configured lanes and one active greedy request. Both arms include the append changes. Each request had one warmup and three timed runs to natural completion.

| Request | ms/round, original → packed | Output tok/s, original → packed | Output tokens / rounds, original → packed |
|---|---:|---:|---|
| Astropy | 20.214 → 19.898 | 257.62 → 262.66 | 1104 / 212 → 1108 / 212 |
| Django | 20.392 → 20.030 | 225.21 → 253.40 | 1589 / 346 → 873 / 172 |
| SymPy | 20.181 → 19.845 | 259.16 → 263.55 | 659 / 126 → 659 / 126 |

The 192-token C1 sampling diagnostic took about 1031 → 1017 ms at 51 rounds. C2 diagnostics include overlapping admission and prefill. Both load plans reported 10,907,958,020 bytes for weights.

### Conversion and verification

Run the converter on a host containing the original file and its adjacent `.conversion.json`. The destination directory must exist and have room for another complete artifact. Use a new filename.

```sh
python3 scripts/repack_dflash2_gate_up.py \
  models/qwen3_8_27b_trellis_sc30-v2.ninfer \
  models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer
```

The converter writes a companion conversion record and checks the inverse permutation against every original gate/up payload. An independent file comparison verified all other payload bytes and padding. The resulting model file was 13,379,877,124 bytes, SHA-256 `7b38a5470a141c8e08127852219914cd9270efb663790233b7b30d02368108aa`. With the updated binary, set `ARTIFACT := models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer` in the serving host's private `local.mk`. Selecting the original artifact restores row-major gate/up storage and the T=8 SIMT gate/up route. Both artifacts retain this PR’s append changes and small-T projection dispatch, including the T=16 MMA routes.

Acceptance passed 230 artifact unit tests, CUDA all-target compilation, 132 kernel CTests, Trellis GPU projection/layer/lane checks, VRAM reload, 131072-token capacity and the service canary. Raw experiment receipts are under `.scratch/dflash-opt/layout/` on the measurement checkout.

## Finding

On these three requests the packed layout reduced measured C1 decode time by 0.316–0.362 ms per round. SymPy gained 1.69% in throughput at matching token and round counts. The counter changes and unchanged reduction support improved load efficiency as the explanation for the isolated MMA speedup.

## Implications

The serving host can select the packed variant without loading a second head or gate/up weight copy. Older artifacts continue to select their original layout. The downloadable model catalog still describes the published original artifact.

## Limits and unknowns

Changing the compute route can change rounding and the proposal trajectory. In the separate Django append diagnostic, one of 5120 feature values differed between T=1 GEMV and the first column of T=2. Both had relative L2 error about 0.000918 against FP64. All ten K/V buffers matched the expected cache update. The first differing target logit was not captured. Django's throughput includes a large change in generated length. The C2 samples do not isolate steady-state two-lane throughput.

## Follow-ups

A claim about steady-state C2 performance would require matched two-lane decode intervals and per-round output counts.
