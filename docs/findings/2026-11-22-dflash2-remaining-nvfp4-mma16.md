# RTX 5080 上 DFlash2 剩余 NVFP4 投影的 MMA16 布局

- Kind: experiment
- Status: current
- Observed: 2026-11-22
- Last verified: 2026-11-23
- Scope: RTX 5080 DFlash2 投影、KV 写入与 artifact 布局
- Related: [前次 gate/up 布局实验](2026-11-21-dflash2-nvfp4-mma-layout.md), [coding-agent benchmark](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

剩余 DFlash2 NVFP4 权重共用 MMA16 存储后，能否缩短真实 coding 请求的解码轮次，同时支持单列 commit、双 lane decode 和宽 prefill？

## Evidence

实验使用 Qwen3.8-27B Trellis、NVFP4 KV、两条配置 lane 和 1024-token prefill chunk。对照 artifact 已将五层 gate/up 转为 MMA16。候选再转写 26 个张量，沿用一个设备权重副本。数值比较区分相同 MMA 归约的布局变换、SIMT 到 MMA 的归约变化，以及只改变输出位置的直写。

### 当前服务基线

冻结的 release 二进制 SHA-256 为 `aa1e1f4389d66d007ffc3e95e0de7f49dcb4ee287afed4e0d28a523fa7264adb`，服务加载的 gate/up artifact SHA-256 为 `7b38a5470a141c8e08127852219914cd9270efb663790233b7b30d02368108aa`。Nsight Systems 在实际请求中记录了 T=8 的 packed gate/up。C1 用固定 SymPy 请求截到 256 个输出 token，共 52 轮，平均 4.92 tokens/round。C2 用两个同样的短 coding 请求，各截到 512 tokens。T=16 gate/up 确认 138 轮双 lane，共有 137 个相邻双 lane 轮间隔，平均 25.334 ms。两条长 SymPy 同时提交的初次试跑没有双 lane decode，已从 C2 数据排除。

| 操作 | C1 逻辑权重 MB/round | C1 GPU µs/round | 双 lane GPU µs/round |
|---|---:|---:|---:|
| MLP down | 250.68 | 414.22 | 943.53 |
| Attention output | 58.98 | 122.27 | 169.72 |
| Draft QKV | 88.47 | 184.26 | 256.01 |
| Commit K/V | 29.49 | 98.88 | 139.04 |
| Attention 和 MLP conv_proj | 36.86 | 106.88 | 237.07 |
| Feature projection | 73.73 | 127.06 | 158.29 |
| 合计 | 538.21 | 1053.57 | 1903.65 |
| QKV 的 15 次 D2D copy，另计 | | 7.94 | 7.85 |

C1 的 commit 执行宽度 T=1 到 8 分别出现 2、7、10、6、6、3、6、12 轮。双 lane 的执行宽度为两倍最长有效前缀，T=2、4、6、8、10、12、14、16 分别出现 6、25、20、23、13、20、9、22 轮。以上字节数计同一 QKV 在 draft 和 commit 的两次读取。按当前 1.054 ms 基线，投影合计省 0.40 或 0.50 ms 需要有效权重带宽约 824 或 972 GB/s。实际 DRAM 流量须由硬件计数器测量。

### 真实权重筛选

微测使用从基线 artifact 提取的真实 NVFP4 权重，BF16 激活由固定种子生成。CUDA event 包含完整调用，QKV 对照包含所有输出 copy。三种顺序各预热后计 50 次，下表用三次中位数的中位数。五层轮转的计时是五次连续调用总和，不包含重排。T=16 的 down 和 output 原生产路由已经使用原布局 MMA。

| 形状和测法 | T | 生产 µs | Packed µs |
|---|---:|---:|---:|
| Down，第 0 层重复 | 8 | 78.34 | 39.26 |
| Down，五层轮转 | 8 | 421.12 | 327.97 |
| Down，五层轮转 | 16 | 758.82 | 513.79 |
| Attention output，五层轮转 | 8 | 120.00 | 51.78 |
| Attention output，五层轮转 | 16 | 154.18 | 82.50 |
| 两种 conv_proj，分别测五层后合计 | 8 | 107.36 | 103.55 |
| 两种 conv_proj，分别测五层后合计 | 16 | 248.77 | 171.10 |
| Feature，按 C1 实际 T 加权 | 1–8 | 109.21 | 87.02 |
| Feature，宽 MMA | 1024 | 2813.12 | 2725.02 |

原布局 MMA 在第 0 层 down T=8 为 82.34 µs。五层 down 和 output 的两个布局在同一 MMA 归约下均逐位一致。conv 在 C1 的提升很小，双 lane 是将其纳入组合的主要理由。Feature 在 T=1 约慢 2 µs；T=1 占此次 C1 的 2/52。宽 feature 在 T=128 有约 3% 微测回退，真实 1024-token chunk 则有所改善。

| 单层实验 | T | 原布局旧输出 µs | 原布局直写或合并 µs | Packed 同输出方式 µs |
|---|---:|---:|---:|---:|
| Draft QKV | 8 | 42.30 | 36.86 | 16.64 |
| Draft QKV | 16 | 57.82 | 51.52 | 27.10 |
| Commit K/V | 1 | 12.64 | 8.48 | 9.89 |
| Commit K/V | 1024 | 300.32 | 295.17 | 286.85 |

按 C1 实际宽度加权，五层 K/V 的原布局两次投影、原布局合并、packed 合并分别耗时 89.38、66.68、57.99 µs。原布局的直写与合并逐元素保持有效输出。以 16 列窄 MMA 循环读取宽 prefill 会重复扫描权重，T=1024 的 K/V 耗时约 1145 µs、feature 约 13.6 ms。最终实现让原有宽 MMA 从 packed tile 异步搬运 codes 和 scales，沿用共享内存中的 BF16 布局和 FP32 归约。真实 K/V 在 T=128、129、1024，feature 在 T=64、127、128、129、1024 的两种宽布局输出逐位一致。

### 数值和缓存策略

真实权重共检查 107 个形状、层和宽度组合，每个组合在相同输入上对生产及候选各取 128 个确定性输出，与从原始 payload 解码的 FP64 点积比较，并检查完整输出的有限性和尾部哨兵。两侧最大相对 L2 均为 0.002127，低于 A16 既有的 1/256 界限。这个检查覆盖实际投影值；接受长度和输出轨迹另由服务请求测量。原布局合并 K/V 的写槽测试在五层交换两条 lane、分别提交 3 和 1 列、跨越 2048 容量的回环边界，10 个完整 K/V 缓冲区与独立构造的期望字节一致，包括未写槽。

以仅转写 down 和 output 的临时 artifact，在同一二进制上交替测 down codes 的默认加载与 L2 优先淘汰。两者具有不同的 SASS 内存描述符；grid、scale 和归约相同。SymPy 固定 256 输出，每次 52 轮，每组预热一次后取三次中位数。按普通、优先淘汰、优先淘汰、普通顺序，分别为 **19.2047、19.2081、19.2129、19.2047 ms/round**。最终组合使用普通加载。

| NCU 匹配轮次 | 位置 | 发出 tokens | DRAM 读 MB，普通 → 提示 | DRAM 写 MB，普通 → 提示 | L2 命中率，普通 → 提示 |
|---|---:|---:|---:|---:|---:|
| 10 | 16754 | 5 | 12196.20 → 12194.88 | 205.51 → 205.87 | 38.20% → 38.17% |
| 20 | 16807 | 3 | 12197.11 → 12195.11 | 191.60 → 190.30 | 38.10% → 38.07% |

NCU 使用既有 Kernel Lab 普通 forward 的 NVTX 轮次和 app-range replay，仅观察这两个完整轮次。两个实验的输出 token、logits hash、所有轮次的位置和输出数均一致。每轮写量变化小且方向不一致，因此未将该提示扩展到 QKV。

### 同一二进制的服务 A/B

两边使用 SHA-256 为 `b64691f6bad74b7ac05737253b4a6ed256b37ec7f99e3a124b952d2296fcefcb` 的 release 二进制。A 加载原 gate/up artifact 并关闭直写和合并，B 加载候选并启用两者。每个固定请求各有一次完整预热和三次顺序计时，返回工具调用即结束。A/B 的 benchmark 有效配置一致，禁用了 prompt 复用和 retained slots。测量时没有默认服务在运行，私有 `local.mk` 中的 `--speech-model` 和 `--diarizer-model` 已不被此二进制接受。显式 benchmark 配置省略了这两个参数，两边的 host KV 池均为 2 GiB，私有 Make 配置为 8 GiB。

| 请求，A → B | ms/round | tokens/round | 总 decode 秒 | 输出 tok/s | 输出 tokens / 轮数 | prefill 秒 |
|---|---:|---:|---:|---:|---:|---:|
| Astropy | 19.431 → 19.152 | 5.226 → 5.266 | 4.119 → 3.964 | 268.97 → 274.94 | 1108 / 212 → 1090 / 207 | 9.285 → 9.292 |
| Django | 19.617 → 19.345 | 5.076 → 5.017 | 3.374 → 3.366 | 258.74 → 259.35 | 873 / 172 → 873 / 174 | 10.655 → 10.665 |
| SymPy | 19.437 → 19.149 | 5.230 → 5.230 | 2.449 → 2.413 | 269.08 → 273.14 | 659 / 126 → 659 / 126 | 7.756 → 7.764 |

每个请求的三次 B 侧 ms/round 均低于该请求最快的 A 侧样本。按项目 11 月 3 日固定历史基准，候选的三项 tok/s 比值依次为 1.324×、1.403×、1.438×；该比值包含期间的全部改动。

三个请求的停止原因均为 `tool_calls`。Astropy 的 reasoning 文本变化，最终工具、参数和路径相同。Django 和 SymPy 的完整返回消息均相同。Django 多用两轮，因而其 tok/s 改善小于每轮耗时改善。SymPy 在相同输出和轮数下，decode 吞吐提升约 1.51%。

C1 sampling 使用 SymPy、seed 7、temperature 1、top-k 20、top-p 0.95 和 256-token 上限。三次均在 81 轮接受 175 个草稿 token。接受长度 0–7 的轮数为 24、18、10、10、6、1、6、6，A/B 完全相同。decode 吞吐中位数为 161.99 → 164.32 tok/s，含 prefill 的 HTTP 吞吐为 27.31 → 27.37 tok/s，停止原因为 `length`。

只做了一项服务消融。保持最终二进制、QKV 输出方式、K/V 合并和其他张量不变，仅把 feature projection 恢复为原布局。在同一 Django 请求的一次预热和三次计时中，每次都生成 1589 tokens、346 轮，中位数为 19.337 ms/round、6.691 s decode、237.50 tok/s。完整候选为 873 tokens、174 轮、3.366 s，且返回与 A 相同的工具调用。单独回退 feature 使模型选择了另一个 grep 参数，未恢复 A 的 172 轮轨迹。保留 packed feature。

A、B 的权重加载计划均为 10,907,958,020 字节，加载时计划分配均为 15,347,915,180 字节，ready 后 NVML 同为 14,388 MiB。两边的 KV 池容量均为 155,904 tokens，单序列上限均为 139,264。候选实际完成无缓存的 131,072 输入 token 容量检查，产出 1 token 后以 `stop` 结束；四项服务 canary 均通过自洽性检查。

### 双 lane 区间和完整请求

同一 release 二进制上，分别用 Nsight Systems 捕获两条 lane 都在 decode 的连续区间。每轮五次 T=16 packed gate/up 用于确认双 lane，取相邻 selector walk 的 GPU 时间差。通过每层 append 前 RoPE 的 grid 宽度核对实际执行列数；扣除单 lane 的产出后计算共同区间的 tokens/round。普通服务另跑三次并发请求对计时，包含两次 prefill 和单 lane 收尾。

| C2 指标 | A | B |
|---|---:|---:|
| 双 lane 轮数 / 单 lane 轮数 | 138 / 24 | 143 / 19 |
| 相邻双 lane 间隔数 | 137 | 142 |
| 双 lane 平均 ms/round | 25.226 | 24.449 |
| 双 lane 中位 ms/round | 25.193 | 24.376 |
| 双 lane 输出 tokens/round，两条合计 | 6.870 | 6.727 |
| 普通服务的双请求吞吐中位数，tok/s | 210.32 | 213.36 |

双 lane 的平均轮间隔缩短 0.777 ms。B 的接受轨迹需要更多轮，两边双请求都输出 1024 tokens。完整请求吞吐提升 1.44%。

### 最终回归

Python 转换测试、artifact crate 测试和 CUDA workspace all-targets 编译通过。GPU runner 的 132/132 项 CTest、四组 Trellis GPU 集成和显存重载全部通过。重载测试冷启动得到 2436 页，三次重载各得到 2429 页，差异 0.29%，在项目的 1% 界限内。另以原输出方式检查了 packed 普通 QKV、分开的 packed K/V，包含单列、尾列和 T=1024，均通过 FP64 与尾部哨兵检查。

## Finding

同一二进制、相同配置下，三个 C1 greedy 请求每轮减少 0.271–0.289 ms，约为 0.40 ms 进取目标的七成。两个活跃 lane 的平均轮间隔减少 0.777 ms。五层权重轮转的 down 收益约 93 µs，单层重复读取会高估它在服务中的机会。实测的整轮时间和每轮输出共同决定 tok/s。

## Implications

候选位于测量机的 `/shared_home/ignis-lab-1005/dflash-remaining-20261122/qwen3_8_27b_trellis_sc30-dflash-projections.ninfer`，SHA-256 为 `0295bd728310a651bbae9e741de99dfe2da8814f0daa04056f71663e08b368eb`。源和目标均为 13,379,877,124 字节。独立比较检查了所有 header 对象、26 个新布局张量的逆变换，以及其余每一段 payload 和数据区 padding；转写总计 508,723,304 字节。

```sh
python3 scripts/repack_dflash2_gate_up.py \
  models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer \
  /path/to/qwen3_8_27b_trellis_sc30-dflash-projections.ninfer \
  --projection mlp/down --projection attention/output \
  --projection attention/query_key_value --projection feature_projection \
  --projection attention/conv_proj --projection mlp/conv_proj
```

上述增量转换的 `.conversion.json` 中，`layout_conversion.tensors` 记录本次转写并逆变换验证的 26 个张量。要在同一条记录中列出全部 31 个，可从原始 `models/qwen3_8_27b_trellis_sc30-v2.ninfer` 运行上述命令，并追加 `--projection mlp/gate_up`。最终各张量的布局可直接从 artifact header 读取。

每个布局由 artifact 的张量标签独立选择。`IGNIS_DFLASH2_DIRECT_QKV=0` 强制投影后拆分，`IGNIS_DFLASH2_COMBINED_KV=0` 强制按 K、V 分别投影，`IGNIS_DFLASH2_DOWN_EVICT_CODES=1` 仅为 packed down 的 codes 选择 L2 优先淘汰。实验目录为测量 checkout 的 `.scratch/dflash-remaining-20261122/`。

## Limits and unknowns

本次服务测量覆盖三个固定 coding 请求、一个采样种子和一对短 coding 请求。三次重复用来观察同一请求的稳定性。三个最终工具调用均与对照一致；这些 fixture 只测下一条响应，没有执行工具或给出任务解决率。每个 C2 arm 的共同区间来自一次独立 profiler capture，完整请求吞吐另取三次普通运行中位数。

## Follow-ups

复用本次实验时，可按新的 commit 宽度分布对测量目录中的 `operator-medians.json` 重新加权。其 `order_medians_us` 保留了每个测量顺序的 50 次中位数。
