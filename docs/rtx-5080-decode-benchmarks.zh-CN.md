# Ignis-5080 的 decode 测量与开源方案对比

2026-11-24 · 单张桌面 RTX 5080 16 GB · Qwen3.8-27B · 单请求（C1）

**在我们同机实测的开源实现中，Ignis-5080 在这组真实 coding-agent 回合上 decode 最快。** 在 48 道冻结请求上，生产版的逐题中位数为 **261.71 tok/s**。相对 fork 初期的原始 Ignis，逐题配对提速 **35.0%**，模型逻辑带宽利用率从 **49.6%** 提升到 **68.1%**。在独立竞品中，我们选择 Ferrite 跑了同一套请求，Ignis 逐题配对领先 **44.0%**。其他项目保留各自在公开工作负载上的测量值。[上游 A/B 报告](findings/2026-11-23-coding-agent-48-decode.md) · [Ferrite 同机报告](findings/2026-11-24-ferrite-coding-agent-48.md)

## 为什么用这 48 道题

早期开发用 Astropy、Django、SymPy 三个公开 SWE-agent 请求做固定基线。三题均在 agent 读过相关实现和测试、准备编辑时截取。收尾评测保留这三题作为历史对照，从同一份 SWE-bench Verified 提交的其余 497 个任务中另选 48 题，检验优化能否覆盖不同仓库、输入长度和工作阶段。[历史三题](../crates/bench/fixtures/coding_agent/README.md)

497 个任务提供了 16,451 个合格候选回合。按模型实际输入 token 数划四个分位档，计算边界时每个任务权重相同；再按探索定位、代码修改、验证调试分三个阶段。每个「长度 × 阶段」组合抽四个不同任务，各任务只出现一次。最终覆盖 12 个仓库，输入从 2,978 到 63,564 tokens。题号、轮次、阶段依据、输入长度和请求哈希均在测速前冻结。[48 题清单](../crates/bench/fixtures/coding_agent_48/README.md) · [冻结 manifest](../crates/bench/fixtures/coding_agent_48/manifest.json)

每个请求保留原回合可见的消息、源码和历史工具返回。一次完整预热后，各题独立生成一次下一条回复，开启最高推理强度，使用 greedy，关闭跨请求前缀复用，尊重自然 EOS，输出上限 8,192 tokens。达到上限的请求保留在统计中。服务端从最终 prefill 与 decode 的交界计时至调度完成，覆盖首个推测轮次。生成的工具调用只记录，解题正确率不在本次测量范围。[协议与计时边界](findings/2026-11-23-coding-agent-48-decode.md)

## 相对 fork 起点的累计收益

A 取 fork 初期加入 benchmark 计时的提交 `e25780e2`，运行当时的推理实现、原始 Trellis 权重、HQ KV 和 262,144 上下文配置。B 取 PR #19 合并后的 `8495f3a0`，加载生产默认的 31 张量重排权重、NVFP4 KV 和 139,264 上下文配置。两组使用同一张卡、相同输入顺序、DFlash2 七个草稿 token、两条配置 lane，每次只有一个请求活跃。[运行配置与逐题测量](findings/2026-11-23-coding-agent-48-decode.md)

| 指标 | A · fork 初期 | B · 优化后 |
|---|---:|---:|
| 逐题 decode 中位数 | 191.34 tok/s | **261.71 tok/s** |
| 总输出 tokens ÷ 总 decode 秒数 | 203.24 tok/s | **280.33 tok/s** |
| 每个推测轮次平均耗时 | 25.98 ms | **19.24 ms** |
| 每轮平均输出 tokens | 5.28 | 5.39 |
| 草稿接受率 | 61.20% | 62.85% |
| 总输出 tokens | 72,371 | 57,357 |

先对每题计算 B/A 的 decode 速率比，再对 48 个比值取几何平均，得到 **1.350×**。B 在 47 题上更快，按长度和阶段分层重采样的 95% 区间为 1.301–1.399×。四个长度档从短到长分别加速 1.300×、1.285×、1.355×、**1.466×**。每轮产出增加约 2.2%，每轮耗时下降约 25.9%。[逐题配对数据](findings/artifacts/2026-11-23-coding-agent-48/paired.json)

### 模型带宽利用率（MBU）

按推测轮次累加 draft、verify、commit 涉及的逻辑权重、KV 和持久状态流量，除以所有请求的服务端 decode 时间，再除以 RTX 5080 标称的 960 GB/s。一轮生成的多个 token 共享权重读取。本轮数值是模型流量估算，未采集 DRAM 硬件计数器。[核算方法与输入假设](findings/2026-11-23-coding-agent-48-decode.md#estimated-model-bandwidth-utilization)

| 指标 | A · fork 初期 | B · 优化后 |
|---|---:|---:|
| 逻辑流量 / 轮 | 12.37 GB | 12.59 GB |
| 有效逻辑带宽 | 475.96 GB/s | **654.07 GB/s** |
| MBU | 49.6% | **68.1%** |

## 开源竞品与参考指标

下面标明每个数字由谁测、用什么输入。**Ferrite 是这次在多种独立竞品中单独挑出来跑完整 48 题的一家。** 其余数据用于了解公开实现的速度范围，不参与 48 题配对统计。

| 项目 | GPU | Decode 速率 | 来源与工作负载 |
|---|---|---:|---|
| Ferrite 0.9.2 | 5080 16 GB | **183.21 tok/s** | 我们同机实测，48 个真实 agent 回合的逐题中位数；MTP + 上下文复制。[报告](findings/2026-11-24-ferrite-coding-agent-48.md) |
| FLYX + MTP | 5080 16 GB | **225.9 / 161.0 / 105.8 tok/s** | 项目实测，实际输入分别约 1K / 128K / 256K，关闭 thinking，要求生成 512 tokens 的 Python 解析器。[报告](https://github.com/enwove/flyx/blob/main/docs/benchmarks.md) |
| llama.cpp + MTP3 | 5080 16 GB | **95.7–96.0 tok/s** | 社区实测，UD-IQ3_S，30K 输入热请求，短输入为 106.8–109.0。[配置与原始表](https://github.com/BFinn/5080-llm-configs/tree/main/models/qwen3.8-27b) |
| NInfer-5080 v1.5 | 5080 16 GB | **112.2 tok/s** | 项目发布版 sustained decode。131,072 是配置的上下文和 KV 容量。[发布记录](https://github.com/toddballinger/ninfer-5080) |
| SGLang + Escha-W2 + MTP | 5080 16 GB | **109.2 tok/s** | 量化项目实测，2-bit 权重、单用户；关闭推测为 61.7。[模型卡](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2) |
| ExLlamaV3 1.6 + DFlash2 | **5090 32 GB** | **274.6 tok/s** | 社区 RigMark coding，EXL3 4.0 bpw，关闭 thinking，五次中位数。[测试记录](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install#what-was-measured) |
| vLLM 0.28 + MTP | **5090 32 GB** | **175.0 tok/s** | 社区 NVFP4 KV 单流代码生成，取对照表中的单卡列。[测量矩阵](https://github.com/adrienbrault/qwen3.8-27b-rtx5090) |

**同机比较。** Ignis 与 Ferrite 的实际输入 token 序列逐题核对一致。Ignis 在 47/48 题上 decode 更快，配对速率比为 1.440×（分层 bootstrap 95% 区间 1.370–1.514×）。探索、修改、调试三组分别为 1.505×、1.397×、1.421×。两版自然生成的回复不同。包含 prefill 的整套墙钟时间为 Ferrite 9 分 24 秒、Ignis 12 分 58 秒，各自输出 53,630 和 57,357 tokens。[逐题明细与两边的时钟边界](findings/2026-11-24-ferrite-coding-agent-48.md)

**上游短题复测。** 旧版 `small-card` 的四条 coding 短题在上游双 lane 配置的 5080 社区报告中为 258.2 tok/s，我们用相同客户端公式测得 363.7 tok/s；三条合成 agent 回合分别为 254.3 和 326.3 tok/s。旧公式从首个流式输出后开始计时，短回复对首轮推测尤其敏感。[上游原始记录](https://github.com/gpillon/ignis/issues/360#issuecomment-6468739578) · [本机复测](findings/artifacts/2026-11-23-coding-agent-48/small-card.json)

**Ferrite 脚本复测。** 将 Ferrite 客户端接到 Ignis，同样关闭 thinking，各跑三次取中位数。LRU 生成题是 Ignis 323.1、Ferrite 公布 298 tok/s；500-token 整文件改名是 283.0 对 692，添加类型标注是 403.5 对 411。这里的 Ferrite 数字来自作者机器。[本机复测](findings/artifacts/2026-11-24-ferrite-48/ferrite-replay-summary.json) · [Ferrite 公布成绩](https://github.com/BHCC2025/ferrite/blob/main/docs/rtx-5080.md#results)
