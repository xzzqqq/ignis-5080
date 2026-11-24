# RTX 5080 主要文本 workload 的 profiling

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: serving / Qwen3.8-27B trellis on RTX 5080: 8K/64K coding decode, two lanes, cold prefill, agent turns and mixed prefill/decode; kernel / Nsight Systems and trellis microbenchmark
- Related: [首次 5080 基准](2026-11-03-rtx5080-first-bench.md), [5090 上的 trellis kernel](2026-10-23-dense-trellis-kernels-on-the-5090.md), [此前的 16 GB 卡速度估计](2026-10-24-27b-16gb-speed-estimate.md), [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md), [ADR 0061](../adr/0061-prefill-attention-reads-the-hq-codec.md), [ADR 0018](../adr/0018-chunk-level-prefill-decode-interleaving.md)
- Superseded by: none

## Question

日常聊天与 coding agent 使用的 27B，在实际 RTX 5080 上把时间花在哪里？短上下文、长上下文、
双请求和 prefill/decode 混合负载的限制是否相同，距离带宽 SOL 还有多远？

## Evidence

### 版本、环境与测量隔离

源码基线 `31ec76a180d0d0fb549848e39e75debc3df97176`，Ignis 0.9.5；release CUDA server 的 SHA-256：
`8b9d555ad3ee36fa66350957d3fe4e4186555855035d5a9372bde267a62a4b39`。

硬件、驱动、模型与[首次基准](2026-11-03-rtx5080-first-bench.md)相同：RTX 5080，84 SM，16,303 MiB，
360 W；i9-13900K，Linux，CUDA 13.0.88。模型为 catalog 的 `qwen3_8_27b_trellis_sc30-v2.ninfer`。
配置保持 `rtx50-16gb`、262,144 context、两条 decode lane、hq-e8-2b KV、1,024-token prefill chunk、
DFlash2 七个 drafts/full head。vision 仍按 profile 加载，但不发送图片；speech/diarizer/UI 关闭，
idle unload 与 loop guard 关闭。无适配、无显存溢出，基准的 NVML 读数为 14,415 MiB。

全部远程操作通过 `python3 scripts/remote.py`。首次同步、`make build` 后运行 baseline；期间另一项任务
修改并同步了源码，后续 `make start` 的 stale 检查正常拒绝启动。没有重建混入新代码：将上述 commit
归档到远端 `.scratch/profile-src/`，复制并核对同一可执行文件，后续通过该目录的 `make start/stop`
运行。第二次 baseline 也使用这份独立副本。此次未修改推理实现。

三个 server launch 的请求时间，均为 Asia/Shanghai：

| Launch | 时间 | 用途 |
|---|---|---|
| baseline | 22:19:31–22:22:10 | 不启用 profiler；冷请求、重复 decode、双 lane、agent |
| profile | 22:23:36–22:26:05 | Nsight Systems 2026.2.1，CUDA graph node trace，既有 chunk CUDA-event profiler |
| baseline2 | 22:26:30–22:28:27 | 独立启动、不启用 profiler，复测及正常 EOS 的 agent 请求 |

混合负载在第一次 baseline 后、停止该 daemon 前测量；trellis 微基准在两次 server 启动之间独占 GPU
运行。每个 GPU 阶段先检查 `make gpu-status`，server 启动经过 guard，没有并行的 GPU 工作。

### Workload 与口径

1. **8K / 64K coding**：用基线仓库 `crates/core/src/*.rs` 的源码构造两份不同的 system 上下文，
   user 请求实现带测试的 bounded LRU cache。固定随机种子，通过 `/v1/tokenize` 测量长度；四份实际长度
   是 8,192 / 8,193 / 65,534 / 65,536 tokens。greedy、thinking off、512 output tokens、`ignore_eos=true`。
   这是可控的源码上下文性能负载，不代表真实 agent 轨迹的质量分布。
2. 每个长度先依次冷跑 A/B，再重复 A，以及同时请求 A/B。第一次 launch 重复三组，第二次独立 launch
   重复一组。两次 launch 的四个冷请求均报告 `cached_tokens=0`。热请求明确使用 prompt checkpoint，
   cached tokens 分别为 8,188 / 8,189 / 65,472 / 65,532；不把它们当冷 prefill。
3. **Agent**：现有 `crates/bench/fixtures/small_card/agent_turn_{1,2,3}.json`，thinking on，最多 512 tokens。
   第二次 baseline 使用正常 EOS：输出分别为 512、512、44 tokens，第三项结束于 `tool_calls`。
4. **混合负载**：复用 64K A 开始 decode，300 ms 后提交一份独立的约 8K 冷 prompt；重复两次，冷 prompt
   使用不同前缀，均确认零 cached tokens。

HTTP decode rate = `(completion_tokens - 1) / (response_end - first_content_event)`，同时统计 reasoning
content。双 lane 聚合率 = `2 × 511 / (最后完成时间 - 最早首 content 时间)`，包括第二条 lane 启动的错位
和末尾单 lane；不是两条单请求速度直接相加，也不是全程端到端吞吐。

round 数与 drafted/accepted 数来自 server 的 `ignis.request.done`，每轮产出用 output tokens / rounds
近似，包含首 token 和最后一轮边界。SSE content event 可以携带多个 tokens，事件间隔不冒充逐 token ITL。

### 不启用 profiler 的性能

| 测量 | baseline | baseline2 |
|---|---:|---:|
| 8K 单 lane A，decode tok/s | 153.74，中位数，三次 | 153.79 |
| 64K 单 lane A，decode tok/s | 132.77，中位数，三次 | 132.57 |
| 8K 双 lane，聚合 decode tok/s | 218.94，中位数，三次 | 218.92 |
| 64K 双 lane，聚合 decode tok/s | 195.80，中位数，三次 | 195.55 |
| 8K A/B 冷 TTFT | 约 3.95–3.98 s | 3.960 / 3.981 s |
| 64K A/B 冷 TTFT | 34.346 / 34.461 s | 34.374 / 34.500 s |

第一次 launch 的 8K 单 lane 三次为 153.87 / 153.74 / 153.50 tok/s；64K 为
132.75 / 132.83 / 132.77。第二次启动的对应单 lane 和双 lane 结果都在第一次中位数的 0.2% 内。

双 lane 的每条请求分别约 117 / 111 tok/s（8K）、104 / 99 tok/s（64K）。聚合吞吐较 A 单 lane
提高约 42% / 47%，每条请求自身变慢。A/B 的内容和接受率不同，这不是严格同一 token 流的 batching A/B。

前缀复用后的单 lane TTFT，8K 约 71 ms；64K 从第一次的 263 ms 到第二次的 511 ms，变化明显，
不能只引用最快的一次。热请求仍有恢复、尾部 prefill 和调度成本。

正常停止条件的三个 agent fixture 在 baseline2 分别为 **219.91 / 256.17 / 204.04 tok/s**。
后两项复用了共同 system 前缀 576 tokens；第三项只有 44 tokens，因此不宜用三项平均值代表长输出能力。
第一次 baseline 对 agent 也强制忽略 EOS，第三项在原本应结束的工具调用后继续输出特殊 token，
得到 75.11 tok/s；**该样本无效，不进入 agent 性能结论**。源码 coding 请求未发现这种特殊 token 尾部。

### 投机接受率解释了短 prompt 与源码 workload 的速度差异

| 单 lane workload | rounds | accepted / drafted | 约 tokens/round | HTTP decode ms/round |
|---|---:|---:|---:|---:|
| 短提示 merge sort warmup | 76 | 82.1% | 6.74 | 22.76 |
| 8K coding A，热请求 | 138 | 39.3% | 3.71 | 24.06–24.12 |
| 64K coding A，热请求 | 127 | 43.3% | 4.03 | 30.29–30.31 |
| Agent fixture 1 | 100 | 59.5% | 5.12 | 23.23 |
| Agent fixture 2 | 86 | 71.0% | 5.95 | 23.19 |

因此首次基准的短 coding 提示约 285 tok/s，不能直接当作 8K 源码 coding 的预期。
此处短提示与 8K coding 的每轮时间只差约 6%，每轮产出却差约 45%；tokens/s 的主要差距来自投机接受率。
也不能沿用此前 5080 估计中约六个 tokens/round 的前提，把其 243–307 tok/s 与本次约 154 直接判成回归。

### Decode 的 kernel 分解

Nsight 以 CUDA graph 的首个 kernel 为 decode 区间起点，保留 graph 外的 fold、拷贝和尾部操作。
每轮 graph 内有 1,204 个 kernel；加上 graph 外的工作，约 1,229 个 kernel/round。
每轮有 338 个 trellis decode kernel：drafter head、336 个 body 投影、target head，另有 rotation。

下表来自各一次热的单 lane capture，138 / 127 rounds。时间是同类 kernel 的 device duration 总和除以
rounds，括号是该时间占整个 decode device 区间的比例。PDL 可使 kernel 时间重叠，因此这些分类不是
严格可加的墙钟分区；GPU active 使用区间并集另算。

| 项目 | 8K，ms/round | 64K，ms/round |
|---|---:|---:|
| Trellis 投影与 rotation | **18.38（75.0%）** | **18.57（60.3%）** |
| Attention / KV，含 drafter SWA 与 RoPE | **1.67（6.8%）** | **7.83（25.4%）** |
| Drafter NVFP4 投影 | 2.11（8.6%） | 2.12（6.9%） |
| GDN recurrence / convolution / gating | 1.57（6.4%） | 1.60（5.2%） |
| 其余 norm、采样、elementwise | 0.38（1.6%） | 0.39（1.3%） |
| 整个 decode device 区间 / rounds | 24.49 | 30.78 |
| GPU kernel + memcpy 区间并集占比 | 97.85% | 98.68% |

Trellis 内部，body kernel 为 16.47 / 16.65 ms；两个 head 合计 1.46 / 1.48 ms；rotation 约 0.45 ms。
因此 output head 不是主项，去掉一个 head 的上限也远小于改善 body。

从 8K 到 64K 的每轮 device 时间增加 6.29 ms，attention/KV 增加 6.16 ms，解释了约 98% 的增量。
按 hq-e8-2b 的 9,216 bytes/token，额外 KV payload 是约 528 MB，除以这个时间得到约 **86 GB/s**。
这是压缩 KV 有效数据量的等效速率，**不是 DRAM counter**；重复读取、解码、计算、归约都在该时间内。

双 lane capture 的 trellis body 仍约 16.8 ms/全局 round；说明两条 lane 在同一遍模型计算中分摊权重读取。
但 attention、drafter 和 GDN 成本增加，整体没有接近两倍的吞吐收益。8K/64K 双 lane capture 的
attention/KV 分别占区间约 10.6% / 35.4%；该区间也含另一条 lane 的短尾部 prefill 和结束时的单 lane。

### CPU 与拷贝：API 时间不等于 PCIe 瓶颈

在单 lane decode capture 中，`cudaMemcpyAsync` 的 CPU API 时间累计为约 23.3 / 29.7 ms/round，
看起来几乎覆盖整个 round；但真正 device memcpy 仅约 **0.074 / 0.078 ms/round**，GPU 约 98% 的时间
有 kernel 或 copy 执行。CPU API 时间与 GPU 执行重叠，包含等待之前工作的时间，不能把它直接算成
PCIe 传输或 CPU 调度开销。

`cudaGraphLaunch` 的累计 CPU 时间约 0.10–0.20 ms/round。当前证据支持优先优化 GPU 上的 body 和
attention；消除可见的 GPU 空闲只能回收约 1–2% 的区间，不足以解释 SOL 差距。这里没有 CPU sampling
或 NCU stall counter，未断言所有 CPU 工作都无优化空间。

### Cold prefill

| Nsight 冷请求，首 kernel 至首 decode graph | 8K | 64K |
|---|---:|---:|
| Device 区间 | 3.966 s | 34.744 s |
| Trellis kernels，累计 | 3.379 s（85.2%） | 26.961 s（77.6%） |
| Attention / KV，累计 | 0.124 s（3.1%） | 3.829 s（11.0%） |
| GDN，累计 | 0.248 s（6.3%） | 1.969 s（5.7%） |
| GPU kernel + memcpy 区间并集占比 | 99.45% | 99.15% |

Cold prefill 在这两个长度仍以 trellis GEMM 为主，64K attention 已开始明显增长。
HTTP 未采样的速率约 2.07K / 1.90K prompt tok/s。

现有 `IGNIS_CHUNK_PROFILE` 与 `IGNIS_CHUNK_PROFILE_LAYERS` 同时记录 CUDA events。
调度器每次只提交一个 chunk，所以各记录的 `span_tokens` 通常为 1,024、`chunk=0`、`entry_gap_ms=0`；
这个零值不能当作 chunk 间没有等待的证据。跨 chunk 的判断使用 Nsight 时间线。

大多数 trellis prefill kernel 的 capture 资源为 256 threads、235–255 registers/thread、
100,352 bytes dynamic shared memory，grid 64 或 84 CTAs。资源占用与此前 5090 观察到的一 SM 一 CTA
相符，但本次没有采集 NCU 的 tensor pipe、issue 或 stall 指标，不把此前的 stall 结论当作 5080 实测。

### 混合负载的输出停顿

64K A 单独 decode 约 133 tok/s；在它开始后 300 ms 加入约 8K 冷请求，重复两次：

- 已在运行的 64K 请求降到 **63.8 tok/s**，decode span 约 8.01 s。
- 两次均出现 **8 个超过 100 ms 的 content-event 间隔**，最大为 **523.6 / 525.0 ms**。
- 新的 8K 请求均为零缓存，第一轮的 TTFT 为 5.56 s。

[ADR 0018](../adr/0018-chunk-level-prefill-decode-interleaving.md) 的 interleaving 在 chunk 边界发生。
本次 1,024-token prefill chunk 需要约半秒，与这八次停顿对应。对交互体验，这是平均 tok/s 之外的主要问题。

### 与 SOL 的距离

[NVIDIA 的 RTX 5080 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)
给出 960 GB/s。现有 `ignis_kernel_trellis_linear_bench` 在独占 GPU 下运行：

```sh
kernel/build/tests/ignis_kernel_trellis_linear_bench --caps 0 --calls 32 --reps 3
```

该工具写死了 **5090 的 1,792 GB/s 和 419 TFLOPS**；原输出中的 `%roof` 和该 TFLOPS 参照值不能用于 5080。
以下重新使用 960 GB/s，或同次运行的实际 streaming/MMA loop 作为分母。

| 测量 | 结果 | 相对上限 |
|---|---:|---:|
| 1 GiB streaming read | **913 GB/s** | 5080 规格的 95.1% |
| Body K mix，1 token，9,135.6 MB | **14.060 ms，650 GB/s** | 规格 67.7%；streaming 71.2% |
| 同一 body，8 tokens | **15.485 ms，590 GB/s** | streaming 64.6%；1 token 时间的 1.101 倍 |
| 服务中 8K 的 body，8-row verify | **16.468 ms，约 555 GB/s** | streaming 60.8% |
| 大矩阵 prefill，1,024 tokens | **约 119–131 TFLOPS** | 实测 register MMA loop 的约 49–54% |
| Register-only FP16-accumulate MMA loop | **242.2 TFLOPS，2,812 MHz** | 实测计算参照 |

服务中的 GB/s 以模型唯一权重字节量除以 kernel 时间，是有效权重吞吐，并非硬件实际 DRAM traffic。
微基准会轮换超过 L2 容量的权重；它与服务的热状态、矩阵序列和中间张量访问仍不同。该微基准只有一次
进程运行，内部三次 repetitions，未把某个矩阵的异常宽度比值当作确定的根因。

就 body 而言，9.136 GB / 913 GB/s 的理想流式读取时间约 **10.0 ms**，实测服务约 **16.5 ms**，
所以 body 尚未接近自己的带宽下界。即使把 body 降到这个理想值，其余成本不变，整个 8K round
仍约 18 ms；这是乐观边界，不是承诺可实现的性能，也不是整个推理路径的严格 SOL。

### 测量扰动、温度与检查

Profile 热单 lane 的 HTTP decode 为 152.25 tok/s（8K）、131.73 tok/s（64K），相对 baseline
中位数约慢 1.0% / 0.8%。Cold profile 的 CPU/GPU 计时也有开销；表中的服务性能只取不启用 profiler 的结果。

按请求时间窗口截取 200 ms 的 `nvidia-smi` 采样，第一次 baseline：

| 区间 | 功率中位数 | SM clock 中位数 | 温度中位数 |
|---|---:|---:|---:|
| 8K cold prefill | 357 W | 2,700 MHz | 76 °C |
| 64K cold prefill | 360 W | 2,670 MHz | 82 °C |
| 8K 单 lane decode | 319 W | 2,737 MHz | 72 °C |
| 64K 单 lane decode | 318 W | 2,745 MHz | 76 °C |

Prefill 接近功率限制；没有固定时钟，也没有采集完整 throttle-reason counters，不能只根据温度断言热降频。
第二次独立启动的 decode 复现降低了温度或单次启动偶然性对结论的影响。

最后运行现有 `ignis-bench canary`：四项 sane、四项 greedy deterministic，self-consistency PASS；
两项达到答案、两项在 thinking 中耗尽预算。这是现有 canary 的通过口径，不是完整质量验证。
结束后 `make stop`，GPU 为 0 MiB、无 compute process。

本任务没有代码变更，未重跑 workspace `cargo test` 或完整 GPU profile；本次 canary 不替代它们。

### 复现与原始证据

远端与 Laptop 的 `.scratch/profile-5080-2026-11-03/` 保存：

- `workload.py`、`mixed.py`、`profile-session.py`、`profile-only.py`、`confirm-session.py`：精确请求与运行命令；
- `prompts.json`、`baseline*-requests.jsonl`、`mixed-requests.jsonl`、`profile-requests.jsonl`：请求、文本、SSE 时间；
- `baseline*-server.log`、`profile-server.log`、每组请求前后的 `.prom`、GPU `.csv`；
- `chunks.jsonl`、`trellis-microbench.txt`、`trace-summary.json`、`head-body-summary.json`；
- `analyze-traces.py`、本地 `plot.py` 与 `profile-summary.png/.svg`，以及 `canary.json`。

七个 `.nsys-rep` 与导出的 `.sqlite` 保留在远端同一目录，不入库。本次使用
`--discard-environment=true`，仍按仓库规则把 capture 留在 scratch。

Nsight 的关键命令如下；server 仍由 `make start` 的 daemon hook 管理，完整 hook 在 `profile-only.py`：

```sh
nsys launch --session-new=ignis5080 --trace=cuda,nvtx --cuda-graph-trace=node \
  target/x86_64-unknown-linux-gnu/release/ignis-server <recorded-server-flags>
nsys start --session=ignis5080 --sample=none --cpuctxsw=none \
  --discard-environment=true --force-overwrite=true -o <capture-path>
# 执行该 capture 对应的一个请求或一个并发请求对。
nsys stop --session=ignis5080
nsys export --type=sqlite --force-overwrite=true --output=<capture.sqlite> <capture.nsys-rep>
```

## Finding

实测结论：

1. **8K coding 的首要 kernel 成本是 trellis body**；它在服务中只达到实测 streaming 带宽约 61% 的
   有效权重吞吐。整条 decode 路径仍有明显 SOL 差距。
2. **长上下文的增量主要来自 attention**：8K 到 64K 每轮多出的约 6.3 ms，约 6.2 ms 来自 attention/KV。
3. **接受率与 kernel 速度必须分开看**：同一构建从约 295 到 154 tok/s，主要可以由约 6.7 到 3.7
   tokens/round 的变化解释；不能用单个短提示的 tok/s 代表 coding agent。
4. **双 lane 能分摊 body 权重成本，但收益有限**：当前 workload 聚合提升约 42–47%，每条 lane 变慢。
5. **冷 prefill 和混合负载体验仍需改善**：64K 冷 TTFT 约 34.4 s；1,024-token chunk 在混合负载中
   带来约半秒的输出间隔。GPU 有工作执行的时间比例很高，降低 host dispatch 不是当前最有证据的主方向。

## Implications

根据测量推导的优化顺序，尚未验证优化收益：

| 优先方向 | 证据与下一步要回答的问题 |
|---|---|
| Trellis body decode / verify | 8K round 的主成本；在 5080 上采集受支持 NCU 的 memory、issue、stall、occupancy counters，区分解码指令、寄存器压力和访存延迟。比较 1/8/16 rows，保留数值正确性门槛。 |
| 长上下文 GQA | 64K 单 lane 已占约四分之一，双 lane约三分之一；检查 KV 解码、query 复用、split/reduction 和实际 DRAM traffic。不能用 KV payload 的 86 GB/s 直接证明 DRAM 本身慢。 |
| 混合负载的 chunk 策略 | 半秒停顿已经直接可见；用相同请求比较 256/512/1,024 chunks 的输出间隔、cold TTFT 和聚合吞吐。较小 chunk 是否划算，需要 A/B，不能只看延迟改善。 |
| 投机窗口与接受率 | 当前源码 workload 只有约 3.7–4.0 tokens/round；按真实任务比较 draft 长度和每轮时间，以最终 tok/s 判定，不能只追求 acceptance 百分比。 |
| Prefill trellis GEMM | 8K/64K 都占大头，大矩阵只到实测 MMA loop 的约一半；验证资源占用与 pipeline/解码/fold 指令的限制。 |

例如只把 trellis 全部成本减半，按当前占比的 Amdahl 估计，8K/64K round 的理想加速约 1.60 / 1.43 倍。
这是按已测占比计算的上界示例，不是性能预测。任何改变 reduction、quant 解码或 speculative 数值路径的优化，
仍需仓库要求的正确性测试和 GPU gates；本报告不改变这些要求。

## Limits and unknowns

- 仅 Qwen3.8-27B trellis 文本路径；未测 vision、speech、Flash-Next、128K/256K 或超过两条 lane。
- 源码上下文与三个内置 agent fixture 可复现，但不是生产 agent trace；未统计真实用户的上下文和任务分布。
- 未与其他引擎 live/live 比较，也未测关闭 speculation 的端到端 A/B。
- Nsight Systems 提供时间与资源信息，不提供此次未采集的 NCU 硬件 stall、实际 DRAM bytes、tensor 利用率。
- 吞吐主结论有两次独立启动；kernel 分解和微基准各一次进程运行，没有跨启动置信区间。
- 固定 512-token coding 测量是性能控制；不验证代码答案质量。正常工具调用必须允许停止，已剔除强制 EOS 的无效 agent 样本。
- 结果针对记录的二进制，不包含同时进行中的 generation timing 或 coding-agent harness 改动。

## Follow-ups

本次未创建跟踪 issue；上述优化方向是依据当前证据的建议，尚未实施。
