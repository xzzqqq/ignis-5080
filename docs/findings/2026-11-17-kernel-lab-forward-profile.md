# 优化后的 5080 coding-agent 前向：decode 成本分布

- Kind: experiment
- Status: current
- Observed: 2026-11-17
- Last verified: 2026-11-17
- Scope: kernel / RTX 5080 Qwen3.8-27B Kernel Lab full-forward prefill/decode attribution
- Related: [静态 decode 与 down reduction](2026-11-16-trellis-static-decode-and-down-reduction.md), [此前算子 profile](2026-11-07-decode-round-operator-profile.md), [Kernel Lab](../user/contributing/kernel-lab.md)
- Superseded by: none

## Question

经过 Trellis 静态调度和 T=8 down reduction 优化后，固定 coding-agent 请求的完整前向时间花在哪里，下一步应优先测哪些算子？

## Evidence

### 条件与覆盖

- 源码 HEAD `ff0989a0c84c`，推理源码没有未提交修改；本次未修改推理代码。
- RTX 5080，84 SM，Linux；`qwen3_8_27b_trellis_sc30-v2.ninfer`，`hq-e8-2b`，DFlash2，context 65,536，prefill chunk 1,024，配置两条 lane、实际 C1。
- 固定 `astropy__astropy-12907`，19,954 个输入 token；保留原请求、greedy、原 8,192 输出上限，自然停止于 913 个输出 token，没有诊断截断。187 轮，平均 4.882 个输出 token/round；Lab 的 4.888 emitted/round 另计最终 stop token。
- 当前候选 `a7d2044a6f34ce2742e6e9498ddfe40daab207895db08aec70f8f3014d4e3615`；baseline `412541a1e7d9c8853c1ce81cd0bc11c3f2c76dac7311f7b8e29e581e038109a6`。
- 开启 Nsight 的 Lab 中执行一次 `profile --forward --candidate current`，0 warmup、1 sample。没有录制算子 corpus。随后关闭 Nsight，重开同一 baseline，执行一次普通 `forward --candidate current`。
- NVTX、CUDA API correlation 和 graph-node ancestry 将全部 270,743 次 request kernel 执行归到 phase/operator；265,909 次有层身份，其他是显式 nonlayer 算子。未归类、unplaced 和 setup kernel 均为 0。prefill 30,507 次、decode 240,236 次。图的重新 capture 在本次没有执行 setup kernel。
- 每次启动/测量前检查 `make gpu-status`；运行期间只有本任务引擎。完成后 `lab.py stop`，GPU 无计算进程、0 MiB 使用。

### 前向墙钟

| 指标 | Nsight profile | 关闭 Nsight 的 control |
|---|---:|---:|
| Prefill | 8,670.146 ms | 8,630.549 ms |
| Decode | 4,176.491 ms | 4,156.004 ms |
| Decode 平均每轮 | 22.334 ms | 22.225 ms |
| Decode 输出速率 | 218.605 tok/s | 219.682 tok/s |

profile 相对 control 的 prefill/decode 墙钟分别高 0.46%/0.49%。两次的完整 token 序列、停止原因、stop token 和 prefill logits SHA-256 一致。profile 本身没有已有 builtin reference，因此其 `numerical_match` 是 null；普通 forward 的 builtin reference 比较通过。这里的跨运行核对是额外证据，不能解释成 profile 已执行独立数值 oracle。

### Decode 阶段

下表按 187 轮平均。累计 kernel duration 与 GPU 首尾跨度分列；占比分母是 decode 的全部累计 kernel duration，不能与跨度、间隙直接相加。

| 阶段 | Kernel 累计 ms/round | GPU 跨度 ms/round | Kernel 占比 |
|---|---:|---:|---:|
| Draft（含 draft head） | 2.679 | 2.716 | 12.12% |
| Verify（含 target head、argmax、accept） | 18.706 | 18.738 | 84.64% |
| Commit（GDN fold、hidden selection、draft context append） | 0.714 | 0.753 | 3.23% |
| KV invalidate（stage 外） | 0.0006 | — | 0.003% |
| 合计 | 22.100 | — | 100% |

每轮 kernel 首尾跨度平均 22.234 ms，kernel 并集 21.868 ms，轮内 kernel 间隙 0.366 ms，重叠超额 0.232 ms。完整 decode GPU 时间线跨度为 4,175.976 ms，其中 kernel 间隙含轮间空档共 86.662 ms；再计 memcpy/memset 后剩余无 GPU activity 覆盖时间为 73.128 ms（1.75%，平均 0.391 ms/round）。这些间隙不能仅凭 Nsight 时间线归因为 CPU 或同步。

### Decode 算子成本

| 算子组 | Kernel ms/round | Kernel 占比 |
|---|---:|---:|
| 主模型 MLP gate/up + SwiGLU | 5.541 | 25.07% |
| 主模型 attention 相关线性投影（GDN QKV/Z/output、GQA Q/K/V/output） | 4.865 | 22.01% |
| 主模型 GQA attention | 3.361 | 15.21% |
| 主模型 MLP down（含 residual） | 2.980 | 13.48% |
| Draft body | 1.964 | 8.89% |
| 主模型 GDN 状态、norm、conv 等 | 1.218 | 5.51% |
| Draft output head | 0.715 | 3.23% |
| Commit | 0.714 | 3.23% |
| 主模型 output head | 0.713 | 3.22% |
| Argmax + accept | 0.020 | 0.09% |
| 主模型其他、KV invalidate | 0.008 | 0.04% |

同一层的不同轮分别计算跨度，未把全请求第一轮和最后一轮包成一个 layer latency。48 个主模型 GDN 层平均每层每轮 kernel 228.322 µs；16 个 GQA 层平均 437.889 µs。GDN 第 57 层的平均跨度为 269.210 µs，GQA 第 59 层为 465.324 µs；不同层的权重/位宽不同，不能把层间差异直接当作调度退化。

进一步拆分热点：

- 主模型 gate/up 的 `mlp_static4_kernel` 为 5.460 ms/round，接近该组全部成本。
- GQA attention partial 为 3.100 ms/round，reduce/output 为 0.261 ms/round。
- 已采用的 `down_t8_partial_kernel` 为 2.343 ms/round；`down_t8_reduce_kernel` 仅 0.075 ms/round。这只覆盖专用 down 路由，不能替代所有 down 的 2.980 ms。
- GDN QKV、Z、output 分别为 1.428、1.028、1.002 ms/round；GQA Q、K、V、output 分别为 0.734、0.180、0.160、0.333 ms/round。
- 两个 output head 共 1.427 ms/round（6.46%）；GDN record 为 0.614 ms/round，commit fold 为 0.360 ms/round。

### Prefill

19,954 tokens 的 prefill kernel 累计 8,592.427 ms。主模型 gate/up 共 3,573.396 ms（41.59%），down 共 1,811.956 ms（21.09%），其余主模型线性投影共 2,359.669 ms（27.46%），GQA attention 436.358 ms（5.08%）。本请求的长 prefill 仍主要花在 Trellis 线性操作；该分布不是稳定 decode 的分布。

### 复现和原始结果

从 Laptop 执行，使用保存的配置；所有 GPU 状态检查均须确认没有其他任务占卡：

```sh
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/lab.py setup --config .scratch/forward-profile-2026-11-17/lab-settings.json --new-baseline --enable-profiling
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/lab.py profile --forward --candidate current
python3 scripts/lab.py stop
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/lab.py setup
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/lab.py forward --candidate current
python3 scripts/lab.py stop
python3 scripts/remote.py exec -- make gpu-status
```

配置、命令日志、结果副本、压缩 timeline、SQLite、分析脚本和 CSV 保存在 [本次 scratch 报告](../../.scratch/forward-profile-2026-11-17/report.md)。原始 profile operation 为 `909b43542c9c49b6a2e471a1039e72c3`，普通 forward 为 `b9de1399af9b4a60ab2dc106286a2edf`。`result.json` 的 `profile.artifacts` 保留远端 `.nsys-rep`、SQLite 和完整 timeline 的路径；这些原始文件不提交。

## Finding

在这个约 20K 输入、C1、T=8 验证的实际请求上，主模型 MLP 仍占 decode kernel 时间 38.56%，attention 相关线性投影占 22.01%，GQA attention 占 15.21%。已有优化后，大型 Trellis 算子仍是最大成本来源。专用 down reducer 只占每轮约 0.075 ms；继续围绕 reducer 本身优化的绝对收益上限很小。

## Implications

以下是根据成本排序的后续实验建议，不是已证明的速度收益：

1. 优先测主模型 gate/up 和 generic projection 的实际输入；以 `mlp_static4_kernel`、GDN QKV/Z/output 为起点，结合 Nsight Compute 判断指令、寄存器和访存瓶颈。仅凭本次 Systems 时长无法声称达到或偏离某个 SOL 百分比。
2. GQA attention partial 是下一条独立路线；在实际 20K 请求上占 14.03% 的 decode kernel 时间，需补短/长上下文控制才能判断长度扩展收益。
3. 两个 head 合计 6.46%，draft body 8.89%，可作为后续方向；本次不支持改变 speculation 或数值契约。
4. Down 如继续优化，应先关注 partial 的 2.343 ms，而非 reducer 的 0.075 ms。维持既定 K grouping、split 和独立数值检查。
5. 时间线剩余间隙约 1.75%，当前优先级低于上述大算子；要解释间隙原因需另做 CPU/API 分析。

## Limits and unknowns

单请求、单次 profile 和单次普通 forward；未锁时钟、没有置信区间。对照约 0.49% 的差异混合了 profiler 开销与运行波动。Engine timing 不含 HTTP 和 scheduler，不代表服务吞吐，也不能与此前不同 prompt/context 的 profile 直接计算优化收益。

本次没有改代码，未运行完整 CPU/GPU 正确性、capacity、Vision 或 canary suites；输出一致性仅覆盖该请求。没有采集 Nsight Compute 的带宽、occupancy 或指令计数。prefill 与 decode 的不同层级统计互相包含，不能跨层级相加。

## Follow-ups

下一轮优化先冻结各算子的契约和真实输入，使用普通 Lab replay/compare 做受控测量；候选收益由同请求 fresh baseline 的 forward 比较确认。代码接受仍按 [测试要求](../agents/testing.md) 和独立数值参考执行。
