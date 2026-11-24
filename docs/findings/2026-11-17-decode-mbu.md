# RTX 5080 decode 的模型流量与 DRAM 带宽利用率

- Kind: experiment
- Status: current
- Observed: 2026-11-17
- Last verified: 2026-11-17
- Scope: kernel / RTX 5080 Trellis + DFlash2 C1 coding-agent decode MBU and DRAM traffic
- Related: [最新前向 profile](2026-11-17-kernel-lab-forward-profile.md), [早期 5080 profile](2026-11-03-rtx5080-workload-profile.md)
- Superseded by: none

## Question

以 RTX 5080 标称 960 GB/s 为分母，当前版本整条 decode 的 MBU 是多少？只统计 body 得出的约 71% 是否代表整轮？

## Evidence

### 身份和测量口径

源码 HEAD `ff0989a0c84cfd54688c729dc52b6e700dbf218b`，测量前的推理源码哈希与同日完整前向 profile 一致。沿用已冻结的 Kernel Lab engine `412541a1e7d9c8853c1ce81cd0bc11c3f2c76dac7311f7b8e29e581e038109a6`，加载候选 `a7d2044a6f34ce2742e6e9498ddfe40daab207895db08aec70f8f3014d4e3615`；候选二进制 SHA-256 为 `a8703289051611fca25183c631300dfb7e62db1962f39058125fde5178d4cc9f`。

硬件为单 RTX 5080 16 GB、84 SM，Linux，模型为 `qwen3_8_27b_trellis_sc30-v2.ninfer`。配置 context 65,536、prefill chunk 1,024、两条配置 lane、实际 C1、hq-e8-2b、DFlash2 七个 drafts/full head。固定 Astropy 请求输入 19,954 tokens，输出预算 8,192，自然停止于 913 tokens，共 187 decode rounds。计时覆盖同步 engine decode 调用，不含 prefill、HTTP 和 scheduler。

每次 GPU 运行前执行 `make gpu-status` 并检查 compute process 列表；runner 持有 `/tmp/ignis-gpu-profile.lock`。GPU 运行通过 `scripts/remote.py exec`，没有改推理实现或数值契约。

采用两个不同的指标：

- **模型流量 MBU**：固定 draft/verify/commit 算法的权重、唯一 KV 数据和持久状态读写负载，除以无 profiler decode 时间及 `960e9`。它包含混合架构和投机路径的工作，不把每个输出 token 当作完整权重遍历。
- **实际 DRAM 带宽利用率**：NCU 测得的 DRAM 读写字节，除以匹配轨迹的无 profiler decode 时间及 `960e9`。它包含实现的额外访问，不能直接解释成算法效率。

### 硬件计数器

使用 Nsight Compute 2025.3.1 的 `app-range` 模式，按 NVTX 的 `phase=decode_round` 范围采集整轮，保留 graph 内的并发与缓存行为。设置 `--cache-control none --clock-control none`，不逐 kernel 清缓存。有效指标为：

```text
dram__bytes.sum
dram__bytes_op_read.sum
dram__bytes_op_write.sum
gpu__time_duration.sum
```

先对 round 10/90/170 做两次有效采样，再完整采集 round 0..186。完整采集恰有 187 条记录，没有缺轮，读加写逐轮与 total 相等。两次代表轮采样相对完整采集同轮的总字节差异在 -0.057% 至 +0.074% 内。

首个探测使用了该设备不支持的 `dram__bytes_read.sum` / `dram__bytes_write.sum`，只返回时间、未返回流量；不计入有效带宽结果。随后用设备的 `--query-metrics` 确认上述 `op_read/op_write` 名称。该失败探测保留在 scratch。

完整计数器采集生成相同的 913 tokens、187 轮、停止原因和 prefill logits 哈希。NCU 使该次 decode 墙钟增至 11.030 s；这个受扰动时间没有用于正常运行的带宽计算。

| 完整 decode 的 DRAM 计数 | 字节 |
|---|---:|
| Read | 2,271,634,850,560 |
| Write | 41,386,139,392 |
| Total | 2,313,020,989,952 |
| 平均每轮 total | 12,369,096,203 |

### 无 profiler 的前后控制

计数器采集前后各启动一次同一 engine/candidate；每次一遍预热、三遍正式测量。六个正式 decode 时间为 4,152.841 / 4,160.991 / 4,162.762 / 4,162.058 / 4,163.432 / 4,164.712 ms，中位数 4,162.410 ms，平均每轮 22.259 ms，219.34 output tok/s。后组中位数比前组高 0.059%。全部八次无 profiler 前向及四次 profiler 前向的完整 token 序列、logits 哈希、停止原因与逐轮 position/emitted/extent 均一致。

总 DRAM 字节除以上述中位数，得到 **555.69 GB/s，即标称 960 GB/s 的 57.885%**。分别代入六次控制耗时，范围为 57.853%–58.018%；这只是控制运行的观测范围，不是统计置信区间。最终运行结束后 GPU 为 0 MiB、无 compute process。

### 模型流量账

直接读取本次模型的对象目录，按实际调用路径计数；Trellis 去除文件对齐 padding，保留压缩码和两组 scale。排除未运行的 Vision、MTP、shortlist head，host-mapped token embedding 不计入 GDDR 带宽。两个 full head 调用各计一次；commit 的 feature projection 和五层 QKV 也单列，不能漏算。

| 每轮平均逻辑负载 | GB，十进制 |
|---|---:|
| Target body Trellis（含 scale） | 9.135555 |
| Target body 其他权重及 final norm | 0.052486 |
| Target head | 0.636206 |
| Draft body | 0.937608 |
| Draft head | 0.636206 |
| Commit projections / norm | 0.162213 |
| Target KV 读取 | 0.218762 |
| Target KV 写入 | 0.000598 |
| Draft KV 读取 | 0.041943 |
| Draft KV 写入 | 0.000100 |
| GDN recurrent state | 0.452985 |
| GDN conv state 读写预算 | 0.008847 |
| Selector gather 上界 | 0.000032 |
| 合计 | 12.283542 |

Target KV 按每轮实际 position 统计，保留 32 sink + 512 recent 的 BF16 residual window，其余按 9,216 bytes/token 的压缩 KV。GDN recurrent state 为 `48 layers × 48 heads × 128 × 128 × 4` bytes；verify 读一次、commit 再读并写回一次，合计三份。Draft KV 按五层 2,048-token BF16 窗口计数。

这是明确口径的模型负载估算，不是每条 load/store 的枚举。Conv state 使用读写上界；残余窗口有效行数、selector 重复候选和末轮提交边界会产生小误差。即使把整个 conv 预算、八行 residual 差额和 selector 上界作为误差预算，其对 MBU 的影响也小于 0.05 个百分点。中间 activation、workspace 和重复访问不加入此逻辑分子，由独立 DRAM 计数体现；二者差值不能直接当成冗余流量，因为缓存也会降低外存访问。

### 原始证据与复现

[Scratch 目录](../../.scratch/mbu-2026-11-17/) 保存 `run.py`、`account.py`、`analyze.py`、artifact 对象目录、完整输入和输出、binary/source identity、控制计时、GPU 采样、NCU CSV、`rounds.csv` 和 `summary.json`。NCU 原始 `.ncu-rep` 留在远端同名目录；没有提交 profiler 原始文件。

```sh
python3 scripts/remote.py exec -- python3 .scratch/mbu-2026-11-17/run.py baseline
python3 scripts/remote.py exec -- python3 .scratch/mbu-2026-11-17/run.py ncuall
python3 scripts/remote.py exec -- python3 .scratch/mbu-2026-11-17/run.py after
python3 .scratch/mbu-2026-11-17/account.py
python3 .scratch/mbu-2026-11-17/analyze.py
```

Runner 自行执行每次运行前的 GPU 检查、锁定和清理。原始计数器命令保存在 `ncuall-command.json`。复现需要已有冻结 baseline、candidate 以及 prepared benchmark input。

## Finding

以六次无 profiler 控制的中位数计算，本次整轮模型流量 MBU 为 **57.48%（按负载账估算）**，实际 DRAM 带宽利用率为 **57.88%（计数器字节 / 匹配的控制耗时）**，分母均为标称 960 GB/s。报告到一位小数为 57.5% 和 57.9%。约 71% 只描述此前用 body 算子累计时间估算的权重吞吐，不能代表整条 decode。精确计算和逐轮表见 scratch 的 `summary.json` 与 `rounds.csv`。

## Implications

后续比较固定模型、上下文、DFlash2 设置、请求和输出轨迹，同时报告每轮时间、每轮输出数量、逻辑 MBU 和实际 DRAM 带宽利用率。主要 kernel 的峰值带宽成绩不能代替包含 attention、状态与 draft/commit 的整轮结果。

## Limits and unknowns

仅一个真实 coding-agent 请求、实际 C1、约 20K context，不代表其他上下文、C2 或关闭投机的结果。完整计数器覆盖一次请求；代表轮做了重复检查，没有第二次完整计数器 capture。使用正常 boost、未锁时钟；带宽字节与无 profiler 时间来自匹配轨迹的不同运行，硬件百分比是这种交叉测量的估计，不是无扰动在线 counter。保留 NCU 关于未控制缓存及频率的警告。

本次没有修改推理代码，未运行新的 workspace CPU suite、完整 GPU correctness、capacity、Vision 或 canary。重复运行的输出一致性不是独立数值 oracle；正确性验收仍由已有测试契约负责。

## Follow-ups

若扩展到其他输入长度、C2 或 speculation-off，应各自重新计数，不能直接外推本次约 58%。
