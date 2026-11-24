# RTX 5080 投机 decode round 的算子级成本

- Kind: experiment
- Status: current
- Observed: 2026-11-07
- Last verified: 2026-11-07
- Scope: kernel / Qwen3.8-27B trellis speculative decode operator attribution and MLP hardware counters
- Related: [此前 workload profile](2026-11-03-rtx5080-workload-profile.md), [coding-agent 基线](2026-11-03-coding-agent-5080-baseline.md)
- Superseded by: none

## Question

一轮 DFlash2 起草、主模型验证和状态提交的成本分布是什么？哪些算子值得优先优化？

## Evidence

### 版本与条件

- 源码 `f09c53f235784a2cbad23bd05008e1f163cfb924`；本次没有修改推理代码。远端 sync 报告 0 changed source files。
- server SHA-256：`82d0901cb59afdc1bf5184d23ff1463e41218e5e711685af9da39fa7dd08930c`。
- RTX 5080 16 GB，Linux，`rtx50-16gb`，hq-e8-2b，DFlash2 7 drafts/full head，两条配置 lane，prefill chunk 1024，Vision 默认加载但没有图片。
- 复用此前冻结的源码 coding prompts；A/B 长度分别为 8192/8193、65534/65536。temperature=0、seed=0、thinking off，固定生成512 tokens，ignore_eos=true，prompt reuse 开启。
- C1/C2 表示一/两条同时请求，不改变模型配置的两条 lane。每个场景先准备 A/B 前缀，再测热请求；尾部 prefill 不算作稳定 decode 的 kernel 工作。
- 不带 profiler 的基准各3次；Nsight Systems 2026.2.1 node trace 各1次，全部来自同一二进制。Nsight Compute 2026.3.1 独立测 MLP 微基准。
- 开始时另一 worktree 正跑基准；等待其完整流程结束且 make gpu-status 无进程/0 MiB 后才启动。本任务只停止自己 runtime 目录的 daemon。结束后 GPU 无计算进程。

### 未采样基准与采样扰动

| 上下文，C1 | decode 中位数 ms | tok/s | rounds | tokens/round | accepted/drafted | profile decode ms | 扰动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 3738.474 | 136.954 | 154 | 3.3247 | 33.68% | 3773.177 | +0.93% |
| 65536 | 3456.180 | 148.140 | 113 | 4.5310 | 50.63% | 3476.171 | +0.58% |

64K 此请求比 8K 的 tok/s 高，来自更高的每轮产出；它的每轮计算实际上更慢。不能把这两个不同输入当作严格的上下文长度因果 A/B。本次 seed=0 等协议不等同此前合成 workload，亦不等同固定 coding-agent benchmark，不报告对其的速度提升。

### 稳定 decode 时间线

相邻两个相同宽度 graph 的首 kernel 开始时间定义一轮窗口，包含 graph 外的提交和下一轮提交前间隙。排除最后一轮、宽度转换和混合 prefill 的窗口；C2 只展示宽度2稳定部分，尾部宽度1另保存在 summary.json。

| 指标 | 8K C1 | 64K C1 | 8K C2 | 64K C2 |
|---|---:|---:|---:|---:|
| 保留的全局 round 数 | 153.0000 | 112.0000 | 150.0000 | 111.0000 |
| 墙钟 ms/round | 24.4954 | 30.7571 | 29.8432 | 41.5248 |
| kernel 累计 ms/round | 24.2022 | 30.4612 | 29.4519 | 41.1851 |
| device memcpy 累计 ms/round | 0.0754 | 0.0776 | 0.0883 | 0.0836 |
| 无 kernel/copy/memset 覆盖的间隙 ms/round | 0.4380 | 0.4398 | 0.4573 | 0.4094 |
| GPU 活跃区间并集 % | 98.2121 | 98.5702 | 98.4676 | 99.0140 |

下面所有算子耗时是 kernel duration 累计/round。百分比分母是该场景的全部 kernel duration 累计，不是墙钟时间；PDL、拷贝和计算可能重叠，不能把这些数与间隙直接相加。

### 按阶段汇总

| 阶段，ms/round | 8K C1 | 64K C1 | 8K C2 | 64K C2 |
|---|---:|---:|---:|---:|
| draft | 2.7227 | 2.7264 | 4.2816 | 4.2864 |
| verify | 20.4826 | 26.7382 | 23.1982 | 34.9274 |
| accept | 0.0211 | 0.0208 | 0.0382 | 0.0378 |
| commit | 0.9758 | 0.9759 | 1.9338 | 1.9335 |

draft 包含 proposal full head；verify 包含 target head；commit 包含 GDN fold、KV 有效位、hidden selection 和 drafter context append。GPU copy 另计。

### 完整算子组明细

每个 graph 按源码执行顺序及 kernel 名称断言映射到64层；GDN 在T=8使用 gating partial+reduce，在T=16使用单一 gating kernel，分析器显式处理该差异。保留逐调用的层编号、graph node、kernel 名称、寄存器和shared memory；未对融合 kernel 内部虚构耗时拆分。表中小数保留4位用于复核，不表示该级精度可跨运行复现。

| 算子组 | 8K C1 ms (%kernel) | 64K C1 ms (%kernel) | 8K C2 ms (%kernel) | 64K C2 ms (%kernel) |
|---|---:|---:|---:|---:|
| `accept.accept` | 0.0048 (0.02%) | 0.0049 (0.02%) | 0.0069 (0.02%) | 0.0070 (0.02%) |
| `accept.argmax` | 0.0163 (0.07%) | 0.0159 (0.05%) | 0.0313 (0.11%) | 0.0308 (0.07%) |
| `commit.draft_context_append` | 0.3662 (1.51%) | 0.3665 (1.20%) | 0.6162 (2.09%) | 0.6165 (1.50%) |
| `commit.gdn_fold` | 0.6072 (2.51%) | 0.6071 (1.99%) | 1.3146 (4.46%) | 1.3140 (3.19%) |
| `commit.kv_validity` | 0.0009 (0.00%) | 0.0008 (0.00%) | 0.0017 (0.01%) | 0.0015 (0.00%) |
| `commit.select_hidden` | 0.0015 (0.01%) | 0.0014 (0.00%) | 0.0014 (0.00%) | 0.0014 (0.00%) |
| `draft.attention_output` | 0.1251 (0.52%) | 0.1253 (0.41%) | 0.2014 (0.68%) | 0.2015 (0.49%) |
| `draft.dynamic_conv` | 0.0204 (0.08%) | 0.0204 (0.07%) | 0.0232 (0.08%) | 0.0232 (0.06%) |
| `draft.dynamic_conv_projection` | 0.1096 (0.45%) | 0.1097 (0.36%) | 0.2465 (0.84%) | 0.2467 (0.60%) |
| `draft.embedding` | 0.0151 (0.06%) | 0.0153 (0.05%) | 0.0398 (0.14%) | 0.0410 (0.10%) |
| `draft.full_head.multiply` | 0.7246 (2.99%) | 0.7250 (2.38%) | 0.7521 (2.55%) | 0.7525 (1.83%) |
| `draft.full_head.rotation` | 0.0015 (0.01%) | 0.0015 (0.00%) | 0.0013 (0.00%) | 0.0013 (0.00%) |
| `draft.mlp.down` | 0.4413 (1.82%) | 0.4420 (1.45%) | 0.7575 (2.57%) | 0.7584 (1.84%) |
| `draft.mlp.gate_up_swiglu` | 0.8925 (3.69%) | 0.8938 (2.93%) | 1.5999 (5.43%) | 1.6013 (3.89%) |
| `draft.norm` | 0.0223 (0.09%) | 0.0223 (0.07%) | 0.0227 (0.08%) | 0.0227 (0.06%) |
| `draft.prepare` | 0.0012 (0.01%) | 0.0012 (0.00%) | 0.0012 (0.00%) | 0.0012 (0.00%) |
| `draft.qkv_projection` | 0.1912 (0.79%) | 0.1914 (0.63%) | 0.3627 (1.23%) | 0.3631 (0.88%) |
| `draft.residual` | 0.0101 (0.04%) | 0.0101 (0.03%) | 0.0103 (0.03%) | 0.0103 (0.02%) |
| `draft.rope` | 0.0061 (0.03%) | 0.0061 (0.02%) | 0.0063 (0.02%) | 0.0064 (0.02%) |
| `draft.selector` | 0.0218 (0.09%) | 0.0218 (0.07%) | 0.0402 (0.14%) | 0.0402 (0.10%) |
| `draft.sliding_window_attention` | 0.0902 (0.37%) | 0.0903 (0.30%) | 0.1399 (0.48%) | 0.1399 (0.34%) |
| `draft.topk` | 0.0498 (0.21%) | 0.0501 (0.16%) | 0.0764 (0.26%) | 0.0764 (0.19%) |
| `verify.embedding` | 0.0049 (0.02%) | 0.0049 (0.02%) | 0.0267 (0.09%) | 0.0265 (0.06%) |
| `verify.final_norm` | 0.0012 (0.01%) | 0.0012 (0.00%) | 0.0013 (0.00%) | 0.0013 (0.00%) |
| `verify.gdn.conv_silu` | 0.1358 (0.56%) | 0.1367 (0.45%) | 0.1437 (0.49%) | 0.1432 (0.35%) |
| `verify.gdn.gated_norm` | 0.0559 (0.23%) | 0.0559 (0.18%) | 0.0561 (0.19%) | 0.0561 (0.14%) |
| `verify.gdn.gating` | 0.2328 (0.96%) | 0.2342 (0.77%) | 0.3976 (1.35%) | 0.3976 (0.97%) |
| `verify.gdn.input_norm` | 0.0567 (0.23%) | 0.0569 (0.19%) | 0.0600 (0.20%) | 0.0600 (0.15%) |
| `verify.gdn.mlp.down.multiply_residual` | 3.1478 (13.01%) | 3.1523 (10.35%) | 2.5152 (8.54%) | 2.5132 (6.10%) |
| `verify.gdn.mlp.down.rotation` | 0.0750 (0.31%) | 0.0752 (0.25%) | 0.0884 (0.30%) | 0.0883 (0.21%) |
| `verify.gdn.mlp.gate_up.multiply_swiglu` | 5.0033 (20.67%) | 5.0066 (16.44%) | 5.5899 (18.98%) | 5.5907 (13.57%) |
| `verify.gdn.mlp.gate_up.rotation` | 0.0606 (0.25%) | 0.0607 (0.20%) | 0.0684 (0.23%) | 0.0684 (0.17%) |
| `verify.gdn.output.multiply_residual` | 1.3486 (5.57%) | 1.3567 (4.45%) | 1.1254 (3.82%) | 1.1280 (2.74%) |
| `verify.gdn.output.rotation` | 0.0629 (0.26%) | 0.0630 (0.21%) | 0.0672 (0.23%) | 0.0672 (0.16%) |
| `verify.gdn.post_norm` | 0.0576 (0.24%) | 0.0576 (0.19%) | 0.0580 (0.20%) | 0.0580 (0.14%) |
| `verify.gdn.qkv.multiply` | 1.4213 (5.87%) | 1.4264 (4.68%) | 1.5987 (5.43%) | 1.6003 (3.89%) |
| `verify.gdn.qkv.rotation` | 0.0596 (0.25%) | 0.0599 (0.20%) | 0.0609 (0.21%) | 0.0610 (0.15%) |
| `verify.gdn.recurrence_record` | 0.6156 (2.54%) | 0.6222 (2.04%) | 1.1398 (3.87%) | 1.1410 (2.77%) |
| `verify.gdn.z.multiply` | 1.0535 (4.35%) | 1.0588 (3.48%) | 1.2136 (4.12%) | 1.2163 (2.95%) |
| `verify.gdn.z.rotation` | 0.0597 (0.25%) | 0.0599 (0.20%) | 0.0616 (0.21%) | 0.0616 (0.15%) |
| `verify.gqa.attention.partial` | 1.3384 (5.53%) | 7.1383 (23.43%) | 2.7525 (9.35%) | 14.0958 (34.23%) |
| `verify.gqa.attention.reduce_gate` | 0.2166 (0.90%) | 0.5578 (1.83%) | 0.3622 (1.23%) | 0.8046 (1.95%) |
| `verify.gqa.input_norm` | 0.0188 (0.08%) | 0.0189 (0.06%) | 0.0199 (0.07%) | 0.0199 (0.05%) |
| `verify.gqa.k.multiply` | 0.1623 (0.67%) | 0.1895 (0.62%) | 0.2291 (0.78%) | 0.2300 (0.56%) |
| `verify.gqa.k.rotation` | 0.0198 (0.08%) | 0.0198 (0.06%) | 0.0200 (0.07%) | 0.0200 (0.05%) |
| `verify.gqa.mlp.down.multiply_residual` | 1.0450 (4.32%) | 1.0584 (3.47%) | 0.9028 (3.07%) | 0.9101 (2.21%) |
| `verify.gqa.mlp.down.rotation` | 0.0254 (0.11%) | 0.0256 (0.08%) | 0.0298 (0.10%) | 0.0297 (0.07%) |
| `verify.gqa.mlp.gate_up.multiply_swiglu` | 1.8091 (7.47%) | 1.8378 (6.03%) | 2.0657 (7.01%) | 2.0792 (5.05%) |
| `verify.gqa.mlp.gate_up.rotation` | 0.0206 (0.08%) | 0.0206 (0.07%) | 0.0234 (0.08%) | 0.0234 (0.06%) |
| `verify.gqa.output.multiply_residual` | 0.6098 (2.52%) | 0.6156 (2.02%) | 0.5733 (1.95%) | 0.4877 (1.18%) |
| `verify.gqa.output.rotation` | 0.0229 (0.09%) | 0.0238 (0.08%) | 0.0244 (0.08%) | 0.0240 (0.06%) |
| `verify.gqa.post_norm` | 0.0193 (0.08%) | 0.0194 (0.06%) | 0.0196 (0.07%) | 0.0195 (0.05%) |
| `verify.gqa.q_gate.multiply` | 0.7319 (3.02%) | 0.7319 (2.40%) | 0.8414 (2.86%) | 0.8423 (2.05%) |
| `verify.gqa.q_gate.rotation` | 0.0197 (0.08%) | 0.0198 (0.06%) | 0.0202 (0.07%) | 0.0203 (0.05%) |
| `verify.gqa.qk_norm_rope` | 0.0308 (0.13%) | 0.0317 (0.10%) | 0.0331 (0.11%) | 0.0333 (0.08%) |
| `verify.gqa.v.multiply` | 0.1685 (0.70%) | 0.1687 (0.55%) | 0.1977 (0.67%) | 0.1979 (0.48%) |
| `verify.gqa.v.rotation` | 0.0201 (0.08%) | 0.0201 (0.07%) | 0.0198 (0.07%) | 0.0198 (0.05%) |
| `verify.head.multiply` | 0.7480 (3.09%) | 0.7484 (2.46%) | 0.7883 (2.68%) | 0.7887 (1.92%) |
| `verify.head.rotation` | 0.0013 (0.01%) | 0.0013 (0.00%) | 0.0013 (0.00%) | 0.0013 (0.00%) |
| `verify.prepare` | 0.0014 (0.01%) | 0.0014 (0.00%) | 0.0012 (0.00%) | 0.0012 (0.00%) |

### 热点的硬件计数器

使用现有 `ignis_kernel_trellis_linear_bench` 的 gate/up SwiGLU 形状：两个 `[17408,5120]` 矩阵、K=3；8/16个位置。权重使用该微基准的合成数据与超过L2的轮转pool，不是服务中的实际权重。通过过滤 trellis decode_kernel、calls=1/reps=1/no-prefill，以 launch-skip=1782/1802 选择8/16位置、warmup后的该形状。不是完整服务的计数器，也不包含单独的activation rotation kernel。

| NCU 指标 | T=8 | T=16 |
|---|---:|---:|
| multiply/fused epilogue 时间 μs | 112.06 | 126.53 |
| DRAM 吞吐 GB/s | 620.51 | 552.99 |
| DRAM 吞吐占 NCU 峰值 | 65.58% | 58.46% |
| SM throughput | 45.73% | 46.54% |
| registers/thread | 128 | 128 |
| 理论 occupancy（寄存器限制） | 33.33% | 33.33% |
| achieved occupancy | 29.90% | 29.87% |
| scheduler no eligible warp | 43.51% | 41.90% |

NCU kernel replay，cache-control=all，clock-control=none；每次多pass，不能拿NCU整段程序耗时计算吞吐。T=8的SM频率约2.67GHz。WarpStateStats提示未提供可选PC-sampling指标，未据此声称具体某条指令的stall占比。工具自动建议的speedup未作为收益预测。

### 复现与原始材料

本地与远端 `.scratch/decode-round-2026-11-07/`：
- `run.py`：固定请求、make start/stop、三个baseline repeats、四个nsys capture和导出。
- `prompts.json`、`baseline-*.json`、`profile-*.json`：实际输入和含ignis_timing的响应。
- `identity.json`、`environment.json`、server日志、GPU采样；`summary.json` 与四个 `.kernels.csv`。
- `analyze.py`：完整算子归因、区间并集和每层统计；`write_report.py`生成本文数表。
- Nsight `.nsys-rep/.sqlite/.ncu-repz` 留在远端，不提交；NCU details文本已回收本地。
- `untimed-preflight/`保存初次缺少ignis_timing的准备请求，不进入正式表格；初次make start因私有local.mk遗留speech参数失败，随后用本次ARGS覆盖。没有修改私有配置。

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- make build CUDA=1 UI=0 PROFILE=release
python3 scripts/remote.py exec -- make gpu-status
# 确认GPU空闲；run.py依赖该目录的prompts.json
python3 scripts/remote.py exec -- python3 .scratch/decode-round-2026-11-07/run.py
python3 scripts/remote.py exec -- python3 .scratch/decode-round-2026-11-07/analyze.py
```

NCU 在服务完全停止且 gpu-status 空闲后单独运行，精确参数：

```sh
ncu --kernel-name-base demangled --kernel-name "regex:ignis_tl::.*decode_kernel" \
  --launch-skip 1782 --launch-count 1 --kill yes --clock-control none --cache-control all \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section LaunchStats --section Occupancy \
  -o mlp-t8 kernel/build/tests/ignis_kernel_trellis_linear_bench \
  --caps 0 --calls 1 --reps 1 --no-prefill
# T=16: launch-skip=1802, output=mlp-t16, 不请求WarpStateStats
```

## Finding

1. 主模型MLP（含投影rotation）为8K C1的11.187 ms，占kernel累计46.22%；64K C1为11.237 ms、36.89%。它比两个head合计约1.475 ms更值得优先研究。
2. GQA attention为1.555→7.696 ms；整轮墙钟24.495→30.757 ms。不同上下文工作负载之间增加的6.262 ms，有6.140 ms体现在GQA attention上，约98%。
3. C2把主模型MLP从约11.2 ms仅提高到11.3 ms，但GQA从1.555→3.115 ms（8K）、7.696→14.900 ms（64K）；起草和状态提交也明显增长。批处理分摊主干权重，不能分摊每条序列自己的全部状态工作。
4. GPU无工作区间约0.41–0.46 ms/round；其占比约1.0–1.8%。即使全消除，也不是当前主要收益来源。该区间不等同纯CPU计算时间。
5. 隔离MLP微基准没有达到DRAM峰值；高寄存器用量限制occupancy且存在较多无eligible warp周期。寄存器/延迟隐藏/流水安排是有证据支持的研究方向，尚未证明降低寄存器一定会提速。

## Implications

优先研究主模型MLP trellis的权重解码、访存与MMA流水，以及长上下文GQA的KV读取/解码/归约。每次修改都需验证数值和完整服务收益；不能以单个kernel变快替代最终decode ms/token。GDN fold约0.61 ms/单lane、状态提交合计约0.98 ms，属于可见但次要成本。

## Limits and unknowns

- 四个capture各一次；三次重复仅覆盖未采样基准。使用正常boost，没有锁频，未提供跨天置信区间。
- 8K/64K是不同内容，C2尾部会转为C1；报告仅对稳定宽度区间计时，完整响应吞吐来自独立的ignis_timing。
- profile对C1 decode时间扰动约+0.93%/+0.58%；不能声称时间线完全无扰动。
- 主模型物理权重位宽按矩阵不同。kernel时间线不能直接读出每个实际矩阵的位宽；本报告按语义/层归因，未伪造逐矩阵DRAM流量。
- NCU仅采MLP两个代表形状的隔离kernel；未测服务中GQA硬件计数器、未证明具体指令stall原因。
- ignore_eos、512-token合成源码请求只用于性能控制；不评估答案质量、工具调用或正常EOS。没有跑新的优化验收、capacity或canary，因为没有修改推理实现。
- 不将本次结果冒充另一worktree正在研究的优化候选结果；版本固定为上述commit/二进制。

## Follow-ups

本次没有实施优化或创建issue。若开展优化，以上测量可作为算子定位证据，性能验收仍使用项目固定coding-agent基准。
