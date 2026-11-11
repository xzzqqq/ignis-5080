# RTX 5080 T=8 MLP：输出划分与任务调度共同决定 PDL 收益

- Kind: experiment
- Status: current
- Observed: 2026-11-11
- Last verified: 2026-11-12
- Scope: kernel / Qwen3.8-27B T=8 MLP scheduling and PDL
- Related: [固定服务测量协议](../user/contributing/coding-agent-benchmark.md)
- Superseded by: none

## Question

固定形状 MLP 的权重流量不变时，输出分工和任务调度如何影响执行效率及 PDL 收益？

## Evidence

范围为单张 84-SM RTX 5080、CUDA 13.0.88、T=8、gate/up 各为 5120→17408、
同位宽 2/3/4-bit、无 SM cap、一个 K split。4 warps×32 outputs 保持每 CTA
128 outputs、八位置权重复用、每个输出的 K 累加顺序及完整 128 维旋转。

- 8×16 改为 4×32 时，寄存器仍为 128/thread，权重读取与 MMA 执行计数不变；
  K3 主激活 shared 读取的逻辑字节数约减半，总 warp 指令减少约 15.7%。
- 4×32 persistent 的同组对照中，PDL off/on 完整操作为 88.320/132.874 μs。
  PDL on 的任务计数诊断出现 16 个 CTA 各完成两份、16 个没有任务；off 时
  272 个 CTA 各一份。源码在完成当前任务前预留下一份任务。
- 静态映射删除调度 ticket，grid 始终覆盖全部 272 份任务。另一组 K3 对照中，
  persistent off / static off / static on 为 88.621/87.650/84.544 μs。
  静态版本每个 CTA 各一份任务，但不保证覆盖全部 SM。
- 在 static on 上，位宽专门化将 K3 寄存器降至 96/thread、无 spill，操作却从
  84.635 增至 95.716 μs。另一布局在早期对照中使 shared wavefronts 减少约 23%，
  放到 static on 基线、实际 9/49/6 层位宽混合下仍约慢 5 μs/round。

服务按固定协议进行 A/B/A/B，每次启动完整预热。两阶段相对各自基线的 C1 decode
吞吐分别提高 3.1–3.3% 和 1.04–1.18%；不能相加当作直接测得的总收益。
每阶段 96 个响应在排除生成的 tool-call ID 后，内容、token 数和投机计数一致。
正式回归另以 fp64 oracle 校验真实形状的同位宽 T=8 输出，并验证 cap fallback
逐位一致，以及不同调用共享 workspace 的连续 graph replay。

可复核源码链为 `1435735a` → `714e7ee7` → `b4201d9f`；后两者为实验冻结快照。
派生结果和请求保存在 `.scratch/mlp-output-2026-11-11/`、
`.scratch/mlp-static-2026-11-11/`，完整实验报告也归档于其中。
隔离操作按归档 request 及其冻结源码复现；服务沿用上面的固定协议。
阶段对照使用独立构建目录；原始 profiler 捕获不进入 Git。

## Finding

1. 输出分工可以在权重流量、每线程寄存器数及理论驻留 warp 数不变时改善性能。激活复用和指令计数
   支持执行效率改善这一解释，尚未隔离各项改动的独立贡献。
2. PDL 与 persistent 任务预留存在交互。静态映射消除了同一 CTA 领取多份任务与空 CTA，
   并使 PDL 恢复正收益；现有证据未唯一确定硬件仲裁或资源竞争的时间占比。
3. 更少寄存器或 shared wavefronts 不保证完整操作更快。优化组合需要在最终
   基线上重测；本次专门化和布局收益没有成立。

## Implications

调整 CTA/warp 输出分工时，同时验证任务调度与 PDL。选优依据包含输入旋转、
partial 汇合和 SwiGLU 的完整操作，以及实际服务吞吐。NCU replay 曾改变候选
排序，只用于机制分析。完整 grid、arrival 可见性及 workspace 复用由实现和
`kernel/tests/test_trellis_linear.cu` 的回归测试维护。

## Limits and unknowns

结果限于上述硬件、形状和工具链。未锁频，SM 时钟遥测存在异常，不能声称等频。
微基准使用合成权重，插桩会影响调度；数值验证的结论限于已覆盖的输入。
剩余 K-loop 成本尚未归因到单一瓶颈，也未建立其他并发度的性能结论。

## Follow-ups

none
