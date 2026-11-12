# 最小实施计划：benchmark 驱动的常驻热加载

状态：计划，尚未实现。审计基于 `c8137d1f` 的代码；后续提交截至
`75462699` 只改了设计文档。本文按最新讨论收敛实施范围。

## 目标与非目标

一个 agent 在独立 worktree 编辑，通过 `scripts/remote.py` 同步到一个可修改的
远端开发 checkout。一个前台驱动持有一个常驻引擎，反复评测当前 kernel。

只提供两级反馈：

1. 回放现有 coding-agent benchmark 中捕获的真实 Trellis 调用。
2. 从同一 benchmark 的原始输入重新执行完整模型前向。

服务验收、HTTP 性能基准、容量/canary 产品检查不属于这次功能范围。代码修改
所需的 workspace 测试、CUDA 编译和 GPU 正确性验证仍按仓库约定执行。
不实现多 agent 调度、后台投递服务、跨任务恢复、全量执行追踪或 KV/GDN 历史快照。
不新增编译缓存系统；复用现有候选构建器。

## 已有能力与 gap

| 部分 | 当前代码 | 最小处理 |
|---|---|---|
| 常驻模型 | Rust example 在 JSONL 循环内保留模型和 pool | 保留 |
| 计算替换 | 模型持有 linear/SwiGLU 函数表；ABI/工作区检查；图重捕获与失败回退 | 保留，不另写重载机制 |
| 候选构建 | 三个 CUDA 单元生成不可变 `.so`，记录源码/编译参数/库哈希 | 保留；补保存实际源码输入和热改范围校验 |
| 迭代驱动 | Python 接收预先给定的候选列表，测试后关闭引擎；GPU 锁只从队列继承 | 补前台交互循环、直接锁和进程清理 |
| benchmark 输入 | bench 通过 HTTP 提交三份完整 conversation/tools；lab 仅接收 token 数组 | 补离线输入准备和引擎回放适配 |
| 完整前向 | lab 默认生成 32 tokens，固定预算、不处理 EOS，不能等同现有 benchmark | 补 benchmark 的生成控制和明确的引擎计时口径 |
| 局部反馈 | `layer` 从手填 BF16 文件开始，历史状态为零 | 用真实 Trellis 调用捕获/回放替代主流程 |
| 图内采集 | 无；构图时的主机函数调用不代表 GPU 重放时的张量或调用次数 | 补受限的录制图和干净计时图切换 |
| 候选局部回放 | 独立算子基准静态链接；不会随另一个进程 reload 改变实现 | 新回放入口必须使用同一个模型的当前候选函数表 |
| 实验记录 | 有输入/库哈希、样本、基线对照 | 扩展真实样本来源、覆盖范围与候选源码；不用任务数据库 |

代码入口：`scripts/kernel_lab.py`、`scripts/build_trellis_candidate.py`、
`crates/core/examples/kernel_lab.rs`、`kernel/src/trellis_body.cu`、
`kernel/src/trellis_candidate.cu`。

## 实施顺序

### P1：固定 benchmark 到引擎的输入契约

复用 `crates/bench/fixtures/coding_agent/` 的 Astropy、Django、SymPy 三份请求和
`crates/bench/src/coding_agent.rs` 的请求参数。通过现有请求规整、
`ArtifactTemplateProvider`、模板和 tokenizer 准备输入，避免另写模板规则。
最小适配器可放在 server crate 的离线 example 中；不启动 HTTP 服务，也不让
core 反向依赖 server。只在确有需要时抽出小的共用准备函数。

产出一个供 lab 使用的输入文件：fixture 与模板/tokenizer 身份、精确 token IDs、
生成控制、有效引擎配置。保留 tools 和历史消息；不执行生成的工具调用。
按 C1 顺序运行三个请求，配置中的两个 decode lanes 不意味着同时发两个请求。

补齐 EOS/输出上限、reasoning 的停止重定向、DFlash2 和冷前缀规则，尽量调用
已有 core 逻辑。区分完整响应与显式截短的快速诊断，不能把固定 32-token 运行
悄悄称为原 benchmark。

完整前向只报告引擎内 prefill/decode、输出 token、停止原因、轮次和接受情况。
它不含服务调度与 HTTP 开销，不能用历史服务 TPS 作直接分母。所有速度比较在
同一引擎配置的 baseline/candidate 之间进行。

验收：三份 fixture 的准备结果与现有模板/分词路径逐 token 一致；生成控制与
计数可检查；baseline 可重复运行且每次序列状态重置。这是捕获和完整前向共享的
唯一输入准备路径。

### P2：接通单写者交互循环

扩展现有 Python driver，复用 `Session`、比较与记录函数：

- 启动时获取 host-wide GPU 锁、检查卡、固定 baseline 二进制/模型/配置，建立
  一个引擎进程。通过 `remote.py exec` 保持一条前台输入输出通道。
- 每次同步评测当前远端源码：确认改动在热改集合内，复制候选编译所需的小文件集，
  调用现有 builder，reload、检查、测量、恢复 baseline，然后返回结果。
- 一次只接收一个评测；收到反馈后再编辑/同步。没有队列、inbox、后台服务或
  自动重试。另一会话遇锁直接失败。
- EOF、停止或中断清理子进程；CUDA 故障结束会话。恢复仍使用原 baseline
  二进制，不能从当前候选源码隐式重建 baseline。

先用 P1 的完整前向证明这个闭环，再加入 P3 的快速局部回放。

验收：两次根据前一次反馈产生的修改，在同一个 engine PID 和模型分配上执行；
编译失败保持会话可用；越界修改要求重新 setup；退出释放 GPU；源码记录能还原
实际编译输入。故障恢复、图回退继续使用已有机制及回归测试。

### P3：从 benchmark 录制并回放真实 Trellis 调用

首批仅针对选定层的 MLP gate/up（SwiGLU）和 down（linear + residual）。它们
已有插件入口，且没有独立的 KV/GDN 历史依赖。需要其他 Trellis 投影时再扩展
筛选项，不增加通用子图切分器。

录制配置固定 fixture、层/投影和少量执行位置。只保留实际出现的形状、量化类型
和路径；设置样本/字节上限，报告缺失覆盖。每个样本包含：

- op、层/权重身份、形状、dtype、stride、epilogue、prefill/decode/verify 来源；
- 实际输入、ADD 操作的初始目标值、baseline 输出；
- fixture、baseline、模型、配置身份及执行位置；
- 可可靠得到的实际出现次数。不能把构图时主机调用次数当成重放次数。

权重继续引用同一 resident model，不复制模型文件。样本只要求在当前会话内
可回放；把少量张量和元数据保存到 `.scratch/` 供检查，不做跨版本回放缓存。

在 `trellis_body` 的既有分发边界设置实验专用采集钩子，不改变插件 ABI。
预分配有界设备捕获缓冲区。录制运行的 decode/verify 图包含必要的 D2D 复制，
获取真正执行时的输入/输出；在选定 round 返回后的同步边界读取样本。
可以用固定缓冲区覆盖各 round，并只保存选中的结果，不记录整条 tensor trace。

录制完成并释放序列后，解除采集并复用现有安全重捕获路径生成干净图；不在
活跃序列中调用要求空 pool 的热加载接口。Prefill 的非图路径同样记录实际执行
的数据。录制运行不产生性能结论，计时图不得保留采集复制。

新增局部回放入口，直接调用模型当前的 linear/SwiGLU 函数表。每次回放恢复
可变输入/目标缓冲和工作区状态；恢复、上传和下载在计时窗口之外。
局部计时图在当前候选激活后建立、替换候选前销毁，避免新增跨候选图缓存。
输出按既有数值门槛比较，保留原始时间样本、预热和 baseline 前后对照。

验收：baseline 回放与捕获输出一致；连续回放不受 ADD 累加污染；错误候选被
拦截；候选替换确实影响局部与完整前向两条路径；解除采集后结果不变且图内无
采集节点。采集、恢复和测量分别计时，不能宣称局部缓存状态等同完整模型。

### P4：收敛为一套评测入口并移除旧默认路径

setup 完成 P1 准备和一次 baseline 录制，比仅加载模型多了基线运行/采集。
分别记录这些一次性成本，不能沿用此前纯模型加载的耗时作为新 setup 总时间。
常规操作只保留：

- `evaluate replay`：构建当前候选，回放捕获集并比较；
- `evaluate forward`：先通过局部检查，再从原始输入完整运行；
- `stop`：释放引擎和 GPU。

这些是拟定操作，不是当前已经存在的命令。完整前向使用候选自身产生的激活、
状态和推测结果，不注入 baseline 的中间张量或强制它沿着 baseline token 轨迹。
模型、引擎、fixture、输入准备或配置身份变化后，重新准备参考与捕获集。

删除主流程中“必须先列出全部候选”的约束、手工拼 token/residual 的准备步骤和
零状态 `layer` 作为一级反馈的语义。新 replay 通过后，可删除 lab example 的旧
`layer` 命令及其专属分支，保留 core 中仍被测试/其他调用者使用的层 API。
原有限列表模式如需用于回归，只作为同一 evaluate 函数的薄循环，不维护第二套
加载、比较或清理逻辑。固定预算诊断不再冒充 benchmark 完整前向。

从主方案移除队列继承租约、逐候选完整 checkout、动态投递服务、跨任务共享与
自动恢复。现有 `scripts/experiments/` 保持独立，本次不批量删除它的代码。
文档中服务验收不再作为本功能的第三测量层级。

验收：在真实 5080 上展示 setup → 修改 A → replay/forward → 修改 B →
replay/forward → stop；整个正常循环只有一次模型加载。与既有构建/测量函数
保持相同 oracle、预热和样本规则；分别报告编译、重载、局部及完整前向成本。
实现阶段按仓库约定完成 CPU、CUDA 和所需 GPU 检查；不新增服务基准验收系统。

## 必须保留的约束

源码/库/输入身份、不可变 `.so`、GPU 独占、空 pool 替换、图与代码生命周期、
数值检查、基线对照和失败记录都保留。它们解决结果可信和进程正确性问题，与
是否支持并发无关。
