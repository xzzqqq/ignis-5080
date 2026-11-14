# 实施计划：四命令驱动的常驻热加载

状态：P1–P4 已实现，验证记录随实现保存。代码底座为 `c8137d1f`。
本文保留初版实施顺序；快速反馈修订见 [设计](README.md) 和
[用户指南](../../user/contributing/kernel-lab.md)。实现使用每会话一个
临时 systemd user unit 管理 owner/引擎子进程清理，不启用自动重启或实验队列。
图捕获使用有界缓冲区，预算不足明确拒绝；自动分组录制未实现，需显式缩小捕获
范围或调整预算，不能把不完整数据集当作成功结果。

## 1. 最终入口与范围

所有命令从 Laptop 的当前 worktree 调用，四个独立进程，无需保留 SSH 会话、
手动 sync 或发送 JSONL：

```sh
python3 scripts/lab.py setup
python3 scripts/lab.py replay
python3 scripts/lab.py forward
python3 scripts/lab.py stop
```

`scripts/lab.py` 在内部调用现有 `scripts/remote.py sync/exec`，复用
连接配置、源码同步及错误处理；不另写 SSH/传输实现。一次性模型路径、引擎配置、
数据目录和捕获范围由 setup 参数/本地配置提供，随后冻结在会话中。
`.remote.json` 继续只负责现有 host/repo 配置，不混入测量参数。

前提是一个 agent、一份可修改的远端开发 checkout、一个活跃会话、一次一个
评测。并发命令返回 busy，不排队。服务验收、HTTP 性能基准、容量/canary 产品
检查不属于本功能；实现需要的 CPU/CUDA/GPU 正确性检查仍按仓库约定执行。
不实现多 agent 调度、任务数据库、自动重试/续跑、通用历史状态快照、通用子图
编译器或新的编译缓存系统。

## 2. 命令契约

| 命令 | 内部步骤 | 完成条件 |
|---|---|---|
| setup | 确认没有冲突会话 → sync → 构建或复用固定 baseline/runtime → 启动后台 owner 与引擎 → 准备 benchmark 输入 → 加载兼容回放集，缺失时生成 | 模型、干净执行图、参考和完整回放集均 ready 后返回；不能仅凭进程启动就报成功 |
| replay | 校验会话归属/配置和热改范围 → sync → 验证同步结果 → 保存候选输入并构建/复用库 → 热加载 → 局部回放与基线对照 → 恢复 baseline | 结果持久化、baseline 恢复成功后返回 |
| forward | 与 replay 共用准备与局部 gate → 从原始 benchmark 输入执行完整前向 → 基线对照 → 恢复 baseline | 局部 gate 通过才进入完整前向；输出候选自身的生成行为与引擎时间 |
| stop | 根据已记录的会话目标核验进程身份 → 停止 owner/引擎及本次构建子进程 → 确认 GPU 资源释放 | 无需 sync、源码可编译或控制进程能正常响应；重复 stop 安全，保留结果和回放集 |

同步/构建失败禁止继续使用旧候选进行本次测量。无源码变化时依靠现有增量同步
和对象/候选缓存；同一会话中，forward 可复用同一候选已通过的 replay。
日常 replay 和 forward 固定使用一条请求，移除前后完整 baseline 回放；
`forward --full` 保留全部请求、配置的预热/采样及前后 baseline 对照。

同一 owner 的 setup 若发现已有匹配会话，只报告现有 ready/busy 和最近结果，
不隐式同步重建或改变基线。不同 owner/config 的活跃会话明确拒绝。发生 CUDA
故障后，默认 setup 复用保留的 baseline 二进制；`setup --new-baseline`
只在旧会话已停止后，从当前源码显式建立新基线。活跃会话下拒绝此选项，不隐式
重启。不能把当前 kernel 候选静默变成恢复后的基线。

## 3. 最小内部结构

- **本地入口 `scripts/lab.py`**：四命令分发、命令锁、调用 remote.py、把反馈和
  结果索引保存到本 worktree 的 `.scratch/lab/`。
- **远端 owner**：扩展/复用 `scripts/kernel_lab.py` 的 Session、比较与记录逻辑。
  它后台存活并直接监督一个 Rust 引擎，通过本机 Unix socket 接受同步操作。
  socket 位于短路径的用户私有运行目录；不开放 TCP/HTTP。
- **已有 Rust/CUDA 引擎**：模型状态、插件构建后的 reload，以及新增的 benchmark
  回放与真实调用捕获。内部继续使用 JSONL；这不是用户需要操作的界面。

只增加薄的本地控制通道。没有 pending job 列表、投递服务、轮询调度器或多引擎池。
会话身份包括 owner worktree、session id、远端目标、PID/启动身份、socket 和
固定配置。复用已有 Linux 进程身份核验方法，不只依赖 PID，也不全局 kill 进程。

### 同步与代码版本

本地非阻塞命令锁覆盖 replay/forward 的检查、sync 和评测。远端也检查 owner、
session id 和 busy 状态；忙/归属不符时在 sync 前拒绝。仅支持该 worktree 的
单写者，不支持第二台客户端复用身份或评测中绕过入口手工同步。

每次 sync 后验证实际远端源码身份与本次提交的源一致。构建从保存的小文件集
读取，避免后续编辑污染候选。编译器、flags、源码和库哈希均落盘。

初始热改集合为 `kernel/src/` 下的 `trellis_linear.cu`、
`trellis_linear_decode.cu`、`trellis_linear_prefill.cu`、
`trellis_linear_internal.cuh`、`trellis_decode.cuh`。host/runtime、公共 ABI、
其他算子或配置变化需要停止后重新 setup；host 源码变化使用显式新基线选项。
文档/笔记变化不触发模型重建。

后台控制代码、构建 helper 和 baseline 二进制从固定版本的运行目录执行，不能
因 sync 覆盖开发目录而混用新旧实现。stop 通过会话记录中的固定清理入口和远端
目标执行，不依赖刚修改的 CUDA 源码，也不把已修改的连接配置误当作会话所在地。
底层仍走 remote.py；必要时只补向它传递已记录目标的小接口。

### 进程与故障

owner 持有 host-wide GPU 独占锁，监督引擎和构建子进程。退出/崩溃必须处理其
子进程，GPU 使用者未退出前不得宣告释放。setup 的后台启动要重定向日志与
标准输入输出，避免继承 SSH 管道导致命令不返回。

stop 是控制操作，允许终止正在运行的评测；不能被评测的本地锁或阻塞 RPC
永久挡住。正常退出超时后，只能升级终止核验过身份的本会话进程组。

四命令模式下，普通命令返回或 SSH 断开不应自动关闭空闲引擎。评测请求一旦
接受，即完成或到达其超时，并写入结果；丢失回复不自动重放。新的 setup 可
查询同一会话的 active/最近结果，下一次评测在 busy 时拒绝。只记录当前操作和
结果文件，不建设可续跑任务系统。失败或未完成文件永远不能冒充成功结果。

编译、checksum、ABI 或工作区拒绝保持 baseline 可用；数值失败恢复 baseline。
CUDA 故障、执行超时或恢复失败终止会话，保留日志并要求 setup。数据目录和
回放集保留；不自动重建、重新排队或迁移回放集。

## 4. 两级测量的数据契约

### benchmark 输入与完整前向

使用 `crates/bench/fixtures/coding_agent/` 中 Astropy、Django、SymPy 三份请求，
复用 `crates/bench/src/coding_agent.rs` 的参数与现有请求规整、
`ArtifactTemplateProvider`、模板和 tokenizer。离线准备器可放在 server crate
的 example 中；不启动 HTTP，也不让 core 反向依赖 server 或另写模板规则。

准备文件保存精确 token IDs、fixture/模板/tokenizer 身份、生成控制和有效配置。
保留 tools 与历史消息，不执行生成的工具调用。三个请求按 C1 顺序运行；配置
中的两个 decode lanes 不意味着同时发两条请求。

补齐 EOS、输出上限、reasoning 停止重定向、DFlash2 与冷前缀规则。完整前向
每次从原始输入与新序列状态开始，不注入保存的中间张量，也不强制沿基线 token
轨迹执行。记录输出 token、停止原因、轮次、接受情况和引擎 prefill/decode 时间。
它不包含服务调度和 HTTP，不能直接用历史服务 TPS 作分母。显式截短的诊断须
标注，固定 32-token/ignore-EOS 运行不能冒充完整 benchmark 回放。

### 捕获与持久化局部回放

覆盖当前可热加载的全部 Trellis linear/SwiGLU：MLP、GQA 投影、GDN 输入/输出
投影和输出头。按两个接口统一实现，用层号、张量角色和权重身份区分调用。
这些算子不直接依赖 KV/GDN 历史；不扩大到 attention、卷积和递归更新本身。

回放集保存所选范围在真实 benchmark 中执行的实际输入、ADD 初始目标、参考
输出，以及 op/层/路径、shape、dtype、stride、权重视图、请求/执行位置与次数。
不能把主机构图调用次数当作 GPU 重放次数。默认不进行隐式采样；统计实际体积，
只有明确选择缩小范围/采样时才改变覆盖。

保存范围与每轮测量范围分开：可以保存完整回放集，setup 固定本次要评测的
op/层/形状或记录范围，默认覆盖其指定的捕获范围。若回放全部数据的 I/O 或计算
不适合快速反馈，显式缩小测量范围并记录覆盖；所有候选使用相同范围，不能根据
某候选的结果挑选有利样本。先实测回放成本，不预设它必然比完整前向快。


使用 manifest 和二进制张量文件保存到配置的数据目录（默认 `.scratch/`）。
磁盘空间/字节预算不足时明确失败；缺失、损坏、未完成的回放集拒绝复用。只有
完整写入并校验后才发布完成标记。模型、baseline、fixture、输入准备、配置、
GPU/工具链、影响 kernel 路径的环境设置、捕获范围和格式身份均匹配时，后续
setup 直接加载；只记录相关环境白名单，不保存整个进程环境。不兼容则生成新的
回放集。复用张量与正确性参考，不复用旧性能数字；当前基线计时仍按相同规则重测。
权重按稳定张量名/视图重新绑定，不复制模型权重，不保存 GPU 地址，不做自动
格式迁移或后台缓存管理。

在 `trellis_body` 的分发边界设置实验专用钩子，保持候选 ABI 不变。使用有界
捕获缓冲区并流式落盘，保存所有范围内调用后才能覆盖缓冲区。若一个图容纳不下
目标数据，可按固定调用分组重跑同一 baseline 收集全部范围，并验证轨迹一致；
不把整轮多次调用压成最后一次，也不为容纳捕获而改动被测引擎配置。

decode/verify 录制图必须在实际执行时复制输入/输出；在安全同步边界读取。
Prefill 非图路径也保存实际调用的数据。录制完成并释放序列后，移除捕获节点，
复用已有安全重捕获路径建立干净图。不在活跃序列中调用要求空 pool 的 reload。
录制运行的耗时不作为性能结果。

局部回放调用同一个模型当前的候选函数表。恢复可变输入/目标和工作区，恢复、
传输、文件 I/O 在计时之外；计时图在候选激活后创建、替换前销毁，不增加跨
候选图缓存。对照保存的参考输出，保留既有严格数值门槛、预热、样本和基线对照。
真实输入不意味着局部缓存状态等同整模型；局部时间不能简单相加为完整收益。

## 5. 实施顺序与验收

### P1：benchmark 输入适配与完整前向

补离线准备器和 lab 的生成控制；复用模板/分词/core 路径。验收三份 fixture
逐 token 一致，EOS/reasoning/预算/DFlash2 行为可检查，baseline 可重复运行、
每次状态重置。此阶段只需内部入口，不新增服务 API。

### P2：四命令与常驻 owner

新增 lab.py 包装 sync/exec；把现有 Python Session 和评测函数用于 Unix socket
控制的后台 owner。完成身份/锁、固定运行代码与 baseline、源码边界、小输入
存档、结果文件和清理。先用 P1 的完整前向打通两次修改仍是同一 engine PID 的
闭环。复用现有 builder 与 CUDA 重载；不另写一套比较或传输代码。

验收 sync 失败不评测旧代码、源码不变缓存命中、编译失败后 baseline 可用、
busy/foreign owner 在 sync 前拒绝、setup 不隐式重设基线、stop 不依赖可编译
源码且可打断评测、掉线不重放、owner 异常后无遗留 GPU 子进程。

### P3：统一的真实调用捕获/回放

补全两种 Trellis 调用的采集与回放，不按 MLP/GQA/GDN 各建系统。验收包括
SwiGLU、普通 linear、ADD、带 stride/视图调用、输出头及实际出现的执行路径。
证明图内数据来自真实执行、调用没有被覆盖漏记、回放输出匹配参考、重复 ADD
不累积、当前候选确实影响 replay 和 forward 两条路径。

录制开/关的生成轨迹应一致，干净图无采集节点。停止后新 setup 复用磁盘回放集，
重新解析权重视图；错误身份、损坏和不完整数据被拒绝。实测并分别记录数据字节量、
首次录制成本、再次加载成本、重载、恢复与测量成本。

### P4：统一入口、清理与端到端验证

接入 setup 的回放集准备、replay 的局部 gate 和 forward 的完整前向。
删除主流程对预声明候选列表、手工 token/residual 和零历史整层 probe 的依赖。
真实 replay 可用后删除 lab example 的旧 `layer` 专属命令/分支；core 层 API
的其他调用者不动。有限列表回归如果保留，只作为同一评测函数的薄循环。

移除前台 SSH/stdin 句柄作为用户入口的要求、队列继承租约和逐候选完整 checkout；
旧 `scripts/experiments/` 独立保留。修订用户/agent 文档，不再描述未实现的交互
入口，不把服务验收放回第三测量层级。

在真实 5080 展示：setup → 修改 A → replay → forward → 修改 B → replay/
forward → stop → setup 复用回放集。正常循环只加载一次模型，完整前向使用候选
自身状态。独立记录自动 sync/控制开销，不能用旧的纯 reload 时间代替整个命令。
相同输入、oracle、预热、样本和基线对照下，与原有测量逻辑比较反馈一致性。
实现完成前运行 workspace CPU、Python、CUDA all-targets 和所需 GPU 回归；
记录失败/跳过。服务基准验收系统不在实施范围。

## 6. 组件取舍

保留现有插件 ABI、工作区检查、空 pool 替换、图/代码生命周期、数值检查与失败
回退。新增代码集中在四命令薄封装、后台进程/本地 socket、benchmark 适配和
真实调用捕获/回放。它们各自有明确职责；不加队列、任务持久化、通用自动恢复、
编译调度或数据分析平台。
