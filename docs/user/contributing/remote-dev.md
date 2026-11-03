---
title: "5080 fork：远端开发"
description: "Laptop 编辑与 CPU 检查，远端 Linux 构建 CUDA 和运行 RTX 5080。"
sidebar:
  order: 7
---

## 分工与配置

此 fork 从 Ignis 起步，目标是 RTX 5080 上的 Qwen3.8-27B 个人聊天与 coding agent
服务。Laptop 负责编辑、CPU 检查、提交和分析；CUDA 构建、GPU 执行及依赖 Linux
的检查在远端进行。上游 5090 的运行结果不代表此 fork 已完成 5080 验证。

Laptop 和远端需要 Python 3.10+、Git；Laptop 还需 OpenSSH 客户端。
`scripts/remote.py` 只使用标准库，远端命令通过 Bash 执行。
在仓库根目录初始化私有配置：

```sh
cp .remote.example.json .remote.json
```

`.remote.json` 已被 Git 忽略，只支持两个字段：

```json
{
  "host": "gpu-dev",
  "repo": "/path/to/ignis-5080"
}
```

`host` 是 `~/.ssh/config` 中的 alias，实际地址、用户、端口、认证方式由 SSH
配置管理；`repo` 是远端现有 checkout 的绝对路径。每个本地 checkout 使用自己根目录
的 `.remote.json`，从其他当前目录调用脚本也不会改用别处的配置。私有配置和凭据不入库。
SSH 使用非交互认证，需事先配置好认证及主机信任；连接失败会直接返回错误。

## 固定机器环境

已验证的工具链放在远端稳定目录，用远端仓库根目录的 `.remote.env` 固定选择。
每次 `exec` 和 `sync` 都在执行命令前加载它，加载失败则不执行命令。这个文件是
Bash 脚本，需保持安静，不向标准输出写内容，例如：

```sh
export RUSTUP_TOOLCHAIN=<installed-version>
export CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu
export PATH="$PWD/.toolchain/bin:$HOME/.cargo/bin:$PATH"
export CUDA_HOME=/path/to/validated/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IGNIS_KERNEL_BUILD_JOBS=12
export CARGO_BUILD_JOBS=12
```

按本机已安装版本填写，不直接执行含占位符的示例。`.remote.env`、`.toolchain/`、
`.remote.cargo.toml` 和已有的 `local.mk` 都不提交，也不随源码同步。工具链、镜像
配置放这些私有位置；模型路径和 Make 参数放 `local.mk`。环境文件选择已安装工具，
不代替安装包或容器镜像；更换机器时仍需准备对应版本。日常不重复做整套环境预检。

## 同步源码

```sh
python3 scripts/remote.py sync --dry-run
python3 scripts/remote.py sync
```

`sync` 将 Laptop 当前工作区同步到远端：包含已提交源码、未提交修改、未被 Git
忽略的新文件，以及源码删除。不要求先提交，也不改变 Laptop 的分支或暂存区。
初次同步要求远端 HEAD 是本地历史中的祖先；此命令不负责创建 checkout。

传输使用压缩的增量 Git bundle。输出中的 `bundle bytes` 是源码包大小，不含 SSH
协议和接收脚本的控制开销；`--dry-run` 计算这个大小但不应用，内容不变时为 0。
它保留远端的模型、`target/`、`kernel/build/`、工具链、私有配置和 `.scratch/`，
不会来回传输构建产物。新增大文件仍可能产生大包，传输前可先看 `--dry-run`。
已同步且内容未变的未提交文件也不重复传输。源码同步不启动服务或触发构建，可以
多次编辑、同步后再构建；通常按一次需要远端验证的改动同步一次。

远端 HEAD 跟随 Laptop HEAD；未提交的工作区快照由内部 Git ref 记录，并出现在
远端暂存区。远端是构建副本，源码继续在 Laptop 编辑：若远端出现独立修改、
非忽略的新文件或不同的 HEAD，下一次同步会拒绝，先保存并处理这些修改再重试。
不要在同步过程中编辑远端源码；完成同步后再构建，避免编译读到新旧文件的混合版本。

## 唯一远端入口

从 Laptop 的仓库根目录运行：

```sh
python3 scripts/remote.py exec -- git rev-parse HEAD
python3 scripts/remote.py exec -- git status --short
python3 scripts/remote.py exec -- cargo test --workspace --target x86_64-unknown-linux-gnu
python3 scripts/remote.py exec -- make build CUDA=1
python3 scripts/remote.py exec -- make check CUDA=1
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- make gpu-guard
```

命令在配置中的远端仓库目录运行，保留标准输入、输出、错误及退出码。参数逐项传递；
需要管道或重定向时，显式调用 `bash -c '…'`，其中的 shell 内容由调用者控制。
环境由 `.remote.env` 加载，避免登录 shell 重新改写 PATH。Laptop 不需要 make 或 CUDA。

`exec` 不同步文件、不 checkout、不提交、不 push、不管理实验队列，也不自动重试。
执行前先 `sync`，再核对 commit 和未提交改动；本地编辑不会自动出现在远端。
退出码 255 通常表示 SSH 连接问题，命令可能已经开始执行，重跑前先
核对远端状态。长时间服务需使用远端持久会话或 daemon；此入口不保证断线后任务继续。

## Linux 服务与开发循环

配置好远端 `local.mk` 中的模型路径、`UI` 和其他运行参数后：

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- make build CUDA=1
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- make start
python3 scripts/remote.py exec -- make status
python3 scripts/remote.py exec -- make stop
```

Linux 复用现有 Make 目标图和参数：`start` 只运行最新构建，源码比构建新时拒绝启动，
CUDA 启动仍经过 GPU guard。后台进程脱离 SSH 会话；启动等待公开的 `/ready` 返回
200，失败或超过 `READY_TIMEOUT` 就停止本次进程并保留日志。`stop` 默认只停止
本 checkout 记录的进程，记录包含 PID、启动时间和系统启动标识，避免误杀复用的 PID。
`FORCE=1` 才额外停止同用户的其他 `ignis-server`。

PID 和日志在 `.scratch/serve/`；再次启动会归档旧日志，`make logs` 跟踪当前日志。
普通修改的循环是 `sync` 后 `exec -- make redeploy`：构建成功才停止旧服务并启动新
服务，编译失败保留旧服务。`restart` 只重启，不构建。

`dev` 是构建后前台运行；代理管理长期服务用 `start/stop`。已有 `make watch`
需要 `cargo-watch`，检测远端文件变更并重新编译、重启进程；它不负责 Laptop 同步。
CUDA 修改重启后会重新加载模型，不是在保留显存状态下热替换代码；CPU mock 的
`make watch CUDA=0` 更适合快速检查服务逻辑。Linux 的 `dev-ui/run-ui` 联动尚未实现，
前端可单独运行已有的 `make web-dev` 使用 Vite 热更新。

## 检查与当前限制

本地 Python 入口测试需要 Gitleaks 8.30.1+，运行时不需要网络或 GPU；push 测试
只向临时目录里的本地 bare 仓库推送：

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

Rust CPU 测试优先在 Laptop 执行。仓库的 Cargo 配置固定 Windows target，其他平台
必须覆盖它；先通过 `rustc -vV` 读取 `host`，再用其值运行：

```sh
cargo test --workspace --target <host>
```

本地缺工具链、依赖或平台能力的检查交给远端 Linux。未执行的检查应列出原因，不能
算通过；macOS Makefile 适配不属于此入口的前提。

GPU 工作前先查看 `make gpu-status` 并通过 `make gpu-guard`，协调全主机独占，
包括其他项目和 worktree。状态检查不是锁；上游 guard 的显存阈值也不证明 5080
有足够余量。不能因检查通过就并行启动第二项 GPU 工作。

Linux 已支持 CUDA build/check、GPU status/guard 和后台服务控制；`make gpu-profile`
仍未实现（见 `mk/os/linux.mk`）。不能把 Windows profile 的 skip 或未运行写成
Linux/5080 验收通过。构建目录长期保留供增量构建使用，不在同步或测试后自动清理。

## 推送前的凭据检查

在负责提交和推送的 Laptop 安装 Gitleaks，再为当前仓库安装 hook：

```sh
brew install gitleaks
python3 scripts/pre_push.py install
```

安装器只安装 `pre-push`，保留已有 `core.hooksPath` 和其他 hook；已有其他来源的
`pre-push` 时拒绝覆盖。Python 和扫描器的本机路径保存在本地 Git 配置中，不写进
提交。更新或搬迁工具后重新执行安装命令即可。已有 `make setup` 配置的 checkout
也使用同一份检查逻辑。

正常执行 `git push` 就会检查，不需要另外运行扫描命令。范围来自 Git 传入的
实际推送分支、标签和远端旧提交；新分支会查询**目标远端**实时公布的 refs，仅排除
其中本地已知的提交历史。空远端或没有可用基线时扫描完整可达历史；未知的旧提交、
远端查询失败或浅克隆会阻止推送，需要先取回完整的目标历史。

检查包含每个新增提交的补丁（含合并提交）、commit message 和 annotated tag
message。因此某个提交写入凭据、下个提交删掉，仍会被拦截。`.remote.json`、
`.remote.env`、`.remote.cargo.toml`、`local.mk`、`.env` 和 `.toolchain/`、
`.scratch/`、`.ssh/` 是私有路径，出现在新增历史中就拒绝，不依赖内容识别。
仅允许推送普通分支和指向 commit 的标签；内部 refs 和 `--mirror` 会被拒绝。

内容检查使用 Gitleaks 默认规则，并启用两层解码和压缩包检查；仓库里的忽略文件、
`gitleaks:allow` 注释及环境变量不能关闭这个 hook 的规则。发现可疑内容时只输出
规则、文件、行号和提交位置，不输出凭据值；扫描器缺失、报错或超时也拒绝推送。
误报需核实后调整检查代码中的精确规则，不通过排除整个目录或跳过 hook 放行。

这是本机防护，不能识别任意格式的自定义密码，也可以被主动禁用或 `--no-verify`
绕过；GitHub 的 push protection 是额外的服务端防线。CI 运行这些临时仓库回归
测试，但 CI 在上传之后执行，不代替本机拦截。安装过程不会推送代码，也不会修改
GitHub 设置。
