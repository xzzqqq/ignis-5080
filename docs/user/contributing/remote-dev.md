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

Laptop 需要 Python 3.10+ 和 OpenSSH 客户端；`scripts/remote.py` 只使用标准库。
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
需要管道、重定向或环境初始化时，显式调用 `bash -lc '…'`，其中的 shell 内容由调用者
控制。CUDA、Rust 等工具需已在远端命令环境中可用。Laptop 不需要 make 或 CUDA。

`exec` 不同步文件、不 checkout、不提交、不 push、不管理实验队列，也不自动重试。
执行前需将待测版本准备到远端 checkout，再核对 commit 和未提交改动；本地编辑不会
自动出现在远端。退出码 255 通常表示 SSH 连接问题，命令可能已经开始执行，重跑前先
核对远端状态。长时间服务需使用远端持久会话或 daemon；此入口不保证断线后任务继续。

## 检查与当前限制

本地 Python 入口测试不需要网络或 GPU：

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

Linux 已支持 CUDA build/check 和 GPU status/guard，但 `make gpu-profile` 及
`make start/stop/status` 仍未实现（见 `mk/os/linux.mk`）。需要这些能力时另行实现并
验证；不能把 Windows profile 的 skip 或未运行写成 Linux/5080 验收通过。
