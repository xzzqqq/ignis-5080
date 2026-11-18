---
title: "推送前的凭据检查"
description: "安装和使用本地 Git pre-push 凭据检查。"
---

在负责提交和推送的本地安装 Gitleaks，再为当前仓库安装 hook：

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
