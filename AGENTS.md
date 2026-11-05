# AGENTS.md

This file is for agents working **on** ignis: its code, its guide, its issues.
An agent here for a user of ignis (to explain it, install it, or connect a
client to it) wants [the agent path](docs/user/paths/agent.md) instead.

## Project

This is an RTX 5080-specific fork of Ignis, focused on Qwen3.8-27B: a
low-latency local inference service for daily personal chat and coding agents.
The performance goal is decode approaching SOL (Speed of Light). The fork
starts from the unmodified upstream implementation.

Optimize for low concurrency and latency. Support text, reasoning, tool calls
and Vision on one RTX 5080 16 GB, using CPU, host RAM, PCIe and SSD as needed.
See [architecture](docs/architecture.md) and ADR 0079 for component responsibilities
and model/platform support.

## Development mode

The Laptop handles editing, CPU checks, commits and result analysis. CUDA
builds and GPU execution run on remote Linux. Run Rust CPU tests locally
where possible; tests that lack the required local platform or dependencies
run on remote Linux.

The single remote entry point from the Laptop is `python3 scripts/remote.py`:
`sync` transfers source, then `exec -- <command>` runs in the remote checkout.
Remote `.remote.env` selects the validated toolchain automatically. The private
connection config is the repository-root `.remote.json`, which is not committed; the tracked,
credential-free template is [.remote.example.json](.remote.example.json).
SSH connection details are managed in `~/.ssh/config`.

Use the Python entry point for remote work and Cargo for local CPU checks.
Cargo defaults to the host target. Commands and Linux target availability are
listed in [Remote development](docs/user/contributing/remote-dev.md).

## Agent skills

### Domain docs

Single-context: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Repo conventions

### Scratch

`.scratch/` is the untracked scratchpad, local to each clone, for temporary
artifacts, experiments, raw logs and workflow output. When something there
must outlive the task, it moves into the repo: Artifact boundaries in
`docs/agents/findings.md` names the usual homes.

### Coding standards

Writing, reviewing or merging a change, or running a skill that writes files:
`CODING_STANDARDS.md`.

### Findings

When recording or consulting durable findings from discovery, research, or
experiments, start from `docs/findings/README.md`.

### ADRs

When recording a decision in `docs/adr/`, write it in the `/domain-modeling`
skill's short format, with the bare title as its heading: the release notes
take the number from the file name. An ADR takes the next free number when it
merges into `main`; if `main` took that number first, the ADR is renumbered
with every line that cites it (b61a224f). When a skill's convention and the
repo's disagree: ADR 0070 lists where the repo differs.

### Testing

Every code change ships with a test, and the task is done only when
`cargo test` passes workspace-wide. Before writing a test or reporting a code
change done: `docs/agents/testing.md`.

Before launching **anything** on the GPU (the profile, a bench run, a gate
leg, one `--ignored` test): check with `make gpu-status` that nothing else
holds the card, from any worktree or agent session. The 5080 fits one run at
a time, and the loser dies with no diagnostic.

### User documentation

Every user-visible change ships with its `docs/user/` update in the same
commit. Before reporting a change done: `docs/agents/user-docs.md` says
whether it is user-visible, where its guide update goes and how it is written.

### Make

An agent runs the server as a daemon, `make start` then `make stop`: a
foreground target holds the shell until Ctrl+C. `start` and `run` launch only
the last build and refuse a stale one, naming the changed files; `make build`
is the fix. Before changing the Makefile: ADR 0027.

Redirect `make start` output to a file when scripting: a server can inherit
an open pipe. Before restarting after an incident, copy the logs out of
`.scratch/serve/`, since a new start rewrites them.
