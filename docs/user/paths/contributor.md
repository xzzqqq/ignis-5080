---
title: "The contributor path"
description: "Work on ignis itself: the ground rules, building from source, the tests and the GPU discipline, make, the records a change leaves, releases."
sidebar:
  order: 4
---

For this RTX 5080 fork, read [remote development](../contributing/remote-dev.md)
first. It describes the Laptop/Linux split, the Python entry point and private
configuration. The upstream paths below remain references where applicable;
[`AGENTS.md`](../../../AGENTS.md) defines this fork's workflow.

For someone changing ignis — a fix, a feature, a kernel, a page of this
guide. The owner works with coding agents, and the rules they follow are the
rules here.

1. **[Contributing](../contributing/index.md)** — the ground rules and the
   repository's layout.
2. **[Building from source](../contributing/building.md)** — the kernel leaf,
   the Rust workspace, the CPU mock that needs no card.
3. **[Testing](../contributing/testing.md)** — the default suite, the GPU
   profile, the Playground's and the guide's checks.
4. **[Working with make](../contributing/make.md)** — the targets, the daemon
   mode agents use, the GPU guard.
5. **[How ignis serves a request](../concepts/how-ignis-serves.md)** and the
   [glossary](../reference/glossary.md) — the vocabulary the code and the
   issues use.
6. **[How work is tracked](../contributing/conventions.md)** — issues and their
   labels, specs, ADRs, contracts, findings, this guide, commits.
7. **[Cutting a release](../contributing/releases.md)** — the version, the
   changelog, the tag, and what the workflows publish.

:::caution
The card runs one GPU job at a time, across every worktree and agent session.
`make gpu-status` before the GPU profile, a bench run or a single `--ignored`
test: two runs on one card means one of them dies with no diagnostic.
:::

:::tip
A user-visible change is not done until this guide says what the code now
does, in the same commit. The pages under `reference/` are generated: change
the code and run `make docs-reference`.
:::
