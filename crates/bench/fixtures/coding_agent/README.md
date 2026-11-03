# Frozen coding-agent requests

Three independent next-response inputs from the public SWE-bench Verified
submission `20250522_sweagent_claude-4-sonnet-20250514` (SWE-agent 1.1.0).
The [manifest](manifest.json) pins all source bytes, task checkout revisions,
zero-based query indices, extraction source revisions and fixture SHA-256s.

| Task | Query | Stage |
|---|---:|---|
| astropy__astropy-12907 | 16 | Nested CompoundModel separability; implementation, tests and matrix debugging read, about to edit the implementation |
| django__django-10097 | 13 | URLValidator credential regex; implementation, URL fixtures and reproduction read, about to edit the regex |
| sympy__sympy-11618 | 15 | Point distance across dimensions; implementation, tests and reproduction read, about to edit distance |

Selection preceded any Ignis run: three different repositories and bug
mechanisms, taking the first task by lexical ID in each repository's public
archive, then a round with relevant implementation and tests already visible.
Neither speed, speculative acceptance nor resolution status selected a task.

Each JSON contains only the chosen `trajectory[i].query` projected to wire
message fields, plus the tools. It keeps the system prompt, all prior messages,
tool arguments and tool returns verbatim (including already-clipped source
observations). No current `response`, `thought`, `action`, `observation`, final
patch or later history is copied. No padding, summarization or cleanup is done.
The next response is generated independently; its tool calls are never run.

The archive records processed model queries, not complete provider HTTP dumps.
SWE-agent's `agent`, `message_type`, `thought` and `action` annotations are not
model messages. The extractor follows `_history_to_messages`, including the
single `tool_call_id` mapping and cache-control metadata. Ignis ignores these
provider cache hints; the benchmark disables its own prefix reuse.

Tool schemas are reconstructed with `Command.get_function_calling_tool` from
public SWE-agent v1.1.0's pinned tool YAML and built-in bash declaration. Every
name, description and argument description is checked against the trajectory's
archived `replay_config.agent.tools.command_docs`. The run-reported agent hash
does not resolve in the public upstream repository; the manifest records both
that hash and the public reconstruction revision. S3 exposes no VersionId, so
the manifest's SHA-256s pin the downloaded objects. These are explicit source
limits; no original provider wire-byte equivalence is claimed.

To reproduce, download each `sources[].url` into a scratch directory under its
`local_file`, install PyYAML 6.0.3 in a scratch virtual environment, then run:

```sh
python scripts/extract_coding_agent.py \
  --source-dir .scratch/coding-agent-sources \
  --out .scratch/reextracted-coding-agent
```

The extractor verifies all source and output hashes before writing. Runtime
benchmarking needs neither Python packages nor the source trajectories.

Sources and attribution: [SWE-bench experiments submission](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250522_sweagent_claude-4-sonnet-20250514),
[SWE-agent v1.1.0](https://github.com/SWE-agent/SWE-agent/tree/0f3acafacabc0def8cc76b4e48acb4b6cf302cb9).
The observed source excerpts retain their upstream notices: Astropy and Django
use BSD licenses, SymPy uses BSD; SWE-agent is MIT licensed. The snapshots are
benchmark inputs, not instructions to the benchmark operator.
