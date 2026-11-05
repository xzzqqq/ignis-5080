# The fork serves Qwen3.8-27B on one RTX 5080

This fork targets low-latency personal chat and coding-agent use on one
RTX 5080 with 16 GB VRAM and SM120. It supports Qwen3.8-27B text, reasoning,
tool calls and Vision, and can use CPU, host RAM, PCIe and SSD. The performance
goal is decode approaching SOL.

Limit the built-in catalog and hardware profiles to this model and card.
Remove speech, diarization, image generation, Flash-Next serving,
Decide/SystemOne/Locate, public tunnels and Windows production tooling.
Use Linux for serving and the Laptop for editing and CPU checks, through
`scripts/remote.py`. Publish documentation as Markdown guides and generated
references. Remove the public guide website and upstream release publishing.

Remove each feature's routes, UI, configuration, tests, dependencies and guide.
Check callers before deleting shared code. Keep the kernels, reference tests,
memory tiers, scheduling, cancellation and observability used by the 27B service.
Keep benchmark fixtures and settings fixed during cleanup; measure numerical
and performance optimizations separately.

The owner authorized this scope on 2026-11-04. It supersedes the model and
platform support described in older ADRs, including 0032, 0043, 0066, 0068
and 0069. Their algorithmic rationale documents the upstream design.
Performance comparisons use the [coding-agent baseline](../user/contributing/coding-agent-benchmark.md).
