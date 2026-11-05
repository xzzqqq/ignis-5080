# RTX 5080 architecture

Ignis-5080 serves Qwen3.8-27B on one RTX 5080 with 16 GB VRAM. It targets
low-concurrency personal chat and coding-agent use, with latency first and
decode approaching the hardware limit. It supports text, reasoning, tool
calls and image input. The server runs on Linux; editing and CPU checks run
on the Laptop through the [remote workflow](user/contributing/remote-dev.md).

## Execution

The Rust server renders and tokenizes input, then sends requests to a dedicated
model thread. `core` handles admission, scheduling and resource accounting;
`runtime` implements its `Compute` interface. A C ABI calls the C++/CUDA
implementation for each prefill chunk and decode round. Active sequences
and activations stay on the GPU. CUDA builds target `120a`.

The trellis body, DFlash2, GQA/GDN, sampling and Vision call project-owned
and vendored kernels. DFlash2 uses NVFP4 operators. Kernel sources and licenses
are recorded in [NOTICE](../kernel/NOTICE) and the
[vendor manifest](../kernel/vendor/manifest.json).

## State and resources

The baseline configures two decode lanes. The scheduler interleaves chunked
prefill and decode, handles request priority and cancellation, and reclaims
sequence resources. Each sequence holds KV pages and GDN/conv state.
Checkpoints and shared prefixes avoid repeated prefill; KV-RAM and KV-disk
store evicted state in host memory and on SSD.

Token embeddings are mapped from host memory. Vision weights are staged from
host memory over PCIe for each encode. These placements leave more VRAM for
active sequences. The load planner budgets device and host memory.

## Service

The server exposes Chat Completions and Responses, SSE and WebSocket streaming,
reasoning, tool calls, image input and tokenization. It provides request
deadlines, cancellation, health/readiness endpoints, API-key authentication,
structured logs and metrics. The UI includes Chat and Monitor. Idle unload
releases the model; the next request reloads it.

The supported model, hardware and removed upstream features are defined in
[ADR 0079](adr/0079-the-fork-serves-qwen38-27b-on-one-rtx-5080.md).

## Verification

Performance comparisons use the [coding-agent benchmark](user/contributing/coding-agent-benchmark.md)
with fixed fixtures, hardware settings, cache policy, response budget, lane
count, draft window and timing. Its 131,072-token capacity check and canary
run separately. Vision, tools and service lifecycle have additional checks.
BF16 references, eager execution and speculation-off runs provide numerical
controls. See [testing](agents/testing.md) for commands and failure handling.
