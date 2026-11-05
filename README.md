# Ignis-5080

An RTX 5080-specific fork of Ignis, focused on Qwen3.8-27B: a low-latency
local inference service for daily personal chat and coding agents.
The performance goal is decode approaching SOL (Speed of Light).
The fork starts from the unmodified upstream implementation.

The service runs on Linux with one RTX 5080 16 GB and uses CPU, host RAM,
PCIe and SSD for inference and state management. It supports text, reasoning,
tool calls and image input through Chat Completions and Responses, with
streaming, a chat UI, metrics and idle unload/reload.

Rust handles HTTP requests, scheduling and model execution. C++/CUDA implements
the GPU operators behind a C ABI. The built-in catalog supplies the 27B trellis
artifact; `rtx50-16gb` supplies the configuration defaults.

- [Architecture](docs/architecture.md)
- [Run and use the service](docs/user/README.md)
- [Laptop / remote Linux development](docs/user/contributing/remote-dev.md)
- [Coding-agent benchmark](docs/user/contributing/coding-agent-benchmark.md)
- [Testing](docs/agents/testing.md)
- [Terminology](GLOSSARY.md)
- [Scope decision](docs/adr/0079-the-fork-serves-qwen38-27b-on-one-rtx-5080.md)
