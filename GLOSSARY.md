# Glossary

| Term | Meaning in this fork |
|---|---|
| SOL (Speed of Light) | The hardware limit for decode computation and data movement. This is the optimization target. |
| Artifact | The `.ninfer` weight container and its verified conversion sidecar. |
| Trellis | The compressed 27B weight representation used by the fixed 16 GB artifact. |
| Kernel leaf | The C++/CUDA implementation called through the step ABI. |
| Step ABI | The C interface for prefill chunks and decode rounds. Active sequence state and activations stay on the GPU. |
| Lane | A sequence decoded in a round. The baseline configures two decode lanes. |
| Prefill / decode | Processing prompt tokens / generating subsequent tokens. |
| GQA / GDN | The model's grouped-query attention / gated-delta recurrent blocks. |
| KV pool | Device pages holding attention keys and values. |
| KV-RAM / KV-disk | Host-memory and SSD storage for evicted or reusable sequence state. |
| Checkpoint / prefix | Saved sequence state / a shared prompt head used to avoid repeated prefill. |
| DFlash2 | The 27B speculative drafter. It uses NVFP4 operators; the main model verifies its token proposals. |
| Vision | The Qwen tower that encodes image input. |
| Profile | A set of configuration defaults. `rtx50-16gb` is the sole built-in profile. |
| Experiment (research scope) | A bounded optimization question with a baseline, metric and stopping rule. It may span several GPU jobs and resident sessions. |
| GPU job | One frozen `remote.py experiment` submission, with its own source snapshot, timeout, results and serial GPU lease. The CLI uses “experiment” for this execution unit. |
| Resident session | One supervised resident engine with fixed artifact and load settings. The planned four-command clients come and go without reloading it; restart creates a new session. |
| Evaluation | One synchronous comparison of a candidate within a resident session: build/check, reload, measure, restore the baseline and record feedback. It is not a queued task. |
| Replay corpus | Persisted benchmark call inputs, initial mutable buffers, reference outputs and provenance, reusable across sessions with matching identities and formats. |
| Candidate | An immutable Trellis library variant identified by its source and binary hashes. One resident session can test several candidates sequentially. |
| ADR | Architecture Decision Record, a general engineering practice. ADR 0079 defines this fork's scope. |
| Lain | An upstream name used in historical development material. |

See [architecture](docs/architecture.md) for execution and resource management,
the [service guide](docs/user/README.md) for usage, and the
[experiment lifecycle](docs/specs/continual-experiments/README.md#experiment-purpose-and-lifecycle)
for optimization work.
