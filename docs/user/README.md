# Run Ignis-5080

This fork serves Qwen3.8-27B on Linux with one RTX 5080 16 GB. It supports
text, reasoning, tool calls and image input. The Chat UI, Monitor, settings,
workspace files and image markup are available at `/ui/` when built in.

## Build and start

Configure the [remote development entry point](contributing/remote-dev.md)
and put `qwen3_8_27b_trellis_sc30-v2.ninfer` with its verified conversion
sidecar in `models/`. The pinned catalog also supports `ignis-server model
download qwen3.8-27b-16g` on the Linux host. Download configuration and
integrity checks are listed in [the reference](reference/cli.md).

From the Laptop:

```sh
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- make build CUDA=1 UI=1
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- make start CUDA=1 UI=1
python3 scripts/remote.py exec -- make status
# After use:
python3 scripts/remote.py exec -- make stop
```

`make start` uses the last build and refuses to start if its sources have changed. It writes logs
under `.scratch/serve/`. Set `API_KEY=auto` to require a generated key; keep
`BIND=127.0.0.1:8000` for a local listener. When the server runs remotely,
use SSH port forwarding to reach it from the Laptop.

## Clients and capabilities

Use base URL `http://127.0.0.1:8000/v1` and model `qwen3.8-27b-16g`.
The family alias `qwen3.8-27b` resolves to the same artifact. Chat Completions
and Responses support streaming, reasoning and tool calls. Send images as
chat image parts. The Vision tower stages weights from host memory; the
profile allows up to 4,096 image tokens per request.

Use `/health` and `/ready` to check service health and readiness. The server
provides request timeouts, cancellation, idle unload/reload, prompt reuse,
KV-RAM, KV-disk, API keys, structured logs and Prometheus metrics.
Model-management routes handle downloads, file removal and loading.
The loader rejects artifacts from other model families.

## Configuration

The built-in `rtx50-16gb` profile configures two decode lanes, DFlash2 with
seven draft tokens and the full draft head, 1,200 MiB VRAM headroom, and
Vision weights in host memory. The load planner adapts the context to
available VRAM.

- [All configuration fields](reference/configuration.md)
- [Profile values](reference/profiles.md)
- [Pinned model files](reference/models.md)
- [HTTP routes](reference/http-routes.md), also `/v1/docs/` on the server
- [Metrics](reference/metrics.md)
- [Command line](reference/cli.md)
- [Performance and capacity protocol](contributing/coding-agent-benchmark.md)

## Removed upstream features

Speech, transcription, dictation, diarization, image generation/editing,
Flash-Next serving, Decide/SystemOne/Locate and public tunnels are removed.
Their former HTTP endpoints return 404. Existing conversation images can
still be viewed. Windows production builds, the public guide website and
upstream release publishing are also removed; see [ADR 0079](../adr/0079-the-fork-serves-qwen38-27b-on-one-rtx-5080.md).
