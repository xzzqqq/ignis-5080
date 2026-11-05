---
title: "Reference"
description: "Configuration, profiles, models, metrics, routes, commands and terminology."
sidebar:
  order: 0
  label: "Overview"
---

`make docs-reference` generates these pages from the server's configuration
tables, catalog, metrics, OpenAPI document and CLI help. `cargo test` checks
that the committed pages match their sources. The glossary is maintained
by hand.

| Page | Contents | Source |
|---|---|---|
| [Configuration](configuration.md) | Flags, environment variables, defaults, validation and live updates | Configuration field table |
| [Profiles](profiles.md) | Hardware profile values | Built-in profiles |
| [Models](models.md) | Model files, pinned revisions, sizes and SHA-256 hashes | Built-in catalog |
| [Metrics](metrics.md) | Prometheus metrics, types and meanings | Metrics listener |
| [HTTP routes](http-routes.md) | API methods and paths | OpenAPI document |
| [Command line](cli.md) | Commands and subcommands | CLI help |
| [Glossary](../../../GLOSSARY.md) | Project terminology | Handwritten |

To inspect the running binary, use `ignis-server help --fields --format json`,
`GET /v1/config/fields`, or `GET /v1/openapi.json`. Browse the API at `/v1/docs/`.
