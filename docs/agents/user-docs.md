# User documentation

Update `docs/user/` in the same commit as any change to routes, configuration,
the model catalog, telemetry, UI, builds or startup behavior. Begin with the
[service guide](../user/README.md); remote development and the coding-agent
benchmark have separate pages.

Generate `docs/user/reference/` from `crates/server/src/user_docs.rs`. Edit the
source, then run `make docs-reference` on Linux or its Cargo command on the
Laptop. Run the reference test to check that the committed pages match the
source. Use relative Markdown links, placeholder hosts and credentials, and
commands for the current Linux/Laptop workflow.
