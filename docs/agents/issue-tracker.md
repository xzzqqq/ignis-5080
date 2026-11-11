# Work records

The RTX 5080 fork takes work from the user's task. No GitHub issue, sub-issue,
label or issue-number suffix is required to plan, implement, validate or commit
it. This replaces the upstream issue workflow recorded in ADR 0070.

- Keep temporary plans, progress, experiment output and handoffs in `.scratch/`.
- Keep specifications that need to survive the task in `docs/specs/`.
- Put reusable conclusions in `docs/findings/` under its
  [authoring convention](findings.md), and architectural decisions in `docs/adr/`.
- Describe the delivered change and validation in the commit and final handoff.
  A pull request, when used, carries its own description; it needs no issue.

When a skill asks for a ticket, use the task brief or linked specification.
An instruction to publish to a tracker does not require creating an issue for
this fork. Keep the result locally unless the user requests publication.
Historical issue links remain useful source references; read them when needed,
but do not create, comment on or close issues as an implicit workflow step.

For commit and merge messages, follow [Coding standards](../../CODING_STANDARDS.md).
Use the configured Git remote and credentials when a push is authorized; the
upstream repository and account are not defaults for this fork.
