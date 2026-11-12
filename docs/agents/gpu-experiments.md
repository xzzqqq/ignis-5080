# GPU experiments for delegated work

The main agent supplies the question, constraints and expected deliverable.
The subagent owns its worktree, request and analysis loop. Use `python3 scripts/remote.py experiment`; do not
sync a candidate over the shared remote checkout or ask the main agent to
run its GPU commands.

Before submitting, state the research question, baseline/source identity,
allowed change scope, workload/metric, numerical gate, budget and stopping
rule in `.scratch/`. One question can require several queue jobs. The
[resident experiment contract](../specs/continual-experiments/README.md#experiment-purpose-and-lifecycle)
defines experiment, job, session and candidate lifetimes.

1. Check `python3 scripts/remote.py experiment status`. Install the worker
   once if absent; upgrades require an empty queue.
2. Write `.scratch/request.json` with a unique id, question, timeout, argv
   steps and required output files. Pin remote inputs with SHA-256 when
   results depend on their contents. Use candidate-relative build paths,
   `{output}` for results and explicit service/model configuration.
3. Run `python3 scripts/remote.py experiment submit .scratch/request.json`
   from the subagent worktree. It freezes working changes without committing.
4. Use `result`, `wait --timeout 60`, `fetch` and `cancel` with the returned id.
   After a lost submission reply use `retry <id>`, which reuses the saved
   submission. A changed submission payload/source/configuration gets a new
   job id. One frozen payload may include several candidate libraries, each
   identified separately by its source and binary hashes.
5. Inspect artifacts and logs, revise and repeat. Command success alone is
   not a correctness or performance verdict. Record skipped/failed checks.
6. Return the conclusion, observations, evidence, known limits, job ids,
   source HEAD/snapshot/tree and local/remote result paths. Identify which
   working changes should be integrated. The main agent decides integration.

For Trellis-only searches, the current hot-loading workflow is a finite batch:
prepare candidate sources, submit a frozen job, build the libraries, then call
`scripts/kernel_lab.py` once with multiple `--candidate` paths. Model setup is
shared within that session. The driver still compiles no code itself, checks
and warms each candidate separately, and closes the engine after the list.
After inspecting feedback, a newly edited worktree needs a new submission;
there is no live delivery into a running job or session shared across agents.
The runner rejects modifications to its frozen tracked source. See the
[lab guide](../user/contributing/kernel-lab.md) for executable commands.

The revised [single-writer design](../specs/continual-experiments/README.md)
removes the proposed shared session and candidate-delivery service. It calls
for a foreground interactive driver using `remote.py exec`, one mutable
checkout and one engine, independently of this queue. That driver extension
is not implemented yet. This page remains the procedure for delegated/frozen
queue jobs: retain their source isolation, timeouts and descendant cleanup.
Do not run the future mutable workflow inside such a job.

End an experiment with its supported/rejected/inconclusive conclusion and
record required checks that failed or were not run. Final service validation
runs after the lab releases the card.

For comparisons using rebuilt binaries, use separate Cargo target and CMake build
directories. Verify the linked native CUDA library or kernel symbols for each
arm; different server hashes alone do not establish different kernel code.

For a service experiment use `make build`, `make start` redirected to a log,
measure, then `make stop` inside the job. The unit cleans up descendants even
on failure or cancellation. Never spawn another unit/container/SSH process
to escape it. Keep shared models and tools read-only. Use `prune <id>` after
collecting a terminal result to remove only its candidate build/cache.

Minimal request (`.scratch/request.json`):

```json
{
  "version": 1,
  "id": "candidate-001",
  "task": "candidate-check",
  "question": "Does this candidate pass its check?",
  "timeout_s": 600,
  "steps": [
    {"name": "check", "argv": ["bash", "scripts/my-check.sh", "{output}"],
     "artifacts": ["report.json"]}
  ]
}
```

Replace the example check with the candidate's actual script. Optional `env`
sets command environment; `inputs` contains absolute remote `path` and `sha256`
pairs checked before/after execution. `config` is descriptive metadata; argv/env
control execution. Artifacts are relative files under `{output}`.

Configure `.remote.json` in this worktree or its main worktree first. The remote
needs Linux, systemd user services, cgroup v2 and the validated `.remote.env`.
`experiment install` enables the worker and user lingering for SSH disconnects.
Each request snapshots source and private remote config into an isolated build;
`local.mk`, existing builds and models are not copied. Supply model paths and
service parameters explicitly. `--commit SHA` selects committed source only.

`stop` cancels the active job and retains queued jobs; `start` resumes consumption.
`status` shows blocked reasons and worker health. `result ID` includes progress;
`fetch ID` downloads terminal results to `.scratch/experiment-results/ID`, verifies
hashes and refuses overwrites. Receipts live in `.scratch/experiment-submissions/ID`.
Never remove a receipt before resolving an uncertain submission. Failed,
cancelled, timed-out or interrupted attempts are not automatically replayed.

Exit codes: 0 success, 1 unsuccessful terminal experiment, 2 control/request
error, 3 still pending/wait timeout, 4 unavailable or blocked worker, 255 usually
SSH failure. Cleanup may extend beyond the requested runtime timeout. The queue
is for trusted collaborators; unmanaged GPU commands must still obey host-wide
exclusive use. Validation: `python3 scripts/validate_experiments.py --out
.scratch/queue-validation-new` (requires an idle queue and uses the real GPU).
