# AGENTS.md

## Project

Ignis-5080 is a local inference service for personal chat and coding agents,
focused on Qwen3.8-27B on one RTX 5080 16 GB.

Optimize for low concurrency and low latency, with decode approaching
SOL (Speed of Light). Support text, reasoning, tool calls and Vision.

## Environment and feedback

Edit and analyze locally; build CUDA and run GPU workloads on remote Linux.
Use `python3 scripts/remote.py` for remote operations. Before launching GPU
work, check `make gpu-status` through the remote entry point and coordinate
ownership of the single GPU. See [remote development](docs/user/contributing/remote-dev.md).

During optimization, use [Kernel Lab](docs/user/contributing/kernel-lab.md)
and operator replay for fast feedback, reusing the resident engine and
recorded inputs where supported. Explore multiple optimization points,
measuring both performance and numerical error. Keep candidates with
substantial speedups as alternatives while investigating numerical differences;
do not discard them solely for failing bitwise comparison. Use targeted
checks and forward measurements when they help diagnose issues or evaluate
combined changes. Run [full acceptance](docs/agents/testing.md) at the end of an optimization
round on the accumulated changes, not after every optimization point.

Operator boundaries and hot-reload coverage are tooling conveniences, not
optimization constraints. Fusion, splitting, cross-operator changes and
host-side changes are allowed within the task scope.

## Numerical correctness

Choose checks according to what the transformation changes:

- Changes to data movement, layout or scheduling that preserve arithmetic
  and rounding order should preserve output bits. Exact semantics such as
  indices and state updates must remain correct.
- For mathematically equivalent transformations, investigate numerical
  differences rather than rejecting the candidate. Bitwise agreement with
  the previous implementation is not required.
- Changes to precision, quantization or approximate algorithms require
  evaluating the additional numerical error and model-quality impact.

Investigate numerical differences and validate or repair candidates
autonomously. Consult the user only when adopting a candidate would
exceed agreed accuracy or quality limits, or when measurements establish a
quality tradeoff that requires a user decision. If no applicable criterion
exists, first gather evidence and propose one. A byte mismatch or an
unfinished error analysis is not itself a reason to stop and ask.
