# C1 active-vocabulary stress screen

Use the experiment checkout described in the [original reproduction instructions](../README.md). Apply the original prototype.patch, then the incremental benchmark.patch in this directory. The additional harness applies the same temperature, seed, top-k and top-p at prefill and verify, records bitmap setup and update time, and supports the 131,072-token input plus output allowance. The long case requires one configured lane. Both arms use the same engine and NVFP4 KV.

From the local checkout, after applying the original patch:

```sh
a=docs/findings/artifacts/2026-11-23-draft-active-vocab/stress
d=.scratch/vocab-stress
git apply "$a/benchmark.patch"
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- cargo build --locked --release --features cuda -p ignis-core --example kernel_lab
python3 scripts/remote.py exec -- mkdir -p "$d"
python3 scripts/remote.py exec -- cp target/x86_64-unknown-linux-gnu/release/examples/kernel_lab "$d/engine"
python3 scripts/remote.py exec -- cp "$a/run.py" "$a/long-files.txt" "$d"
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- python3 "$d/run.py" full-a active-a active-b full-b
```

The driver locks the GPU, checks ownership before each launch and warms up each engine. Short requests run in all four launches. The 128 Ki-token prompt uses ordered source excerpts from revision 2e7e73ba, truncates only the source body, and checks the prepared token hash. It runs once in each arm. Runs with pinning enable blocks 1937 and 1938 in addition to the fixed 256-block prefix and the request's context blocks. Where the prompt already includes both, the masks are identical and the duplicate execution is omitted. The pin comparison reverses the order on the second active launch.

The generated token sequence excludes the terminating stop token. Decode throughput uses those output tokens and includes round and bitmap-update time. Initialization of the bitmap is part of prefill. For each short comparison pool the two matched runs per arm. With temperature 1, top-k 20 and top-p 0.95, the three seeds use the same Chinese prompt and a fixed 256-token cap. Its timing compares equal token budgets; individual sampled paths can differ. The tool-format prompt requests textual tags without supplying a service tool schema. Executable SHA-256 for the reported run was `22a750cec6dfab469e25d235859c8237d7bad3af08b84d94fff6f8105daeaa02`.
