# Reproduce the C1 active draft vocabulary experiment

The patch is the Kernel Lab prototype measured at revision `2e7e73baa7ce`. It uses a device bitmap for a single active request and exposes the opt-in in the benchmark executable. The full arm uses the original draft head. The real prompts are prepared from the repository's coding-agent fixtures. The driver defines the synthetic rewrite prompt.

Apply the patch in a local experiment checkout and sync it to its configured remote checkout. Copy the driver into the remote scratch directory so each arm writes its own receipt there. Run from the local checkout:

```sh
a=docs/findings/artifacts/2026-11-23-draft-active-vocab
d=.scratch/active-vocab-20261123
git apply "$a/prototype.patch"
python3 scripts/remote.py sync
python3 scripts/remote.py exec -- mkdir -p "$d"
python3 scripts/remote.py exec -- cp "$a/run.py" "$d/run.py"
python3 scripts/remote.py exec -- cargo build --locked --release -p ignis-core --features cuda --example kernel_lab
python3 scripts/remote.py exec -- cargo build --locked --release -p ignis-server --example lab_prepare
artifact=models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer
python3 scripts/remote.py exec -- target/x86_64-unknown-linux-gnu/release/examples/lab_prepare "$artifact" "$d/inputs.json"
python3 scripts/remote.py exec -- make gpu-status
python3 scripts/remote.py exec -- python3 "$d/run.py" full-a 256-a 256-b full-b
python3 scripts/remote.py exec -- python3 "$d/run.py" 192-pair-a 256-pair-a 256-pair-b 192-pair-b
```

The driver acquires the shared profiling lock, checks GPU ownership before each engine launch and warms up each resident engine. A row in [forward.csv](forward.csv) represents one request on one launch. The 1940-block diagnostic used a 128-token output cap. Full-output and stop comparisons in that row use the uncapped baseline; the reported matching prefix covers all 128 generated tokens. The zero-prefix ablation and its Django repeat ran separately.

For the projection microbenchmark, the head starts at byte 10556772352 in this specific artifact (4194304-byte payload base plus object offset 10552578048). The probe uses its actual K4 codes and scales with deterministic synthetic BF16 activations. Run from the same checkout after building the patched library:

```sh
python3 scripts/remote.py exec -- .toolchain/cuda/bin/nvcc -O2 -std=c++17 -Ikernel/include "$a/head.cpp" kernel/build/libignis_kernel.a -o "$d/head"
python3 scripts/remote.py exec -- flock -n /tmp/ignis-gpu-profile.lock bash -c 'set -e; make gpu-status; make gpu-guard; .scratch/active-vocab-20261123/head models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer 10556772352 < docs/findings/artifacts/2026-11-23-draft-active-vocab/masks.txt'
```

Each line in [head-results.txt](head-results.txt) gives the enabled block count, full-head ms, masked-head ms and number of incorrect BF16 words. Each timing is the median of seven measurements of twenty calls. The probe alternates execution order and checks every retained logit against full and every excluded logit against negative infinity.
