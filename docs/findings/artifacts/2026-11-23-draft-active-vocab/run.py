import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path(__file__).resolve().parent
engine = Path(os.environ.get("IGNIS_VOCAB_LAB_ENGINE", "target/x86_64-unknown-linux-gnu/release/examples/kernel_lab")).resolve()
artifact = Path("models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer")
cases = json.loads((out / "inputs.json").read_text())["cases"]
rewrite = "Copy the following Python definitions in order, changing only the local variable payload to segment. Preserve every function name. Return only the code.\n\n" + "\n\n".join(
    f"def _{name}(payload):\n    return payload.get('{name}', 0)"
    for name in ["barycentric_jacobian", "conjugacy_orbit", "ephemeris_epoch", "idempotent_retraction",
                 "schreier_transversal", "noncentral_stabilizer", "wigner_seitz_cell", "lindblad_superoperator",
                 "eigenspace_multiplicity", "contravariant_tensor", "schur_complement", "symplectic_involution"])
msg = "<|im_start|>user\n" + rewrite + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
(out / "settings.json").write_text(json.dumps(dict(max_context=40960, prefill_chunk=1024, lanes=2, kv_format="nvfp4", speculation="dflash2")))

def arm(label):
    mode = label.split("-")[0]
    env = dict(os.environ)
    env.pop("IGNIS_LAB_ACTIVE_VOCAB", None)
    if mode != "full":
        env["IGNIS_LAB_ACTIVE_VOCAB"] = mode
    subprocess.run(["make", "gpu-status"], check=True)
    subprocess.run(["make", "gpu-guard"], check=True)
    with (out / (label + ".stderr")).open("w") as log, subprocess.Popen(
            [engine, artifact, out / "settings.json"], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True) as proc:
        def receive():
            while line := proc.stdout.readline():
                if line.startswith("{"):
                    return json.loads(line)
            raise RuntimeError((out / (label + ".stderr")).read_text()[-2400:])
        def ask(**command):
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()
            reply = receive()
            if not reply.get("ok"):
                raise RuntimeError(str(reply))
            return reply["result"]
        try:
            assert receive()["event"] == "ready"
            tokens = ask(op="tokenize", text=msg)["tokens"]
            edit = dict(cases[0], id="unseen-identifier-rewrite", tokens=tokens, thinking_open=False,
                        output_tokens=512)
            warm = dict(cases[0], id="warmup", tokens=cases[0]["tokens"][:512], output_tokens=40)
            ask(op="benchmark",case=warm)
            inputs = [dict(cases[0],output_tokens=128)] if mode=="1940" else [*cases,edit]
            if selected := os.environ.get("IGNIS_VOCAB_CASE"):
                inputs = [c for c in inputs if c["id"] == selected]
                assert len(inputs) == 1
            for case in inputs:
                res = ask(op="benchmark", case=case)
                (out / f"{label}.{case['id']}.json").write_text(json.dumps(res))
                count, rounds = res["output_tokens"], len(res["rounds"])
                print(label, case["id"], count, rounds, round(res["decode_ms"]/rounds,3),
                      round(count*1000/res["decode_ms"],2), res["final_active_blocks"],flush=True)
        finally:
            proc.stdin.close()
            proc.wait(30)
        if proc.returncode:
            raise RuntimeError(f"lab exit {proc.returncode}")

with open("/tmp/ignis-gpu-profile.lock","a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    for label in sys.argv[1:]:
        arm(label)
