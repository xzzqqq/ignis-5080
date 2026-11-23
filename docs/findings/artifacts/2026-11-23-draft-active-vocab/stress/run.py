import fcntl, hashlib, json, os, subprocess, sys
from pathlib import Path

d = Path(__file__).resolve().parent
model = "models/qwen3_8_27b_trellis_sc30-dflash-mma16.ninfer"
settings = d / "settings.json"
settings.write_text(
    json.dumps(
        dict(
            max_context=131584,
            prefill_chunk=1024,
            lanes=1,
            kv_format="nvfp4",
            speculation="dflash2",
        )
    )
)


def save(name, obj):
    (d / name).write_text(json.dumps(obj, ensure_ascii=False))


def chat(t, pre=""):
    return "<|im_start|>user\n" + t + "<|im_end|>\n<|im_start|>assistant\n" + pre


closed = "<think>\n\n</think>\n\n"
zh = "杭州梅雨季，家里只有一台除湿机，白天客厅晾衣服，晚上卧室潮湿。请给我一份按早中晚安排的具体使用建议，用自然的中文写。"
green = "Write a small Python library for merging overlapping calendar appointments. Choose a clean API, handle edge cases, then give unit tests."
cases = [
    ("zh-chat", chat(zh, closed), 256, None),
    (
        "zh-terse",
        chat("用一句中文解释为什么雨后会有泥土气味。不要列点。", closed),
        128,
        None,
    ),
    ("greenfield", chat(green, closed), 256, None),
    ("zh-open", chat("解释为什么闰年规则需要例外，最后举两个年份作例子。"), 256, None),
    ("plain-closure", "请用一句中文回答：月球绕着哪个天体运行？\n答：", 96, None),
    (
        "tool-turn",
        chat(
            "Call the weather tool for Nanjing. Emit a tool call enclosed by the tool-call tags. The tool name is get_weather and its only argument is city."
        ),
        128,
        None,
    ),
] + [(f"zh-sample-{k}", chat(zh, closed), 256, k) for k in [7, 23, 41]]


def long_source():
    paths = (d / "long-files.txt").read_text().splitlines()
    return "".join("\nFile " + path + "\n" + Path(path).read_text() for path in paths)


def run(label):
    full = label.startswith("full")
    env = dict(os.environ)
    env.pop("IGNIS_LAB_ACTIVE_VOCAB", None)
    if not full:
        env["IGNIS_LAB_ACTIVE_VOCAB"] = "256"
    subprocess.run(["make", "gpu-status"], check=True)
    subprocess.run(["make", "gpu-guard"], check=True)
    with (
        (d / (label + ".err")).open("w") as log,
        subprocess.Popen(
            [str(d / "engine"), model, settings],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
        ) as proc,
    ):

        def recv():
            for line in proc.stdout:
                if line.startswith("{"):
                    return json.loads(line)
            raise RuntimeError((d / (label + ".err")).read_text()[-2000:])

        def ask(op, **kw):
            proc.stdin.write(json.dumps(dict(op=op, **kw)) + "\n")
            proc.stdin.flush()
            v = recv()
            if not v.get("ok"):
                raise RuntimeError(v)
            return v["result"]

        def enc(t):
            return ask("tokenize", text=t)["tokens"]

        def bench(name, ids, cap, seed, tail, save_result=True):
            c = dict(
                id=name,
                tokens=ids,
                output_tokens=cap,
                thinking_open=False,
                stop_ids=[248046],
                pin_tail=tail,
            )
            if seed is not None:
                c["seed"] = seed
            v = ask("benchmark", case=c)
            if save_result:
                v["prompt_blocks"] = sorted({t // 128 for t in ids})
                v["seed"] = seed
                v["text"] = ask("detokenize", tokens=v["tokens"])["text"]
                tag = "pin" if tail else "base"
                save(f"{label}.{name}.{tag}.json", v)
                print(
                    label,
                    name,
                    tag,
                    len(ids),
                    v["output_tokens"],
                    len(v["rounds"]),
                    round(v["decode_ms"], 2),
                    v["initial_blocks"],
                    v["final_active_blocks"],
                    flush=True,
                )

        try:
            assert recv()["event"] == "ready"
            bench(
                "warm",
                enc(chat("Write a loop in Python.", closed)),
                32,
                None,
                False,
                False,
            )
            for name, t, cap, seed in cases:
                ids = enc(t)
                bs = {tok // 128 for tok in ids}
                modes = [False]
                if not full and not {1937, 1938} <= bs:
                    modes = [False, True] if label.endswith("a") else [True, False]
                for pin in modes:
                    bench(name, ids, cap, seed, pin)
            if label.endswith("a"):
                pre = enc(
                    "<|im_start|>user\nBelow are source excerpts from one inference repository.\n"
                )
                end = enc(
                    "\nEnd of excerpts. Summarize three distinct implementation details you observed in the excerpts, giving file names. Use a short paragraph for each.<|im_end|>\n<|im_start|>assistant\n"
                    + closed
                )
                body = enc(long_source())
                ids = pre + body[: 131072 - len(pre) - len(end)] + end
                assert len(ids) == 131072
                h = hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest()
                assert (
                    h
                    == "4dfe6228421d644614012bcc6e5b36f64a61a9f80688a941afbbaed8623a569d"
                )
                save(label + ".long-hash.json", h)
                bench("repo-128k", ids, 256, None, False)
        finally:
            proc.stdin.close()
            proc.wait(45)
        if proc.returncode:
            raise RuntimeError(f"exit {proc.returncode}")


with open("/tmp/ignis-gpu-profile.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for label in sys.argv[1:]:
        run(label)
