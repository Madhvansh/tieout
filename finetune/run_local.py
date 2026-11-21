"""Register the Kaggle-built GGUF with Ollama and score it on the val set on this machine.

Usage: python run_local.py [--model planner-ft] [--out out] [--data data/val.jsonl] [--system system_prompt.txt]
         [--schema plan_schema.json] [--format schema|json|none] [--cpu] [--skip-create] [--limit N]
Reads <out>/Modelfile + <out>/*.gguf, the val file, the system prompt and the JSON schema.
Reports JSON validity, exact match (parsed-JSON equality) and seconds per question (warm model).
Stdlib only (urllib); needs `ollama` on PATH and the server on localhost:11434.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def post(path, body, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def create(model, OUT):
    if not os.path.exists(os.path.join(OUT, "Modelfile")):
        sys.exit("out/Modelfile missing: run push_kaggle.py first (or --download-only)")
    gg = glob.glob(os.path.join(OUT, "*.gguf"))
    if not gg:
        sys.exit("no .gguf in out/")
    cmd = ["ollama", "create", model, "-f", "Modelfile"]
    if "q8_0" in os.path.basename(gg[0]).lower():      # fallback export: quantize here instead
        cmd += ["-q", "q4_K_M"]
    print("$", " ".join(cmd), f"(gguf {os.path.getsize(gg[0]) / 1e6:.0f} MB)")
    t = time.time()
    subprocess.run(cmd, cwd=OUT, check=True)
    print(f"ollama create took {time.time() - t:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="planner-ft")
    ap.add_argument("--format", choices=["schema", "json", "none"], default="schema")
    ap.add_argument("--cpu", action="store_true", help="force CPU (num_gpu 0)")
    ap.add_argument("--skip-create", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--system", default="system_prompt.txt")
    ap.add_argument("--schema", default="plan_schema.json")
    a = ap.parse_args()
    OUT = os.path.join(HERE, a.out)
    if not a.skip_create:
        create(a.model, OUT)
    system = open(os.path.join(HERE, a.system), encoding="utf-8").read().strip()
    schema = json.load(open(os.path.join(HERE, a.schema)))
    rows = [json.loads(l) for l in open(os.path.join(HERE, a.data), encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    opts = {"temperature": 0, "num_predict": 384}
    if a.cpu:
        opts["num_gpu"] = 0
    fmt = {"schema": schema, "json": "json", "none": None}[a.format]

    def ask(q):
        body = {"model": a.model, "prompt": q, "system": system, "stream": False, "options": opts, "keep_alive": "15m"}
        if fmt is not None:
            body["format"] = fmt
        return post("/api/generate", body)

    t = time.time(); ask(rows[0]["question"]); load_s = time.time() - t           # warm-up (model load)
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip()
    print(ps)
    valid = exact = 0; times = []; toks = []; fails = []
    for r in rows:
        t = time.time(); res = ask(r["question"]); dt = time.time() - t
        times.append(dt)
        if res.get("eval_duration"):
            toks.append(res["eval_count"] / (res["eval_duration"] / 1e9))
        txt = res.get("response", "").strip()
        try:
            obj = json.loads(txt); ok = True
        except Exception:
            obj, ok = None, False
        hit = ok and obj == r["plan"]
        valid += ok; exact += hit
        if not hit and len(fails) < 8:
            fails.append({"question": r["question"], "gold": r["plan"], "pred": txt})
    n = len(rows)
    times_sorted = sorted(times)
    m = {"model": a.model, "format": a.format, "device": "cpu" if a.cpu else "auto(gpu if fits)", "n": n,
         "json_valid": round(valid / n, 4), "exact_match": round(exact / n, 4),
         "sec_per_q_mean": round(sum(times) / n, 3), "sec_per_q_p50": round(times_sorted[n // 2], 3),
         "sec_per_q_max": round(times_sorted[-1], 3), "gen_tok_per_s": round(sum(toks) / max(1, len(toks)), 1),
         "first_call_incl_load_s": round(load_s, 2), "ollama_ps": ps, "sample_failures": fails}
    tag = f"{a.format}_{'cpu' if a.cpu else 'gpu'}"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"local_metrics_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps({k: v for k, v in m.items() if k not in ("sample_failures", "ollama_ps")}, indent=2))
    for f in fails[:3]:
        print("FAIL", json.dumps(f)[:400])


if __name__ == "__main__":
    main()
