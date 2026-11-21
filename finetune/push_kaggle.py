"""Push data + train_kaggle.py to Kaggle (PRIVATE dataset + PRIVATE GPU kernel), wait, download outputs.

Usage: python push_kaggle.py [--data-dir data] [--system system_prompt.txt] [--out out]
         [--dataset-slug planner-ft-data] [--kernel-slug planner-lora-ft] [--set EPOCHS=4 ...]
         [--no-wait] [--download-only] [--timeout-min 120] [--machine NvidiaTeslaT4]
Needs the `kaggle` CLI on PATH and ~/.kaggle/kaggle.json. Never prints the API key.
Uploads <data-dir>/train.jsonl, <data-dir>/val.jsonl and the system prompt (as system_prompt.txt),
plus overrides.json from --set (hyperparameter overrides read by train_kaggle.py). Downloads into --out.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, ".kaggle_stage")
KAGGLE = shutil.which("kaggle") or "kaggle"


def run(args, check=True, quiet=False):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # Windows cp1252 console breaks `kernels output`
    p = subprocess.run([KAGGLE, "-W"] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = (p.stdout + p.stderr).strip()
    if not quiet and out:
        print(out)
    if check and p.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed ({p.returncode})")
    return p.returncode, out


def username():
    _, out = run(["config", "view"], quiet=True)
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        sys.exit("could not read the Kaggle username from `kaggle config view`")
    return m.group(1)


def push_dataset(user, a):
    d = os.path.join(STAGE, a.dataset_slug)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for src, dst in ((os.path.join(a.data_dir, "train.jsonl"), "train.jsonl"),
                     (os.path.join(a.data_dir, "val.jsonl"), "val.jsonl"), (a.system, "system_prompt.txt")):
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            sys.exit(f"missing {p} (run make_dummy.py or drop in the real file)")
        shutil.copy(p, os.path.join(d, dst))
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = json.loads(v)
        except json.JSONDecodeError:
            ov[k] = v
    with open(os.path.join(d, "overrides.json"), "w") as fh:
        json.dump(ov, fh)
    print("overrides", ov)
    ref = f"{user}/{a.dataset_slug}"
    with open(os.path.join(d, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": a.dataset_slug, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, fh)
    rc, out = run(["datasets", "status", ref], check=False, quiet=True)
    if rc == 0 and "ready" in out.lower():
        print(f"dataset {ref} exists -> new version")
        run(["datasets", "version", "-p", d, "-m", f"update {time.strftime('%Y-%m-%d %H:%M')}"])
    else:
        print(f"creating PRIVATE dataset {ref}")
        run(["datasets", "create", "-p", d])            # private is the CLI default (no -u)
    for _ in range(60):                                  # wait until processed
        time.sleep(10)
        _, out = run(["datasets", "status", ref], check=False, quiet=True)
        if "ready" in out.lower():
            print("dataset ready"); return ref
    sys.exit("dataset never became ready")


def push_kernel(user, dataset_ref, machine, slug):
    d = os.path.join(STAGE, "kernel-" + slug)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    shutil.copy(os.path.join(HERE, "train_kaggle.py"), d)
    ref = f"{user}/{slug}"
    meta = {"id": ref, "title": slug, "code_file": "train_kaggle.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [dataset_ref], "competition_sources": [],
            "kernel_sources": [], "model_sources": [], "machine_shape": machine}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"pushing PRIVATE kernel {ref} on {machine}")
    run(["kernels", "push", "-p", d])
    return ref


def wait(ref, timeout_min):
    t0 = time.time(); last = None
    while time.time() - t0 < timeout_min * 60:
        _, out = run(["kernels", "status", ref], check=False, quiet=True)
        m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out)
        st = m.group(1).upper() if m else out[-120:]
        if st != last:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {st}", flush=True); last = st
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            return st, (time.time() - t0) / 60
        time.sleep(45)
    return "TIMEOUT", timeout_min


def download(ref, out):
    os.makedirs(out, exist_ok=True)
    run(["kernels", "output", ref, "-p", out, "-o"])
    mp = os.path.join(out, "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        ev = {k: v for k, v in m.get("eval_hf", {}).items() if k != "sample_failures"}
        print(json.dumps({"train": m.get("train"), "eval_hf": ev, "gguf": m.get("gguf_file"),
                          "gguf_mb": m.get("gguf_mb"), "total_min": m.get("elapsed_min_total")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--timeout-min", type=float, default=120)
    ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--system", default="system_prompt.txt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dataset-slug", default="planner-ft-data")
    ap.add_argument("--kernel-slug", default="planner-lora-ft")
    ap.add_argument("--set", action="append", default=[], help="hyperparameter override KEY=VALUE (JSON value)")
    a = ap.parse_args()
    out = os.path.join(HERE, a.out)
    user = username()
    print(f"Kaggle user: {user}")
    ref = f"{user}/{a.kernel_slug}"
    if not a.download_only:
        ref = push_kernel(user, push_dataset(user, a), a.machine, a.kernel_slug)
        if a.no_wait:
            return
        st, mins = wait(ref, a.timeout_min)
        print(f"kernel finished: {st} after {mins:.1f} min wall (queue + run)")
    download(ref, out)


if __name__ == "__main__":
    main()
