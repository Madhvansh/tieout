"""Fallback: run train_kaggle.py on this machine's GPU (RTX 3050 6 GB) instead of Kaggle.

Usage: .venv\\Scripts\\python.exe train_local.py [--data-dir data_real] [--system system_prompt.real.txt]
                                                [--out out_local] [--set BASE_MODEL="Qwen/Qwen3-0.6B" ...]
Download first (curl): https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/<file> into .local_stage/models/Qwen3-0.6B.
Uses finetune/.venv (torch + transformers + peft; the repo's main venv stays torch-free).
Stages the inputs exactly like push_kaggle.py and points train_kaggle.py at them through env vars.
Defaults: Qwen3-0.6B, EVAL_BATCH 8, GGUF_ROUTE f16 (no C++ toolchain here; run_local.py quantizes
to q4_K_M through `ollama create -q`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Python TLS on this box rejects huggingface.co (a self-signed root in the chain), while curl (Windows
# schannel) works, so the base model is fetched with curl into .local_stage/models and loaded offline.
LOCAL_MODEL = os.path.join(HERE, ".local_stage", "models", "Qwen3-0.6B")
LOCAL_DEFAULTS = {"BASE_MODEL": LOCAL_MODEL, "EVAL_BATCH": 8, "BATCH_SIZE": 8, "GRAD_ACCUM": 2, "GGUF_ROUTE": "f16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_real")
    ap.add_argument("--system", default="system_prompt.real.txt")
    ap.add_argument("--out", default="out_local")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    stage = os.path.join(HERE, ".local_stage", "input")
    shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    shutil.copy(os.path.join(HERE, a.data_dir, "train.jsonl"), stage)
    shutil.copy(os.path.join(HERE, a.data_dir, "val.jsonl"), stage)
    shutil.copy(os.path.join(HERE, a.system), os.path.join(stage, "system_prompt.txt"))
    ov = dict(LOCAL_DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = json.loads(v)
        except json.JSONDecodeError:
            ov[k] = v
    json.dump(ov, open(os.path.join(stage, "overrides.json"), "w"))
    env = dict(os.environ, FT_INPUT=stage, FT_WORK=os.path.join(HERE, a.out),
               FT_SCRATCH=os.path.join(HERE, ".local_stage", "scratch"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1")
    print("overrides", ov)
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "train_kaggle.py")], env=env))


if __name__ == "__main__":
    main()
