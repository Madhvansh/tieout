"""Publish the fine-tuned planner so the system runs with no API key.

Hugging Face (when a write token is in .env as HF_TOKEN): uploads the GGUF, the Modelfile and the
model card to <user>/tieout-planner. Otherwise a GitHub release asset on the project repository
(files up to 2 GB), uploaded with the `gh` CLI.

Usage: python finetune/publish.py finetune/out/v2_q17 [--name tieout-planner] [--tag planner-v2] [--github]
Prints the download location and the two commands a reader runs to use it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "Madhvansh/tieout"


def token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--name", default="tieout-planner")
    ap.add_argument("--tag", default="planner-v2")
    ap.add_argument("--github", action="store_true", help="publish as a GitHub release asset even if HF_TOKEN is set")
    a = ap.parse_args()
    folder = pathlib.Path(a.folder)
    gguf = next(folder.glob("*.gguf"))
    card, modelfile = ROOT / "finetune" / "MODEL_CARD.md", folder / "Modelfile"
    tok = token()
    if tok.startswith("hf_") and not a.github:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        user = api.whoami()["name"]
        repo_id = f"{user}/{a.name}"
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        for path, dest in ((gguf, gguf.name), (modelfile, "Modelfile"), (card, "README.md")):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=repo_id)
        print(f"published: https://huggingface.co/{repo_id}")
        print(f"use: ollama pull hf.co/{repo_id}  (or download {gguf.name} and run: ollama create {a.name} -f Modelfile)")
        return
    if gguf.stat().st_size >= 2 * 1024 ** 3:
        sys.exit(f"{gguf.name} is over GitHub's 2 GB asset limit; it needs a Hugging Face token")
    gh = shutil.which("gh") or "gh"
    notes = f"Fine-tuned planner {gguf.name} for TieOut, with its Modelfile. See finetune/MODEL_CARD.md."
    subprocess.run([gh, "release", "create", a.tag, "--repo", REPO, "--title", f"Local planner ({a.tag})",
                    "--notes", notes, str(gguf), str(modelfile)], check=True)
    print(f"published: https://github.com/{REPO}/releases/tag/{a.tag}")
    print(f"use: download {gguf.name} and Modelfile into one folder, then: ollama create {a.name} -f Modelfile")


if __name__ == "__main__":
    main()
