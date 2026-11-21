"""Bring a Colab training run's outputs to the laptop through the browser.

A 2.5 GB file in one browser download tends to break, so the GGUF is cut into 500 MB parts with
a SHA-256 of the whole file; the laptop joins the parts and checks the hash. Run inside the
notebook kernel (files.download needs it):  %run /content/tieout/finetune/colab_fetch.py
"""
import glob
import hashlib
import os
import subprocess

from google.colab import files  # noqa: available on Colab only

OUT = "/content/out"
gguf = sorted(glob.glob(f"{OUT}/*.gguf"))[0]
stem = os.path.basename(gguf).rsplit(".", 1)[0]
if not glob.glob(f"{OUT}/{stem}.part*"):
    subprocess.run(["split", "-b", "500M", "-d", gguf, f"{OUT}/{stem}.part"], check=True)
    h = hashlib.sha256()
    with open(gguf, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    with open(f"{OUT}/{stem}.sha256", "w") as fh:
        fh.write(f"{h.hexdigest()}  {os.path.basename(gguf)}\n")
small = [f"{OUT}/{n}" for n in ("Modelfile", "val_predictions.jsonl", "train_log.txt", f"{stem}.sha256")]
small += [p for p in ["/content/plans4b.json"] if os.path.exists(p)]
for path in small + sorted(glob.glob(f"{OUT}/{stem}.part*")):
    print("downloading", path, os.path.getsize(path))
    files.download(path)
