"""Download the filings listed in data/manifest.csv into data/pdf/.

PDFs are not committed to the repo. This script fetches each one from the
URL in the manifest (the company's own investor-relations host), checks that
it is a PDF, and records its size and sha256 back into the manifest so a
reader can confirm they parsed the same bytes.

Usage: python scripts/fetch_corpus.py [--only TICKER ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
OUT = ROOT / "data" / "pdf"
# several investor-relations hosts refuse the default python client string
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
MAX_BYTES = 90 * 1024 * 1024


def fetch(url: str, dest: pathlib.Path) -> tuple:
    h = hashlib.sha256()
    n = 0
    with requests.get(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"},
                      stream=True, timeout=90, allow_redirects=True) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            first = True
            for chunk in r.iter_content(1 << 16):
                if first:
                    if not chunk.lstrip()[:5].startswith(b"%PDF"):
                        raise ValueError("not a PDF (first bytes %r)" % chunk[:12])
                    first = False
                n += len(chunk)
                if n > MAX_BYTES:
                    raise ValueError("larger than size cap")
                h.update(chunk)
                f.write(chunk)
        tmp.replace(dest)
    return n, h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    for extra in ("bytes", "sha256", "status"):
        if extra not in fields:
            fields.append(extra)
    failed = 0
    for row in rows:
        doc = row["doc_id"]
        if args.only and row["ticker"] not in args.only and doc not in args.only:
            continue
        dest = OUT / f"{doc}.pdf"
        if dest.exists() and row.get("sha256"):
            continue
        try:
            n, digest = fetch(row["pdf_url"], dest)
            row["bytes"], row["sha256"], row["status"] = str(n), digest, "ok"
            print(f"ok   {doc:22s} {n / 1e6:6.2f} MB", flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue with the rest
            row["status"] = f"FAILED: {type(e).__name__}: {e}"[:160]
            failed += 1
            print(f"FAIL {doc:22s} {e}", flush=True)
        time.sleep(0.6)
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
