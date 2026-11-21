"""PDF to words, once.

Every later stage works from word boxes (text plus x0, y0, x1, y1 on a page),
so the PDF is read exactly once per filing and cached as gzipped JSON.
pdfplumber (MIT licence) does the extraction.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import pdfplumber

from .tables import Word

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"
WORD_DIR = ROOT / "data" / "words"


def extract_words(pdf_path: str) -> list:
    """Return one dict per page: {"w": width, "h": height, "words": [[x0, y0, x1, y1, text], ...]}."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False,
                                       use_text_flow=False)
            pages.append({
                "w": float(page.width), "h": float(page.height),
                "words": [[round(float(w["x0"]), 2), round(float(w["top"]), 2),
                           round(float(w["x1"]), 2), round(float(w["bottom"]), 2), w["text"]]
                          for w in words if w["text"].strip()],
            })
    return pages


def cache_path(doc_id: str) -> pathlib.Path:
    return WORD_DIR / f"{doc_id}.json.gz"


def _extract_to_cache(doc_id: str) -> tuple:
    out = cache_path(doc_id)
    if out.exists():
        return doc_id, -1
    pages = extract_words(str(PDF_DIR / f"{doc_id}.pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(pages, f)
    return doc_id, len(pages)


def extract_all(doc_ids: list, workers: int = 8) -> None:
    todo = [d for d in doc_ids if (PDF_DIR / f"{d}.pdf").exists() and not cache_path(d).exists()]
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for doc_id, n in ex.map(_extract_to_cache, todo):
            print(f"words  {doc_id:24s} {n} pages", flush=True)


def load_pages(doc_id: str) -> list:
    """Pages as lists of Word objects (page index 0 is page 1 of the PDF)."""
    with gzip.open(cache_path(doc_id), "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return [{"w": p["w"], "h": p["h"], "words": [Word(*w) for w in p["words"]]} for p in raw]
