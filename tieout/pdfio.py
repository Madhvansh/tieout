"""PDF to words, once.

Every later stage works from word boxes (text plus x0, y0, x1, y1 on a page),
so the PDF is read exactly once per filing and cached as gzipped JSON.
pdfplumber (MIT licence) does the extraction.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import pdfplumber

from . import glyphs
from .calc import coverage, discover
from .tables import Word, build_table, group_lines

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"
WORD_DIR = ROOT / "data" / "words"


WORD_OPTS = dict(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False, use_text_flow=False)


def _rows(words: list) -> list:
    return [[round(float(w["x0"]), 2), round(float(w["top"]), 2),
             round(float(w["x1"]), 2), round(float(w["bottom"]), 2), w["text"]]
            for w in words if w["text"].strip()]


def extract_words(pdf_path: str, report: Optional[dict] = None) -> list:
    """Return one dict per page: {"w": width, "h": height, "words": [[x0, y0, x1, y1, text], ...]}.

    Pages whose fonts lost their character map ("(cid:N)" glyphs) are decoded
    by tieout.glyphs; when report is given it receives the decoding summary.
    """
    pages, cipher, keys = [], {}, {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(**WORD_OPTS)
            pages.append({"w": float(page.width), "h": float(page.height), "words": _rows(words)})
            if any(glyphs.CID.match(c["text"]) for c in page.chars):
                cipher[i], keys[i] = page.chars, glyphs.font_keys(page)
    if cipher:
        doc = pathlib.Path(pdf_path).stem
        decoded, units = glyphs.decode_pages(cipher, keys, vocabulary(exclude=doc), _arithmetic(cipher, keys, doc))
        for i, chars in decoded.items():
            pages[i]["words"] = _rows(pdfplumber.utils.extract_words(chars, **WORD_OPTS))
        if report is not None:
            report.update(pages_decoded=len(cipher), **glyphs.summary(units))
    return pages


def _arithmetic(cipher: dict, keys: dict, doc: str):
    """Judge for digit glyphs the crib left open: the reading under which the
    most table cells foot (tieout.calc formula recovery) wins."""
    def judge(units: dict, unit: glyphs.FontUnit, codes: list):
        digits = [c for c in codes if unit.candidates[c] <= set("0123456789")]
        pages = [i for i, chars in cipher.items()
                 if any(glyphs.cid_of(ch["text"]) in digits
                        and glyphs.unit_key(keys[i], i, ch["fontname"]) == unit.key for ch in chars)]
        if not digits or not pages:
            return None

        def footing(trial: dict) -> int:
            n = 0
            for i in pages:
                chars = glyphs.decode_chars(cipher[i], keys[i], i, units, unit, trial)
                words = [Word(*r) for r in _rows(pdfplumber.utils.extract_words(chars, **WORD_OPTS))]
                table = build_table(doc, i + 1, group_lines(words))
                n += coverage(table, discover(table))[0] if table is not None else 0
            return n
        return digits, footing
    return judge


def vocabulary(exclude: str = "") -> Counter:
    """How often each word is printed in the other cached filings: the crib for glyph decoding."""
    vocab: Counter = Counter()
    for f in sorted(WORD_DIR.glob("*.json.gz")):
        if f.name != f"{exclude}.json.gz":
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for p in json.load(fh):
                    vocab.update(w[4] for w in p["words"])
    return vocab


def cache_path(doc_id: str) -> pathlib.Path:
    return WORD_DIR / f"{doc_id}.json.gz"


def _extract_to_cache(doc_id: str) -> tuple:
    out = cache_path(doc_id)
    if out.exists():
        return doc_id, -1
    report: dict = {}
    pages = extract_words(str(PDF_DIR / f"{doc_id}.pdf"), report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(pages, f)
    if report:
        record_decoding(doc_id, report)
    return doc_id, len(pages)


def record_decoding(doc_id: str, report: dict) -> None:
    """Keep the glyph-decoding summary of each decoded filing in data/decoded_fonts.json."""
    path = ROOT / "data" / "decoded_fonts.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known[doc_id] = report
    path.write_text(json.dumps(known, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_all(doc_ids: list, workers: int = 2) -> None:
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
