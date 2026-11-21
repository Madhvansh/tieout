"""Conventional retrieval baseline: page chunks, BM25 + TF-IDF, fused by reciprocal rank.

Chunk   one PDF page of one filing. Text = the page's words grouped into lines with
        tieout.tables.group_lines (reading order: top to bottom, left to right), with a
        header line "<company> (<ticker>) | <form> | fiscal year <FY> | period ended <date>
        | <doc_id> | page <n>" prepended so retrieval can tell filings apart.
        Page numbers are 1-based (PDF page 1 = page 1), the same convention as the graph.
Skipped MCD_FY2025_10K (unreadable text layer) and pages with no words.

Tuning choices (all standard defaults, none tuned on the benchmark):
  tokenizer   lowercase, split on [a-z0-9]+ ; numbers like 1,234 become two tokens, as in any
              off-the-shelf keyword index. No stemming, no stop list for BM25.
  BM25        rank_bm25.BM25Okapi with its defaults k1=1.5, b=0.75, epsilon=0.25.
  TF-IDF      sklearn TfidfVectorizer, same tokenizer, word unigrams+bigrams, sublinear_tf=True,
              min_df=2 (drops one-off tokens), English stop words removed, L2-normalised rows,
              so the dot product with the query vector is the cosine.
  fusion      reciprocal rank fusion, score = sum over the two rankings of 1 / (60 + rank),
              rank 1-based, k=60 as in Cormack et al. (2009); each ranking is cut at its top 200.
  wide gaps   a horizontal gap > 12 pt between two words on a line is written as " | " so a
              table row keeps visible column breaks for the reader (the LLM); tokens unaffected.
"""
from __future__ import annotations

import csv
import hashlib
import pathlib
import pickle
import re
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "baseline" / "cache"
INDEX_PATH = CACHE_DIR / "index.pkl"
MANIFEST = ROOT / "data" / "manifest.csv"
SKIP = {"MCD_FY2025_10K"}
VERSION = "page-bm25-tfidf-rrf-v1"

BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60
RRF_DEPTH = 200
GAP_PT = 12.0

_TOKEN = re.compile(r"[a-z0-9]+")
_INDEX = None


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


def _manifest() -> dict:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def page_text(words: list) -> str:
    from tieout.tables import group_lines
    out = []
    for ln in group_lines(words):
        parts = [ln[0].text]
        for prev, w in zip(ln, ln[1:]):
            parts.append((" | " if w.x0 - prev.x1 > GAP_PT else " ") + w.text)
        out.append("".join(parts))
    return "\n".join(out)


def build_chunks(verbose: bool = True) -> list:
    from tieout import pdfio
    chunks = []
    for doc_id, m in _manifest().items():
        if doc_id in SKIP or not pdfio.cache_path(doc_id).exists():
            continue
        pages = pdfio.load_pages(doc_id)
        for i, p in enumerate(pages):
            if not p["words"]:
                continue
            header = (f"{m['company']} ({m['ticker']}) | {m['form']} | fiscal year {m['fiscal_year']} | "
                      f"period ended {m['period_end']} | {doc_id} | page {i + 1}")
            chunks.append({"doc": doc_id, "page": i + 1, "header": header, "body": page_text(p["words"])})
        if verbose:
            print(f"chunks {doc_id:24s} {len(pages)} pages", flush=True)
    return chunks


def _fingerprint() -> str:
    from tieout import pdfio
    parts = [VERSION]
    for doc_id in sorted(_manifest()):
        p = pdfio.cache_path(doc_id)
        if doc_id not in SKIP and p.exists():
            st = p.stat()
            parts.append(f"{doc_id}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build(verbose: bool = True) -> dict:
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.time()
    chunks = build_chunks(verbose)
    texts = [c["header"] + "\n" + c["body"] for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in texts], k1=BM25_K1, b=BM25_B)
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2, stop_words="english")
    matrix = tfidf.fit_transform(texts)
    tfidf.stop_words_ = None          # the min_df-pruned term list is only for introspection; saves disk
    idx = {"fingerprint": _fingerprint(), "chunks": chunks, "bm25": bm25, "tfidf": tfidf, "matrix": matrix,
           "build_seconds": round(time.time() - t0, 1)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        mb = INDEX_PATH.stat().st_size / 1e6
        print(f"index  {len(chunks)} pages, built in {idx['build_seconds']} s, {mb:.1f} MB at {INDEX_PATH}")
    return idx


def load(rebuild: bool = False) -> dict:
    global _INDEX
    if _INDEX is not None and not rebuild:
        return _INDEX
    if INDEX_PATH.exists() and not rebuild:
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)
        if idx.get("fingerprint") == _fingerprint():
            _INDEX = idx
            return idx
    _INDEX = build()
    return _INDEX


def _ranks(scores: np.ndarray, depth: int) -> list:
    top = np.argsort(-scores, kind="stable")[:depth]
    return [int(i) for i in top if scores[i] > 0]


def retrieve(question: str, k: int = 8) -> list:
    """Top-k pages for the question: [{"doc", "page", "text", "score"}], best first."""
    idx = load()
    bm = idx["bm25"].get_scores(tokenize(question))
    tf = (idx["matrix"] @ idx["tfidf"].transform([question]).T).toarray().ravel()
    fused: dict = {}
    for ranking in (_ranks(bm, RRF_DEPTH), _ranks(tf, RRF_DEPTH)):
        for r, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
    best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
    out = []
    for i in best:
        c = idx["chunks"][i]
        out.append({"doc": c["doc"], "page": c["page"], "text": c["header"] + "\n" + c["body"],
                    "score": round(fused[i], 6)})
    return out


if __name__ == "__main__":
    import sys
    from baseline import index as _mod          # pickle must reference baseline.index.tokenize, not __main__
    _mod.load(rebuild="--rebuild" in sys.argv)
    for q in [a for a in sys.argv[1:] if not a.startswith("--")]:
        for h in _mod.retrieve(q, 8):
            print(f"{h['score']:.5f}  {h['doc']:24s} p{h['page']}")
