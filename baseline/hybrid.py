"""The v2 conventional retriever: passages, BM25 + dense embeddings fused by reciprocal rank,
searched inside the companies a question names.

Index     baseline/passages.py: about 28,500 passages of 1,000 to 1,500 characters, each starting with
          a line naming company, ticker, form, fiscal year, period end, doc id and page. The indexed
          text and the text shown to the LLM are that line plus the passage body.
BM25      rank_bm25.BM25Okapi, defaults k1=1.5 b=0.75, the v1 tokenizer (lowercase [a-z0-9]+).
Dense     BAAI/bge-small-en-v1.5 (baseline/dense.py), the question with the model's query instruction.
          Passage vectors are cached on disk in 2,000-passage shards, so an interrupted build resumes.
Fusion    reciprocal rank fusion, score = sum of 1 / (60 + rank) over the two rankings, each cut at
          its top 200 inside the searched scope (Cormack et al. 2009; the same fusion as v1).
Scope     baseline/scope.py names the companies in the question. Each named company is searched
          alone (its own filings), and the passage budget is shared out by round robin: the best
          passage of each company in the order the question names them, then the second best of each,
          and so on, until the character budget (or k passages) is spent. No company named: one
          search over every passage.
Budget    at most BUDGET_CHARS characters of filing text (passage bodies plus their header lines).
"""
from __future__ import annotations

import hashlib
import pathlib
import pickle
import time

import numpy as np

from baseline import dense, index, passages, scope

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "baseline" / "cache" / "v2"
PASSAGES_PATH = DIR / "passages.pkl"
VERSION = "passage-hybrid-v2"
BUDGET_CHARS = 16_000
MODE = "hybrid"                 # "hybrid" (BM25 + dense, RRF), "dense" or "bm25"; see baseline/README.md
RRF_K = 60
RRF_DEPTH = 200
SHARD = 2000

_INDEX = None
_COMPANIES = None


def text_of(p: dict) -> str:
    return p["meta"] + "\n" + p["body"]


def _fingerprint() -> str:
    from tieout import pdfio
    parts = [VERSION, dense.REVISION, hashlib.sha256(pathlib.Path(passages.__file__).read_bytes()).hexdigest()]
    for doc_id in sorted(index.manifest()):
        path = pdfio.cache_path(doc_id)
        if doc_id not in index.SKIP and path.exists():
            parts.append(f"{doc_id}:{path.stat().st_size}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _embed(texts: list, verbose: bool) -> np.ndarray:
    """Passage vectors, from the shard files where they exist, else encoded now and saved."""
    emb_dir = DIR / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros((len(texts), dense.DIM), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(texts), SHARD):
        chunk = texts[s:s + SHARD]
        key = hashlib.sha256("\n".join(chunk).encode()).hexdigest()[:12]
        path = emb_dir / f"shard_{s // SHARD:03d}_{key}.npy"
        if path.exists():
            out[s:s + len(chunk)] = np.load(path)
        else:
            out[s:s + len(chunk)] = dense.encoder().encode(chunk)
            np.save(path, out[s:s + len(chunk)])
        if verbose:
            done = min(len(texts), s + SHARD)
            print(f"embed  {done:6d}/{len(texts)}  {time.time() - t0:6.0f} s", flush=True)
    return out


def build(verbose: bool = True) -> dict:
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    ps = passages.build(verbose=False)
    t_pass = time.time() - t0
    texts = [text_of(p) for p in ps]
    t1 = time.time()
    bm25 = BM25Okapi([index.tokenize(t) for t in texts], k1=index.BM25_K1, b=index.BM25_B)
    t_bm25 = time.time() - t1
    t2 = time.time()
    vectors = _embed(texts, verbose)
    t_dense = time.time() - t2
    idx = {"fingerprint": _fingerprint(), "passages": ps, "bm25": bm25, "vectors": vectors,
           "seconds": {"passages": round(t_pass, 1), "bm25": round(t_bm25, 1), "dense": round(t_dense, 1),
                       "total": round(time.time() - t0, 1)}}
    DIR.mkdir(parents=True, exist_ok=True)
    with open(PASSAGES_PATH, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"index  {len(ps)} passages built in {idx['seconds']}  {PASSAGES_PATH.stat().st_size / 1e6:.0f} MB")
    return idx


def load(rebuild: bool = False) -> dict:
    global _INDEX
    if _INDEX is not None and not rebuild:
        return _INDEX
    if PASSAGES_PATH.exists() and not rebuild:
        with open(PASSAGES_PATH, "rb") as f:
            idx = pickle.load(f)
        if idx.get("fingerprint") == _fingerprint():
            _INDEX = _prepare(idx)
            return _INDEX
    _INDEX = _prepare(build())
    return _INDEX


def _prepare(idx: dict) -> dict:
    """Lookup arrays derived from the passages (not stored)."""
    ps = idx["passages"]
    by_ticker: dict = {}
    for i, p in enumerate(ps):
        by_ticker.setdefault(p["ticker"], []).append(i)
    idx["by_ticker"] = {t: np.array(v) for t, v in by_ticker.items()}
    idx["all"] = np.arange(len(ps))
    return idx


def companies() -> dict:
    global _COMPANIES
    if _COMPANIES is None:
        from tieout.graph import Graph
        _COMPANIES = Graph.load().companies
    return _COMPANIES


def _top(scores: np.ndarray, members: np.ndarray, depth: int) -> list:
    """The `depth` best passages among `members`, by score, ties by position; zero scores dropped."""
    s = scores[members]
    order = np.argsort(-s, kind="stable")[:depth]
    return [int(members[i]) for i in order if s[i] > 0]


def rank_scope(question: str, members: np.ndarray, mode: str, bm: np.ndarray, qv) -> list:
    """Passage indexes of one scope, best first. mode: bm25, dense or hybrid."""
    idx = load()
    rankings = []
    if mode in ("bm25", "hybrid"):
        rankings.append(_top(bm, members, RRF_DEPTH))
    if mode in ("dense", "hybrid"):
        rankings.append(_top(idx["vectors"] @ qv, members, RRF_DEPTH))
    if len(rankings) == 1:
        return rankings[0]
    fused: dict = {}
    for ranking in rankings:
        for r, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
    return sorted(fused, key=lambda i: (-fused[i], i))


def retrieve(question: str, budget_chars: int = BUDGET_CHARS, k: int = None, mode: str = None,
             company_aware: bool = True) -> list:
    """Passages for the question, as dicts {"doc", "page", "part", "ticker", "scope", "rank", "text"}.

    The list is in selection order. Stops at k passages when k is given, else when the next passage
    would take the prompt past budget_chars characters."""
    idx = load()
    mode = mode or MODE
    named = scope.detect(question, companies()) if company_aware else []
    named = [t for t in named if t in idx["by_ticker"]]
    scopes = [idx["by_ticker"][t] for t in named] or [idx["all"]]
    bm = idx["bm25"].get_scores(index.tokenize(question)) if mode in ("bm25", "hybrid") else None
    qv = dense.embed_query(question) if mode in ("dense", "hybrid") else None
    ranked = [rank_scope(question, m, mode, bm, qv) for m in scopes]
    chosen, used, depth = [], 0, 0
    live = list(range(len(scopes)))
    while live and (k is None or len(chosen) < k):
        nxt = []
        for s in live:
            if k is not None and len(chosen) >= k:
                break
            if depth >= len(ranked[s]):
                continue
            i = ranked[s][depth]
            size = len(text_of(idx["passages"][i])) + 2
            if k is None and used + size > budget_chars:
                continue                        # does not fit: this scope is done
            chosen.append((s, depth, i))
            used += size
            nxt.append(s)
        live, depth = nxt, depth + 1
    out = []
    for s, rank, i in chosen:
        p = idx["passages"][i]
        out.append({"doc": p["doc"], "page": p["page"], "part": p["part"], "ticker": p["ticker"], "scope": s,
                    "rank": rank + 1, "text": text_of(p)})
    return out


def peak_memory_mb() -> float:
    import psutil
    return psutil.Process().memory_info().peak_wset / 1e6


if __name__ == "__main__":
    import sys
    load(rebuild="--rebuild" in sys.argv)
    for q in [a for a in sys.argv[1:] if not a.startswith("--")]:
        for h in retrieve(q):
            print(f"s{h['scope']} r{h['rank']:2d}  {h['doc']:22s} p{h['page']:<4d} part {h['part']}")
