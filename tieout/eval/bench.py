"""Score a system on the benchmark: accuracy, precision and recall, by question type.

Definitions (fixed before any system was run on the questions)

Accuracy     correct answers / questions.
               numeric      within the question's relative tolerance of the gold value
               text         same after normalisation (case, punctuation, "LLP")
               company      the same company; for a ranked list, the same companies in the same order
               unanswerable correct only if the system abstains

Retrieved    every fact the system read to produce its answer: for the graph system, every
facts        leaf of the proof tree, including every company scanned for a ranking. Nothing
             is trimmed after the fact.

Gold facts   written by hand from the PDF pages as (document, page, printed value), before the
             system was run. They do not refer to graph ids.

A retrieved fact matches a gold fact when it is the same company and the same value in full
units (so the same number read from another page or another filing still counts, and is
reported separately as not page-exact).

Precision    matched retrieved facts / retrieved facts.   Abstention with nothing retrieved: 1.
Recall       matched gold facts / gold facts.             Unanswerable questions have no gold facts
                                                          and are left out of recall.
Both are reported macro (mean over questions) and micro (pooled counts).
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import re
from collections import defaultdict
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark" / "questions.jsonl"
TYPES = ("single_fact", "multi_hop", "cross_filing")


def load_benchmark(path: pathlib.Path = BENCH) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_text(x) -> str:
    s = str(x).lower()
    s = re.sub(r"\b(llp|inc|corp|corporation|company|co|the|plc|n\.a)\b\.?", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def company_to_ticker(x, companies: dict) -> Optional[str]:
    """Gold answers name companies as people do; map them to tickers."""
    if x in companies:
        return x
    n = _norm_text(x)
    alias = {"google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "jj": "JNJ", "johnsonjohnson": "JNJ",
             "jpmorgan": "JPM", "jpmorganchase": "JPM", "bankofamerica": "BAC", "bofa": "BAC", "exxon": "XOM",
             "exxonmobil": "XOM", "cocacola": "KO", "coke": "KO", "pg": "PG", "proctergamble": "PG",
             "unitedhealth": "UNH", "unitedhealthgroup": "UNH", "wellsfargo": "WFC", "goldmansachs": "GS",
             "goldman": "GS", "morganstanley": "MS", "americanexpress": "AXP", "amex": "AXP", "homedepot": "HD",
             "generalmotors": "GM", "gm": "GM", "citigroup": "C", "citi": "C", "merck": "MRK", "pfizer": "PFE"}
    if n in alias:
        return alias[n]
    for t, c in companies.items():
        cn = _norm_text(c["name"])
        if n and (n == cn or n in cn or cn in n):
            return t
    return None


def is_correct(q: dict, pred: dict, companies: dict) -> bool:
    """pred = {"kind": number|text|companies|abstain, "value": ...}"""
    gold = q.get("answer_value")
    if gold is None or "unanswerable" in q.get("tags", []):
        return pred["kind"] == "abstain"
    if pred["kind"] == "abstain":
        return False
    unit = q.get("answer_unit", "")
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        if pred["kind"] != "number":
            return False
        tol = q.get("tolerance", 0.005) or 0.0
        v = pred["value"]
        if abs(gold) < 1e-12:
            return abs(v) < 1e-9
        return abs(v - gold) <= max(tol, 1e-9) * abs(gold) or (unit == "USD" and abs(abs(v) - abs(gold)) <= max(tol, 1e-9) * abs(gold))
    if isinstance(gold, list):
        want = [company_to_ticker(x, companies) or _norm_text(x) for x in gold]
        if pred["kind"] != "companies":
            return False
        got = [t for t, _ in pred["value"]]
        return got[:len(want)] == want
    if unit == "company":
        want = company_to_ticker(gold, companies)
        if pred["kind"] == "companies":
            return bool(pred["value"]) and pred["value"][0][0] == want
        return company_to_ticker(pred["value"], companies) == want
    g, p = _norm_text(gold), _norm_text(pred["value"])
    return bool(g) and (g == p or g in p or p in g)


def fact_match(q: dict, retrieved: list, companies: dict) -> dict:
    """retrieved = [{"ticker", "value", "doc", "page"}, ...] -> counts for precision and recall."""
    gold = q.get("supporting_facts", []) or []
    gold_keys = []
    for gf in gold:
        ticker = gf["doc"].split("_")[0]
        gold_keys.append((ticker, _vkey(gf.get("value", gf.get("printed"))), gf["doc"], gf.get("page")))
    used = set()
    matched_ret = page_exact = 0
    for rf in retrieved:
        key = (rf["ticker"], _vkey(rf["value"]))
        hit = next((i for i, gk in enumerate(gold_keys) if (gk[0], gk[1]) == key), None)
        if hit is not None:
            matched_ret += 1
            used.add(hit)
            if any(gk[2] == rf["doc"] and gk[3] == rf["page"] and (gk[0], gk[1]) == key for gk in gold_keys):
                page_exact += 1
    return {"retrieved": len(retrieved), "matched_retrieved": matched_ret, "gold": len(gold_keys),
            "matched_gold": len(used), "page_exact": page_exact}


def _vkey(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return ("n", round(abs(float(v)), 4))        # outflows are printed negative and asked as amounts
    s = str(v).strip()
    if re.fullmatch(r"\(?-?[\d,]+(\.\d+)?\)?", s):
        return ("n", round(abs(float(s.strip("()").replace(",", ""))), 4))
    return ("t", _norm_text(s))


def score(rows: list) -> dict:
    """rows: per-question dicts with type, correct, match (from fact_match), abstained."""
    out = {}
    for name, subset in [("overall", rows)] + [(t, [r for r in rows if r["type"] == t]) for t in TYPES]:
        n = len(subset)
        if not n:
            continue
        k = sum(1 for r in subset if r["correct"])
        prec = [(r["match"]["matched_retrieved"] / r["match"]["retrieved"]) if r["match"]["retrieved"] else
                (1.0 if (r["abstained"] or r["match"]["gold"] == 0) else 0.0) for r in subset]
        rec = [r["match"]["matched_gold"] / r["match"]["gold"] for r in subset if r["match"]["gold"]]
        tr = sum(r["match"]["retrieved"] for r in subset)
        tg = sum(r["match"]["gold"] for r in subset)
        out[name] = {
            "n": n, "correct": k, "accuracy": k / n, "accuracy_ci95": wilson(k, n),
            "precision_macro": sum(prec) / len(prec), "recall_macro": (sum(rec) / len(rec)) if rec else None,
            "precision_micro": (sum(r["match"]["matched_retrieved"] for r in subset) / tr) if tr else None,
            "recall_micro": (sum(r["match"]["matched_gold"] for r in subset) / tg) if tg else None,
            "page_exact_share": (sum(r["match"]["page_exact"] for r in subset) /
                                 max(1, sum(r["match"]["matched_retrieved"] for r in subset))),
            "abstained": sum(1 for r in subset if r["abstained"]),
            "answered_accuracy": (sum(1 for r in subset if r["correct"] and not r["abstained"]) /
                                  max(1, sum(1 for r in subset if not r["abstained"]))),
        }
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def paired_bootstrap(a: list, b: list, n_boot: int = 10000, seed: int = 11) -> dict:
    """a, b: 0/1 correctness of two systems on the same questions. Difference a - b."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return {"diff": (sum(a) - sum(b)) / n, "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]],
            "a_only": sum(1 for x, y in zip(a, b) if x and not y), "b_only": sum(1 for x, y in zip(a, b) if y and not x),
            "n": n, "n_boot": n_boot, "seed": seed}


def run_graph_system(questions: list, g, answer_fn: Callable, label: str = "tieout") -> list:
    """answer_fn(question dict) -> the dict returned by tieout.ask.answer."""
    rows = []
    for q in questions:
        out = answer_fn(q)
        res = out["result"]
        retrieved = [{"ticker": lf.ticker, "value": lf.value, "doc": lf.doc, "page": lf.page} for lf in res.leaves]
        pred = {"kind": res.kind, "value": res.value}
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "gold": q.get("answer"),
            "answer": out["answer"], "correct": is_correct(q, pred, g.companies),
            "abstained": res.kind == "abstain", "match": fact_match(q, retrieved, g.companies),
            "plan": out.get("plan"), "notes": list(dict.fromkeys(res.notes))[:6],
            "learned": [{k: e.get(k) for k in ("ticker", "concept", "state", "label")} for e in out.get("learned", [])],
            "seconds": round(out.get("seconds", 0.0), 2), "system": label, "tags": q.get("tags", []),
        })
    return rows
