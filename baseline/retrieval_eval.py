"""Retrieval evaluation with no LLM: gold page recall on the development questions, v1 against v2.

    python -m baseline.retrieval_eval [--out baseline/retrieval_eval.json]

Questions   the development sets only: benchmark/round2/dev.jsonl, benchmark/questions.jsonl and
            benchmark/dev.jsonl, de-duplicated by question id; unanswerable questions (no gold
            supporting facts) are left out.
Metric      a gold supporting fact is found when its (doc, page) is among the retrieved passages (v2)
            or pages (v1). Reported per type as
              recall   mean over questions of (gold facts found / gold facts)
              all      share of questions with every gold fact found
            at k = 4, 8, 12, 16, 24 retrieved items (v1: pages, v2: passages; the first k in selection
            order), and for the deployed prompts: v1 the top 8 pages (about 19,000 characters), v2 as
            many passages as fit 16,000 characters.
Variants    v1 pages (BM25 + TF-IDF); v2 hybrid with company scope (the default); and three ablations
            of v2: BM25 only, dense only, hybrid without company scope.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline import hybrid, index          # noqa: E402

DEV_FILES = ["benchmark/round2/dev.jsonl", "benchmark/questions.jsonl", "benchmark/dev.jsonl"]
TYPES = ["single_fact", "multi_hop", "cross_filing"]
KS = [4, 8, 12, 16, 24]


def dev_questions() -> list:
    seen, out = set(), []
    for f in DEV_FILES:
        for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                if q["id"] not in seen and q.get("supporting_facts") and "unanswerable" not in q.get("tags", []):
                    seen.add(q["id"])
                    out.append(q)
    return out


def found(q: dict, hits: list) -> tuple:
    got = {(h["doc"], h["page"]) for h in hits}
    gold = [(f["doc"], f.get("page")) for f in q["supporting_facts"]]
    return sum(g in got for g in gold), len(gold)


def summarise(rows: list) -> dict:
    out = {}
    for name, sub in [("all", rows)] + [(t, [r for r in rows if r["type"] == t]) for t in TYPES]:
        if sub:
            out[name] = {"n": len(sub), "recall": statistics.mean(r["f"] / r["g"] for r in sub),
                         "all": statistics.mean(r["f"] == r["g"] for r in sub)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "baseline" / "retrieval_eval.json"))
    args = ap.parse_args()
    qs = dev_questions()
    print(f"{len(qs)} answerable development questions: "
          + ", ".join(f"{t} {sum(q['type'] == t for q in qs)}" for t in TYPES), flush=True)
    t0 = time.time()
    index.load()
    hybrid.load()
    print(f"indexes loaded in {time.time() - t0:.1f} s", flush=True)

    variants = {
        "v1 pages (BM25+TF-IDF)": lambda q, k: index.retrieve(q, k),
        "v2 hybrid + company scope": lambda q, k: hybrid.retrieve(q, k=k),
        "v2 BM25 only + scope": lambda q, k: hybrid.retrieve(q, k=k, mode="bm25"),
        "v2 dense only + scope": lambda q, k: hybrid.retrieve(q, k=k, mode="dense"),
        "v2 hybrid, no scope": lambda q, k: hybrid.retrieve(q, k=k, company_aware=False),
    }
    result = {"questions": len(qs), "ks": KS, "variants": {}, "deployed": {}}
    for name, fn in variants.items():
        t1 = time.time()
        full = [(q, fn(q["question"], max(KS))) for q in qs]
        result["variants"][name] = {}
        for k in KS:
            rows = [{"type": q["type"], **dict(zip("fg", found(q, hits[:k])))} for q, hits in full]
            result["variants"][name][k] = summarise(rows)
        print(f"{name:28s} {time.time() - t1:5.1f} s", flush=True)

    deployed = {"v1 top-8 pages": lambda q: index.retrieve(q, 8),
                "v2 16,000-character budget": lambda q: hybrid.retrieve(q)}
    for name, fn in deployed.items():
        rows, sizes = [], []
        for q in qs:
            hits = fn(q["question"])
            rows.append({"type": q["type"], **dict(zip("fg", found(q, hits)))})
            sizes.append(len(hits))
        result["deployed"][name] = {"scores": summarise(rows), "mean_items": statistics.mean(sizes)}

    print("\nrecall (mean over questions of gold facts found / gold facts); all-gold share in brackets")
    for name, by_k in result["variants"].items():
        print(f"\n{name}")
        print(f"  {'k':>3s} " + " ".join(f"{t:>20s}" for t in ["all"] + TYPES))
        for k, s in by_k.items():
            print(f"  {k:>3s} " if isinstance(k, str) else f"  {k:>3d} ", end="")
            print(" ".join(f"{s[t]['recall']:12.3f} ({s[t]['all']:4.2f})" if t in s else f"{'-':>20s}"
                           for t in ["all"] + TYPES))
    print("\nDEPLOYED prompts")
    for name, d in result["deployed"].items():
        s = d["scores"]
        print(f"  {name:28s} items {d['mean_items']:5.1f}  " + "  ".join(
            f"{t} {s[t]['recall']:.3f} ({s[t]['all']:.2f})" for t in ["all"] + TYPES if t in s))
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
