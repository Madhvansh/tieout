"""Run the conventional RAG baseline over a benchmark file and score it.

    python scripts/run_baseline.py [benchmark.jsonl] [--k 8] [--limit N] [--out results/baseline.json]

Two precision/recall definitions, both reported:
  (a) fact-level  retrieved facts = the facts_used the LLM lists (ticker from the doc id, value in
                  full units), matched with tieout.eval.bench.fact_match; same rule as the graph system.
  (b) page-level  retrieved = the k pages given to the LLM. A gold fact is recalled if its (doc, page)
                  is among them; precision = share of the k pages holding at least one gold fact.
                  Questions with no gold facts (unanswerable) are left out of (b).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline import answer as ba            # noqa: E402
from baseline import index                   # noqa: E402
from tieout import llm                       # noqa: E402
from tieout.eval import bench                # noqa: E402


def scoring_pred(pred: dict, companies: dict) -> dict:
    """Company names as the LLM wrote them -> tickers, the form is_correct compares against."""
    if pred["kind"] != "companies":
        return pred
    return {"kind": "companies",
            "value": [[bench.company_to_ticker(n, companies) or bench._norm_text(n), v] for n, v in pred["value"]]}


def _page_texts(pages: list) -> dict:
    idx = index.load()
    want = {(p["doc"], p["page"]) for p in pages}
    return {(c["doc"], c["page"]): c["body"] for c in idx["chunks"] if (c["doc"], c["page"]) in want}


def page_match(q: dict, pages: list) -> dict:
    """Strict (b): gold (doc, page) among the k pages. Relaxed (b'), supplementary: the gold fact's
    printed figure appears on a retrieved page of the same company, any filing, any page; this
    mirrors fact_match, which also accepts the same number read from another page or filing."""
    got = {(p["doc"], p["page"]) for p in pages}
    gfs = q.get("supporting_facts", []) or []
    gold = [(gf["doc"], gf.get("page")) for gf in gfs]
    gold_set = set(gold)
    texts = _page_texts(pages)
    relaxed = 0
    for gf in gfs:
        tick, printed = gf["doc"].split("_")[0], str(gf.get("printed", "")).strip()
        if (gf["doc"], gf.get("page")) in got or (printed and any(
                d.split("_")[0] == tick and printed in t for (d, _), t in texts.items())):
            relaxed += 1
    return {"pages": len(got), "gold": len(gold), "recalled": sum(1 for g in gold if g in got),
            "relevant_pages": sum(1 for p in got if p in gold_set), "recalled_relaxed": relaxed}


def page_scores(rows: list) -> dict:
    out = {}
    for name, sub in [("overall", rows)] + [(t, [r for r in rows if r["type"] == t]) for t in bench.TYPES]:
        sub = [r for r in sub if r["page_match"]["gold"]]
        if not sub:
            continue
        pm = [r["page_match"] for r in sub]
        out[name] = {
            "n": len(sub),
            "precision_macro": sum(m["relevant_pages"] / m["pages"] for m in pm if m["pages"]) / len(pm),
            "recall_macro": sum(m["recalled"] / m["gold"] for m in pm) / len(pm),
            "precision_micro": sum(m["relevant_pages"] for m in pm) / max(1, sum(m["pages"] for m in pm)),
            "recall_micro": sum(m["recalled"] for m in pm) / max(1, sum(m["gold"] for m in pm)),
            "all_gold_pages_retrieved": sum(1 for m in pm if m["recalled"] == m["gold"]) / len(pm),
            "recall_relaxed_macro": sum(m["recalled_relaxed"] / m["gold"] for m in pm) / len(pm),
        }
    return out


def fmt(x) -> str:
    return "  -  " if x is None else f"{x:5.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?", default=str(bench.BENCH))
    ap.add_argument("--k", type=int, default=ba.K)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline.json"))
    args = ap.parse_args()

    from tieout.graph import Graph
    companies = Graph.load().companies           # only for mapping names to tickers when scoring

    questions = bench.load_benchmark(pathlib.Path(args.bench))
    if args.limit:
        questions = questions[: args.limit]
    model = llm.default_model()
    t_index = time.time()
    index.load()
    t_index = time.time() - t_index
    print(f"baseline  model={model}  k={args.k}  page_budget={ba.PAGE_BUDGET}  questions={len(questions)}  "
          f"index ready in {t_index:.1f} s", flush=True)

    rows = []
    for q in questions:
        out = ba.answer(q["question"], k=args.k, model=model)
        pred = scoring_pred(out["pred"], companies)
        row = {
            "id": q["id"], "type": q["type"], "question": q["question"], "tags": q.get("tags", []),
            "correct": bench.is_correct(q, pred, companies), "abstained": pred["kind"] == "abstain",
            "match": bench.fact_match(q, out["facts_used"], companies),
            "page_match": page_match(q, out["retrieved_pages"]),
            "answer": out["answer"], "pred": pred, "gold": q.get("answer"), "gold_value": q.get("answer_value"),
            "retrieved_pages": out["retrieved_pages"], "facts_used": out["facts_used"],
            "seconds": out["seconds"], "llm_calls": out["llm_calls"], "error": out["error"], "system": "baseline",
        }
        rows.append(row)
        print(f"{'OK ' if row['correct'] else 'ABS' if row['abstained'] else 'no '} {q['id']:14s} {q['type']:12s} "
              f"{out['seconds']:6.1f}s  pages {row['page_match']['recalled']}/{row['page_match']['gold']}  "
              f"facts {row['match']['matched_gold']}/{row['match']['gold']}  {out['answer'][:60]!r}", flush=True)

    fact_level = bench.score(rows)
    page_level = page_scores(rows)
    print("\n(a) FACT-LEVEL (retrieved = facts_used listed by the LLM; bench.fact_match, same rule as the graph)")
    print(f"{'type':13s} {'n':>3s} {'acc':>5s} {'ci95':>13s} {'P mac':>5s} {'R mac':>5s} {'P mic':>5s} {'R mic':>5s} "
          f"{'abst':>4s} {'ans.acc':>7s}")
    for t, s in fact_level.items():
        ci = f"[{s['accuracy_ci95'][0]:.2f},{s['accuracy_ci95'][1]:.2f}]"
        print(f"{t:13s} {s['n']:3d} {s['accuracy']:5.3f} {ci:>13s} {fmt(s['precision_macro'])} {fmt(s['recall_macro'])} "
              f"{fmt(s['precision_micro'])} {fmt(s['recall_micro'])} {s['abstained']:4d} {s['answered_accuracy']:7.3f}")
    print("\n(b) PAGE-LEVEL (retrieved = the k pages given to the LLM; answerable questions only)")
    print(f"{'type':13s} {'n':>3s} {'P mac':>5s} {'R mac':>5s} {'P mic':>5s} {'R mic':>5s} {'all-gold':>8s} "
          f"{'R relaxed mac':>13s}")
    for t, s in page_level.items():
        print(f"{t:13s} {s['n']:3d} {fmt(s['precision_macro'])} {fmt(s['recall_macro'])} {fmt(s['precision_micro'])} "
              f"{fmt(s['recall_micro'])} {s['all_gold_pages_retrieved']:8.3f} {fmt(s['recall_relaxed_macro']):>13s}")
    print("  R relaxed (supplementary): gold figure printed on any retrieved page of the same company")

    result = {"system": "baseline (page BM25 + TF-IDF, RRF, LLM reads and computes)", "model": model,
              "benchmark": str(args.bench), "params": {"k": args.k, "page_budget_chars": ba.PAGE_BUDGET,
                                                       "bm25_k1": index.BM25_K1, "bm25_b": index.BM25_B,
                                                       "rrf_k": index.RRF_K, "rrf_depth": index.RRF_DEPTH},
              "llm_calls": dict(llm.CALLS), "score_fact_level": fact_level, "score_page_level": page_level,
              "rows": rows}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}  (llm calls: {llm.CALLS['live']} live, {llm.CALLS['cached']} cached)")


if __name__ == "__main__":
    main()
