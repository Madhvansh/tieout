"""Run the conventional RAG baseline over a benchmark file and score it.

    python scripts/run_baseline.py [benchmark.jsonl] [--retriever v2|v1] [--budget 16000] [--k 8]
                                   [--limit N] [--ids A01,A02] [--out results/baseline.json]

--retriever v2 (default) is the passage hybrid retriever (baseline/hybrid.py): at most --budget characters
of filing text per prompt. --retriever v1 is the page BM25 + TF-IDF baseline: the top --k pages.

Two precision/recall definitions, both reported:
  (a) fact-level  retrieved facts = the facts_used the LLM lists (ticker from the doc id, value in
                  full units), matched with tieout.eval.bench.fact_match; same rule as the graph system.
  (b) page-level  retrieved = the pages (v1) or passages (v2) given to the LLM. A gold fact is recalled if
                  its (doc, page) is among them; precision = share of the distinct pages holding a gold fact.
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


def _page_texts(pages: list, retrieved_text: list) -> dict:
    """{(doc, page): text} of what the LLM was shown (v2: the passage texts; v1: the page chunks)."""
    if retrieved_text:
        out: dict = {}
        for p, text in zip(pages, retrieved_text):
            out[(p["doc"], p["page"])] = out.get((p["doc"], p["page"]), "") + "\n" + text
        return out
    idx = index.load()
    want = {(p["doc"], p["page"]) for p in pages}
    return {(c["doc"], c["page"]): c["body"] for c in idx["chunks"] if (c["doc"], c["page"]) in want}


def page_match(q: dict, pages: list, retrieved_text: list = ()) -> dict:
    """Strict (b): gold (doc, page) among the k pages. Relaxed (b'), supplementary: the gold fact's
    printed figure appears on a retrieved page of the same company, any filing, any page; this
    mirrors fact_match, which also accepts the same number read from another page or filing."""
    got = {(p["doc"], p["page"]) for p in pages}
    gfs = q.get("supporting_facts", []) or []
    gold = [(gf["doc"], gf.get("page")) for gf in gfs]
    gold_set = set(gold)
    texts = _page_texts(pages, list(retrieved_text))
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
    ap.add_argument("--retriever", choices=["v2", "v1"], default="v2")
    ap.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="",
                    help="v2 ranking: hybrid (default, BM25 + dense by RRF), dense, or bm25")
    ap.add_argument("--budget", type=int, default=0, help="v2: characters of filing text per prompt (default 16000)")
    ap.add_argument("--k", type=int, default=ba.K, help="v1: pages per prompt")
    ap.add_argument("--model", default="", help="provider:model, as TIEOUT_LLM (default: TIEOUT_LLM or the llm.py default)")
    ap.add_argument("--ids", default="", help="comma-separated question ids to run")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to wait after each question that made a live call (per-minute token limits)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline.json"))
    args = ap.parse_args()

    from tieout.graph import Graph
    companies = Graph.load().companies           # only for mapping names to tickers when scoring

    questions = bench.load_benchmark(pathlib.Path(args.bench))
    if args.ids:
        want = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in want]
    if args.limit:
        questions = questions[: args.limit]
    model = args.model or llm.default_model()
    t_index = time.time()
    if args.retriever == "v1":
        index.load()
    else:
        from baseline import hybrid
        hybrid.load()
    t_index = time.time() - t_index
    budget = args.budget or (hybrid.BUDGET_CHARS if args.retriever == "v2" else 0)
    print(f"baseline  retriever={args.retriever}  model={model}  "
          + (f"budget={budget} chars" if args.retriever == "v2" else f"k={args.k}  page_budget={ba.PAGE_BUDGET}")
          + f"  questions={len(questions)}  index ready in {t_index:.1f} s", flush=True)

    rows, stopped = [], None
    for q in questions:
        try:
            out = ba.answer(q["question"], k=args.k, model=model, retriever=args.retriever, budget_chars=budget or None,
                            mode=args.mode or None)
        except llm.DailyLimit as e:                       # a run stops here; no abstentions are recorded
            stopped = f"{q['id']}: {e}"
            print(f"STOP daily limit at {stopped}", flush=True)
            break
        pred = scoring_pred(out["pred"], companies)
        row = {
            "id": q["id"], "type": q["type"], "question": q["question"], "tags": q.get("tags", []),
            "correct": bench.is_correct(q, pred, companies), "abstained": pred["kind"] == "abstain",
            "match": bench.fact_match(q, out["facts_used"], companies),
            "page_match": page_match(q, out["retrieved_pages"], out["retrieved_text"]),
            "answer": out["answer"], "pred": pred, "gold": q.get("answer"), "gold_value": q.get("answer_value"),
            "retrieved_pages": out["retrieved_pages"], "facts_used": out["facts_used"],
            "seconds": out["seconds"], "retrieval_seconds": out["retrieval_seconds"], "llm_calls": out["llm_calls"],
            "error": out["error"], "system": "baseline",
            "prompt_chars": out["prompt_chars"], "filing_chars": out["filing_chars"],
            "est_tokens_in": out["est_tokens_in"], "live_calls": out["live_calls"],
            "tokens_in": out.get("tokens_in"), "tokens_out": out.get("tokens_out"),
        }
        rows.append(row)
        if args.pace and out["live_calls"]:
            time.sleep(args.pace)
        print(f"{'OK ' if row['correct'] else 'ABS' if row['abstained'] else 'no '} {q['id']:14s} {q['type']:12s} "
              f"{out['seconds']:6.1f}s  {out['prompt_chars']:6d}ch  pages {row['page_match']['recalled']}/{row['page_match']['gold']}  "
              f"facts {row['match']['matched_gold']}/{row['match']['gold']}  {out['answer'][:60]!r}", flush=True)

    if not rows:
        print("no rows")
        return
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

    live = [r for r in rows if r["live_calls"]]
    tokens = {"questions": len(rows), "live_questions": len(live),
              "prompt_chars_mean": sum(r["prompt_chars"] for r in rows) / len(rows),
              "est_tokens_in_mean": sum(r["est_tokens_in"] for r in rows) / len(rows),
              "tokens_in_mean_live": sum(r["tokens_in"] for r in live) / len(live) if live else None,
              "tokens_out_mean_live": sum(r["tokens_out"] for r in live) / len(live) if live else None}
    print(f"\ntokens per question: {tokens}")
    params = ({"retriever": "v2", "budget_chars": budget, "passage_chars": "1000-1500", "mode": args.mode or hybrid.MODE, "dense_model": "BAAI/bge-small-en-v1.5",
               "bm25_k1": index.BM25_K1, "bm25_b": index.BM25_B, "rrf_k": hybrid.RRF_K, "rrf_depth": hybrid.RRF_DEPTH}
              if args.retriever == "v2" else
              {"retriever": "v1", "k": args.k, "page_budget_chars": ba.PAGE_BUDGET, "bm25_k1": index.BM25_K1,
               "bm25_b": index.BM25_B, "rrf_k": index.RRF_K, "rrf_depth": index.RRF_DEPTH})
    system = ("baseline v2 (passage BM25 + dense bge-small, RRF, company scope, LLM reads and computes)"
              if args.retriever == "v2" else "baseline v1 (page BM25 + TF-IDF, RRF, LLM reads and computes)")
    result = {"system": system, "model": model, "benchmark": str(args.bench), "params": params,
              "stopped": stopped, "tokens": tokens, "llm_calls": dict(llm.CALLS),
              "score_fact_level": fact_level, "score_page_level": page_level, "rows": rows}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}  (llm calls: {llm.CALLS['live']} live, {llm.CALLS['cached']} cached)")


if __name__ == "__main__":
    main()
