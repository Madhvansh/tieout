"""Run the benchmark and write results.

  python scripts/run_benchmark.py                       # the system as shipped (LLM plans, cached)
  python scripts/run_benchmark.py --oracle FILE         # hand-written plans: isolates the graph from the planner
  python scripts/run_benchmark.py --ablate concepts     # no shared concepts: a fact is found only under its literal label
  python scripts/run_benchmark.py --ablate solve        # no solving: no fourth quarter from the year, no identities
  python scripts/run_benchmark.py --ablate link         # no linker: only the three statements' shared concepts
  python scripts/run_benchmark.py --ablate checks       # serve readings whose formulas fail as if they were fine
  python scripts/run_benchmark.py --ablate store        # start from an empty learned store (learning still on)
  python scripts/run_benchmark.py --ablate living       # seed semantics only: nothing learned, before or during
  python scripts/run_benchmark.py --strict              # answer only from verified leaves
  python scripts/run_benchmark.py --noise 0.05          # rebuild the graph with 5% of statement cells damaged
  python scripts/run_benchmark.py --offline             # replay from the cache only; no LLM call is made
  python scripts/run_benchmark.py --ids A01,A06         # only these questions
  python scripts/run_benchmark.py --no-repair           # no second planner call after an executor failure
  python scripts/run_benchmark.py --persist             # development only: keep in the store the links that
                                                        # led to an answer its gold confirmed (outcome-gated)
  python scripts/run_benchmark.py --model M --checker C --arbiter A1,A2
                                                        # two readers and an arbiter (tieout/readers.py): M and C each
                                                        # plan every question; A1 (then A2 ...) is called on disagreement

Each row records the model calls the question made (live or replayed from the cache) with their
tokens and seconds. The result file is rewritten after every question, so a run that stops keeps
what it answered. If a provider's daily quota runs out the run stops with "complete": false, and
a rerun replays the answered questions from the cache at no cost. A question that raises an error
is recorded as wrong, with the error, and the run goes on.

Results go to results/<name>.json; a summary table is printed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib

from tieout import readers
from tieout.ask import answer
from tieout.concepts import audit_bindings, norm_label
from tieout.eval.bench import load_benchmark, run_graph_system, score
from tieout.execute import Executor
from tieout.graph import Graph, build
from tieout.link import load_links, start_question
from tieout.living import load_absent, load_learned
from tieout.llm import CALL_LOG, CALLS, DailyLimit

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]

# the plainest printed wording of each concept; with the concept layer off, only rows printed
# exactly like this are found
LITERAL = {
    "Revenue": {"revenue", "revenues"}, "CostOfRevenue": {"cost of revenue", "cost of revenues"},
    "GrossProfit": {"gross profit"}, "ResearchAndDevelopment": {"research and development"},
    "SellingGeneralAdmin": {"selling, general and administrative"}, "OperatingIncome": {"operating income"},
    "PretaxIncome": {"income before income taxes"}, "IncomeTax": {"income tax expense"},
    "NetIncome": {"net income"}, "EPSDiluted": {"diluted earnings per share"}, "Cash": {"cash and cash equivalents"},
    "TotalAssets": {"total assets"}, "TotalLiabilities": {"total liabilities"},
    "TotalCurrentAssets": {"total current assets"}, "TotalCurrentLiabilities": {"total current liabilities"},
    "TotalEquity": {"total equity", "total stockholders' equity", "total shareholders' equity"},
    "OperatingCashFlow": {"net cash provided by operating activities"},
    "CapitalExpenditures": {"capital expenditures"}, "Goodwill": {"goodwill"}, "Inventory": {"inventories", "inventory"},
    "LongTermDebt": {"long-term debt"}, "DividendsPaid": {"dividends paid"},
    "ShareRepurchases": {"repurchases of common stock"}, "StockBasedCompensation": {"stock-based compensation"},
    "DepreciationAmortization": {"depreciation and amortization"},
}


def strip_concept_layer(g: Graph) -> int:
    """Ablation: forget every binding whose printed label is not the literal concept wording."""
    dropped = 0
    for key in list(g.facts):
        f = g.facts[key]
        keep = [i for i in f.readings if norm_label(g.readings[i].label) in LITERAL.get(f.concept, set())]
        if keep:
            f.readings = keep
        else:
            del g.facts[key]
            dropped += 1
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "benchmark" / "questions.jsonl"))
    ap.add_argument("--oracle", default=None, help="JSON file of hand-written plans keyed by question id")
    ap.add_argument("--ablate", choices=["concepts", "solve", "link", "checks", "store", "living", "derivations"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-learning", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ids", default=None, help="comma-separated question ids to run")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--persist", action="store_true",
                    help="development runs only: save to the learned store the links behind answers that matched "
                         "their gold. A link from a wrong or abstained answer is not kept. The store is the "
                         "learning history a held-out run starts from; a held-out run never persists")
    ap.add_argument("--checker", default=None, help="second, independent planner model (see tieout/readers.py)")
    ap.add_argument("--arbiter", default="", help="comma-separated arbiter models, strongest first")
    ap.add_argument("--pick-checker", default=None,
                    help="second, independent model for row picks (the four-eyes rule in tieout/link.py); the "
                         "arbiters above settle its disagreements with the first picker")
    a = ap.parse_args()
    if a.pick_checker:                                # read by tieout/link.py when a row is picked
        os.environ["TIEOUT_LINK_CHECKER"] = a.pick_checker
        os.environ["TIEOUT_LINK_ARBITER"] = a.arbiter

    questions = load_benchmark(pathlib.Path(a.bench))
    if a.ids:
        keep = set(a.ids.split(","))
        questions = [q for q in questions if q["id"] in keep]
    fresh_store = a.ablate in ("living", "store")
    if a.noise > 0:
        g = build(verbose=False, learned=not fresh_store, corrupt=(a.noise, a.seed))
    else:
        g = Graph.load()
        if a.ablate != "checks":
            audit_bindings(g)                         # the arithmetic vetoes wrong bindings (off with the checks)
        if not fresh_store:
            load_learned(g)
            load_links(g)
            load_absent(g)
    if a.ablate == "concepts":
        print("concept layer off: facts dropped =", strip_concept_layer(g))
    ex = Executor(g, use_checks=a.ablate != "checks", use_derivations=a.ablate != "derivations",
                  use_solving=a.ablate != "solve", use_links=a.ablate != "link", require_verified=a.strict)
    oracle = {}
    if a.oracle:
        oracle = json.loads(pathlib.Path(a.oracle).read_text(encoding="utf-8"))

    panel = readers.Readers(a.model, a.checker, tuple(m for m in a.arbiter.split(",") if m)) if a.checker else None

    def run(q):
        common = dict(question_id=q["id"], offline=a.offline, executor=ex,
                      allow_learning=not (a.no_learning or a.ablate == "living" or a.noise > 0),
                      persist=False, repair=not a.no_repair)   # nothing is stored before its outcome is known
        if panel and not a.oracle:
            out = readers.answer(q["question"], g, panel, **common)
        else:
            start_question(readers.pick_arbiter_budget())
            out = answer(q["question"], g, model=a.model, plan=oracle.get(q["id"]) if a.oracle else None, **common)
        served.append(out)
        return out

    served: list = []
    keep = a.persist and not a.ablate and a.noise == 0

    name = a.name or "_".join(x for x in ["tieout", "oracle" if a.oracle else "", f"no-{a.ablate}" if a.ablate else "",
                                          "strict" if a.strict else "", f"noise{a.noise}" if a.noise else ""] if x)
    path = ROOT / "results" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    rows, complete = [], True
    for q in questions:
        start = len(CALL_LOG)
        try:
            row = run_graph_system([q], g, run, label=name)[0]
        except DailyLimit as e:
            print(f"STOPPED at {q['id']}: {e}. Rerun later; answered questions replay from the cache.")
            complete = False
            break
        except Exception as e:  # noqa: BLE001 - one question must never stop the run
            row = error_row(q, name, e)
        row.update(call_totals(CALL_LOG[start:]))
        if keep and row.get("correct") and served:
            row["kept_links"] = keep_links(served[-1], q["id"])
        rows.append(row)
        save(path, name, a, complete, rows, g)
    table = save(path, name, a, complete, rows, g)
    print(f"\n== {name} ==")
    print(f"{'type':14s} {'n':>3s} {'correct':>7s} {'accuracy':>9s} {'P macro':>8s} {'R macro':>8s} "
          f"{'P micro':>8s} {'R micro':>8s} {'abstain':>7s}")
    for t, s in table.items():
        fmt = lambda x: f"{x:8.3f}" if x is not None else "     n/a"  # noqa: E731
        print(f"{t:14s} {s['n']:3d} {s['correct']:7d} {s['accuracy']:9.3f} {fmt(s['precision_macro'])} "
              f"{fmt(s['recall_macro'])} {fmt(s['precision_micro'])} {fmt(s['recall_micro'])} {s['abstained']:7d}")
    if "overall" in table:
        bt = table["overall"]["by_tier"]
        print("trust: " + ", ".join(f"{t} {v['correct']}/{v['n']}" for t, v in bt.items())
              + f"; silent wrong (wrong while VERIFIED): {table['overall']['silent_wrong']}"
              + f"; wrong while WITNESSED: {table['overall'].get('witnessed_wrong', 0)}")
    for r in rows:
        if not r["correct"]:
            print(f"  x {r['id']} [{r['type']}] [{r.get('tier', '')}] gold={str(r['gold'])[:40]!r} got={r['answer'][:70]!r}")
    if any(r.get("reading") for r in rows):
        print_readings(rows)
    print(f"LLM calls: live {CALLS['live']}, cached {CALLS['cached']}" + ("" if complete else "  (INCOMPLETE)"))


def print_readings(rows: list) -> None:
    """Two readers and an arbiter: how often each case happened and how often it was right."""
    cases: dict = {}
    for r in rows:
        n, k = cases.get(r.get("reading"), (0, 0))
        cases[r.get("reading")] = (n + 1, k + bool(r["correct"]))
    print("readings: " + "; ".join(f"{case} {k}/{n}" for case, (n, k) in cases.items()))
    for role in ("maker", "checker", "arbiter"):
        seen = [x for r in rows for x in r.get("readers", []) if x["role"] == role]
        if seen:
            print(f"  {role} alone: {sum(x['correct'] for x in seen)}/{len(seen)}")


def keep_links(out: dict, question_id: str) -> int:
    """Outcome gate: write to the store the links and bindings this answer learned, now that its gold
    confirmed the answer. Returns how many were written."""
    from tieout.link import save_link
    from tieout.living import save_entry
    kept = 0
    for entry in out.get("learned", []):
        if entry.get("state") not in ("active", "flagged"):
            continue
        entry = {**entry, "confirmed_by": f"the answer to {question_id} matched its gold"}
        (save_link if entry.get("kind") in ("item", "mention", "breakdown") else save_entry)(entry)
        kept += 1
    return kept


def call_totals(calls: list) -> dict:
    """Model calls one question made, live or replayed, with the tokens and seconds of the original call."""
    by_kind: dict = {}
    by_model: dict = {}                               # the arbiter's daily quota is counted per model and key
    live_by_model: dict = {}                          # what a first run spends (a repeat is served from the cache)
    for c in calls:                                   # plan, link, labels, breakdown, bind ...
        kind = c["tag"].split("_")[0] if c["tag"] else "other"
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_model[c["model"]] = by_model.get(c["model"], 0) + 1
        if not c["cached"]:
            live_by_model[c["model"]] = live_by_model.get(c["model"], 0) + 1
    return {"llm_calls": len(calls), "llm_live": sum(1 for c in calls if not c["cached"]),
            "tokens_in": sum(c["prompt_tokens"] for c in calls), "tokens_out": sum(c["completion_tokens"] for c in calls),
            "llm_seconds": round(sum(c["seconds"] for c in calls), 2), "models": sorted({c["model"] for c in calls}),
            "calls_by_kind": by_kind, "calls_by_model": by_model, "live_by_model": live_by_model}


def error_row(q: dict, label: str, e: Exception) -> dict:
    """A question that raised: recorded as wrong, with the error, so the run can go on."""
    return {"id": q["id"], "type": q["type"], "question": q["question"], "gold": q.get("answer"),
            "answer": f"ERROR: {type(e).__name__}: {str(e)[:160]}", "correct": False, "abstained": False,
            "match": {"retrieved": 0, "matched_retrieved": 0, "gold": len(q.get("supporting_facts") or []),
                      "matched_gold": 0, "page_exact": 0},
            "plan": None, "notes": [], "learned": [], "seconds": 0.0, "system": label, "tags": q.get("tags", []),
            "error": True}


def save(path: pathlib.Path, name: str, a, complete: bool, rows: list, g) -> dict:
    """Write the result file atomically and return the score table."""
    table = score(rows) if rows else {}
    out = {"name": name, "args": vars(a), "complete": complete, "score": table, "rows": rows,
           "llm": dict(CALLS), "graph": g.stats()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return table


if __name__ == "__main__":
    main()
