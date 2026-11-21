"""Run the benchmark and write results.

  python scripts/run_benchmark.py                       # the system as shipped (LLM plans, cached)
  python scripts/run_benchmark.py --oracle              # hand-written plans: isolates the graph from the planner
  python scripts/run_benchmark.py --ablate concepts     # no shared concepts: a fact is found only under its literal label
  python scripts/run_benchmark.py --ablate derivations  # no metric formulas, no period lattice
  python scripts/run_benchmark.py --ablate living       # seed semantics only, nothing learned from questions
  python scripts/run_benchmark.py --strict              # answer only from verified leaves
  python scripts/run_benchmark.py --noise 0.05          # rebuild the graph with 5% of statement cells damaged
  python scripts/run_benchmark.py --offline             # replay from the cache only; no LLM call is made

Results go to results/<name>.json; a summary table is printed.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib

from tieout.ask import answer
from tieout.concepts import norm_label
from tieout.eval.bench import load_benchmark, run_graph_system, score
from tieout.execute import Executor
from tieout.graph import Graph, build
from tieout.living import load_learned
from tieout.llm import CALLS

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
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--ablate", choices=["concepts", "derivations", "living", "checks"])
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-learning", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    questions = load_benchmark(pathlib.Path(a.bench))
    if a.noise > 0:
        g = build(verbose=False, learned=a.ablate != "living", corrupt=(a.noise, a.seed))
    else:
        g = Graph.load()
        if a.ablate != "living":
            load_learned(g)
    if a.ablate == "concepts":
        print("concept layer off: facts dropped =", strip_concept_layer(g))
    ex = Executor(g, use_checks=a.ablate != "checks", use_derivations=a.ablate != "derivations",
                  require_verified=a.strict)
    oracle = {}
    if a.oracle:
        oracle = json.loads((ROOT / "benchmark" / "oracle_plans.json").read_text(encoding="utf-8"))

    def run(q):
        return answer(q["question"], g, model=a.model, question_id=q["id"], offline=a.offline,
                      plan=oracle.get(q["id"]) if a.oracle else None, executor=ex,
                      allow_learning=not (a.no_learning or a.ablate == "living" or a.noise > 0), persist=False)

    name = a.name or "_".join(x for x in ["tieout", "oracle" if a.oracle else "", f"no-{a.ablate}" if a.ablate else "",
                                          "strict" if a.strict else "", f"noise{a.noise}" if a.noise else ""] if x)
    rows = run_graph_system(questions, g, run, label=name)
    table = score(rows)
    out = {"name": name, "args": vars(a), "score": table, "rows": rows,
           "llm": dict(CALLS), "graph": g.stats()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\n== {name} ==")
    print(f"{'type':14s} {'n':>3s} {'correct':>7s} {'accuracy':>9s} {'P macro':>8s} {'R macro':>8s} {'P micro':>8s} {'R micro':>8s} {'abstain':>7s}")
    for t, s in table.items():
        fmt = lambda x: f"{x:8.3f}" if x is not None else "     n/a"
        print(f"{t:14s} {s['n']:3d} {s['correct']:7d} {s['accuracy']:9.3f} {fmt(s['precision_macro'])} {fmt(s['recall_macro'])} "
              f"{fmt(s['precision_micro'])} {fmt(s['recall_micro'])} {s['abstained']:7d}")
    wrong = [r for r in rows if not r["correct"]]
    for r in wrong:
        print(f"  x {r['id']} [{r['type']}] gold={str(r['gold'])[:40]!r} got={r['answer'][:70]!r}")
    print(f"LLM calls: live {CALLS['live']}, cached {CALLS['cached']}")


if __name__ == "__main__":
    main()
