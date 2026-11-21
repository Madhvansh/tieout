"""The teaching run: a log of plain questions, one per (company, concept) the seed semantics could not bind.

Each question ("What was Citigroup's net income in its latest fiscal year?") is one the graph
cannot answer yet. The living layer asks an LLM to pick the row from the labels, the structural
gate judges the pick, and the outcome is written to
    semantics/learned.jsonl        admitted bindings (applied on every later build)
    semantics/review_queue.jsonl   proposals the gate could not confirm (for a person)
    results/teaching_log.jsonl     every probe with the proposal, the evidence and the objections
The benchmark questions are not part of this log.

Usage: python scripts/teach.py [--limit N] [--concepts A,B] [--rpm 13]
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

from tieout.concepts import concept_defs
from tieout.graph import Graph
from tieout.living import SAYS, learn, load_learned, save_entry

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "teaching_log.jsonl"
BANK_ONLY = {"NetInterestIncome", "ProvisionForCreditLosses", "NoninterestExpense", "Deposits"}
NOT_FOR_BANKS = {"GrossProfit", "CostOfRevenue", "Inventory", "TotalCurrentAssets", "TotalCurrentLiabilities",
                 "ResearchAndDevelopment", "OperatingIncome", "SellingGeneralAdmin"}
# the order in which gaps are probed: statements' backbone first
ORDER = ["NetIncome", "Revenue", "TotalAssets", "TotalEquity", "OperatingCashFlow", "Cash", "PretaxIncome", "IncomeTax",
         "OperatingIncome", "CostOfRevenue", "TotalLiabilitiesAndEquity", "InvestingCashFlow", "FinancingCashFlow",
         "CapitalExpenditures", "DepreciationAmortization", "DividendsPaid", "ShareRepurchases",
         "LongTermDebt", "TotalLiabilities", "StockBasedCompensation", "Deposits", "NetInterestIncome",
         "ProvisionForCreditLosses", "NoninterestExpense", "ResearchAndDevelopment", "SellingGeneralAdmin", "GrossProfit",
         "Goodwill", "Inventory", "TotalCurrentAssets", "TotalCurrentLiabilities"]


def gaps(g: Graph) -> list:
    latest = {f["ticker"]: d for d, f in g.filings.items() if f["role"] == "latest" and f.get("readable", True)}
    have = {(g.filings[b.doc]["ticker"], b.concept) for b in g.bindings if b.state == "active" and b.doc in latest.values()}
    out = []
    for c in ORDER:
        for t in sorted(latest):
            if (t, c) in have:
                continue
            bank = g.companies[t]["cluster"] == "banks_finance"
            if (c in BANK_ONLY and not bank) or (c in NOT_FOR_BANKS and bank):
                continue
            out.append((t, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--concepts", default="")
    ap.add_argument("--rpm", type=float, default=13.0)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    g = Graph.load()
    load_learned(g)
    done = set()
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            done.add((e["ticker"], e["concept"]))
    todo = [x for x in gaps(g) if x not in done]
    if a.concepts:
        keep = set(a.concepts.split(","))
        todo = [x for x in todo if x[1] in keep]
    todo = todo[:a.limit]
    print(f"{len(todo)} probes to run ({len(done)} already logged)", flush=True)
    LOG.parent.mkdir(exist_ok=True)
    gap = 60.0 / a.rpm
    counts = {"active": 0, "proposed": 0, "rejected": 0}
    for n, (t, c) in enumerate(todo):
        t0 = time.time()
        q = f"What was {g.companies[t]['name']}'s {SAYS.get(c, c).split(' (')[0]} in its latest fiscal year?"
        e = learn(g, t, c, question_id=f"T{len(done) + n + 1:03d}", question=q, model=a.model)
        counts[e["state"]] += 1
        if e["state"] != "rejected":
            save_entry(e)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")
        print(f"{n + 1:3d}/{len(todo)} {t:5s} {c:26s} {e['state']:9s} {str(e['label'])[:44]!r:48s} "
              f"{('- ' + e['objections'][0][:60]) if e['objections'] else ''}", flush=True)
        if any("proposer failed" in o for o in e["objections"]):
            time.sleep(20)                                # quota or outage: back off, the probe is retried on a rerun
        time.sleep(max(0.0, gap - (time.time() - t0)))
    print("done:", counts)


if __name__ == "__main__":
    main()
