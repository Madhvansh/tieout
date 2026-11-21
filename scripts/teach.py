"""The coverage pass: bind every concept the statements print, and record what a company does not report.

For each company's latest annual report and each primary statement, the concepts with no row bound
yet are proposed in ONE model call (row labels, sections and roles only; never a value). Each
proposal goes through the structural gate of tieout.living: admitted bindings are applied and saved
to semantics/learned.jsonl; held ones are only logged.

A concept that is still unbound becomes a stored fact "this company does not report it"
(semantics/absent.jsonl) only when two pieces of evidence agree:
  1. the statement was read whole: at least 80% of its amount rows sit inside passing formulas,
     so a missing line is not a parsing gap;
  2. no table row or sentence anywhere in the company's filings carries the concept's words
     (the linker's own candidate search finds nothing that covers them).
Otherwise the gap stays open, and a question that needs it tries the linker at question time.

Usage: python scripts/teach.py [--model groq:openai/gpt-oss-20b@low] [--tickers AAPL,MSFT] [--rpm 4]
Writes results/coverage_pass.json: counts before and after, every proposal, every absence.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import time

from tieout.concepts import concept_defs
from tieout.execute import concept_words
from tieout.graph import Graph
from tieout.link import candidates, load_links, store
from tieout.living import (ABSENT, SAYS, apply_entry, closure, gate, latest_doc, load_learned, propose_rows,
                           row_table, save_entry)
from tieout.llm import DailyLimit

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]
BANK_ONLY = {"NetInterestIncome", "ProvisionForCreditLosses", "NoninterestExpense", "NoninterestIncome", "Deposits", "Loans"}
NOT_FOR_BANKS = {"GrossProfit", "CostOfRevenue", "Inventory", "TotalCurrentAssets", "TotalCurrentLiabilities",
                 "ResearchAndDevelopment", "SellingGeneralAdmin", "SalesAndMarketing", "GeneralAndAdministrative"}
CLOSED = 0.80


def unbound(g: Graph, ticker: str) -> dict:
    """{statement: [concepts with no active binding in the latest filing]}."""
    doc = latest_doc(g, ticker)
    have = {b.concept for b in g.bindings if b.doc == doc and b.state == "active"}
    out: dict = {}
    for name, c in concept_defs().items():
        if name not in have and not any(k.startswith(f"{ticker}|{name}|") for k in g.facts):
            out.setdefault(c.statement, []).append(name)
    return out


def meaning(concept: str) -> str:
    """The concept's meaning in plain words, as far as the first bracket or slash ("diluted earnings per share")."""
    return re.split(r"[(/;]", SAYS.get(concept, ""), maxsplit=1)[0].strip() or concept_words(concept)


def applicable(g: Graph, ticker: str, concept: str) -> bool:
    bank = g.companies[ticker]["cluster"] == "banks_finance"
    return not ((concept in BANK_ONLY and not bank) or (concept in NOT_FOR_BANKS and bank))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="the model that proposes rows (TIEOUT_LINK_LLM otherwise)")
    ap.add_argument("--tickers", default="")
    ap.add_argument("--rpm", type=float, default=4.0)
    a = ap.parse_args()
    g = Graph.load()
    load_learned(g)
    load_links(g)
    tickers = a.tickers.split(",") if a.tickers else sorted(g.companies)
    before = {t: sum(len(v) for v in unbound(g, t).values()) for t in tickers}
    log = {"proposals": [], "absent": [], "before": before}
    st = store()
    gap = 60.0 / a.rpm
    absent_lines = []
    try:
        for t in tickers:
            doc = latest_doc(g, t)
            if doc is None:
                continue
            for kind, concepts in sorted(unbound(g, t).items()):
                asked = [c for c in concepts if applicable(g, t, c)]
                t0 = time.time()
                try:
                    picks = propose_rows(g, t, kind, asked, model=a.model) if asked else {}
                except DailyLimit:
                    raise
                except Exception as e:  # noqa: BLE001 - one statement's failure is logged, the pass goes on
                    print(f"{t} {kind}: proposer failed ({type(e).__name__}); left open", flush=True)
                    continue
                rows = row_table(g, doc, kind)
                for c, row in picks.items():
                    if row is None:
                        continue
                    ok, evidence, objections = gate(g, doc, c, row, rows)
                    entry = {"ticker": t, "concept": c, "kind": kind, "doc": doc, "taught_by": "coverage-pass",
                             "question": "", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "state": "active" if ok else "proposed", "label": rows[row]["label"], "row": row,
                             "evidence": evidence, "objections": objections, "proposer": a.model or ""}
                    if ok:
                        apply_entry(g, entry)
                        save_entry(entry)
                    log["proposals"].append({k: entry[k] for k in ("ticker", "concept", "state", "label", "objections")})
                verified, total = closure(g, doc, kind)
                said_none = {c for c, row in picks.items() if row is None}       # asked, and the proposer found no row
                for c in unbound(g, t).get(kind, []):
                    if applicable(g, t, c) and c not in said_none:
                        continue                      # proposed but held, or never asked: the gap stays open
                    words = concept_words(c)
                    fy = g.filings[doc]["fiscal_year"]
                    found = [x for phrase in (words, meaning(c))
                             for x in candidates(st, g, t, phrase, {fy, fy + "@end"}, fy) if x["covered"]]
                    if total and verified / total >= CLOSED and not found:
                        evidence = (f"its {kind} statement ({doc}) was read whole: {verified} of {total} amount rows sit "
                                    f"inside passing formulas, none of them is {c}, and no table row or sentence in "
                                    f"its filings carries the words '{words}'")
                        absent_lines.append({"ticker": t, "concept": c, "doc": doc, "evidence": evidence})
                        log["absent"].append({"ticker": t, "concept": c})
                if asked:
                    time.sleep(max(0.0, gap - (time.time() - t0)))
            print(f"{t:5s} unbound before {before[t]:3d}  after {sum(len(v) for v in unbound(g, t).values()):3d}", flush=True)
    except DailyLimit as e:
        print("STOPPED:", e)
    kept = [line for line in (ABSENT.read_text(encoding="utf-8").splitlines() if ABSENT.exists() else [])
            if line.strip() and json.loads(line)["ticker"] not in tickers]          # other companies stay as they were
    ABSENT.write_text("".join(x + "\n" for x in kept) + "".join(json.dumps(x) + "\n" for x in absent_lines),
                      encoding="utf-8")
    log["after"] = {t: sum(len(v) for v in unbound(g, t).values()) for t in tickers}
    log["summary"] = {"unbound_before": sum(log["before"].values()), "unbound_after": sum(log["after"].values()),
                      "admitted": sum(1 for p in log["proposals"] if p["state"] == "active"),
                      "held": sum(1 for p in log["proposals"] if p["state"] != "active"),
                      "absent_recorded": len(absent_lines)}
    (ROOT / "results" / "coverage_pass.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps(log["summary"]))


if __name__ == "__main__":
    main()
