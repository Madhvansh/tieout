"""Run the text-fact rules over every manifest filing with a word cache.

Writes data/textfacts.json (a list of fact dicts) and prints:
  * coverage per attribute over the 10-K filings,
  * per filing, the attributes that were not found,
  * self-checks: fiscal_year_end vs manifest period_end (10-Ks, within 20 days),
    and the sanity bounds the rules already enforce.

usage: python scripts/textfacts_report.py [--show TICKER[,TICKER...]]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout import pdfio  # noqa: E402
from tieout.textfacts import extract_text_facts  # noqa: E402

ATTRS = ["auditor", "auditor_since", "auditor_location", "state_of_incorporation",
         "fiscal_year_end", "hq_city", "hq_state", "exchange", "trading_symbol", "employees",
         "shares_outstanding", "public_float_usd", "ceo", "cfo"]
TENK_ONLY = {"auditor", "auditor_since", "auditor_location", "fiscal_year_end", "employees",
             "public_float_usd"}


def load_manifest() -> list:
    with open(ROOT / "data" / "manifest.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if pdfio.cache_path(r["doc_id"]).exists()]


def self_checks(row: dict, facts: list) -> list:
    msgs = []
    for f in facts:
        if f.attr == "fiscal_year_end" and row["form"] == "10-K" and row.get("period_end"):
            d = abs((dt.date.fromisoformat(f.value) - dt.date.fromisoformat(row["period_end"])).days)
            if d > 20:
                msgs.append(f"fiscal_year_end {f.value} is {d} days from manifest {row['period_end']}")
        if f.attr == "employees" and not 1_000 <= int(f.value) <= 3_000_000:
            msgs.append(f"employees {f.value} out of range")
        if f.attr == "auditor_since" and not 1850 <= int(f.value) <= 2026:
            msgs.append(f"auditor_since {f.value} out of range")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default="", help="print every fact for these tickers")
    args = ap.parse_args()
    show = {t.strip().upper() for t in args.show.split(",") if t.strip()}

    rows = load_manifest()
    all_facts, by_doc, checks = [], {}, []
    t0 = time.time()
    for row in rows:
        facts = extract_text_facts(row["doc_id"], pdfio.load_pages(row["doc_id"]))
        by_doc[row["doc_id"]] = facts
        all_facts += facts
        checks += [f"{row['doc_id']}: {m}" for m in self_checks(row, facts)]
        if row["ticker"].upper() in show:
            print(f"\n== {row['doc_id']}")
            for f in facts:
                print(f"  {f.attr:22s} {f.value:28s} p{f.page:<4d} {f.confidence:4s} | {f.quote[:150]}")

    out = ROOT / "data" / "textfacts.json"
    out.write_text(json.dumps([asdict(f) for f in all_facts], indent=1, ensure_ascii=False), encoding="utf-8")

    tenk = [r for r in rows if r["form"] == "10-K"]
    print(f"\n{len(all_facts)} facts from {len(rows)} filings in {time.time() - t0:.0f}s -> {out.relative_to(ROOT)}")
    print(f"\nCoverage over {len(tenk)} 10-K filings (high-confidence in brackets)")
    for a in ATTRS:
        got = [d for d in (r["doc_id"] for r in tenk) if any(f.attr == a for f in by_doc[d])]
        hi = [d for d in got if any(f.attr == a and f.confidence == "high" for f in by_doc[d])]
        print(f"  {a:24s} {len(got):3d} / {len(tenk)}   [{len(hi)}]")

    print("\nMissing attributes per filing")
    for r in rows:
        have = {f.attr for f in by_doc[r["doc_id"]]}
        want = [a for a in ATTRS if r["form"] == "10-K" or a not in TENK_ONLY]
        miss = [a for a in want if a not in have]
        print(f"  {r['doc_id']:24s} {r['form']:5s} " + (", ".join(miss) if miss else "-"))

    print("\nSelf-checks")
    for m in checks or ["all passed"]:
        print("  " + m)


if __name__ == "__main__":
    main()
