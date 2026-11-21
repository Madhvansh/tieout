"""Dev metric for the raw table layer: every numeric table outside the primary
statements, per filing.

Per document it prints: tables found, numeric cells, the share of money cells
inside a passing recovered formula (calc.discover + calc.coverage), and the
share of tables whose columns all got a period (a year or a date). Totals go
to data/alltables_summary.json.

Usage:
    python scripts/alltables_report.py [DOC_ID or TICKER ...]
    python scripts/alltables_report.py --show DOC PAGE      print the tables parsed on one page
    python scripts/alltables_report.py --include-primary    do not skip the statement pages
"""
import csv
import json
import logging
import pathlib
import sys
import time

from tieout import pdfio
from tieout.alltables import find_all_tables
from tieout.calc import coverage, discover, explain
from tieout.statements import find_statements
from tieout.tables import show

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def doc_ids(filters: list) -> list:
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", newline="", encoding="utf-8")))
    seen, docs = set(), []
    for r in rows:
        d = r["doc_id"]
        if d in seen or not pdfio.cache_path(d).exists():
            continue
        seen.add(d)
        if not filters or d in filters or d.split("_")[0] in filters:
            docs.append(d)
    return docs


def statement_pages(doc: str, pages: list) -> set:
    return {st.table.page for st in find_statements(doc, pages).values()}


def summarize(doc: str, tables: list) -> dict:
    cells = hit = money = dated = 0
    for t in tables:
        cells += sum(1 for r in t.rows for c in r.cells.values() if c.value is not None)
        ok, total = coverage(t, discover(t))
        hit += ok
        money += total
        dated += all(c.year or c.end_date for c in t.columns)
    n = len(tables)
    return {"doc": doc, "tables": n, "cells": cells, "money_cells": money, "verified_cells": hit,
            "verified_share": round(hit / money, 4) if money else 0.0,
            "dated_tables": dated, "dated_share": round(dated / n, 4) if n else 0.0,
            "scale_sources": {s: sum(1 for t in tables if t.scale_source == s)
                              for s in ("caption", "intro", "inherited")}}


def show_page(doc: str, page_no: int) -> None:
    pages = pdfio.load_pages(doc)
    # the whole document is read so the Item / Note context and a title carried
    # over from the previous page come out as in a full run
    for t in [t for t in find_all_tables(doc, pages) if t.page == page_no]:
        checks = discover(t)
        ok, total = coverage(t, checks)
        print(show(t))
        print(f"  context: {t.context!r}   scale_source: {t.scale_source}   verified {ok}/{total}")
        if t.pct_columns:
            print("  dropped percent columns:", [c.header for c in t.pct_columns])
        print(explain(t, checks))
        print()


def main() -> int:
    argv = sys.argv[1:]
    if "--show" in argv:
        i = argv.index("--show")
        show_page(argv[i + 1], int(argv[i + 2]))
        return 0
    include_primary = "--include-primary" in argv
    docs = doc_ids([a for a in argv if not a.startswith("--")])
    t0 = time.time()
    per_doc = []
    for doc in docs:
        pages = pdfio.load_pages(doc)
        skip = set() if include_primary else statement_pages(doc, pages)
        s = summarize(doc, find_all_tables(doc, pages, skip_pages=skip))
        s["skipped_pages"] = sorted(skip)
        per_doc.append(s)
        print(f"{doc:22s} tables {s['tables']:4d}  cells {s['cells']:6d}  "
              f"verified {s['verified_cells']:5d}/{s['money_cells']:<6d} {s['verified_share']:6.1%}  "
              f"dated {s['dated_share']:6.1%}", flush=True)
    tot = {k: sum(d[k] for d in per_doc) for k in ("tables", "cells", "money_cells", "verified_cells", "dated_tables")}
    tot["verified_share"] = round(tot["verified_cells"] / tot["money_cells"], 4) if tot["money_cells"] else 0.0
    tot["dated_share"] = round(tot["dated_tables"] / tot["tables"], 4) if tot["tables"] else 0.0
    tot["documents"] = len(per_doc)
    tot["seconds"] = round(time.time() - t0, 1)
    print(f"\nTOTAL {tot['documents']} filings, {tot['tables']} tables, {tot['cells']} cells, "
          f"verified {tot['verified_share']:.1%} of money cells, dated {tot['dated_share']:.1%} of tables, "
          f"{tot['seconds']}s")
    if not [a for a in argv if not a.startswith("--")]:
        out = ROOT / "data" / "alltables_summary.json"
        out.write_text(json.dumps({"total": tot, "documents": per_doc}, indent=1), encoding="utf-8")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
