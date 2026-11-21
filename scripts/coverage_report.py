"""Dev metric: for every downloaded filing, how much of each primary statement
sits inside a recovered, passing formula. This number is what the parser is tuned against.

Usage: python scripts/coverage_report.py [DOC_ID ...] [--detail]
"""
import csv
import pathlib
import sys

from tieout import pdfio
from tieout.calc import coverage, discover, explain
from tieout.statements import find_statements

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    detail = "--detail" in sys.argv
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", newline="", encoding="utf-8")))
    docs = [r["doc_id"] for r in rows if (pdfio.PDF_DIR / f"{r['doc_id']}.pdf").exists()]
    if args:
        docs = [d for d in docs if d in args or d.split("_")[0] in args]
    pdfio.extract_all(docs)
    tot_hit = tot_all = 0
    for doc in docs:
        pages = pdfio.load_pages(doc)
        found = find_statements(doc, pages)
        best = {k: (None, st.table, st.checks, st.verified, st.cells) for k, st in found.items()}
        parts = []
        for kind in ("IS", "BS", "CF"):
            if kind in best:
                _, t, checks, hit, total = best[kind]
                dated = sum(1 for c in t.columns if c.year)
                parts.append(f"{kind} p{t.page:<3d} {hit:3d}/{total:<3d} cols {dated}/{t.ncols}")
                tot_hit += hit
                tot_all += total
            else:
                parts.append(f"{kind} MISSING            ")
        print(f"{doc:20s} " + " | ".join(parts), flush=True)
        if detail:
            for kind in ("IS", "BS", "CF"):
                if kind in best:
                    _, t, checks, hit, total = best[kind]
                    print(f"  -- {kind} page {t.page}: {t.title[:70]} scale={t.scale:g}")
                    print("     cols:", [(c.header[:28], c.end_date, c.months) for c in t.columns])
                    print(explain(t, checks))
                    ok_rows = {r for ch in checks if ch.status != "fail" for r in ch.rows}
                    print("     loose:", [r.label[:34] for r in t.rows if r.idx not in ok_rows])
    if tot_all:
        print(f"\nTOTAL verified cells {tot_hit}/{tot_all} = {tot_hit / tot_all:.1%} over {len(docs)} filings")


if __name__ == "__main__":
    main()
