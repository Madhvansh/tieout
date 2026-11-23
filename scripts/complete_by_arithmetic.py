"""Residue that completes an equation: a table whose total fails to foot by exactly a number the page prints but the
graph does not hold (a ledger residue line) gets that line back, marked as recovered by arithmetic.

HoloClean's idea (Rekatsinas et al., VLDB 2017) in its smallest form: the footing equations a table already obeys
are the constraints, and a missing operand is the value that restores them. A candidate is accepted only when
ONE residue line of the same page completes EVERY failing column of the check (the gap of each column equals one
of the line's numbers, in order), the gap is at least 10 printed units, and no other residue line of the page
completes it as well. The control runs the same test with the residue of another page of the same filing
(page + 7): every match there is a coincidence, so it measures how often this rule would invent a row.

    python scripts/complete_by_arithmetic.py --residue graph/residue.json.gz [--docs DOC ...] [--out FILE]
"""
import argparse
import gzip
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tieout import pdfio  # noqa: E402
from tieout.alltables import find_all_tables  # noqa: E402
from tieout.calc import discover  # noqa: E402
from tieout.graph import Graph  # noqa: E402

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def tables_of(g, doc):
    """The tables build_raw reads for one filing (same arguments)."""
    f = g.filings[doc]
    pages = pdfio.load_pages(doc)
    skip = {p for t in g.tables.values() if t["doc"] == doc for p in t.get("pages", [t["page"]])}
    scales = Counter(t["scale"] for t in g.tables.values() if t["doc"] == doc and t.get("scale"))
    default = scales.most_common(1)[0][0] if scales else None
    own = defaultdict(set)
    for r in g.readings.values():
        if r.doc == doc:
            own[r.page].add(r.text)
    return list(find_all_tables(doc, pages, skip_pages=skip, default_scale=default, statement_cells=own)), \
        int(f["fiscal_year"].lstrip("FY"))


def completes(gaps: list, numbers: list) -> bool:
    """The line's printed numbers, in order, contain one number for each failing column's gap, in order."""
    vals = [abs(round(n["value"])) for n in numbers if not n.get("pct")]
    j = 0
    for gap in gaps:
        while j < len(vals) and vals[j] != gap:
            j += 1
        if j == len(vals):
            return False
        j += 1
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residue", default="graph/residue.json.gz")
    ap.add_argument("--docs", nargs="*", default=None)
    ap.add_argument("--out", default="results/complete_by_arithmetic.json")
    a = ap.parse_args()
    residue = json.load(gzip.open(a.residue, "rt", encoding="utf-8"))
    by_page = defaultdict(list)
    for r in residue:
        if r.get("zone", "body") == "body":
            by_page[(r["doc"], r["page"])].append(r)
    g = Graph.load()
    docs = a.docs or sorted(d for d, f in g.filings.items() if f.get("readable", True))
    stats, found, control = Counter(), [], []
    for doc in docs:
        tables, _ = tables_of(g, doc)
        pages = sorted({t.page for t in tables})
        for t in tables:
            for ch in discover(t):
                stats["checks"] += 1
                if ch.status != "fail":
                    continue
                stats["failing"] += 1
                gaps = [abs(ch.residuals[c]) for c in ch.failing_cols if ch.residuals[c] is not None]
                if not gaps or min(gaps) < 10:
                    stats["gap under 10"] += 1
                    continue
                total = t.rows[ch.total].label if ch.total < len(t.rows) else ""
                for tag, page, sink in (("real", t.page, found),
                                        ("control", pages[(pages.index(t.page) + 7) % len(pages)] if pages else t.page,
                                         control)):
                    hits = [r for r in by_page.get((doc, page), []) if completes(gaps, r["numbers"])]
                    if len(hits) == 1:
                        sink.append({"doc": doc, "page": t.page, "table_title": (t.title or "")[:80], "total": total,
                                     "gaps": gaps, "line": hits[0]["line"], "line_page": page,
                                     "class": hits[0]["class"], "read_by": "residue+arithmetic"})
                    elif len(hits) > 1:
                        stats[f"{tag} ambiguous"] += 1
        print(f"{doc:22s} checks {stats['checks']:6d} failing {stats['failing']:5d} recovered {len(found):4d} "
              f"control {len(control):3d}", flush=True)
    print(f"\nfooting checks {stats['checks']}, failing {stats['failing']} (gap under 10: {stats['gap under 10']}); "
          f"completed by one residue line of the same page: {len(found)}; by a line of another page (control, "
          f"coincidences): {len(control)}; ambiguous: real {stats['real ambiguous']}, control {stats['control ambiguous']}")
    Path(a.out).write_text(json.dumps({"stats": dict(stats), "recovered": found, "control": control}, indent=1),
                           encoding="utf-8")
    for x in found[:25]:
        print(f"  {x['doc']} p{x['page']} total {x['total'][:40]!r} gaps {x['gaps']} <- [{x['class']}] {x['line'][:90]}")


if __name__ == "__main__":
    main()
