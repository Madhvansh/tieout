"""Build the raw layer (every non-primary table of every filing) and write graph/raw.json.gz."""
import logging
from collections import Counter

from tieout.graph import Graph
from tieout.rawlayer import build_raw, save_raw, search

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def main():
    g = Graph.load()
    rows = build_raw(g)
    save_raw(rows)
    cells = sum(len(r["cells"]) for r in rows)
    ok = sum(1 for r in rows for c in r["cells"] if c["ok"])
    print(f"rows {len(rows)}  cells {cells}  verified cells {ok} ({ok / cells:.1%})")
    print("roles", dict(Counter(r["role"] for r in rows)))
    print("rows that tie to a bound concept:", sum(1 for r in rows if r["ties_to"]),
          "| rows anchored through their total:", sum(1 for r in rows if r["anchor"]))
    for ticker, doc, phrase in (("AAPL", "AAPL_FY2026_10K", "iPhone net sales"),
                                ("GOOGL", "GOOGL_FY2025_10K", "Google Cloud revenue"),
                                ("WMT", "WMT_FY2026_10K", "Sam's Club net sales")):
        print(f"\nsearch {ticker}: {phrase!r}")
        for r in search(rows, ticker, doc, phrase, k=4):
            print(f"   p{r['page']:<3d} {r['label'][:34]:34s} | {r['title'][:46]:46s} | {r['role']:7s} "
                  f"ok={r['verified']} anchor={r['anchor']} ties={r['ties_to']} | {[c['text'] for c in r['cells'][:3]]}")


if __name__ == "__main__":
    main()
