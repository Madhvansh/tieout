"""Count the changes between a company's first and latest filing, by cause, and write results/restatements.json.

For every statement column printed in two filings the line items are compared
(tieout/restate.py). A change is a reclassification (a line and its named
counterpart moved by equal amounts under an unchanged subtotal), a restatement
(the totals moved) or a presentation change (a line or block added or dropped).
The file keeps the counts by kind and one example of each, with the filing's own
explanatory sentence where one was found.

    python scripts/restate_stats.py
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

from tieout.graph import Graph
from tieout.restate import events

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    g = Graph.load()
    found = events(g, explain=True)
    by_kind = Counter(e["kind"] for e in found)
    examples = {}
    for e in found:
        examples.setdefault(e["kind"], e)
    out = {"events": len(found), "by_kind": dict(by_kind), "examples": examples,
           "all_events": [{k: e[k] for k in ("ticker", "statement", "period", "kind", "old_doc", "new_doc")}
                          | {"line": e["line"].get("label") if isinstance(e["line"], dict) else str(e["line"])}
                          for e in found]}
    path = ROOT / "results" / "restatements.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(len(found), dict(by_kind))
    for k, e in examples.items():
        print(k, e["ticker"], e["statement"], e["period"], e["line"])


if __name__ == "__main__":
    main()
