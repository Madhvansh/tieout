"""Count what the shipped graph holds and write results/graph_stats.json.

Every number the README and the results PDF quote about the size of the graph
comes from this file. It reads graph/graph.json.gz, graph/raw.json.gz and
graph/mentions.json.gz and nothing else.

    python scripts/graph_stats.py
"""
from __future__ import annotations

import gzip
import json
import pathlib
from collections import Counter

from tieout.graph import Graph

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _gz(name: str):
    with gzip.open(ROOT / "graph" / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    g = Graph.load()
    readings = list(g.readings.values())
    checks = Counter(c.type for c in g.checks.values())
    check_status = Counter(f"{c.type}:{c.status}" for c in g.checks.values())
    raw = _gz("raw.json.gz")
    mentions = _gz("mentions.json.gz")
    decoded = sorted(json.loads((ROOT / "data" / "decoded_fonts.json").read_text(encoding="utf-8")))
    by_doc = {d: {k: [sum(1 for r in readings if r.doc == d and r.kind == k and r.status == "verified"),
                      sum(1 for r in readings if r.doc == d and r.kind == k)] for k in ("IS", "BS", "CF")}
              for d in decoded}
    out = {
        "filings": len(g.filings),
        "companies": len(g.companies),
        "readings": len(readings),
        "readings_verified": sum(1 for r in readings if r.status == "verified"),
        "checks": len(g.checks),
        "checks_by_type": dict(checks),
        "checks_by_type_and_status": dict(check_status),
        "facts": len(g.facts),
        "bindings": len(g.bindings),
        "bindings_active": sum(1 for b in g.bindings if b.state == "active"),
        "text_facts": len(g.textfacts),
        "raw_rows": len(raw),
        "raw_cells": sum(len(r["cells"]) for r in raw),
        "raw_cells_with_value": sum(1 for r in raw for c in r["cells"] if c.get("value") is not None),
        "mentions": len(mentions),
        "mention_numbers": sum(len(m["numbers"]) for m in mentions),
        "decoded_font_filings_verified_and_total_readings_by_statement": by_doc,
    }
    path = ROOT / "results" / "graph_stats.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
