"""Check the brief's rule on result files: every proof leaf of every served answer is a node of the graph.

    python scripts/check_brief.py results/heldout_v2.json [more result files]

The brief: "Answers must come from the graph, not from plain vector search over text chunks."

For each file this prints the answers served, their proof leaves, how many of those leaves are nodes of
the graph (a reading of the built graph, a cell of a raw-layer row, a figure of a sentence record, at
the filing and page the leaf states), and every leaf that is not. It also counts the answers whose
lines a reader of the text pointed at: their figures are graph nodes like any other. The exit code is
1 when any leaf is outside the graph or an answer was served with no leaf at all (no source). Result files
written before 23 Nov carry no leaves and are
reported as "not recorded": rerun them to check them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tieout.eval.brief import from_graph  # noqa: E402
from tieout.graph import Graph  # noqa: E402


def check(path: pathlib.Path, g) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    served = [r for r in rows if not r.get("abstained")]
    if served and not any("leaves" in r for r in served):
        return {"file": path.name, "recorded": False}
    out = {"file": path.name, "recorded": True, "served": len(served), "leaves": 0, "nodes": 0, "outside": [],
           "no_leaf": [r["id"] for r in served if not r.get("leaves")],
           "pointed": sum(1 for r in served if "pointed at" in str(r.get("reading")) or "text reader" in str(r.get("reading")))}
    for r in served:
        found = from_graph(r.get("leaves") or [], g)
        out["leaves"] += found["leaves"]
        out["nodes"] += found["nodes"]
        out["outside"] += [{"question": r["id"], **leaf} for leaf in found["outside"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default="", help="also write the report to this file")
    a = ap.parse_args()
    g = Graph.load()
    reports = [check(pathlib.Path(f), g) for f in a.files]
    bad = 0
    for rep in reports:
        if not rep["recorded"]:
            print(f"{rep['file']}: proof leaves not recorded in this file (written before 23 Nov)")
            continue
        bad += len(rep["outside"]) + len(rep["no_leaf"])     # an answer with no leaf has no source
        print(f"{rep['file']}: {rep['served']} answers served, {rep['leaves']} proof leaves, {rep['nodes']} are graph nodes, "
              f"{len(rep['outside'])} outside the graph; {rep['pointed']} answers on lines a text reader pointed at; "
              f"{len(rep['no_leaf'])} served with no leaf {rep['no_leaf'][:8]}")
        for leaf in rep["outside"][:20]:
            print(f"   OUTSIDE {leaf['question']}: {leaf.get('id')} ({leaf.get('doc')} page {leaf.get('page')}): {leaf['why']}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print("PASS: every served answer has proof leaves and every proof leaf is a node of the graph" if not bad
          else f"FAIL: {bad} leaves are not nodes of the graph, or answers were served with no leaf")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
