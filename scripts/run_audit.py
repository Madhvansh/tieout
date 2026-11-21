"""Audit the graph against the SEC's XBRL data (never used to build it). Writes results/xbrl_audit.json.

Usage: python scripts/run_audit.py [--no-learned]
"""
import json
import pathlib
import sys

from tieout.eval.xbrl_audit import audit
from tieout.graph import Graph
from tieout.living import load_learned

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    g = Graph.load()
    learned = "--no-learned" not in sys.argv
    if learned:
        load_learned(g)
    res = audit(g)
    # which bindings came from the living layer
    origin = {}
    for b in g.bindings:
        if b.state == "active":
            origin[(g.filings[b.doc]["ticker"], b.concept)] = "learned" if b.origin.startswith("learned") else b.origin
    by_origin = {"seed": {"agree": 0, "disagree": 0}, "learned": {"agree": 0, "disagree": 0}, "human": {"agree": 0, "disagree": 0}}
    for d in res["disagreements"]:
        by_origin.setdefault(origin.get((d["ticker"], d["concept"]), "seed"), {"agree": 0, "disagree": 0})["disagree"] += 1
    res["disagreements_by_origin"] = {k: v["disagree"] for k, v in by_origin.items()}
    name = "xbrl_audit" + ("" if learned else "_seed_only")
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"readings checked against XBRL: {res['readings_checked']}  confirmed: {res['confirmed']} "
          f"({res['confirmed_share']:.1%})  period not in XBRL feed: {res['period_not_in_xbrl']}")
    for s, d in res["by_status"].items():
        n = d["confirmed"] + d["not_confirmed"]
        if n:
            print(f"   {s:10s} confirmed {d['confirmed']:5d} of {n:5d} = {d['confirmed'] / n:.1%}")
    print(f"concept bindings compared: {res['bindings_compared']}  agree: {res['bindings_agree']} "
          f"({res['bindings_agree_share']:.1%})  disagreements by origin: {res['disagreements_by_origin']}")
    for c, v in sorted(res["by_concept"].items(), key=lambda kv: -kv[1].get("disagree", 0))[:12]:
        print(f"   {c:26s} agree {v.get('agree', 0):4d} disagree {v.get('disagree', 0):3d} no xbrl value {v.get('no_xbrl_value', 0):3d}")
    errs = {t: v["error"] for t, v in res["per_company"].items() if "error" in v}
    if errs:
        print("fetch errors:", errs)


if __name__ == "__main__":
    main()
