"""Fault-injection study over every parsed primary statement. Writes results/faults.json.

The first session's study is kept in results/faults_v1.json. This run repeats it on the current
graph and adds two outside evidence channels (sentence figures that pin the scale of a table, and
last year's column of a later filing that must equal the earlier filing): see tieout/eval/faults.py.
"""
import json
import logging
import pathlib

from tieout import pdfio
from tieout.eval.faults import MIN_SIGNIFICANT, Case, Evidence, chance_pins, study
from tieout.graph import Graph, load_manifest
from tieout.mentions import load_mentions
from tieout.statements import find_statements, readable

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_cases() -> dict:
    by_kind = {"IS": [], "BS": [], "CF": []}
    for m in load_manifest():
        if not pdfio.cache_path(m["doc_id"]).exists():
            continue
        pages = pdfio.load_pages(m["doc_id"])
        if not readable(pages):
            continue
        for kind, st in find_statements(m["doc_id"], pages).items():
            by_kind[kind].append(Case(m["doc_id"], kind, st.table))
    return by_kind


def pct(x):
    return f"{100 * x:5.1f}" if x is not None else "  n/a"


def main():
    by_kind = parse_cases()
    cases = [c for cs in by_kind.values() for c in cs]
    print(len(cases), "statements")
    ev = Evidence(Graph.load(), load_mentions())          # graph and mentions are read once
    res = {"all": study(cases, evidence=ev)}
    for kind, cs in by_kind.items():
        res[kind] = study(cs, evidence=ev)
    real_pins = [ev.pins(c) for c in cases]
    res["_notes"] = {"tables": len(cases), "tables_with_a_pin": sum(1 for p in real_pins if p),
                     "pins": sum(len(p) for p in real_pins), "min_significant_digits": MIN_SIGNIFICANT,
                     "chance_control": chance_pins(cases, ev.figures),
                     "cells_with_a_carried_partner": sum(1 for v in ev.partners.values() if v)}
    out = ROOT / "results" / "faults.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["_notes"])
    print(f"{'fault':13s} {'n':>5s} {'before':>7s} {'after':>6s} | pin: cover  catch(avail) | carry: cover  catch(avail)")
    for f, d in res["all"].items():
        p, c = d["pin"], d["carry"]
        print(f"{f:13s} {d['injected']:5d} {pct(d['catch_rate'])}% {pct(d['catch_rate_after'])}% | "
              f"{pct(p['coverage'])}% {pct(p['catch_rate_of_available'])}% | {pct(c['coverage'])}% {pct(c['catch_rate_of_available'])}%")


if __name__ == "__main__":
    main()
