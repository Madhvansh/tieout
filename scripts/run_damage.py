"""Damage curve: what the checks do to wrong numbers as extraction gets worse.

A growing share of the money cells of every parsed statement is damaged with a mix of fault types
(sign flip, digit error, one cell a thousand times too large, a row's value moved one period column,
two neighbouring rows trading values). The graph is built again from the damaged tables, so the
checks see only what a real pipeline would see. For every fact of the clean graph the best reading
is graded against the clean graph's own value, under three policies:

    off      every reading is served as if verified
    on       the graph as built: only a verified reading is trusted
    on+      the same, and a reading that a sentence figure or a carried prior-year column
             contradicts is not trusted either

Counts per fact: correct, silent_wrong (served as verified and wrong), flagged (wrong but unverified,
failing or contradicted), withheld (right but not served, or gone). Writes results/damage_curve.json.
The statements are parsed once and copied for every run; nothing is written to the graph files.

Usage: python scripts/run_damage.py [--rates 0,0.005,0.01,0.02,0.05,0.1,0.2] [--seeds 1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pathlib
import types

from tieout import graph as graph_mod
from tieout import pdfio
from tieout.eval.faults import channel_flags, carry_groups, classify, damage_statements, sentence_groups
from tieout.graph import build
from tieout.mentions import load_mentions
from tieout.statements import find_statements, readable

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = {"rate": 0.0, "seed": 0, "damaged": 0}
PARSED: dict = {}


def parse_once() -> None:
    for m in graph_mod.load_manifest():
        doc = m["doc_id"]
        if not pdfio.cache_path(doc).exists():
            continue
        pages = pdfio.load_pages(doc)
        PARSED[doc] = find_statements(doc, pages) if readable(pages) else None


def _load_pages(doc):
    return ["readable"] if PARSED.get(doc) is not None else []


def _find(doc, pages):
    statements = copy.deepcopy(PARSED[doc])
    if RUN["rate"] > 0:
        RUN["damaged"] += damage_statements(statements, doc, RUN["rate"], RUN["seed"])
    return statements


def build_with(rate: float, seed: int):
    RUN.update(rate=rate, seed=seed, damaged=0)
    g = build(verbose=False, learned=True)
    return g, RUN["damaged"]


def mean_sd(xs: list) -> tuple:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return round(m, 2), round(sd, 2)


def write_results(runs: list, rates: list, cells: int) -> None:
    summary = []
    for rate in rates:
        rs = [r for r in runs if r["rate"] == rate]
        if not rs:
            continue
        entry = {"rate": rate, "seeds": len(rs), "damaged_cells": mean_sd([r["damaged_cells"] for r in rs])}
        for policy in ("off", "on", "on_plus"):
            entry[policy] = {k: mean_sd([r[policy][k] for r in rs]) for k in rs[0][policy]}
        summary.append(entry)
    out = {"facts": runs[0]["off"]["facts"], "usd_readings": cells,
           "grading": "each fact's best reading against the clean graph's own value",
           "policies": {"off": "every reading served as verified", "on": "only verified readings trusted",
                        "on_plus": "verified and not contradicted by a sentence figure or a carried column"},
           "columns": "mean and sd over seeds", "summary": summary, "runs": runs}
    (ROOT / "results" / "damage_curve.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,0.005,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    mentions = load_mentions()                                  # read once
    parse_once()
    real_pdfio = graph_mod.pdfio
    graph_mod.pdfio = types.SimpleNamespace(cache_path=pdfio.cache_path, load_pages=_load_pages)
    graph_mod.readable = lambda pages: bool(pages)
    graph_mod.find_statements = _find
    try:
        clean, _ = build_with(0.0, 0)
        groups, carries = sentence_groups(clean, mentions), carry_groups(clean)
        cells = sum(1 for r in clean.readings.values() if r.unit == "USD")
        print(f"clean graph: {len(clean.facts)} facts, {len(groups)} sentence pins, {len(carries)} carries", flush=True)
        runs = []
        for rate in rates:
            for seed in (seeds if rate > 0 else seeds[:1]):
                g, damaged = build_with(rate, seed)
                flags = channel_flags(g, groups, carries)
                row = {"rate": rate, "seed": seed, "damaged_cells": damaged,
                       "off": classify(clean, g, checks=False),
                       "on": classify(clean, g),
                       "on_plus": classify(clean, g, flagged=flags)}
                runs.append(row)
                print(f"rate {rate:5.3f} seed {seed} damaged {damaged:5d} | silent wrong off {row['off']['silent_wrong']:4d} "
                      f"on {row['on']['silent_wrong']:4d} on+ {row['on_plus']['silent_wrong']:4d} | on: correct "
                      f"{row['on']['correct']} flagged {row['on']['flagged']} withheld {row['on']['withheld']}", flush=True)
                write_results(runs, rates, cells)          # a limit or a crash loses at most one run
    finally:
        graph_mod.pdfio = real_pdfio


if __name__ == "__main__":
    main()
