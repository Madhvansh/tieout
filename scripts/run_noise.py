"""Noisy-graph experiment: graph quality against answer quality, with and without the verification gate.

For each fault rate, a share of the money cells of every statement is damaged before the checks
run (see tieout/eval/noise.py), the graph is rebuilt, and the benchmark is answered under two
policies with the same cached plans:

    serve all       answer from whatever is bound
    verified only   abstain unless every leaf of the proof is verified

Counts of correct, wrong and abstained answers per rate and policy go to results/noise.json.
No LLM call is made: plans and binding proposals are replayed from the cache.

Usage: python scripts/run_noise.py [--rates 0,0.01,0.03,0.05,0.1] [--seeds 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib

from tieout.ask import answer
from tieout.eval.bench import is_correct, load_benchmark
from tieout.execute import Executor
from tieout.graph import build
from tieout.items import load_items

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,0.01,0.03,0.05,0.1")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    questions = load_benchmark()
    rates = [float(x) for x in a.rates.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    out = []
    for rate in rates:
        for seed in (seeds if rate > 0 else seeds[:1]):
            g = build(verbose=False, learned=True, corrupt=(rate, seed) if rate > 0 else None)
            load_items(g)
            damaged = sum(f.get("corrupted_cells", 0) for f in g.filings.values())
            stats = g.stats()["reading_status"]
            for policy, strict in (("serve_all", False), ("verified_only", True)):
                ex = Executor(g, require_verified=strict)
                c = {"correct": 0, "wrong": 0, "abstain": 0}
                wrong_ids = []
                for q in questions:
                    try:
                        res = answer(q["question"], g, question_id=q["id"], offline=True, executor=ex,
                                     allow_learning=True, persist=False)["result"]
                    except RuntimeError:
                        c["abstain"] += 1
                        continue
                    ok = is_correct(q, {"kind": res.kind, "value": res.value, "extra": res.extra}, g.companies)
                    if ok:
                        c["correct"] += 1
                    elif res.kind == "abstain":
                        c["abstain"] += 1
                    else:
                        c["wrong"] += 1
                        wrong_ids.append(q["id"])
                row = {"rate": rate, "seed": seed, "policy": policy, "damaged_cells": damaged, **c,
                       "wrong_ids": wrong_ids, "verified": stats.get("verified", 0), "unverified": stats.get("unverified", 0),
                       "failed": stats.get("failed", 0)}
                out.append(row)
                print(f"rate {rate:5.2f} seed {seed} {policy:13s} correct {c['correct']:2d} wrong {c['wrong']:2d} "
                      f"abstain {c['abstain']:2d} | damaged {damaged} | verified {row['verified']} unverified {row['unverified']} "
                      f"failed {row['failed']}", flush=True)
            (ROOT / "results" / "noise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
