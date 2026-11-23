"""The fresh rounds in one table, and the trust tiers pooled over every first run.

    python scripts/pooled_first_runs.py [--after h3]   # prints the tables, writes results/pooled_first_runs.json

A first run is the one run made on a round before any of its answers was read
(results/first_runs/round<N>.json). Each is rescored here under both scorers of tieout/eval/bench.py
("final" and "strict", the rules of 21 Nov), from the answer it stored, so that every round is judged by
the same rules whatever the scorer was on the night it ran. "After" is the same round on the frozen code
(results/<after>_r<N>.json, one run of every round on the frozen commit; `--after` names the prefix, h3 =
commit 5431e1c), when the round had become development material. No model call is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.eval.bench import is_correct, load_benchmark, wilson  # noqa: E402
from tieout.graph import Graph  # noqa: E402

RES = ROOT / "results"
ROUNDS = [3, 4, 5, 6, 7, 8, 9, 10, 11]
TIERS = ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED", "ABSTAIN")
DEVELOPMENT = [("round 1 development", "r1dev"), ("round 1", "r1"), ("round 2 development", "r2dev")]


def _parser():
    spec = importlib.util.spec_from_file_location("rescore_rendered", ROOT / "scripts" / "rescore_rendered.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_answer


def rows_of(name: str):
    path = RES / name
    return json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="first runs of every fresh round, pooled")
    ap.add_argument("--after", default="h3", help="prefix of the frozen commit's per-round files results/<after>_r<N>.json")
    after_prefix = ap.parse_args().after
    companies = Graph.load().companies
    parse = _parser()
    out: dict = {"rounds": [], "pooled_tiers": {}, "development": []}
    pooled = {t: [0, 0] for t in TIERS}                 # tier -> [right, n] under the final scorer, first runs
    pooled_strict = {t: [0, 0] for t in TIERS}
    total = [0, 0, 0]                                   # right final, right strict, n
    print(f"{'round':9s} {'n':>3s} {'first run':>10s} {'strict':>7s} {'plan road alone':>16s} {'after, frozen code':>19s}")
    for n in ROUNDS:
        first = rows_of(f"first_runs/round{n}.json")
        if first is None:
            continue
        bench = {q["id"]: q for q in load_benchmark(ROOT / "benchmark" / f"round{n}" / "dev.jsonl")}
        right = strict = alone = 0
        has_road = any(r.get("plan_road") for r in first)
        for r in first:
            q = bench.get(r["id"])
            if q is None:
                continue
            pred = r["pred"] if isinstance(r.get("pred"), dict) else parse(str(r.get("answer") or ""), companies)
            ok = {m: bool(is_correct(q, pred, companies, m)) for m in ("final", "strict")}
            right += ok["final"]
            strict += ok["strict"]
            alone += bool((r.get("plan_road") or {}).get("correct"))
            tier = r.get("tier") or "FLAGGED"
            if tier in pooled:
                pooled[tier][0] += ok["final"]
                pooled[tier][1] += 1
                pooled_strict[tier][0] += ok["strict"]
                pooled_strict[tier][1] += 1
        after = rows_of(f"{after_prefix}_r{n}.json")
        after_right = sum(bool(r.get("correct")) for r in after) if after else None
        total[0] += right
        total[1] += strict
        total[2] += len(first)
        out["rounds"].append({"round": n, "n": len(first), "first_run_final": right, "first_run_strict": strict,
                              "first_run_plan_road_alone": alone if has_road else None,
                              "after_frozen_code": after_right, "after_n": len(after) if after else None})
        print(f"round {n:<3d} {len(first):3d} {right:10d} {strict:7d} {(str(alone) if has_road else '-'):>16s} "
              f"{(str(after_right) + '/' + str(len(after))) if after else '-':>19s}")
    print(f"{'pooled':9s} {total[2]:3d} {total[0]:10d} {total[1]:7d}   = {total[0] / max(1, total[2]):.1%} (final), "
          f"{total[1] / max(1, total[2]):.1%} (strict)")
    out["pooled"] = {"n": total[2], "right_final": total[0], "right_strict": total[1],
                     "ci95_final": wilson(total[0], total[2]) if total[2] else None}
    print("\ntrust tiers, pooled over the first runs (final scorer; 95% Wilson interval):")
    for t in TIERS:
        k, m = pooled[t]
        if m:
            lo, hi = wilson(k, m)
            out["pooled_tiers"][t] = {"right": k, "n": m, "ci95": [lo, hi], "right_strict": pooled_strict[t][0]}
            print(f"  {t:13s} {k:3d} of {m:3d}   {lo:.0%} to {hi:.0%}   (strict scorer: {pooled_strict[t][0]})")
    print("\ndevelopment sets on the frozen code:")
    dev = [0, 0]
    out["after_files"] = f"results/{after_prefix}_r<N>.json"
    for label, short in DEVELOPMENT:
        name = f"{after_prefix}_{short}.json"
        rows = rows_of(name)
        if rows:
            k = sum(bool(r.get("correct")) for r in rows)
            dev[0] += k
            dev[1] += len(rows)
            out["development"].append({"set": label, "right": k, "n": len(rows), "file": name})
            print(f"  {label:22s} {k}/{len(rows)}")
    if dev[1] and total[2]:
        gap = dev[0] / dev[1] - total[0] / total[2]
        out["gap_points"] = round(100 * gap, 1)
        print(f"  development {dev[0]}/{dev[1]} = {dev[0] / dev[1]:.1%}; fresh first runs {total[0] / total[2]:.1%}; "
              f"gap {100 * gap:.1f} points")
    (RES / "pooled_first_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
