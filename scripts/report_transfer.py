"""Transfer-aware report of the learned store (item 3, 23 Nov; the reporting of the LCG paper,
research/17 Part 2 idea 1): the same questions answered with the store and without it, split by
whether a question touches a stored (ticker, phrase) pair, with Wilson 95 percent intervals.

A question "touches" the store when a step of any reader's plan names a (company, item phrase) or a
(company, breakdown phrase) that the store holds a link for, or a (company, concept) it holds a learned
binding for. Phrases are compared as the linker compares them (link._phrase_key).

  python scripts/report_transfer.py results/heldout_v2.json results/heldout_abl_store.json \
      [--store semantics/learned.jsonl] [--scorer final|strict] [--out results/transfer_heldout.json]

Reads only result files and the store; never a question file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.eval.bench import wilson  # noqa: E402
from tieout.link import _phrase_key  # noqa: E402


def stored_pairs(path: pathlib.Path) -> set:
    pairs = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("state") not in ("active", "flagged"):
            continue
        if e.get("phrase"):
            pairs.add(("phrase", _phrase_key(e["ticker"], e["phrase"])))
        if e.get("concept") and e.get("kind") in ("IS", "BS", "CF"):
            pairs.add(("concept", e["ticker"], e["concept"]))
    return pairs


def plan_pairs(plan: dict) -> set:
    out = set()
    for s in (plan or {}).get("steps", []) or []:
        tickers = [s.get("company")] if isinstance(s.get("company"), str) else []
        if isinstance(s.get("over"), list):
            tickers += [t for t in s["over"] if isinstance(t, str)]
        tickers = [t for t in tickers if t and not t.startswith("$")]
        measure = s.get("by") if isinstance(s.get("by"), dict) else s
        for t in tickers:
            for key in ("item", "of"):
                if isinstance(measure.get(key), str):
                    out.add(("phrase", _phrase_key(t, measure[key])))
                if isinstance(s.get(key), str):
                    out.add(("phrase", _phrase_key(t, s[key])))
            if isinstance(measure.get("concept"), str):
                out.add(("concept", t, measure["concept"]))
    return out


def touches(row: dict, pairs: set) -> bool:
    plans = [row.get("plan")] + [x.get("plan") for x in row.get("readers", []) or []]
    return any(plan_pairs(p) & pairs for p in plans if isinstance(p, dict))


def ok(row: dict, scorer: str) -> bool:
    return bool(row.get(f"correct_{scorer}", row.get("correct")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("with_store")
    ap.add_argument("without_store")
    ap.add_argument("--store", default=str(ROOT / "semantics" / "learned.jsonl"))
    ap.add_argument("--scorer", default="final", choices=["final", "strict"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pairs = stored_pairs(pathlib.Path(a.store))
    rows_w = {r["id"]: r for r in json.loads(pathlib.Path(a.with_store).read_text(encoding="utf-8"))["rows"]}
    rows_o = {r["id"]: r for r in json.loads(pathlib.Path(a.without_store).read_text(encoding="utf-8"))["rows"]}
    ids = sorted(set(rows_w) & set(rows_o))
    report = {"scorer": a.scorer, "store_pairs": len(pairs), "n": len(ids), "groups": {}}
    for name, keep in (("all", lambda i: True),
                       ("touches the store", lambda i: touches(rows_w[i], pairs) or touches(rows_o[i], pairs)),
                       ("does not touch it", lambda i: not (touches(rows_w[i], pairs) or touches(rows_o[i], pairs)))):
        sub = [i for i in ids if keep(i)]
        kw = sum(ok(rows_w[i], a.scorer) for i in sub)
        ko = sum(ok(rows_o[i], a.scorer) for i in sub)
        report["groups"][name] = {"n": len(sub), "with_store": kw, "without_store": ko,
                                  "with_ci95": wilson(kw, len(sub)), "without_ci95": wilson(ko, len(sub)),
                                  "only_with": [i for i in sub if ok(rows_w[i], a.scorer) and not ok(rows_o[i], a.scorer)],
                                  "only_without": [i for i in sub if ok(rows_o[i], a.scorer) and not ok(rows_w[i], a.scorer)]}
    for name, g in report["groups"].items():
        print(f"{name:18s} n={g['n']:3d}  with store {g['with_store']:3d} {g['with_ci95']}  "
              f"without {g['without_store']:3d} {g['without_ci95']}  only with {len(g['only_with'])}, "
              f"only without {len(g['only_without'])}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
