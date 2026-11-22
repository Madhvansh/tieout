"""Run the scale test and write results/scale_test.json (see tieout/eval/scale.py).

    python scripts/run_scale_test.py [--families fact,growth] [--companies AAPL,MSFT] [--out PATH]

Questions keyed by the SEC's XBRL data are executed as plans, with no model anywhere. Prints the outcomes by
question kind and the largest groups of wrong answers, and writes totals overall, by kind, company and
concept, what the test covers, and every wrong answer with its id and evidence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import Counter

from tieout.__main__ import load_graph
from tieout.eval import scale
from tieout.execute import Executor

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scale_test.json"

# (company, concept) pairs the SEC feed cannot key, found by checking wrong answers against the filing:
# the filer tags a sub-line with the element that other filers use for the whole amount.
KEY_DEFECTS = {
    ("JNJ", "ResearchAndDevelopment"): "the SEC element holds a different amount from the printed "
                                       "income-statement line (1,841 against 17,232 for fiscal 2024)",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scale test keyed by the SEC's XBRL data.")
    p.add_argument("--families", default=",".join(scale.FAMILIES), help="comma separated; default: all")
    p.add_argument("--companies", default="", help="tickers, comma separated; default: all")
    p.add_argument("--out", default=str(OUT))
    return p.parse_args(argv)


def print_table(summary: dict) -> None:
    names = ("n", "correct", "abstained", "unverified", "silent")
    print(f"{'':12s}" + "".join(f"{n:>11s}" for n in names) + f"{'accuracy':>10s}")
    for name, s in [*summary["by_family"].items(), ("overall", summary["overall"])]:
        print(f"{name:12s}" + "".join(f"{s[k]:>11d}" for k in ("n", *scale.CLASSES)) + f"{s['accuracy']:>10.1%}")


def print_wrong_groups(wrong: list) -> None:
    print(f"\nlargest groups of wrong answers ({len(wrong)} in all):")
    groups = Counter((r["family"], r["concept"]) for r in wrong)
    for (family, concept), n in groups.most_common(10):
        silent = sum(1 for r in wrong if (r["family"], r["concept"]) == (family, concept) and r["class"] == "wrong_silent")
        print(f"  {family:9s} {concept:26s} {n:4d}  silent {silent}")


def main(argv=None) -> int:
    args = parse_args(argv)
    start = time.time()
    families = tuple(f for f in args.families.split(",") if f)
    unknown = [f for f in families if f not in scale.FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; choose from {', '.join(scale.FAMILIES)}")
    g = load_graph()
    tickers = [t for t in args.companies.split(",") if t] or sorted(g.companies)
    missing = [t for t in tickers if t not in g.companies]
    if missing:
        raise SystemExit(f"unknown companies {missing}")
    keys = scale.load_keys(g, tickers, frozenset(KEY_DEFECTS))
    questions = scale.build_questions(g, keys, families)
    rows = scale.run(questions, Executor(g), g)
    wrong = scale.wrong_rows(rows)
    report = {
        "meta": {"n": len(rows), "families": list(families), "seconds": round(time.time() - start, 1),
                 "executor_errors": sum(1 for r in rows if "error" in r),
                 "correct_by_magnitude_only": sum(1 for r in rows if r.get("sign_differs")),
                 "restated_questions_whose_key_is_not_zero": sum(1 for q in questions if q.family == "restated" and q.key),
                 "key_defects_blocked": {f"{t}/{c}": why for (t, c), why in KEY_DEFECTS.items()},
                 "alternatives": scale.ALTERNATIVES, "key_elements": scale.KEY_TAGS},
        "coverage": scale.coverage(g, keys),
        "summary": scale.summarize(rows),
        "wrong": wrong,
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print_table(report["summary"])
    print_wrong_groups(wrong)
    print(f"\n{len(rows)} questions in {report['meta']['seconds']} s; executor errors "
          f"{report['meta']['executor_errors']}; written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
