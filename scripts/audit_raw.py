"""Audit the raw table layer and the sentence amounts against the SEC's XBRL data.

Runs audits 3 and 4 of tieout/eval/xbrl_audit.py over graph/raw.json.gz and
graph/mentions.json.gz, writes results/xbrl_audit_raw.json and prints the
summary, the tables read at the wrong scale and the columns given the wrong
period.

    python scripts/audit_raw.py [--raw FILE] [--mentions FILE] [--show N]
"""
import argparse
import json
import pathlib

from tieout.eval.xbrl_audit import audit_raw
from tieout.graph import Graph
from tieout.mentions import load_mentions
from tieout.rawlayer import load_raw

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "xbrl_audit_raw.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=40, help="errors of each kind to print")
    ap.add_argument("--raw", help="raw layer file (default graph/raw.json.gz)")
    ap.add_argument("--mentions", help="mentions file (default graph/mentions.json.gz)")
    ap.add_argument("--out", help="result file (default results/xbrl_audit_raw.json)")
    args = ap.parse_args()
    raw = load_raw(pathlib.Path(args.raw)) if args.raw else load_raw()
    mentions = load_mentions(pathlib.Path(args.mentions)) if args.mentions else load_mentions()
    res = audit_raw(Graph.load(), raw, mentions)
    out = pathlib.Path(args.out) if args.out else OUT
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    n = res["cells_tested"]
    print(f"raw cells tested {n}: " + ", ".join(f"{k} {v} ({v / n:.1%})" for k, v in sorted(res["cells"].items())))
    print(f"value and end date confirmed: {res['value_and_date_confirmed_share']:.1%}")
    for unit, c in sorted(res["by_unit"].items()):
        t = sum(c.values())
        print(f"   {unit:10s} {t:6d}  confirmed {c.get('confirmed', 0) / t:.1%}  as balance or flow the other way "
              f"{c.get('kind_differs', 0)}  other length {c.get('length_differs', 0)}")
    print(f"tables tested {res['tables_tested']}, with a confirmed cell {res['tables_with_a_confirmed_cell']}")
    print(f"tables at the wrong scale: {len(res['scale_errors'])}")
    for e in res["scale_errors"][:args.show]:
        print(f"   x{e['factor']:g}  {e['doc']:18s} p{e['page']:<4} {e['cells_at_factor']}/{e['tested']} cells  {e['title']}")
    print(f"columns tested {res['columns_tested']}, with the wrong period: {len(res['period_errors'])}")
    for e in res["period_errors"][:args.show]:
        print(f"   {e['doc']:18s} p{e['page']:<4} col {e['column']} {e['pid']:12s} quarters off {e['quarters_off']} "
              f"other length {e['other_length']}/{e['tested']}  {e['title']}")
    m = res["mentions_tested"]
    if m:
        print(f"sentence amounts tested {m}: confirmed {res['mentions'].get('confirmed', 0)} "
              f"({res['mentions'].get('confirmed', 0) / m:.1%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
