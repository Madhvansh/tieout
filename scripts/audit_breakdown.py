"""Why are some readings not confirmed by the SEC feed? Break the unconfirmed ones down.

    wrong period   the same amount is in the feed for this company, but under another period:
                   the number was read correctly and its period was not (or the company prints
                   the same amount twice)
    not in feed    the amount is nowhere in the company's feed: a line tagged with a company
                   extension element or a dimension (both are left out of "companyfacts"), or
                   a misread number
Writes results/xbrl_audit_breakdown.json.
"""
import json
import pathlib
from collections import defaultdict

from tieout.eval.xbrl_audit import _index, _months, _near, _period_months, fetch_companyfacts
from tieout.graph import Graph
from tieout.living import load_learned

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    g = Graph.load()
    load_learned(g)
    scale_of = {tid: t["scale"] for tid, t in g.tables.items()}
    out = defaultdict(int)
    by_kind = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    for ticker, comp in sorted(g.companies.items()):
        try:
            by_end, _ = _index(fetch_companyfacts(comp["cik"]))
        except Exception:  # noqa: BLE001
            continue
        all_vals = defaultdict(set)
        for end, rows in by_end.items():
            for s, e, v, tag in rows:
                all_vals[abs(v)].add(end)
        for r in g.readings.values():
            if r.ticker != ticker or r.unit != "USD" or r.value == 0:
                continue
            cands = _near(by_end, r.period_end)
            if not cands:
                continue
            want_m = _period_months(r.period)
            half = 0.5 * scale_of.get(r.table, 1e6)
            hit = any(abs(abs(v) - abs(r.value)) <= half and
                      (want_m is None or s is None or abs(_months(s, e) - want_m) <= 0.6) for s, e, v, _ in cands)
            if hit:
                cat = "confirmed"
            elif any(abs(av - abs(r.value)) <= half for av in all_vals):
                cat = "wrong_period_or_duration"
            else:
                cat = "not_in_feed"
            out[cat] += 1
            by_kind[r.kind][cat] += 1
            if cat != "confirmed" and len(examples[cat]) < 40:
                examples[cat].append({"doc": r.doc, "page": r.page, "label": r.label[:70], "period": r.period,
                                      "printed": r.text, "status": r.status})
    total = sum(out.values())
    res = {"total": total, "counts": dict(out), "by_statement": {k: dict(v) for k, v in by_kind.items()},
           "examples": examples}
    (ROOT / "results" / "xbrl_audit_breakdown.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"{k:26s} {v:6d}  {v / total:.1%}")
    for kind, d in by_kind.items():
        n = sum(d.values())
        print(f"  {kind}: " + ", ".join(f"{k} {v / n:.1%}" for k, v in sorted(d.items())))
    for cat in ("wrong_period_or_duration", "not_in_feed"):
        print(f"\n{cat} examples:")
        for e in examples[cat][:12]:
            print(f"   {e['doc']:20s} p{e['page']:<4d} {e['period']:12s} {e['printed']:>12s}  {e['label'][:56]}")


if __name__ == "__main__":
    main()
