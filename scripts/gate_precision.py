"""How good is the structural gate? Every binding proposal scored against the SEC's own figures.

The gate decides whether a row a model proposed for a concept may serve answers. This script takes
every proposal on record (semantics/learned.jsonl: admitted; semantics/review_queue.jsonl and
results/coverage_pass.json: held) and, where the SEC's XBRL data has the concept for that company
and year, checks whether the proposed row carries the SEC's value. It reports precision of what was
admitted against what was held, with Wilson intervals, and the binding audit's catch against the
same key. The SEC data is an answer key here only; the build never reads it.

Usage: python scripts/gate_precision.py   ->  results/gate_precision.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
from collections import defaultdict

from tieout.concepts import audit_bindings, concept_defs
from tieout.eval.xbrl_audit import TAG_MAP, _index, _months
from tieout.graph import Graph
from tieout.living import row_table

ROOT = pathlib.Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [0.0, 0.0]
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 3), round(min(1.0, mid + half), 3)]


def sec_values(g: Graph) -> dict:
    """{(ticker, concept, period end): [values]} from the SEC data for the concepts the audit maps."""
    out, defs = defaultdict(list), concept_defs()
    for t, comp in g.companies.items():
        p = ROOT / "data" / "sec" / f"CIK{int(comp['cik']):010d}.json"
        if not p.exists():
            continue
        _by_end, by_tag = _index(json.loads(p.read_text(encoding="utf-8")))
        for c, tags in TAG_MAP.items():
            for tag in tags:
                for (tg, end), recs in by_tag.items():
                    if tg == tag:
                        out[(t, c, end)] += [v for s, e, v, _ in recs
                                             if defs[c].nature == "stock" or s is None or abs(_months(s, e) - 12) <= 0.6]
    return out


def verdict(g: Graph, key: dict, doc: str, kind: str, row: int, concept: str):
    """True if the row carries the SEC value for its filing's year end, False if not, None if unkeyed."""
    if not doc or doc not in g.filings:
        return None
    f, t = g.filings[doc], g.filings[doc]["ticker"]
    rows = row_table(g, doc, kind)
    if row not in rows:
        return None
    end = dt.date.fromisoformat(f["period_end"])
    xs = next((v for (tk, c, e), v in key.items() if tk == t and c == concept and v
               and abs((dt.date.fromisoformat(e) - end).days) <= 3), None)
    if not xs:
        return None
    vals = [r.value for r in rows[row]["readings"] if r.period_end and
            abs((dt.date.fromisoformat(r.period_end) - end).days) <= 3 and r.value is not None]
    if not vals:
        return None
    return any(abs(abs(v) - abs(x)) <= max(500000, 0.001 * abs(x)) for v in vals for x in xs)


def main() -> None:
    g = Graph.load()
    key = sec_values(g)
    proposals = []
    for path, admitted in ((ROOT / "semantics" / "learned.jsonl", True), (ROOT / "semantics" / "review_queue.jsonl", False)):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("concept") and e.get("row") is not None and e.get("doc"):
                        proposals.append((e, admitted and e.get("state") == "active"))
    cov = ROOT / "results" / "coverage_pass.json"
    if cov.exists():
        seen = {(e["ticker"], e["concept"], e.get("label")) for e, _ in proposals}
        for p in json.loads(cov.read_text(encoding="utf-8")).get("proposals", []):
            if p["state"] != "active" and (p["ticker"], p["concept"], p["label"]) not in seen:
                doc = next((d for d, f in g.filings.items() if f["ticker"] == p["ticker"] and f["role"] == "latest"), None)
                kind = concept_defs()[p["concept"]].statement
                row = next((r for r, d in row_table(g, doc, kind).items() if d["label"] == p["label"]), None) if doc else None
                if row is not None:
                    proposals.append(({"ticker": p["ticker"], "concept": p["concept"], "doc": doc, "kind": kind,
                                       "row": row, "label": p["label"], "objections": p.get("objections", [])}, False))
    tally = {"admitted": [0, 0], "held": [0, 0]}
    wrong_admitted, right_held = [], []
    for e, admitted in proposals:
        kind = e.get("kind") or concept_defs()[e["concept"]].statement
        v = verdict(g, key, e["doc"], kind, e["row"], e["concept"])
        if v is None:
            continue
        side = "admitted" if admitted else "held"
        tally[side][0] += int(v)
        tally[side][1] += 1
        if admitted and not v:
            wrong_admitted.append(f"{e['ticker']} {e['concept']}: '{(e.get('label') or '')[:60]}'")
        if not admitted and v:
            right_held.append(f"{e['ticker']} {e['concept']}: '{(e.get('label') or '')[:60]}' held because "
                              f"{(e.get('objections') or [''])[0][:80]}")
    seed = [b for b in g.bindings if b.state == "active" and b.origin == "seed"]
    seed_keyed = [(b, verdict(g, key, b.doc, b.kind, b.row, b.concept)) for b in seed]
    seed_keyed = [(b, v) for b, v in seed_keyed if v is not None]
    dropped = audit_bindings(g)
    caught = [d for d in dropped if verdict(g, key, d["doc"], concept_defs()[d["concept"]].statement,
                                            next((b.row for b in g.bindings if b.doc == d["doc"] and b.label == d["label"]
                                                  and b.concept == d["concept"]), -1), d["concept"]) is False]
    out = {
        "gate": {side: {"right": k, "keyed": n, "precision": round(k / n, 3) if n else None, "ci95": wilson(k, n)}
                 for side, (k, n) in tally.items()},
        "seed_bindings": {"keyed": len(seed_keyed), "right": sum(1 for _, v in seed_keyed if v),
                          "precision": round(sum(1 for _, v in seed_keyed if v) / max(1, len(seed_keyed)), 4)},
        "audit": {"dropped": len(dropped), "dropped_and_wrong_by_key": len(caught),
                  "dropped_unkeyed_or_right": len(dropped) - len(caught)},
        "wrong_admitted": wrong_admitted, "right_but_held": right_held[:40],
    }
    (ROOT / "results" / "gate_precision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("gate", "seed_bindings", "audit")}, indent=1))
    print("admitted but wrong by the key:", *wrong_admitted, sep="\n  ")


if __name__ == "__main__":
    main()
