"""How many printed facts the development questions rest on are addressable nodes?

For every numeric supporting fact of the development questions (doc, page,
printed, value), look for a node of the same filing that carries that value in
full units with a compatible unit:

    statement  a reading of graph/graph.json.gz (the three primary statements)
    raw        a cell of graph/raw.json.gz (every other table)
    mention    a number of graph/mentions.json.gz (a sentence of running text)

Prints recall overall and by the kind of node that carries the fact, then every
miss. Text-valued facts (names, places) are out of scope.

    python scripts/addressability.py [--same-page] [--show-near]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUESTION_FILES = [ROOT / "benchmark" / "round2" / "dev.jsonl", ROOT / "benchmark" / "candidates.jsonl"]
KINDS = ("statement", "raw", "mention")
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
COUNT_WORDS = re.compile(r"employ|staff|member|store|warehouse|associate|deliver|vehicle|unit|location|"
                         r"supercenter|club|headcount|customer|subscriber", re.I)


def _num(value):
    """A fact value as a number, or None for a text fact ("Oregon")."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*\$?\s*([\d,]*\.?\d+)\s*(thousand|million|billion|trillion)?\s*", str(value), re.I)
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * SCALE.get((m.group(2) or "").lower(), 1.0)


def expected_unit(sf: dict) -> str:
    scale, printed, label = str(sf.get("scale", "")).lower(), str(sf.get("printed", "")), str(sf.get("label", ""))
    if "per share" in scale:
        return "USD/share"
    if "%" in printed or "percent" in scale:
        return "percent"
    if "$" in printed or "dollar" in scale:
        return "USD"
    if re.search(r"\bshares\b|stock outstanding", label, re.I):
        return "shares"
    if scale.startswith("units") or isinstance(sf.get("value"), str) or COUNT_WORDS.search(label) and "thousand" in scale:
        return "count"
    return "USD"


COMPATIBLE = {"USD": {"USD"}, "shares": {"shares"}, "count": {"count"}, "USD/share": {"USD/share", "ratio"},
              "percent": {"percent", "ratio"}}


def load_facts() -> list:
    out = []
    for path in QUESTION_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for sf in item.get("supporting_facts", []):
                v = _num(sf.get("value"))
                if v is None or not sf.get("doc"):
                    continue
                out.append({"qid": item["id"], "doc": sf["doc"], "page": sf.get("page"), "printed": sf.get("printed"),
                            "value": v, "unit": expected_unit(sf), "label": sf.get("label", "")})
    return out


def _gz(path: pathlib.Path):
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_nodes(raw_path=None, mentions_path=None) -> dict:
    """doc -> list of (kind, page, value, unit, where)."""
    nodes = defaultdict(list)
    blob = _gz(ROOT / "graph" / "graph.json.gz") or {}
    for r in blob.get("readings", []):
        nodes[r["doc"]].append(("statement", r["page"], r["value"], r["unit"], r["id"]))
    for row in _gz(pathlib.Path(raw_path or ROOT / "graph" / "raw.json.gz")) or []:
        if any(c.get("rid") for c in row["cells"]):
            continue                                   # primary-statement rows repeat the statement readings
        for c in row["cells"]:
            if c.get("value") is not None:
                nodes[row["doc"]].append(("raw", row["page"], c["value"], c.get("unit", row["unit"]), row["id"]))
    for m in _gz(pathlib.Path(mentions_path or ROOT / "graph" / "mentions.json.gz")) or []:
        for n in m["numbers"]:
            nodes[m["doc"]].append(("mention", m["page"], n["value"], n["unit"], m["id"]))
    return nodes


def same(a: float, b: float) -> bool:
    return abs(abs(a) - abs(b)) <= 1e-6 * abs(b) + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-page", action="store_true", help="also require the node to sit on the fact's page")
    ap.add_argument("--raw", help="raw layer file to measure (default graph/raw.json.gz)")
    ap.add_argument("--mentions", help="mentions file to measure (default graph/mentions.json.gz)")
    ap.add_argument("--show-near", action="store_true", help="for misses, show nodes with the value in another unit")
    args = ap.parse_args()
    facts, nodes = load_facts(), load_nodes(args.raw, args.mentions)
    by_kind, hit_any, misses = Counter(), 0, []
    anywhere, outside, outside_hit = Counter(), 0, 0
    for f in facts:
        kinds, wrong_unit = set(), []
        for kind, page, value, unit, where in nodes.get(f["doc"], []):
            if not isinstance(value, (int, float)):
                continue
            # a percentage may be stored as printed (14.6) or as a fraction (0.146)
            if not (same(value, f["value"]) or f["unit"] == "percent" and same(value, 100 * f["value"])):
                continue
            if args.same_page and page != f["page"]:
                continue
            if unit in COMPATIBLE[f["unit"]]:
                kinds.add(kind)
            else:
                wrong_unit.append(f"{kind}:{unit}:p{page}")
        for k in kinds:
            anywhere[k] += 1
        if "statement" not in kinds:
            outside += 1
            outside_hit += bool(kinds)
        if kinds:
            hit_any += 1
            by_kind[next(k for k in KINDS if k in kinds)] += 1
        else:
            misses.append((f, wrong_unit))
    n = len(facts)
    print(f"numeric supporting facts: {n}   addressable: {hit_any} ({hit_any / n:.1%})")
    print("first kind that carries the fact (statement > raw > mention):",
          ", ".join(f"{k} {by_kind[k]}" for k in KINDS))
    print("facts carried by each kind at all:", ", ".join(f"{k} {anywhere[k]}" for k in KINDS))
    print(f"facts outside the primary statements: {outside}   addressable: {outside_hit} "
          f"({outside_hit / max(1, outside):.1%})")
    print(f"\nmisses ({len(misses)}):")
    for f, wrong in misses:
        extra = f"   value found with unit {sorted(set(wrong))[:3]}" if wrong and args.show_near else ""
        print(f"  {f['qid']:4s} {f['doc']:22s} p{f['page']!s:4s} {str(f['printed'])[:28]!r:30s} "
              f"{f['value']:>20,.2f} {f['unit']}{extra}")


if __name__ == "__main__":
    main()
