"""Command line: python -m tieout build | stats | concepts"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    from .graph import Graph, build, load_manifest
    if cmd == "build":
        from . import pdfio
        docs = [r["doc_id"] for r in load_manifest()]
        pdfio.extract_all(docs)
        g = build(sys.argv[2:] or None)
        g.save()
        print(json.dumps(g.stats(), indent=1))
        return 0
    g = Graph.load()
    if cmd == "ask":
        from .ask import answer
        from .living import load_learned
        load_learned(g)
        out = answer(" ".join(sys.argv[2:]), g, question_id="cli")
        print(json.dumps(out["plan"]))
        print("ANSWER:", out["answer"])
        for lf in out["result"].leaves[:8]:
            print(f"  source: {lf.doc} page {lf.page} | {lf.label[:50]} | printed {lf.printed} | {lf.status}")
        if len(out["result"].leaves) > 8:
            print(f"  ... {len(out['result'].leaves)} sources in all")
        for n in dict.fromkeys(out["result"].notes):
            print("  note:", n)
        for e in out["learned"]:
            print(f"  learned: {e['ticker']} {e['concept']} -> {e['state']} ({e.get('label')})")
        return 0
    if cmd == "stats":
        print(json.dumps(g.stats(), indent=1))
    elif cmd == "concepts":
        # coverage of each concept across the latest 10-K of every company
        latest = {d: f for d, f in g.filings.items() if f["role"] == "latest"}
        have = defaultdict(dict)
        for b in g.bindings:
            if b.doc in latest and b.state != "rejected":
                have[b.concept][latest[b.doc]["ticker"]] = b.state
        tickers = sorted({f["ticker"] for f in latest.values()})
        for concept, m in sorted(have.items(), key=lambda kv: -len(kv[1])):
            act = sum(1 for s in m.values() if s == "active")
            missing = [t for t in tickers if t not in m]
            print(f"{concept:26s} active {act:2d}  proposed {len(m) - act:2d}  missing {len(missing):2d} {' '.join(missing)[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
