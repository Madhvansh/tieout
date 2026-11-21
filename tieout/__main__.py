"""Command line.

    python -m tieout build [doc ids]     words, tables, formulas, graph
    python -m tieout stats               what the shipped graph holds
    python -m tieout concepts            which companies each concept is bound for
    python -m tieout ask "question"      answer one question, with plan, sources and notes
    python -m tieout language            the plan language: rules, vocabulary, worked examples
    python -m tieout show "question"     the source page with the answer's cells and formula drawn on it
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict

logging.getLogger("pdfminer").setLevel(logging.ERROR)
MAX_SOURCES_SHOWN = 10


def load_graph():
    """The shipped graph with everything learned so far applied on top."""
    from .graph import Graph
    from .link import load_links
    from .living import load_absent, load_learned
    g = Graph.load()
    load_learned(g)
    load_links(g)
    load_absent(g)
    return g


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "build":
        from . import pdfio
        from .graph import build, load_manifest
        pdfio.extract_all([r["doc_id"] for r in load_manifest()])
        g = build(sys.argv[2:] or None)
        g.save()
        print(json.dumps(g.stats(), indent=1))
    elif cmd == "ask":
        from .ask import answer
        out = answer(" ".join(sys.argv[2:]), load_graph(), question_id="cli")
        result = out["result"]
        print("PLAN:  ", json.dumps(out["plan"]))
        print("ANSWER:", out["answer"])
        print("TRUST: ", result.tier or "FLAGGED", *(f"\n  why: {x}" for x in result.tier_reasons))
        for lf in result.leaves[:MAX_SOURCES_SHOWN]:
            print(f"  source: {lf.doc} page {lf.page} | {lf.label[:60]} | printed {lf.printed} | {lf.status}")
        if len(result.leaves) > MAX_SOURCES_SHOWN:
            print(f"  ... {len(result.leaves)} sources in all")
        for note in result.notes:
            print("  note:", note)
        for quote in result.extra.get("diagnosis", {}).get("explanations", []):
            print(f"  the filing says ({quote['doc']} page {quote['page']}): {quote['sentence'][:300]}")
        for e in out["learned"]:
            print(f"  learned: {e['ticker']} {e.get('concept') or e.get('phrase')} -> {e['state']} ({e.get('label')})")
    elif cmd == "show":
        from .show import main as show_main
        return show_main(sys.argv[2:])
    elif cmd == "language":
        from .graph import Graph
        from .planner import system_prompt
        print(system_prompt(Graph.load()))
    elif cmd == "concepts":
        from .graph import Graph
        g = Graph.load()
        latest = {d: f for d, f in g.filings.items() if f["role"] == "latest"}
        have = defaultdict(dict)
        for b in g.bindings:
            if b.doc in latest and b.state != "rejected":
                have[b.concept][latest[b.doc]["ticker"]] = b.state
        tickers = sorted({f["ticker"] for f in latest.values()})
        for concept, m in sorted(have.items(), key=lambda kv: -len(kv[1])):
            active = sum(1 for s in m.values() if s == "active")
            missing = [t for t in tickers if t not in m]
            print(f"{concept:26s} active {active:2d}  proposed {len(m) - active:2d}  missing {len(missing):2d} {' '.join(missing)[:90]}")
    else:
        from .graph import Graph
        print(json.dumps(Graph.load().stats(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
