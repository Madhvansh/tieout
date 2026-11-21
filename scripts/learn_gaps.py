"""Dev check of the living layer: try to learn bindings for some (ticker, concept) gaps.

Usage: python scripts/learn_gaps.py [--save] TICKER:Concept [TICKER:Concept ...]
"""
import sys

from tieout.graph import Graph
from tieout.living import learn, load_learned, save_entry


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv
    g = Graph.load()
    load_learned(g)
    for a in args:
        ticker, concept = a.split(":")
        e = learn(g, ticker, concept, question_id="dev", question=f"dev probe {a}")
        print(f"{a:28s} -> {e['state']:9s} row={e['row']} label={str(e['label'])[:48]!r}")
        print(f"{'':28s}    reason: {e.get('reason', '')[:110]}")
        for x in e["evidence"]:
            print(f"{'':28s}    + {x}")
        for x in e["objections"]:
            print(f"{'':28s}    - {x}")
        if save:
            save_entry(e)


if __name__ == "__main__":
    main()
