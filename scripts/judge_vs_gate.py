"""Who should decide whether a proposed binding may serve answers: the proposer, a model judge, or structure?

Every binding proposal on record (the rows a model proposed for a concept) is scored against the SEC's
own figures, as in scripts/gate_precision.py. Three admission rules are compared on the same proposals:
  self         every proposal is admitted (the proposer grades itself);
  model judge  a second, independent model reads the row's label, section and role (never a value)
               and answers whether the row reports the concept for the whole company;
  structure    the gate in tieout/living.py: unit, a passing formula, the concept's structural rule,
               the row's role elsewhere, and the vetoes.
For each: precision of what it admits, and how many right proposals it keeps. The SEC data is the key
here only. The judge's calls are cached like every other model call.

Usage: python scripts/judge_vs_gate.py [--judge ollama:phi4-mini]   ->  results/judge_vs_gate.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

from tieout.concepts import concept_defs
from tieout.graph import Graph
from tieout.living import SAYS, row_table
from tieout.llm import ask_json

ROOT = pathlib.Path(__file__).resolve().parents[1]
JUDGE = """You check one proposed mapping from a financial concept to a row of a company's financial statement.
You see the row's printed label, the section heading above it and its role in the statement's arithmetic.
You never see values. Answer whether this row reports exactly the concept for the whole company (not a
part of it, not a broader line that includes other items, not a gross amount when the concept is net).
Answer as JSON: {"reports": true or false, "reason": "<one short sentence>"}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="ollama:phi4-mini")
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp", ROOT / "scripts" / "gate_precision.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    g = Graph.load()
    key = gp.sec_values(g)
    proposals = []
    for path, admitted in ((ROOT / "semantics" / "learned.jsonl", True), (ROOT / "semantics" / "review_queue.jsonl", False)):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("concept") and e.get("row") is not None and e.get("doc"):
                        proposals.append((e, admitted and e.get("state") == "active"))
    rows_out = []
    for e, gate_admits in proposals:
        kind = e.get("kind") or concept_defs()[e["concept"]].statement
        right = gp.verdict(g, key, e["doc"], kind, e["row"], e["concept"])
        if right is None:
            continue
        d = row_table(g, e["doc"], kind).get(e["row"])
        if d is None:
            continue
        user = (f"Concept: {e['concept']}: {SAYS.get(e['concept'], '')}\nStatement: {kind}\n"
                f"Row label: {d['label'][:120]}\nSection: {d['section'][:80]}\nRole: {d['role']}")
        try:
            ans = ask_json(JUDGE, user, model=a.judge, tag="judge")
            judge_admits = bool(ans.get("reports")) if isinstance(ans, dict) else False
        except Exception:  # noqa: BLE001 - an unanswered judgement counts as not admitted
            judge_admits = False
        rows_out.append({"ticker": e["ticker"], "concept": e["concept"], "label": d["label"][:80], "right": right,
                         "gate": gate_admits, "judge": judge_admits})

    def score(rule: str) -> dict:
        kept = [r for r in rows_out if rule == "self" or r[rule]]
        right = sum(1 for r in kept if r["right"])
        return {"admitted": len(kept), "right": right, "precision": round(right / len(kept), 3) if kept else None,
                "ci95": gp.wilson(right, len(kept)), "right_kept_of_all_right": f"{right}/{sum(1 for r in rows_out if r['right'])}"}

    out = {"judge_model": a.judge, "proposals_keyed": len(rows_out),
           "rules": {"self": score("self"), "model_judge": score("judge"), "structure": score("gate")},
           "both_structure_and_judge": score("gate") if False else None, "rows": rows_out}
    both = [r for r in rows_out if r["gate"] and r["judge"]]
    out["both_structure_and_judge"] = {"admitted": len(both), "right": sum(r["right"] for r in both)}
    (ROOT / "results" / "judge_vs_gate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
