"""The four-eyes rule, after the fact, for the concept bindings already in a learned store.

Every binding in the store was proposed by ONE model (living.learn while answering, or the coverage
pass) and admitted by the structural gate. Since 23 Nov an answer that rests on such a binding is
FLAGGED unless a second model made the same binding (execute.trust). This pass asks a second model
the same question the proposer was asked (row labels, sections and roles; never a value) and records
what it said:

    it names the stored row                       picked = "two readers agree"   (WITNESSED from now on)
    it names another row, the arbiter the stored  picked = "arbiter and one reader agree"
    it and the arbiter make the same other choice state = "disputed": two of three models chose another
      (another row, or no row at all)             row or none, so the binding is not applied (a person
                                                  can approve it in semantics/review_decisions.yaml)
    otherwise                                     picked = "one reader only" or "arbiter alone":
                                                  applied, and every answer on it is FLAGGED

A binding's row is never changed here.

  python scripts/witness_bindings.py IN.jsonl OUT.jsonl --checker gemini:gemini-3.1-flash-lite \
      [--arbiter gemini:gemini-3.6-flash,gemini:gemini-3.5-flash] [--max-arbiter 40] [--offline]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.concepts import audit_bindings, norm_label  # noqa: E402
from tieout.graph import Graph  # noqa: E402
from tieout.living import SYSTEM, binding_prompt, latest_doc  # noqa: E402
from tieout.llm import DailyLimit, ask_json  # noqa: E402
from tieout.witness import AGREE, ARBITER_ALONE, ARBITER_AND_READER, ONE_READER  # noqa: E402

KINDS = ("IS", "BS", "CF")


def ask_row(user: str, model: str, offline: bool):
    try:
        ans = ask_json(SYSTEM, user, model=model, tag="bind", offline=offline)
    except DailyLimit:
        raise
    except Exception as e:  # noqa: BLE001 - a failed call is recorded as no answer
        return None, f"{type(e).__name__}"
    row = ans.get("row") if isinstance(ans, dict) else None
    if isinstance(row, str) and row.strip().isdigit():
        row = int(row)
    return (row if isinstance(row, int) and not isinstance(row, bool) else None), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--checker", default="gemini:gemini-3.1-flash-lite")
    ap.add_argument("--arbiter", default="")
    ap.add_argument("--max-arbiter", type=int, default=40)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    g = Graph.load()
    audit_bindings(g)                                  # the seed bindings only: the state a proposer saw first
    arbiters = [m for m in a.arbiter.split(",") if m]
    budget = a.max_arbiter
    lines = pathlib.Path(a.inp).read_text(encoding="utf-8").splitlines()
    out, tally = [], {}
    for line in lines:
        if not line.strip():
            continue
        e = json.loads(line)
        if (e.get("kind") not in KINDS or e.get("state") != "active" or not e.get("label")
                or e.get("picked") in (AGREE, ARBITER_AND_READER)):
            out.append(json.dumps(e))
            continue
        ticker, concept = e["ticker"], e["concept"]
        doc = e.get("doc") if e.get("doc") in g.filings else latest_doc(g, ticker)
        user, rows = binding_prompt(g, ticker, concept, doc)
        stored = {r for r, d in rows.items() if norm_label(d["label"]) == norm_label(e["label"])}
        if not stored:
            e["picked"], e["witness_note"] = ONE_READER, "the stored label is not printed in this filing's statement"
        else:
            checker = a.checker if a.checker != e.get("proposer") else (arbiters[0] if arbiters else a.checker)
            r2, err = ask_row(user, checker, a.offline)
            seen = [{"model": checker, "row_label": rows[r2]["label"] if r2 in rows else None, "error": err}]
            if r2 in stored:
                e["picked"] = AGREE
            else:
                e["picked"] = ONE_READER
                for m in arbiters:
                    if budget <= 0 or m == checker:
                        continue
                    budget -= 1
                    try:
                        r3, err3 = ask_row(user, m, a.offline)
                    except DailyLimit:
                        continue
                    seen.append({"model": m, "row_label": rows[r3]["label"] if r3 in rows else None, "error": err3})
                    if r3 in stored:
                        e["picked"] = ARBITER_AND_READER
                    elif not err and not err3 and (r3 == r2 or (r3 is None and r2 is None)):
                        # two of three models made the same other choice (another row, or no row at all)
                        # against the one model that proposed this binding: it is not applied; a person
                        # can approve it through semantics/review_decisions.yaml
                        e["state"], e["picked"] = "disputed", ARBITER_ALONE
                        e["disputed"] = ("the second model and the arbiter both found "
                                         + ("no row that reports the concept" if r3 is None
                                            else f"another row ('{rows[r3]['label'][:50]}')"))
                    elif r3 is not None:
                        e["picked"] = ARBITER_ALONE
                    break
            e["witnesses"] = seen
        e["witness_pass"] = "scripts/witness_bindings.py (23 Nov)"
        tally[e["picked"]] = tally.get(e["picked"], 0) + 1
        out.append(json.dumps(e))
    pathlib.Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("witness pass:", tally, f"(arbiter calls left {budget})")


if __name__ == "__main__":
    main()
