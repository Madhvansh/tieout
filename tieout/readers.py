"""Two readers and an arbiter: no single model's reading of a question is trusted.

The graph can prove a number: it sits inside a formula that holds. It cannot prove that the
plan asked for the right number. A plan over the wrong set of companies, or the wrong year,
ties out just as well. That reading is the one thing a language model decides alone.

So the question is read twice, by two different models: a hosted model with a large free quota
(the maker) and the small fine-tuned local model (the checker). Each writes its own plan without
seeing the other's, and each plan runs on the graph. They are not independent: the local model was
trained on plans written with the maker family's help, and the word rules of ask.py rewrite both
plans alike (see witness.py), so they can share an error; agreement lowers the error rate, and the
measured rate is what the write-up reports.

    the two answers are the same       served: the reading has two witnesses
    they differ, or one has no answer  one call to the strongest free model (the arbiter),
                                       which plans from the question alone and sees neither
                                       plan. If it equals one reader's answer, that answer is
                                       served as witnessed. If all three differ, the answer
                                       whose numbers are best verified is served, flagged

Agreement is judged on what the plans compute (the same figure, the same names in the same
order), never on how the plans are written, and never against a gold answer.

The strong free models allow about twenty calls a day. Spending them only on disagreements
lets a thirty-question run fit one free key in one day, and every answer says which case it was.
"""
from __future__ import annotations

import os
from typing import Optional

from . import ask
from .execute import Result, render, same_name
from .link import start_question
from .llm import DailyLimit, impatient
from .witness import (AGREE, AGREE_OF_THREE, AGREE_ON_ORDER, ARBITER_ALONE, ARBITER_AND_READER,  # noqa: F401
                      ONE_READER, WITNESSED, Panel as Readers)

CORROBORATED = "CORROBORATED"                         # a tier: a FLAGGED answer that a second method repeats
READER_AND_PASSAGE = "one reader and the passage reader agree"
ARBITER_AND_PASSAGE = "the arbiter and the passage reader agree"
POINTED_NODE = "one node the text reader pointed at"  # three cases where a reader of the text named the
POINTED_LINES = "a plan on lines the text reader pointed at"          # lines and the graph gave the figures
LINES_OVERRULED = "the text reader's lines against the linker's"
# Two rules added after round 10's misses were read (23 Nov), each with a switch so its gain can be measured:
NODE_OVER_BELIEF = os.environ.get("TIEOUT_NODE_OVER_BELIEF", "1") == "1"       # a node at a named place outranks a decline
COMPLETE_OTHER_PLANS = os.environ.get("TIEOUT_COMPLETE_PLANS", "1") == "1"     # point at the gap of another planner's plan
# Session A, 23 Nov 04:05: the places the reader cited for the WHOLE question (passage.by_citing, no new call)
CITED_LINES = "a plan on lines the text reader cited for the whole question"
CITED_OVERRULED = "the lines the text reader cited against the linker's"
CITED_ON = os.environ.get("TIEOUT_CITED", "1") == "1"                          # an abstention, another plan's gap
# A row dispute: the served plan gives one figure on the linker's rows and the reader's own figure on the rows
# the reader cited. Scored on rounds 3 to 10 before it was switched on (dev_scripts/a/phase6/cited_eval.py, 284
# questions, 4 disputes): for a figure the cited rows were right 3 times of 3; for a ranking 0 of 1 (round 8 W18:
# a binding that repeats the reader's order with an absurd figure). So it overrules for a figure only.
CITED_OVERRULE = os.environ.get("TIEOUT_CITED_OVERRULE", "1") == "1"
CITED_OTHER = os.environ.get("TIEOUT_CITED_OTHER", "0") == "1"                 # a reading dispute (recorded; off)
# Lane J, 23 Nov: a FLAGGED answer the text reader disputes. When the reader's own answer can be computed from
# graph nodes (candidate B: one printed figure, a planner's plan on the places the reader cited, or the reader's
# own arithmetic evaluated on those nodes), one arbiter call judges which of the two DERIVATIONS answers the
# question as asked. It picks a derivation, never writes a figure. TIEOUT_JUDGE: "1" on (default), "record" asks
# and records the pick but never serves, "0" off.
JUDGED = "a judge chose the text reader's derivation, computed on graph nodes, over the served one"


def judge_mode() -> str:
    raw = os.environ.get("TIEOUT_JUDGE", "1").strip().lower()
    return "record" if raw == "record" else ("off" if raw in ("0", "off", "") else "on")


UNWITNESSED = "the planners read the question differently; this reading has no second witness"
FIGURES_DIFFER = "two readers rank these in the same order but do not compute every figure the same way"


def answer(question: str, g, readers: Readers, **kw) -> dict:
    """`ask.answer` under the four-eyes rule, then the second road when the first leaves doubt.
    Returns the served answer's dict with "reading": the case, and what each reader answered."""
    served = _plan_road(question, g, readers, **kw)
    if getattr(readers, "passage", ""):
        served = second_road(question, g, served, readers, kw)
    return served


def second_road(question: str, g, served: dict, readers: Readers, kw: dict) -> dict:
    """A reader of the text as witness and pointer (tieout/passage.py). It never supplies an answer:
    the brief asks that answers come from the graph, not from search over text chunks. This function
    handles an answer the plan road did not end VERIFIED or WITNESSED (those go to _disputed below).

      the planners have no answer, and no arbiter   -> the reader is asked where the answer is printed.
      and no two planners say the filings lack it      One printed figure: the graph node at the place it
                                                       names is the answer (passage.as_answer). Otherwise
                                                       each line a planner's plan is missing is pointed
                                                       at, linked, and the plan runs on the graph
                                                       (passage.by_pointing). Both FLAGGED
      the served reading has no second witness      -> that planner's answer
      (arbiter alone, one reader only) and the
      reader's figure is another planner's figure
      the reader repeats a FLAGGED answer           -> CORROBORATED: two different methods reached it
                                                       (right 28 of 30 on rounds 7 to 9, against 31 of 53
                                                       for a FLAGGED answer it does not repeat)
      the reader says something else                -> one more test: the served plan runs again on the
                                                       lines the reader points at
          that run gives the reader's own figure    -> the linker read a wrong line for a right plan:
                                                       the run on the pointed lines is served, FLAGGED
          anything else                             -> the served answer stays and names the dissent

    The reader's own answer never replaces a served one: simulated on saved results that lost more than
    it won (141 of 178 right against 144, dev_scripts/fable/two_roads_sim.py), and it would break the brief."""
    from . import passage
    res, reading = served["result"], served["reading"]
    reading["plan_road"] = {"answer": served["answer"], "tier": res.tier or "ABSTAIN", "kind": res.kind,
                            "value": res.value, "extra": res.extra}
    if res.ok and res.tier in ("VERIFIED", "WITNESSED"):
        return _disputed(question, g, served, readers, kw)
    reply = passage.read(question, model=readers.passage, offline=kw.get("offline", False))
    if reply is None:
        reading["passage"] = {"answer": None, "note": "the text reader was not available"}
        return served
    reading["passage"] = {"answer": reply.get("answer"), "abstain": reply.get("abstain"), "pred": reply.get("pred"),
                          "facts_ok": reply.get("facts_ok"), "cited": passage.cited(reply)}
    seen = reading["readers"]
    answers = not reply.get("abstain") and bool(reply.get("facts_ok"))
    shown = passage.head(reply, g.companies) if answers else None
    declines = [s for s in seen if s["declined"]]
    lacking = len(declines) >= 2 or any(s["role"] == "arbiter" for s in declines)   # the planners say it is not stated
    pointed = _pointer(g, readers, kw)
    cited = _citer(g, reply, shown, kw)

    if not res.ok:
        out = passage.as_answer(reply, g) if answers else None
        if out is not None and (not lacking or NODE_OVER_BELIEF):
            # one printed figure: the node at the place the reader names. A planner that says "filings do
            # not report this" states a belief about filings in general; a node at a named place is the
            # filing itself (round 10, N05: an employee turnover rate two planners said no filing gives)
            if lacking:
                out["result"].tier_reasons.append("the planners said the filings do not state this; the graph holds "
                                                  "the figure at the place the reader names")
            reading["passage"]["used"] = POINTED_NODE
            return {**out, "reading": {**reading, "case": POINTED_NODE, "witnessed": False}}
        if lacking:
            return served
        for s in seen if CITED_ON and answers else []:
            # a planner's plan on the places the reader cited for the whole question: served only if it gives
            # the reader's own figure (two readings agree, every figure a node); no new model call
            got = cited(s["plan"])
            if got is not None:
                return _serve_pointed(got, s["plan"], reading, CITED_LINES,
                                      f"no plan ran on the linker's lines; this {s['role']}'s plan, run on the graph nodes "
                                      f"a reader of the text cited for the whole question ({passage.cited(reply)}), gives "
                                      "that reader's own figure")
        for s in seen:                                # a planner's plan, its missing lines pointed at
            got = pointed(s["plan"], s["result"].missing)
            if got is not None:
                return _serve_pointed(got, s["plan"], reading, POINTED_LINES)
        return served
    if not answers:
        if reading["case"] in WITNESSED:              # the reader has no whole answer: test the plan's lines
            got = pointed(served["plan"])
            if got is not None and same_answer(got[0], res):
                reading["passage"]["pointed"] = got[1]
                res.tier = CORROBORATED
                res.tier_reasons = ["asked for each line of this plan on its own, a reader of the text points at lines "
                                    "that give the same answer"] + list(res.tier_reasons)
                reading["passage"]["used"] = "the pointed lines repeat the served answer"
        return served
    if reading["case"] in (ARBITER_ALONE, ONE_READER) and not passage.same(res, shown):
        for s in seen:
            if s["result"] is not res and s["result"].ok and passage.same(s["result"], shown):
                other = s["result"]
                other.tier = CORROBORATED
                other.tier_reasons = [f"two methods agree: this reader's plan and a reader of the text ({passage.cited(reply)}); "
                                      "no second planner shares the reading"] + \
                                     [r for r in other.tier_reasons if r != UNWITNESSED]
                reading["passage"]["used"] = READER_AND_PASSAGE
                base = s.get("out") or {"answer": s["answer"], "plan": s["plan"], "learned": []}
                return {**base, "result": other, "reading": {**reading, "case": READER_AND_PASSAGE, "witnessed": False}}
        for s in seen if COMPLETE_OTHER_PLANS else []:
            # another planner's plan stopped at a line the linker could not find (round 10, N13: a head-count
            # of two years earlier, "linked but has no value for FY2023"). With that line pointed at, the plan
            # runs on the graph; it is served only if it gives the figure the reader gives for the whole question
            if not s["result"].ok and s["result"].missing:
                got = cited(s["plan"]) if CITED_ON else None
                if got is not None:
                    return _serve_pointed(got, s["plan"], reading, CITED_LINES,
                                          "this planner's plan stopped at a line the linker could not find; on the graph "
                                          "nodes a reader of the text cited for the whole question it gives that reader's "
                                          f"own figure ({passage.cited(reply)})")
                got = pointed(s["plan"], s["result"].missing)
                if got is not None and passage.same(got[0], shown):
                    return _serve_pointed(got, s["plan"], reading, POINTED_LINES,
                                          "this planner's plan stopped at a line the linker could not find; with that line "
                                          "pointed at it gives the figure a reader of the text gives for the whole question "
                                          f"({passage.cited(reply)})")
    if passage.same(res, shown):
        res.tier = CORROBORATED
        res.tier_reasons = [f"a second method repeats this figure: a reader of the text read it in {passage.cited(reply)}"] \
            + list(res.tier_reasons)
        reading["passage"]["used"] = "repeats the served answer"
        return served
    got = pointed(served["plan"])
    if got is not None:
        reading["passage"]["pointed"] = got[1]
        if passage.same(got[0], shown) and not same_answer(got[0], res):
            return _serve_pointed(got, served["plan"], reading, LINES_OVERRULED, _overruled(served, reply))
        # The plan on the pointed lines repeats the graph while the reader's own answer differs: the two
        # agree on the lines the plan names and disagree on what the question asks. Not corroborated
        # (round 8, W05: the plan dropped "firmwide", and both read the channel total).
    alt = cited(served["plan"])                       # the served plan on the places the reader cited: always
    runs = []                                         # (result, nodes), plan, whose plan: candidates for the judge
    if alt is not None and not same_answer(alt[0], res):   # recorded (no model call), served only when switched on
        reading["passage"]["cited_alt"] = {"answer": render(alt[0]), "kind": alt[0].kind, "value": alt[0].value,
                                           "unit": alt[0].unit, "extra": alt[0].extra, "nodes": alt[1]}
        if CITED_OVERRULE and shown is not None and shown[0] == "n":
            return _serve_pointed(alt, served["plan"], reading, CITED_OVERRULED,
                                  f"the linker's lines give {served['answer']}; on the graph nodes a reader of the text "
                                  f"cited for the whole question ({passage.cited(reply)}) the same plan gives that "
                                  "reader's own figure. Every figure here is a node of the graph")
        runs.append((alt, served["plan"], "served planner"))
    if alt is None and shown is not None:
        # A reading dispute: ANOTHER planner's plan gives the reader's own figure on the rows the reader cited,
        # so a planner and the reader agree against the served reading. Recorded (no model call); served only
        # when switched on, for a figure only
        for s in seen:
            other = cited(s["plan"]) if s["plan"] is not served["plan"] else None
            if other is not None and not same_answer(other[0], res):
                reading["passage"]["cited_other"] = {"role": s["role"], "answer": render(other[0]), "kind": other[0].kind,
                                                     "value": other[0].value, "unit": other[0].unit,
                                                     "extra": other[0].extra, "nodes": other[1]}
                if CITED_OTHER and shown[0] == "n":
                    return _serve_pointed(other, s["plan"], reading, CITED_LINES,
                                          f"the served reading gives {served['answer']}; the {s['role']}'s plan, run on the "
                                          f"graph nodes a reader of the text cited for the whole question "
                                          f"({passage.cited(reply)}), gives that reader's own figure: a planner and the "
                                          "reader agree")
                runs.append((other, s["plan"], s["role"]))
                break
    if shown is not None and res.tier == "FLAGGED" and judge_mode() != "off":
        judged = _judge(question, g, served, reply, shown, readers, kw, runs)
        if judged is not None:
            return judged
    res.tier_reasons = list(res.tier_reasons) + [
        f"second reading: a reader of the text answers {str(reply.get('answer'))[:80]!r} ({passage.cited(reply)})"]
    reading["passage"].setdefault("used", "differs: not served")
    return served


_JUDGE_SYSTEM = (
    "You audit answers to questions about SEC filings. Two derivations, A and B, answer the same question from "
    "figures printed in the filings; each lists the printed figures it read (label, filing, page, period), its "
    "arithmetic and its result. Decide which derivation answers the question AS ASKED: the lines the question names "
    "(not a look-alike line, a segment instead of a total, or a total instead of a part), the company and the "
    "period the question asks for, the arithmetic the question implies (a ratio, a change, a sum, a margin), and a "
    "plausible scale for the result. Do not compute an answer of your own. If neither answers the question as "
    'asked, say neither. Reply JSON {"pick": "A" | "B" | "neither", "reason": "one or two sentences"}.')


def _tree_words(tree, depth: int = 0) -> str:
    """The arithmetic of a result tree in words: each operation and what it reads, without its figures."""
    if not isinstance(tree, dict) or depth > 5:
        return ""
    kids = [k for k in (_tree_words(c, depth + 1) for c in tree.get("children") or []) if k]
    out = f"{tree.get('op', '')} {tree.get('what', '')}".strip()
    return (out + (" of (" + "; ".join(kids) + ")" if kids else ""))[:700]


def _derivation(res: Result, plan: Optional[dict], how: str = "", names: Optional[list] = None) -> str:
    """One derivation as the judge reads it: the printed figures read, the arithmetic, the result."""
    import json
    rows = []
    for i, lf in enumerate(res.leaves[:12]):
        name = f"{names[i]} = " if names and i < len(names) else ""
        rows.append(f"  - {name}'{str(lf.label)[:140]}' | {lf.doc} page {lf.page} | period {lf.period or 'not stated'} "
                    f"| printed {lf.printed}")
    steps = (plan or {}).get("steps") if isinstance(plan, dict) else None
    text = "figures read:\n" + ("\n".join(rows) or "  (none listed)")
    text += f"\narithmetic: {how or _tree_words(res.tree) or 'the figure as read'}"
    if steps:
        text += f"\nplan: {json.dumps(steps, default=str)[:900]}"
    return text + f"\nresult: {render(res)}"


def _candidate_b(question: str, g, reply: dict, shown: tuple, readers: Readers, kw: dict, res: Result,
                 runs: list) -> Optional[dict]:
    """The reader's answer computed from graph nodes, tried in this order: (b1) one printed figure, the node at
    the place the reader names (passage.as_answer); (b2) a planner's plan run on the nodes the reader cited for
    the whole question that gives the reader's own figure (passage.by_citing, recorded by second_road); (b3) the
    reader's own arithmetic, one expression evaluated by code on those nodes (passage.by_expression). None when
    none of them gives an answer that differs from the served one."""
    from . import passage
    try:
        one = passage.as_answer(reply, g)
    except Exception:  # noqa: BLE001 - no node store: no candidate
        one = None
    if one is not None and passage.same(one["result"], shown) and not same_answer(one["result"], res):
        return {"source": "b1", "result": one["result"], "plan": one["plan"],
                "how": "the figure itself, as printed (no arithmetic)", "names": None}
    for (got, plan, role) in runs:
        if got is not None and not same_answer(got[0], res):
            return {"source": f"b2 ({role})", "result": got[0], "plan": plan, "how": "", "names": None,
                    "nodes": got[1]}
    try:
        expr = passage.by_expression(question, reply, g, shown, model=readers.passage, offline=kw.get("offline", False))
    except Exception:  # noqa: BLE001
        expr = None
    if expr is not None and not same_answer(expr[0], res):
        b, text, nodes = expr
        return {"source": "b3", "result": b, "nodes": nodes, "how": f"the expression {text}",
                "names": [n["name"] for n in nodes],
                "plan": {"steps": [], "answer": None,
                         "expression": {"expr": text, "nodes": [n["id"] for n in nodes], "named_by": "the text reader"}}}
    return None


def _ask_judge(user: str, arbiters: tuple, offline: bool) -> tuple:
    """(pick "A" | "B" | "NEITHER", reason, model) from the first arbiter of the chain that answers; (None, why,
    None) when none can (every quota spent, every call failed)."""
    from .llm import ask_json
    for model in arbiters:
        try:
            with impatient(model):                   # an overloaded arbiter is skipped for the next in the chain
                got = ask_json(_JUDGE_SYSTEM, user, model=model, tag="judge", offline=offline)
        except DailyLimit:
            continue
        except Exception:  # noqa: BLE001 - a failed call or a reply that is not JSON: the next arbiter
            continue
        pick = str((got or {}).get("pick") or "").strip().upper() if isinstance(got, dict) else ""
        return (pick if pick in ("A", "B") else "NEITHER"), str((got or {}).get("reason") or "")[:500], model
    return None, "no arbiter answered", None


def _judge(question: str, g, served: dict, reply: dict, shown: tuple, readers: Readers, kw: dict,
           runs: list) -> Optional[dict]:
    """A FLAGGED answer the reader disputes, once no other rule settled it: candidate B (_candidate_b), then one
    arbiter call that picks the derivation which answers the question as asked. The reader's derivation picked:
    candidate B is served, FLAGGED (every figure a graph node). Anything else, or "record" mode: None, and the
    served answer stands. The pick and its reason are recorded in reading["passage"]["judge"] in every case."""
    import hashlib
    res, reading = served["result"], served["reading"]
    rec = reading["passage"]["judge"] = {"mode": judge_mode()}
    cand = _candidate_b(question, g, reply, shown, readers, kw, res, runs)
    if cand is None:
        rec["candidate"] = None
        return None
    b = cand["result"]
    rec.update(candidate=cand["source"], B={"answer": render(b), "kind": b.kind, "value": b.value, "unit": b.unit,
                                            "extra": b.extra, "nodes": cand.get("nodes"),
                                            "leaves": [lf.id for lf in b.leaves]})
    a_text = _derivation(res, served.get("plan"))
    b_text = _derivation(b, cand["plan"], cand["how"], cand["names"])
    reader_first = int(hashlib.sha1(question.encode("utf-8")).hexdigest()[:8], 16) % 2 == 1
    first, second = (b_text, a_text) if reader_first else (a_text, b_text)
    user = f"Question: {question}\n\nDerivation A\n{first}\n\nDerivation B\n{second}"
    pick, reason, model = _ask_judge(user, readers.arbiters, kw.get("offline", False))
    reader_letter = "A" if reader_first else "B"
    chose = None if pick is None else ("neither" if pick == "NEITHER" else ("reader" if pick == reader_letter else "served"))
    rec.update(order="reader first" if reader_first else "served first", pick=pick, chose=chose, reason=reason,
               model=model)
    if chose != "reader" or judge_mode() != "on":
        return None
    b.tier = "FLAGGED"
    b.tier_reasons = [f"the served derivation ({served['answer']}) and a derivation of the text reader's answer, computed "
                      f"on graph nodes ({cand['source']}: {render(b)}), disagree; a judge ({model}) chose the reader's: "
                      f"{reason[:200]}"] + [r for r in b.tier_reasons]
    reading["passage"]["used"] = JUDGED
    reading["passage"]["pointed"] = cand.get("nodes")
    return {"answer": render(b), "result": b, "plan": cand["plan"], "learned": [], "seconds": 0.0, "plan_errors": [],
            "repaired": False, "reading": {**reading, "case": JUDGED, "witnessed": False}}


def _disputed(question: str, g, served: dict, readers: Readers, kw: dict) -> dict:
    """A VERIFIED or WITNESSED answer rests on two cheap planners reading the question alike. They can
    share a misreading, and then every number ties out for the wrong question (round 8, W11: "what
    gross margin did Adobe earn on its subscription revenue" was read by both as the company's gross
    margin). So the text reader is asked here too. Its answer is never served; a different figure from
    it has two effects.

    A WITNESSED answer rests on lines two pickers chose from one shortlist. The plan runs again on the
    lines the reader points at; if that run gives the reader's own figure, the shortlist led both
    pickers to a wrong line (round 8, W02: a reserves row in millions of barrels that three plans
    reached) and the run on the pointed lines is served, FLAGGED.

    Otherwise the strongest free model plans the question as well, unless it already has:

        the arbiter repeats the two planners      -> the answer stands, with its tier
        the arbiter repeats the text reader       -> the arbiter's answer (its plan, on the graph), CORROBORATED
        the arbiter says a third thing            -> the answer stands, FLAGGED: three readings
        no arbiter has quota left                 -> the answer stands, and names the second reading

    A dissent costs one strong call; agreement or silence of the reader costs none."""
    from . import passage
    res, reading = served["result"], served["reading"]
    reply = passage.read(question, model=readers.passage, offline=kw.get("offline", False))
    if reply is None:
        reading["passage"] = {"answer": None, "note": "the text reader was not available"}
        return served
    reading["passage"] = {"answer": reply.get("answer"), "abstain": reply.get("abstain"), "pred": reply.get("pred"),
                          "facts_ok": reply.get("facts_ok"), "cited": passage.cited(reply)}
    shown = passage.head(reply, g.companies) if not reply.get("abstain") and reply.get("facts_ok") else None
    if shown is None:
        return served
    if passage.same(res, shown):
        reading["passage"]["used"] = "repeats the served answer"
        return served
    second = f"second reading: a reader of the text answers {str(reply.get('answer'))[:80]!r} ({passage.cited(reply)})"
    seen = reading["readers"]
    if res.tier == "WITNESSED":
        got = _pointer(g, readers, kw)(served["plan"])
        if got is not None:
            reading["passage"]["pointed"] = got[1]
            if passage.same(got[0], shown) and not same_answer(got[0], res):
                return _serve_pointed(got, served["plan"], reading, LINES_OVERRULED, _overruled(served, reply))
    if any(s["role"] == "arbiter" for s in seen):     # the arbiter has read it, and one planner agreed with it
        reading["passage"]["used"] = "differs; the arbiter had already sided with a planner"
        return served
    third, model = _arbitrate(question, g, readers.arbiters, kw)
    if third is None:
        res.tier_reasons = list(res.tier_reasons) + [second]
        reading["passage"]["used"] = "differs; no arbiter available"
        return served
    seen.append(_seen("arbiter", model, third))
    if third["result"].ok and same_answer(third["result"], res):
        reading["passage"]["used"] = "differs; the arbiter repeats the planners"
        return served
    if third["result"].ok and passage.same(third["result"], shown):
        other = third["result"]
        other.tier = CORROBORATED
        other.tier_reasons = [f"two cheap planners read the question another way ({served['answer']}); the arbiter's plan and "
                              f"a reader of the text ({passage.cited(reply)}) agree on this figure"] + list(other.tier_reasons)
        reading["passage"]["used"] = ARBITER_AND_PASSAGE
        return {**third, "reading": {**reading, "case": ARBITER_AND_PASSAGE, "witnessed": False}}
    res.tier = "FLAGGED"
    res.tier_reasons = ["two planners agree, but the arbiter and a reader of the text each read the question differently"] \
        + list(res.tier_reasons) + [second]
    reading["passage"]["used"] = "differs; three readings"
    return served


def _pointer(g, readers: Readers, kw: dict):
    """passage.by_pointing for this question: (plan, missing lines or None for every line) ->
    (the executor's result on the pointed lines, the nodes), or None when the plan cannot be run so."""
    from . import passage

    def pointed(plan, missing=None):
        if kw.get("executor") is None or not isinstance(plan, dict):
            return None
        try:
            return passage.by_pointing(plan, g, kw["executor"], model=readers.passage,
                                       offline=kw.get("offline", False), missing=missing)
        except Exception:  # noqa: BLE001 - a planner's plan the executor itself refused can be malformed
            return None
    return pointed


def _citer(g, reply: dict, shown, kw: dict):
    """passage.by_citing for this question: plan -> (the executor's result on the nodes the reader cited for
    the whole question, the nodes) when that result is the reader's own figure, else None. No model call."""
    from . import passage

    def cited(plan):
        if kw.get("executor") is None or not isinstance(plan, dict) or shown is None:
            return None
        try:
            return passage.by_citing(plan, g, kw["executor"], reply, shown)
        except Exception:  # noqa: BLE001 - a planner's plan the executor itself refused can be malformed
            return None
    return cited


def _overruled(served: dict, reply: dict) -> str:
    from . import passage
    return (f"the linker's lines give {served['answer']}; a reader of the text gives another figure ({passage.cited(reply)}), "
            "and this plan run on the graph lines that reader points at gives that figure too. Every figure here is a "
            "node of the graph")


def _serve_pointed(got: tuple, plan: dict, reading: dict, case: str, why: str = "") -> dict:
    """An answer the executor computed on lines the text reader pointed at, in the shape ask.answer returns."""
    res, nodes = got
    if why:
        res.tier_reasons = [why] + list(res.tier_reasons)
    reading["passage"]["used"] = case
    reading["passage"]["pointed"] = nodes
    return {"answer": render(res), "result": res, "plan": plan, "learned": [], "seconds": 0.0, "plan_errors": [],
            "repaired": False, "reading": {**reading, "case": case, "witnessed": False}}


def _plan_road(question: str, g, readers: Readers, **kw) -> dict:
    """The first road: two planners and an arbiter (see the module docstring)."""
    start_question(pick_arbiter_budget())             # this question's arbiter calls for disputed row picks
    first = ask.answer(question, g, model=readers.maker, **kw)
    second = ask.answer(question, g, model=readers.checker, **kw)
    seen = [_seen("maker", readers.maker, first), _seen("checker", readers.checker, second)]
    if _both_decline(first, second) or same_answer(first["result"], second["result"]):
        return _serve(_prefer(first, second), AGREE, seen)
    if same_order(first["result"], second["result"]):     # a ranking is its order; the figures are its support
        served = _serve(_prefer(first, second), AGREE_ON_ORDER, seen)
        res = served["result"]
        res.tier, res.tier_reasons = "FLAGGED", [FIGURES_DIFFER] + [r for r in res.tier_reasons if r != FIGURES_DIFFER]
        return served
    readers_out = [first, second]
    if getattr(readers, "third", ""):
        # item 8 (23 Nov): a third cheap reader of another model family breaks the tie before any arbiter call
        try:
            extra = ask.answer(question, g, model=readers.third, **kw)
        except DailyLimit:
            extra = None
        if extra is not None:
            seen.append(_seen("third", readers.third, extra))
            readers_out.append(extra)
            for o in (first, second):
                if (extra["result"].ok and same_answer(extra["result"], o["result"])) or (declined(extra) and declined(o)):
                    return _serve(_prefer(o, extra), AGREE_OF_THREE, seen)
    third, model = _arbitrate(question, g, readers.arbiters, kw)
    if third is None:                                # no arbiter configured, or none has quota left
        return _serve(_prefer(first, second), ONE_READER, seen)
    seen.append(_seen("arbiter", model, third))
    if third["result"].ok or declined(third):
        if any(same_answer(third["result"], o["result"]) or (declined(third) and declined(o)) for o in readers_out):
            return _serve(third, ARBITER_AND_READER, seen)
        # three readers, three readings: structure decides first (an answer before none, then the one
        # whose every number ties out); when it cannot, the arbiter's reading, the strongest model's.
        # Measured 22 Nov on 61 such cases with the local 4B as checker (rounds 1 to 6): the arbiter
        # was right 34 times, the reading served by the earlier order (maker, checker) 32. Still FLAGGED.
        return _serve(_prefer(third, first, second), ARBITER_ALONE, seen)
    answered = [o for o in (first, second) if o["result"].ok]
    if answered:                                     # the arbiter's plan did not run; a reader's did
        return _serve(_prefer(*answered), ONE_READER, seen)
    return _serve(third, ARBITER_ALONE, seen)


def pick_arbiter_budget() -> Optional[int]:
    """Arbiter calls allowed per question for disputed row picks (TIEOUT_PICK_ARBITER_MAX, default 3;
    "none" for no limit). Readings come first: a thirty-question run must fit one free key's day."""
    raw = os.environ.get("TIEOUT_PICK_ARBITER_MAX", "3").strip().lower()
    return None if raw in ("", "none") else int(raw)


def same_answer(a: Result, b: Result) -> bool:
    """Do two results say the same thing? Figures to nine digits, names in the same order. One
    figure and a list whose first line carries that figure are the same answer (the scorer reads
    a list of lines by its first line too)."""
    if not (a.ok and b.ok):
        return False
    if {a.kind, b.kind} == {"number", "lines"}:
        figure, listed = (a, b) if a.kind == "number" else (b, a)
        return bool(listed.value) and _close(figure.value, listed.value[0][1])
    if a.kind != b.kind:
        return False
    if a.kind == "number":
        return _close(a.value, b.value) or (a.unit == b.unit == "USD" and _close(abs(a.value), abs(b.value)))
    if a.kind == "text":
        return same_name(a.value, b.value)
    if len(a.value) != len(b.value):                 # companies, lines, series: (name, value) pairs in order
        return False
    return all(str(n1) == str(n2) and (_close(v1, v2) if _is_number(v1) and _is_number(v2) else str(v1) == str(v2))
               for (n1, v1), (n2, v2) in zip(a.value, b.value))


def same_order(a: Result, b: Result) -> bool:
    """Two rankings that name the same members in the same order, whatever their figures."""
    if not (a.ok and b.ok) or a.kind != b.kind or a.kind not in ("companies", "lines"):
        return False
    return len(a.value) > 1 and [str(n) for n, _ in a.value] == [str(n) for n, _ in b.value]


def declined(out: dict) -> bool:
    """The planner said the filings cannot answer this (as opposed to a plan that failed)."""
    why = str((out.get("plan") or {}).get("cannot") or "")
    return bool(why) and not why.startswith("planner failed") and "returned no plan" not in why


def _both_decline(a: dict, b: dict) -> bool:
    return declined(a) and declined(b)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(x, y) -> bool:
    if not (_is_number(x) and _is_number(y)):
        return False
    return abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1e-12)


def _prefer(*outs: dict) -> dict:
    """Among unreconciled readers: an answer over none, then one whose every number ties out."""
    rank = {"VERIFIED": 0, "WITNESSED": 1, "FLAGGED": 2}
    return min(outs, key=lambda o: (not o["result"].ok, rank.get(o["result"].tier, 3)))


def _arbitrate(question: str, g, arbiters: tuple, kw: dict) -> tuple:
    for model in arbiters:
        try:
            with impatient(model):                   # an overloaded arbiter is skipped for the next in the chain
                out = ask.answer(question, g, model=model, **kw)
        except DailyLimit:
            continue
        if str((out.get("plan") or {}).get("cannot") or "").startswith("planner failed"):
            continue                                 # the call itself failed (a timeout): the next arbiter reads
        return out, model
    return None, None


def _seen(role: str, model: Optional[str], out: dict) -> dict:
    res = out["result"]
    return {"role": role, "model": model, "answer": out["answer"], "tier": res.tier or "ABSTAIN", "plan": out["plan"],
            "declined": declined(out), "result": res, "out": out}


def _serve(out: dict, case: str, seen: list) -> dict:
    """The served answer, with its case. A reading no second model shares cannot be VERIFIED."""
    res = out["result"]
    if res.ok and case not in WITNESSED:
        res.tier, res.tier_reasons = "FLAGGED", [UNWITNESSED] + [r for r in res.tier_reasons if r != UNWITNESSED]
    return {**out, "reading": {"case": case, "witnessed": case in WITNESSED, "readers": seen}}
