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
from .execute import Result, same_name
from .link import start_question
from .llm import DailyLimit, impatient
from .witness import (AGREE, AGREE_OF_THREE, AGREE_ON_ORDER, ARBITER_ALONE, ARBITER_AND_READER,  # noqa: F401
                      ONE_READER, WITNESSED, Panel as Readers)

CORROBORATED = "CORROBORATED"                         # a tier: a FLAGGED answer that a second method repeats
PASSAGE_ALONE = "the passage reader alone"            # two more cases of a reading, from the second road
READER_AND_PASSAGE = "one reader and the passage reader agree"
ARBITER_AND_PASSAGE = "the arbiter and the passage reader agree"
PLAN_ON_PASSAGES = "a plan on facts read from passages"
PASSAGES_OVERRULE = "two passage readings against the graph's line"
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
    """A passage reader as a witness of another kind (tieout/passage.py). It is asked only when the
    plan road did not end VERIFIED or WITNESSED, and it changes the served answer in two cases only,
    both measured before they were built (rounds 7 to 9, 23 Nov: 139 of 178 right without it, 144 with):

      the plan road has no answer, and no arbiter and no two planners say the filings lack it
          -> the passage answer, FLAGGED (it is one model reading text; nothing ties it out)
      the served reading has no second witness (arbiter alone, one reader only) and the passage
      reader's figure is the figure of another reader's plan
          -> that reader's answer

    A FLAGGED answer whose figure the passage reader repeats is CORROBORATED: two different methods
    reached it (right 28 of 30 on those rounds, against 31 of 53 for a FLAGGED answer it does not
    repeat). When the passage reader says something else, serving its answer lost more than it won
    (141 of 178), so one more vote is taken: the served plan is run again on facts read from passages
    (passage.run_plan: the plan road's reading, the passage road's lines).

      that run gives the passage reader's own figure  -> the graph read a wrong line for a right plan:
                                                         the passage figure is served, FLAGGED
      that run gives the graph's figure, and the      -> the lines are corroborated: CORROBORATED
      reading had two planners
      neither                                         -> the served answer stays and names the second reading

    The same run answers a question the graph has no line for when the passage reader could not answer
    it whole (several filings, several steps), FLAGGED."""
    from . import passage
    res, reading = served["result"], served["reading"]
    reading["plan_road"] = {"answer": served["answer"], "tier": res.tier or "ABSTAIN", "kind": res.kind,
                            "value": res.value, "extra": res.extra}
    if res.ok and res.tier in ("VERIFIED", "WITNESSED"):
        return _disputed(question, g, served, readers, kw)
    offline = kw.get("offline", False)
    reply = passage.read(question, model=readers.passage, offline=offline)
    if reply is None:
        reading["passage"] = {"answer": None, "note": "the passage reader was not available"}
        return served
    reading["passage"] = {"answer": reply.get("answer"), "abstain": reply.get("abstain"), "pred": reply.get("pred"),
                          "facts_ok": reply.get("facts_ok"), "cited": passage.cited(reply)}
    seen = reading["readers"]
    answers = not reply.get("abstain") and bool(reply.get("facts_ok"))
    shown = passage.head(reply, g.companies) if answers else None
    declines = [s for s in seen if s["declined"]]
    lacking = len(declines) >= 2 or any(s["role"] == "arbiter" for s in declines)   # the planners say it is not stated

    def on_passages(plan):
        """The plan's answer on facts read from passages, or None."""
        if kw.get("executor") is None or not isinstance(plan, dict):
            return None
        return passage.run_plan(plan, g, kw["executor"], model=readers.passage, offline=offline)

    if not res.ok:
        if lacking:
            return served
        if answers:
            out = passage.as_answer(reply, g.companies)
            reading["passage"]["used"] = PASSAGE_ALONE
            return {**out, "reading": {**reading, "case": PASSAGE_ALONE, "witnessed": False}}
        for s in seen:                                # no whole answer: a planner's plan on passage facts
            ran = on_passages(s["plan"])
            if ran is not None:
                reading["passage"]["used"] = PLAN_ON_PASSAGES
                reading["passage"]["plan_on_passages"] = ran["value"]
                return {**passage.plan_answer(ran, s["plan"]),
                        "reading": {**reading, "case": PLAN_ON_PASSAGES, "witnessed": False}}
        return served
    if not answers:
        if reading["case"] in WITNESSED:              # the passage reader has no whole answer: check the lines
            ran = on_passages(served["plan"])
            if ran is not None and passage.same(res, ("n", ran["value"])):
                reading["passage"]["plan_on_passages"] = ran["value"]
                res.tier = CORROBORATED
                res.tier_reasons = ["the same plan on facts read from passages gives the same figure"] + list(res.tier_reasons)
                reading["passage"]["used"] = "the plan on passage facts repeats the served answer"
        return served
    if reading["case"] in (ARBITER_ALONE, ONE_READER) and not passage.same(res, shown):
        for s in seen:
            if s["result"] is not res and s["result"].ok and passage.same(s["result"], shown):
                other = s["result"]
                other.tier = CORROBORATED
                other.tier_reasons = [f"two methods agree: this reader's plan and a passage reader ({passage.cited(reply)}); "
                                      "no second planner shares the reading"] + \
                                     [r for r in other.tier_reasons if r != UNWITNESSED]
                reading["passage"]["used"] = READER_AND_PASSAGE
                base = s.get("out") or {"answer": s["answer"], "plan": s["plan"], "learned": []}
                return {**base, "result": other, "reading": {**reading, "case": READER_AND_PASSAGE, "witnessed": False}}
    if passage.same(res, shown):
        res.tier = CORROBORATED
        res.tier_reasons = [f"a second method repeats this figure: a passage reader read it in {passage.cited(reply)}"] \
            + list(res.tier_reasons)
        reading["passage"]["used"] = "repeats the served answer"
        return served
    ran = on_passages(served["plan"]) if shown is not None and shown[0] == "n" else None
    if ran is not None:
        reading["passage"]["plan_on_passages"] = ran["value"]
        if passage.same_number(ran["value"], shown[1]):   # two passage readings against the graph's line
            out = passage.as_answer(reply, g.companies)
            out["result"].tier_reasons = [
                f"the graph's line gives {served['answer']}; the same plan on facts read from passages and the passage "
                f"reader's own reading both give this figure ({passage.cited(reply)}). Served on the passages, not tied by arithmetic"]
            reading["passage"]["used"] = PASSAGES_OVERRULE
            return {**out, "plan": served["plan"], "learned": [],
                    "reading": {**reading, "case": PASSAGES_OVERRULE, "witnessed": False}}
        # The plan on passage facts repeats the graph while the passage reader's own answer differs: the
        # two roads agree on the line the plan names, and disagree on what the question asks. Not
        # corroborated (round 8, W05: the plan dropped "firmwide", and both roads read the channel total).
    res.tier_reasons = list(res.tier_reasons) + [
        f"second reading: a passage reader answers {str(reply.get('answer'))[:80]!r} ({passage.cited(reply)})"]
    reading["passage"].setdefault("used", "differs: not served")
    return served


def _disputed(question: str, g, served: dict, readers: Readers, kw: dict) -> dict:
    """A VERIFIED or WITNESSED answer rests on two cheap planners reading the question alike. They can
    share a misreading, and then every number ties out for the wrong question (round 8, W11: "what
    gross margin did Adobe earn on its subscription revenue" was read by both as the company's gross
    margin). So the passage reader is asked here too. When it gives a different figure, the strongest
    free model plans the question as well, unless it already has:

        the arbiter repeats the two planners      -> the answer stands, with its tier
        the arbiter repeats the passage reader    -> the arbiter's answer, CORROBORATED
        the arbiter says a third thing            -> the answer stands, FLAGGED: three readings
        no arbiter has quota left                 -> the answer stands, and names the second reading

    A dissent costs one strong call; agreement or silence of the passage reader costs none."""
    from . import passage
    res, reading = served["result"], served["reading"]
    reply = passage.read(question, model=readers.passage, offline=kw.get("offline", False))
    if reply is None:
        reading["passage"] = {"answer": None, "note": "the passage reader was not available"}
        return served
    reading["passage"] = {"answer": reply.get("answer"), "abstain": reply.get("abstain"), "pred": reply.get("pred"),
                          "facts_ok": reply.get("facts_ok"), "cited": passage.cited(reply)}
    shown = passage.head(reply, g.companies) if not reply.get("abstain") and reply.get("facts_ok") else None
    if shown is None:
        return served
    if passage.same(res, shown):
        reading["passage"]["used"] = "repeats the served answer"
        return served
    second = f"second reading: a passage reader answers {str(reply.get('answer'))[:80]!r} ({passage.cited(reply)})"
    seen = reading["readers"]
    if res.tier == "WITNESSED" and shown[0] == "n" and kw.get("executor") is not None:
        # WITNESSED rests on lines two models chose from one shortlist. The same plan on facts read from
        # passages is the test of those lines: if it gives the passage reader's figure, the shortlist
        # led both pickers to a wrong line (round 8, W02: Exxon's reserves, a row in millions of barrels
        # that three planners' plans all reached)
        ran = passage.run_plan(served["plan"], g, kw["executor"], model=readers.passage, offline=kw.get("offline", False))
        if ran is not None:
            reading["passage"]["plan_on_passages"] = ran["value"]
            if passage.same_number(ran["value"], shown[1]):
                out = passage.as_answer(reply, g.companies)
                out["result"].tier_reasons = [
                    f"the graph's line gives {served['answer']}; the same plan on facts read from passages and the passage "
                    f"reader's own reading both give this figure ({passage.cited(reply)}). Served on the passages, not tied by arithmetic"]
                reading["passage"]["used"] = PASSAGES_OVERRULE
                return {**out, "plan": served["plan"], "learned": [],
                        "reading": {**reading, "case": PASSAGES_OVERRULE, "witnessed": False}}
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
                              f"a passage reader ({passage.cited(reply)}) agree on this figure"] + list(other.tier_reasons)
        reading["passage"]["used"] = ARBITER_AND_PASSAGE
        return {**third, "reading": {**reading, "case": ARBITER_AND_PASSAGE, "witnessed": False}}
    res.tier = "FLAGGED"
    res.tier_reasons = ["two planners agree, but the arbiter and a passage reader each read the question differently"] \
        + list(res.tier_reasons) + [second]
    reading["passage"]["used"] = "differs; three readings"
    return served


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
