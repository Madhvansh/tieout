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

UNWITNESSED = "the planners read the question differently; this reading has no second witness"
FIGURES_DIFFER = "two readers rank these in the same order but do not compute every figure the same way"


def answer(question: str, g, readers: Readers, **kw) -> dict:
    """`ask.answer` under the four-eyes rule. Returns the served answer's dict with "reading":
    the case, and what each reader answered."""
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
            "declined": declined(out), "result": res}


def _serve(out: dict, case: str, seen: list) -> dict:
    """The served answer, with its case. A reading no second model shares cannot be VERIFIED."""
    res = out["result"]
    if res.ok and case not in WITNESSED:
        res.tier, res.tier_reasons = "FLAGGED", [UNWITNESSED] + [r for r in res.tier_reasons if r != UNWITNESSED]
    return {**out, "reading": {"case": case, "witnessed": case in WITNESSED, "readers": seen}}
