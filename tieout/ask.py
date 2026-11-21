"""Question in, sourced answer out.

    question -> plan (language model, vocabulary only) -> validate -> execute on the graph
             -> where the graph has no tie yet for something the plan names, the living
                layer tries to learn one, then the plan runs again:
                    a shared concept with no row bound at this company   (living.learn)
                    a line item or a figure stated in a sentence         (link.learn_link)
                    the parts of a total                                 (link.learn_breakdown)
             -> if the plan still cannot run, or its answer is of the wrong type for the question
                (a list where "how many" was asked, an amount where a share was asked, one number
                where "which of A or B" was asked), the planner sees its plan and the reason once,
                with figures masked, and may repair it; the repair is kept only if it runs and fits
             -> answer with its proof leaves, or an abstention that names what is missing

Learning never changes a number. It only records which printed row or sentence
a phrase refers to, admitted by the structural gate and stored with the
question that taught it.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from .concepts import concept_defs
from .execute import Executor, Result, abstain, render
from .graph import Graph
from .link import learn_breakdown, learn_link, save_link
from .living import learn, save_entry
from .llm import DailyLimit
from .planner import plan_question

LEARNING_ROUNDS = 3                               # a ranking can surface several gaps; a few rounds settle them
_FIGURE = re.compile(r"\$?-?\(?\d[\d,]*\.?\d*\)?%?")


def answer(question: str, g: Graph, model: Optional[str] = None, allow_learning: bool = True,
           question_id: str = "", offline: bool = False, plan: Optional[dict] = None,
           executor: Optional[Executor] = None, persist: bool = True, repair: bool = True) -> dict:
    """Return {"answer", "result", "plan", "learned", "seconds", "plan_errors", "repaired"}."""
    t0 = time.time()
    ex = executor or Executor(g)
    errors: list = []
    planned_here = plan is None
    if plan is None:
        plan, errors = _plan(question, g, model, offline)
    learn_args = dict(allow_learning=allow_learning, question=question, question_id=question_id, model=model,
                      offline=offline, persist=persist)
    res, learned = _execute(plan, g, ex, **learn_args)
    repaired = False
    misfit = shape_misfit(question, plan, res)
    if planned_here and repair and (misfit or _repairable(res, plan)):
        problem = misfit or _problem(res, plan, errors)
        fixed, errs2 = _plan(question, g, model, offline, feedback={"plan": plan, "problem": problem})
        if fixed.get("steps") and not errs2 and fixed != plan:
            res2, learned2 = _execute(fixed, g, ex, **learn_args)
            learned += learned2
            if res2.ok and not shape_misfit(question, fixed, res2):
                res, plan, errors, repaired = res2, fixed, [], True
    out = _package(res, plan, learned, t0, errors)
    out["repaired"] = repaired
    return out


def _plan(question: str, g: Graph, model, offline: bool, feedback: Optional[dict] = None) -> tuple:
    """(plan, validation errors left). A model that returns no usable plan must not crash an answer."""
    try:
        planned = plan_question(question, g, model=model, offline=offline, feedback=feedback)
        return planned["plan"], planned["errors"]
    except DailyLimit:
        raise                                     # a spent quota stops the run; it is not an answer
    except Exception as e:  # noqa: BLE001
        return {"steps": [], "answer": None, "cannot": f"planner failed ({type(e).__name__}: {str(e)[:80]})"}, []


def _execute(plan: dict, g: Graph, ex: Executor, allow_learning: bool, question: str, question_id: str,
             model, offline: bool, persist: bool) -> tuple:
    """Run a plan, closing the gaps it reports by learning, a few rounds at most. Returns (result, learned)."""
    if plan.get("cannot") or not plan.get("steps"):
        return abstain("planner: " + str(plan.get("cannot", "question is outside the graph's vocabulary"))), []
    res = ex.run(plan)
    learned: list = []
    tried: set = set()
    for _ in range(LEARNING_ROUNDS if allow_learning else 0):
        gaps = [gap for gap in dict.fromkeys(_hashable(m) for m in res.missing) if gap not in tried]
        if not gaps:
            break
        for gap in gaps:
            tried.add(gap)
            entry = _learn(g, ex, gap, question, question_id, model, offline)
            if entry is not None:
                learned.append(entry)
                if persist:
                    (save_link if entry.get("kind") in ("item", "mention", "breakdown") else save_entry)(entry)
        res = ex.run(plan)
    return res, learned


_HOW_MANY = re.compile(r"(?<!by )(?<!by roughly )(?<!by about )(?<!by approximately )\bhow many\b"
                       r"(?! (times|percentage points?|dollars|billion|million|basis points))", re.I)
_SHARE = re.compile(r"\b(what|which) (share|fraction|percentage|proportion|portion)\b|\bas a (share|percentage|fraction)\b"
                    r"|\bmargin\b|\bratio\b|\bpercent of\b|\bslice of\b", re.I)
_EITHER = re.compile(r"\b(which|who)\b[^?]*\bor\b", re.I)
_QUARTER = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(fiscal\s+)?quarter\b|\bQ[1-4]\b", re.I)
_SO_FAR = re.compile(r"\b(nine|six|three)\s+months\b|\byear[- ]to[- ]date\b|\bfirst (half|nine months)\b", re.I)


def shape_misfit(question: str, plan: dict, res: Result) -> str:
    """Why the answer's type cannot answer the question, or "" if it can (or the plan did not run).

    Three checks only, each a clear mismatch between the words of the question and the kind of
    answer: "how many" answered with a list, an amount where a share is asked, and a single
    number where the question asks which of two named companies. Nothing here looks at a value.
    """
    if not res.ok or not plan.get("steps"):
        return ""
    last = next((s for s in plan["steps"] if s.get("id") == plan.get("answer")), {})
    if _HOW_MANY.search(question) and res.kind in ("companies", "lines") and last.get("op") != "count":
        return "the question asks how many: end the plan with a count step over the set"
    if not _HOW_MANY.search(question) and last.get("op") == "count":
        return "the question does not ask how many: the answer is not a count; end on the step that answers it"
    if _SHARE.search(question) and res.kind == "number" and res.unit == "USD":
        return "the question asks for a share, margin or ratio, but the plan returns an amount in dollars"
    periods = [p for s in plan["steps"] for p in (s.get("period"), s.get("from"), s.get("to"),
                                                   (s.get("where") or {}).get("period")) if isinstance(p, dict)]
    quarter, so_far = _QUARTER.search(question), _SO_FAR.search(question)
    if quarter and not so_far and not any(p.get("q") for p in periods):
        return "the question asks for one quarter: give the period as {\"fy\": Y, \"q\": Q}"
    if quarter and not so_far and any(p.get("ytd") for p in periods):
        return "the question asks for one quarter, not the year to date: drop \"ytd\""
    if so_far and not any(p.get("ytd") or p.get("q") for p in periods):
        return "the question asks for a part of the year (nine or six months): use {\"fy\": Y, \"q\": Q, \"ytd\": true}"
    named = [s for s in plan["steps"] if s.get("op") in ("get", "change", "metric")]
    if (_EITHER.search(question) and last.get("op") == "calc" and len({s.get("company") for s in named}) == 2
            and not re.search(r"\bhow much (more|less|higher|lower|larger|smaller)\b", question, re.I)):
        return ("the question asks which of two companies: rank over the two with k=2, which gives both values, "
                "the winner and the gap")
    return ""


def _repairable(res: Result, plan: dict) -> bool:
    """Worth one repair: the plan ran into a wall that a different plan might not hit.

    A deliberate "cannot" and a fact the company is recorded as not reporting are answers, not walls.
    """
    if res.ok or plan.get("cannot") or not plan.get("steps"):
        return False
    return not res.extra.get("not_reported")


def _problem(res: Result, plan: dict, errors: list) -> str:
    """What went wrong, in words, with every figure masked so the planner never sees a value."""
    text = "; ".join(errors + res.notes) or "the plan returned nothing"
    return _FIGURE.sub(lambda m: m.group(0) if re.fullmatch(r"(19|20)\d\d", m.group(0)) else "<n>", text)[:600]


def _hashable(missing: tuple) -> tuple:
    ticker, what, spec = missing
    return ticker, what, tuple(sorted(spec.items())) if isinstance(spec, dict) else str(spec)


def _learn(g: Graph, ex: Executor, gap: tuple, question: str, question_id: str, model, offline: bool) -> Optional[dict]:
    """Try to close one gap the executor reported. Returns the store entry, or None if not learnable."""
    ticker, what, spec_items = gap
    spec = dict(spec_items) if isinstance(spec_items, tuple) else {}
    common = dict(question=question, question_id=question_id, model=model, offline=offline)
    if isinstance(what, str) and what.startswith(("item:", "members:")):
        wanted = {ex._pid(ticker, spec, "flow"), ex._pid(ticker, spec, "stock")}
        fiscal_year = f"FY{ex._fy(ticker, spec)}"
        if what.startswith("item:"):
            return learn_link(g, ticker, what[len("item:"):], wanted, fiscal_year, **common)
        return learn_breakdown(g, ticker, what[len("members:"):], wanted, fiscal_year, **common)
    if what in concept_defs() and not any(k.startswith(f"{ticker}|{what}|") for k in g.facts):
        return learn(g, ticker, what, **common)
    return None


def _package(res: Result, plan: dict, learned: list, t0: float, errors: list) -> dict:
    return {"answer": render(res), "result": res, "plan": plan, "learned": learned,
            "seconds": time.time() - t0, "plan_errors": errors}
