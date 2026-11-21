"""Question in, sourced answer out.

    question -> plan (language model, vocabulary only) -> validate -> execute on the graph
             -> where the graph has no tie yet for something the plan names, the living
                layer tries to learn one, then the plan runs again:
                    a shared concept with no row bound at this company   (living.learn)
                    a line item or a figure stated in a sentence         (link.learn_link)
                    the parts of a total                                 (link.learn_breakdown)
             -> if the plan still cannot run, the planner sees its plan and the executor's
                reason once (figures masked) and may repair it; the repair is kept only if it runs
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
    if planned_here and repair and _repairable(res, plan):
        problem = _problem(res, plan, errors)
        fixed, errs2 = _plan(question, g, model, offline, feedback={"plan": plan, "problem": problem})
        if fixed.get("steps") and not errs2 and fixed != plan:
            res2, learned2 = _execute(fixed, g, ex, **learn_args)
            learned += learned2
            if res2.ok:
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
