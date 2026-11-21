"""Question in, sourced answer out.

    question -> plan (language model, vocabulary only) -> validate -> execute on the graph
             -> where the graph has no tie yet for something the plan names, the living
                layer tries to learn one, then the plan runs again:
                    a shared concept with no row bound at this company   (living.learn)
                    a line item or a figure stated in a sentence         (link.learn_link)
                    the parts of a total                                 (link.learn_breakdown)
             -> answer with its proof leaves, or an abstention that names what is missing

Learning never changes a number. It only records which printed row or sentence
a phrase refers to, admitted by the structural gate and stored with the
question that taught it.
"""
from __future__ import annotations

import time
from typing import Optional

from .concepts import concept_defs
from .execute import Executor, Result, abstain, render
from .graph import Graph
from .link import learn_breakdown, learn_link, save_link
from .living import learn, save_entry
from .planner import plan_question

LEARNING_ROUNDS = 3                               # a ranking can surface several gaps; a few rounds settle them


def answer(question: str, g: Graph, model: Optional[str] = None, allow_learning: bool = True,
           question_id: str = "", offline: bool = False, plan: Optional[dict] = None,
           executor: Optional[Executor] = None, persist: bool = True) -> dict:
    """Return {"answer", "result", "plan", "learned", "seconds", "plan_errors"}."""
    t0 = time.time()
    ex = executor or Executor(g)
    errors: list = []
    if plan is None:
        try:
            planned = plan_question(question, g, model=model, offline=offline)
            plan, errors = planned["plan"], planned["errors"]
        except Exception as e:  # noqa: BLE001 - a model that returns no usable plan must not crash an answer
            plan = {"steps": [], "answer": None, "cannot": f"planner failed ({type(e).__name__}: {str(e)[:80]})"}
    if plan.get("cannot") or not plan.get("steps"):
        res = abstain("planner: " + str(plan.get("cannot", "question is outside the graph's vocabulary")))
        return _package(res, plan, [], t0, errors)
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
    return _package(res, plan, learned, t0, errors)


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
