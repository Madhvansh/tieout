"""Question in, sourced answer out.

    question -> plan (LLM, vocabulary only) -> validate -> execute on the graph
             -> if a (company, concept) has no binding: the living layer tries to
                learn one through the structural gate, then the plan runs again
             -> answer with its proof leaves, or an abstention that names what is missing
"""
from __future__ import annotations

import time
from typing import Optional

from .concepts import concept_defs
from .execute import Executor, Result, render
from .graph import Graph
from .items import learn_item, save_item
from .living import learn, save_entry
from .rawlayer import load_raw

_RAW: list = []


def _raw() -> list:
    """The raw layer is large; load it the first time a question needs it."""
    if not _RAW:
        _RAW.extend(load_raw())
    return _RAW
from .planner import plan_question


def answer(question: str, g: Graph, model: Optional[str] = None, allow_learning: bool = True,
           question_id: str = "", offline: bool = False, plan: Optional[dict] = None,
           executor: Optional[Executor] = None, persist: bool = True) -> dict:
    """Return {"answer", "result", "plan", "learned", "seconds", "plan_errors"}."""
    t0 = time.time()
    ex = executor or Executor(g)
    errors: list = []
    if plan is None:
        try:
            p = plan_question(question, g, model=model, offline=offline)
            plan, errors = p["plan"], p["errors"]
        except Exception as e:  # noqa: BLE001 - a model that returns no usable plan must not crash an answer
            plan, errors = {"steps": [], "answer": None, "cannot": f"planner failed ({type(e).__name__}: {str(e)[:80]})"}, []
    if plan.get("cannot") or not plan.get("steps"):
        res = Result("abstain", notes=["planner: " + str(plan.get("cannot", "question is outside the graph's vocabulary"))])
        return {"answer": render(res), "result": res, "plan": plan, "learned": [], "seconds": time.time() - t0,
                "plan_errors": errors}
    res = ex.run(plan)
    learned = []
    if allow_learning:
        concepts = concept_defs()
        tried = set()
        for _ in range(3):                         # a ranking can surface several gaps; a few rounds settle them
            gaps = list(dict.fromkeys((t, c) for (t, c, _p) in _all_missing(res) if c in concepts and (t, c) not in tried
                                      and not any(k.startswith(f"{t}|{c}|") for k in g.facts)))
            item_gaps = list(dict.fromkeys((t, c[5:]) for (t, c, _p) in _all_missing(res)
                                           if isinstance(c, str) and c.startswith("item:") and (t, c) not in tried))
            if not gaps and not item_gaps:
                break
            for t, phrase in item_gaps:
                tried.add((t, "item:" + phrase))
                e = learn_item(g, _raw(), t, phrase, question_id=question_id, question=question, model=model,
                               offline=offline)
                learned.append(e)
                if persist:
                    save_item(e)
            for t, c in gaps:
                tried.add((t, c))
                e = learn(g, t, c, question_id=question_id, question=question, model=model, offline=offline)
                learned.append(e)
                if persist:
                    save_entry(e)
            res = ex.run(plan)
    return {"answer": render(res), "result": res, "plan": plan, "learned": learned,
            "seconds": time.time() - t0, "plan_errors": errors}


def _all_missing(res: Result) -> list:
    """Facts the executor wanted and did not find, from an abstention or from a ranking's skips."""
    return list(res.missing)
