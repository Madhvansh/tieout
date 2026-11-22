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

import json
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
    if planned_here and misfit and any(w in misfit for w in ("quarter", "part of the year", "named", "threshold",
                                                              "audits")):
        fixed = fix_from_words(question, fix_named_set(question, fix_periods(question, plan), g))   # no model call
        if fixed != plan:
            res2, learned2 = _execute(fixed, g, ex, **learn_args)
            learned += learned2
            if res2.ok and not shape_misfit(question, fixed, res2):
                res, plan, misfit, repaired = res2, fixed, "", True
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
_AUDITOR = re.compile(r"\b(audits?|auditor|accounting firm)\b", re.I)
_DIVIDED = re.compile(r"\bdivided by\b|\bdividing\b", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_AMOUNT = re.compile(r"(\$)?\s*(\d[\d,]*\.?\d*)\s*(%|percent\b|thousand\b|million\b|billion\b|trillion\b)?", re.I)


def stated_amounts(question: str) -> list:
    """Every amount the question states, in full units ($10 billion -> 1e10, 20 percent -> 0.2).
    Years and counts written without a unit are left out: they are not thresholds."""
    out = []
    for m in _AMOUNT.finditer(question):
        dollar, number, unit = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "").lower()
        if not (dollar or unit) or not number.replace(".", "", 1).isdigit():
            continue
        v = float(number)
        if unit in ("%", "percent"):
            out.append(v / 100)
        else:
            out.append(v * _SCALE.get(unit, 1.0))
    return out


def _same_kind(x: float, stated: list) -> list:
    """The stated amounts a threshold can be compared with: shares and ratios (at most 1.5) with
    percentages, money (a million or more) with money. Counts are never checked: questions state
    them without a unit, and a count of 1,000 must not be read as a dollar amount."""
    if abs(x) <= 1.5:
        return [v for v in stated if abs(v) <= 1.5]
    if abs(x) >= 1e6:
        return [v for v in stated if abs(v) >= 1e6]
    return []


def _thresholds(step: dict) -> list:
    w = step.get("where") if isinstance(step.get("where"), dict) else {}
    return [float(w[k]) for k in ("gt", "lt", "ge", "le") if isinstance(w.get(k), (int, float))
            and not isinstance(w.get(k), bool) and w[k] != 0]


def _divides(plan: dict) -> bool:
    text = json.dumps(plan)
    return '"/' in text or "/ " in text or '"ratio"' in text or '"metric"' in text


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
    for s in plan["steps"]:
        for x in _thresholds(s):
            same_kind = _same_kind(x, stated_amounts(question))
            if same_kind and not any(abs(x - v) <= 1e-9 * max(1.0, abs(v)) for v in same_kind):
                return (f"the threshold {x:g} is not an amount the question states; the question states "
                        + ", ".join(f"{v:g}" for v in same_kind))
    if _AUDITOR.search(question) and not any(s.get("op") == "attr" and s.get("attr") == "auditor" for s in plan["steps"]):
        return "the question asks which firm audits: end with an attr step for the auditor of the company found"
    if _DIVIDED.search(question) and not _divides(plan):
        return "the question divides one figure by another: use a ratio measure or a calc with /"
    last = next((s for s in plan["steps"] if s.get("id") == plan.get("answer")), {})
    if _HOW_MANY.search(question) and res.kind in ("companies", "lines") and last.get("op") != "count":
        return "the question asks how many: end the plan with a count step over the set"
    if not _HOW_MANY.search(question) and last.get("op") == "count":
        return "the question does not ask how many: the answer is not a count; end on the step that answers it"
    if _SHARE.search(question) and res.kind == "number" and res.unit == "USD":
        return "the question asks for a share, margin or ratio, but the plan returns an amount in dollars"
    ranks = [s for s in plan["steps"] if s.get("op") in ("rank", "filter", "agg")]
    named = named_companies(question)
    if len(named) >= 3 and ranks and not isinstance(ranks[0].get("over"), list) \
            and not str(ranks[0].get("over", "")).startswith("$"):
        return "the question names the companies to compare: use them as the set, not a cluster or all"
    for s in ranks:
        if (s.get("op") == "rank" and isinstance(s.get("over"), list) and int(s.get("k", 1)) < len(s["over"])
                and re.search(r"\brank\b", question, re.I) and not re.search(r"\btop\b", question, re.I)):
            return "the question asks to rank the named companies: keep all of them (k = their number)"
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


def named_companies(question: str) -> list:
    """Tickers of the listed companies the question names, in the order it names them (common names
    from semantics/company_aliases.yaml; tickers of one or two letters are not matched on their own)."""
    found = []
    for ticker, names in _aliases().items():
        for name in names:
            m = re.search(rf"(?<![\w&]){re.escape(name)}(?![\w&])", question)   # proper nouns: case matters
            if m:
                found.append((m.start(), ticker))
                break
    return [t for _, t in sorted(found)]


_ALIASES: dict = {}


def _aliases() -> dict:
    if not _ALIASES:
        import pathlib

        import yaml
        path = pathlib.Path(__file__).resolve().parents[1] / "semantics" / "company_aliases.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _ALIASES.update({t: [n for n in names if len(n) > 2] for t, names in raw.items()})
    return _ALIASES


def fix_named_set(question: str, plan: dict, g=None) -> dict:
    """Rank or compare over the companies the question names (in place of a cluster or "all"), and rank
    all of them when it asks to rank them. Only the first set step is touched."""
    named = named_companies(question)
    steps, done = [], False
    for s in plan.get("steps", []):
        s = dict(s)
        if not done and s.get("op") in ("rank", "filter", "agg"):
            if len(named) >= 3 and not isinstance(s.get("over"), list) and not str(s.get("over", "")).startswith("$"):
                s["over"] = named
            if (s.get("op") == "rank" and isinstance(s.get("over"), list) and re.search(r"\brank\b", question, re.I)
                    and not re.search(r"\btop\b", question, re.I)):
                s["k"] = len(s["over"])
            done = True
        steps.append(s)
    return {**plan, "steps": steps}


def fix_from_words(question: str, plan: dict) -> dict:
    """Two repairs read straight from the question: a single threshold the plan got wrong becomes the one
    amount the question states (in the same direction), and a question asking who audits the company a
    ranking or filter finds gets that hop appended."""
    steps = []
    for s in plan.get("steps", []):
        s = dict(s)
        w = s.get("where")
        if isinstance(w, dict) and len(_thresholds(s)) == 1:
            same_kind = _same_kind(_thresholds(s)[0], stated_amounts(question))
            if len(same_kind) == 1:
                k = next(k for k in ("gt", "lt", "ge", "le") if isinstance(w.get(k), (int, float)) and w[k] != 0)
                s["where"] = {**w, k: same_kind[0]}
        steps.append(s)
    plan = {**plan, "steps": steps}
    if _AUDITOR.search(question) and not any(s.get("op") == "attr" and s.get("attr") == "auditor" for s in steps):
        finder = next((s for s in reversed(steps) if s.get("op") in ("rank", "filter")), None)
        if finder is not None:
            sid = f"s{len(steps) + 1}"
            plan = {**plan, "steps": steps + [{"id": sid, "op": "attr", "company": "$" + finder["id"], "attr": "auditor"}],
                    "answer": sid}
    return plan


_QWORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_MONTHS = {"three": 1, "six": 2, "nine": 3}


def fix_periods(question: str, plan: dict) -> dict:
    """Set the periods a question names in words: "the third quarter" is {"q": 3} with no year to date,
    "the first nine months" is {"q": 3, "ytd": true}. Only periods that already name a year are touched."""
    quarter, so_far = _QUARTER.search(question), _SO_FAR.search(question)
    if quarter and not so_far:
        word = (quarter.group(1) or "").lower()
        q = _QWORD.get(word) or int(re.search(r"[1-4]", quarter.group(0)).group(0))
        change = {"q": q, "ytd": None}
    elif so_far:
        m = re.search(r"\b(nine|six|three)\s+months\b", question, re.I)
        change = {"q": _MONTHS[m.group(1).lower()] if m else 3, "ytd": True}
    else:
        return plan
    steps = []
    for s in plan.get("steps", []):
        s = dict(s)
        for key in ("period", "from", "to"):
            p = s.get(key)
            if isinstance(p, dict) and "fy" in p:
                p = {**p, "q": change["q"]}
                if change["ytd"]:
                    p["ytd"] = True
                else:
                    p.pop("ytd", None)
                s[key] = p
        steps.append(s)
    return {**plan, "steps": steps}


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
