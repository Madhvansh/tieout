"""The second road: a passage reader, a witness of another kind.

The graph answers from structure: a plan names lines, the executor reads them, arithmetic vouches for
them. That road fails in two ways the arithmetic cannot see: the line was never linked (a fact stated
only in a sentence, a table the build read badly), or one reader's plan is the only reading there is.
Two models reading the same shortlist share those failures.

So a different method reads the same question: the conventional pipeline this repository also
measures as its baseline (baseline/: passages of the filings, BM25 and dense vectors fused, one hosted
model reads them and does its own arithmetic). Alone it is right about half as often as the plan
road. Its mistakes are different ones, and that is what makes it a witness:

  measured on rounds 7 to 9 (23 Nov, dev_scripts/fable/two_roads_sim.py): a FLAGGED answer the
  passage reader repeats was right 28 times of 30, one it does not repeat 31 of 53.

readers.second_road uses it three ways and never serves it as verified:
  the plan road has no answer and no planner says the filings lack it  ->  the passage answer, FLAGGED
  a reading no second planner shares, which the passage reader repeats ->  that reading, CORROBORATED
  a FLAGGED answer the passage reader repeats                          ->  CORROBORATED

Every reply is kept under cache/passage/ keyed by the question, so a run replays with no index, no
embedding model and no key.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Optional

from .execute import Leaf, Result, render
from .llm import DailyLimit

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "cache" / "passage"
VERSION = "passage-2"            # the baseline's v2 retriever in hybrid mode and its prompt, with BUDGET characters
BUDGET = 48_000                  # of filing text per question. The baseline keeps its conventional 16,000; three
                                 # times as much made the reader right on 20, 21 and 22 questions of rounds 9, 7
                                 # and 8 instead of 17, 20 and 20 (23 Nov, results/baseline_r<N>_hybrid_b48.json)
TOLERANCE = 0.005                # two roads round differently: figures within half a percent are the same figure
                                 # (the benchmark's own tolerance for a computed answer). One percent was too
                                 # wide: round 8 V07, 182,186 and 183,417 dollars per employee rest on two
                                 # different head-counts and were called the same
SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
UNITS = {"usd": "USD", "ratio": "ratio", "shares": "shares", "count": "count", "usd/share": "USD/share"}


def _path(question: str, model: Optional[str]) -> pathlib.Path:
    key = hashlib.sha1(f"{VERSION}|{model or ''}|{question}".encode("utf-8")).hexdigest()[:24]
    return DIR / f"{key}.json"


def read(question: str, model: Optional[str] = None, offline: bool = False) -> Optional[dict]:
    """What the passage reader answers: {"abstain", "answer", "pred", "unit", "facts", "facts_ok"},
    or None when this road is not available (offline with no saved reply, no index, no quota)."""
    path = _path(question, model)
    if path.exists():
        _log(path)
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        return None
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from baseline import answer as ba
        out = ba.answer(question, model=model, retriever="v2", mode="hybrid", budget_chars=BUDGET)
    except DailyLimit:
        return None
    except Exception:  # noqa: BLE001 - no index, no embedding model: the first road still answers
        return None
    if out.get("error"):                              # a failed call is not an abstention: ask again next time
        return None
    shown = list(zip(out.get("retrieved_pages") or [], out.get("retrieved_text") or []))
    texts = {}
    for page, text in shown:
        texts[(page["doc"], page["page"])] = texts.get((page["doc"], page["page"]), "") + "\n" + text
    facts = []
    for f in out.get("facts_used") or []:
        f = dict(f)
        if not (f.get("in_context") and _printed(f.get("value"), texts.get((f.get("doc"), f.get("page")), ""))):
            # the reader sometimes writes a passage's number where the page belongs (6 of 66 answers, 23 Nov):
            # the fact is then looked for in the passages of that filing it was shown
            for page, text in shown:
                if page["doc"] == f.get("doc") and _printed(f.get("value"), text):
                    f.update(page=page["page"], in_context=True, page_found_by_figure=True)
                    break
        f["printed"] = bool(f.get("in_context")) and _printed(f.get("value"), texts.get((f.get("doc"), f.get("page")), ""))
        facts.append(f)
    pred = out["pred"]
    reply = {"version": VERSION, "model": model, "question": question, "abstain": pred.get("kind") == "abstain",
             "answer": out.get("answer", ""), "pred": pred, "unit": out.get("unit"), "facts": facts,
             "facts_ok": bool(facts) and all(f.get("in_context") and f["printed"] for f in facts)}
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reply, indent=1, default=str), encoding="utf-8")
    _log(path)
    return reply


def _log(path: pathlib.Path) -> None:
    """Name the saved reply in the run's cache list (TIEOUT_CACHE_LOG), as llm.py does for model calls."""
    log = os.environ.get("TIEOUT_CACHE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"passage/{path.name}\n")


def _printed(value, text: str) -> bool:
    """Are the figure's leading digits printed in the passage it is cited from? The reader gives
    full units ("1.9 trillion" as 1900000000000), so only the significant digits can be compared."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    digits = re.sub(r"\D", "", f"{abs(value):.6f}".rstrip("0")).lstrip("0").rstrip("0")
    return len(digits) >= 1 and digits[:6] in re.sub(r"[^0-9]", "", text)


def head(reply: dict, companies: dict) -> Optional[tuple]:
    """What of a passage answer is compared with a plan's result: ("n", figure), or ("s", the names
    of a ranking in order, as tickers where they are companies of the corpus)."""
    pred = reply.get("pred") or {}
    kind, value = pred.get("kind"), pred.get("value")
    if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("n", float(value))
    if kind == "companies" and value:
        names = [v[0] if isinstance(v, (list, tuple)) else v for v in value]
        return ("s", tuple(_ticker(str(n), companies) for n in names))
    return None


def _ticker(name: str, companies: dict) -> str:
    """The company's ticker when the name is one of the corpus's companies, else the name as written."""
    from .eval.bench import company_to_ticker
    return company_to_ticker(name, companies) or name.strip()


def same(result: Result, shown: Optional[tuple]) -> bool:
    """Does the plan's result say what the passage reader says? A figure within one percent, also when
    the reader left the page's scale on it (23,545 for $23.545 billion) or wrote a percentage for a
    fraction; for a ranking or a list, the same first name."""
    if shown is None or not result.ok:
        return False
    if shown[0] == "s":
        if result.kind not in ("companies", "lines") or not result.value:
            return False
        mine = [str(n).strip().lower() for n, _ in result.value]
        theirs = [str(n).strip().lower() for n in shown[1]]
        if len(mine) <= 2:                            # which of two: the winner is the answer
            return bool(theirs) and theirs[0] == mine[0] and theirs == mine[:len(theirs)]
        return theirs == mine                         # a ranking of three or more: the whole order (round 8, W13:
        #                                               the same first name stood for two different rankings)
    figure = result.value if result.kind == "number" else (
        result.value[0][1] if result.kind == "lines" and result.value else None)
    if isinstance(figure, bool) or not isinstance(figure, (int, float)):
        return False
    x, y = abs(float(figure)), abs(shown[1])
    factors = SCALES + ((100.0,) if result.unit in ("ratio", "percent", "percent_points", "") else ())
    for k in factors:
        for a, b in ((x, y * k), (x * k, y)):
            if abs(a - b) <= TOLERANCE * max(a, b, 1e-12) or _rounds_to(a, b) or _rounds_to(b, a):
                return True
    return False


def _rounds_to(exact: float, coarse: float) -> bool:
    """Is `coarse` the figure `exact` rounded to two or three significant digits, the way a sentence
    states a table's figure ("$6.7 billion" for 6,661 million)? A figure with one significant digit
    (50, 0.3) is too coarse to say, and one with more than three is not a rounding."""
    if exact <= 0 or coarse <= 0:
        return False
    from math import floor, log10
    for digits in (2, 3):
        step = 10 ** (floor(log10(coarse)) - digits + 1)
        if abs(coarse / step - round(coarse / step)) > 1e-6:      # coarse itself has more digits than this
            continue
        if digits == 2 and abs(coarse / (step * 10) - round(coarse / (step * 10))) <= 1e-6:
            return False                                           # one significant digit only
        return abs(round(exact / step) * step - coarse) <= 1e-6 * coarse
    return False


# ----------------------------------------------------------------- the same plan, on facts read from passages
#
# The passage reader answers a whole question well when it is one printed fact, and badly when it needs
# several filings or several steps. A plan already says which facts are needed. So the second road can
# also run a planner's plan: each line the plan names is asked of the passage reader as its own small
# question, and the plan's arithmetic is done here. That gives a third answer, built from the plan road's
# reading and the passage road's facts. readers.second_road uses it as a vote:
#   it equals the passage reader's own answer, against the graph's figure -> the graph read a wrong line
#   it equals the graph's figure                                          -> the lines are corroborated

ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}
MAX_LEAVES = 6                   # a plan that needs more lines than this is not run on passages


def fact_question(g, ex, ticker: str, words: str, spec: dict) -> Optional[str]:
    """One line of a plan as a question for the passage reader."""
    if ticker not in g.companies or not words or not isinstance(spec, dict):
        return None
    try:
        year = ex._fy(ticker, spec)
    except Exception:  # noqa: BLE001 - a period the executor cannot resolve
        return None
    quarter = spec.get("q")
    if quarter and spec.get("ytd"):
        when = f"the first {3 * int(quarter)} months of fiscal year {year} (year to date)"
    elif quarter:
        when = f"the {ORDINALS.get(int(quarter), str(quarter))} quarter of fiscal year {year}"
    else:
        when = f"fiscal year {year} (for a balance: at the end of fiscal year {year})"
    return f"What was {words} of {g.companies[ticker]['name']} for {when}? Give the one figure the filing prints for it."


def read_fact(g, ex, ticker: str, words: str, spec: dict, model: Optional[str], offline: bool) -> Optional[tuple]:
    """(figure, the reader's reply) for one line of a plan, or None (no answer, or its fact is not on
    the page it cites)."""
    question = fact_question(g, ex, ticker, words, spec)
    if question is None:
        return None
    reply = read(question, model=model, offline=offline)
    if not reply or reply.get("abstain") or not reply.get("facts_ok"):
        return None
    value = (reply.get("pred") or {}).get("value")
    if (reply.get("pred") or {}).get("kind") != "number" or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value), reply


def _words(step: dict) -> Optional[str]:
    """The line a get or change step names, in words: its item phrase, or its concept's name spelled out.
    A metric, a breakdown or a ranking is not one printed line: None."""
    if any(k in step for k in ("metric", "by", "of", "over", "where")):
        return None
    if isinstance(step.get("item"), str):
        return step["item"]
    if isinstance(step.get("concept"), str):
        return re.sub(r"(?<!^)(?=[A-Z])", " ", step["concept"]).lower()
    return None


def _calc(expr: str, values: dict) -> Optional[float]:
    """The plan language's arithmetic (+ - * / ** abs, earlier steps by id), nothing else."""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
            return abs(ev(node.args[0]))
        raise ValueError("not plan arithmetic")
    try:
        return float(ev(ast.parse(str(expr), mode="eval")))
    except Exception:  # noqa: BLE001 - a division by zero, a name that is not a step, anything else
        return None


def run_plan(plan: dict, g, ex, model: Optional[str] = None, offline: bool = False) -> Optional[dict]:
    """The plan's answer when every line it names is read from passages: {"value", "leaves"} or None
    when the plan has a step this road cannot run (a ranking, a breakdown, a metric), a line the
    passage reader does not give, or more than MAX_LEAVES lines."""
    steps, answer = (plan or {}).get("steps"), (plan or {}).get("answer")
    if not isinstance(steps, list) or not steps or not isinstance(answer, str):
        return None
    needed = sum(2 if s.get("op") == "change" else 1 for s in steps if isinstance(s, dict) and s.get("op") in ("get", "change"))
    if needed == 0 or needed > MAX_LEAVES:
        return None
    values: dict = {}
    leaves = []
    for s in steps:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str):
            return None
        op = s.get("op")
        if op == "calc":
            v = _calc(s.get("expr"), values)
        elif op in ("get", "change"):
            words = _words(s)
            if words is None or s.get("as"):
                return None
            specs = [s.get("period")] if op == "get" else [s.get("from"), s.get("to")]
            read_ = [read_fact(g, ex, s.get("company"), words, spec, model, offline) for spec in specs]
            if any(x is None for x in read_):
                return None
            got = [x[0] for x in read_]
            for _, fact_reply in read_:
                leaves += fact_reply.get("facts") or []
            if op == "get":
                v = got[0]
            elif s.get("mode") == "pct":
                v = (got[1] - got[0]) / abs(got[0]) if got[0] else None
            else:
                v = got[1] - got[0]
        else:
            return None
        if v is None:
            return None
        values[s["id"]] = v
    return {"value": values[answer], "leaves": leaves} if answer in values else None


def same_number(x, y) -> bool:
    """Two figures of the passage road, or one of each road, to the tolerance of same()."""
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    return same(Result("number", float(x), ""), ("n", float(y)))


def cited(reply: dict) -> str:
    """"DOC page N" of the first fact the passage reader used."""
    for f in reply.get("facts") or []:
        if f.get("doc"):
            return f"{f['doc']} page {f.get('page')}"
    return "its passages"


def _leaves(facts: list) -> list:
    """The passage reader's cited facts as proof leaves (unverified: nothing in the graph ties them)."""
    return [Leaf(id=f"{f.get('doc')}:p{f.get('page')}:passage{i}", ticker=str(f.get("ticker") or ""),
                 what="passage:" + str(f.get("label") or "")[:60], period="", doc=str(f.get("doc") or ""),
                 page=int(f.get("page") or 0), label=str(f.get("label") or ""), printed=str(f.get("value")),
                 status="unverified", value=f.get("value"))
            for i, f in enumerate(facts or [])]


def plan_answer(ran: dict, plan: dict) -> dict:
    """The answer of a plan run on facts read from passages (run_plan), in the shape ask.answer returns."""
    last = next((s for s in plan.get("steps") or [] if isinstance(s, dict) and s.get("id") == plan.get("answer")), {})
    a_rate = last.get("mode") == "pct" or (last.get("op") == "calc" and "/" in str(last.get("expr")))
    res = Result("number", float(ran["value"]), "ratio" if a_rate else "", _leaves(ran.get("leaves")))
    res.tier = "FLAGGED"
    res.tier_reasons = ["the graph had no line for this plan; each line the plan names was read from passages of the "
                        "filings by one model, and the plan's arithmetic was done on those figures. Not tied by arithmetic"]
    return {"answer": render(res), "result": res, "plan": plan, "learned": [], "seconds": 0.0, "plan_errors": [],
            "repaired": False}


def as_answer(reply: dict, companies: dict) -> dict:
    """The passage answer in the shape ask.answer returns, FLAGGED: one model read it from passages,
    nothing in the graph vouches for it."""
    pred = reply["pred"]
    kind, value = pred.get("kind"), pred.get("value")
    leaves = _leaves(reply.get("facts"))
    if kind == "number":
        res = Result("number", float(value), UNITS.get(str(reply.get("unit") or "").lower(), ""), leaves)
    elif kind == "companies":
        names = [str(v[0] if isinstance(v, (list, tuple)) else v) for v in value]
        res = Result("companies", [(_ticker(n, companies), None) for n in names], "", leaves)
    else:
        res = Result("text", str(value), "", leaves)
    res.tier = "FLAGGED"
    res.tier_reasons = ["no line of the graph answered this; one model read it from passages of the filing "
                        f"({cited(reply)}), where the figure is printed. It is not tied by arithmetic"]
    try:
        text = render(res)
    except Exception:  # noqa: BLE001 - a list without figures: the reader's own words
        text = str(reply.get("answer") or value)
    if kind != "number":
        text = str(reply.get("answer") or text)
    return {"answer": text, "result": res, "plan": {"steps": [], "answer": None, "road": "passage"}, "learned": [],
            "seconds": 0.0, "plan_errors": [], "repaired": False}
