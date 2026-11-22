"""The plan language: what a question is compiled into.

A plan is a flat list of typed steps. Each step names graph nodes (company,
concept, period, attribute) and an operation; later steps refer to earlier
ones by id. The planner (a language model) writes the plan from the question
and the vocabulary below. It never sees a number and cannot write one into an
answer: numbers enter only through steps that read the graph.

Measures (what can be read for one company and one period)
    {"concept": C}                    a concept shared across companies
    {"metric": M}                     a derived metric (a formula over concepts)
    {"item": "words"}                 a line of any table, or a figure stated in a sentence,
                                      linked to the graph the first time a question asks for it
    {"attr": A}                       a numeric attribute (employees, shares outstanding)
    {"growth": measure}               percentage change into the period from the year before
    {"delta": measure}                the same change as an amount
    {"ratio": [measure, measure]}     one measure divided by another, same period
    {"value": true}                   the printed value of a line (inside rank, filter, agg over lines)

Operations
    get       one measure for one company and period; "as": "first_reported" reads the
              figure as its first filing printed it, before any later revision
    metric    shorthand for get with a metric
    calc      arithmetic over earlier steps, e.g. "s1 / s2", "abs(s1) / s2"
    change    growth of a measure between two named periods (pct or abs)
    rank      order a set by a measure, keep the top k; over two named companies the
              answer also carries the gap between them
    compare   order figures worked out in earlier steps (one company's segment margin against
              another's): the same answer as rank, for sides that each need their own steps
    filter    keep the members of a set that meet a condition
    agg       sum, average, median, minimum or maximum of a measure over a set
    count     size of a set
    series    one measure over several periods, optionally the period where it peaks
    members   the lines that add up to a total (segments, products, regions)
    attr      a text fact of one company (auditor, state of incorporation, ...)
    restated  first-reported against latest-reported value, with the structural reason

Sets
    "all" | {"cluster": name} | [tickers] | "$s1" (what an earlier step kept: companies or lines)

References
    "company": "$s1" names the top member of an earlier rank or filter step: a company, or,
    when that step ranked the lines of a members step, that line (get then reads its value)

Periods
    {"fy": 2025}                      fiscal year as that company labels it
    {"fy": 2025, "q": 3}              a fiscal quarter
    {"fy": 2025, "q": 3, "ytd": true} year to date at that quarter
    {"fy": "latest"} / {"fy": "prior"}
"""
from __future__ import annotations

import ast
import pathlib
import re
from typing import Optional

import yaml

from .concepts import concept_defs

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPS = ("get", "metric", "calc", "change", "rank", "compare", "filter", "agg", "count", "series", "members", "attr",
       "restated")
TEXT_ATTRS = ("auditor", "auditor_since", "auditor_location", "state_of_incorporation", "fiscal_year_end",
              "hq_city", "hq_state", "exchange", "trading_symbol", "employees", "shares_outstanding",
              "public_float_usd", "ceo", "cfo", "cluster")
NUMERIC_ATTRS = {"employees": "employees", "shares_outstanding": "shares", "public_float_usd": "USD",
                 "auditor_since": "year"}
CLUSTERS = ("big_tech", "banks_finance", "consumer_retail", "health", "industrial_auto_energy")
AGG_FNS = ("sum", "avg", "median", "min", "max")
COMPARATORS = ("gt", "lt", "ge", "le")

_SEMANTICS: Optional[dict] = None


def _semantics() -> dict:
    global _SEMANTICS
    if _SEMANTICS is None:
        _SEMANTICS = yaml.safe_load((ROOT / "semantics" / "metrics.yaml").read_text(encoding="utf-8"))
    return _SEMANTICS


def metric_defs() -> dict:
    """Derived metrics by name: formula over concepts, unit, one-line meaning."""
    return {m["name"]: m for m in _semantics()["metrics"]}


def identity_defs() -> list:
    """Accounting identities as {concept: coefficient} maps whose terms sum to zero."""
    return list(_semantics().get("identities", []))


def vocabulary(graph) -> dict:
    """Everything a planner is allowed to name. This, not the data, is what the LLM sees."""
    from .living import SAYS                         # one-line meaning of each concept
    return {
        "companies": {t: c["name"] for t, c in sorted(graph.companies.items())},
        "clusters": list(CLUSTERS),
        "concepts": {n: SAYS.get(n, f"{d.statement}, {d.nature}") for n, d in concept_defs().items()},
        "metrics": {n: m["says"] for n, m in metric_defs().items()},
        "attributes": list(TEXT_ATTRS),
        "ops": list(OPS),
    }


def step_measure(step: dict) -> Optional[dict]:
    """The measure a step reads, whichever way the step spells it.

    `{"op": "get", "concept": "Revenue"}` and `{"op": "get", "by": {"concept": "Revenue"}}`
    mean the same thing; plans written for the first version of the language use the former.
    """
    if isinstance(step.get("by"), dict):
        return step["by"]
    for key in ("concept", "metric", "item", "attr"):
        if step.get(key) and not (key == "attr" and step.get("op") == "attr"):
            return {key: step[key]}
    return None


def _key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _as_measure(m):
    """A measure written loosely, made exact from the vocabulary when there is one clear reading:
    "net_income" or {"metric": "inventory"} name a concept; {"delta": "NetIncome"} wraps it.
    Anything unclear is left as it was, for the validator to report."""
    concepts, metrics = concept_defs(), metric_defs()
    by_key = {**{_key(c): {"concept": c} for c in concepts}, **{_key(n): {"metric": n} for n in metrics}}
    if isinstance(m, str):
        return dict(by_key.get(_key(m), {"item": m}))
    if not isinstance(m, dict):
        return m
    for wrap in ("growth", "delta"):
        if wrap in m:
            return {wrap: _as_measure(m[wrap])}
    if "ratio" in m and isinstance(m["ratio"], list):
        return {"ratio": [_as_measure(x) for x in m["ratio"]]}
    if "concept" in m and m["concept"] not in concepts:
        key = _key(m["concept"])
        if key in by_key:
            return dict(by_key[key])
        attr = next((a for a in NUMERIC_ATTRS if _key(a) == key), None)
        if attr:
            return {"attr": attr}                        # "employees" is an attribute, not a concept
        # a word that is not in the concept list names a narrower line: an item, in the planner's own words
        return {"item": re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", str(m["concept"])).strip()}
    if "metric" in m and m["metric"] not in metrics and _key(m["metric"]) in by_key:
        return dict(by_key[_key(m["metric"])])
    return m


def normalize(plan: dict) -> dict:
    """Fix the small slips a planner makes without changing what the plan asks.

    Deterministic and vocabulary-only: a measure spelled as a string or with the wrong case, a
    concept named as a metric (or the reverse), a step that reads a line picked earlier but
    forgot the period (it takes the period of the step it refers to). No value is involved.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return plan
    steps, period_of = [], {}
    for s in plan["steps"]:
        if not isinstance(s, dict):
            steps.append(s)
            continue
        s = dict(s)
        if isinstance(s.get("over"), str) and s["over"] != "all" and not s["over"].startswith("$"):
            s["over"] = [s["over"]]                                     # one ticker given as a set
        where = s.get("where")
        if (s.get("op") in ("get", "change", "restated") and isinstance(where, dict) and isinstance(s.get("item"), str)
                and not any(k in where for k in ("gt", "lt", "ge", "le", "by", "attr"))):
            # a read never filters: words put in a "where" belong to the line asked for ("stores in California")
            s["item"] = (s["item"] + " in " + " ".join(str(v) for v in where.values())).strip()
            del s["where"]
        if s.get("op") == "calc" and isinstance(s.get("expr"), str):
            s["expr"] = re.sub(r"\$(s\d+)\b", r"\1", s["expr"])          # "$s1 / $s2" -> "s1 / s2"
        if isinstance(s.get("by"), (str, dict)):
            s["by"] = _as_measure(s["by"])
        if isinstance(s.get("where"), dict) and "by" in s["where"]:
            s["where"] = {**s["where"], "by": _as_measure(s["where"]["by"])}
        for key in ("concept", "metric"):
            if isinstance(s.get(key), str) and s.get("op") != "attr":
                fixed = _as_measure({key: s[key]})
                if fixed != {key: s[key]}:
                    del s[key]
                    if s.get("op") == "metric" and "concept" in fixed:
                        s["op"] = "get"
                    if s.get("op") == "get" and "metric" in fixed:
                        s["op"] = "metric"
                    if "item" in fixed and isinstance(s.get("item"), str) and s["item"].strip():
                        continue                                        # the step already words its item
                    s.update(fixed)
        if s.get("as") not in (None, "first_reported", "latest"):
            del s["as"]                                                 # not a way of reading the language has
        ref = s.get("company") if isinstance(s.get("company"), str) and s["company"].startswith("$") else None
        if s.get("op") in ("get", "metric", "members", "restated") and not s.get("period") and ref:
            if ref[1:] in period_of:
                s["period"] = period_of[ref[1:]]
        if s.get("id") and isinstance(s.get("period"), dict):
            period_of[s["id"]] = s["period"]
        steps.append(s)
    answer = plan.get("answer")
    last = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    if (last.get("op") == "compare" and isinstance(last.get("of"), list) and len(set(map(str, last["of"]))) == 1
            and answer == last.get("id") and len(steps) > 1):
        steps, answer = steps[:-1], str(last["of"][0]).lstrip("$")     # a ranking compared with itself: the ranking
        last = steps[-1] if isinstance(steps[-1], dict) else {}
    if isinstance(answer, str) and last.get("id") not in (None, answer) and f"${answer}" in (last.get("company"), last.get("of"),
                                                                                             last.get("over")):
        answer = last["id"]                           # the last step asks about what the answer step found: it is the answer
    return {**plan, "steps": steps, "answer": answer}


_CALC_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.Call, ast.Name, ast.Load, ast.Constant)


def calc_errors(expr, defined: set, sid: str) -> list:
    """Why a calc expression cannot run: not arithmetic, or it names a step that is not defined before it."""
    if not isinstance(expr, str) or not expr.strip():
        return [f"{sid}: calc needs expr"]
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return [f"{sid}: calc expr is not arithmetic over step ids (write s1 / s2, not $s1 / $s2)"]
    errs = []
    for node in ast.walk(tree):
        if not isinstance(node, _CALC_NODES):
            errs.append(f"{sid}: calc expr may only use + - * / ( ), numbers, step ids and abs()")
            break
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") != "abs":
            errs.append(f"{sid}: the only function a calc may call is abs()")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            errs.append(f"{sid}: calc constants must be numbers")
        if isinstance(node, ast.Name) and node.id != "abs" and node.id not in defined:
            errs.append(f"{sid}: calc uses {node.id}, which is not a step defined before it")
    return errs


def measure_errors(m, where: str) -> list:
    """Why `m` is not a well-formed measure (empty list if it is)."""
    if not isinstance(m, dict) or not m:
        return [f"{where}: a measure must be an object such as {{\"concept\": \"Revenue\"}}"]
    concepts, metrics = concept_defs(), metric_defs()
    if "growth" in m or "delta" in m:
        return measure_errors(m.get("growth") or m.get("delta"), where)
    if "ratio" in m:
        parts = m["ratio"]
        if not (isinstance(parts, list) and len(parts) == 2):
            return [f"{where}: ratio needs a list of two measures"]
        return measure_errors(parts[0], where) + measure_errors(parts[1], where)
    if "value" in m:
        return []                                    # the printed value of a line from a members step
    if "concept" in m:
        return [] if m["concept"] in concepts else [f"{where}: unknown concept {m['concept']}"]
    if "metric" in m:
        return [] if m["metric"] in metrics else [f"{where}: unknown metric {m['metric']}"]
    if "attr" in m:
        return [] if m["attr"] in NUMERIC_ATTRS else [f"{where}: attribute {m['attr']} is not a number"]
    if "item" in m:
        return [] if isinstance(m["item"], str) and m["item"].strip() else [f"{where}: item needs words"]
    return [f"{where}: a measure needs one of concept, metric, item, attr, growth, delta, ratio"]


def validate(plan: dict, graph) -> list:
    """Static checks before anything runs. Returns a list of error strings."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["plan has no steps"]
    errs: list = []
    seen: set = set()

    def ref_ok(ref, sid: str) -> None:
        if isinstance(ref, str) and ref.startswith("$") and ref[1:] not in seen:
            errs.append(f"{sid}: refers to {ref} before it is defined")

    for s in steps:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str):
            errs.append("every step is an object with a string id such as \"s1\"")
            continue
        sid, op = s.get("id"), s.get("op")
        if not sid or sid in seen:
            errs.append(f"step id missing or repeated: {sid}")
        if op not in OPS:
            errs.append(f"{sid}: unknown op {op}")
            seen.add(sid)
            continue
        company = s.get("company")
        if company is not None and not isinstance(company, str):
            errs.append(f"{sid}: company is one ticker or \"$sN\", never a list; for several companies use rank, "
                        "filter or agg over a set")
            company = None
        ref_ok(company, sid)
        if isinstance(company, str) and not company.startswith("$") and company not in graph.companies:
            errs.append(f"{sid}: unknown company {company}")
        over = s.get("over")
        ref_ok(over, sid)
        if isinstance(over, dict) and over.get("cluster") not in CLUSTERS:
            errs.append(f"{sid}: unknown cluster {over.get('cluster')}")
        if isinstance(over, list):
            if not all(isinstance(t, str) for t in over):
                errs.append(f"{sid}: a set lists tickers, such as [\"AMZN\", \"GOOGL\"]; to compare figures "
                            "worked out in earlier steps use compare")
            else:
                errs.extend(f"{sid}: unknown company {t}" for t in over if t not in graph.companies)
        measure = step_measure(s)
        if (op in ("rank", "filter", "agg") and "value" in str(measure or (s.get("where") or {}).get("by") or "")
                and not (isinstance(over, str) and over.startswith("$"))):
            errs.append(f"{sid}: {{\"value\": true}} is the figure of a line from a members step; to order figures "
                        "worked out in earlier steps use compare with \"of\": [their step ids]")
        line_ref = op == "get" and measure is None and isinstance(company, str) and company.startswith("$")
        if op in ("get", "metric", "change", "rank", "agg", "series", "restated") and not line_ref:
            errs.extend(measure_errors(measure, sid))
        if s.get("as") not in (None, "first_reported", "latest"):
            errs.append(f"{sid}: \"as\" is \"first_reported\" or \"latest\"")
        if op in ("get", "metric", "restated", "members") and not (company and s.get("period")):
            errs.append(f"{sid}: {op} needs company and period")
        if op == "change" and not (company and s.get("from") and s.get("to")):
            errs.append(f"{sid}: change needs company, from and to")
        if op in ("rank", "agg") and not s.get("period"):
            errs.append(f"{sid}: {op} needs period")
        if op == "agg" and s.get("fn") not in AGG_FNS:
            errs.append(f"{sid}: agg needs fn, one of {', '.join(AGG_FNS)}")
        if op == "filter":
            w = s.get("where") or {}
            if "attr" in w and "eq" in w:
                if w["attr"] not in TEXT_ATTRS:
                    errs.append(f"{sid}: unknown attribute {w['attr']}")
            elif isinstance(w.get("by"), dict) and any(k in w for k in COMPARATORS):
                errs.extend(measure_errors(w["by"], sid))
                if not (w.get("period") or s.get("period")) and "value" not in w["by"]:
                    errs.append(f"{sid}: filter needs a period")
            else:
                errs.append(f"{sid}: filter needs where {{attr, eq}} or where {{by, period, gt|lt|ge|le}}")
        if op == "series" and not (company and (s.get("periods") or s.get("last"))):
            errs.append(f"{sid}: series needs company and periods or last")
        if op == "members" and not s.get("of"):
            errs.append(f"{sid}: members needs of (the total whose parts are wanted)")
        if op == "attr" and s.get("attr") not in TEXT_ATTRS:
            errs.append(f"{sid}: unknown attribute {s.get('attr')}")
        if op == "calc":
            errs.extend(calc_errors(s.get("expr"), seen, sid))
        if op == "compare":
            of = s.get("of")
            if not (isinstance(of, list) and len(of) >= 2 and all(isinstance(i, str) for i in of)):
                errs.append(f"{sid}: compare needs of, a list of two or more earlier step ids")
            else:
                errs.extend(f"{sid}: compares {i}, which is not a step defined before it"
                            for i in of if i.lstrip("$") not in seen)
                kinds = {x["id"]: x.get("op") for x in steps if isinstance(x, dict) and isinstance(x.get("id"), str)}
                if len(set(of)) < len(of) or any(kinds.get(i.lstrip("$")) in ("rank", "filter", "members", "attr") for i in of):
                    errs.append(f"{sid}: compare takes two or more different steps that each give one figure (get, "
                                "metric, calc, change); a rank over the companies already compares them")
        if op == "count":
            ref_ok(s.get("of"), sid)
            if not s.get("of"):
                errs.append(f"{sid}: count needs of")
        seen.add(sid)
    if plan.get("answer") not in seen:
        errs.append(f"answer refers to unknown step {plan.get('answer')}")
    return errs
