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

Operations
    get       one measure for one company and period
    metric    shorthand for get with a metric
    calc      arithmetic over earlier steps, e.g. "s1 / s2"
    change    growth of a measure between two named periods (pct or abs)
    rank      order a set by a measure, keep the top k
    filter    keep the members of a set that meet a condition
    agg       sum, average, median, minimum or maximum of a measure over a set
    count     size of a set
    series    one measure over several periods, optionally the period where it peaks
    members   the lines that add up to a total (segments, products, regions)
    attr      a text fact of one company (auditor, state of incorporation, ...)
    restated  first-reported against latest-reported value, with the structural reason

Sets
    "all" | {"cluster": name} | [tickers] | "$s1" (what an earlier step kept)

Periods
    {"fy": 2025}                      fiscal year as that company labels it
    {"fy": 2025, "q": 3}              a fiscal quarter
    {"fy": 2025, "q": 3, "ytd": true} year to date at that quarter
    {"fy": "latest"} / {"fy": "prior"}
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from .concepts import concept_defs

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPS = ("get", "metric", "calc", "change", "rank", "filter", "agg", "count", "series", "members", "attr", "restated")
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
        sid, op = s.get("id"), s.get("op")
        if not sid or sid in seen:
            errs.append(f"step id missing or repeated: {sid}")
        if op not in OPS:
            errs.append(f"{sid}: unknown op {op}")
            seen.add(sid)
            continue
        company = s.get("company")
        ref_ok(company, sid)
        if isinstance(company, str) and not company.startswith("$") and company not in graph.companies:
            errs.append(f"{sid}: unknown company {company}")
        over = s.get("over")
        ref_ok(over, sid)
        if isinstance(over, dict) and over.get("cluster") not in CLUSTERS:
            errs.append(f"{sid}: unknown cluster {over.get('cluster')}")
        if isinstance(over, list):
            errs.extend(f"{sid}: unknown company {t}" for t in over if t not in graph.companies)
        measure = step_measure(s)
        if op in ("get", "metric", "change", "rank", "agg", "series", "restated"):
            errs.extend(measure_errors(measure, sid))
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
        if op == "calc" and not s.get("expr"):
            errs.append(f"{sid}: calc needs expr")
        if op == "count":
            ref_ok(s.get("of"), sid)
            if not s.get("of"):
                errs.append(f"{sid}: count needs of")
        seen.add(sid)
    if plan.get("answer") not in seen:
        errs.append(f"answer refers to unknown step {plan.get('answer')}")
    return errs
