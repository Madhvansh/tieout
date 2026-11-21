"""The plan language: what a question is compiled into.

A plan is a flat list of typed steps. Each step names graph nodes (company,
concept, period, attribute) and an operation; later steps refer to earlier
ones by id. The planner (an LLM) writes the plan from the question and the
vocabulary below. It never sees a number and cannot write one into an
answer: numbers only enter through `get`, which reads the graph.

Operations
    get       one fact: company, concept, period
    metric    a derived metric for one company and period
    calc      arithmetic over earlier steps, e.g. "s1 / s2"
    change    growth of a concept or metric between two periods (abs or pct)
    rank      order a set of companies by a concept or metric, keep the top k
    filter    keep the companies whose attribute or metric meets a condition
    attr      a text fact of one company (auditor, state of incorporation, ...)
    count     size of a company set
    restated  first-reported value against the latest-reported value

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
OPS = ("get", "metric", "calc", "change", "rank", "filter", "attr", "count", "restated")
TEXT_ATTRS = ("auditor", "auditor_since", "auditor_location", "state_of_incorporation", "fiscal_year_end",
              "hq_city", "hq_state", "exchange", "trading_symbol", "employees", "shares_outstanding",
              "public_float_usd", "ceo", "cfo", "cluster")
NUMERIC_ATTRS = {"employees": "employees", "shares_outstanding": "shares", "public_float_usd": "USD",
                 "auditor_since": "year"}
CLUSTERS = ("big_tech", "banks_finance", "consumer_retail", "health", "industrial_auto_energy")

_METRICS: Optional[dict] = None


def metric_defs() -> dict:
    global _METRICS
    if _METRICS is None:
        raw = yaml.safe_load((ROOT / "semantics" / "metrics.yaml").read_text(encoding="utf-8"))["metrics"]
        _METRICS = {m["name"]: m for m in raw}
    return _METRICS


def _says() -> dict:
    from .living import SAYS                         # one-line meaning of each concept
    return SAYS


def vocabulary(graph) -> dict:
    """Everything a planner is allowed to name. This, not the data, is what the LLM sees."""
    return {
        "companies": {t: c["name"] for t, c in sorted(graph.companies.items())},
        "clusters": list(CLUSTERS),
        "concepts": {n: _says().get(n, f"{d.statement}, {d.nature}") for n, d in concept_defs().items()},
        "metrics": {n: m["says"] for n, m in metric_defs().items()},
        "attributes": list(TEXT_ATTRS),
        "ops": list(OPS),
    }


def validate(plan: dict, graph) -> list:
    """Static checks before anything runs. Returns a list of error strings."""
    errs = []
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["plan has no steps"]
    concepts, metrics = concept_defs(), metric_defs()
    seen = set()
    for s in steps:
        sid, op = s.get("id"), s.get("op")
        if not sid or sid in seen:
            errs.append(f"step id missing or repeated: {sid}")
        if op not in OPS:
            errs.append(f"{sid}: unknown op {op}")
            continue
        comp = s.get("company")
        if isinstance(comp, str) and not comp.startswith("$") and comp not in graph.companies:
            errs.append(f"{sid}: unknown company {comp}")
        if isinstance(comp, str) and comp.startswith("$") and comp[1:] not in seen:
            errs.append(f"{sid}: refers to {comp} before it is defined")
        for key in ("concept",):
            if s.get(key) and s[key] not in concepts:
                errs.append(f"{sid}: unknown concept {s[key]}")
        if s.get("metric") and s["metric"] not in metrics:
            errs.append(f"{sid}: unknown metric {s['metric']}")
        by = s.get("by") or (s.get("where") or {}).get("by")
        if isinstance(by, dict):
            if by.get("concept") and by["concept"] not in concepts:
                errs.append(f"{sid}: unknown concept {by['concept']}")
            if by.get("metric") and by["metric"] not in metrics:
                errs.append(f"{sid}: unknown metric {by['metric']}")
        if op in ("get", "restated") and not (s.get("concept") and comp and s.get("period")):
            errs.append(f"{sid}: {op} needs company, concept and period")
        if op == "metric" and not (s.get("metric") and comp and s.get("period")):
            errs.append(f"{sid}: metric needs company, metric and period")
        if op == "attr" and s.get("attr") not in TEXT_ATTRS:
            errs.append(f"{sid}: unknown attribute {s.get('attr')}")
        if op == "calc" and not s.get("expr"):
            errs.append(f"{sid}: calc needs expr")
        if op == "change" and not (s.get("from") and s.get("to") and (s.get("concept") or s.get("metric"))):
            errs.append(f"{sid}: change needs from, to and a concept or metric")
        if op == "rank" and not (s.get("by") and s.get("period")):
            errs.append(f"{sid}: rank needs by and period")
        over = s.get("over")
        if isinstance(over, str) and over.startswith("$") and over[1:] not in seen:
            errs.append(f"{sid}: refers to {over} before it is defined")
        if isinstance(over, dict) and over.get("cluster") not in CLUSTERS:
            errs.append(f"{sid}: unknown cluster {over.get('cluster')}")
        if isinstance(over, list):
            errs.extend(f"{sid}: unknown company {t}" for t in over if t not in graph.companies)
        seen.add(sid)
    if plan.get("answer") not in seen:
        errs.append(f"answer refers to unknown step {plan.get('answer')}")
    return errs
