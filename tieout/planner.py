"""Question to plan.

The planner is the only place a language model touches a question. It is
given the graph's vocabulary (companies, concepts, metrics, attributes,
operations) and writes a plan in that vocabulary. It is not given any data,
so it has nothing to copy a number from. The plan is validated before it
runs; a plan that fails validation is sent back once with the errors.

Plans are cached by content hash (see llm.py), so the benchmark can be
re-run offline and gives the same result every time.
"""
from __future__ import annotations

import json
from typing import Optional

from .llm import ask_json
from .plan import validate, vocabulary

PROMPT_VERSION = "p6"

RULES = """You translate a question about US public companies into a query plan over a knowledge graph
of facts taken from their SEC filings. Output JSON only: {"steps": [...], "answer": "<step id>"}.

You never state a number yourself. Numbers come only from steps that read the graph.

OPERATIONS (each step has "id" like "s1" and "op")
- get      {"op":"get","company":T,"concept":C,"period":P}             one reported fact
- metric   {"op":"metric","company":T,"metric":M,"period":P}           a derived metric
- calc     {"op":"calc","expr":"s1 - s2"}                              arithmetic on earlier steps: + - * / ( )
- change   {"op":"change","company":T,"concept":C or "metric":M,"from":P,"to":P,"mode":"pct"|"abs"}
- rank     {"op":"rank","over":SET,"by":{"concept":C} or {"metric":M},"period":P,"order":"desc"|"asc","k":N}
- filter   {"op":"filter","over":SET,"where":{"attr":A,"eq":"value"}}
           or "where":{"by":{"concept":C} or {"metric":M},"period":P,"gt"|"lt"|"ge"|"le": number}
- attr     {"op":"attr","company":T,"attr":A}                          a text fact (auditor, state, employees ...)
- count    {"op":"count","of":"$s1"}                                   how many companies a filter or rank kept
- restated {"op":"restated","company":T,"concept":C,"period":P}        latest-reported minus first-reported value

T is a ticker from the list, or "$s1" meaning the top company of an earlier rank or filter step.
SET is "all", {"cluster": name}, a list of tickers, or "$s1" (the companies an earlier step kept).
P is {"fy": 2025} (the fiscal year as that company labels it), {"fy": "latest"}, {"fy": "prior"},
  {"fy": 2025, "q": 3} for a fiscal quarter, or {"fy": 2025, "q": 3, "ytd": true} for year to date.

RULES
1. Use only the tickers, concepts, metrics and attributes listed below. If the question needs something
   that is not listed, output {"steps": [], "answer": null, "cannot": "<what is missing>"}.
2. A ratio, margin, growth rate or difference is never a `get`: use metric, change or calc.
3. "Which company has the highest/lowest ..." is a rank with k = 1. "Top three" is k = 3.
4. "How many companies ..." is a filter followed by count.
5. A question that hops (find a company, then ask something about it) uses "$sN" as the company.
6. If no year is given use {"fy": "latest"}. "Fiscal 2025", "FY2025" and "in 2025" all mean {"fy": 2025}.
7. Amounts in the question (thresholds) are written in full units: 10 billion dollars is 10000000000.
   Percent thresholds are fractions: 20 percent is 0.2.
8. "How much more/less was A's X than B's" asks for an amount: get each value and finish with calc.
   "Which of A or B had the higher X" (with or without "and by how much") asks for a company:
   rank over the two tickers with k = 2, so the answer names the winner and shows both values.
9. Balance-sheet items use the same {"fy": N}: the graph takes the value at the end of that fiscal year.
10. You may use general knowledge to work out WHICH listed company a question means (a brand it owns, a
    product, a nickname, "Google's parent"). You may not use general knowledge for any figure or fact.
11. Choose the closest listed concept even when the question words it differently (for example
    "depreciation, amortization and other" is DepreciationAmortization; "bottom line" is NetIncome).
    Use "cannot" only when nothing listed is close.
12. Use a metric only when the question asks for exactly what the metric means. When the question
    names the line items itself ("dividends as a share of earnings", "R&D relative to gross profit"),
    get each item and combine them with calc. "Cash dividends" alone is DividendsPaid, never the
    dividends-plus-buybacks metric.
"""

EXAMPLES = [
    ("What was Apple's total revenue in fiscal 2026?",
     {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026}}], "answer": "s1"}),
    ("What was Amazon's gross margin in 2025?",
     {"steps": [{"id": "s1", "op": "metric", "company": "AMZN", "metric": "gross_margin", "period": {"fy": 2025}}], "answer": "s1"}),
    ("By what percentage did Nvidia's revenue grow in its most recent fiscal year?",
     {"steps": [{"id": "s1", "op": "change", "company": "NVDA", "concept": "Revenue", "from": {"fy": "prior"}, "to": {"fy": "latest"}, "mode": "pct"}], "answer": "s1"}),
    ("How much larger were JPMorgan's total assets than Bank of America's at the end of 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "JPM", "concept": "TotalAssets", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "BAC", "concept": "TotalAssets", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 - s2"}], "answer": "s3"}),
    ("Which was more profitable in fiscal 2025, Visa or Mastercard, and by how much?",
     {"steps": [{"id": "s1", "op": "rank", "over": ["V", "MA"], "by": {"concept": "NetIncome"}, "period": {"fy": 2025}, "order": "desc", "k": 2}], "answer": "s1"}),
    ("Which bank or financial company had the highest net income in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "banks_finance"}, "by": {"concept": "NetIncome"}, "period": {"fy": 2025}, "order": "desc", "k": 1}], "answer": "s1"}),
    ("Who audits the company with the largest research and development spending among the big tech companies?",
     {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "big_tech"}, "by": {"concept": "ResearchAndDevelopment"}, "period": {"fy": "latest"}, "order": "desc", "k": 1},
                {"id": "s2", "op": "attr", "company": "$s1", "attr": "auditor"}], "answer": "s2"}),
    ("How many of the companies audited by Deloitte had an operating margin above 20 percent in their latest fiscal year?",
     {"steps": [{"id": "s1", "op": "filter", "over": "all", "where": {"attr": "auditor", "eq": "Deloitte"}},
                {"id": "s2", "op": "filter", "over": "$s1", "where": {"by": {"metric": "operating_margin"}, "period": {"fy": "latest"}, "gt": 0.2}},
                {"id": "s3", "op": "count", "of": "$s2"}], "answer": "s3"}),
    ("What was Alphabet's revenue in the fourth quarter of 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "GOOGL", "concept": "Revenue", "period": {"fy": 2025, "q": 4}}], "answer": "s1"}),
    ("Did Coca-Cola restate its 2024 operating income in the following year's filing, and by how much?",
     {"steps": [{"id": "s1", "op": "restated", "company": "KO", "concept": "OperatingIncome", "period": {"fy": 2024}}], "answer": "s1"}),
    ("What is Apple's revenue per employee for fiscal 2026?",
     {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026}},
                {"id": "s2", "op": "attr", "company": "AAPL", "attr": "employees"},
                {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}),
    ("What was Tesla's headcount of robots in 2025?",
     {"steps": [], "answer": None, "cannot": "no concept or attribute for a count of robots"}),
]


def system_prompt(graph) -> str:
    v = vocabulary(graph)
    parts = [RULES, "COMPANIES (ticker: name)\n" + "\n".join(f"{t}: {n}" for t, n in v["companies"].items()),
             "CLUSTERS: " + ", ".join(v["clusters"]),
             "CONCEPTS (name: meaning)\n" + "\n".join(f"{c}: {d}" for c, d in v["concepts"].items()),
             "METRICS (name: meaning)\n" + "\n".join(f"{m}: {d}" for m, d in v["metrics"].items()),
             "ATTRIBUTES: " + ", ".join(v["attributes"]),
             "EXAMPLES\n" + "\n".join(f"Q: {q}\nA: {json.dumps(p, separators=(',', ':'))}" for q, p in EXAMPLES)]
    return "\n\n".join(parts)


def plan_question(question: str, graph, model: Optional[str] = None, offline: bool = False) -> dict:
    """Return {"plan": ..., "errors": [...], "attempts": n}."""
    system = system_prompt(graph)
    user = f"Q: {question}\nA:"
    plan = ask_json(system, user, model=model, tag=f"plan_{PROMPT_VERSION}", offline=offline)
    if plan.get("cannot") or not plan.get("steps"):
        return {"plan": plan, "errors": [], "attempts": 1}
    errs = validate(plan, graph)
    if not errs:
        return {"plan": plan, "errors": [], "attempts": 1}
    retry = (f"Q: {question}\nYour previous plan was rejected by the validator:\n"
             + "\n".join("- " + e for e in errs) + f"\nPrevious plan: {json.dumps(plan)}\nWrite a corrected plan.\nA:")
    plan2 = ask_json(system, retry, model=model, tag=f"plan_{PROMPT_VERSION}r", offline=offline)
    errs2 = [] if (plan2.get("cannot") or not plan2.get("steps")) else validate(plan2, graph)
    return {"plan": plan2, "errors": errs2, "attempts": 2}
