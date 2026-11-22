"""Question to plan.

The planner is the only place a language model reads a question. It is given
the graph's vocabulary (companies, concepts, metrics, attributes, operations)
and writes a plan in that vocabulary. It is given no data, so it has nothing
to copy a number from.

The prompt is the plan language (RULES), the vocabulary, and a few worked
examples. The examples come from a fixed pool written from the language, one
or more per operation; the ones whose wording is closest to the question are
shown, so the prompt stays short enough for free-tier rate limits.

Two chances to repair: a plan the validator rejects is sent back once with the
errors, and `ask.answer` sends a plan back once more when the executor cannot
run it (with the executor's reason). A repair never sees a value from the graph.

Every call is cached by content hash (see llm.py), so a run can be replayed
offline and gives the same result every time.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter
from typing import Optional

from .llm import ask_json
from .plan import metric_defs, normalize, validate, vocabulary

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_VERSION = "p14"
N_EXAMPLES = 6

RULES = """Translate a question about US public companies into a query plan over a knowledge graph of their
SEC filings. Output JSON only: {"steps":[...],"answer":"<id of the answering step>"}. Never write a figure
from your own knowledge; the only numbers you write are thresholds stated in the question.

STEPS (each has "id" s1, s2, ... and "op")
get      {"op":"get","company":T,"concept":C,"period":P}; instead of "concept": "metric":M, "item":"words"
         or "by":MEASURE. "as":"first_reported" reads the figure as first printed
calc     {"op":"calc","expr":"s1 / s2"}  + - * / ( ) abs() over earlier steps
change   {"op":"change","company":T,"concept":C,"from":P,"to":P,"mode":"pct"|"abs"}
rank     {"op":"rank","over":SET,"by":MEASURE,"period":P,"order":"desc"|"asc","k":N}
compare  {"op":"compare","of":["s3","s6"],"order":"desc"|"asc"}  orders figures worked out in earlier steps,
         one per side; like rank it gives the winner, every side's figure and the gap
filter   {"op":"filter","over":SET,"where":{"by":MEASURE,"period":P,"gt"|"lt"|"ge"|"le":X}} or
         {"op":"filter","over":SET,"where":{"attr":A,"eq":"text"}}
count    {"op":"count","of":"$sN"}  how many step sN kept, and which
agg      {"op":"agg","over":SET,"by":MEASURE,"period":P,"fn":"sum"|"avg"|"median"|"min"|"max"}
members  {"op":"members","company":T,"of":"words naming the breakdown","period":P}  the lines adding up to
         a total: segments, products, regions, businesses
series   {"op":"series","company":T,"by":MEASURE,"last":N}  "pick":"max"|"min" for the peak year
attr     {"op":"attr","company":T,"attr":A}  a text fact (auditor, state of incorporation, CEO ...)
restated {"op":"restated","company":T,"concept":C or "item":"words","period":P}  first against latest
         report of a figure, and why it changed
MEASURE  {"concept":C} {"metric":M} {"item":"words"} {"attr":"employees"|"shares_outstanding"}
         {"growth":MEASURE} percent change from the year before, {"delta":MEASURE} the same as an amount,
         {"ratio":[MEASURE,MEASURE]}, {"value":true} a line's own figure (over lines)
T        one ticker, or "$sN" = the top member of step sN (a company, or a line if sN ranked lines).
         Never a list: for several companies use rank, filter or agg over a SET
SET      "all" | {"cluster":name} | [tickers] | "$sN"
P        {"fy":2025} {"fy":2025,"q":3} {"fy":2025,"q":3,"ytd":true} {"fy":"latest"} {"fy":"prior"}

RULES
1. A concept is a company-wide line of the income statement, balance sheet or cash flow statement.
   Anything narrower or elsewhere is an item, written as the filing labels it with the segment or product
   named: a segment, product, region or business line; a count (stores, warehouses, members, vehicles,
   people in a region); ARR, backlog, remaining performance obligation, assets under management, client
   assets, net new money, capital ratios. A line narrower than a concept (net sales without membership
   fees, one share class) is an item too.
2. Copy the fiscal year named: "fiscal 2025", "FY25", "2025" are {"fy":2025}; a balance "at the end of
   fiscal 2025" is {"fy":2025}. No year: {"fy":"latest"}; the year before it: {"fy":"prior"}. A quarter
   is {"fy":Y,"q":Q} with Y the fiscal year it belongs to; "nine months"/"year to date" add "ytd":true.
   A fourth quarter is {"fy":Y,"q":4}: the graph derives it from the year and the nine months, so never
   subtract periods yourself, even when the question explains the derivation.
3. Ratios, shares, margins, per-unit figures, growth and differences are computed: a metric only when it
   means exactly what is asked, otherwise get the parts and calc, or a growth, delta or ratio measure.
4. "Which company has the highest X": rank, k=1. "Rank A, B and C": rank over them, k = their number.
5. "Which of A or B had more X (and by how much)": rank over [A,B], k=2; it gives both values, the winner
   and the gap. Also when X is a growth rate or a ratio. When each side needs its own words or its own
   arithmetic (one company's segment margin against another's, or a line each company labels
   differently), work each side out with get and calc, then end with compare over those steps.
6. "How many ...": filter then count. Do not end on count unless the question asks how many.
7. A hop (find a company or line, then ask about it) names the earlier step as "$sN".
8. "Which segment/product/region/business ...": members, then rank or filter over "$sN" by {"value":true},
   {"growth":{"value":true}} (fastest growing, largest percentage change) or {"delta":{"value":true}}
   (largest increase or decline in dollars). A ratio per line (a segment's margin) is
   {"ratio":[{"value":true},{"item":"revenue"}]}: an item inside a measure over lines is that line's own
   other figure. For a figure of the chosen line add get with "company":"$sN" and the period (its own
   value) or with "item":"words" (another figure of that line, such as its revenue). The figure of one
   segment the question names needs no members: it is an item.
9. Thresholds in full units: $10 billion is 10000000000; 20 percent is 0.2.
10. The size of a loss, gap or deduction uses abs() in calc: losses, costs, incentives and discounts may be
    printed negative, so wrap them in abs() before adding them back or taking their share.
11. General knowledge may only say WHICH listed company is meant (a brand, "Google's parent").
12. {"steps":[],"answer":null,"cannot":"<why>"} when a company is not listed, or for what filings do not
    print (stock prices, market share, forecasts, ratings) or leave to the proxy (audit fees, pay).
    Never answer cannot because of the year (the filings run to fiscal 2026, whatever today's date
    seems to be) or because a figure is narrower than a concept (that is an item).
13. If the question also asks why a figure changed, plan the figure (restated explains why).
14. Attributes are facts of the latest filing and have no period. A head-count or share count for a named
    earlier year, by region or by share class, is an item.
"""

# The example pool: one or more per operation and shape, written from the language above.
# None is a benchmark question; companies and wording were chosen to differ from them.
EXAMPLES = [
    ("What were Caterpillar's total sales and revenues in 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "CAT", "concept": "Revenue", "period": {"fy": 2025}}], "answer": "s1"}),
    ("What was Oracle's operating margin in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "metric", "company": "ORCL", "metric": "operating_margin", "period": {"fy": 2025}}], "answer": "s1"}),
    ("How much net income did Pfizer earn in the third quarter of 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "PFE", "concept": "NetIncome", "period": {"fy": 2025, "q": 3}}], "answer": "s1"}),
    ("What was Nike's revenue in the fourth quarter of fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "NKE", "concept": "Revenue", "period": {"fy": 2025, "q": 4}}], "answer": "s1"}),
    ("How much cash did Target generate from operations in the first nine months of fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "TGT", "concept": "OperatingCashFlow", "period": {"fy": 2025, "q": 3, "ytd": True}}], "answer": "s1"}),
    ("How much revenue did Microsoft's More Personal Computing segment bring in for fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "MSFT", "item": "More Personal Computing revenue", "period": {"fy": 2025}}], "answer": "s1"}),
    ("How many restaurants did McDonald's operate in France at the end of 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "MCD", "item": "number of restaurants in France", "period": {"fy": 2025}}], "answer": "s1"}),
    ("What order backlog did Caterpillar report at the end of 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "CAT", "item": "order backlog", "period": {"fy": 2025}}], "answer": "s1"}),
    ("As first reported in its fiscal 2025 annual report, what was Nike's inventory at the end of fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "NKE", "concept": "Inventory", "period": {"fy": 2025}, "as": "first_reported"}], "answer": "s1"}),
    ("How much more did Microsoft spend on research and development than Apple in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "MSFT", "concept": "ResearchAndDevelopment", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "AAPL", "concept": "ResearchAndDevelopment", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 - s2"}], "answer": "s3"}),
    ("What share of Alphabet's 2025 revenue came from YouTube ads?",
     {"steps": [{"id": "s1", "op": "get", "company": "GOOGL", "item": "YouTube ads revenue", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "GOOGL", "concept": "Revenue", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}),
    ("What was Apple's net income per employee in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "NetIncome", "period": {"fy": 2025}},
                {"id": "s2", "op": "attr", "company": "AAPL", "attr": "employees"},
                {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}),
    ("Dividing Visa's fiscal 2025 net revenue by the number of processed transactions, how much did it earn per transaction?",
     {"steps": [{"id": "s1", "op": "get", "company": "V", "concept": "Revenue", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "V", "item": "processed transactions", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 / s2"}], "answer": "s3"}),
    ("Combined, how much did Apple's Mac and iPad product lines sell in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "item": "Mac net sales", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "AAPL", "item": "iPad net sales", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 + s2"}], "answer": "s3"}),
    ("How big was Alphabet's Other Bets operating loss relative to its total operating income in 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "GOOGL", "item": "Other Bets operating income (loss)", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "GOOGL", "concept": "OperatingIncome", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "abs(s1) / s2"}], "answer": "s3"}),
    ("By what percentage did Costco's net income grow in fiscal 2025?",
     {"steps": [{"id": "s1", "op": "change", "company": "COST", "concept": "NetIncome", "from": {"fy": 2024}, "to": {"fy": 2025}, "mode": "pct"}], "answer": "s1"}),
    ("By how much did Home Depot's operating income change from fiscal 2023 to fiscal 2024?",
     {"steps": [{"id": "s1", "op": "change", "company": "HD", "concept": "OperatingIncome", "from": {"fy": 2023}, "to": {"fy": 2024}, "mode": "abs"}], "answer": "s1"}),
    ("How fast did AWS revenue grow in Amazon's latest fiscal year?",
     {"steps": [{"id": "s1", "op": "change", "company": "AMZN", "item": "AWS net sales", "from": {"fy": "prior"}, "to": {"fy": "latest"}, "mode": "pct"}], "answer": "s1"}),
    ("Which health care company had the highest net margin in 2025?",
     {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "health"}, "by": {"metric": "net_margin"}, "period": {"fy": 2025}, "order": "desc", "k": 1}], "answer": "s1"}),
    ("Which three companies in the set had the lowest effective tax rate in their latest fiscal year?",
     {"steps": [{"id": "s1", "op": "rank", "over": "all", "by": {"metric": "effective_tax_rate"}, "period": {"fy": "latest"}, "order": "asc", "k": 3}], "answer": "s1"}),
    ("Rank Visa, Mastercard and American Express by number of employees, largest first.",
     {"steps": [{"id": "s1", "op": "rank", "over": ["V", "MA", "AXP"], "by": {"attr": "employees"}, "period": {"fy": "latest"}, "order": "desc", "k": 3}], "answer": "s1"}),
    ("Rank Home Depot, Target and Walmart by the number of stores at the end of each one's latest fiscal year, most first.",
     {"steps": [{"id": "s1", "op": "rank", "over": ["HD", "TGT", "WMT"], "by": {"item": "total number of stores"}, "period": {"fy": "latest"}, "order": "desc", "k": 3}], "answer": "s1"}),
    ("Who held more cash at the end of 2025, Meta or Alphabet, and by how much?",
     {"steps": [{"id": "s1", "op": "rank", "over": ["META", "GOOGL"], "by": {"concept": "Cash"}, "period": {"fy": 2025}, "order": "desc", "k": 2}], "answer": "s1"}),
    ("Which grew revenue faster in fiscal 2025, Microsoft or Oracle, and by how many percentage points?",
     {"steps": [{"id": "s1", "op": "rank", "over": ["MSFT", "ORCL"], "by": {"growth": {"concept": "Revenue"}}, "period": {"fy": 2025}, "order": "desc", "k": 2}], "answer": "s1"}),
    ("Which bank ran leaner in 2025, Citigroup or Wells Fargo, measured as noninterest expense over total revenue?",
     {"steps": [{"id": "s1", "op": "rank", "over": ["C", "WFC"], "by": {"ratio": [{"concept": "NoninterestExpense"}, {"concept": "Revenue"}]}, "period": {"fy": 2025}, "order": "asc", "k": 2}], "answer": "s1"}),
    ("Who audits the consumer company with the largest inventory at the end of its latest fiscal year?",
     {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "consumer_retail"}, "by": {"concept": "Inventory"}, "period": {"fy": "latest"}, "order": "desc", "k": 1},
                {"id": "s2", "op": "attr", "company": "$s1", "attr": "auditor"}], "answer": "s2"}),
    ("What was the effective tax rate of the bank with the most deposits at the end of 2025?",
     {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "banks_finance"}, "by": {"concept": "Deposits"}, "period": {"fy": 2025}, "order": "desc", "k": 1},
                {"id": "s2", "op": "metric", "company": "$s1", "metric": "effective_tax_rate", "period": {"fy": 2025}}], "answer": "s2"}),
    ("How many of the big tech companies had an operating margin above 30 percent in their latest fiscal year, and which?",
     {"steps": [{"id": "s1", "op": "filter", "over": {"cluster": "big_tech"}, "where": {"by": {"metric": "operating_margin"}, "period": {"fy": "latest"}, "gt": 0.3}},
                {"id": "s2", "op": "count", "of": "$s1"}], "answer": "s2"}),
    ("Which of Ford, General Motors and Tesla earned less net income in 2025 than in 2024?",
     {"steps": [{"id": "s1", "op": "filter", "over": ["F", "GM", "TSLA"], "where": {"by": {"delta": {"concept": "NetIncome"}}, "period": {"fy": 2025}, "lt": 0}}], "answer": "s1"}),
    ("Among the companies audited by KPMG, which had the highest revenue in its latest fiscal year?",
     {"steps": [{"id": "s1", "op": "filter", "over": "all", "where": {"attr": "auditor", "eq": "KPMG"}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"concept": "Revenue"}, "period": {"fy": "latest"}, "order": "desc", "k": 1}], "answer": "s2"}),
    ("What was the combined 2025 net income of JPMorgan, Bank of America and Citigroup?",
     {"steps": [{"id": "s1", "op": "agg", "over": ["JPM", "BAC", "C"], "by": {"concept": "NetIncome"}, "period": {"fy": 2025}, "fn": "sum"}], "answer": "s1"}),
    ("What was the average gross margin of the consumer companies in their latest fiscal year?",
     {"steps": [{"id": "s1", "op": "agg", "over": {"cluster": "consumer_retail"}, "by": {"metric": "gross_margin"}, "period": {"fy": "latest"}, "fn": "avg"}], "answer": "s1"}),
    ("In which of the last three fiscal years did Merck's net income peak?",
     {"steps": [{"id": "s1", "op": "series", "company": "MRK", "by": {"concept": "NetIncome"}, "last": 3, "pick": "max"}], "answer": "s1"}),
    ("Which of Alphabet's segments had the highest operating income in 2025?",
     {"steps": [{"id": "s1", "op": "members", "company": "GOOGL", "of": "operating income by segment", "period": {"fy": 2025}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"value": True}, "period": {"fy": 2025}, "order": "desc", "k": 1}], "answer": "s2"}),
    ("Which of Ford's segments lost money before interest and taxes in 2025?",
     {"steps": [{"id": "s1", "op": "members", "company": "F", "of": "EBIT by segment", "period": {"fy": 2025}},
                {"id": "s2", "op": "filter", "over": "$s1", "where": {"by": {"value": True}, "period": {"fy": 2025}, "lt": 0}}], "answer": "s2"}),
    ("Which of Microsoft's product and service lines grew revenue fastest in fiscal 2025, and how much revenue did it bring in?",
     {"steps": [{"id": "s1", "op": "members", "company": "MSFT", "of": "revenue by product and service", "period": {"fy": 2025}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"growth": {"value": True}}, "period": {"fy": 2025}, "order": "desc", "k": 1},
                {"id": "s3", "op": "get", "company": "$s2", "period": {"fy": 2025}}], "answer": "s3"}),
    ("Which of Caterpillar's segments had the highest profit margin in 2025, and what was the margin?",
     {"steps": [{"id": "s1", "op": "members", "company": "CAT", "of": "profit by segment", "period": {"fy": 2025}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"ratio": [{"value": True}, {"item": "sales and revenues"}]},
                 "period": {"fy": 2025}, "order": "desc", "k": 1}], "answer": "s2"}),
    ("Which Procter & Gamble segment earned the most before tax in fiscal 2025, and what share of its own net sales was that?",
     {"steps": [{"id": "s1", "op": "members", "company": "PG", "of": "earnings before income taxes by segment", "period": {"fy": 2025}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"value": True}, "period": {"fy": 2025}, "order": "desc", "k": 1},
                {"id": "s3", "op": "get", "company": "$s2", "period": {"fy": 2025}},
                {"id": "s4", "op": "get", "company": "$s2", "item": "net sales", "period": {"fy": 2025}},
                {"id": "s5", "op": "calc", "expr": "s3 / s4"}], "answer": "s5"}),
    ("Which of Walmart's segments lost the most net sales in dollars between fiscal 2024 and fiscal 2025, and how much?",
     {"steps": [{"id": "s1", "op": "members", "company": "WMT", "of": "net sales by segment", "period": {"fy": 2025}},
                {"id": "s2", "op": "rank", "over": "$s1", "by": {"delta": {"value": True}}, "period": {"fy": 2025}, "order": "asc", "k": 1}], "answer": "s2"}),
    ("Which turned more of its segment revenue into operating income in 2025, Microsoft's More Personal Computing or Meta's Family of Apps, and by how many points?",
     {"steps": [{"id": "s1", "op": "get", "company": "MSFT", "item": "More Personal Computing operating income", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "MSFT", "item": "More Personal Computing revenue", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 / s2"},
                {"id": "s4", "op": "get", "company": "META", "item": "Family of Apps income from operations", "period": {"fy": 2025}},
                {"id": "s5", "op": "get", "company": "META", "item": "Family of Apps revenue", "period": {"fy": 2025}},
                {"id": "s6", "op": "calc", "expr": "s4 / s5"},
                {"id": "s7", "op": "compare", "of": ["s3", "s6"], "order": "desc"}], "answer": "s7"}),
    ("What was Target's operating margin in the fourth quarter of fiscal 2025, worked out from the annual report and the nine-month figures?",
     {"steps": [{"id": "s1", "op": "metric", "company": "TGT", "metric": "operating_margin", "period": {"fy": 2025, "q": 4}}], "answer": "s1"}),
    ("How many more people did Cisco employ at the end of fiscal 2025 than at the end of fiscal 2023?",
     {"steps": [{"id": "s1", "op": "get", "company": "CSCO", "item": "number of employees", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "CSCO", "item": "number of employees", "period": {"fy": 2023}},
                {"id": "s3", "op": "calc", "expr": "s1 - s2"}], "answer": "s3"}),
    ("Before deducting customer rebates, what were Merck's gross sales in 2025?",
     {"steps": [{"id": "s1", "op": "get", "company": "MRK", "concept": "Revenue", "period": {"fy": 2025}},
                {"id": "s2", "op": "get", "company": "MRK", "item": "customer rebates", "period": {"fy": 2025}},
                {"id": "s3", "op": "calc", "expr": "s1 + abs(s2)"}], "answer": "s3"}),
    ("Did Pfizer's 2025 annual report change the 2024 revenue it first reported, and why?",
     {"steps": [{"id": "s1", "op": "restated", "company": "PFE", "concept": "Revenue", "period": {"fy": 2024}}], "answer": "s1"}),
    ("Did Walmart's latest annual report revise the prior year's accrued liabilities balance?",
     {"steps": [{"id": "s1", "op": "restated", "company": "WMT", "item": "Accrued liabilities", "period": {"fy": "prior"}}], "answer": "s1"}),
    ("In which state is Chevron incorporated?",
     {"steps": [{"id": "s1", "op": "attr", "company": "CVX", "attr": "state_of_incorporation"}], "answer": "s1"}),
    ("Which firm audits the company that makes Gillette razors?",
     {"steps": [{"id": "s1", "op": "attr", "company": "PG", "attr": "auditor"}], "answer": "s1"}),
    ("What was Boeing's operating cash flow in 2025?",
     {"steps": [], "answer": None, "cannot": "Boeing is not among the listed companies"}),
    ("How much did Microsoft pay its chief executive in fiscal 2025?",
     {"steps": [], "answer": None, "cannot": "executive pay is reported in the proxy statement, not the annual report"}),
    ("What is Tesla's share price today?",
     {"steps": [], "answer": None, "cannot": "the graph holds filing facts, not market prices"}),
]

_WORD = re.compile(r"[a-z&%]+")                  # letters only: years and figures say nothing about shape
_STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "it", "what", "was", "were", "did", "does",
         "is", "how", "much", "which", "who", "at", "on", "that", "this", "with", "from", "as", "be", "or",
         "fiscal", "fy", "year", "company", "companies", "s", "inc", "corp", "co", "group", "report", "reported"}


def _words(text: str, ignore: frozenset = frozenset()) -> list:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and w not in ignore]


def pick_examples(question: str, n: int = N_EXAMPLES, ignore: frozenset = frozenset()) -> list:
    """The n pool examples whose wording is closest to the question in shape.

    Company names, years and filler are ignored, so "how many stores ... at the end of" finds the
    store-count example whatever the company; rarer shared words count more.
    """
    words = [set(_words(q, ignore)) for q, _ in EXAMPLES]
    df = Counter(w for ws in words for w in ws)
    want = set(_words(question, ignore))
    scored = [(sum(math.log((1 + len(EXAMPLES)) / (1 + df[w])) for w in want & ws), i) for i, ws in enumerate(words)]
    chosen = sorted(i for _, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:n])
    return [EXAMPLES[i] for i in chosen]


def _name_words(companies: dict) -> frozenset:
    """Words of company names and tickers: they identify a company, not the shape of a question."""
    return frozenset(w for t, name in companies.items() for w in _WORD.findall(f"{t} {name}".lower()))


CONCEPT_NOTES = """NetIncome is net income attributable to the company's shareholders (the usual "net income", "profit",
"earnings"); ConsolidatedNetIncome includes noncontrolling interests. ShareholdersEquity is stockholders'
equity (book value); TotalEquity includes noncontrolling interests. Revenue is total revenue or net sales
(banks: total net revenue). TotalLiabilities only where printed; the metric total_liabilities also solves it.
CapitalExpenditures, DividendsPaid, ShareRepurchases are cash paid, from the cash-flow statement, as are
DepreciationAmortization and StockBasedCompensation. NetInterestIncome, NoninterestIncome, NoninterestExpense,
ProvisionForCreditLosses, Deposits, Loans are bank lines. SharesDiluted and SharesBasic are weighted-average
share counts used for EPS (shares outstanding on the cover is the attribute shares_outstanding)."""


def system_prompt(graph, question: str = "") -> str:
    v = vocabulary(graph)
    shots = pick_examples(question, ignore=_name_words(v["companies"])) if question else EXAMPLES
    formulas = {m["name"]: m["formula"] for m in metric_defs().values()}
    parts = [RULES,
             "COMPANIES " + "; ".join(f"{t}: {n}" for t, n in v["companies"].items()),
             "CLUSTERS " + ", ".join(v["clusters"]),
             "CONCEPTS " + ", ".join(v["concepts"]) + "\n" + CONCEPT_NOTES,
             "METRICS " + "; ".join(f"{m} = {formulas[m]}" for m in v["metrics"]),
             "ATTRIBUTES " + ", ".join(v["attributes"]),
             "EXAMPLES\n" + "\n".join(f"Q: {q}\nA: {json.dumps(p, separators=(',', ':'))}" for q, p in shots)]
    return "\n\n".join(parts)


LOCAL_PLANNER = "tieout-planner"                  # the fine-tuned model's name in Ollama (finetune/README)


def is_local_planner(model: Optional[str]) -> bool:
    return bool(model) and model.split(":", 1)[-1].startswith(LOCAL_PLANNER)


def local_prompt() -> str:
    """The short prompt the fine-tuned planner was trained with: task, operations and vocabulary only."""
    path = ROOT / "finetune" / "system_prompt.v2.txt"
    return path.read_text(encoding="utf-8").strip()


def plan_question(question: str, graph, model: Optional[str] = None, offline: bool = False,
                  feedback: Optional[dict] = None) -> dict:
    """Return {"plan": ..., "errors": [...], "attempts": n}.

    `feedback` = {"plan": earlier plan, "problem": what went wrong} asks for a repaired plan.
    The fine-tuned local planner was trained on question and plan pairs only, so it gets its
    short prompt and the bare question, and is not asked for repairs it was never taught.
    """
    if is_local_planner(model):
        if feedback:
            return {"plan": feedback["plan"], "errors": [], "attempts": 0}
        plan = _as_plan(ask_json(local_prompt(), question, model=model, tag=f"plan_{LOCAL_PLANNER}", offline=offline))
        errs = [] if (plan.get("cannot") or not plan.get("steps")) else validate(plan, graph)
        return {"plan": plan, "errors": errs, "attempts": 1}
    system = system_prompt(graph, question)
    if feedback:
        user = (f"Q: {question}\nAn earlier plan for this question could not be used.\n"
                f"Plan: {json.dumps(feedback['plan'], separators=(',', ':'))}\nProblem: {feedback['problem']}\n"
                "Write a corrected plan for the question. Change only what the problem points at; if the "
                "question cannot be planned with the listed vocabulary, answer cannot.\nA:")
        tag = f"plan_{PROMPT_VERSION}fix"
    else:
        user, tag = f"Q: {question}\nA:", f"plan_{PROMPT_VERSION}"
    try:
        plan = _as_plan(ask_json(system, user, model=model, tag=tag, offline=offline))
    except ValueError:                                # the reply was not JSON: ask once more, saying so
        plan = _as_plan(ask_json(system, user + " (Answer with one valid JSON object and nothing else.)",
                                 model=model, tag=tag, offline=offline))
    if plan.get("cannot") or not plan.get("steps"):
        return {"plan": plan, "errors": [], "attempts": 1}
    errs = validate(plan, graph)
    if not errs:
        return {"plan": plan, "errors": [], "attempts": 1}
    retry = (f"Q: {question}\nYour plan was rejected by the validator:\n" + "\n".join("- " + e for e in errs)
             + f"\nRejected plan: {json.dumps(plan, separators=(',', ':'))}\nWrite a corrected plan.\nA:")
    plan2 = _as_plan(ask_json(system, retry, model=model, tag=f"{tag}r", offline=offline))
    errs2 = [] if (plan2.get("cannot") or not plan2.get("steps")) else validate(plan2, graph)
    return {"plan": plan2, "errors": errs2, "attempts": 2}


def _as_plan(obj) -> dict:
    """Some models wrap the plan ({"plan": {...}}) or return a bare list of steps; unwrap both."""
    if isinstance(obj, dict) and "steps" not in obj and isinstance(obj.get("plan"), dict):
        obj = obj["plan"]
    if isinstance(obj, list):
        obj = {"steps": obj, "answer": obj[-1].get("id") if obj and isinstance(obj[-1], dict) else None}
    return normalize(obj) if isinstance(obj, dict) else {"steps": [], "answer": None, "cannot": "planner returned no plan"}
