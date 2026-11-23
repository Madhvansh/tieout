"""The scale test: thousands of questions keyed by the SEC's own XBRL data, run with no language model.

The benchmark measures the whole system with a few careful questions. This test measures the graph and the
executor alone, with thousands of questions whose answers the SEC already holds, and above all how often an
answer is wrong without saying so. The SEC serves every filing's tagged values as JSON ("companyfacts",
cached in data/sec); the graph is built from the PDFs and never reads it.

The key
    A concept's value is the first us-gaap element of KEY_TAGS that has a value. Each SEC value is placed on
    the company's own fiscal calendar by the rule that dates the graph's columns (tieout/periods.py): a year
    spans 12 months, a quarter 3, the year to date at the third quarter 9 (each within 0.6 month), and a
    balance is an instant. The SEC's own fiscal-year field is never used.
    Filings may report one period with different values (restatements). The latest value is the one in the
    latest corpus report that prints that period as a column; the first is the one in the earliest. A corpus
    report is found in the SEC data as the filing of the same form whose own period ends within 3 days of
    the report's. A question is asked only for a period the corpus prints, and only when its key exists.

Question kinds, each a one-step plan executed directly
    fact      one concept for one period (earnings per share included)
    growth    percentage change from the prior fiscal year
    metric    each metric of semantics/metrics.yaml, keyed by the same formula
    solve     the fourth quarter as the full year minus the nine months
    restated  latest-reported minus first-reported value of the earlier annual report's year
    rank      the companies of a cluster ordered by a concept in each one's latest year (four or more)

Four outcomes
    correct           within tolerance: half a unit of the last printed digit for a number read from one
                      cell, 0.5 percent (a ratio also 0.0005) for a computed one; a dollar amount is compared
                      as a magnitude, as the benchmark does, and such answers are counted separately
    abstained         the executor declined, or crashed (the error is kept)
    wrong_unverified  wrong, and some source cell was not verified
    wrong_silent      wrong, and every source cell was verified: nothing in the answer says it is weak
A wrong answer equal to a second definition a careful reader could defend (ALTERNATIVES) is also marked
alt_definition; it stays wrong.
"""
from __future__ import annotations

import ast
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterator, Optional

from ..concepts import concept_defs
from ..periods import FiscalCalendar
from ..plan import metric_defs
from .xbrl_audit import SEC_CACHE, TAG_MAP, printed_number

# Concepts whose element TAG_MAP (written for the binding audit) leaves out, or whose TAG_MAP list is wrong
# for a question about one amount. A balance of long-term debt is the non-current part alone, since the
# fallback element also counts the current part. Consolidated net income has no fallback: the attributable
# figure is a different amount. Revenue lists the net-of-interest total before contract revenue, which a
# card or payments filer tags for only a part of its revenue.
KEY_TAGS = {
    **TAG_MAP,
    "Revenue": ["Revenues", "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "ConsolidatedNetIncome": ["ProfitLoss"],
    "LongTermDebt": ["LongTermDebtNoncurrent"],
    "EPSDiluted": ["EarningsPerShareDiluted"],
    "EPSBasic": ["EarningsPerShareBasic"],
    "PretaxIncome": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                     "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "SellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense"],
    "SalesAndMarketing": ["SellingAndMarketingExpense"],
    "GeneralAndAdministrative": ["GeneralAndAdministrativeExpense"],
    "SharesDiluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "SharesBasic": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "Inventory": ["InventoryNet"],
    "NoninterestIncome": ["NoninterestIncome"],
    "NoninterestExpense": ["NoninterestExpense"],
}

# Second definitions a careful reader could defend: the filing prints both figures and the concept's words do
# not settle which one is meant.
ALTERNATIVES = {
    "NetIncome": ["ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ConsolidatedNetIncome": ["NetIncomeLoss"],
    "ShareholdersEquity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "TotalEquity": ["StockholdersEquity"],
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "InterestExpense": ["InterestExpenseNonoperating", "InterestExpenseDebt"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "AccountsReceivable": ["ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"],
    "LongTermDebt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "CapitalExpenditures": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "InvestingCashFlow": ["NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "FinancingCashFlow": ["NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "EPSDiluted": ["IncomeLossFromContinuingOperationsPerDilutedShare"],
    "EPSBasic": ["IncomeLossFromContinuingOperationsPerBasicShare"],
    "PretaxIncome": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
}

FAMILIES = ("fact", "growth", "metric", "solve", "restated", "rank")
CLASSES = ("correct", "abstained", "wrong_unverified", "wrong_silent")
REPORT_FORMS = ("10-K", "10-Q")
SEC_UNITS = ("USD", "USD/shares", "shares")
SPANS = (3, 9, 12)             # months a key period spans: a quarter, nine months, a year
SPAN_SLACK = 0.6               # months
END_SLACK = 3                  # days between a corpus report's period end and its SEC filing's
DERIVED_TOLERANCE = 0.005      # a computed number: 0.5 percent of the key
RATIO_TOLERANCE = 0.0005       # a computed ratio may also be off by this much
MAX_CHOICES = 64               # combinations of second definitions tried for one key
RANK_MIN = 4                   # companies a ranking needs


# ------------------------------------------------------------------ periods and the key

@dataclass(frozen=True, order=True)
class Period:
    """A period as a plan names it: a fiscal year, a fiscal quarter, or the year to date at a quarter."""
    fy: int
    q: int = 0
    ytd: bool = False

    def spec(self) -> dict:
        return {"fy": self.fy, **({"q": self.q} if self.q else {}), **({"ytd": True} if self.ytd else {})}

    def pid(self, nature: str) -> str:
        """The graph's period id for a concept of this nature (see tieout/periods.py)."""
        base = f"FY{self.fy}"
        if self.q:
            base += f"Q{self.q}" + ("YTD" if self.ytd and nature == "flow" else "")
        return base + ("@end" if nature == "stock" else "")

    def label(self) -> str:
        return f"FY{self.fy}" + (f"Q{self.q}" + ("YTD" if self.ytd else "") if self.q else "")


@dataclass(frozen=True)
class Report:
    """One filing of the corpus."""
    doc: str
    form: str
    fy: int
    period_end: str


def reports_of(g, ticker: str) -> list:
    """The company's corpus filings, latest first."""
    out = [Report(doc, f["form"], int(f["fiscal_year"].lstrip("FY")), f["period_end"])
           for doc, f in g.filings.items() if f["ticker"] == ticker]
    return sorted(out, key=lambda r: r.period_end, reverse=True)


def report_periods(r: Report, calendar: FiscalCalendar) -> dict:
    """{statement: [Period]}: the columns one report prints. An annual report has three years of flows and
    two balance sheets; a third-quarter report the quarter and the nine months, this year and last, and the
    balance sheet at the quarter end."""
    if r.form == "10-K":
        return {"IS": [Period(r.fy - b) for b in range(3)], "CF": [Period(r.fy - b) for b in range(3)],
                "BS": [Period(r.fy - b) for b in range(2)]}
    q = int(re.search(r"Q(\d)", calendar.period_id(date.fromisoformat(r.period_end), None, True)).group(1))
    to_date = [Period(y, q, True) for y in (r.fy, r.fy - 1)] if q > 1 else []
    quarter = [Period(y, q) for y in (r.fy, r.fy - 1)]
    return {"IS": quarter + to_date, "CF": to_date or quarter, "BS": [Period(r.fy, q)]}


def period_ids(r: Report, calendar: FiscalCalendar) -> set:
    return {p.pid("stock" if st == "BS" else "flow") for st, ps in report_periods(r, calendar).items() for p in ps}


@dataclass(frozen=True)
class Record:
    """One value the SEC holds for one element and one period."""
    value: float
    tag: str
    form: str
    filed: str
    accn: str


def period_of(calendar: FiscalCalendar, start: Optional[str], end: str) -> Optional[str]:
    """The period id an SEC value has on the company's calendar; None if it spans no key period."""
    end_day = date.fromisoformat(end)
    if not start:
        return calendar.period_id(end_day, None, instant=True)
    months = (end_day - date.fromisoformat(start)).days / 30.44
    span = next((m for m in SPANS if abs(months - m) <= SPAN_SLACK), None)
    return None if span is None else calendar.period_id(end_day, span, instant=False)


class CompanyKey:
    """What the SEC holds for one company, by element and period id, with the corpus reports located."""

    def __init__(self, facts: dict, calendar: FiscalCalendar, reports: list, blocked: frozenset = frozenset()):
        self.blocked = blocked      # concepts whose SEC element is known not to hold the amount they mean
        gaap = facts.get("facts", {}).get("us-gaap", {})
        elements = {t for tags in (*KEY_TAGS.values(), *ALTERNATIVES.values()) for t in tags}
        self.records = _records(gaap, calendar, elements)
        ends = _filing_ends(gaap)
        located = [(_accession(ends, r), period_ids(r, calendar)) for r in reports]
        self.filings = [(accn, ids) for accn, ids in located if accn]     # latest report first

    def _pool(self, tags: list, pid: str) -> list:
        return [r for tag in tags for r in self.records.get((tag, pid), ())]

    def _in_report(self, tags: list, pid: str, filings: list) -> Optional[Record]:
        for accn, ids in filings:
            for tag in tags:
                hit = next((r for r in self.records.get((tag, pid), ()) if r.accn == accn and pid in ids), None)
                if hit:
                    return hit
        return None

    def latest(self, tags: list, pid: str) -> Optional[Record]:
        """The value in the latest corpus report that prints the period, else the most recently filed."""
        pool = self._pool(tags, pid)
        return self._in_report(tags, pid, self.filings) or (
            max(pool, key=lambda r: (r.filed, -tags.index(r.tag))) if pool else None)

    def first(self, tags: list, pid: str) -> Optional[Record]:
        """The value in the earliest corpus report that prints the period, else the earliest filed."""
        pool = self._pool(tags, pid)
        return self._in_report(tags, pid, self.filings[::-1]) or (
            min(pool, key=lambda r: (r.filed, tags.index(r.tag))) if pool else None)


def _records(gaap: dict, calendar: FiscalCalendar, elements: set) -> dict:
    """{(element, period id): [Record]} from original 10-K and 10-Q filings."""
    out = defaultdict(list)
    for tag in elements:
        for unit, rows in gaap.get(tag, {}).get("units", {}).items():
            for x in rows if unit in SEC_UNITS else ():
                pid = period_of(calendar, x.get("start"), x["end"]) if x.get("form") in REPORT_FORMS else None
                if pid:
                    out[(tag, pid)].append(Record(x["val"], tag, x["form"], x["filed"], x["accn"]))
    return dict(out)


def _filing_ends(gaap: dict) -> dict:
    """{accession: (form, latest end date of a duration it reports)} for 10-K and 10-Q filings."""
    out: dict = {}
    for body in gaap.values():
        for rows in body.get("units", {}).values():
            for x in rows:
                if x.get("start") and x.get("form") in REPORT_FORMS:
                    out[x["accn"]] = (x["form"], max(out.get(x["accn"], ("", ""))[1], x["end"]))
    return out


def _accession(ends: dict, report: Report) -> Optional[str]:
    """The SEC filing that is this corpus report: the same form, its own period ending within 3 days."""
    day = date.fromisoformat(report.period_end)
    hits = sorted(accn for accn, (form, end) in ends.items()
                  if form == report.form and abs((date.fromisoformat(end) - day).days) <= END_SLACK)
    return hits[0] if hits else None


def load_keys(g, tickers: list, blocked: frozenset = frozenset()) -> dict:
    """{ticker: CompanyKey} for every company whose companyfacts file is in the cache. `blocked` holds
    (company, concept) pairs the SEC data cannot key."""
    keys = {}
    for t in tickers:
        path = SEC_CACHE / f"CIK{int(g.companies[t]['cik']):010d}.json"
        if path.exists():
            keys[t] = CompanyKey(json.loads(path.read_text(encoding="utf-8")), g.calendars[t], reports_of(g, t),
                                 frozenset(c for tk, c in blocked if tk == t))
    return keys


# ------------------------------------------------------------------ keys of computed answers

class Reader:
    """Reads key values through the key's elements, or through chosen second definitions, and remembers
    the SEC records it used."""

    def __init__(self, key: CompanyKey, choice: Optional[dict] = None):
        self.key, self.choice, self.used = key, choice or {}, []

    def __call__(self, concept: str, pid: str, first: bool = False) -> Optional[float]:
        tags = self.choice.get(concept) or KEY_TAGS[concept]
        if concept in self.key.blocked:
            return None
        rec = (self.key.first if first else self.key.latest)(tags, pid)
        if rec is None:
            return None
        self.used.append({"concept": concept, "period": pid, "as": "first" if first else "latest",
                          "tag": rec.tag, "value": rec.value, "form": rec.form, "filed": rec.filed})
        return rec.value


def keyed(compute: Callable, key: CompanyKey, concepts: list) -> tuple:
    """(key, alternative keys, SEC records used): `compute(read)` over the key's elements, then over every
    combination of the concepts' second definitions."""
    primary = Reader(key)
    value = compute(primary)
    alts: list = []
    if value is not None:
        options = [[None] + ALTERNATIVES.get(c, []) for c in concepts]
        for combo in itertools.islice(itertools.product(*options), 1, MAX_CHOICES):
            v = compute(Reader(key, {c: [tag] for c, tag in zip(concepts, combo) if tag}))
            if v is not None and v != value and v not in alts:
                alts.append(v)
    return value, alts, primary.used


def formula_concepts(formula: str) -> list:
    """The concept names a metric formula uses."""
    tree = ast.parse(formula, mode="eval")
    calls = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in calls})


def evaluate(formula: str, value: Callable[[str], Optional[float]]) -> Optional[float]:
    """Arithmetic over concept names (+ - * / and abs); None when a term is missing or a divisor is zero."""
    def ev(node) -> Optional[float]:
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if a is None or b is None:
                return None
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b if b else None}
            return ops[type(node.op)]()
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "abs":
            v = ev(node.args[0])
            return None if v is None else abs(v)
        if isinstance(node, ast.Name):
            return value(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"unsupported element in formula {formula}")
    return ev(ast.parse(formula, mode="eval").body)


def _growth(before: Optional[float], after: Optional[float]) -> Optional[float]:
    return None if before is None or after is None or before == 0 else (after - before) / abs(before)


def _minus(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


# ------------------------------------------------------------------ questions

@dataclass
class Question:
    id: str
    family: str
    company: str              # ticker, or "cluster:<name>" for a ranking
    concept: str              # concept or metric
    period: str               # period id or label; "latest" for a ranking
    unit: str                 # USD | USD/shares | shares | ratio
    plan: dict
    key: Any                  # a number, or the tickers in the key's order for a ranking
    alt_keys: list = field(default_factory=list)
    terms: list = field(default_factory=list)       # SEC records the key was computed from
    members: dict = field(default_factory=dict)     # ranking: ticker -> key value


@dataclass
class Company:
    """What question generation needs about one company."""
    ticker: str
    key: CompanyKey
    reports: list             # corpus reports, latest first
    printed: dict             # statement -> {period id: Period}


def make_company(g, ticker: str, key: CompanyKey) -> Company:
    reports = reports_of(g, ticker)
    printed: dict = {"IS": {}, "CF": {}, "BS": {}}
    for r in reports:
        for statement, periods in report_periods(r, g.calendars[ticker]).items():
            nature = "stock" if statement == "BS" else "flow"
            printed[statement].update({p.pid(nature): p for p in periods})
    return Company(ticker, key, reports, printed)


def _ask(co: Company, family: str, concept: str, period: str, unit: str, step: dict,
         compute: Callable, concepts: list) -> list:
    """The question as a one-step plan, if its key exists."""
    value, alts, used = keyed(compute, co.key, concepts)
    if value is None:
        return []
    plan = {"steps": [{"id": "s1", **step}], "answer": "s1"}
    return [Question(f"{family}:{co.ticker}:{concept}:{period}", family, co.ticker, concept, period, unit,
                     plan, value, alts, used)]


def fact_questions(co: Company, defs: dict, metrics: dict) -> Iterator[Question]:
    for c in (c for c in KEY_TAGS if c in defs):
        for pid, period in co.printed[defs[c].statement].items():
            step = {"op": "get", "company": co.ticker, "concept": c, "period": period.spec()}
            yield from _ask(co, "fact", c, pid, defs[c].unit, step, lambda read: read(c, pid), [c])


def growth_questions(co: Company, defs: dict, metrics: dict) -> Iterator[Question]:
    for c in (c for c in KEY_TAGS if c in defs):
        prints = co.printed[defs[c].statement]
        for pid, period in prints.items():
            before = Period(period.fy - 1).pid(defs[c].nature)
            if period.q or before not in prints:
                continue
            step = {"op": "change", "company": co.ticker, "concept": c, "from": prints[before].spec(),
                    "to": period.spec(), "mode": "pct"}
            yield from _ask(co, "growth", c, pid, "ratio", step,
                            lambda read: _growth(read(c, before), read(c, pid)), [c])


def metric_questions(co: Company, defs: dict, metrics: dict) -> Iterator[Question]:
    periods = sorted({p for prints in co.printed.values() for p in prints.values()})
    for name, m in metrics.items():
        names = formula_concepts(m["formula"])
        if not all(n in KEY_TAGS and n in defs for n in names):
            continue
        asked: set = set()
        for period in periods:
            pids = {n: period.pid(defs[n].nature) for n in names}
            if any(pids[n] not in co.printed[defs[n].statement] for n in names) or frozenset(pids.values()) in asked:
                continue
            asked.add(frozenset(pids.values()))
            step = {"op": "metric", "company": co.ticker, "metric": name, "period": period.spec()}
            yield from _ask(co, "metric", name, period.label(), m["unit"], step,
                            lambda read: evaluate(m["formula"], lambda n: read(n, pids[n])), names)


def solve_questions(co: Company, defs: dict, metrics: dict) -> Iterator[Question]:
    for c in (c for c in KEY_TAGS if c in defs and defs[c].nature == "flow" and defs[c].unit == "USD"):
        prints = co.printed[defs[c].statement]
        for pid, period in prints.items():
            nine = Period(period.fy, 3, True).pid("flow")
            if period.q or nine not in prints:
                continue
            step = {"op": "get", "company": co.ticker, "concept": c, "period": {"fy": period.fy, "q": 4}}
            yield from _ask(co, "solve", c, f"FY{period.fy}Q4", "USD", step,
                            lambda read: _minus(read(c, pid), read(c, nine)), [c])


def restated_questions(co: Company, defs: dict, metrics: dict) -> Iterator[Question]:
    annual = [r for r in co.reports if r.form == "10-K"]
    if len(annual) < 2:
        return
    earlier = Period(annual[1].fy)
    for c in (c for c in KEY_TAGS if c in defs):
        pid = earlier.pid(defs[c].nature)
        step = {"op": "restated", "company": co.ticker, "concept": c, "period": earlier.spec()}
        yield from _ask(co, "restated", c, pid, defs[c].unit, step,
                        lambda read: _minus(read(c, pid), read(c, pid, first=True)), [c])


def rank_questions(companies: list, clusters: dict, defs: dict) -> Iterator[Question]:
    """Within each cluster, for each concept keyed for enough of its companies: order them by it."""
    for cluster in sorted(set(clusters.values())):
        members = [co for co in companies if clusters[co.ticker] == cluster]
        for c in (c for c in KEY_TAGS if c in defs):
            values = {}
            for co in members:
                latest_fy = max(r.fy for r in co.reports if r.form == "10-K")
                value = Reader(co.key)(c, Period(latest_fy).pid(defs[c].nature))
                if value is not None:
                    values[co.ticker] = value
            if len(values) < RANK_MIN:
                continue
            step = {"op": "rank", "over": sorted(values), "by": {"concept": c}, "period": {"fy": "latest"},
                    "order": "desc", "k": len(values)}
            yield Question(f"rank:{cluster}:{c}:latest", "rank", f"cluster:{cluster}", c, "latest", defs[c].unit,
                           {"steps": [{"id": "s1", **step}], "answer": "s1"},
                           sorted(values, key=lambda t: (-values[t], t)), members=values)


PER_COMPANY = {"fact": fact_questions, "growth": growth_questions, "metric": metric_questions,
               "solve": solve_questions, "restated": restated_questions}


def build_questions(g, keys: dict, families: tuple = FAMILIES) -> list:
    """Every question of the asked families whose key exists, in a stable order."""
    defs, metrics = concept_defs(), metric_defs()
    companies = [make_company(g, t, keys[t]) for t in sorted(keys)]
    out = [q for co in companies for family in families if family in PER_COMPANY
           for q in PER_COMPANY[family](co, defs, metrics)]
    if "rank" in families:
        clusters = {t: g.companies[t]["cluster"] for t in keys}
        out.extend(rank_questions(companies, clusters, defs))
    return out


def coverage(g, keys: dict) -> dict:
    """What the test covers and what it leaves out, so the reader need not guess."""
    defs, metrics = concept_defs(), metric_defs()
    return {"companies": sorted(keys), "companies_without_sec_file": sorted(set(g.companies) - set(keys)),
            "concepts": sorted(c for c in defs if c in KEY_TAGS),
            "concepts_out": sorted(c for c in defs if c not in KEY_TAGS),
            "metrics": sorted(m for m, d in metrics.items() if set(formula_concepts(d["formula"])) <= set(KEY_TAGS)),
            "metrics_out": sorted(m for m, d in metrics.items() if not set(formula_concepts(d["formula"])) <= set(KEY_TAGS))}


# ------------------------------------------------------------------ judging

def is_verified(r) -> bool:
    """Every source cell of the answer is verified. (The executor carries no trust tier of its own.)"""
    return all(lf.status == "verified" for lf in r.leaves)


def read_tolerance(leaf, g) -> float:
    """Half a unit of the last digit a cell printed: half the table's scale for an amount printed in
    millions, half a cent for a per-share amount printed to the cent. With no digits in the printed text,
    half the table's scale for an amount and exact for anything else."""
    p = printed_number(leaf.printed)
    if p and isinstance(leaf.value, (int, float)) and leaf.value:
        number, decimals, _ = p
        return 0.5 * 10 ** -decimals * abs(float(leaf.value)) / number
    reading = g.readings.get(leaf.id)
    if reading is None or reading.unit != "USD":
        return 0.0
    return g.tables.get(reading.table, {}).get("scale", 0.0) / 2


def is_read(r) -> bool:
    """A number taken straight from one printed cell (its magnitude, for an outflow)."""
    return (len(r.leaves) == 1 and isinstance(r.leaves[0].value, (int, float))
            and abs(r.value) == abs(r.leaves[0].value))


def tolerance(q: Question, r, g) -> float:
    if q.family == "restated":     # the difference of two printed cells: the sum of their precisions
        return 2 * max((read_tolerance(lf, g) for lf in r.leaves if lf.what == q.concept), default=0.0)
    if is_read(r):
        return read_tolerance(r.leaves[0], g)
    return max(DERIVED_TOLERANCE * abs(q.key), RATIO_TOLERANCE if q.unit == "ratio" else 0.0)


def close(got: float, want: float, tol: float, unit: str) -> bool:
    """Within tolerance. An amount in dollars is compared as a magnitude, as the benchmark's scoring does."""
    return abs(got - want) <= tol or (unit == "USD" and abs(abs(got) - abs(want)) <= tol)


def judge_number(q: Question, r, g) -> tuple:
    """(correct?, equal to a second definition?, correct by magnitude only?)"""
    if r.kind != "number" or isinstance(r.value, bool) or not isinstance(r.value, (int, float)):
        return False, False, False
    tol = tolerance(q, r, g)
    if close(r.value, q.key, tol, q.unit):
        return True, False, abs(r.value - q.key) > tol
    return False, any(close(r.value, alt, tol, q.unit) for alt in q.alt_keys), False


def judge_rank(q: Question, r) -> tuple:
    """Correct when the companies come in the key's order (ties in either order)."""
    if r.kind != "companies":
        return False, False, False
    order = [name for name, _ in r.value]
    ok = sorted(order) == sorted(q.key) and all(q.members[a] >= q.members[b] for a, b in zip(order, order[1:]))
    return ok, False, False


def _num(x: Any) -> Any:
    """A number as the results file shows it: whole amounts exact, ratios to six significant digits."""
    if isinstance(x, float):
        return int(round(x)) if x.is_integer() or abs(x) >= 1e6 else float(f"{x:.6g}")
    return x


def judge(q: Question, r, g) -> dict:
    """One row: the outcome, and what is needed to investigate it when wrong."""
    row = {"id": q.id, "family": q.family, "company": q.company, "concept": q.concept, "period": q.period,
           "key": _num(q.key)}
    if r.kind == "abstain":
        return {**row, "class": "abstained", "why": "; ".join(r.notes)[:160]}
    correct, alt, by_magnitude = judge_rank(q, r) if q.family == "rank" else judge_number(q, r, g)
    row["got"] = [name for name, _ in r.value] if q.family == "rank" and r.kind == "companies" else _num(r.value)
    if correct:
        return {**row, "class": "correct", "sign_differs": by_magnitude}
    row.update({"class": "wrong_silent" if is_verified(r) else "wrong_unverified", "alt_definition": alt,
                "plan": q.plan, "key_terms": q.terms,
                "leaves": [{"doc": lf.doc, "page": lf.page, "label": lf.label[:60], "printed": lf.printed,
                            "period": lf.period, "value": _num(lf.value), "status": lf.status} for lf in r.leaves]})
    return row


def run(questions: list, executor, g) -> list:
    """Execute every plan directly and judge it. A plan that crashes the executor counts as abstained and
    keeps its error, so a crash is never scored as a wrong number."""
    rows = []
    for q in questions:
        try:
            result = executor.run(q.plan)
        except Exception as e:  # noqa: BLE001 - recorded in the row
            rows.append({"id": q.id, "family": q.family, "company": q.company, "concept": q.concept,
                         "period": q.period, "key": _num(q.key), "class": "abstained",
                         "error": f"{type(e).__name__}: {e}"[:160]})
            continue
        rows.append(judge(q, result, g))
    return rows


# ------------------------------------------------------------------ summary

def stats(rows: list) -> dict:
    c = Counter(r["class"] for r in rows)
    n, answered = len(rows), len(rows) - c["abstained"]
    return {"n": n, **{k: c[k] for k in CLASSES},
            "silent_wrong_equal_to_second_definition": sum(1 for r in rows if r["class"] == "wrong_silent"
                                                           and r["alt_definition"]),
            "accuracy": round(c["correct"] / n, 4) if n else None,
            "accuracy_when_answered": round(c["correct"] / answered, 4) if answered else None,
            "silent_wrong_rate": round(c["wrong_silent"] / n, 4) if n else None}


def grouped(rows: list, field_name: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[field_name]].append(r)
    return {k: stats(v) for k, v in sorted(groups.items())}


def summarize(rows: list) -> dict:
    """The outcomes overall and by question kind, company and concept."""
    return {"overall": stats(rows), "by_family": grouped(rows, "family"), "by_company": grouped(rows, "company"),
            "by_concept": grouped(rows, "concept")}


def wrong_rows(rows: list) -> list:
    return [r for r in rows if r["class"].startswith("wrong")]
