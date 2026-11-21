"""Independent audit of the graph against the SEC's own structured data.

Every 10-K is also filed with XBRL tags, and the SEC serves the tagged values
as JSON (data.sec.gov, free, no key). The pipeline never reads that data: the
graph is built from the PDFs alone. Here, after the build, the two are
compared.

Audit 1, value and period fidelity (label-free)
    For each money reading in the graph: does the SEC hold, for that company,
    a value with the same period end (within 3 days), the same duration for
    flows, and the same amount at printed precision? A match means the number
    and its period were read correctly. It says nothing about the label.
    Lines the company tagged with its own extension elements are not in the
    SEC's "companyfacts" feed, so "not confirmed" is an upper bound on error.

Audit 2, concept bindings
    For a fixed map from us-gaap tags to graph concepts: where the SEC has a
    value for (company, tag, period) and the graph has a fact for the mapped
    concept and period, do the two amounts agree? This measures whether rows
    were bound to the right concept.

The SEC asks automated clients to identify themselves; set SEC_USER_AGENT
("Name email") in the environment or in .env.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from collections import defaultdict
from datetime import date

import requests

from ..llm import load_env

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEC_CACHE = ROOT / "data" / "sec"

# us-gaap elements that mean each graph concept (first match with data wins per period)
TAG_MAP = {
    "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "GrossProfit": ["GrossProfit"],
    "ResearchAndDevelopment": ["ResearchAndDevelopmentExpense"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "IncomeTax": ["IncomeTaxExpenseBenefit"],
    "NetIncome": ["NetIncomeLoss"],
    "ConsolidatedNetIncome": ["ProfitLoss", "NetIncomeLoss"],
    "InterestExpense": ["InterestExpense", "InterestExpenseNonoperating"],
    "AccountsReceivable": ["AccountsReceivableNetCurrent"],
    "PropertyPlantEquipment": ["PropertyPlantAndEquipmentNet"],
    "AccountsPayable": ["AccountsPayableCurrent"],
    "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
    "ShareholdersEquity": ["StockholdersEquity"],
    "TotalAssets": ["Assets"],
    "TotalCurrentAssets": ["AssetsCurrent"],
    "TotalLiabilities": ["Liabilities"],
    "TotalCurrentLiabilities": ["LiabilitiesCurrent"],
    "TotalEquity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "TotalLiabilitiesAndEquity": ["LiabilitiesAndStockholdersEquity"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks"],
    "Goodwill": ["Goodwill"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "InvestingCashFlow": ["NetCashProvidedByUsedInInvestingActivities"],
    "FinancingCashFlow": ["NetCashProvidedByUsedInFinancingActivities"],
    "CapitalExpenditures": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "StockBasedCompensation": ["ShareBasedCompensation"],
    "DividendsPaid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "ShareRepurchases": ["PaymentsForRepurchaseOfCommonStock"],
    "NetInterestIncome": ["InterestIncomeExpenseNet"],
    "Deposits": ["Deposits"],
}


def fetch_companyfacts(cik: str) -> dict:
    """Download (once) and cache the SEC's companyfacts JSON for one company."""
    load_env()
    SEC_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEC_CACHE / f"CIK{int(cik):010d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set (SEC asks for a name and email in the User-Agent)")
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
                     headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=120)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    time.sleep(0.3)                                   # SEC fair access: stay far below 10 requests a second
    return r.json()


def _index(facts: dict) -> tuple:
    """(by_end, by_tag): USD facts keyed by period end date, and by (tag, end)."""
    by_end = defaultdict(list)
    by_tag = defaultdict(list)
    for tax in ("us-gaap", "ifrs-full"):
        for tag, body in facts.get("facts", {}).get(tax, {}).items():
            for unit, rows in body.get("units", {}).items():
                if unit != "USD":
                    continue
                for x in rows:
                    rec = (x.get("start"), x["end"], x["val"], tag)
                    by_end[x["end"]].append(rec)
                    by_tag[(tag, x["end"])].append(rec)
    return by_end, by_tag


def _near(by_end: dict, end: str, days: int = 3) -> list:
    d0 = date.fromisoformat(end)
    out = []
    for e, rows in by_end.items():
        if abs((date.fromisoformat(e) - d0).days) <= days:
            out.extend(rows)
    return out


def _months(start, end) -> float:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days / 30.44


def _period_months(pid: str):
    """Length in months implied by a period id; None for an instant."""
    if pid.endswith("@end"):
        return None
    if "Q" not in pid:
        return 12
    tail = pid.split("Q", 1)[1]
    q = int(tail[0])
    return 3 * q if tail.endswith("YTD") else 3


def audit(g) -> dict:
    """Run both audits over every company that has a CIK. Returns a summary dict."""
    per_company = {}
    tot = defaultdict(int)
    concept = defaultdict(lambda: defaultdict(int))
    disagreements = []
    for ticker, comp in sorted(g.companies.items()):
        if not comp.get("cik"):
            continue
        try:
            by_end, by_tag = _index(fetch_companyfacts(comp["cik"]))
        except Exception as e:  # noqa: BLE001 - report and continue
            per_company[ticker] = {"error": f"{type(e).__name__}: {e}"[:120]}
            continue
        c = defaultdict(int)
        scale_of = {tid: t["scale"] for tid, t in g.tables.items()}
        for r in g.readings.values():
            if r.ticker != ticker or r.unit != "USD" or r.value == 0:
                continue
            cands = _near(by_end, r.period_end)
            if not cands:
                c["period_not_in_xbrl"] += 1
                continue
            want_m = _period_months(r.period)
            half = 0.5 * scale_of.get(r.table, 1e6)
            hit = any(abs(abs(v) - abs(r.value)) <= half and
                      (want_m is None or s is None or abs(_months(s, e) - want_m) <= 0.6)
                      for s, e, v, _tag in cands)
            key = "confirmed" if hit else "not_confirmed"
            c[key] += 1
            c[f"{key}_{r.status}"] += 1
        per_company[ticker] = dict(c)
        for k, v in c.items():
            tot[k] += v
        # audit 2: bindings
        for cname, tags in TAG_MAP.items():
            for key, f in g.facts.items():
                t, cn, pid = key.split("|")
                if t != ticker or cn != cname:
                    continue
                rd = g.best_reading(f)
                want_m = _period_months(pid)
                xs = []
                for tag in tags:
                    for s, e, v, _ in _near({k2[1]: by_tag[k2] for k2 in by_tag if k2[0] == tag}, rd.period_end):
                        if want_m is None or s is None or abs(_months(s, e) - want_m) <= 0.6:
                            xs.append(v)
                    if xs:
                        break
                if not xs:
                    concept[cname]["no_xbrl_value"] += 1
                    continue
                half = 0.5 * scale_of.get(rd.table, 1e6)
                if any(abs(abs(v) - abs(rd.value)) <= half for v in xs):
                    concept[cname]["agree"] += 1
                else:
                    concept[cname]["disagree"] += 1
                    disagreements.append({"ticker": ticker, "concept": cname, "period": pid, "graph": rd.value,
                                          "xbrl": xs[:3], "label": rd.label, "doc": rd.doc, "page": rd.page,
                                          "binding": f.binding_state})
    checked = tot["confirmed"] + tot["not_confirmed"]
    agree = sum(v["agree"] for v in concept.values())
    dis = sum(v["disagree"] for v in concept.values())
    return {
        "readings_checked": checked,
        "confirmed": tot["confirmed"], "confirmed_share": tot["confirmed"] / checked if checked else None,
        "by_status": {s: {"confirmed": tot[f"confirmed_{s}"], "not_confirmed": tot[f"not_confirmed_{s}"]}
                      for s in ("verified", "unverified", "failed")},
        "period_not_in_xbrl": tot["period_not_in_xbrl"],
        "bindings_compared": agree + dis, "bindings_agree": agree,
        "bindings_agree_share": agree / (agree + dis) if agree + dis else None,
        "by_concept": {k: dict(v) for k, v in concept.items()},
        "disagreements": disagreements, "per_company": per_company,
    }
