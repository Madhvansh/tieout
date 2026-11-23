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

Audit 3, every other table (the raw layer)
    Audit 1 for each dated cell of the raw layer: money, share counts, other
    counts and per-share amounts, at the precision the cell was printed in.
    "Confirmed" means the SEC holds the same amount for the same end date;
    whether it holds it as a balance or as a flow is counted separately,
    because a table headed only by a year does not say which it is.
    The errors a table parser makes are structural: a whole table read at
    the wrong scale, a whole column given the wrong period. They are judged
    per table and per column, never per cell: the SEC holds tens of
    thousands of amounts per company, so one cell matching some amount a
    thousand times larger is a coincidence, but a table where no cell is
    confirmed and several match at the same factor is a scale error.

Audit 4, numbers in sentences (the mention layer)
    A sentence carries no period column, so a dollar amount stated with a
    scale word ("$416.2 billion") is compared with every value the SEC holds
    for that company within two years before the filing's period end, at the
    precision it was written in.

The SEC asks automated clients to identify themselves; set SEC_USER_AGENT
("Name email") in the environment or in .env.
"""
from __future__ import annotations

import json
import bisect
import os
import pathlib
import re
import time
from collections import Counter, defaultdict
from datetime import date, timedelta

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


# ------------------------------------------------------------------ audits 3 and 4

SEC_UNIT = {"USD": "USD", "shares": "shares", "USD/shares": "USD/share"}
NOT_AMOUNTS = {"pure", "Rate", "Year", "Y", "D", "M", "H"}     # ratios and durations are not compared
FACTORS = (1e3, 1e-3, 1e6, 1e-6)                               # a scale word read wrongly
TESTED_UNITS = ("USD", "shares", "count", "USD/share")
MIN_HITS = 2                                                   # agreeing cells needed to judge a table or column
SCALE_SHARE = 0.3                                              # ... and the share of a table's tested cells
SHIFT_SHARE = 0.5                                              # ... and of a column's cells with four digits


class SecValues:
    """Every amount the SEC holds for one company, by unit, found by period end or by size."""

    def __init__(self, facts: dict):
        self.by_day = defaultdict(lambda: defaultdict(list))    # unit -> day -> [(start, end, value, tag)]
        sized = defaultdict(dict)                                # unit -> {(|value|, start, end, tag): None}
        for tags in facts.get("facts", {}).values():
            for tag, body in tags.items():
                for unit, rows in body.get("units", {}).items():
                    if unit in NOT_AMOUNTS:
                        continue
                    kind = SEC_UNIT.get(unit, "count")           # "employee", "store", "segment": counted things
                    for x in rows:
                        self.by_day[kind][date.fromisoformat(x["end"]).toordinal()].append(
                            (x.get("start"), x["end"], x["val"], tag))
                        sized[kind][(abs(x["val"]), x.get("start"), x["end"], tag)] = None
        self.sized = {u: sorted(v, key=lambda r: r[0]) for u, v in sized.items()}
        self.sizes = {u: [r[0] for r in v] for u, v in self.sized.items()}

    def near(self, unit: str, end: str, days: int = 3) -> list:
        """Values whose period ends within `days` of `end`."""
        day, rows = date.fromisoformat(end).toordinal(), self.by_day.get(unit, {})
        return [r for d in range(day - days, day + days + 1) for r in rows.get(d, [])]

    def sized_between(self, unit: str, lo: float, hi: float) -> list:
        """Values between lo and hi in absolute amount, whatever their date."""
        sizes = self.sizes.get(unit, [])
        return self.sized[unit][bisect.bisect_left(sizes, lo):bisect.bisect_right(sizes, hi)] if sizes else []


def printed_number(text: str):
    """(number as printed, decimals, significant digits) of a printed amount, or None."""
    m = re.search(r"\d[\d,]*(?:\.(\d+))?", text or "")
    if not m:
        return None
    digits = m.group(0).replace(",", "")
    if float(digits) == 0:
        return None
    decimals = len(m.group(1) or "")
    body = digits.replace(".", "").lstrip("0")
    return float(digits), decimals, len(body if decimals else body.rstrip("0"))


def _covers(start, end, months) -> bool:
    """The SEC value spans the period read: an instant for a balance, the same length for a flow."""
    if months is None:
        return start is None
    return start is not None and abs(_months(start, end) - months) <= 0.6


def cell_evidence(value, text: str, pid: str, end: str, unit: str, sec: SecValues):
    """What the SEC holds for one dated value, or None when it cannot be tested.

    confirmed       the same amount for the same end date and the same kind of period
    kind_differs    the same amount and end date, but held as a balance where the cell
                    was read as a flow, or the reverse
    length_differs  the same amount and end date, as a flow of another length
    factor          no amount matches, but one 1,000 or 1,000,000 times larger or smaller
                    does, for that end date
    shift           quarters between the cell's end date and other dates the SEC holds
                    the amount for (cells with four or more significant digits only)
    """
    p = printed_number(text)
    if p is None or not value:
        return None
    number, decimals, digits = p
    v = abs(float(value))
    half = 0.5 * 10 ** -decimals * v / number                 # half a unit of the last printed digit
    months = _period_months(pid)
    near = sec.near(unit, end)
    same = [r for r in near if abs(abs(r[2]) - v) <= half]
    ev = {"confirmed": any(_covers(r[0], r[1], months) for r in same)}
    if ev["confirmed"]:
        return ev
    ev["kind_differs"] = any((r[0] is None) != (months is None) for r in same)
    ev["length_differs"] = not ev["kind_differs"] and bool(same)
    if same or digits < 3:
        return ev
    ev["factor"] = next((k for k in FACTORS if any(abs(abs(r[2]) - v * k) <= half * k for r in near)), None)
    if digits >= 4:
        d0 = date.fromisoformat(end)
        ev["shift"] = sorted({round((d0 - date.fromisoformat(r[2])).days / 91.3)
                              for r in sec.sized_between(unit, v - half, v + half)
                              if 60 <= abs((d0 - date.fromisoformat(r[2])).days) <= 1200})
    return ev


def mention_confirmed(n: dict, period_end: str, sec: SecValues):
    """True when the SEC holds a dollar amount written with a scale word ("$416.2 billion"),
    at the precision it was written in, for a date in the two years up to the filing's
    end; None when the amount cannot be tested (no scale word, under three digits)."""
    p = printed_number(n["text"])
    if p is None or p[2] < 3 or not re.search(r"million|billion|trillion", n["text"], re.I):
        return None
    number, decimals, _ = p
    v = abs(float(n["value"]))
    half = 0.5 * 10 ** -decimals * v / number
    end = date.fromisoformat(period_end)
    lo, hi = (end - timedelta(days=740)).isoformat(), (end + timedelta(days=5)).isoformat()
    return any(lo <= r[2] <= hi for r in sec.sized_between("USD", v - half, v + half))


def _verdict(ev: dict) -> str:
    for key in ("confirmed", "kind_differs", "length_differs"):
        if ev.get(key):
            return key
    return "unmatched"


def audit_raw(g, raw: list, mentions: list) -> dict:
    """Audits 3 (raw-layer cells, judged per table and per column) and 4 (sentence amounts)."""
    cells, by_unit = Counter(), defaultdict(Counter)
    tables = defaultdict(lambda: {"tested": 0, "right": 0, "factor": Counter()})
    columns = defaultdict(lambda: {"tested": 0, "right": 0, "length": 0, "long": 0, "shift": Counter()})
    where = {}
    said = Counter()
    rows_of, mentions_of = defaultdict(list), defaultdict(list)
    for row in raw:
        if not any(c.get("rid") for c in row["cells"]):         # primary statements are audit 1
            rows_of[row["ticker"]].append(row)
    for m in mentions:
        mentions_of[m["ticker"]].append(m)
    for ticker, comp in sorted(g.companies.items()):
        if not comp.get("cik"):
            continue
        sec = SecValues(fetch_companyfacts(comp["cik"]))
        for row in rows_of[ticker]:
            for c in row["cells"]:
                unit = c.get("unit", row["unit"])
                if unit not in TESTED_UNITS or not c.get("pid") or not isinstance(c.get("value"), (int, float)):
                    continue
                ev = cell_evidence(c["value"], c["text"], c["pid"], c["end"], unit, sec)
                if ev is None:
                    continue
                verdict = _verdict(ev)
                cells[verdict] += 1
                by_unit[unit][verdict] += 1
                right = ev["confirmed"] or ev.get("kind_differs", False)
                tid = row.get("table_id") or f"{row['doc']}:p{row['page']}:x{row['table']}"
                t, col = tables[tid], columns[(tid, c.get("ci", c.get("col", "")), c["pid"])]
                where.setdefault(tid, (row["doc"], row["page"], row["title"][:70]))
                t["tested"] += 1
                t["right"] += right
                col["tested"] += 1
                col["right"] += right
                col["length"] += ev.get("length_differs", False)
                col["long"] += "shift" in ev
                if ev.get("factor"):
                    t["factor"][ev["factor"]] += 1
                for s in ev.get("shift", []):
                    col["shift"][s] += 1
        for m in mentions_of[ticker]:
            for n in m["numbers"]:
                if n["unit"] == "USD":
                    ok = mention_confirmed(n, g.filings[m["doc"]]["period_end"], sec)
                    if ok is not None:
                        said["confirmed" if ok else "unmatched"] += 1
    scale_errors = []
    for tid, t in tables.items():
        k, hits = max(t["factor"].items(), key=lambda kv: kv[1], default=(None, 0))
        if t["right"] == 0 and hits >= max(MIN_HITS, SCALE_SHARE * t["tested"]):
            doc, page, title = where[tid]
            scale_errors.append({"table": tid, "doc": doc, "page": page, "title": title, "factor": k,
                                 "cells_at_factor": hits, "tested": t["tested"]})
    period_errors = []
    for (tid, ci, pid), col in columns.items():
        shift, hits = max(col["shift"].items(), key=lambda kv: kv[1], default=(None, 0))
        shifted = hits >= max(MIN_HITS, SHIFT_SHARE * col["long"])
        other_length = col["length"] >= max(MIN_HITS, SHIFT_SHARE * col["tested"])
        if col["right"] == 0 and (shifted or other_length):
            doc, page, title = where[tid]
            period_errors.append({"table": tid, "doc": doc, "page": page, "title": title, "column": ci, "pid": pid,
                                  "quarters_off": shift if shifted else 0, "cells_shifted": hits,
                                  "other_length": col["length"], "tested": col["tested"]})
    tested = sum(cells.values())
    return {
        "cells_tested": tested, "cells": dict(cells),
        "value_and_date_confirmed_share": (cells["confirmed"] + cells["kind_differs"]) / tested if tested else None,
        "by_unit": {u: dict(v) for u, v in by_unit.items()},
        "tables_tested": len(tables), "tables_with_a_confirmed_cell": sum(1 for t in tables.values() if t["right"]),
        "scale_errors": sorted(scale_errors, key=lambda e: (e["doc"], e["page"])),
        "columns_tested": len(columns), "period_errors": sorted(period_errors, key=lambda e: (e["doc"], e["page"])),
        "mentions_tested": sum(said.values()), "mentions": dict(said),
    }
