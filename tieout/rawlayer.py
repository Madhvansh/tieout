"""The raw layer: every other table of every filing, waiting to be bound.

The base graph binds the three primary statements to shared concepts at build
time. Everything else a filing prints in tables (revenue by product, segment
or region, debt maturities, lease costs, store counts, headcount) is parsed and
checked the same way but left unbound: nobody has asked about it yet. Each row
keeps its page, its table (`table_id`), its table title, the note it sits
under, its printed label, and what the recovered formulas say about it:

    role      total / operand / loose, from the formulas recovered in its table
    parent    the row id of the subtotal whose formula includes this row
    children  on a subtotal: the row ids of its operands (with `signs`)
    verified  the row's numbers sit inside a passing formula
    ties_to   a concept of the base graph that prints the same value for the
              same period (the product table's total is Revenue)
    anchor    the concept its table's total ties to, so an operand of that
              total is known to be a component of that concept

Every value cell is kept, dated or not: `col` is its column header, `ci` its
column index, `pid`/`end` its period when the header is one. `value` is in
full units and `unit` is one of USD, shares, count, percent, ratio, USD/share.
Money versus count is decided from evidence on the page: dollar signs in the
table, its caption and header words. A count (stores, employees, square feet)
is never multiplied by the filing's money scale; it takes a scale only from its
own caption, column header or row label ("(in thousands)").

When a question names a line item that is not a seed concept, the living
layer searches these rows by their wording, lets an LLM pick one from the
labels, and admits the binding only on that structural evidence.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from typing import Optional

from . import pdfio
from .alltables import find_all_tables
from .calc import PER_SHARE_RE, discover, row_pool, row_section, verified_cells
from .concepts import norm_label
from .periods import column_period
from .tables import duration, printed_date

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "graph" / "raw.json.gz"
STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "total", "net", "from", "on", "at", "s", "was", "what"}
UNITS = ("USD", "shares", "count", "percent", "ratio", "USD/share")

SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "000": 1e3}
SCALE_WORD_RE = re.compile(r"\b(?:in|of)\s+(thousands|millions|billions)\b|\b(thousands|millions|billions)\s+of\b|"
                           r"\bin\s+(000)['’]?s\b", re.I)
# "(in millions, except remaining performance obligation, which is presented in billions)"
EXCEPT_RE = re.compile(r"except(?:\s+for)?\s+([a-z][\w ,'’-]{2,70}?),?\s+which\s+(?:is|are)\s+"
                       r"(?:presented|reported|stated|shown|expressed|disclosed)\s+in\s+(thousands|millions|billions)", re.I)
DOLLAR_RE = re.compile(r"\$|\bdollars?\b|\bUSD\b")
RATIO_LABEL_RE = re.compile(r"\bratios?\b|\(to 1\)|\btimes\b|margin|\byields?\b|\brates?\b|return on|leverage|"
                            r"\bbps\b|basis points|\bdays\b|percent|%|multiple|coverage", re.I)
COUNT_RE = re.compile(r"\b(?:stores?|store counts?|warehouses?|clubs|locations|retail units|unit counts?|"
                      r"employees|employed|employment|"
                      r"headcount|staff|associates|members|cardholders|vehicles|deliveries|square (?:feet|footage)|"
                      r"sq\.?\s*ft|(?:distribution|fulfillment|supply chain|manufacturing) (?:centers|facilities)|"
                      r"branches|atms|restaurants|subscribers|barrels|wells|acres|number of)\b", re.I)
# things that are counted, not paid; a row label needs a stronger cue than a table
ROW_COUNT_RE = re.compile(r"\bnumber of\b|\bcounts?\b|headcount|\bemployees\b|direct staff|\bstaff \(|\(actual\)|"
                          r"square feet|\bmembers\b|cardholders|\bprocessed transactions\b", re.I)
# a table titled with money words holds money even when it prints no dollar sign
MONEY_TITLE_RE = re.compile(r"expenditure|capital|revenue|\bcosts?\b|expense|income|earnings|cash|dollars|price|fees|"
                            r"compensation|amounts?\b", re.I)
PRICE_COL_RE = re.compile(r"price|fair value|per share|grant date", re.I)
SHARES_TABLE_RE =re.compile(r"\bshares\b|common stock|stock units|\bRSUs?\b|stock options|options outstanding", re.I)
MONEY_WORD_RE = re.compile(r"sales|revenue|cost|expense|income|earnings|asset|liabilit|fee|price|value|amount|capital|"
                           r"cash|payment|compensation|\bper\b|debt|loss|gain|tax|dividend|equity|spend|investment|"
                           r"wage|salar(?:y|ies)\b|benefit|obligation|deposit|loan|rent", re.I)


def _scale_of(text: str) -> Optional[float]:
    m = SCALE_WORD_RE.search(text or "")
    if not m:
        return None
    return SCALES[(m.group(1) or m.group(2) or m.group(3)).lower().rstrip("s")]


def _full(value: float, scale: float):
    """value * scale, as an int when it is a whole number."""
    v = value * scale
    return int(round(v)) if abs(v - round(v)) < 1e-6 * max(1.0, abs(v)) else round(v, 6)


class _Profile:
    """What the page says about the units of one table."""

    def __init__(self, t):
        cells = [c for r in t.rows for c in r.cells.values()]
        text = " ".join([t.title or "", getattr(t, "intro", "") or "", t.caption or "", t.header_text or ""])
        self.dollar = any(c.has_dollar or "$" in c.text for c in cells) or bool(DOLLAR_RE.search(text))
        names = [r.label + " " + r.section for r in t.rows]
        counted = sum(1 for n in names if COUNT_RE.search(n))
        heads = text + " " + " ".join(c.header for c in t.columns)
        self.count = not self.dollar and (bool(COUNT_RE.search(heads)) or counted >= max(1, 0.3 * len(names)))
        # "The following number of shares ...": a table of share counts
        self.shares = self.count and bool(re.search(r"\bshares\b", heads, re.I)) and \
            not COUNT_RE.search(re.sub(r"number of", " ", heads, flags=re.I))
        # a scale printed in a column header (or a caption over some columns) belongs to those columns only
        self.col_scale = {c.idx: getattr(c, "scale", None) or _scale_of(c.header) for c in t.columns
                          if getattr(c, "scale", None) or _scale_of(c.header)}
        own = t.caption + " " + t.header_text + " " + (getattr(t, "intro", "") or "")
        for c in t.columns:
            if c.idx in self.col_scale:
                for piece in (c.header.split("|")[-1].strip(), getattr(c, "scale_text", "")):
                    if piece:
                        own = own.replace(piece, " ")
        m = EXCEPT_RE.search(own + " " + (t.title or ""))
        self.exception = (m.group(1).strip(), SCALES[m.group(2).lower().rstrip("s")]) if m else None
        rest = EXCEPT_RE.sub(" ", own)
        self.own_scale = _scale_of(rest)          # the table's own scale word, outside any exception clause
        self.money_scale = t.scale
        if self.exception and getattr(t, "scale_source", "") != "inherited" and self.own_scale:
            self.money_scale = self.own_scale     # build_table read the exception's scale as the table's
        self.share_scale = t.share_scale


def row_unit(t, r, prof: _Profile) -> str:
    """One of UNITS for a table row, from its own printing and its table's evidence."""
    cells = list(r.cells.values())
    pool = row_pool(t, r)
    row_dollar = any(c.has_dollar for c in cells)
    counted = prof.count or (ROW_COUNT_RE.search(r.label) and not MONEY_WORD_RE.search(r.label) and not row_dollar)
    amount = "shares" if prof.shares else ("count" if counted and not row_dollar else "USD")
    if pool == "shares":
        return "shares"
    if pool == "money":
        return amount
    if cells and sum(1 for c in cells if c.is_percent) * 2 >= len(cells):
        return "percent"
    if PER_SHARE_RE.search(r.label) or PER_SHARE_RE.search(row_section(t, r)):
        return "USD/share"
    if RATIO_LABEL_RE.search(r.label):
        return "ratio"
    # printed decimals that are amounts: next to whole amounts in the same row ("13,427 ... (1.0)" with the
    # variance columns in billions), or in a table or row whose own caption gives a scale ("72.4" in billions)
    mixed = any("." not in c.text and not c.is_percent and c.value for c in cells)
    if mixed or _row_scale(t, r, prof) or prof.col_scale or \
            (prof.money_scale >= 1e9 and getattr(t, "scale_source", "") != "inherited"):
        return amount
    if row_dollar:
        return "USD/share"
    return "ratio"


def _row_scale(t, r, prof: _Profile) -> Optional[float]:
    """A scale that belongs to this row alone: "Direct staff (in thousands)", or the
    exception clause of the caption naming it."""
    s = _scale_of(r.label)
    if s:
        return s
    if prof.exception:
        want = set(tokens(prof.exception[0]))
        if want and want <= set(tokens(r.label)):
            return prof.exception[1]
    return None


def _cell_scale(t, r, ci: int, unit: str, prof: _Profile) -> float:
    own = _row_scale(t, r, prof)
    if own:
        return own
    if unit == "count" or unit == "shares" and prof.shares:
        if ci in prof.col_scale:
            return prof.col_scale[ci]
        # a one-line table is all count: its caption ("in millions") is said about that line
        return (prof.own_scale or 1.0) if prof.count or len(t.rows) == 1 else 1.0
    if unit == "shares":
        return prof.share_scale
    if unit == "USD":
        return prof.col_scale.get(ci, prof.money_scale)
    return 1.0


def table_rows(doc: str, ticker: str, k: int, t, periods: list, anchors: dict, cal=None, form: str = "",
               filing_end: Optional[str] = None) -> list:
    """Rows of one parsed table in the raw-layer shape. With the company's calendar, a date line
    inside the table dates the rows below it (_marked_periods). When two or more cells of one row
    share a period id (segment, region, gross and net columns), each of them also carries `member`:
    its column header without the date words, so a reader can tell the cells apart."""
    checks = discover(t)
    ok = verified_cells(t, checks)
    passing = [ch for ch in checks if ch.status != "fail"]
    totals = {ch.total for ch in passing}
    parent = {row: ch.total for ch in passing for row, _ in ch.operands}
    children = {}
    for ch in passing:
        children.setdefault(ch.total, ch.operands)
    prof = _Profile(t)
    table_id = f"{doc}:p{t.page}:x{k}"
    out = {}
    for r in t.rows:
        unit = row_unit(t, r, prof)
        cells, ties = [], defaultdict(int)
        row_periods = _marked_periods(cal, t, r, periods, form, filing_end) if cal is not None else periods
        for ci in sorted(r.cells):
            cell = r.cells[ci]
            if cell.value is None:
                continue
            pid, end = row_periods[ci]
            cunit = "percent" if cell.is_percent else unit
            if unit in ("count", "shares") and (cell.has_dollar or "." in cell.text and
                                                PRICE_COL_RE.search(t.columns[ci].header + " " + t.header_text)):
                cunit = "USD/share"                     # "Weighted-average grant date fair value" beside a count
            if cunit in ("USD", "count", "shares"):
                value = _full(cell.value, _cell_scale(t, r, ci, cunit, prof))
            else:
                value = cell.value
            c = {"pid": pid, "end": end, "value": value, "text": cell.text, "ok": (r.idx, ci) in ok,
                 "col": t.columns[ci].header[:60], "ci": ci, "bbox": [round(x, 1) for x in cell.bbox]}
            if cunit != unit:
                c["unit"] = cunit
            cells.append(c)
            if cunit == "USD" and end and (end, value) in anchors:
                ties[anchors[(end, value)]] += 1
        if not cells:
            continue
        shared = Counter(c["pid"] for c in cells if c["pid"])
        for c in cells:
            if c["pid"] and shared[c["pid"]] > 1:
                c["member"] = member_of(t.columns[c["ci"]].header)
        out[r.idx] = {
            "id": f"{table_id}:r{r.idx}", "doc": doc, "ticker": ticker, "page": r.page, "table": k,
            "table_id": table_id, "title": (t.title or "")[:160], "context": (getattr(t, "context", "") or "")[:120],
            "label": r.label, "section": r.section, "unit": unit,
            "role": "total" if r.idx in totals else ("operand" if r.idx in parent else "loose"),
            "verified": any(c["ok"] for c in cells),
            "ties_to": max(ties, key=ties.get) if ties else None, "anchor": None,
            "parent": None, "cells": cells}
    for idx, row in out.items():
        if idx in parent and parent[idx] in out:
            row["parent"] = out[parent[idx]]["id"]
        if idx in children:
            ops = [(i, s) for i, s in children[idx] if i in out]
            row["children"] = [out[i]["id"] for i, _ in ops]
            row["signs"] = [s for _, s in ops]
        # an operand inherits the concept its total (or its total's total) ties to
        p, hops = parent.get(idx), 0
        while p is not None and hops < 4:
            if p in out and out[p]["ties_to"]:
                row["anchor"] = out[p]["ties_to"]
                break
            p, hops = parent.get(p), hops + 1
    return list(out.values())


def build_raw(g, docs: Optional[list] = None, verbose: bool = True) -> list:
    """Raw rows of every readable filing, or only of `docs`."""
    rows = []
    for doc, f in g.filings.items():
        if docs is not None and doc not in docs:
            continue
        if not f.get("readable", True):
            continue
        ticker = f["ticker"]
        cal = g.calendars[ticker]
        pages = pdfio.load_pages(doc)
        skip = {p for t in g.tables.values() if t["doc"] == doc for p in t.get("pages", [t["page"]])}
        # values of bound facts in this filing, to recognise a table that restates one of them
        anchors = {}
        for fact in g.facts.values():
            if fact.ticker != ticker:
                continue
            for rid in fact.readings:
                r = g.readings.get(rid)
                if r is not None and r.doc == doc and r.unit == "USD" and abs(r.value) >= 1_000_000:
                    anchors[(r.period_end, r.value)] = fact.concept
        # a table that prints no unit of its own is read in the unit of the filing's statements
        scales = Counter(t["scale"] for t in g.tables.values() if t["doc"] == doc and t.get("scale"))
        default = scales.most_common(1)[0][0] if scales else None
        n0 = len(rows)
        fiscal_year = int(f["fiscal_year"].lstrip("FY"))
        own = defaultdict(set)                   # printed texts of the primary statements, by page
        for r in g.readings.values():
            if r.doc == doc:
                own[r.page].add(r.text)
        base, extra = 0, 0
        for t in find_all_tables(doc, pages, skip_pages=skip, default_scale=default, statement_cells=own):
            # tables the older reader also found keep their number; a one-line table or a table on a
            # statement page is numbered after the table before it ("12e1"), so no older row id moves
            if getattr(t, "added_by", ""):
                extra += 1
                k = f"{base - 1}e{extra}"
            else:
                k, base, extra = base, base + 1, 0
            buckets = _maturity_columns(t, fiscal_year)
            periods = [(None, None) if c.idx in buckets else _period(cal, c, f["form"], f["period_end"], _heading(t))
                       for c in t.columns]
            rows.extend(table_rows(doc, ticker, k, t, periods, anchors, cal, f["form"], f["period_end"]))
        rows.extend(_primary_rows(g, doc))
        if verbose:
            print(f"raw    {doc:22s} {len(rows) - n0:5d} rows", flush=True)
    return rows


def _primary_rows(g, doc: str) -> list:
    """Rows of the three primary statements, in the same shape, so that any printed line
    (not only the seed concepts) can be bound on demand. Cells point at the readings the
    base graph already holds."""
    grouped: dict = {}
    for r in g.readings.values():
        if r.doc == doc and r.kind in ("IS", "BS", "CF"):
            grouped.setdefault((r.table, r.row), []).append(r)
    bound = {(b.kind, b.row): b.concept for b in g.bindings if b.doc == doc and b.state == "active"}
    # formula structure from the graph's SUM checks
    key = {r.id: (r.table, r.row) for rs in grouped.values() for r in rs}
    parent, children = {}, {}
    for ch in g.checks.values():
        if ch.type != "SUM" or ch.status == "fail" or ch.doc != doc:
            continue
        tot = [key[m[0]] for m in ch.members if m[1] == "total" and m[0] in key]
        ops = []
        for rid, sign in ch.members:
            if sign != "total" and rid in key and (key[rid], sign) not in ops:
                ops.append((key[rid], sign))
        if tot:
            children.setdefault(tot[0], ops)
            for o, _ in ops:
                parent.setdefault(o, tot[0])
    rid_of = lambda k: f"{k[0]}:r{k[1]}"
    out = []
    for (table, row), rs in grouped.items():
        rs.sort(key=lambda r: r.col)
        roles = set()
        for r in rs:
            for cid in r.checks:
                ch = g.checks.get(cid)
                if ch is not None and ch.type == "SUM" and ch.status != "fail":
                    roles |= {"total" if role == "total" else "operand" for rid, role in ch.members if rid == r.id}
        first = rs[0]
        cols = g.tables[table].get("columns", [])
        rec = {
            "id": rid_of((table, row)), "doc": doc, "ticker": first.ticker, "page": first.page, "table": first.kind,
            "table_id": table, "title": g.tables[table]["title"][:160], "context": "primary financial statement",
            "label": first.label, "section": first.section,
            "unit": first.unit if first.unit in UNITS else "ratio",
            "role": "total" if "total" in roles else ("operand" if roles else "loose"),
            "verified": any(r.status == "verified" for r in rs),
            "ties_to": bound.get((first.kind, row)), "anchor": None,
            "parent": rid_of(parent[(table, row)]) if (table, row) in parent else None,
            "cells": [{"pid": r.period, "end": r.period_end, "value": r.value, "text": r.text,
                       "ok": r.status == "verified", "col": cols[r.col]["header"][:60] if r.col < len(cols) else "",
                       "ci": r.col, "bbox": r.bbox, "rid": r.id} for r in rs]}
        if (table, row) in children:
            rec["children"] = [rid_of(k) for k, _ in children[(table, row)]]
            rec["signs"] = [s for _, s in children[(table, row)]]
        out.append(rec)
    return out


VINTAGE_RE = re.compile(r"\bvintage|origination year|year of origination|by year originated|term loans by", re.I)


def _maturity_columns(t, fiscal_year: int) -> set:
    """Columns whose years are buckets of a schedule, not periods.

    "2026  2027  2028  2029  2030  Thereafter  Total" in a 2025 report says when amounts owed
    at the report date fall due. A filing reports nothing after its own fiscal year, so years
    that run upward past it are buckets of a schedule; "Thereafter" and "Total" beside them are
    buckets too. Every column of such a table keeps its header as the bucket name and gets no
    period id. In a table of loans by vintage the years say when loans were made.
    """
    years = [c.year for c in t.columns if c.year]
    if VINTAGE_RE.search(" ".join([t.title or "", t.caption or "", t.header_text or ""])):
        return {c.idx for c in t.columns if c.year}           # years of origination, not periods
    if len(years) < 2 or max(years) <= fiscal_year or years != sorted(set(years)):
        return set()
    return {c.idx for c in t.columns}


QUARTER_HEAD_RE = re.compile(r"quarter|\bq[1-4]\b|months|weeks|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _marked_periods(cal, t, r, periods: list, form: str, filing_end: Optional[str] = None) -> list:
    """The periods of one row's columns when a date line sits above the row inside the table.

    "As of December 2024" between two stacked blocks of the same columns dates the second
    block; "2025" over a block of segment columns dates columns the header leaves undated.
    The line replaces the header's period only when the header gives every dated column one
    and the same period, or gives none (and the columns are not quarters or months).
    """
    mark = getattr(r, "period_mark", "")
    years = re.findall(r"\b((?:19|20)\d\d)\b", mark)
    if not years:
        return periods
    dated = {p for p in periods if p[0]}
    if len(dated) > 1 or not dated and any(QUARTER_HEAD_RE.search(c.header) for c in t.columns):
        return periods
    year = int(years[-1])
    out = []
    for c, p in zip(t.columns, periods):
        if dated and not p[0]:
            out.append(p)                       # a column the header leaves undated ("Total") stays so
            continue
        col = replace(c, year=year, end_date=printed_date(mark, year), months=duration(mark) or c.months)
        out.append(_period(cal, col, form, filing_end, _heading(t) + " " + mark))
    return out


DATE_WORD_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?:\s+\d{1,2}\b,?)?"
    r"|\b(?:19|20)\d\d\b|\bq[1-4]\b|\b(?:first|second|third|fourth)\s+quarter\b"
    r"|\b(?:three|six|nine|twelve|3|6|9|12|13|26|39|52|53)\s+(?:months|weeks)\b"
    r"|\b(?:fiscal(?:\s+years?)?|(?:years?|quarters?|months|weeks|periods?)\s+(?:ended|ending)|ended|ending|"
    r"ytd|year-to-date)\b",
    re.I)
EDGE_WORD_RE = re.compile(r"^(?:at|as\s+of|for\s+the|for|and|in|of)\b\s*|\s*\b(?:at|as\s+of|for\s+the|for|and|in|of)$",
                          re.I)


def member_of(header: str) -> str:
    """A column header with its date and year words removed: what the column is a member of
    ("Global Banking | 2025" -> "Global Banking", "September 30, 2025" -> "")."""
    text = DATE_WORD_RE.sub(" ", header or "")
    text = re.sub(r"[|,;:()\[\]⁠]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—.")
    while True:
        cut = EDGE_WORD_RE.sub("", text).strip(" -–—.")
        if cut == text:
            break
        text = cut
    return text if re.search(r"[A-Za-z]", text) else ""


def _period(cal, col, form: str, filing_end: Optional[str] = None, heading: str = "") -> tuple:
    """(period id, ISO end date) of one column: periods.column_period, as a flow."""
    return column_period(cal, col, form, instant=False, filing_end=filing_end, heading=heading)


def _heading(t) -> str:
    """What a table prints above its columns: title, caption, header lines and introducing sentence."""
    return " ".join([t.title or "", t.caption or "", t.header_text or "", getattr(t, "intro", "") or ""])


def save_raw(rows: list, path: pathlib.Path = RAW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f)


def load_raw(path: pathlib.Path = RAW_PATH) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def summary(rows: list) -> dict:
    cells = sum(len(r["cells"]) for r in rows)
    return {"rows": len(rows), "cells": cells, "units": dict(Counter(r["unit"] for r in rows)),
            "roles": dict(Counter(r["role"] for r in rows))}


def tokens(text: str) -> list:
    out = []
    for w in re.findall(r"[a-z0-9&]+", norm_label(text)):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def search(rows: list, ticker: str, doc: str, phrase: str, k: int = 14) -> list:
    """Rows of one filing whose wording overlaps the phrase, best first. Wording only: no values."""
    want = set(tokens(phrase))
    if not want:
        return []
    scored = []
    for r in rows:
        if r["doc"] != doc or not any(c["pid"] for c in r["cells"]):
            continue
        lab = set(tokens(r["label"]))
        ctx = set(tokens(r["section"] + " " + r["title"] + " " + r["context"]))
        hit_l, hit_c = len(want & lab), len(want & ctx)
        if hit_l == 0:
            continue
        score = 3.0 * hit_l + 1.0 * hit_c - 0.15 * len(lab - want) + (0.5 if r["verified"] else 0) \
            + (0.5 if r["anchor"] or r["ties_to"] else 0)
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]
