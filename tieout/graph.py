"""The graph: what it holds and how it is built from the filings.

Node kinds
    Filing, Table        where things are printed
    Reading              one printed number in one cell, with page and bounding box
    Check                a relation the numbers must satisfy
                           SUM    recovered formula inside a statement
                           TIE    the same value printed in two places of one filing
                           CARRY  the same line and period printed in two filings
    Concept, Period      shared across companies; the join points
    Company
    Fact                 (company, concept, period): the thing readings report
    Binding              "this printed row means this concept", with origin and state
    TextFact             a non-numeric fact with its quoted source

Reading states
    verified     inside at least one passing check
    unverified   printed, but no check covers it
    failed       inside a formula that does not hold in its column
A reading contradicted by a later filing is marked superseded; it is kept.
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

from . import pdfio
from .calc import _ints, row_pool, verified_cells
from .concepts import Binding, concept_defs, gate, load_overrides, norm_label, propose
from .periods import FiscalCalendar
from .statements import find_statements, readable

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
GRAPH_PATH = ROOT / "graph" / "graph.json.gz"
MIN_TIE = 1000           # printed units; smaller equal numbers are not evidence of identity


@dataclass
class Reading:
    id: str
    doc: str
    ticker: str
    page: int
    bbox: list
    table: str
    kind: str
    row: int
    col: int
    label: str
    section: str
    text: str                 # as printed
    value: float              # full units (USD, shares) or the printed ratio
    unit: str                 # USD | shares | USD/share | percent | ratio
    period: str               # period id on the company's fiscal calendar
    period_end: str
    status: str = "unverified"
    checks: list = field(default_factory=list)
    superseded: bool = False


@dataclass
class Check:
    id: str
    type: str                 # SUM | TIE | CARRY
    status: str               # pass | rounded | fail | restated
    doc: str
    members: list             # [[reading id, sign or role], ...]
    note: str = ""


@dataclass
class Fact:
    ticker: str
    concept: str
    period: str
    readings: list = field(default_factory=list)      # reading ids, any filing
    binding_state: str = "active"


class Graph:
    def __init__(self):
        self.filings: dict = {}
        self.companies: dict = {}
        self.calendars: dict = {}
        self.tables: dict = {}
        self.readings: dict = {}
        self.checks: dict = {}
        self.bindings: list = []
        self.facts: dict = {}
        self.textfacts: list = []

    # ------------------------------------------------------------- lookups

    def fact(self, ticker: str, concept: str, period: str) -> Optional[Fact]:
        return self.facts.get(f"{ticker}|{concept}|{period}")

    def best_reading(self, fact: Fact, as_first_reported: bool = False) -> Optional[Reading]:
        """The reading that speaks for a fact.

        By default the latest filing wins (as latest reported); with
        as_first_reported the earliest filing wins. Inside one filing a
        verified reading is preferred.
        """
        rs = [self.readings[i] for i in fact.readings]
        if not rs:
            return None
        order = {d: self.filings[d]["period_end"] for d in {r.doc for r in rs}}
        rs.sort(key=lambda r: (order[r.doc], r.status == "verified"), reverse=not as_first_reported)
        if as_first_reported:
            first_doc = rs[0].doc
            same = [r for r in rs if r.doc == first_doc]
            same.sort(key=lambda r: r.status != "verified")
            return same[0]
        return rs[0]

    def periods_of(self, ticker: str, concept: str) -> list:
        pre = f"{ticker}|{concept}|"
        return sorted(k[len(pre):] for k in self.facts if k.startswith(pre))

    # ----------------------------------------------------------- persistence

    def save(self, path: pathlib.Path = GRAPH_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "filings": self.filings, "companies": self.companies, "tables": self.tables,
            "readings": [asdict(r) for r in self.readings.values()],
            "checks": [asdict(c) for c in self.checks.values()],
            "bindings": [b.to_dict() for b in self.bindings],
            "facts": [asdict(f) for f in self.facts.values()],
            "textfacts": self.textfacts,
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: pathlib.Path = GRAPH_PATH) -> "Graph":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            blob = json.load(f)
        g = cls()
        g.filings, g.companies, g.tables = blob["filings"], blob["companies"], blob["tables"]
        g.readings = {r["id"]: Reading(**r) for r in blob["readings"]}
        g.checks = {c["id"]: Check(**c) for c in blob["checks"]}
        g.bindings = [Binding(**b) for b in blob["bindings"]]
        g.facts = {f"{f['ticker']}|{f['concept']}|{f['period']}": Fact(**f) for f in blob["facts"]}
        g.textfacts = blob.get("textfacts", [])
        for t, c in g.companies.items():
            g.calendars[t] = FiscalCalendar(t, c["fye_month"], c["fye_day"], c["fy_offset"])
        return g

    def stats(self) -> dict:
        rs = list(self.readings.values())
        by = defaultdict(int)
        for r in rs:
            by[r.status] += 1
        ck = defaultdict(int)
        for c in self.checks.values():
            ck[f"{c.type}:{c.status}"] += 1
        return {"filings": len(self.filings), "companies": len(self.companies), "tables": len(self.tables),
                "readings": len(rs), "reading_status": dict(by), "checks": dict(ck),
                "facts": len(self.facts), "bindings": len(self.bindings),
                "bindings_active": sum(1 for b in self.bindings if b.state == "active"),
                "textfacts": len(self.textfacts)}


# ------------------------------------------------------------------- build

def load_manifest() -> list:
    return list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))


def build(doc_ids: Optional[list] = None, verbose: bool = True, learned: bool = True,
          corrupt: Optional[tuple] = None) -> Graph:
    """Build the graph. `corrupt=(rate, seed)` damages that share of statement cells
    before the checks run; it exists only for the noisy-graph experiment."""
    g = Graph()
    g._corrupt = corrupt
    rows = [r for r in load_manifest() if pdfio.cache_path(r["doc_id"]).exists()]
    for r in rows:                                    # fiscal calendars come from each latest 10-K
        if r["role"] == "latest":
            cal = FiscalCalendar.from_latest(r["ticker"], r["fiscal_year"], r["period_end"])
            g.calendars[r["ticker"]] = cal
            g.companies[r["ticker"]] = {"name": r["company"], "cluster": r["cluster"], "cik": r["cik"],
                                        "fye_month": cal.fye_month, "fye_day": cal.fye_day,
                                        "fy_offset": cal.offset}
    if doc_ids:
        rows = [r for r in rows if r["doc_id"] in doc_ids or r["ticker"] in doc_ids]
    for r in rows:
        _add_filing(g, r, verbose)
    _carry_checks(g)
    _apply_overrides(g)
    _load_textfacts(g)
    if learned:
        from .living import load_learned
        load_learned(g)                               # semantics learned from earlier questions
    return g


def _column_period(cal: FiscalCalendar, col, kind: str, form: str) -> tuple:
    """(period id, ISO end date) for one table column, or (None, None)."""
    instant = kind == "BS"
    if col.end_date:
        try:
            end = date.fromisoformat(col.end_date)
        except ValueError:
            return None, None
    elif col.year:
        end = cal.end_of_label(col.year)              # header printed only a fiscal-year label
    else:
        return None, None
    months = None if instant else (col.months or (12 if form == "10-K" else None))
    return cal.period_id(end, months, instant), end.isoformat()


def _add_filing(g: Graph, m: dict, verbose: bool) -> None:
    doc, ticker = m["doc_id"], m["ticker"]
    cal = g.calendars.get(ticker)
    if cal is None:
        return
    g.filings[doc] = {"ticker": ticker, "form": m["form"], "fiscal_year": m["fiscal_year"],
                      "period_end": m["period_end"], "url": m["pdf_url"], "sha256": m.get("sha256", ""),
                      "role": m["role"]}
    pages = pdfio.load_pages(doc)
    g.filings[doc]["readable"] = readable(pages)
    if not g.filings[doc]["readable"]:
        # fonts without a Unicode map: the text layer is glyph codes. Say so; extract nothing.
        if verbose:
            print(f"{doc:22s} UNREADABLE text layer (fonts carry no Unicode map); no facts extracted", flush=True)
        return
    statements = find_statements(doc, pages)
    if getattr(g, "_corrupt", None):
        from .eval.noise import corrupt_statements
        g.filings[doc]["corrupted_cells"] = corrupt_statements(statements, doc, *g._corrupt)
    proposals = {k: propose(st) for k, st in statements.items()}
    bindings, ratio_checks = gate(doc, statements, proposals)
    g.bindings.extend(bindings)
    cell_id: dict = {}
    for kind, st in statements.items():
        t = st.table
        tid = f"{doc}:p{t.page}:{kind}"
        g.tables[tid] = {"doc": doc, "page": t.page, "kind": kind, "title": t.title, "scale": t.scale,
                         "columns": [{"header": c.header, "end": c.end_date, "months": c.months} for c in t.columns],
                         "verified": st.verified, "cells": st.cells}
        ok = verified_cells(t, st.checks)
        bad = {(row, c) for ch in st.checks if ch.status == "fail" for row in ch.rows for c in ch.failing_cols}
        periods = [_column_period(cal, c, kind, m["form"]) for c in t.columns]
        for r in t.rows:
            pool = row_pool(t, r)
            for ci, cell in r.cells.items():
                pid, end = periods[ci]
                if cell.value is None or pid is None:
                    continue
                if pool == "money":
                    value, unit = int(round(cell.value * t.scale)), "USD"
                elif pool == "shares":
                    value, unit = int(round(cell.value * t.share_scale)), "shares"
                else:
                    value = cell.value
                    unit = "percent" if cell.is_percent else ("USD/share" if "share" in (r.section + r.label).lower() else "ratio")
                rid = f"{doc}:p{t.page}:{kind}:r{r.idx}:c{ci}"
                status = "verified" if (r.idx, ci) in ok else ("failed" if (r.idx, ci) in bad else "unverified")
                g.readings[rid] = Reading(id=rid, doc=doc, ticker=ticker, page=cell.page, bbox=[round(x, 1) for x in cell.bbox],
                                          table=tid, kind=kind, row=r.idx, col=ci, label=r.label, section=r.section,
                                          text=cell.text, value=value, unit=unit, period=pid, period_end=end, status=status)
                cell_id[(kind, r.idx, ci)] = rid
        for n, ch in enumerate(st.checks):            # SUM checks, one per column that could be evaluated
            members = [[cell_id.get((kind, ch.total, c)), "total"] for c in range(t.ncols)]
            cid = f"{tid}:sum{n}"
            mem = []
            for c in range(t.ncols):
                if ch.residuals[c] is None or (kind, ch.total, c) not in cell_id:
                    continue
                mem.append([cell_id[(kind, ch.total, c)], "total"])
                for row, sign in ch.operands:
                    if (kind, row, c) in cell_id:
                        mem.append([cell_id[(kind, row, c)], sign])
            if not mem:
                continue
            g.checks[cid] = Check(id=cid, type="SUM", status=ch.status, doc=doc, members=mem,
                                  note=f"{t.rows[ch.total].label[:60]} = " + " ".join(
                                      ("+" if s > 0 else "-") + t.rows[i].label[:30] for i, s in ch.operands)[:300])
            for rid, _ in mem:
                g.readings[rid].checks.append(cid)
            del members
        for n, (a, b) in enumerate(st.ties):          # TIE inside one statement
            mem = [[cell_id[(kind, row, c)], "same"] for row in (a, b) for c in range(t.ncols)
                   if (kind, row, c) in cell_id]
            if mem:
                cid = f"{tid}:tie{n}"
                g.checks[cid] = Check(id=cid, type="TIE", status="pass", doc=doc, members=mem,
                                      note=f"{t.rows[a].label[:50]} == {t.rows[b].label[:50]}")
                for rid, _ in mem:
                    g.readings[rid].checks.append(cid)
                    g.readings[rid].status = "verified"
    for n, (kind, rc) in enumerate(ratio_checks):       # per-share identity: NI / diluted shares = diluted EPS
        mem = [[cell_id[(kind, row, c)], role] for c in rc["cols"]
               for row, role in zip(rc["rows"], ("eps", "net_income", "shares")) if (kind, row, c) in cell_id]
        cid = f"{doc}:ratio{n}"
        g.checks[cid] = Check(id=cid, type="RATIO", status="pass", doc=doc, members=mem,
                              note="net income / diluted shares = diluted EPS (pins the income-statement scale)")
        for rid, _ in mem:
            g.readings[rid].checks.append(cid)
            g.readings[rid].status = "verified"
    _cross_statement_ties(g, doc)
    # facts: every bound row contributes its readings
    for b in bindings:
        st = statements[b.kind]
        for ci in st.table.rows[b.row].cells:
            rid = cell_id.get((b.kind, b.row, ci))
            if rid is None:
                continue
            rd = g.readings[rid]
            key = f"{ticker}|{b.concept}|{rd.period}"
            f = g.facts.setdefault(key, Fact(ticker, b.concept, rd.period, [], b.state))
            f.readings.append(rid)
            if b.state == "active":
                f.binding_state = "active"
    if verbose:
        parts = [f"{k} p{st.table.page} {st.verified}/{st.cells}" for k, st in statements.items()]
        act = sum(1 for b in bindings if b.state == "active")
        print(f"{doc:22s} {' | '.join(parts):62s} bindings {act}/{len(bindings)}", flush=True)


def _cross_statement_ties(g: Graph, doc: str) -> None:
    """The same large number for the same period end in two statements of one filing.

    Net income on the income statement and the cash-flow statement, cash on
    the balance sheet and at the foot of the cash-flow statement.
    """
    groups = defaultdict(list)
    for r in g.readings.values():
        if r.doc == doc and r.unit == "USD" and abs(r.value) >= MIN_TIE * 1000:
            groups[(r.period_end, r.value)].append(r)
    n = 0
    for (end, value), rs in groups.items():
        if len({r.kind for r in rs}) < 2 or len(rs) > 4:
            continue
        cid = f"{doc}:xtie{n}"
        n += 1
        g.checks[cid] = Check(id=cid, type="TIE", status="pass", doc=doc,
                              members=[[r.id, "same"] for r in rs],
                              note=" == ".join(f"{r.kind}:{r.label[:36]}" for r in rs))
        for r in rs:
            r.checks.append(cid)
            r.status = "verified"


def _carry_checks(g: Graph) -> None:
    """The same line and period printed in two filings of one company.

    Equal values corroborate each other. Different values mean the company
    restated or reclassified the figure; the older reading is superseded and
    the pair is recorded, never dropped.
    """
    groups = defaultdict(list)
    for r in g.readings.values():
        if r.unit in ("USD", "shares"):
            groups[(r.ticker, r.kind, norm_label(r.label), norm_label(r.section), r.period)].append(r)
    n = 0
    for key, rs in groups.items():
        docs = {r.doc for r in rs}
        if len(docs) < 2:
            continue
        by_doc = {}
        for r in rs:
            by_doc.setdefault(r.doc, r)
        ordered = sorted(by_doc.values(), key=lambda r: g.filings[r.doc]["period_end"])
        same = len({r.value for r in ordered}) == 1
        cid = f"carry{n}"
        n += 1
        g.checks[cid] = Check(id=cid, type="CARRY", status="pass" if same else "restated", doc=ordered[-1].doc,
                              members=[[r.id, g.filings[r.doc]["fiscal_year"]] for r in ordered],
                              note=f"{key[0]} {key[2][:50]} {key[4]}: " + " -> ".join(r.text for r in ordered))
        for r in ordered:
            r.checks.append(cid)
            if same:
                r.status = "verified"
        if not same:
            for r in ordered[:-1]:
                r.superseded = True


def _apply_overrides(g: Graph) -> None:
    """Human-approved corrections from semantics/overrides.yaml."""
    for o in load_overrides():
        for b in g.bindings:
            if b.doc == o["doc"] and b.kind == o["kind"] and b.concept == o["concept"]:
                b.state = "rejected"
                b.evidence.append(f"replaced by human override: {o.get('reason', '')}")
        # the override names a printed label; bind every reading of that row
        want = norm_label(o["label"])
        rows = {(r.table, r.row) for r in g.readings.values()
                if r.doc == o["doc"] and r.kind == o["kind"] and norm_label(r.label) == want}
        for table, row in rows:
            g.bindings.append(Binding(doc=o["doc"], kind=o["kind"], row=row, label=o["label"],
                                      concept=o["concept"], origin="human", state="active",
                                      evidence=[f"human override: {o.get('reason', '')} (page {o.get('page', '?')})"]))
            ticker = g.filings[o["doc"]]["ticker"]
            for key in [k for k in g.facts if k.startswith(f"{ticker}|{o['concept']}|")]:
                g.facts[key].readings = [i for i in g.facts[key].readings if g.readings[i].doc != o["doc"]]
            for r in g.readings.values():
                if r.table == table and r.row == row:
                    f = g.facts.setdefault(f"{ticker}|{o['concept']}|{r.period}",
                                           Fact(ticker, o["concept"], r.period, [], "active"))
                    f.readings.append(r.id)
                    f.binding_state = "active"
    for key in [k for k, f in g.facts.items() if not f.readings]:
        del g.facts[key]


def _load_textfacts(g: Graph) -> None:
    p = ROOT / "data" / "textfacts.json"
    if p.exists():
        g.textfacts = [t for t in json.loads(p.read_text(encoding="utf-8")) if t["doc"] in g.filings]
