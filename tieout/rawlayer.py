"""The raw layer: every other table of every filing, waiting to be bound.

The base graph binds the three primary statements to shared concepts at build
time. Everything else a filing prints in tables (revenue by product, segment
or region, debt maturities, lease costs, tax reconciliations) is parsed and
checked the same way but left unbound: nobody has asked about it yet. Each
row keeps its page, its table title, the note it sits under, its printed
label, and what the recovered formulas say about it:

    role      total / operand / loose, from the formulas recovered in its table
    verified  the row's numbers sit inside a passing formula
    ties_to   a concept of the base graph that prints the same value for the
              same period (the product table's total is Revenue)
    anchor    the concept its table's total ties to, so an operand of that
              total is known to be a component of that concept

When a question names a line item that is not a seed concept, the living
layer searches these rows by their wording, lets an LLM pick one from the
labels, and admits the binding only on that structural evidence.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
from collections import defaultdict
from datetime import date

from . import pdfio
from .alltables import find_all_tables
from .calc import _ints, discover, row_pool, verified_cells
from .concepts import norm_label

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "graph" / "raw.json.gz"
STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "total", "net", "from", "on", "at", "s", "was", "what"}


def build_raw(g, verbose: bool = True) -> list:
    rows = []
    for doc, f in g.filings.items():
        if not f.get("readable", True):
            continue
        ticker = f["ticker"]
        cal = g.calendars[ticker]
        pages = pdfio.load_pages(doc)
        skip = {t["page"] for t in g.tables.values() if t["doc"] == doc}
        # values of bound facts in this filing, to recognise a table that restates one of them
        anchors = {}
        for fact in g.facts.values():
            if fact.ticker != ticker:
                continue
            for rid in fact.readings:
                r = g.readings[rid]
                if r.doc == doc and r.unit == "USD" and abs(r.value) >= 1_000_000:
                    anchors[(r.period_end, r.value)] = fact.concept
        n0 = len(rows)
        for k, t in enumerate(find_all_tables(doc, pages, skip_pages=skip)):
            checks = discover(t)
            ok = verified_cells(t, checks)
            totals = {ch.total for ch in checks if ch.status != "fail"}
            parent = {row: ch.total for ch in checks if ch.status != "fail" for row, _ in ch.operands}
            periods = [_period(cal, c, f["form"]) for c in t.columns]
            table_rows = {}
            for r in t.rows:
                pool = row_pool(t, r)
                cells, ties = [], defaultdict(int)
                for ci, cell in r.cells.items():
                    pid, end = periods[ci]
                    if cell.value is None:
                        continue
                    if pool == "money":
                        value, unit = int(round(cell.value * t.scale)), "USD"
                    elif pool == "shares":
                        value, unit = int(round(cell.value * t.share_scale)), "shares"
                    else:
                        value, unit = cell.value, ("percent" if cell.is_percent else "ratio")
                    cells.append({"pid": pid, "end": end, "value": value, "text": cell.text, "ok": (r.idx, ci) in ok,
                                  "col": t.columns[ci].header[:40], "bbox": [round(x, 1) for x in cell.bbox]})
                    if end and (end, value) in anchors:
                        ties[anchors[(end, value)]] += 1
                if not cells:
                    continue
                table_rows[r.idx] = {
                    "id": f"{doc}:p{t.page}:x{k}:r{r.idx}", "doc": doc, "ticker": ticker, "page": r.page,
                    "table": k, "title": (t.title or "")[:160], "context": (getattr(t, "context", "") or "")[:120],
                    "label": r.label, "section": r.section, "unit": cells[0]["value"] is not None and
                    ("USD" if pool == "money" else "shares" if pool == "shares" else "ratio"),
                    "role": "total" if r.idx in totals else ("operand" if r.idx in parent else "loose"),
                    "verified": any(c["ok"] for c in cells),
                    "ties_to": max(ties, key=ties.get) if ties else None, "anchor": None, "cells": cells}
            # an operand inherits the concept its total (or its total's total) ties to
            for idx, row in table_rows.items():
                p, hops = parent.get(idx), 0
                while p is not None and hops < 4:
                    if p in table_rows and table_rows[p]["ties_to"]:
                        row["anchor"] = table_rows[p]["ties_to"]
                        break
                    p, hops = parent.get(p), hops + 1
            rows.extend(table_rows.values())
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
    out = []
    for (table, row), rs in grouped.items():
        rs.sort(key=lambda r: r.col)
        roles = set()
        for r in rs:
            for cid in r.checks:
                ch = g.checks[cid]
                if ch.type == "SUM" and ch.status != "fail":
                    roles |= {"total" if role == "total" else "operand" for rid, role in ch.members if rid == r.id}
        first = rs[0]
        out.append({
            "id": f"{table}:r{row}", "doc": doc, "ticker": first.ticker, "page": first.page, "table": first.kind,
            "title": g.tables[table]["title"][:160], "context": "primary financial statement",
            "label": first.label, "section": first.section,
            "unit": "USD" if first.unit == "USD" else ("shares" if first.unit == "shares" else "ratio"),
            "role": "total" if "total" in roles else ("operand" if roles else "loose"),
            "verified": any(r.status == "verified" for r in rs),
            "ties_to": bound.get((first.kind, row)), "anchor": None,
            "cells": [{"pid": r.period, "end": r.period_end, "value": r.value, "text": r.text,
                       "ok": r.status == "verified", "col": "", "bbox": r.bbox, "rid": r.id} for r in rs]})
    return out


def _period(cal, col, form: str) -> tuple:
    if col.end_date:
        try:
            end = date.fromisoformat(col.end_date)
        except ValueError:
            return None, None
    elif col.year and 1995 <= col.year <= 2035:
        end = cal.end_of_label(col.year)
    else:
        return None, None
    return cal.period_id(end, col.months or (12 if form == "10-K" else None), False), end.isoformat()


def save_raw(rows: list, path: pathlib.Path = RAW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f)


def load_raw(path: pathlib.Path = RAW_PATH) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


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
