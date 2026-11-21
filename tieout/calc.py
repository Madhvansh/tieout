"""Recover the erased formulas of a statement from its numbers.

A printed statement is a spreadsheet whose formulas were dropped on export.
The numbers still over-determine each other: a subtotal equals a signed sum of
rows above it in every period column at once. Two or three columns of five and
six digit numbers do not agree by accident, so a formula that holds in all
columns is accepted as structure. No labels, no LLM and no XBRL are used.

Three shapes cover real statements (O is the list of rows not yet absorbed by
a subtotal, in print order):

    sum         r = O[-k] + ... + O[-1]            Total current assets
    base_minus  r = O[-(k+1)] - (O[-k] + ... )     Gross margin, Operating income
    signed      r = +/-O[-k] ... +/-O[-1], k <= 6  rare; needs an exact match

All arithmetic is on integers in printed units.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional

from .tables import Table

MAX_SUM_RUN = 30
MAX_BASE_RUN = 12
MAX_SIGNED_RUN = 6
MIN_TOTAL = 50          # a "total" smaller than this (printed units) is not evidence

PER_SHARE_RE = re.compile(r"per\s+(?:common\s+|ordinary\s+|diluted\s+|basic\s+)?share|per-share|earnings per|\beps\b", re.I)
# share-count rows only; an equity line that merely mentions "shares authorized" is money
SHARES_RE = re.compile(r"weighted[- ]average|shares used in|shares outstanding|number of shares|"
                       r"dilutive|^\s*(basic|diluted)\s*$", re.I)
NOT_SHARES_RE = re.compile(r"par value|authorized|paid-in|issued", re.I)


@dataclass
class SumCheck:
    total: int                       # row index of the subtotal
    operands: list                   # [(row index, sign)]
    form: str                        # sum | base_minus | signed
    residuals: list                  # per column: total - combination, None if unknown
    status: str                      # pass | rounded | fail
    failing_cols: list = field(default_factory=list)

    @property
    def rows(self) -> list:
        return [self.total] + [r for r, _ in self.operands]


def row_pool(table: Table, r) -> str:
    """money rows sum with money rows; per-share and share-count rows do not.

    The row's own label decides. An equity line such as "Common stock, par
    value $1.00 per share (authorized 4,320,000,000 shares)" is money even
    though it mentions shares and a per-share amount. The section heading is
    only consulted for the bare sub-rows "Basic" and "Diluted".
    """
    cells = list(r.cells.values())
    if any(c.is_percent for c in cells) or any("." in c.text for c in cells):
        return "ratio"                              # printed decimals: per-share amounts, ratios
    if NOT_SHARES_RE.search(r.label):
        return "money"
    if PER_SHARE_RE.search(r.label):
        return "ratio"
    bare = re.fullmatch(r"\s*(basic|diluted)\b.{0,12}", r.label, re.I)
    if bare and PER_SHARE_RE.search(r.section) and not re.search(r"shares|weighted", r.section, re.I):
        return "ratio"
    if (SHARES_RE.search(r.label) or (bare and re.search(r"shares|weighted", r.section, re.I))) \
            and not any(c.has_dollar for c in cells):
        return "shares"
    return "money"


def _ints(table: Table, r) -> list:
    out = []
    for c in range(table.ncols):
        cell = r.cells.get(c)
        out.append(None if cell is None or cell.value is None else int(round(cell.value)))
    return out


def _combine(terms: list, ncols: int) -> list:
    """Signed sum per column. A blank operand cell is a printed zero."""
    out = [0] * ncols
    for vals, sign in terms:
        for c in range(ncols):
            if vals[c] is not None:
                out[c] += sign * vals[c]
    return out


def _judge(total: list, combo: list, n_terms: int):
    """Compare a candidate formula with the printed total, column by column."""
    res = [None if t is None else t - s for t, s in zip(total, combo)]
    known = [x for x in res if x is not None]
    if not known:
        return None
    tol = max(1, (n_terms + 1) // 2)        # rounding: each rounded term can be off by half a unit
    exact = sum(1 for x in known if x == 0)
    near = sum(1 for x in known if abs(x) <= tol)
    return res, exact, near, len(known)


def discover(table: Table) -> list:
    """Return the SumChecks found in one table, in print order."""
    ncols = table.ncols
    pools: dict = {"money": [], "shares": []}      # open rows per pool: [(row idx, ints)]
    checks: list = []
    seq = [(r, row_pool(table, r), _ints(table, r)) for r in table.rows]
    for pos, (r, pool, vals) in enumerate(seq):
        if pool not in pools:
            continue
        if all(v is None for v in vals):
            continue
        open_rows = pools[pool]
        best = _explain(vals, open_rows, ncols)
        if best is not None and best[0] != "sum" and best[4] != "fail":
            # Any accounting identity can be rearranged to "explain" a leaf
            # (AOCI = assets - liabilities - stock - retained earnings). If a
            # plain sum a few rows below needs this row and one of the rows the
            # rearranged formula would absorb, this row is a leaf of that sum.
            later = [(r2.idx, v2) for r2, p2, v2 in seq[pos + 1:pos + 9]
                     if p2 == pool and not all(v is None for v in v2)]
            if _is_leaf_of_later_sum(r.idx, vals, open_rows, best[1], later, ncols):
                best = None
        if best is not None:
            form, k, signs, res, status, failing = best
            used = open_rows[-k:]
            checks.append(SumCheck(total=r.idx,
                                   operands=[(idx, s) for (idx, _), s in zip(used, signs)],
                                   form=form, residuals=res, status=status, failing_cols=failing))
            if status != "fail":
                del open_rows[-k:]
        open_rows.append((r.idx, vals))
    return checks


def _is_leaf_of_later_sum(idx: int, vals: list, open_rows: list, k: int, later: list, ncols: int) -> bool:
    absorbed = {i for i, _ in open_rows[-k:]}
    trial = list(open_rows) + [(idx, vals)]
    for idx2, vals2 in later:
        got = _explain(vals2, trial, ncols)
        if got is not None and got[0] == "sum" and got[4] != "fail":
            used = {i for i, _ in trial[-got[1]:]}
            if idx in used and used & absorbed:
                return True
        trial.append((idx2, vals2))
    return False


def ties(table: Table, min_abs: int = 1000) -> list:
    """Pairs of rows printed with identical values in every column.

    Total assets and Total liabilities and equity is the usual one.
    """
    seen: dict = {}
    out = []
    for r in table.rows:
        if row_pool(table, r) != "money":
            continue
        vals = tuple(_ints(table, r))
        if any(v is None for v in vals) or min(abs(v) for v in vals) < min_abs:
            continue
        if vals in seen:
            out.append((seen[vals], r.idx))
        else:
            seen[vals] = r.idx
    return out


def _explain(vals: list, open_rows: list, ncols: int):
    """Find the formula over the open rows that reproduces this row, if any."""
    big = max((abs(v) for v in vals if v is not None), default=0)
    if big < MIN_TOTAL or len(open_rows) < 2:
        return None
    candidates = []          # (rank, form, k, signs, residuals, status, failing)

    def consider(form, k, signs, order):
        terms = [(v, s) for (_, v), s in zip(open_rows[-k:], signs)]
        nonzero = sum(1 for v, _ in terms if any(x for x in v if x))
        if nonzero < 2:
            return
        j = _judge(vals, _combine(terms, ncols), len(terms))
        if j is None:
            return
        res, exact, near, known = j
        if near == known:
            # a single known column is weak evidence: demand an exact, large match
            if known == 1 and (exact != 1 or big < 1000):
                return
            if form == "signed" and exact != known:
                return
            status = "pass" if exact == known else "rounded"
            candidates.append(((0, 0 if status == "pass" else 1, order, k),
                               form, k, signs, res, status, []))
        elif known >= 3 and near == known - 1 and form != "signed":
            failing = [c for c, x in enumerate(res) if x is not None
                       and abs(x) > max(1, (len(terms) + 1) // 2)]
            candidates.append(((1, 0, order, k), form, k, signs, res, "fail", failing))

    n = len(open_rows)
    for k in range(2, min(n, MAX_SUM_RUN) + 1):
        consider("sum", k, [1] * k, 0)
    for k in range(2, min(n, MAX_BASE_RUN + 1) + 1):
        consider("base_minus", k, [1] + [-1] * (k - 1), 1)
    for k in range(3, min(n, MAX_SIGNED_RUN) + 1):
        for tail in itertools.product((1, -1), repeat=k - 1):
            signs = [1, *tail]
            if all(s == 1 for s in signs) or all(s == -1 for s in tail):
                continue
            consider("signed", k, signs, 2)
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    top = candidates[0]
    if top[5] == "fail":
        # report a broken formula only when it is the single near-miss
        if sum(1 for c in candidates if c[5] == "fail") != 1:
            return None
    return top[1:]


def verified_cells(table: Table, checks: list) -> set:
    """(row, col) pairs that sit inside at least one passing formula."""
    ok = set()
    for ch in checks:
        if ch.status == "fail":
            continue
        for row in ch.rows:
            for c, res in enumerate(ch.residuals):
                if res is not None and c in table.rows[row].cells:
                    ok.add((row, c))
    return ok


def coverage(table: Table, checks: list) -> tuple:
    """(verified, total) over money cells. Per-share and share-count rows
    do not add up to anything, so they are not part of this measure."""
    ok = verified_cells(table, checks)
    total = 0
    for r in table.rows:
        if row_pool(table, r) == "money":
            total += sum(1 for c in r.cells.values() if c.value is not None)
    hit = sum(1 for (row, c) in ok if row_pool(table, table.rows[row]) == "money")
    return hit, total


def explain(table: Table, checks: list) -> str:
    lines = []
    for ch in checks:
        tot = table.rows[ch.total].label
        parts = " ".join(("+ " if s > 0 else "- ") + table.rows[i].label[:26] for i, s in ch.operands)
        lines.append(f"  [{ch.status:7s}] {tot[:44]:44s} = {parts[:150]}  res={ch.residuals}")
    return "\n".join(lines)
