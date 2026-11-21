"""How much does the check layer really catch? Inject known faults and count.

Text PDFs rarely misread a digit. The errors a table parser makes are
structural: a value lands in the wrong period column, a parenthesis is lost
and the sign flips, a wrapped row is dropped, two rows trade values, one
cell or the whole table is read at the wrong scale. Each of those is
injected into the real parsed statements, the formulas are recovered again
from scratch, and the corrupted cell is looked up:

    caught   the cell is no longer inside any passing formula
             (it is unverified or sits in a failing one), so the graph
             would flag it or refuse to answer from it
    silent   the cell is still verified: the fault got through

Only cells that were verified before the fault are used, so the rate is the
share of faults that survive on numbers the graph vouches for.
"""
from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

from ..calc import discover, row_pool, ties, verified_cells

FAULTS = ("sign_flip", "digit_error", "column_shift", "row_swap", "drop_row", "cell_scale", "table_scale")


def _verified(table) -> set:
    checks = discover(table)
    ok = verified_cells(table, checks)
    for a, b in ties(table):                       # a tie also vouches for both rows
        for c in range(table.ncols):
            if c in table.rows[a].cells and c in table.rows[b].cells:
                ok.add((a, c))
                ok.add((b, c))
    return ok


def _money_cells(table, ok: set) -> list:
    return [(r.idx, c) for r in table.rows if row_pool(table, r) == "money"
            for c, cell in r.cells.items() if (r.idx, c) in ok and cell.value not in (None, 0)]


def inject(table, fault: str, row: int, col: int, rng: random.Random):
    """Return (corrupted copy, [(row, col), ...] cells whose printed value is now wrong) or None."""
    t = copy.deepcopy(table)
    r = t.rows[row]
    if fault == "sign_flip":
        r.cells[col].value = -r.cells[col].value
        return t, [(row, col)]
    if fault == "digit_error":
        v = r.cells[col].value
        k = rng.randrange(0, max(1, len(str(int(abs(v))))))
        r.cells[col].value = v + rng.choice((1, 2, 3, 5, 7)) * 10 ** k
        return t, [(row, col)]
    if fault == "cell_scale":
        r.cells[col].value = r.cells[col].value * 1000
        return t, [(row, col)]
    if fault == "column_shift":
        # every value of the row moves one period column to the right
        if t.ncols < 2:
            return None
        vals = {c: cell.value for c, cell in r.cells.items()}
        wrong = []
        for c in sorted(r.cells, reverse=True):
            if c - 1 in vals:
                if r.cells[c].value != vals[c - 1]:
                    wrong.append((row, c))
                r.cells[c].value = vals[c - 1]
        first = min(r.cells)
        r.cells[first].value = None
        return (t, wrong) if wrong else None
    if fault == "row_swap":
        # two neighbouring money rows trade their values (numbers glued to the wrong labels)
        nxt = next((x for x in t.rows[row + 1:row + 3] if row_pool(t, x) == "money" and col in x.cells), None)
        if nxt is None or nxt.cells[col].value == r.cells[col].value:
            return None
        for c in set(r.cells) & set(nxt.cells):
            r.cells[c].value, nxt.cells[c].value = nxt.cells[c].value, r.cells[c].value
        return t, [(row, col), (nxt.idx, col)]
    if fault == "drop_row":
        # the row vanishes (a wrapped label swallowed it); the cells that depended on it are its neighbours in the sum
        del t.rows[row]
        for i, x in enumerate(t.rows):
            x.idx = i
        return t, []                                  # judged by how many other cells lose their verification
    if fault == "table_scale":
        for x in t.rows:
            for cell in x.cells.values():
                if cell.value is not None and row_pool(t, x) == "money":
                    cell.value = cell.value * 1000
        return t, [(row, col)]
    raise ValueError(fault)


def study(tables: list, per_table: int = 8, seed: int = 7) -> dict:
    """Run every fault type on every table. Returns counts per fault."""
    rng = random.Random(seed)
    out = {f: {"injected": 0, "caught": 0, "silent": 0} for f in FAULTS}
    for table in tables:
        ok = _verified(table)
        cells = _money_cells(table, ok)
        if len(cells) < 6:
            continue
        for fault in FAULTS:
            picks = rng.sample(cells, min(per_table, len(cells)))
            if fault == "table_scale":
                picks = picks[:1]
            for row, col in picks:
                res = inject(table, fault, row, col, rng)
                if res is None:
                    continue
                bad, wrong = res
                ok2 = _verified(bad)
                out[fault]["injected"] += 1
                if fault == "drop_row":
                    # caught if the formula the row belonged to no longer verifies its total
                    lost = len(ok) - len(ok2) - sum(1 for (r_, c_) in ok if r_ == row)
                    caught = lost > 0
                else:
                    caught = all(cell not in ok2 for cell in wrong)
                out[fault]["caught" if caught else "silent"] += 1
    for f, d in out.items():
        n = d["injected"]
        d["catch_rate"] = d["caught"] / n if n else None
        d["ci95"] = wilson(d["caught"], n) if n else None
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 4), round(mid + half, 4))
