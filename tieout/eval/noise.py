"""Noisy-graph experiment: how does answer quality fall as extraction gets worse,
and how much of the damage does verification turn from wrong answers into abstentions?

A share of the money cells in every parsed statement is damaged with the
fault types a table parser really produces (sign flip, digit error, value in
the wrong period column, one cell at the wrong scale). The formulas are then
recovered again from the damaged tables, exactly as in a normal build, so
the checks see only what a real pipeline would see. The benchmark is run on
each damaged graph under two policies:

    serve all        answer from whatever is bound, verified or not
    verified only    abstain unless every leaf of the proof is verified

Reported per fault rate: correct, wrong and abstained answers under each
policy. Plans are held fixed (oracle or cached), so no LLM is involved and
the only thing that varies is graph quality.
"""
from __future__ import annotations

import random
import zlib

from ..calc import coverage, discover, row_pool, ties

CELL_FAULTS = ("sign_flip", "digit_error", "cell_scale", "column_shift")


def corrupt_statements(statements: dict, doc: str, rate: float, seed: int) -> int:
    """Damage `rate` of the money cells of each statement in place; return how many were damaged."""
    rng = random.Random(seed * 1_000_003 + zlib.crc32(doc.encode()))
    damaged = 0
    for st in statements.values():
        t = st.table
        for r in t.rows:
            if row_pool(t, r) != "money":
                continue
            for c in sorted(r.cells):
                cell = r.cells[c]
                if cell.value in (None, 0) or rng.random() >= rate:
                    continue
                fault = rng.choice(CELL_FAULTS)
                if fault == "sign_flip":
                    cell.value = -cell.value
                elif fault == "digit_error":
                    k = rng.randrange(0, max(1, len(str(int(abs(cell.value))))))
                    cell.value = cell.value + rng.choice((1, 2, 3, 5, 7)) * 10 ** k
                elif fault == "cell_scale":
                    cell.value = cell.value * 1000
                else:                                   # this cell takes the value of its left neighbour's period
                    other = r.cells.get(c - 1) or r.cells.get(c + 1)
                    if other is None or other.value == cell.value:
                        cell.value = -cell.value
                    else:
                        cell.value = other.value
                damaged += 1
        st.checks = discover(t)                         # the checks only ever see the damaged table
        st.ties = ties(t)
        st.verified, st.cells = coverage(t, st.checks)
    return damaged
