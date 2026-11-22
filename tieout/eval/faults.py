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

Two evidence channels that live outside the statement's own arithmetic are
measured on top of that ("after"):

    scale pin   a figure written in a sentence with its unit ("$416.2 billion")
                ties to a cell of the table at the precision written. A fault
                that makes a pinned cell stop tying to its sentence is caught.
                A table read a thousand times too large loses every pin.
    carry       last year's column of a later filing must equal the earlier
                filing's current column line by line. A changed cell that has
                such a partner and no longer equals it is caught.

Coverage (how many faults had the channel at all) is reported apart from the
catch rate among the faults that had it.
"""
from __future__ import annotations

import bisect
import copy
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass

from ..calc import coverage, discover, row_pool, ties, verified_cells
from ..mentions import _precision

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


def inject(table, fault: str, row: int, col: int, rng: random.Random, in_place: bool = False):
    """Return (corrupted copy, [(row, col), ...] cells whose printed value is now wrong) or None.

    With in_place the table itself is damaged and returned (used by the damage curve,
    which chains many faults on one table).
    """
    t = table if in_place else copy.deepcopy(table)
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


# ------------------------------------------------------------ evidence channels

MIN_PIN = 1_000_000          # a sentence figure below a million full units matches too many cells
MIN_SIGNIFICANT = 4          # three digits tie to about half the tables by chance, four to one in ten (see chance_pins)


@dataclass
class Case:
    """One parsed statement and where it sits in the graph."""
    doc: str
    kind: str
    table: object

    @property
    def tid(self) -> str:
        return f"{self.doc}:p{self.table.page}:{self.kind}"

    def rid(self, row: int, col: int) -> str:
        return f"{self.tid}:r{row}:c{col}"


@dataclass
class Pin:
    """A sentence figure that ties to cells of one table."""
    value: float
    tol: float
    cells: tuple            # ((row, col), ...) that tied


def significant_digits(text: str) -> int:
    digits = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return len(re.sub(r"\D", "", digits.group(0)).lstrip("0")) if digits else 0


def usd(table, value) -> int:
    """A printed value in full dollars, the unit the graph and the sentences use."""
    return int(round(value * table.scale))


def money_values(table) -> dict:
    """(row, col) -> full-dollar value for every money cell with a value."""
    return {(r.idx, c): usd(table, cell.value) for r in table.rows if row_pool(table, r) == "money"
            for c, cell in r.cells.items() if cell.value is not None}


def sentence_figures(mentions: list) -> dict:
    """doc -> [(value, tolerance)] for the dollar figures a sentence pins hard enough to mean something."""
    out = defaultdict(set)
    for m in mentions:
        for n in m["numbers"]:
            v = n["value"]
            if n["unit"] != "USD" or not isinstance(v, (int, float)) or abs(v) < MIN_PIN:
                continue
            if significant_digits(n["text"]) < MIN_SIGNIFICANT:
                continue
            sw = re.search(r"(thousand|million|billion|trillion)", n["text"], re.I)
            out[m["doc"]].add((abs(v), _precision(n["text"], sw.group(1).lower() if sw else None)))
    return {d: sorted(v) for d, v in out.items()}


def chance_pins(cases: list, figures: dict, repeats: int = 10, seed: int = 3) -> dict:
    """Control: how often do figures that mean nothing still "pin" a table? Every sentence figure is
    moved by a factor that matches no real cell on purpose; what still ties is coincidence."""
    rng = random.Random(seed)
    pins = tables = 0.0
    for case in cases:
        vals = money_values(case.table)
        for _ in range(repeats):
            fake = [(v * rng.choice((1.13, 1.37, 0.71, 2.3, 0.47)), t) for v, t in figures.get(case.doc, [])]
            n = len(pins_of_table(vals, fake))
            pins += n / repeats
            tables += (n > 0) / repeats
    return {"pins": round(pins, 1), "tables": round(tables, 1)}


class SortedValues:
    """Absolute values kept sorted so "does any cell tie to this figure" is a bisect."""

    def __init__(self, values):
        self.v = sorted(abs(x) for x in values)

    def ties(self, value: float, tol: float) -> bool:
        i = bisect.bisect_left(self.v, value - tol)
        return i < len(self.v) and self.v[i] <= value + tol


def pins_of_table(vals: dict, figures: list) -> list:
    """The sentence figures that tie to at least one cell of this table, with the cells they tie to."""
    by_abs = SortedValues(vals.values())
    out = []
    for value, tol in figures:
        t = max(tol, 1e-6 * value)
        if not by_abs.ties(value, t):
            continue
        cells = tuple(k for k, x in vals.items() if abs(abs(x) - value) <= t)
        out.append(Pin(value, tol, cells))
    return out


class Evidence:
    """The two outside channels, built once from the graph and the sentence mentions."""

    def __init__(self, graph, mentions: list):
        self.figures = sentence_figures(mentions)
        self.graph = graph
        self.partners = defaultdict(list)             # reading id -> ids of its carried copies in other filings
        for ch in graph.checks.values():
            if ch.type == "CARRY" and ch.status == "pass":
                ids = [m[0] for m in ch.members]
                for i in ids:
                    self.partners[i].extend(j for j in ids if j != i)
        self._pins: dict = {}

    def pins(self, case: Case) -> list:
        if case.tid not in self._pins:
            self._pins[case.tid] = pins_of_table(money_values(case.table), self.figures.get(case.doc, []))
        return self._pins[case.tid]

    def partner_values(self, case: Case, row: int, col: int) -> list:
        """Values (full dollars) the same line printed in other filings, if this cell is carried."""
        g = self.graph.readings
        mine = g.get(case.rid(row, col))
        if mine is None:
            return []
        return [g[j].value for j in self.partners.get(mine.id, []) if j in g]


def changed_cells(table, bad) -> list:
    """[(row, col, new value)] for cells whose value differs between a table and its damaged copy."""
    out = []
    for r in table.rows:
        for c, cell in r.cells.items():
            new = bad.rows[r.idx].cells[c].value
            if new != cell.value and row_pool(table, r) == "money":
                out.append((r.idx, c, new))
    return out


def pin_verdict(case: Case, ev: Evidence, bad, changed: list, whole_table: bool) -> tuple:
    """(available, caught): does a sentence that tied before still tie to the damaged table?"""
    pins = ev.pins(case)
    touched = {(r, c) for r, c, _ in changed}
    hit = [p for p in pins if whole_table or touched & set(p.cells)]
    if not hit:
        return False, False
    now = SortedValues(money_values(bad).values())
    return True, any(not now.ties(p.value, max(p.tol, 1e-6 * p.value)) for p in hit)


def carry_verdict(case: Case, ev: Evidence, bad, changed: list) -> tuple:
    """(available, caught): a changed cell with a carried partner that no longer equals it."""
    avail = caught = False
    for r, c, new in changed:
        if new is None:
            continue
        for partner in ev.partner_values(case, r, c):
            avail = True
            if usd(bad, new) != partner:
                caught = True
    return avail, caught


# ------------------------------------------------------------------- the study

def _blank() -> dict:
    return {"injected": 0, "caught": 0, "silent": 0, "caught_after": 0,
            "pin": {"available": 0, "caught": 0}, "carry": {"available": 0, "caught": 0},
            "new_only": 0}


def study(cases: list, per_table: int = 8, seed: int = 7, evidence: Evidence = None) -> dict:
    """Run every fault type on every table. Returns counts per fault.

    `cases` are Case objects (or bare tables, which carry no outside evidence).
    "caught" is the first session's number; "caught_after" adds the two channels.
    """
    rng = random.Random(seed)
    out = {f: _blank() for f in FAULTS}
    for case in cases:
        if not isinstance(case, Case):
            case = Case("", "", case)
        table = case.table
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
                d = out[fault]
                d["injected"] += 1
                if fault == "drop_row":
                    # caught if the formula the row belonged to no longer verifies its total
                    lost = len(ok) - len(ok2) - sum(1 for (r_, c_) in ok if r_ == row)
                    caught = lost > 0
                else:
                    caught = all(cell not in ok2 for cell in wrong)
                d["caught" if caught else "silent"] += 1
                pin = carry = (False, False)
                if evidence is not None and fault != "drop_row":
                    changed = changed_cells(table, bad)
                    pin = pin_verdict(case, evidence, bad, changed, fault == "table_scale")
                    carry = carry_verdict(case, evidence, bad, changed)
                for key, (avail, hit) in (("pin", pin), ("carry", carry)):
                    d[key]["available"] += avail
                    d[key]["caught"] += hit
                after = caught or pin[1] or carry[1]
                d["caught_after"] += after
                d["new_only"] += after and not caught
    for f, d in out.items():
        n = d["injected"]
        d["catch_rate"] = d["caught"] / n if n else None
        d["ci95"] = wilson(d["caught"], n) if n else None
        d["catch_rate_after"] = d["caught_after"] / n if n else None
        d["ci95_after"] = wilson(d["caught_after"], n) if n else None
        for key in ("pin", "carry"):
            c = d[key]
            c["coverage"] = c["available"] / n if n else None
            c["coverage_ci95"] = wilson(c["available"], n) if n else None
            c["catch_rate_of_available"] = c["caught"] / c["available"] if c["available"] else None
            c["ci95"] = wilson(c["caught"], c["available"]) if c["available"] else None
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if not n:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 4), round(mid + half, 4))


# ------------------------------------------------------------- the damage curve

CELL_MIX = ("sign_flip", "digit_error", "cell_scale", "column_shift", "row_swap")


def damage_statements(statements: dict, doc: str, rate: float, seed: int) -> int:
    """Damage `rate` of the money cells of every statement of one filing, in place, with a mix of
    fault types; the checks are recovered again from the damaged table. Returns cells changed."""
    rng = random.Random(seed * 1_000_003 + sum(ord(ch) * (i + 1) for i, ch in enumerate(doc)))
    changed = 0
    for st in statements.values():
        t = st.table
        before = {(r.idx, c): cell.value for r in t.rows for c, cell in r.cells.items()}
        picks = [(r.idx, c) for r in t.rows if row_pool(t, r) == "money"
                 for c, cell in sorted(r.cells.items()) if cell.value not in (None, 0) and rng.random() < rate]
        for row, col in picks:
            if t.rows[row].cells.get(col) is None or t.rows[row].cells[col].value is None:
                continue
            inject(t, rng.choice(CELL_MIX), row, col, rng, in_place=True)
        changed += sum(1 for r in t.rows for c, cell in r.cells.items() if cell.value != before.get((r.idx, c)))
        st.checks = discover(t)
        st.ties = ties(t)
        st.verified, st.cells = coverage(t, st.checks)
    return changed


def sentence_groups(clean, mentions: list) -> list:
    """[(doc, value, tolerance, [reading ids])]: sentence figures that tie to readings of the clean graph."""
    figures = sentence_figures(mentions)
    by_doc = defaultdict(list)
    for r in clean.readings.values():
        if r.unit == "USD":
            by_doc[r.doc].append(r)
    out = []
    for doc, figs in figures.items():
        rs = by_doc.get(doc, [])
        for value, tol in figs:
            t = max(tol, 1e-6 * value)
            ids = [r.id for r in rs if abs(abs(r.value) - value) <= t]
            if ids:
                out.append((doc, value, tol, ids))
    return out


def carry_groups(clean) -> list:
    """Reading-id lists of CARRY checks that pass in the clean graph."""
    return [[m[0] for m in ch.members] for ch in clean.checks.values() if ch.type == "CARRY" and ch.status == "pass"]


def channel_flags(g, groups: list, carries: list) -> set:
    """Reading ids that the two outside channels contradict in graph `g`."""
    flagged = set()
    docs = defaultdict(list)
    for r in g.readings.values():
        if r.unit == "USD":
            docs[r.doc].append(r.value)
    sv = {d: SortedValues(v) for d, v in docs.items()}
    for doc, value, tol, ids in groups:
        if doc in sv and not sv[doc].ties(value, max(tol, 1e-6 * value)):
            flagged.update(ids)
    for ids in carries:
        vals = {g.readings[i].value for i in ids if i in g.readings}
        if len(vals) > 1:
            flagged.update(i for i in ids if i in g.readings)
    return flagged


def same(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def classify(clean, g, flagged: set = None, checks: bool = True) -> dict:
    """What would be served for every fact of the clean graph, graded against the clean graph's own value.

    checks on:  a verified reading is served; a reading that is unverified, failed or contradicted by an
                outside channel is not (flagged when it is wrong, withheld when it was right)
    checks off: every reading is served as if verified
    correct       served and right
    silent_wrong  served as verified and wrong: nothing warns
    flagged       wrong, and the checks mark it unverified, failing or contradicted
    withheld      right but not served (unverified or contradicted), or the fact is gone
    """
    flagged = flagged or set()
    out = {"correct": 0, "silent_wrong": 0, "flagged": 0, "withheld": 0, "facts": 0, "missing": 0,
           "withheld_right": 0}
    for key, f in clean.facts.items():
        truth = clean.best_reading(f)
        if truth is None:
            continue
        out["facts"] += 1
        f2 = g.facts.get(key)
        got = g.best_reading(f2) if f2 is not None else None
        if got is None:
            out["withheld"] += 1
            out["missing"] += 1
            continue
        right = same(got.value, truth.value)
        trusted = (not checks) or (got.status == "verified" and got.id not in flagged)
        if trusted:
            out["correct" if right else "silent_wrong"] += 1
        elif right:
            out["withheld"] += 1
            out["withheld_right"] += 1
        else:
            out["flagged"] += 1
    return out
