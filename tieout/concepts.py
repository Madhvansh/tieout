"""Bind printed rows to shared concepts, and let the formula graph judge the binding.

A label match is only a proposal: "Gross margin" probably means GrossProfit.
The recovered formulas then act as the admission gate. The proposal becomes
active when the structure agrees with it (the row really is revenue minus
cost of revenue; total assets really prints the same values as total
liabilities and equity; net income really reappears on the cash-flow
statement). A proposal the structure cannot confirm stays "proposed" and is
served with a flag. Nothing here looks at a company name.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml

from .calc import _ints, row_pool, verified_cells

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = ROOT / "semantics" / "seed_concepts.yaml"
OVERRIDES = ROOT / "semantics" / "overrides.yaml"


@dataclass
class ConceptDef:
    name: str
    statement: str
    unit: str
    nature: str
    pick: str
    labels: list
    structure: dict = field(default_factory=dict)
    section: Optional[str] = None
    prefer_section: Optional[str] = None


@dataclass
class Binding:
    doc: str
    kind: str                 # IS | BS | CF
    row: int
    label: str
    concept: str
    origin: str               # seed | learned:<question id> | human
    state: str                # active | proposed | rejected
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def norm_label(s: str) -> str:
    """Lower case, straight quotes, no footnote marks or note references."""
    s = s.lower().replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\((notes?|see note)[^)]*\)?", " ", s)          # (Note 5), (Notes 1 and 3
    s = re.sub(r"\([^)]*\d[^)]*\)", " ", s)                      # (includes $76,569 and $44,855 at fair value)
    s = re.sub(r"\(\d{1,2}\)|\([a-z]\)|\*+", " ", s)               # (1), (a), *
    s = re.sub(r"[:$]", " ", s)
    s = re.sub(r"(\. ?){2,}", " ", s)                              # dot leaders
    s = re.sub(r"\s+", " ", s).strip(" ,.;-")
    return s


_SEED_CACHE: Optional[list] = None


def load_seed() -> list:
    global _SEED_CACHE
    if _SEED_CACHE is None:
        raw = yaml.safe_load(SEED.read_text(encoding="utf-8"))["concepts"]
        _SEED_CACHE = [ConceptDef(name=c["name"], statement=c["statement"], unit=c["unit"],
                                  nature=c["nature"], pick=c.get("pick", "first"),
                                  labels=[re.compile(p) for p in c["labels"]],
                                  structure=c.get("structure", {}) or {}, section=c.get("section"),
                                  prefer_section=c.get("prefer_section"))
                       for c in raw]
    return _SEED_CACHE


def concept_defs() -> dict:
    return {c.name: c for c in load_seed()}


def propose(st, defs: Optional[list] = None) -> dict:
    """Label proposals for one statement: {concept name: row index}."""
    defs = defs or load_seed()
    t = st.table
    totals = {ch.total for ch in st.checks if ch.status != "fail"}
    out: dict = {}
    for c in defs:
        if c.statement != st.kind:
            continue
        want_pool = {"USD/share": "ratio", "shares": "shares"}.get(c.unit, "money")
        hits = []
        for r in t.rows:
            if row_pool(t, r) != want_pool:
                continue
            if c.section and not re.search(c.section, r.section + " " + r.label, re.I):
                continue
            lab = norm_label(r.label)
            if any(p.search(lab) for p in c.labels):
                hits.append(r.idx)
        if not hits:
            continue
        if c.prefer_section:
            # the same label under two headings ("Preferred stock" / "Common stock"): take the preferred heading
            pref = [h for h in hits if re.search(c.prefer_section, t.rows[h].section, re.I)]
            hits = pref or hits
        if c.pick == "total":
            tot = [h for h in hits if h in totals]
            out[c.name] = tot[0] if tot else hits[0]
        elif c.pick == "last":
            out[c.name] = hits[-1]
        else:
            out[c.name] = hits[0]
    return out


def gate(doc: str, statements: dict, proposals: dict) -> tuple:
    """Judge every proposal against the recovered structure.

    Returns (bindings, ratio_checks). A ratio check is the per-share identity:
    net income divided by diluted shares must reproduce the printed diluted
    earnings per share. Per-share rows belong to no sum, so this is what
    verifies them, and because earnings per share is printed in plain
    dollars it also pins the scale of the income statement.
    """
    defs = concept_defs()
    bindings = []
    ratio_checks = []
    cf = statements.get("CF")
    eps_ok = None
    if "IS" in statements:
        eps_ok = _eps_identity(statements["IS"].table, proposals.get("IS", {}))
        if eps_ok:
            ratio_checks.append(("IS", eps_ok))
    for kind, st in statements.items():
        t = st.table
        ok_cells = verified_cells(t, st.checks)
        by_total = {ch.total: ch for ch in st.checks if ch.status != "fail"}
        prop = proposals.get(kind, {})
        for concept, row in prop.items():
            c = defs[concept]
            r = t.rows[row]
            evidence, confirmed = [], None
            if any((row, col) in ok_cells for col in r.cells):
                evidence.append("value sits inside a passing formula")
            s = c.structure
            if s.get("total_of"):
                confirmed = row in by_total
                if confirmed:
                    evidence.append(f"is the total of a recovered formula over {len(by_total[row].operands)} rows")
            if s.get("operand_of"):
                inside = _within(by_total, prop.get(s["operand_of"]), row)
                confirmed = bool(confirmed) and inside if s.get("total_of") else inside
                if inside:
                    evidence.append(f"is an operand of the formula for {s['operand_of']}")
            if s.get("ties_with"):
                other = prop.get(s["ties_with"])
                confirmed = other is not None and _same_values(t, row, other)
                if confirmed:
                    evidence.append(f"prints the same values as {s['ties_with']}")
            if s.get("equals"):
                confirmed = _formula_is(by_total.get(row), prop, s["equals"])
                if confirmed:
                    evidence.append("recovered formula is " + " ".join(
                        (x if x.startswith("-") else "+" + x) for x in s["equals"]))
            if s.get("ties_with_cf_first_row") and cf is not None:
                hit = _appears_in(t, row, cf.table)
                confirmed = hit is not None
                if confirmed:
                    evidence.append(f"same values reappear on the cash-flow statement (row '{cf.table.rows[hit].label[:40]}')")
                elif row in by_total:
                    # attributable net income = consolidated net income - noncontrolling interests:
                    # the consolidated operand is the one the cash-flow statement starts from
                    for op_row, _sign in by_total[row].operands:
                        if _appears_in(t, op_row, cf.table) is not None:
                            confirmed = True
                            evidence.append("recovered formula derives it from '" + t.rows[op_row].label[:40]
                                            + "', which reappears on the cash-flow statement")
                            break
            if s.get("eps_identity"):
                confirmed = bool(eps_ok)
                if confirmed:
                    evidence.append("net income divided by diluted shares reproduces diluted earnings per share")
            if confirmed is None:
                state = "active" if evidence else "proposed"       # no structural rule: a verified value is enough
            else:
                state = "active" if confirmed else "proposed"
            bindings.append(Binding(doc=doc, kind=kind, row=row, label=r.label, concept=concept,
                                    origin="seed", state=state, evidence=evidence))
    return bindings, ratio_checks


def _eps_identity(t, prop: dict):
    """{"rows": (eps, net income, shares), "cols": [...]} if NI / diluted shares = diluted EPS in every column."""
    rows = [prop.get("EPSDiluted"), prop.get("NetIncome"), prop.get("SharesDiluted")]
    if any(x is None for x in rows):
        return None
    eps, ni, sh = (t.rows[x] for x in rows)
    cols = []
    for c in range(t.ncols):
        if not (c in eps.cells and c in ni.cells and c in sh.cells):
            continue
        e, n, d = eps.cells[c].value, ni.cells[c].value, sh.cells[c].value
        if not d or e is None or n is None:
            return None
        implied = (n * t.scale) / (d * t.share_scale)
        if abs(implied - e) > max(0.02, 0.012 * abs(e)):      # cents rounding; preferred dividends at a few filers
            return None
        cols.append(c)
    return {"rows": tuple(rows), "cols": cols} if cols else None


def _within(by_total: dict, top, row: int, depth: int = 2) -> bool:
    """Is `row` an operand of the formula for `top`, directly or one level down?

    Total liabilities and equity = liabilities + total equity, and total equity =
    shareholders' equity + noncontrolling interests: shareholders' equity is two levels down.
    """
    check = by_total.get(top)
    if check is None or depth == 0:
        return False
    return any(op == row or _within(by_total, op, row, depth - 1) for op, _ in check.operands)


def _same_values(t, a: int, b: int) -> bool:
    va, vb = _ints(t, t.rows[a]), _ints(t, t.rows[b])
    return all(x is not None and x == y for x, y in zip(va, vb))


def _formula_is(check, prop: dict, spec: list) -> bool:
    if check is None:
        return False
    want = set()
    for item in spec:
        sign = -1 if item.startswith("-") else 1
        row = prop.get(item.lstrip("-"))
        if row is None:
            return False
        want.add((row, sign))
    return set(check.operands) == want


def _appears_in(t, row: int, other) -> Optional[int]:
    """Row index in `other` that prints the same values for the same period ends."""
    src = t.rows[row]
    for r in other.rows[:6]:                       # net income opens the cash-flow statement
        pairs = 0
        for ci, cell in src.cells.items():
            if cell.value is None:
                continue
            end = t.columns[ci].end_date or t.columns[ci].year
            for cj, c2 in r.cells.items():
                end2 = other.columns[cj].end_date or other.columns[cj].year
                if end == end2:
                    if c2.value != cell.value:
                        pairs = -99
                    else:
                        pairs += 1
        if pairs >= 2:
            return r.idx
    return None


def load_overrides() -> list:
    """Human-approved corrections: each has doc, kind, label, concept, reason, page."""
    if not OVERRIDES.exists():
        return []
    return yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or []
