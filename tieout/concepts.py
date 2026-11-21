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
    by_total = {ch.total: ch for ch in st.checks if ch.status != "fail"}
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
        not_after = c.structure.get("not_after")
        if not_after:
            # "Net income applicable to common shareholders" is net income minus preferred
            # dividends: a different quantity. The recovered formula says which row is which.
            hits = [h for h in hits if not _subtracts(t, by_total.get(h), not_after)
                    and not (h not in by_total and re.search(r"\bto common\b", norm_label(t.rows[h].label)))]
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


def _subtracts(t, check, pattern: str) -> bool:
    """Does this recovered formula have an operand whose label matches `pattern`?"""
    return check is not None and any(re.search(pattern, t.rows[row].label, re.I) for row, _sign in check.operands)


def structural_proposals(st, proposals: dict) -> dict:
    """Proposals that need no label at all: {concept: (row, evidence)}.

    Balance sheet: the two rows that print identical values in every column are
    total assets and total liabilities and equity, in that order. Merck prints
    both with no label; the tie still names them.

    Income statement: a per-share row e, a share-count row s and an income row n
    with n / s = e in every column are earnings per share, its share count and
    the income it is computed on. Of two such pairs the one with more shares is
    the diluted one. Wording is not consulted unless only one pair exists.
    """
    t, out = st.table, {}
    if st.kind == "BS" and st.ties:
        a, b = max(st.ties, key=lambda pair: max(abs(v) for v in _ints(t, t.rows[pair[0]]) if v is not None))
        why = "prints the same values in every column as the row that closes the statement"
        out["TotalAssets"] = (min(a, b), why)
        out["TotalLiabilitiesAndEquity"] = (max(a, b), why)
    pairs = per_share_identities(t, proposals) if st.kind == "IS" else []
    if pairs:
        eps_rows, share_rows = sorted({p["eps"] for p in pairs}), sorted({p["shares"] for p in pairs})
        income = t.rows[pairs[0]["income"]].label[:40]

        def first_value(row: int) -> float:
            return next(c.value for _, c in sorted(t.rows[row].cells.items()) if c.value)

        def choose(rows: list, word: str, largest: bool) -> int:
            """The row whose wording says `word`; without such wording, the largest or smallest."""
            said = [r for r in rows if word in (t.rows[r].label + " " + t.rows[r].section).lower()]
            return said[0] if said else sorted(rows, key=first_value, reverse=largest)[0]

        # dilution adds shares and lowers the per-share amount
        chosen = {"Diluted": (choose(eps_rows, "dilut", False), choose(share_rows, "dilut", True))}
        if len(eps_rows) > 1 or len(share_rows) > 1:
            chosen["Basic"] = (choose(eps_rows, "basic", True), choose(share_rows, "basic", False))
        for kind, (eps_row, share_row) in chosen.items():
            why = f"'{income}' divided by the share count reproduces the printed per-share amount in every column"
            out[f"EPS{kind}"] = (eps_row, why)
            out[f"Shares{kind}"] = (share_row, why)
    return out


def per_share_identities(t, proposals: dict) -> list:
    """Every (per-share row, share-count row, income row) with income / shares = per-share amount.

    Found from arithmetic alone. The share scale is solved for as well: a filer
    that prints dollars in millions and shares in thousands satisfies the
    identity at exactly one scale.
    """
    start = min((proposals[c] for c in ("NetIncome", "ConsolidatedNetIncome") if c in proposals), default=None)
    if start is None:
        return []
    income_rows = [r for r in t.rows if r.idx >= start and row_pool(t, r) == "money"]
    share_rows = [r for r in t.rows if row_pool(t, r) == "shares"]
    amount_rows = [r for r in t.rows if row_pool(t, r) == "ratio" and not any(c.is_percent for c in r.cells.values())]
    found = []
    for e in amount_rows:
        for s in share_rows:
            for n in income_rows:
                cols = [c for c in range(t.ncols) if all(c in r.cells and r.cells[c].value for r in (e, s, n))]
                if len(cols) < 2:
                    continue
                for share_scale in dict.fromkeys((t.share_scale, 1.0, 1e3, 1e6)):
                    if all(abs(n.cells[c].value * t.scale / (s.cells[c].value * share_scale) - e.cells[c].value)
                           <= max(0.02, 0.012 * abs(e.cells[c].value)) for c in cols):
                        found.append({"eps": e.idx, "shares": s.idx, "income": n.idx, "cols": cols,
                                      "share_scale": share_scale})
                        break
                else:
                    continue
                break                                  # first income row that fits this pair is enough
    return found


def gate(doc: str, statements: dict, proposals: dict) -> tuple:
    """Judge every proposal against the recovered structure.

    Returns (bindings, ratio_checks). A ratio check is a per-share identity:
    income divided by a share count reproduces a printed per-share amount.
    Per-share rows belong to no sum, so this is what verifies them, and because
    earnings per share is printed in plain dollars it also pins the scale of
    the income statement.
    """
    defs = concept_defs()
    bindings = []
    ratio_checks = []
    cf = statements.get("CF")
    structural: dict = {}
    for kind, st in statements.items():
        for concept, (row, why) in structural_proposals(st, proposals.get(kind, {})).items():
            if concept not in defs:
                continue
            worded = proposals.setdefault(kind, {}).get(concept)
            if kind == "BS" and worded not in (None, row):
                continue                             # wording and the tie disagree: the gate below holds it
            proposals[kind][concept] = row           # per-share rows: the identity outranks wording
            structural[(kind, concept)] = why
    if "IS" in statements and "IS" in proposals:
        t = statements["IS"].table
        pairs = per_share_identities(t, proposals["IS"])
        scales = {p["share_scale"] for p in pairs}
        if len(scales) == 1:
            t.share_scale = scales.pop()             # the identity holds at exactly this share scale
        for p in pairs:
            ratio_checks.append(("IS", {"rows": (p["eps"], p["income"], p["shares"]), "cols": p["cols"]}))
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
                confirmed = (kind, concept) in structural
            if (kind, concept) in structural:
                evidence.append(structural[(kind, concept)])
            if confirmed is None:
                state = "active" if evidence else "proposed"       # no structural rule: a verified value is enough
            else:
                state = "active" if confirmed else "proposed"
            bindings.append(Binding(doc=doc, kind=kind, row=row, label=r.label, concept=concept,
                                    origin="seed", state=state, evidence=evidence))
    return bindings, ratio_checks


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
