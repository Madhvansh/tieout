"""Why did a figure change between two filings?

A company prints last year's statement again in this year's filing. When a
line differs from what it first printed, the graph already knows the two
numbers (a CARRY check with status "restated"). This module explains the
change structurally, from the statements themselves:

1. Align. Take the same statement kind and the same period column in the
   earliest and the latest filing that print it. Both print their lines in
   the same order, so rows are aligned as two sequences of label keys
   (normalised label without parentheticals or digits); inside a stretch that
   differs, rows pair by equal value (a renamed line) or shared label words.
   A row printed only in the later filing is "new", one printed only in the
   earlier is "dropped". Printed page numbers caught as cells are ignored.
2. Climb. The SUM checks (later filing first) say which subtotal each line
   feeds. From the changed line go up subtotal by subtotal. If a subtotal
   printed the same in both filings is reached, and the changes beneath it net
   to zero, money moved between lines: a RECLASSIFICATION, and the other
   changed lines beneath it are the counterparts. If the climb leaves the
   statement with totals still moving, the figures themselves changed: a
   RESTATEMENT. A line, or a whole block with its subtotal, that was only
   added or dropped (a supplemental line) is a PRESENTATION change.
3. Quote. Sentences of the later filing that say why prior-period figures
   changed (reclassified, retrospectively, recast, ...) are quoted verbatim
   with their page. Nothing is generated.

All amounts are integers in full units. A printed rounding of one unit per
term is tolerated when sums are compared.
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

from .concepts import norm_label

KIND_ORDER = ("IS", "BS", "CF")
# Why prior-period figures changed, strongest wording first. Legal "Restated Certificate"
# and the routine "reclassified into earnings" of hedge accounting are not explanations.
STRONG = re.compile(r"retrospectiv|conform(?:ed|s)? (?:\w+ ){0,3}current[- ](?:year|period)|"
                    r"prior[- ](?:year|period)s?\b.{0,80}\b(?:reclassif|recast|revised|restated)", re.I)
MEDIUM = re.compile(r"reclassif|\brecast\b|\brestat(?:ed|ement)", re.I)
WEAK = re.compile(r"\brevised\b|held for sale|changes? in accounting", re.I)
NOISE = re.compile(r"restated (?:certificate|bylaws|by-laws|declaration)|amended and restated|"
                   r"reclassified (?:in)?to (?:net )?(?:earnings|income)|reclassified from accumulated", re.I)
STOP = {"and", "the", "of", "for", "net", "other", "total", "assets", "liabilities", "from", "with", "less"}


# ------------------------------------------------------------------ readings

def _ref(r) -> dict:
    """What an answer needs to cite one printed number."""
    return {"id": r.id, "doc": r.doc, "page": r.page, "label": r.label, "printed": r.text,
            "value": int(r.value), "period": r.period}


def _unit(r) -> int:
    """One printed unit in full units (the table scale), read off the cell itself."""
    digits = re.sub(r"\D", "", r.text.split(".")[0])
    if digits and int(digits):
        return max(1, round(abs(r.value) / int(digits)))
    return 1


class Pair:
    """One statement line aligned across the earlier (old) and later (new) filing."""

    def __init__(self, old=None, new=None):
        self.old, self.new = old, new

    @property
    def label(self) -> str:
        return (self.new or self.old).label

    @property
    def delta(self) -> int:
        return int((self.new.value if self.new else 0) - (self.old.value if self.old else 0))

    @property
    def unit(self) -> int:
        return max(_unit(r) for r in (self.old, self.new) if r is not None)

    @property
    def changed(self) -> bool:
        return self.delta != 0                         # a new or dropped line printed as "—" is no change

    def row(self) -> dict:
        return {"label": self.label, "old": int(self.old.value) if self.old else None,
                "new": int(self.new.value) if self.new else None, "delta": self.delta,
                "old_printed": self.old.text if self.old else None,
                "new_printed": self.new.text if self.new else None,
                "new_line": self.old is None, "dropped": self.new is None,
                "readings": [_ref(r) for r in (self.old, self.new) if r is not None]}


# ----------------------------------------------------------------- alignment

def _folio(r) -> bool:
    """A printed page number caught as a statement cell: a bare small integer, a few
    pages below the PDF page it sits on, inside no check."""
    return not r.checks and re.fullmatch(r"\d{1,3}", r.text) is not None and 0 <= r.page - int(r.text) <= 12


class Index:
    """Readings by statement column and SUM checks by filing, built once per graph pass."""

    def __init__(self, g):
        self.columns: dict = defaultdict(list)
        for r in g.readings.values():
            if r.unit == "USD" and not _folio(r):
                self.columns[(r.ticker, r.kind, r.period)].append(r)
        self.sums: dict = defaultdict(list)
        for c in g.checks.values():
            if c.type == "SUM" and c.status != "fail":
                self.sums[c.doc].append(c)


def _statement(g, idx: Index, ticker: str, kind: str, period: str) -> dict:
    """{doc: [readings]} for one statement column, docs in filing order."""
    by_doc: dict = defaultdict(lambda: defaultdict(list))
    for r in idx.columns.get((ticker, kind, period), []):
        by_doc[r.doc][r.table].append(r)
    out = {}
    for doc in sorted(by_doc, key=lambda d: g.filings[d]["period_end"]):
        rs = max(by_doc[doc].values(), key=len)       # the statement table, if a doc has two
        out[doc] = sorted(rs, key=lambda r: r.row)
    return out


def _key(label: str) -> str:
    """A label without its parenthetical (closed or cut off by a line break) and digits."""
    s = re.sub(r"\([^)]*(\)|$)", " ", norm_label(label))
    return " ".join(re.sub(r"[^a-z' ]+", " ", s).split()) or norm_label(label)


def _words(label: str) -> set:
    return set(re.findall(r"[a-z]{3,}", norm_label(label)))


def align(old: list, new: list) -> list:
    """Pair the rows of one statement column in two filings (see module doc, step 1).

    Both filings print the lines in the same order, so the rows are aligned as
    two sequences of label keys. Inside a stretch that differs, rows pair by
    equal value (a renamed line) or by shared label words; whatever is left
    is a new or dropped line, and a last pass pairs those by equal value.
    """
    pairs, left, right = [], [], []
    sm = SequenceMatcher(None, [_key(r.label) for r in old], [_key(r.label) for r in new], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = list(old[i1:i2]), list(new[j1:j2])
        if op == "equal":
            pairs += [Pair(x, y) for x, y in zip(a, b)]
            continue
        for x in list(a):
            y = next((y for y in b if y.value == x.value and x.value != 0), None) or next(
                (y for y in b if len(_words(x.label) & _words(y.label)) * 2 > len(_words(x.label) | _words(y.label))),
                None)
            if y is not None:
                pairs.append(Pair(x, y))
                a.remove(x)
                b.remove(y)
        left, right = left + a, right + b
    for x in list(left):                               # a line that moved: same value anywhere
        y = next((y for y in right if y.value == x.value and x.value != 0), None)
        if y is not None:
            pairs.append(Pair(x, y))
            left.remove(x)
            right.remove(y)
    pairs += [Pair(x, None) for x in left] + [Pair(None, y) for y in right]
    return sorted(pairs, key=lambda p: (p.new.row if p.new else p.old.row + 0.5))


def _parents(g, checks: list, period: str) -> dict:
    """reading id -> (total id, [(operand id, sign)]) from the given SUM checks."""
    out: dict = {}
    for c in checks:
        groups, cur = [], None
        for rid, role in c.members:                    # members: total, operands, total, operands ...
            if role == "total":
                cur = [rid, []]
                groups.append(cur)
            elif cur is not None:
                cur[1].append((rid, int(role)))
        for total, ops in groups:
            if total in g.readings and g.readings[total].period == period:
                for rid, _ in ops:
                    out.setdefault(rid, (total, ops))
    return out


class Column:
    """One statement column aligned across its first and latest filing."""

    def __init__(self, g, ticker: str, kind: str, period: str, idx: Optional[Index] = None):
        idx = idx or Index(g)
        self.kind, self.period = kind, period
        st = _statement(g, idx, ticker, kind, period)
        self.docs = list(st)
        self.old_doc, self.new_doc = (self.docs[0], self.docs[-1]) if self.docs else (None, None)
        self.pairs = align(st[self.old_doc], st[self.new_doc]) if len(self.docs) > 1 else []
        self.by_id = {r.id: p for p in self.pairs for r in (p.old, p.new) if r is not None}
        self.parent = {**_parents(g, idx.sums.get(self.old_doc, []), period),
                       **_parents(g, idx.sums.get(self.new_doc, []), period)}
        self._up = {id(p): self._climb(p) for p in self.pairs}

    def up(self, p: Pair):
        """(total pair, [(operand pair, sign)]) of the subtotal p feeds, preferring the later filing."""
        return self._up.get(id(p))

    def _climb(self, p: Pair):
        for r in (p.new, p.old):
            if r is not None and r.id in self.parent:
                total, ops = self.parent[r.id]
                tp = self.by_id.get(total)
                if tp is not None:
                    return tp, [(self.by_id[i], s) for i, s in ops if i in self.by_id]
        return None

    def beneath(self, total: Pair) -> list:
        """Every line inside a subtotal, recursively."""
        kids = {id(p): p for p in self.pairs if (u := self.up(p)) and u[0] is total}
        out = list(kids.values())
        for k in kids.values():
            out += self.beneath(k)
        return out

    def changes(self) -> list:
        return [p for p in self.pairs if p.changed]


# ----------------------------------------------------------------- diagnosis

def classify(col: Column, p: Pair) -> dict:
    """Reclassification or restatement of one changed line (module doc, step 2)."""
    cur, seen, unchanged, path = p, set(), [], [p]
    while (u := col.up(cur)) and id(cur) not in seen:
        seen.add(id(cur))
        total, ops = u
        path.append(total)
        same = bool(total.old and total.new) and abs(total.delta) <= total.unit
        nets = abs(sum(s * o.delta for o, s in ops)) <= sum(o.unit for o, _ in ops)
        if same and (unchanged or nets):
            unchanged.append(total)                    # first one must also net to zero beneath
        elif unchanged:
            break                                      # the chain of unchanged subtotals ended
        cur = total
    if not unchanged:
        if abs(p.delta) <= p.unit and p.old and p.new:
            kind = "unchanged"                         # one printed unit: rounding
        elif all(x.old is None or x.new is None for x in path):
            kind = "presentation"                      # a line, or a block of lines, added or dropped
        else:
            kind = "restatement"
        return {"kind": kind, "unchanged_total": None, "unchanged_totals": [], "counterparts": []}
    inside = col.beneath(unchanged[0])
    parts = [o for o in inside if o.changed and o is not p]
    tot = lambda t: {"label": t.label, "value": int(t.new.value), "printed": t.new.text,
                     "readings": [_ref(t.old), _ref(t.new)]}
    return {"kind": "reclassification", "unchanged_total": tot(unchanged[0]),
            "unchanged_totals": [tot(t) for t in unchanged], "counterparts": [o.row() for o in parts]}


def _find(g, ticker: str, period: str, label: Optional[str], concept: Optional[str], kind: Optional[str]):
    """The reading the question names: by concept through the fact, or by printed label."""
    if concept:
        f = g.fact(ticker, concept, period)
        rs = [g.readings[i] for i in f.readings] if f else []
    else:
        want = norm_label(label or "")
        cand = [r for r in g.readings.values() if r.ticker == ticker and r.period == period and r.unit == "USD"]
        rs = [r for r in cand if norm_label(r.label) == want] or \
             [r for r in cand if norm_label(r.label).startswith(want)]
    rs = [r for r in rs if kind is None or r.kind == kind]
    if not rs:
        return None
    return max(rs, key=lambda r: (g.filings[r.doc]["period_end"], -KIND_ORDER.index(r.kind)
                                  if r.kind in KIND_ORDER else -9))


def diagnose(g, ticker: str, period: str, label: Optional[str] = None, concept: Optional[str] = None,
             kind: Optional[str] = None, explain: bool = True) -> dict:
    """Why the line (named by printed label or by concept) differs between two filings."""
    r = _find(g, ticker, period, label, concept, kind)
    if r is None:
        return {"kind": "not_found", "summary": f"No printed line found for {label or concept} {period}."}
    col = Column(g, ticker, r.kind, period)
    p = col.by_id.get(r.id)
    if p is None:                                      # printed in one filing only
        return {"kind": "single_filing", "line": {"label": r.label, "old": None, "new": _ref(r)}, "delta": 0,
                "counterparts": [], "unchanged_total": None, "changed_lines": 0, "explanations": [],
                "summary": f"{r.label} for {period} is printed in one filing only ({r.doc} p{r.page}: {r.text})."}
    changes = col.changes()
    out = classify(col, p) if p.changed else \
        {"kind": "unchanged", "unchanged_total": None, "unchanged_totals": [], "counterparts": []}
    out.update(line={"label": p.label, "old": _ref(p.old) if p.old else None, "new": _ref(p.new) if p.new else None},
               delta=p.delta, changed_lines=len(changes), changes=[c.row() for c in changes],
               statement=r.kind, period=period, old_doc=col.old_doc, new_doc=col.new_doc)
    words = [p.label] + [c["label"] for c in out["counterparts"]]
    out["explanations"] = explanations(col.new_doc, (p.new or p.old).page, words) \
        if explain and out["kind"] != "unchanged" else []
    out["summary"] = _summary(out)
    return out


def _fmt(p: Optional[dict]) -> str:
    return p["printed"] if p else "not printed"


def _summary(d: dict) -> str:
    ln, k = d["line"], d["kind"]
    head = f"{ln['label']} ({d['statement']} {d['period']}) was {_fmt(ln['old'])} in {d['old_doc']} " \
           f"and is {_fmt(ln['new'])} in {d['new_doc']}"
    if k == "unchanged":
        return f"{head}: unchanged" + (" apart from rounding." if d["delta"] else ".")
    if k == "reclassification":
        cps = "; ".join(f"{c['label']} {c['old_printed'] or 'not printed'} -> {c['new_printed'] or 'not printed'}"
                        + (" (new line)" if c["new_line"] else "") for c in d["counterparts"]) or "no other line"
        tots = " and ".join(f"{t['label']} {t['printed']}" for t in d["unchanged_totals"])
        return f"{head}: a reclassification; the amount moved to or from {cps}, while {tots} did not change."
    if k == "presentation":
        return f"{head}: a presentation change; the line sits outside every subtotal of the statement."
    return f"{head}: a restatement; totals moved and {d['changed_lines']} lines of the statement changed."


# -------------------------------------------------------------- explanations

def _columns(page: dict) -> list:
    """The words of a page split at a column gutter, if the page has one.

    A gutter is a vertical line near the middle that almost no word crosses
    while plenty of words sit on each side of it.
    """
    words, best = page["words"], None
    for x in range(int(page["w"] * 0.4), int(page["w"] * 0.6), 2):
        cross = sum(1 for w in words if w.x0 < x < w.x1)
        left = sum(1 for w in words if w.x1 <= x)
        if left > 40 and len(words) - left - cross > 40 and cross <= 0.01 * len(words):
            if best is None or cross < best[0]:
                best = (cross, x)
    if best is None:
        return [words]
    x = best[1]
    return [[w for w in words if (w.x0 + w.x1) / 2 < x], [w for w in words if (w.x0 + w.x1) / 2 >= x]]


@lru_cache(maxsize=8)
def _sentences(doc: str) -> tuple:
    """(page, sentence) for every sentence of a filing, from the cached word boxes."""
    from .pdfio import load_pages
    from .statements import page_lines
    out = []
    for i, page in enumerate(load_pages(doc)):
        text = " ".join(" ".join(w.text for w in sorted(ln, key=lambda w: w.x0))
                        for col in _columns(page) for ln in page_lines({**page, "words": col}))
        for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text):
            s = s.strip()
            if 30 <= len(s) <= 700 and len(re.findall(r"\d[\d,.]*", s)) < 12:
                out.append((i + 1, s))
    return tuple(out)


def explanations(doc: str, page: int, labels: list, limit: int = 3) -> list:
    """Up to `limit` sentences of `doc` that say why prior-period figures changed.

    Sentences with a strong pattern (reclassified, retrospectively, recast,
    restated, conformed to the current presentation) or mentioning a word of a
    changed line's label come first; ties go to pages nearer the statement.
    """
    words = {w for lab in labels for w in re.findall(r"[a-z]{4,}", norm_label(lab)) if w not in STOP}
    hits = []
    for pg, s in _sentences(doc):
        if NOISE.search(s) and not STRONG.search(s):
            continue
        tier = 4 if STRONG.search(s) else 2 if MEDIUM.search(s) else 1 if WEAK.search(s) else 0
        if not tier:
            continue
        named = sum(1 for w in words if re.search(rf"\b{w}", s, re.I))
        hits.append(((-tier - min(named, 2), abs(pg - page)), pg, s))
    hits.sort(key=lambda h: h[0])
    return [{"doc": doc, "page": pg, "sentence": s} for _, pg, s in hits[:limit]]


# --------------------------------------------------------------------- corpus

def events(g, explain: bool = False) -> list:
    """Every statement column printed differently by two filings, classified.

    An event is a restatement when any changed line moves totals, else a
    reclassification when a changed line climbs to an unchanged subtotal,
    else a presentation change.
    """
    idx = Index(g)
    out = []
    for ticker, kind, period in sorted(idx.columns):
        if len({r.doc for r in idx.columns[(ticker, kind, period)]}) < 2:
            continue
        col = Column(g, ticker, kind, period, idx)
        changes = col.changes()
        if not changes:
            continue
        verdicts = [classify(col, p) for p in changes]
        kinds = {v["kind"] for v in verdicts}
        ev = next((k for k in ("restatement", "reclassification", "presentation") if k in kinds), None)
        if ev is None:
            continue                                   # rounding differences only
        totals = {t["label"]: t["value"] for v in verdicts for t in v["unchanged_totals"][:1]}
        e = {"ticker": ticker, "statement": kind, "period": period, "kind": ev, "old_doc": col.old_doc,
             "new_doc": col.new_doc, "changed_lines": len(changes), "changes": [c.row() for c in changes],
             "unchanged_totals": totals}
        if explain:
            e["explanations"] = explanations(col.new_doc, (changes[0].new or changes[0].old).page,
                                             [c.label for c in changes])
        out.append(e)
    return out
