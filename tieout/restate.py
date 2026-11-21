"""Why did a figure change between two filings?

A company prints last year's statements again in this year's filing. When a
line differs from what it first printed, the graph already holds both numbers
(a CARRY check with status "restated"). This module explains the change from
the statements themselves.

1. Align. Take one statement kind and one period column in the first and the
   latest filing that print it. Both list their lines in the same order, so the
   rows are aligned as two sequences of label keys (the label without
   parentheticals or digits). Inside a stretch that differs, rows pair by equal
   value (a renamed line) or by shared label words. A row printed only in the
   later filing is "new"; one printed only in the earlier filing is "dropped".
   A bare number on a row with fewer cells than the statement has columns, in
   no check that holds, is left out: it is a page number, a year in a footer or
   a number inside a footnote sentence, not a statement figure.
2. Climb. The SUM checks say which subtotal each line feeds. From a changed
   line go up subtotal by subtotal until one is printed the same in both
   filings and the changes beneath it cancel. Then money moved between lines:
   a RECLASSIFICATION, and its counterparts are the fewest other changed lines
   there whose changes cancel this one. One exception: when a line of that
   group is a figure the filing also prints on another statement (net income
   opens the cash-flow statement) and that statement restated it, the group
   only absorbs a restatement, so it is a RESTATEMENT. If the climb leaves the
   statement with totals still moving, the figures themselves changed: a
   RESTATEMENT. A line, or a block with its own subtotal, that was only added
   or dropped is a PRESENTATION change.
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
from itertools import combinations
from typing import Optional

from .concepts import norm_label

KIND_ORDER = ("IS", "BS", "CF")
STATEMENT = {"IS": "income statement", "BS": "balance sheet", "CF": "cash-flow statement"}
MAX_GROUP = 6                     # most other lines one reclassification is searched across
# Why prior-period figures changed, strongest wording first.
STRONG = re.compile(r"retrospectiv|conform(?:ed|s)? (?:\w+ ){0,3}current[- ](?:year|period)|"
                    r"prior[- ](?:year|period)s?\b.{0,80}\b(?:reclassif|recast|revised|restated)", re.I)
MEDIUM = re.compile(r"reclassif|\brecast\b|\brestat(?:ed|ement)", re.I)
WEAK = re.compile(r"\brevised\b|held for sale|changes? in accounting", re.I)
# Not explanations: legal names, the routine "reclassified into earnings" of hedge accounting,
# compensation clawback policies, and sentences about what can or may happen.
NOISE = re.compile(r"restated (?:certificate|bylaws|by-laws|declaration)|amended and restated|"
                   r"reclassified (?:in)?to (?:net )?(?:earnings|income)|reclassified from accumulated|"
                   r"(?:accounting|financial) restatement|event of a restatement", re.I)
MODAL = re.compile(r"\b(?:can|could|may|might|will|would)\b")
STOP = {"and", "the", "of", "for", "net", "other", "total", "assets", "liabilities", "from", "with", "less"}


# ------------------------------------------------------------------ readings

def _ref(r) -> Optional[dict]:
    """What an answer needs to cite one printed number."""
    if r is None:
        return None
    return {"id": r.id, "doc": r.doc, "page": r.page, "label": r.label, "printed": r.text, "value": int(r.value)}


def _unit(r) -> int:
    """One printed unit in full units (the table scale), read off the cell itself."""
    digits = re.sub(r"\D", "", r.text.split(".")[0])
    return max(1, round(abs(r.value) / int(digits))) if digits and int(digits) else 1


class Pair:
    """One statement line aligned across the earlier (old) and the later (new) filing."""

    def __init__(self, old=None, new=None):
        self.old, self.new = old, new
        self.label = (new or old).label
        self.delta = int((new.value if new else 0) - (old.value if old else 0))
        self.unit = max(_unit(r) for r in (old, new) if r is not None)
        self.changed = self.delta != 0                 # a new or dropped line printed as "—" is no change
        self.both = old is not None and new is not None

    def row(self) -> dict:
        return {"label": self.label, "old": int(self.old.value) if self.old else None,
                "new": int(self.new.value) if self.new else None, "delta": self.delta,
                "old_printed": self.old.text if self.old else None,
                "new_printed": self.new.text if self.new else None,
                "new_line": self.old is None, "dropped": self.new is None,
                "old_reading": _ref(self.old), "new_reading": _ref(self.new)}


# ----------------------------------------------------------------- alignment

class Index:
    """The statement columns and SUM checks of one graph, and the columns aligned so far."""

    def __init__(self, g):
        self.g = g
        cells, cols = defaultdict(int), defaultdict(set)
        for r in g.readings.values():
            cells[(r.table, r.row)] += 1
            cols[r.table].add(r.col)
        holds = {c.id for c in g.checks.values() if c.type != "CARRY" and c.status != "fail"}
        self.columns: dict = defaultdict(list)
        for r in g.readings.values():
            stray = re.fullmatch(r"\d+", r.text) and cells[(r.table, r.row)] < len(cols[r.table]) \
                and not holds.intersection(r.checks)
            if r.unit == "USD" and not stray:
                self.columns[(r.ticker, r.kind, r.period)].append(r)
        self.sums: dict = defaultdict(list)
        for c in g.checks.values():
            if c.type == "SUM" and c.status != "fail":
                self.sums[c.doc].append(c)
        self.aligned: dict = {}

    def column(self, ticker: str, kind: str, period: str) -> "Column":
        key = (ticker, kind, period)
        if key not in self.aligned:
            self.aligned[key] = Column(self, *key)
        return self.aligned[key]

    def statement(self, ticker: str, kind: str, period: str) -> dict:
        """{doc: readings in row order} for one statement column, docs in filing order."""
        by_doc: dict = defaultdict(lambda: defaultdict(list))
        for r in self.columns.get((ticker, kind, period), []):
            by_doc[r.doc][r.table].append(r)
        docs = sorted(by_doc, key=lambda d: self.g.filings[d]["period_end"])
        return {d: sorted(max(by_doc[d].values(), key=len), key=lambda r: r.row) for d in docs}


def _key(label: str) -> str:
    """A label without its parenthetical (closed or cut off by a line break) and digits."""
    s = re.sub(r"\([^)]*(\)|$)", " ", norm_label(label))
    return " ".join(re.sub(r"[^a-z' ]+", " ", s).split()) or norm_label(label)


def _words(label: str) -> set:
    return set(re.findall(r"[a-z]{3,}", norm_label(label)))


def align(old: list, new: list) -> list:
    """Pair the rows of one statement column in two filings (module doc, step 1).

    Rows align as two sequences of label keys. Inside a stretch that differs,
    rows pair by equal value (a renamed line) or by shared label words; what is
    left is a new or dropped line, and a last pass pairs those by equal value.
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
    """reading id -> (subtotal reading id, sign) from SUM checks, for subtotals of the period."""
    out: dict = {}
    for c in checks:
        total = None
        for rid, role in c.members:                    # members: total, operands, total, operands ...
            if role == "total":
                total = rid if rid in g.readings and g.readings[rid].period == period else None
            elif total is not None:
                out.setdefault(rid, (total, int(role)))
    return out


class Column:
    """One statement column aligned across the first and the latest filing that print it."""

    def __init__(self, idx: Index, ticker: str, kind: str, period: str):
        self.idx, self.ticker, self.kind, self.period = idx, ticker, kind, period
        st = idx.statement(ticker, kind, period)
        docs = list(st)
        self.old_doc, self.new_doc = (docs[0], docs[-1]) if docs else (None, None)
        self.pairs = align(st[docs[0]], st[docs[-1]]) if len(docs) > 1 else []
        self.by_id = {r.id: p for p in self.pairs for r in (p.old, p.new) if r is not None}
        parent = {**_parents(idx.g, idx.sums.get(self.old_doc, []), period),
                  **_parents(idx.g, idx.sums.get(self.new_doc, []), period)}
        self.up: dict = {}                             # id(pair) -> (subtotal pair, sign), later filing first
        self.kids: dict = defaultdict(list)
        for p in self.pairs:
            for r in (p.new, p.old):
                if r is not None and r.id in parent and self.by_id.get(parent[r.id][0]) not in (None, p):
                    self.up[id(p)] = (self.by_id[parent[r.id][0]], parent[r.id][1])
                    self.kids[id(self.up[id(p)][0])].append((p, parent[r.id][1]))
                    break

    def beneath(self, total: Pair, sign: int = 1, seen: Optional[set] = None) -> list:
        """[(pair, sign as it feeds `total`)] for every line inside a subtotal, recursively."""
        seen = {id(total)} if seen is None else seen
        out = []
        for p, s in self.kids[id(total)]:
            if id(p) not in seen:
                seen.add(id(p))
                out += [(p, sign * s)] + self.beneath(p, sign * s, seen)
        return out

    def changes(self) -> list:
        return [p for p in self.pairs if p.changed]


# ----------------------------------------------------------------- diagnosis

def _same(p: Pair) -> bool:
    return p.both and abs(p.delta) <= p.unit


def _fewest(delta: int, unit: int, others: list) -> list:
    """The fewest of `others` [(pair, sign)] whose changes cancel `delta`; all of them if none do sooner."""
    for n in range(1, min(len(others), MAX_GROUP) + 1):
        for combo in combinations(others, n):
            if abs(delta + sum(s * q.delta for q, s in combo)) <= unit + sum(q.unit for q, _ in combo):
                return [q for q, _ in combo]
    return [q for q, _ in others]


def _restated_elsewhere(col: Column, p: Pair) -> Optional[dict]:
    """The same figure on another statement of the filing (a TIE check), if that statement restated it."""
    g = col.idx.g
    for r in (x for x in (p.new, p.old) if x is not None):
        for rid, _ in (m for cid in r.checks if g.checks[cid].type == "TIE" for m in g.checks[cid].members):
            m = g.readings.get(rid)
            if m is None or m.kind == col.kind:
                continue
            other = col.idx.column(col.ticker, m.kind, m.period)
            q = other.by_id.get(m.id)
            if q is not None and classify(other, q, follow=False)["kind"] == "restatement":
                return {"line": p.label, "statement": m.kind, "label": q.label,
                        "old_printed": q.old.text if q.old else None, "new_printed": q.new.text if q.new else None}
    return None


def classify(col: Column, p: Pair, follow: bool = True) -> dict:
    """{kind, counterparts, totals, carried_from, block} for one line (module doc, step 2).

    counterparts and totals are pairs: the lines that moved with this one and the
    unchanged subtotals above them. `follow` looks across to other statements.
    """
    out = {"kind": "unchanged", "counterparts": [], "totals": [], "carried_from": None, "block": p}
    if not p.changed or _same(p):
        return out                                     # no change, or one printed unit of rounding
    path, sign = [p], 1
    while id(path[-1]) in col.up and len(path) <= len(col.pairs):
        total, s = col.up[id(path[-1])]
        path.append(total)
        sign *= s
        leaves = [(q, t) for q, t in col.beneath(total) if q.changed and not any(k.changed for k, _ in col.kids[id(q)])]
        if _same(total) and abs(sum(t * q.delta for q, t in leaves)) <= sum(q.unit for q, _ in leaves):
            break
    else:                                              # totals still move: restated, or only added or dropped
        moved = any(x.both for x in path)
        return {**out, "kind": "restatement" if moved else "presentation", "block": path[-1]}
    mine = {id(x) for x in path} | {id(q) for q, _ in col.beneath(p)}
    group = _fewest(sign * p.delta, p.unit, [(q, t) for q, t in leaves if id(q) not in mine])
    totals = [path[-1]]
    while id(totals[-1]) in col.up and _same(col.up[id(totals[-1])][0]) and len(totals) <= len(col.pairs):
        totals.append(col.up[id(totals[-1])][0])        # the chain of unchanged subtotals above it
    carried = next(filter(None, (_restated_elsewhere(col, q) for q in [p] + group)), None) if follow else None
    return {**out, "kind": "restatement" if carried else "reclassification", "counterparts": group,
            "totals": totals, "carried_from": carried}


def _total(t: Pair) -> dict:
    return {"label": t.label, "value": int(t.new.value), "printed": t.new.text,
            "old_reading": _ref(t.old), "new_reading": _ref(t.new)}


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
        return {"kind": "not_found", "counterparts": [], "explanations": [],
                "summary": f"No printed line found for {label or concept} {period}."}
    col = Index(g).column(ticker, r.kind, period)
    p = col.by_id.get(r.id)
    if p is None:
        return {"kind": "single_filing", "counterparts": [], "explanations": [],
                "summary": f"{r.label} for {period} is printed in one filing only ({r.doc} p{r.page}: {r.text})."}
    v = classify(col, p)
    out = {"kind": v["kind"], "line": p.row(), "counterparts": [q.row() for q in v["counterparts"]],
           "unchanged_totals": [_total(t) for t in v["totals"]], "carried_from": v["carried_from"],
           "changes": [c.row() for c in col.changes()], "statement": r.kind, "period": period,
           "old_doc": col.old_doc, "new_doc": col.new_doc}
    words = [p.label] + [c["label"] for c in out["counterparts"]]
    out["explanations"] = explanations(col.new_doc, (p.new or p.old).page, words) \
        if explain and v["kind"] != "unchanged" else []
    out["summary"] = _summary(out)
    return out


def _moved(c: dict) -> str:
    return f"{c['label']} ({c['old_printed'] or 'not printed'} -> {c['new_printed'] or 'not printed'})"


def _summary(d: dict) -> str:
    ln, k, c = d["line"], d["kind"], d["carried_from"]
    head = f"{_moved(ln)} ({d['statement']} {d['period']}, {d['old_doc']} -> {d['new_doc']})"
    tots = " and ".join(f"{t['label']} {t['printed']}" for t in d["unchanged_totals"])
    if k == "unchanged":
        return f"{head}: unchanged" + (" apart from rounding." if ln["delta"] else ".")
    if k == "presentation":
        return f"{head}: a presentation change; the line was added or dropped without moving a total both print."
    if k == "reclassification":
        cps = "; ".join(_moved(x) for x in d["counterparts"]) or "no other line"
        return f"{head}: a reclassification; the amount moved to or from {cps}, while {tots} did not change."
    if c:
        rest = ", ".join(x["label"] for x in [ln] + d["counterparts"] if x["label"] != c["line"])
        return (f"{head}: a restatement; the {STATEMENT[c['statement']]} restated {c['label']} "
                f"({c['old_printed']} -> {c['new_printed']}), and {rest} offset it here, so {tots} did not change.")
    return f"{head}: a restatement; totals moved and {len(d['changes'])} lines of the statement changed."


# -------------------------------------------------------------- explanations

def _columns(page: dict) -> list:
    """The words of a page split at a column gutter, if it has one: a vertical line near the
    middle that almost no word crosses while plenty of words sit on each side of it."""
    words, best = page["words"], None
    for x in range(int(page["w"] * 0.4), int(page["w"] * 0.6), 2):
        cross = sum(1 for w in words if w.x0 < x < w.x1)
        left = sum(1 for w in words if w.x1 <= x)
        if left > 40 and len(words) - left - cross > 40 and cross <= 0.01 * len(words):
            if best is None or cross < best[0]:
                best = (cross, x)
    if best is None:
        return [words]
    return [[w for w in words if (w.x0 + w.x1) / 2 < best[1]], [w for w in words if (w.x0 + w.x1) / 2 >= best[1]]]


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

    Strong wording (retrospectively, conformed to the current presentation,
    prior periods recast) ranks first, then mention of a changed line's label
    words, then nearness to the statement page.
    """
    words = {w for lab in labels for w in re.findall(r"[a-z]{4,}", norm_label(lab)) if w not in STOP}
    hits = []
    for pg, s in _sentences(doc):
        tier = 4 if STRONG.search(s) else 2 if MEDIUM.search(s) else 1 if WEAK.search(s) else 0
        if not tier or NOISE.search(s) or MODAL.search(s):
            continue
        named = sum(1 for w in words if re.search(rf"\b{w}", s, re.I))
        hits.append(((-tier - min(named, 2), abs(pg - page)), pg, s))
    hits.sort(key=lambda h: h[0])
    return [{"doc": doc, "page": pg, "sentence": s} for _, pg, s in hits[:limit]]


# --------------------------------------------------------------------- corpus

def _merge(sets: list) -> list:
    """Join overlapping sets until none overlap."""
    out: list = []
    for s in sets:
        for o in [o for o in out if o & s]:
            out.remove(o)
            s = s | o
        out.append(s)
    return out


def events(g, explain: bool = False) -> list:
    """Every change between the first and the latest filing that print a statement column.

    The changed lines of a column split by cause: a reclassification is a line
    with its counterparts, a presentation change is a line or block added or
    dropped, and all lines a restatement moves are one event. Rounding is none.
    """
    idx, out = Index(g), []
    for key in sorted(idx.columns):
        col = idx.column(*key)
        found = {id(p): (p, classify(col, p)) for p in col.changes()}
        found = {i: (p, v) for i, (p, v) in found.items() if v["kind"] != "unchanged"}
        links = [{i, *(id(q) for q in v["counterparts"]), "restated" if v["kind"] == "restatement" else i,
                  ("block", id(v["block"]))} for i, (p, v) in found.items()]
        for ids in _merge(links):
            lines = sorted((found[i] for i in ids if i in found), key=lambda pv: col.pairs.index(pv[0]))
            kinds = {v["kind"] for _, v in lines}
            p, v = next((pv for pv in lines if pv[0].both), lines[0])
            e = {"ticker": key[0], "statement": key[1], "period": key[2],
                 "kind": next(k for k in ("restatement", "reclassification", "presentation") if k in kinds),
                 "old_doc": col.old_doc, "new_doc": col.new_doc, "line": p.row(),
                 "counterparts": [{**q.row(), "kind": w["kind"]} for q, w in lines if q is not p],
                 "unchanged_totals": [_total(t) for t in v["totals"]], "carried_from": v["carried_from"]}
            if explain:
                e["explanations"] = explanations(col.new_doc, (p.new or p.old).page, [q.label for q, _ in lines])
            out.append(e)
    return out
