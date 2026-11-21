"""Find the primary financial statements inside a filing and parse them.

A statement page is recognised by its printed title near the top of the page
and by being dense with numbers. The table of contents and the audit report
mention the same titles but carry few numbers, so they drop out. A filing
usually has several pages that pass (five-year summaries, parent-only
statements, average-balance tables at banks), so candidates are scored and
the best income statement, balance sheet and cash-flow statement are chosen
together: real primary statements have two or three period columns, foot
almost completely, and sit within a few pages of each other.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, replace
from typing import Optional

from .calc import coverage, discover, ties
from .tables import Table, build_table, group_lines, is_numeric

KINDS = {
    "IS": re.compile(
        r"statements? of (consolidated )?(operations|income|earnings)\b|income statements?\b|"
        r"consolidated results of operations|statements? of (consolidated )?net income", re.I),
    "BS": re.compile(
        r"balance sheets?\b|statements? of (consolidated )?financial (position|condition)|"
        r"consolidated financial position", re.I),
    "CF": re.compile(r"statements? of (consolidated )?cash flows?|cash flows? statements?", re.I),
}
NOT_PRIMARY = re.compile(
    r"comprehensive (income|loss|earnings)|stockholders|shareholders|shareowners|changes in (total )?equity|"
    r"parent company|schedule|supplement|quarterly|selected|pro forma|management|discussion|"
    r"notes to|report of|index to|average balance|item 7|information relating to|relating to the", re.I)
OTHER_TITLE = re.compile(
    r"statements? of (consolidated )?(comprehensive|stockholders|shareholders|shareowners|changes in|"
    r"equity|redeemable|total equity)|comprehensive (income|loss|earnings)\b", re.I)
YEAR_ONLY = re.compile(r"^(19|20)\d\d$")
COMBINED_OK = re.compile(r"statements? of (consolidated )?(operations|income|earnings) and comprehensive", re.I)
ANY_TITLE = re.compile(r"statements?\b|balance sheets?|notes to|report of independent|financial position", re.I)
MIN_NUMBERS = 18
MAX_CONTINUATION_PAGES = 2


@dataclass
class Statement:
    kind: str
    table: Table
    checks: list
    ties: list
    verified: int
    cells: int
    score: float = 0.0

    @property
    def ratio(self) -> float:
        return self.verified / self.cells if self.cells else 0.0


def page_lines(page: dict) -> list:
    """Text lines of a page without the page-number band at the very bottom.

    The top band is kept: some filers print the statement title there.
    """
    h = page["h"]
    words = [w for w in page["words"] if 12 < w.yc < h - 24]
    return group_lines(words)


def readable(pages: list) -> bool:
    """False when the PDF's fonts carry no Unicode map, so the text layer is
    glyph codes ("(cid:29)(cid:69)...") rather than characters."""
    words = [w.text for p in pages[: min(len(pages), 80)] for w in p["words"]]
    if not words:
        return False
    return sum(1 for w in words if "(cid:" in w) / len(words) < 0.2


def split_regions(lines: list) -> list:
    """Cut a page at statement titles and return [(kind, title, lines)].

    A title is a short heading line with no values on it. One page can hold
    two statements (income and comprehensive income), and a statement can
    start halfway down a page after an introduction, so titles are looked for
    on the whole page, not only at the top.
    """
    marks = []                                    # (line index, kind or None, title)
    texts = [" ".join(w.text for w in ln) for ln in lines]
    for i, ln in enumerate(lines):
        text = texts[i]
        if re.search(r"\b(statements?|sheets?) of$", text) and i + 1 < len(lines) and len(texts[i + 1]) < 40:
            text = text + " " + texts[i + 1]      # a title wrapped onto a second line
        if len(text) > 110 or text.rstrip().endswith((".", ";")) or text[:1].islower():
            continue                              # a sentence that mentions a statement
        if any(is_numeric(w.text) and len(w.text) >= 2 and not YEAR_ONLY.match(w.text) for w in ln):
            continue                              # a table row or an index line with page numbers
        if OTHER_TITLE.search(text) and not COMBINED_OK.search(text):
            marks.append((i, None, text))
            continue
        if NOT_PRIMARY.search(text) and not COMBINED_OK.search(text):
            continue
        for kind, pat in KINDS.items():
            if pat.search(text):
                marks.append((i, kind, text))
                break
    regions = []
    for j, (i, kind, title) in enumerate(marks):
        if kind is None:
            continue
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        # a title repeated on the next line ("... (continued)") does not end the region
        while j + 1 < len(marks) and marks[j + 1][0] - i <= 2 and marks[j + 1][1] == kind:
            j += 1
            end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        regions.append((kind, title, lines[i:end]))
    return regions


def candidates(doc_id: str, pages: list) -> list:
    out = []
    continued = set()                             # (kind, page) absorbed into the statement before it
    for i, page in enumerate(pages):
        lines = page_lines(page)
        if not lines:
            continue
        for kind, title, region in split_regions(lines):
            numbers = sum(1 for ln in region for w in ln if is_numeric(w.text) and len(w.text) >= 3)
            if numbers < MIN_NUMBERS:
                continue
            t = build_table(doc_id, i + 1, region, title)
            if t is None or len(t.rows) < 8:
                continue
            st = _statement(kind, t)
            for nxt in range(i + 1, min(i + 1 + MAX_CONTINUATION_PAGES, len(pages))):
                longer = _with_next_page(doc_id, st, pages[nxt], nxt + 1)
                if longer is None:
                    break
                st = longer
                continued.add((kind, nxt + 1))
            out.append(st)
    return [st for st in out if (st.kind, st.table.page) not in continued]


def _statement(kind: str, t: Table) -> Statement:
    checks = discover(t)
    ok, total = coverage(t, checks)
    st = Statement(kind, t, checks, ties(t), ok, total)
    st.score = _score(st)
    return st


def _with_next_page(doc_id: str, st: Statement, page: dict, page_no: int) -> Optional[Statement]:
    """The statement extended by the next page, if that page continues it.

    A long cash-flow statement or a bank's balance sheet runs over two pages.
    The next page is taken as a continuation when it repeats the title with
    "(continued)", or when it carries no title of its own and the arithmetic
    joins the pages: a formula of the combined table has rows on both (the
    subtotal that closes "financing activities" sits on the second page and
    sums rows of the first). A separate statement that merely follows, such as
    comprehensive income, adds up only within itself and is left alone.
    """
    lines = page_lines(page)
    if not lines:
        return None
    head = " ".join(w.text for ln in lines[:8] for w in ln)
    same_kind = [region for kind, _title, region in split_regions(lines) if kind == st.kind]
    titled = bool(same_kind) and re.search(r"\bcontinued\b", head, re.I) is not None
    if titled:
        region = same_kind[0]
    elif ANY_TITLE.search(head):
        return None                               # another statement or a note starts here
    else:
        region = lines
    t2 = build_table(doc_id, page_no, region, st.table.title)
    if t2 is None or t2.ncols != st.table.ncols:
        return None
    first_page_rows = len(st.table.rows)
    rows = list(st.table.rows) + [replace(r, idx=first_page_rows + n) for n, r in enumerate(t2.rows)]
    longer = _statement(st.kind, replace(st.table, rows=rows))
    # a formula that holds in all but one column across the page break is still evidence of the join
    joined = any(min(ch.rows) < first_page_rows <= max(ch.rows) for ch in longer.checks)
    if longer.cells - st.cells < 4 or not (titled or joined):
        return None
    return longer


def _score(st: Statement) -> float:
    n = st.table.ncols
    if st.kind == "BS":
        shape = {2: 1.0, 3: 0.6}.get(n, 0.2)
    else:
        shape = 1.0 if 2 <= n <= 4 else 0.3
    dated = sum(1 for c in st.table.columns if c.year)
    return 2.0 * st.ratio + shape + min(st.cells, 60) / 60 + (0.5 if dated == n else 0.0)


def select_primary(cands: list) -> dict:
    """Choose one statement per kind, preferring a tight cluster of pages."""
    by_kind = {k: [c for c in cands if c.kind == k] for k in ("IS", "BS", "CF")}
    kinds = [k for k in by_kind if by_kind[k]]
    if not kinds:
        return {}
    best, best_val = None, -1e9
    for combo in itertools.product(*[sorted(by_kind[k], key=lambda c: -c.score)[:8] for k in kinds]):
        pages = [c.table.page for c in combo]
        val = sum(c.score for c in combo) - 0.04 * (max(pages) - min(pages))
        if val > best_val:
            best, best_val = combo, val
    return {c.kind: c for c in best}


def find_statements(doc_id: str, pages: list) -> dict:
    return select_primary(candidates(doc_id, pages))
