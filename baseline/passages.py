"""Passage chunks for the v2 retriever: pieces of one page, about 1,000 to 1,500 characters.

Lines    a page's words grouped into lines (tieout.tables.group_lines), a horizontal gap wider
         than 12 pt written as " | ": the same line text as the v1 page chunks.
Kinds    row    a line with a wide gap and a number that is not a year (or a dash cell)
         head   a statement title, an "in millions" caption, or column headings (years, dates)
         label  any other line of at most 60 characters ("Net sales:", "Revenue")
         text   prose
Tables   a run of row, head and label lines holding at least one row; prose ends the run. A
         table's header is the head and label lines just above its first row (at most 6 lines
         and 400 characters). Head lines between rows that are followed by more rows start a
         new table in the same run (a section such as "During 2024"), which shares the column
         headings of the run's first table.
Blocks   lines split at vertical gaps wider than 1.5 times the page's median line gap
         (paragraph breaks, gaps between table sections), never right after a header line.
Packing  blocks in reading order, at most 1,500 characters of body per passage. A block that
         does not fit closes the passage when the passage already holds 1,000 characters or the
         block is a table that fits a passage on its own; otherwise the block is split at line
         boundaries to fill the passage. A passage that starts inside a table first gets a copy
         of that table's head lines (and of the run's column headings), so each piece of a cut
         table still names its columns and units. A piece under 250 characters joins its
         neighbour. Pieces do not overlap.
Metadata every passage carries a line naming the company, ticker, form, fiscal year, period
         end, doc id and page. It is indexed with the passage and shown to the LLM.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from baseline import index
from tieout import tables

MAX_CHARS = 1500
MIN_CHARS = 1000
MIN_PIECE = 250
GAP_FACTOR = 1.5
HEADER_LINES = 6
HEADER_CHARS = 400

_YEAR = re.compile(r"^\(?(19|20)\d\d\)?$")


@dataclass
class Line:
    text: str
    y0: float
    y1: float
    kind: str
    table: int = -1
    is_header: bool = False


def _kind(words: list, text: str) -> str:
    """row, head, label or text (see the module docstring)."""
    toks = [w.text for w in words]
    wide = " | " in text
    values = [t for t in toks if tables.is_numeric(t) and not _YEAR.match(t)]
    if wide and (values or any(t in tables.DASH_TOKENS for t in toks)):
        return "row"
    if tables.CAPTION_RE.search(text) or (len(text) <= 120 and tables.TITLE_RE.search(text)):
        return "head"
    if wide or (len(text) <= 80 and tables.HEADER_HINT.search(text)):
        return "head"
    return "label" if len(text) <= 60 else "text"


def _split_long(text: str) -> list:
    """A line longer than a passage, cut at spaces into pieces that fit."""
    pieces, rest = [], text
    while len(rest) > MAX_CHARS:
        cut = rest.rfind(" ", 0, MAX_CHARS)
        cut = cut if cut > 0 else MAX_CHARS
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip()
    return pieces + [rest]


def page_lines(words: list) -> list:
    """The page's lines in reading order, each with its kind."""
    out = []
    for ws in tables.group_lines(words):
        text = index.line_text(ws)
        y0, y1 = min(w.y0 for w in ws), max(w.y1 for w in ws)
        kind = _kind(ws, text)
        for piece in _split_long(text):
            out.append(Line(piece, y0, y1, kind if piece == text else "text"))
    return out


def _header(lines: list, pending: list) -> list:
    """The last head and label lines above a table's first row, within the size limits."""
    out, size = [], 0
    for j in reversed(pending):
        size += len(lines[j].text) + 1
        if len(out) == HEADER_LINES or size > HEADER_CHARS:
            break
        out.insert(0, j)
    return out


def mark_tables(lines: list) -> list:
    """Set Line.table and Line.is_header for each table on the page.

    Returns, for each table, the first table of its run of non-prose lines: a later table in
    the same run (a new section such as "During 2024") shares that first table's columns."""
    table, firsts, pending = -1, [], []
    for i, ln in enumerate(lines):
        if ln.kind == "text":
            table, pending = -1, []
            continue
        if ln.kind != "row":
            pending.append(i)
            continue
        members = pending + [i]
        if table < 0 or any(lines[j].kind == "head" for j in pending):
            header = _header(lines, pending)
            for j in pending:
                if j not in header:
                    lines[j].table = table      # above the header: the previous table's tail, or none
            firsts.append(firsts[table] if table >= 0 else len(firsts))
            table = len(firsts) - 1
            for j in header:
                lines[j].is_header = True
            members = header + [i]
        for j in members:
            lines[j].table = table
        pending = []
    return firsts


def _prefixes(lines: list, firsts: list):
    """A function giving the header copy for a passage that starts at a given line.

    Inside the first table of a run, the copy is that table's head lines (title, caption,
    column headings). Inside a later table of the run it is the first table's column lines
    (those with column gaps, and unit captions), then the later table's own head lines."""
    heads, columns = [], []
    for t in range(len(firsts)):
        header = [ln for ln in lines if ln.table == t and ln.is_header]
        heads.append([ln.text for ln in header if ln.kind == "head"] or [ln.text for ln in header])
        columns.append([ln.text for ln in header if " | " in ln.text or tables.CAPTION_RE.search(ln.text)])

    def prefix(ln: Line) -> list:
        t = ln.table
        if t < 0:
            return []
        first = firsts[t]
        if t == first:
            return [] if ln.is_header else list(heads[t])
        return columns[first] + ([] if ln.is_header else heads[t])
    return prefix


def blocks(lines: list) -> list:
    """Line indexes grouped at wide vertical gaps; a header line stays with the line below it."""
    if not lines:
        return []
    gaps = [lines[i].y0 - lines[i - 1].y1 for i in range(1, len(lines))]
    positive = [g for g in gaps if g > 0]
    cut = GAP_FACTOR * statistics.median(positive) if positive else float("inf")
    out = [[0]]
    for i in range(1, len(lines)):
        prev = lines[i - 1]
        glued = prev.is_header and prev.table >= 0 and prev.table == lines[i].table
        if gaps[i - 1] > cut and not glued:
            out.append([i])
        else:
            out[-1].append(i)
    return out


def _size(texts: list) -> int:
    return sum(len(t) + 1 for t in texts)


def _merge_small(pieces: list) -> list:
    """A piece under MIN_PIECE characters joins the next piece, or the previous one if last."""
    out = []
    for i, p in enumerate(pieces):
        if out and _size(out[-1]) < MIN_PIECE:
            out[-1] = out[-1] + p
        elif out and i == len(pieces) - 1 and _size(p) < MIN_PIECE:
            out[-1] = out[-1] + p
        else:
            out.append(list(p))
    return out


def pack(lines: list, firsts: list) -> list:
    """Passage bodies for one page (see Packing in the module docstring)."""
    prefix = _prefixes(lines, firsts)
    out, cur = [], []

    def add(i: int) -> None:
        nonlocal cur
        ln = lines[i]
        if cur and _size(cur) + len(ln.text) + 1 > MAX_CHARS:
            out.append(cur)
            cur = []
        if not cur:
            cur = prefix(ln)
        cur.append(ln.text)

    for block in blocks(lines):
        texts = [lines[i].text for i in block]
        table_alone = any(lines[i].kind == "row" for i in block) and _size(texts) <= MAX_CHARS
        if cur and _size(cur) + _size(texts) > MAX_CHARS and (_size(cur) >= MIN_CHARS or table_alone):
            out.append(cur)
            cur = []
        for i in block:
            add(i)
    if cur:
        out.append(cur)
    return ["\n".join(p) for p in _merge_small(out)]


def page_passages(words: list) -> list:
    """Passage bodies of one page, in reading order."""
    lines = page_lines(words)
    return pack(lines, mark_tables(lines))


def meta_line(m: dict, doc_id: str, page: int) -> str:
    """The metadata line that starts every passage."""
    fy = m["fiscal_year"].removeprefix("FY")
    return (f"{m['company']} ({m['ticker']}) | {m['form']} | fiscal year {fy} | period ended {m['period_end']} | "
            f"{doc_id} | page {page}")


def build(verbose: bool = True) -> list:
    """Every passage of every readable filing: {"doc", "ticker", "page", "part", "meta", "body"}."""
    from tieout import pdfio
    out = []
    for doc_id, m in index.manifest().items():
        if doc_id in index.SKIP or not pdfio.cache_path(doc_id).exists():
            continue
        pages = pdfio.load_pages(doc_id)
        before = len(out)
        for n, p in enumerate(pages, start=1):
            if not p["words"]:
                continue
            for part, body in enumerate(page_passages(p["words"])):
                out.append({"doc": doc_id, "ticker": m["ticker"], "page": n, "part": part,
                            "meta": meta_line(m, doc_id, n), "body": body})
        if verbose:
            print(f"passages {doc_id:24s} {len(pages):4d} pages {len(out) - before:5d} passages", flush=True)
    return out
