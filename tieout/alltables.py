"""Find and parse every numeric table in a filing, not only the primary statements.

A filing prints dozens of tables outside the three statements: revenue by
segment, product or geography, debt maturities, lease costs, tax rate
reconciliations, quarterly data. None of them has ruling lines either, so a
page is cut into table regions by looking at the shape of each text line:

    num      a label on the left, then a run of values set off by a wide gap
    hdr      a line of years ("2026  Change  2025  Change  2024")
    colhdr   text sitting over the value columns ("Year Ended December 31,")
    label    a short line on the left ("Net sales:", a wrapped label)
    prose    a long line of running text
    note     a footnote ("(1) China includes Hong Kong and Taiwan.")

A region is a run of num lines with the hdr, colhdr and label lines that sit
between them. Prose, footnotes, big vertical gaps, a fresh year header after
values, and a jump in the value columns all close a region. Each region is
handed to tables.build_table unchanged; this module only decides where tables
start and stop, what they are called, and what scale they are printed in.

Nothing here knows any company or any accounting.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Optional

from .tables import (CAPTION_RE, DASH_TOKENS, HEADER_HINT, MONTH_ABBR, MONTHS, TITLE_RE, YEAR_RE, Table,
                     _mostly_years, _scale, build_table, is_numeric, right_edge)
from .statements import page_lines

LABEL_MARK = "⁠"            # appended to a number that is part of a label, so it stops looking numeric
VALUE_WORDS = {"$", "%", "NA", "NM", "N/A", "n/a", "nm", "N/M", "n/m", "*", "—%", "-%"}
MARKER_RE = re.compile(r"^\(?[a-z]\)$|^\(\d{1,2}\)$|^\*{1,3}$")
# "(in millions)", "(dollars in millions)", "(millions of dollars)", "$ in millions"
SCALE_WORD_RE = re.compile(r"\bin (millions|thousands|billions)\b|"
                           r"\b(millions|thousands|billions) of (u\.s\. )?dollars\b", re.I)
UNIT_CAPTION_RE = re.compile(r"(?:^|\()\s*(?:[\w$, ]{0,25}?\bin\s+(millions|thousands|billions)\b|"
                             r"(millions|thousands|billions)\s+of\s+(?:u\.s\.\s+)?dollars\b)", re.I)
NOTE_RE = re.compile(r"^note\s+\d{1,2}\b", re.I)
NOTE_HEAD_RE = re.compile(r"^note\s+\d{1,2}\s*[.:\-–—]", re.I)   # "Note 14 – Segment ...", "NOTE 14:"
ITEM_RE = re.compile(r"^item\s+\d{1,2}[ABC]?\s*[.:\-–—]?\s+[A-Z][^\d]*$", re.I)   # "Item 7. Management's ..."
INTRO_RE = re.compile(r"^(?:the\s+)?(?:following|below)\s+tables?\s+(?:below\s+)?"
                      r"(?:shows?|presents?|summarizes?|sets?\s+forth|provides?|reflects?|details?|"
                      r"includes?|lists?|illustrates?|reconciles?|disaggregates?|contains?)\s+", re.I)
CONTINUED_RE = re.compile(r"^\(?\s*(table\s+)?continued\b|\bcontinued (from|on) (the )?(previous|next|following) page", re.I)
SENTENCE_SPLIT =re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
TITLE_MAX = 160


# ------------------------------------------------------------------ word prep

def _wordy(text: str) -> bool:
    return any(ch.isalpha() for ch in text) and text not in VALUE_WORDS and not MARKER_RE.match(text)


def _bind_label_numbers(ln: list) -> list:
    """Return a copy of the line in which numbers that belong to the label are
    no longer numeric.

    Handles: "China (1)  82,378" (footnote glued to a label), "3.625% Notes
    due 2031  1,000" (numbers inside a label), "2027  $ 10,000" (a maturity
    table whose label is a year) and "(1) China includes ..." (a footnote line).
    tables.find_columns would otherwise turn those into value columns.
    """
    out = []
    for k, w in enumerate(ln):
        if is_numeric(w.text):
            prev = ln[k - 1] if k else None
            nxt = ln[k + 1] if k + 1 < len(ln) else None
            # "January 31, 2024" inside a label ("Balances as of January 31, 2024")
            # (a date in a column header has no amounts after it and stays a year)
            dated = prev is not None and re.fullmatch(r"\d{1,2},", prev.text) and any(
                is_numeric(x.text) and not YEAR_RE.match(x.text) for x in ln[k + 1:])
            glued_prev = prev is not None and (_wordy(prev.text) or dated) and w.x0 - prev.x1 < 6
            glued_next = nxt is not None and nxt.text[:1].isalpha() and nxt.text not in VALUE_WORDS \
                and nxt.x0 - w.x1 < 6
            lead_label = k == 0 and nxt is not None and nxt.text[:1].isalpha() and nxt.text not in VALUE_WORDS \
                and nxt.x0 - w.x1 < 15
            lead_year = k == 0 and YEAR_RE.match(w.text) and any(
                is_numeric(x.text) and not YEAR_RE.match(x.text) for x in ln[1:])
            if glued_prev or glued_next or lead_label or lead_year:
                w = replace(w, text=w.text + LABEL_MARK)
        out.append(w)
    return out


def _strip_percent_signs(ln: list, pct: set) -> list:
    """Drop stand-alone "%" tokens ("11 %") and remember which number they follow."""
    out = []
    for w in ln:
        if w.text == "%" and out and (is_numeric(out[-1].text) or out[-1].text in DASH_TOKENS) \
                and w.x0 - out[-1].x1 < 8:
            p = out[-1]
            pct.add((round(p.x0, 1), round(p.y0, 1)))
            continue
        out.append(w)
    return out


# ------------------------------------------------------------ line classes

@dataclass
class Line:
    words: list
    kind: str
    text: str
    edges: list = field(default_factory=list)     # right edges of the values on a num line
    split: int = 0                                # label line with column headers after a wide gap

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def height(self) -> float:
        hs = sorted(w.y1 - w.y0 for w in self.words)
        return hs[len(hs) // 2]


def _value_run(ln: list) -> int:
    """Index where the trailing run of value tokens starts."""
    j = len(ln)
    while j > 0:
        t = ln[j - 1].text
        if is_numeric(t) or t in DASH_TOKENS or t in VALUE_WORDS or MARKER_RE.match(t):
            j -= 1
        else:
            break
    return j


DAY_RE = re.compile(r"^\d{1,2},?$")


def _date_header(ln: list) -> bool:
    """"Years Ended  July 25, 2026  July 26, 2025  July 27, 2024": two or more
    years, and from the first year on only dates and capitalised column words.
    "during 2026 compared to 2025" fails on "compared"."""
    first = next((k for k, w in enumerate(ln) if YEAR_RE.match(w.text)), None)
    if first is None or sum(1 for w in ln if YEAR_RE.match(w.text)) < 2:
        return False
    for w in ln[first:]:
        t = w.text.rstrip(",")
        month = t.lower() in MONTHS or (len(t.rstrip(".")) <= 4 and t.lower()[:3] in MONTH_ABBR)
        if not (YEAR_RE.match(t) or DAY_RE.match(w.text) or month or t[:1].isupper() or t in VALUE_WORDS):
            return False
    return True


def _classify(ln: list, left: float, right: float) -> Line:
    text = " ".join(w.text for w in ln).replace(LABEL_MARK, "")
    width = right - left
    j = _value_run(ln)
    vals = [w for w in ln[j:] if is_numeric(w.text) or w.text in DASH_TOKENS]
    if all(MARKER_RE.match(w.text) for w in ln) or CONTINUED_RE.search(text):
        return Line(ln, "mark", text)                   # "(g)   (g)" markers over cells, "(Table continued ...)"
    # values count only when set off from the words before them by a wide
    # gap; "compared to 2025" inside a sentence is not a column
    set_off = bool(vals) and (j == 0 or ln[j].x0 - ln[j - 1].x1 >= 8)
    if _mostly_years(ln) and (set_off or _date_header(ln)):
        return Line(ln, "hdr", text)                    # "2026  Change  2025  Change  2024"
    if set_off:
        # a label, then values set off by a wide gap: a table row
        return Line(ln, "num", text, [right_edge(w) if is_numeric(w.text) else w.x1 for w in vals])
    if ln[0].text.endswith(LABEL_MARK) and MARKER_RE.match(ln[0].text.rstrip(LABEL_MARK)) or text.startswith("*"):
        return Line(ln, "note", text)                   # "(1) China includes Hong Kong and Taiwan."
    if ln[0].x0 > left + 0.25 * width:
        return Line(ln, "colhdr", text)                 # "Year Ended December 31," over the columns
    if CAPTION_RE.search(text) and len(text) < 90:
        return Line(ln, "caption", text)                # "(In millions, except per share data)"
    split = _right_split(ln, left, width)
    if not split and ln[-1].x1 - ln[0].x0 > 0.62 * width and len(ln) >= 7:
        return Line(ln, "prose", text)
    return Line(ln, "label", text, split=split)


def _right_split(ln: list, left: float, width: float) -> int:
    """Index of the first column-header word on a line that starts with a
    title on the left: "Costs Incurred in Property Acquisitions,      Canada/".
    0 when the line is one piece."""
    gaps = [(b.x0 - a.x1, k + 1) for k, (a, b) in enumerate(zip(ln, ln[1:]))]
    if not gaps:
        return 0
    gap, k = max(gaps)
    if gap >= 40 and ln[k].x0 > left + 0.4 * width and not any(is_numeric(w.text) for w in ln[k:]):
        return k
    return 0


# ------------------------------------------------------------------ regions

@dataclass
class Region:
    start: int
    end: int = -1                 # index of the last num line
    nnum: int = 0
    edges: list = field(default_factory=list)


def _big_gap(a: Line, b: Line) -> bool:
    return b.y0 - a.y1 > 2.6 * max(a.height, b.height)


def _new_columns(reg: Region, ln: Line) -> bool:
    """A num line whose values share no column edge with the region so far
    starts another table (two tables with different layouts, no gap)."""
    if reg.nnum < 2:
        return False
    return not any(abs(e - f) <= 5 for e in ln.edges for f in reg.edges)


def _segment(lines: list) -> list:
    regions, cur = [], None

    def close():
        nonlocal cur
        if cur is not None and cur.nnum >= 2:
            regions.append(cur)
        cur = None

    for i, ln in enumerate(lines):
        if i > 0 and cur is not None and _big_gap(lines[i - 1], ln):
            close()                                     # whitespace ends a table (and a stray header)
        if ln.kind == "num":
            if cur is not None and _new_columns(cur, ln):
                close()
            if cur is None:
                cur = Region(start=i)
            cur.nnum += 1
            cur.end = i
            cur.edges.extend(ln.edges)
        elif ln.kind == "hdr":
            # a year header after values starts the next table
            if cur is not None and cur.nnum:
                close()
            if cur is None:
                cur = Region(start=i)
        elif ln.kind in ("prose", "note"):
            close()
        elif ln.kind == "colhdr" and cur is None:
            cur = Region(start=i)
        # label, caption, mark and colhdr inside a region: keep going
    close()
    return regions


def _extend_up(lines: list, start: int, floor: int) -> int:
    """Pull in the header block above the first line of a region: column
    headers, the scale caption, and label-column header words such as
    "As of or for the year ended" / "December 31,"."""
    i = start
    while i - 1 > floor and start - (i - 1) <= 8:
        ln = lines[i - 1]
        if _big_gap(ln, lines[i]):
            break
        ok = ln.kind in ("hdr", "colhdr", "caption", "mark") or (ln.kind == "label" and ln.split) or (
            ln.kind == "label" and (ln.text.endswith(":") or HEADER_HINT.search(ln.text)) and len(ln.text) < 60)
        if not ok:
            break
        i -= 1
    return i


# -------------------------------------------------------------------- titles

def _is_heading(ln: Line) -> bool:
    t = ln.text.strip()
    return (ln.kind == "label" and 2 < len(t) <= 100 and t[:1].isupper()
            and not t.endswith((".", ",", ";")) and not CAPTION_RE.search(t))


def _last_sentence(block: list) -> str:
    text = " ".join(ln.text for ln in block).strip()
    parts = SENTENCE_SPLIT.split(text)
    return parts[-1].strip() if parts else text


def _clean_title(text: str) -> str:
    """"The following table shows net sales by category for 2026 (dollars in
    millions):" becomes "Net sales by category for 2026"."""
    t = re.sub(r"^table of contents\s+", "", text.strip(), flags=re.I)   # running page header
    t = INTRO_RE.sub("", t).rstrip(":").strip()
    t = re.sub(r"\s*\((?:[^()]*\b(?:in|amounts in|dollars in)\s+(?:millions|thousands|billions)[^()]*)\)\s*$",
               "", t, flags=re.I).rstrip(":,; ").strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if len(t) > TITLE_MAX:
        t = t[:TITLE_MAX - 3].rsplit(" ", 1)[0] + "..."
    return t


def _blocks_above(lines: list, start: int, floor: int) -> list:
    """Text blocks above a region, nearest first. A block is a run of
    consecutive text lines without a big gap; a heading is its own block."""
    blocks, cur = [], []
    for i in range(start - 1, floor, -1):
        ln = lines[i]
        if ln.kind in ("num", "hdr"):
            break                                       # another table above
        if ln.kind in ("note", "mark"):
            continue                                    # the footnotes of the table above
        if cur and (_big_gap(ln, cur[0]) or _is_heading(ln)):
            blocks.append(cur)
            cur = []
        cur.insert(0, ln)
        if _is_heading(ln) and len(cur) == 1:
            blocks.append(cur)
            cur = []
        if len(blocks) >= 4:
            break
    if cur:
        blocks.append(cur)
    return blocks


def _title(lines: list, start: int, floor: int) -> tuple:
    """(title, intro text) for the region starting at `start`.

    Rule 1: a paragraph right above that ends with ":" introduces the table;
            its last sentence is the title ("The following table shows ...:").
    Rule 2: otherwise the nearest heading line above ("Disaggregated Revenues").
    Rule 3: otherwise the last sentence of the nearest paragraph.
    Rule 0 (first): a title printed on the same lines as the column headers,
            left of them ("Costs Incurred in Property Acquisitions, ...").
    """
    left_parts = []
    for ln in lines[start:]:
        if ln.kind in ("num", "hdr"):
            break
        if ln.kind == "label" and ln.split:
            left_parts.append(" ".join(w.text for w in ln.words[:ln.split]))
    if left_parts:
        return _clean_title(" ".join(left_parts[:2])), ""
    blocks = [b for b in _blocks_above(lines, start, floor)
              if not all(ln.kind in ("caption", "mark") for ln in b)]
    lead = []
    for ln in lines[start:]:
        if ln.kind != "caption":
            break
        lead.append(ln)                     # caption lines the region pulled in
    lead_text = " ".join(ln.text for ln in lead)
    if not blocks:
        if lead_text.endswith(":") and len(lead_text) > 30:
            return _clean_title(_last_sentence(lead)), lead_text
        return "", ""
    first = blocks[0]
    first_is_heading = len(first) == 1 and _is_heading(first[0])
    if lead and not first_is_heading and not first[-1].text.rstrip().endswith((":", ".")):
        first = first + lead                # "...of our customers (in" / "millions):"
    elif lead_text.endswith(":") and len(lead_text) > 30:
        # the introducing sentence is itself the caption line:
        # "The following table presents revenues by type (in millions):"
        return _clean_title(_last_sentence(lead)), lead_text
    if not first_is_heading:
        sent = _last_sentence(first)
        if sent.endswith(":"):
            return _clean_title(sent), sent
    for b in blocks:
        if len(b) == 1 and _is_heading(b[0]):
            return _clean_title(b[0].text), ""
    sent = _last_sentence(first)
    return _clean_title(sent), sent


# --------------------------------------------------------------------- scale

def infer_default_scale(pages: list) -> float:
    """The scale a filing states for its financial statements.

    Votes over the short unit captions in the whole document, each caption
    line once: "(In millions, except per-share amounts)", "(dollars in
    millions)", "(millions of dollars)", "Millions of dollars, except ...".
    Only captions that open a line or a bracket count, and only money units
    ("Thousands of barrels per day" does not vote). The most common unit wins;
    with no caption at all the default is 1 (units as printed).
    """
    votes: Counter = Counter()
    for page in pages:
        for ln in page_lines(page):
            text = " ".join(w.text for w in ln)
            if len(text) > 160:
                continue
            m = UNIT_CAPTION_RE.search(text)
            if m:
                votes[(m.group(1) or m.group(2)).lower()] += 1
    if not votes:
        return 1.0
    return {"thousands": 1e3, "millions": 1e6, "billions": 1e9}[votes.most_common(1)[0][0]]


def _apply_scale(t: Table, intro: str, default_scale: float) -> None:
    """Scale from the table's own caption (build_table already read it), else
    from the sentence that introduces it ("... (dollars in millions):"), else
    the document default."""
    own = t.caption + " " + t.header_text
    if t.scale != 1.0 or SCALE_WORD_RE.search(own) or re.search(r"^\(?\s*(millions|thousands)\b", own.strip(), re.I):
        t.scale_source = "caption"
        return
    if SCALE_WORD_RE.search(intro):
        t.scale, t.share_scale = _scale(intro)
        t.scale_source = "intro"
        return
    t.scale = t.share_scale = default_scale
    t.scale_source = "inherited"


# --------------------------------------------------------------- post-build

def _mark_percent_cells(t: Table, pct: set) -> None:
    for r in t.rows:
        for c in r.cells.values():
            if (round(c.bbox[0], 1), round(c.bbox[1], 1)) in pct:
                c.is_percent = True


def _percent_columns(t: Table) -> list:
    """Change columns ("11 %", "(4) %") are derived from the amounts beside
    them and never foot. A column is a percent column when many of its cells
    carry "%", or when any does and all its values are small. A table of
    nothing but percentages keeps its columns (returns [])."""
    out = []
    for c in t.columns:
        cells = [r.cells[c.idx] for r in t.rows if c.idx in r.cells]
        if not cells:
            continue
        frac = sum(1 for x in cells if x.is_percent) / len(cells)
        small = all(x.value is None or abs(x.value) <= 100.5 for x in cells)
        headed = re.search(r"%|\bpercent", c.header, re.I)       # "% Change '25 vs. '24"
        if frac >= 0.3 or (frac > 0 and small and not any(x.has_dollar for x in cells)) \
                or (headed and small):
            out.append(c)
    return out if len(out) < t.ncols else []


def _without_columns(region: list, drop: list, cols: list) -> list:
    """The region's word lines with the dropped columns erased: their values,
    and on year-header lines every word whose centre is nearer a dropped
    column than a kept one ("Change", "July 25, 2026" over a percent column).
    A second build_table then gives the period headers to the amount columns
    only. Spanning lines such as "Year Ended December 31," are left alone."""
    def centre(c):
        return (c.x_left + c.x_right) / 2

    dropped = {c.idx for c in drop}

    def nearest_dropped(w):
        xr = right_edge(w) if is_numeric(w.text) else w.x1
        mid = (w.x0 + xr) / 2
        return min(cols, key=lambda c: abs(centre(c) - mid)).idx in dropped

    def in_col(w, c):
        xr = right_edge(w) if is_numeric(w.text) else w.x1
        return c.x_left - 6 <= (w.x0 + xr) / 2 <= c.x_right + 6

    out = []
    for ln in region:
        keep = []
        for w in ln.words:
            value = is_numeric(w.text) or w.text in DASH_TOKENS
            if ln.kind == "hdr" and nearest_dropped(w):
                continue
            if value and ln.kind != "hdr" and any(in_col(w, c) for c in drop):
                continue
            keep.append(w)
        if keep:
            out.append(keep)
    return out


def _defuse_titles(region: list) -> list:
    """build_table stops its header search at any line that reads like a
    statement title (tables.TITLE_RE). Inside a note table a section label
    such as "Selected balance sheet data (period-end)" sits between the year
    header and the first row and would cut the header off. Lines below the
    first header line get a mark inside each word so they no longer match;
    the mark is removed from labels afterwards."""
    first_hdr = next((i for i, ln in enumerate(region) if ln.kind == "hdr"), 0)
    out = []
    for i, ln in enumerate(region):
        if i > first_hdr and ln.kind == "label" and TITLE_RE.search(ln.text):
            words = [replace(w, text=w.text[:1] + LABEL_MARK + w.text[1:]) if len(w.text) > 1 else w
                     for w in ln.words]
            ln = replace(ln, words=words)
        out.append(ln)
    return out


def _build(doc_id: str, page_no: int, region: list, title: str, pct: set) -> Optional[Table]:
    """build_table, then once more without the percent-change columns."""
    region = _defuse_titles(region)
    t = build_table(doc_id, page_no, [ln.words for ln in region], title)
    if t is None:
        return None
    _mark_percent_cells(t, pct)
    drop = _percent_columns(t)
    if drop:
        t = build_table(doc_id, page_no, _without_columns(region, drop, t.columns), title)
        if t is None:
            return None
        _mark_percent_cells(t, pct)
    t.pct_columns = drop
    return t


def _clean_labels(t: Table) -> None:
    t.caption = t.caption.replace(LABEL_MARK, "")
    t.header_text = t.header_text.replace(LABEL_MARK, "")
    for c in t.columns:
        c.header = c.header.replace(LABEL_MARK, "")
    for r in t.rows:
        r.label = _clean_label(r.label)
        r.section = _clean_label(r.section)


def _clean_label(text: str) -> str:
    """Drop the word marks, dot leaders ("Americas ........") and trailing
    footnote markers ("Services (1)", "Home Lending(b)")."""
    t = text.replace(LABEL_MARK, "")
    t = re.sub(r"\s*\.{2,}[.\s]*", " ", t).strip()
    t = re.sub(r"(\s*\((?:\d{1,2}|[a-z])\))+\s*$", "", t).strip()
    return t


def _group_headers(t: Table, region: list) -> None:
    """Attach spanning group headers ("Corporate   Reconciling Items   Total"
    printed over repeated "2025 2024 2023" columns) to the columns below them.
    Groups are the runs of columns between restarts of the year sequence."""
    years = [c.year for c in t.columns]
    if t.ncols < 4 or any(y is None for y in years):
        return
    groups, cur = [], [t.columns[0]]
    for a, b in zip(t.columns, t.columns[1:]):
        if b.year >= a.year if years[0] > years[1] else b.year <= a.year:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    if len(groups) < 2:
        return
    first_hdr = next((i for i, ln in enumerate(region) if ln.kind == "hdr"), None)
    if first_hdr is None:
        return
    for ln in region[:first_hdr]:
        if ln.kind != "colhdr":
            continue
        for phrase in _phrases(ln.words):
            x0, x1 = phrase[0].x0, phrase[-1].x1
            mid = (x0 + x1) / 2
            for g in groups:
                lo = min(c.x_left for c in g) - 10
                hi = max(c.x_right for c in g) + 10
                if lo <= mid <= hi:
                    name = " ".join(w.text for w in phrase)
                    for c in g:
                        c.group = (getattr(c, "group", "") + " " + name).strip()
    for c in t.columns:
        if getattr(c, "group", ""):
            c.header = f"{c.group} | {c.header}" if c.header else c.group


def _text_headers(t: Table, region: list) -> None:
    """Category column headers ("United States  Canada/Other Americas  Europe",
    "Pension Plans", "Equity  Fixed income  Total"). build_table keeps only
    period words; here the other header words above the first row are given
    to the column under them (stacked lines are joined top to bottom), or to
    every column a wide phrase spans. Stored as column.group and prefixed to
    column.header. Skipped when _group_headers already named the columns."""
    if any(getattr(c, "group", "") for c in t.columns):
        return
    label_limit = min(c.x_left for c in t.columns) - 4
    centres = [(c.x_left + c.x_right) / 2 for c in t.columns]
    pitch = (centres[-1] - centres[0]) / (len(centres) - 1) if len(centres) > 1 else 80.0
    names = [[] for _ in t.columns]
    dropped = [(c.x_left + c.x_right) / 2 for c in getattr(t, "pct_columns", [])]
    for ln in region:
        if ln.kind == "num":
            break
        if ln.kind == "hdr":
            continue                                    # period lines: build_table read them
        words =[w for w in ln.words if w.x0 >= label_limit - 10 and not is_numeric(w.text)]
        if not words:
            continue
        for phrase in _phrases(words):
            text = " ".join(w.text for w in phrase).replace(LABEL_MARK, "")
            # pieces of dates or of a wrapped caption ("27,", "value)") are not names
            if re.search(r"\d", text) or not text[:1].isalpha() or text.count(")") > text.count("(") \
                    or text.lower().rstrip(".,") in MONTHS:
                continue
            if HEADER_HINT.search(text) or CAPTION_RE.search(text) or CONTINUED_RE.search(text) \
                    or MARKER_RE.match(text) or text in VALUE_WORDS:
                continue
            x0, x1 = phrase[0].x0, phrase[-1].x1
            mid = (x0 + x1) / 2
            if dropped and min(abs(d - mid) for d in dropped) < min(abs(cx - mid) for cx in centres):
                continue                                # "PERCENTAGE" over dropped percent columns
            hit = [i for i, cx in enumerate(centres) if x0 - 8 <= cx <= x1 + 8]
            if x1 - x0 < 1.3 * pitch or not hit:
                near = min(range(len(centres)), key=lambda i: abs(centres[i] - (x0 + x1) / 2))
                hit = [near] if abs(centres[near] - (x0 + x1) / 2) < 0.7 * pitch else []
            for i in hit:
                names[i].append(text)
    for c, parts in zip(t.columns, names):
        if parts:
            c.group = " ".join(parts)
            c.header = f"{c.group} | {c.header}" if c.header else c.group


def _repair_wrapped_labels(t: Table) -> None:
    """build_table reads "Amortization of prior service cost" / "(credit)  1  1"
    as a section heading and a row labelled "(credit)": a label cannot start
    with a bracket or a lower-case word, so the heading is the first half of
    it. Also drops a footnote digit glued to a label ("assets1", "(gain)2")."""
    prev_section, undo = "", {}           # wrongly made section -> the section it hid
    for r in t.rows:
        r.label = re.sub(r"(?<=[a-z)])\d$", "", r.label)
        if r.section in undo:
            r.section = undo[r.section]   # later rows were filed under the label half too
        elif r.section and r.section != prev_section and re.match(r"^[(a-z]", r.label):
            r.label = f"{r.section} {r.label}"
            undo[r.section] = prev_section
            r.section = prev_section
        prev_section = r.section


def _spread_panel_year(t: Table) -> None:
    """One date printed once over category columns ("December 31, 2025" over
    "Level 1  Level 2  Level 3  Total") lands on a single column; it belongs
    to all of them."""
    dated = [c for c in t.columns if c.year]
    if t.ncols < 2 or len(dated) != 1 or len(re.findall(r"\b(?:19|20)\d\d\b", t.header_text)) != 1:
        return
    src = dated[0]
    for c in t.columns:
        c.year, c.end_date = src.year, src.end_date
        c.months = c.months or src.months


def _phrases(words: list) -> list:
    out, cur = [], [words[0]]
    for a, b in zip(words, words[1:]):
        if b.x0 - a.x1 > 9:
            out.append(cur)
            cur = [b]
        else:
            cur.append(b)
    out.append(cur)
    return out


def _fallback_title(t: Table, region: list, start: int, prev_title: str, earlier: list) -> str:
    """No heading or sentence above the table.

    Rule 1: a statement-style title the region pulled in with its header
            ("Consolidated Statements of Shareholders' Equity", centred).
    Rule 2: a table at the top of a page continues the last table of the
            previous page: "<that title> (continued)".
    Rule 3: a table right under another one is named by its own first section
            label ("Long-lived assets:"), else by the table above it.
    """
    for ln in region:
        if ln.kind in ("hdr", "num"):
            break
        if TITLE_RE.search(ln.text) and len(ln.text) < 110 and not ln.text.rstrip().endswith("."):
            return _clean_title(ln.text)
    if start <= 3 and prev_title:
        return prev_title if prev_title.endswith("(continued)") else prev_title + " (continued)"
    if t.rows and t.rows[0].section:
        return _clean_title(t.rows[0].section)
    if earlier:
        return earlier[-1].title
    return ""


def _looks_like_index(t: Table) -> bool:
    """A table of contents or exhibit index: the last column is small rising
    whole numbers (page numbers) and no column carries a period."""
    if any(c.year or c.end_date for c in t.columns):
        return False
    last = t.ncols - 1
    vals = [r.cells[last].value for r in t.rows if last in r.cells and r.cells[last].value is not None]
    if t.ncols > 1 and len(vals) < 4:
        return False
    return len(vals) >= 2 and all(v == int(v) and 0 < v < 1000 for v in vals) and vals == sorted(vals)


def _acceptable(t: Optional[Table]) -> bool:
    """Precision over recall: drop anything that does not look like a table."""
    if t is None or not t.columns:
        return False
    numeric_rows = [r for r in t.rows if any(c.value is not None for c in r.cells.values())]
    if len(numeric_rows) < 2:
        return False
    labelled = [r for r in numeric_rows if r.label]
    if len(labelled) < 0.5 * len(numeric_rows):
        return False                     # numbers with no labels: labels went to the wrong place
    if sum(len(r.label) for r in labelled) / len(labelled) > 110:
        return False                     # prose with numbers in it
    if _looks_like_index(t):
        return False
    return True


# ---------------------------------------------------------------- contexts

class _Context:
    """The Item and Note headings in force at each point of the document."""

    def __init__(self):
        self.item = ""
        self.note = ""

    def feed(self, ln: Line) -> None:
        t = ln.text.strip()
        if len(t) > 110 or t.endswith(".") and not re.search(r"\d\.\s*$", t[:12]):
            return
        if ITEM_RE.match(t) and not re.search(r"\s\d{1,3}$", t):     # not a contents line ending in a page number
            self.item, self.note = t, ""
        elif NOTE_HEAD_RE.match(t) and ln.kind not in ("num", "hdr") and not re.search(r"\s\d{1,3}$", t):
            self.note = t

    @property
    def text(self) -> str:
        return " / ".join(x for x in (self.item, self.note) if x)


# ----------------------------------------------------------------------- API

# ------------------------------------------------------- two-column pages

def _gutter(lines: list, page_w: float) -> Optional[float]:
    """The x in the middle band of the page crossed by the fewest lines (the
    centre of the widest such run), or None if every x is crossed by most lines."""
    lo, hi = int(0.3 * page_w), int(0.7 * page_w)
    hits = [0] * (hi - lo + 2)
    for ln in lines:                                   # how many lines cross each x
        covered = set()
        for w in ln:
            a, b = max(lo, int(w.x0) + 1), min(hi, int(w.x1 - 1e-9))
            covered.update(range(a, b + 1))
        for x in covered:
            hits[x - lo] += 1
    counts = [(lo + i, hits[i]) for i in range(hi - lo)]
    best, run = None, []
    low = min(c for _, c in counts)
    if low > 0.4 * len(lines):
        return None
    for x, c in counts:
        if c == low:
            run.append(x)
            if best is None or len(run) > len(best):
                best = list(run)
        else:
            run = []
    return (best[0] + best[-1]) / 2


def _alpha_first(parts: list) -> int:
    return sum(1 for p in parts if p and p[0].text[:1].isalpha() and p[0].text not in VALUE_WORDS)


def _two_columns(zone: list, g: float) -> bool:
    """Both halves carry their own left-aligned text: prose in two columns, or
    two tables side by side, each with its own label column. A one-column
    table also has an empty strip between labels and values, but the right
    side of it holds numbers, not text."""
    left = [[w for w in ln if w.x1 <= g] for ln in zone]
    right = [[w for w in ln if w.x0 >= g] for ln in zone]
    left, right = [p for p in left if p], [p for p in right if p]
    if len(left) < 3 or len(right) < 3:
        return False
    return _alpha_first(left) >= 3 and _alpha_first(right) >= 3 and _alpha_first(right) >= 0.4 * len(right)


def _runs(items: list, flags: list) -> list:
    """[(flag, [items])] for consecutive items with the same flag."""
    out: list = []
    for it, f in zip(items, flags):
        if out and out[-1][0] == f:
            out[-1][1].append(it)
        else:
            out.append((f, [it]))
    return out


def _is_value(text: str) -> bool:
    return is_numeric(text) or text in DASH_TOKENS or text in VALUE_WORDS


def _cut_spanning_tables(zone: list, g: float) -> list:
    """A full-width table whose columns happen to leave the gutter empty
    (labels on the left half, values running on into the right half) sits
    inside a two-column zone. Its rows end the left half with a value and
    start the right half with one ("... 2025 2024 2023 | 2025 2024 2023",
    "... 15,118 | $ 53,766"). Those rows, the short rows between them, and the
    header lines above them (group headers across both halves, period words,
    a "(Table continued ...)" note and one heading) are cut out as full width."""
    def halves(ln):
        return [w for w in ln if w.x1 <= g], [w for w in ln if w.x0 >= g]

    flags = []
    for ln in zone:
        left, right = halves(ln)
        flags.append(bool(left and right and _is_value(left[-1].text) and _is_value(right[0].text)))
    hits = [i for i, f in enumerate(flags) if f]
    if len(hits) < 2:
        return [(True, zone)]
    for a, b in zip(hits, hits[1:]):
        if b - a <= 4:                                 # wrapped labels, "(e)" markers between rows
            for k in range(a, b):
                flags[k] = True
    for i in range(len(zone)):
        if flags[i] and (i == 0 or not flags[i - 1]):
            j, heading_taken = i, False
            while j > 0 and not flags[j - 1] and i - j < 6 and not _gap_too_big(zone[j - 1], zone[j]):
                left, right = halves(zone[j - 1])
                text = " ".join(w.text for w in zone[j - 1])
                headerish = (left and right) or HEADER_HINT.search(text) or CAPTION_RE.search(text) \
                    or CONTINUED_RE.search(text)
                if not headerish:
                    sentence = len(zone[j - 1]) > 4 and text.rstrip().endswith((".", ",", ":"))
                    if heading_taken or len(zone[j - 1]) > 10 or sentence:
                        break
                    heading_taken = True               # the table's own heading line
                j -= 1
                flags[j] = True
    return [(not f, part) for f, part in _runs(zone, flags)]


def _gap_too_big(a: list, b: list) -> bool:
    ha = max(w.y1 - w.y0 for w in a)
    return min(w.y0 for w in b) - max(w.y1 for w in a) > 2.6 * ha


def _streams(lines: list, page_w: float) -> list:
    """Reading-order streams of lines. A one-column page is one stream. A page
    printed in two columns (an annual-report style 10-K) becomes left column
    then right column, zone by zone; full-width parts (a line with a word
    across the gutter) stay whole."""
    g = _gutter(lines, page_w) if len(lines) >= 6 else None
    if g is None:
        return [lines]
    zones = _runs(lines, [not any(w.x0 < g < w.x1 for w in ln) for ln in lines])
    split: list = []
    for two, zone in zones:
        split.extend(_cut_spanning_tables(zone, g) if two else [(False, zone)])
    out, full = [], []
    for two, zone in split:
        if two and len(zone) >= 4 and _two_columns(zone, g):
            if full:
                out.append(full)
                full = []
            for side in ([w for w in ln if w.x1 <= g] for ln in zone), ([w for w in ln if w.x0 >= g] for ln in zone):
                stream = [p for p in side if p]
                if stream:
                    out.append(stream)
        else:
            full.extend(zone)
    if full:
        out.append(full)
    return out


def page_tables(doc_id: str, page_no: int, page: dict, default_scale: float = 1.0,
                prev_title: str = "", ctx: Optional[_Context] = None) -> list:
    """Every table on one page (page_no is 1-based)."""
    raw = page_lines(page)
    if not raw:
        return []
    out: list = []
    for stream in _streams(raw, page["w"]):
        out.extend(_stream_tables(doc_id, page_no, stream, default_scale,
                                  prev_title if not out else "", ctx, out))
    return out


def _demote_superscripts(lines: list) -> None:
    """Footnote references set as superscripts sometimes land on their own
    line ("4   5   6" between two rows). Such a line has no label, only single
    digits, and smaller type than the page; it is a marker, not a row."""
    hs = sorted(ln.height for ln in lines)
    body = hs[len(hs) // 2]
    for ln in lines:
        if ln.kind == "num" and all(re.fullmatch(r"\d", w.text) for w in ln.words) \
                and ln.height < 0.85 * body:
            ln.kind = "mark"


def _stream_tables(doc_id: str, page_no: int, raw: list, default_scale: float,
                   prev_title: str, ctx: Optional[_Context], earlier: list) -> list:
    """Segment one stream of lines into tables and build them."""
    pct: set = set()
    prepared = []
    for ln in raw:
        ln = _strip_percent_signs(_bind_label_numbers(ln), pct)
        if ln:
            prepared.append(ln)
    if not prepared:
        return []
    left = min(ln[0].x0 for ln in prepared)
    right = max(ln[-1].x1 for ln in prepared)
    lines = [_classify(ln, left, right) for ln in prepared]
    _demote_superscripts(lines)
    out, floor = [], -1
    regions = _segment(lines)
    for reg in regions:
        start = _extend_up(lines, reg.start, floor)
        region = lines[start:reg.end + 1]
        title, intro = _title(lines, start, floor)
        context = ""
        if ctx is not None:
            # headings above the table set its context; a heading caught
            # inside the region (between two tables) still moves the context on
            for ln in lines[floor + 1:start]:
                ctx.feed(ln)
            context = ctx.text
            for ln in region:
                if ln.kind == "label":
                    ctx.feed(ln)
        floor = reg.end
        t = _build(doc_id, page_no, region, title, pct)
        if t is None:
            continue
        if not title:
            t.title = _fallback_title(t, region, start, prev_title, earlier + out)
        _clean_labels(t)
        if not _acceptable(t):
            continue
        _repair_wrapped_labels(t)
        _group_headers(t, region)
        _text_headers(t, region)
        _spread_panel_year(t)
        _apply_scale(t, intro, default_scale)
        t.intro = intro
        t.context = context
        t.bbox_y = (region[0].y0, region[-1].y1)
        out.append(t)
    if ctx is not None:
        for ln in lines[floor + 1:]:
            ctx.feed(ln)
    return out


def find_all_tables(doc_id: str, pages: list, skip_pages: set | None = None,
                    default_scale: float | None = None) -> list:
    """Every numeric table in a filing, page by page, in print order.

    skip_pages: 1-based page numbers to leave out (the primary statements).
    default_scale: scale for tables that print no unit and have no unit in
    the sentence introducing them. None means infer_default_scale(pages).

    Each Table gets extra attributes: context (Item / Note heading in force),
    intro (the introducing sentence, if any), scale_source (caption | intro |
    inherited), pct_columns (dropped percent-change columns), bbox_y.
    """
    skip = skip_pages or set()
    if default_scale is None:
        default_scale = infer_default_scale(pages)
    ctx = _Context()
    out, prev_title = [], ""
    for i, page in enumerate(pages):
        page_no = i + 1
        if page_no in skip:
            # still read the headings so the context stays right
            for ln in page_lines(page):
                ctx.feed(Line(ln, "label", " ".join(w.text for w in ln)))
            prev_title = ""
            continue
        tables = page_tables(doc_id, page_no, page, default_scale, prev_title, ctx)
        out.extend(tables)
        prev_title = tables[-1].title if tables else ""
    return out
