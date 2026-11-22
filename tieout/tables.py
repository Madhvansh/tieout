"""Rebuild tables from PDF word coordinates.

A 10-K statement in a text PDF has no ruling lines. What it does have is
right-aligned numbers: every value in a period column ends at the same x.
So columns are found by clustering the right edges of numeric tokens, rows by
clustering word baselines, and every cell keeps its page and bounding box.

Nothing here knows any accounting. It returns what is printed and where.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Optional


# "28," (a day inside a printed date) must not count as a value, so a comma is
# only allowed as a thousands separator.
NUM_RE = re.compile(r"^\(?-?\$?\(?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?\)?$")
DASH_TOKENS = {"—", "–", "-", "‒", "―", "——"}
YEAR_RE = re.compile(r"^(19|20)\d\d$")
FOOTNOTE_RE = re.compile(r"^\(?\d{1,2}\)$|^\([a-z]\)$")
TITLE_RE = re.compile(r"CONSOLIDATED|STATEMENTS? OF|BALANCE SHEETS?|FINANCIAL CONDITION", re.I)
CAPTION_RE = re.compile(r"\b(in|\$)\s+(millions|thousands|billions)\b|^\(?\s*(millions|thousands)\b|"
                        r"except per[- ]share|unaudited", re.I)
HEADER_HINT = re.compile(
    r"\b(19|20)\d\d\b|\bended\b|\bas of\b|\bweeks\b|\bfiscal\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", re.I)
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
    )
}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
LABEL_MARK = "⁠"     # a word joiner glued to a number that belongs to a label (alltables), not a printed character
# a line inside a table that holds only a date starts a block of rows for that date
# ("December 2024" between two stacked blocks of the same columns)
PERIOD_LINE_RE = re.compile(
    r"^(?:(?:as of|as at|at|years? ended|fiscal years? ended|(?:three|six|nine|twelve) months ended|quarter ended)\s+)?"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?)?(?:19|20)\d\d:?$", re.I)
# the same with a full date, at the start of a line that goes on with column names
PERIOD_HEAD_RE = re.compile(
    r"^(?:(?:as of|as at|at|years? ended|fiscal years? ended|(?:three|six|nine|twelve) months ended|quarter ended)\s+)?"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d\d\b", re.I)


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Cell:
    page: int
    bbox: tuple
    text: str
    value: Optional[float]      # as printed, before scale
    col: int
    is_percent: bool = False
    has_dollar: bool = False


@dataclass
class Row:
    idx: int
    page: int
    y: float
    label: str
    indent: float
    label_bbox: Optional[tuple]
    cells: dict = field(default_factory=dict)   # col index -> Cell
    section: str = ""                           # nearest label-only row above
    period_mark: str = ""                       # nearest date-only line above inside the table ("December 2024")

    def values(self, ncols: int) -> list:
        return [self.cells[c].value if c in self.cells else None for c in range(ncols)]


@dataclass
class Column:
    idx: int
    x_right: float
    x_left: float
    header: str = ""
    year: Optional[int] = None
    end_date: Optional[str] = None   # ISO date if a full date was printed
    months: Optional[int] = None     # duration length; None means unknown


@dataclass
class Table:
    doc: str
    page: int                 # 1-based page of the first row
    title: str
    caption: str
    scale: float              # multiplier for currency amounts
    share_scale: float        # multiplier for share counts
    columns: list
    rows: list
    header_text: str = ""

    @property
    def ncols(self) -> int:
        return len(self.columns)


# ----------------------------------------------------------------- words/lines

def group_lines(words: list, tol: float = 2.5) -> list:
    """Cluster words into text lines by vertical centre."""
    lines: list = []
    for w in sorted(words, key=lambda w: (w.yc, w.x0)):
        if lines and abs(lines[-1][0].yc - w.yc) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    # Some filings set the numbers a few points above or below the label on the
    # same row (a different font or baseline). Two neighbouring "lines" closer
    # than half a line pitch whose words do not overlap sideways are one row.
    merged: list = []
    for ln in lines:
        if merged:
            prev = merged[-1]
            dy = abs(prev[0].yc - ln[0].yc)
            apart = max(w.x1 for w in prev) < min(w.x0 for w in ln) or max(w.x1 for w in ln) < min(w.x0 for w in prev)
            if dy < 5.0 and apart:
                prev.extend(ln)
                continue
        merged.append(list(ln))
    for ln in merged:
        ln.sort(key=lambda w: w.x0)
    return merged


def is_numeric(tok: str) -> bool:
    return bool(NUM_RE.match(tok)) and any(ch.isdigit() for ch in tok)


def right_edge(w: Word) -> float:
    """Right edge of the last digit: a closing parenthesis or % hangs past it."""
    t = w.text
    n_hang = len(t) - len(t.rstrip(")%"))
    if n_hang and len(t) > n_hang:
        # parentheses and percent signs are narrower than digits in most fonts
        return w.x1 - n_hang * (w.x1 - w.x0) / len(t) * 0.62
    return w.x1


def parse_number(tok: str) -> Optional[float]:
    t = tok.replace("$", "").replace(",", "").replace("%", "")
    neg = t.startswith("(") or t.endswith(")") or t.startswith("-")
    t = t.strip("()-")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


# --------------------------------------------------------------------- columns

def find_columns(lines: list, tol: float = 3.5, min_hits: int = 3) -> list:
    """Cluster the right edges of numeric tokens into value columns."""
    edges = []
    for ln in lines:
        for w in ln:
            if is_numeric(w.text) and not (YEAR_RE.match(w.text) and len(ln) < 12 and _mostly_years(ln)):
                edges.append((right_edge(w), w.x0))
            elif w.text in DASH_TOKENS:
                edges.append((w.x1, w.x0))
    edges.sort()
    clusters: list = []
    for xr, x0 in edges:
        if clusters and xr - clusters[-1]["hi"] <= tol:
            c = clusters[-1]
            c["hi"] = xr
            c["xs"].append(xr)
            c["x0s"].append(x0)
        else:
            clusters.append({"hi": xr, "xs": [xr], "x0s": [x0]})
    # Numbers inside long labels ("100,000 shares authorized; 10,731 issued")
    # sometimes line up three times by chance. A real value column is hit by a
    # large share of the rows, so thin clusters are dropped.
    most = max((len(c["xs"]) for c in clusters), default=0)
    need = max(min_hits, 0.25 * most)
    cols = []
    for c in clusters:
        if len(c["xs"]) >= need:
            xs = sorted(c["xs"])
            cols.append((xs[len(xs) // 2], min(c["x0s"])))
    return [Column(i, xr, xl) for i, (xr, xl) in enumerate(cols)]


def _mostly_years(ln: list) -> bool:
    """True for a header line of years: "2025 2024 2023", also when a date
    such as "December 31" or "June 30" is printed on the same line."""
    years, days, other, stub = 0, 0, 0, True
    for i, w in enumerate(ln):
        if not is_numeric(w.text):
            continue
        if YEAR_RE.match(w.text):
            years += 1
        elif i and ln[i - 1].text.lower().rstrip(".")[:3] in MONTH_ABBR and w.text.isdigit() and int(w.text) <= 31:
            days += 1                              # the day of a printed date: "Year ended December 31"
        else:
            other += 1
            stub = stub and len(w.text) <= 2
    if (years > 0 or days > 0) and other == 0:
        return True
    # a year header whose last year is cut off at the page edge: "2025  2024  20"
    return years >= 2 and other == 1 and stub


def _note_columns(lines: list, cols: list) -> set:
    """Indexes of columns that hold only note references: whole numbers below 100, never a year."""
    seen: dict = {}
    for ln in lines:
        for w in ln:
            if is_numeric(w.text) and not YEAR_RE.match(w.text):
                ci = _col_of(w, cols)
                if ci is not None:
                    seen.setdefault(ci, []).append(w.text)
    out = {ci for ci, texts in seen.items() if len(texts) >= 3 and all(re.fullmatch(r"\d{1,2}", t) for t in texts)}
    return out if len(out) < len(cols) else set()


def _col_of(w: Word, cols: list, tol: float = 4.5) -> Optional[int]:
    xr = right_edge(w) if is_numeric(w.text) else w.x1
    best, bd = None, tol
    for c in cols:
        d = abs(xr - c.x_right)
        if d <= bd:
            best, bd = c.idx, d
    return best


# ---------------------------------------------------------------------- header

DATE_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?!\d),?"
                     r"(?:\s+((?:19|20)\d\d))?", re.I)
MONTH_YEAR_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+((?:19|20)\d\d)\b", re.I)
DAY_END_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?$", re.I)   # "... December 31,"
DURATIONS = ((r"three months|quarter ended|13 weeks|12 weeks|thirteen weeks|16 weeks", 3),
             (r"six months|26 weeks|twenty-six weeks|24 weeks|28 weeks", 6),
             (r"nine months|39 weeks|thirty-nine weeks|36 weeks|40 weeks", 9),
             (r"years? ended|fiscal years?|twelve months|52 weeks|53 weeks|for the years?", 12))


def duration(text: str) -> Optional[int]:
    """Months of the period a header phrase names ("Nine Months Ended" is 9), or None."""
    for pat, months in DURATIONS:
        if re.search(pat, text, re.I):
            return months
    return None


def _part_year(text: str) -> Optional[int]:
    """The one period shorter than a year that a title names ("for the fiscal nine months" is 9);
    None when it names none or several ("the three and nine months ended")."""
    named = {m for pat, m in DURATIONS if m < 12 and re.search(pat, text, re.I)}
    if len(named) != 1 or re.search(r"\b(?:three|six|nine)\s+and\s+(?:three|six|nine)\b", text, re.I):
        return None
    return named.pop()


def printed_date(text: str, year: Optional[int] = None, month_only: bool = True) -> Optional[str]:
    """ISO date of the first date printed in text, with `year` when the text prints none.
    "December 2025" (a month without a day) is the last day of that month."""
    dm = DATE_RE.search(text)
    if dm:
        yr = year or (int(dm.group(3)) if dm.group(3) else None)
        if not yr:
            return None
        mon = MONTH_ABBR[dm.group(1)[:3].lower()]
        return f"{yr:04d}-{mon:02d}-{min(int(dm.group(2)), calendar.monthrange(yr, mon)[1]):02d}"
    my = MONTH_YEAR_RE.search(text) if month_only else None
    if my:
        yr, mon = year or int(my.group(2)), MONTH_ABBR[my.group(1)[:3].lower()]
        return f"{yr:04d}-{mon:02d}-{calendar.monthrange(yr, mon)[1]:02d}"
    return None


def _phrases_of(ln: list) -> list:
    """A header line cut into phrases at wide gaps. A phrase that repeats one dated piece
    ("FISCAL 2026 FISCAL 2025", "2025 2024", "December 31, 2024 December 31, 2023") is one
    phrase per piece, each ending at its year."""
    phrases, cur = [], [ln[0]] if ln else []
    for a, b in zip(ln, ln[1:]):
        if b.x0 - a.x1 > 9:
            phrases.append(cur)
            cur = [b]
        else:
            cur.append(b)
    if cur:
        phrases.append(cur)
    out = []
    for ph in phrases:
        ends = [k + 1 for k, w in enumerate(ph) if YEAR_RE.match(w.text.replace(LABEL_MARK, ""))]
        pieces = [ph[a:b] for a, b in zip([0] + ends, ends)]
        shapes = {tuple(re.sub(r"\d", "#", w.text.replace(LABEL_MARK, "").lower()) for w in p) for p in pieces}
        if len(ends) < 2 or ends[-1] != len(ph) or len(shapes) != 1:
            out.append(ph)
        else:
            out.extend(pieces)
    return out


def _parse_header(header_lines: list, cols: list, context: str = "", title: str = "") -> str:
    """Attach printed period information to each column.

    A header word belongs to a column if its centre falls inside the column's
    horizontal band. Words that span several bands ("Years ended") apply to
    every column they cover. A date or a year sits over one column, the
    nearest. The length of a column's period comes from its own words, else
    from the nearest phrase that names a length ("Three Months Ended" over the
    first pair of columns, "Nine Months Ended" over the second), else from the
    caption, else from a title that names one length shorter than a year.
    """
    if not cols:
        return ""
    bands = []
    for i, c in enumerate(cols):
        left = cols[i - 1].x_right + 2 if i else min(c.x_left, c.x_right - 70) - 25
        bands.append((left, c.x_right + 14))
    full = " ".join(w.text for ln in header_lines for w in ln)
    per_col = [[] for _ in cols]
    lengths = []                                   # (x0, x1, months) of phrases that name a length
    for ln in header_lines:
        for ph in _phrases_of(ln):
            x0, x1 = ph[0].x0, ph[-1].x1
            text = " ".join(w.text for w in ph)
            centre = (x0 + x1) / 2
            if duration(text):
                lengths.append((x0, x1, duration(text)))
            dated = bool(re.search(r"(19|20)\d\d|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text, re.I))
            if dated and x1 - x0 < 110:
                # a date or a year sits over exactly one column: the nearest one
                # Headers are right-aligned with the numbers or centred over them, so
                # compare right edges and centres with the column's own values.
                def gap(i):
                    return min(abs(x1 - cols[i].x_right), abs(centre - (cols[i].x_right - 14)))
                near = min(range(len(cols)), key=gap)
                if gap(near) > 30 and not (bands[near][0] <= centre <= bands[near][1]):
                    continue        # a year over a column that is not there (cut off at the page edge)
                per_col[near].append(text)
            else:
                hit = [i for i, (l, r) in enumerate(bands) if not (x1 < l or x0 > r)]
                for i in hit:
                    per_col[i].append(text)
    wide = full + " || " + context
    for c, parts in zip(cols, per_col):
        c.header = " ".join(parts).strip()
        ym = re.findall(r"\b((?:19|20)\d\d)\b", c.header)
        if ym:
            c.year = int(ym[-1])
    several_years = len({c.year for c in cols if c.year}) > 1
    for c in cols:
        # "December 31," printed once above "2025  2024" belongs to every column. A column without
        # a year of its own beside columns of different years ("Thereafter", "Total"), or with only
        # part of a date of its own ("31,"), is given none.
        shared = c.year or not (several_years or re.search(r"\d", c.header))
        c.end_date = printed_date(c.header, c.year) or (
            printed_date(full, c.year) or printed_date(wide, c.year, month_only=False) if shared else None)
        mid = (c.x_left + c.x_right) / 2
        nearest = min(lengths, key=lambda s: max(s[0] - mid, mid - s[1], 0), default=None)
        c.months = duration(c.header) or (nearest[2] if nearest else None) or duration(wide) or _part_year(title)
    return full


# ----------------------------------------------------------------------- table

SCALE_RE = re.compile(r"\(?\s*(?:dollars |amounts |\$ )?in (millions|thousands|billions)", re.I)


def _scale(caption: str) -> tuple:
    """Currency scale and share-count scale from the printed caption.

    "(In millions, except number of shares, which are reflected in thousands,
    and per-share amounts)" gives 1e6 for money and 1e3 for shares.
    """
    cap = caption.lower()
    unit = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
    scale = 1.0
    share_scale = None
    for m in re.finditer(r"(thousands|millions|billions)", cap):
        before = cap[max(0, m.start() - 60):m.start()]
        if "share" in before and scale != 1.0:
            share_scale = unit[m.group(1)]       # the second scale word, said about shares
            continue
        if scale == 1.0:
            scale = unit[m.group(1)]
    if share_scale is None:
        if re.search(r"except[^)]{0,40}share (data|amounts|counts)|except[^)]{0,30}shares\b", cap) and "per share" not in cap:
            share_scale = 1.0
        else:
            share_scale = scale
    return scale, share_scale


def build_table(doc_name: str, page_no: int, lines: list, title: str = "", min_hits: int = 3) -> Optional[Table]:
    """Turn the text lines of one table region into a Table. `min_hits`: values a column needs (a
    one-line table under a year header has one value per column)."""
    cols = find_columns(lines, min_hits=min_hits)
    if not cols:
        return None
    label_limit = min(c.x_left for c in cols) - 4

    def cells_of(ln):
        found = {}
        for w in ln:
            if w.text in DASH_TOKENS and w.x0 > label_limit - 30:
                ci = _col_of(w, cols, tol=9)
                if ci is not None:
                    found[ci] = (w, 0.0)
            elif is_numeric(w.text):
                ci = _col_of(w, cols)
                if ci is not None and w.x0 >= cols[ci].x_left - 45:
                    found[ci] = (w, parse_number(w.text))
        return found

    # A "Note" column holds reference numbers (1, 12, 7), not values. Drop it:
    # its tokens would otherwise shift the period headers onto the wrong columns.
    note_cols = _note_columns(lines, cols)
    if note_cols:
        note_words = {id(w) for ln in lines for w in ln
                      if is_numeric(w.text) and _col_of(w, cols) in note_cols}
        lines = [[w for w in ln if id(w) not in note_words] for ln in lines]
        lines = [ln for ln in lines if ln]
        cols = [Column(i, c.x_right, c.x_left) for i, c in enumerate(c for c in cols if c.idx not in note_cols)]
        label_limit = min(c.x_left for c in cols) - 4

    first_num = next((i for i, ln in enumerate(lines)
                      if cells_of(ln) and not _mostly_years(ln)), None)
    if first_num is None:
        return None
    # Walk up from the first numeric row. Lines that carry period words and reach
    # into the column area are header; short left-aligned lines in between are
    # section labels ("Net sales:"); anything else ends the header (title, caption).
    header_idx: list = []
    i = first_num - 1
    while i >= 0 and first_num - i <= 14:
        ln = lines[i]
        text = " ".join(w.text for w in ln)
        in_cols = [w for w in ln if w.x1 > label_limit - 20]
        if in_cols and (HEADER_HINT.search(" ".join(w.text for w in in_cols)) or all(is_month(w.text) for w in in_cols)):
            # the scale caption often shares a line with the years:
            # "(Amounts in millions, except per share data)  2026  2025  2024"
            header_idx.append(i)
        elif TITLE_RE.search(text) and len(text) < 130:
            break                                   # statement title: nothing above belongs to the table
        elif CAPTION_RE.search(text):
            pass                                    # "(In millions, except ...)" above or below the header
        elif header_idx:
            break                                   # something else above the header
        # else: a label between the header and the first numeric row
        # ("ASSETS:", "Net sales:", first half of a wrapped label); keep climbing
        i -= 1
    header_idx.sort()
    # only the words over the value columns are header; a caption on the same line is not
    header_lines = [[w for w in lines[j] if w.x1 > label_limit - 20] for j in header_idx]
    body_start = header_idx[-1] + 1 if header_idx else first_num
    caption = " ".join(w.text for ln in lines[max(0, i):first_num] for w in ln)
    header_text = _parse_header(header_lines, cols, context=caption, title=title)
    scale, share_scale = _scale(caption + " " + header_text)

    rows: list = []
    section = ""
    # a date-only line above the first row ("2025" over a block of segment columns) dates the rows below it
    period_mark = next((_plain(ln).rstrip(":") for ln in reversed(lines[max(0, i):first_num])
                        if PERIOD_LINE_RE.match(_plain(ln))), "")
    pending_label: list = []          # label-only line that may continue below
    for ln in lines[body_start:]:
        found = cells_of(ln)
        # a block header inside the table: "As of December 2024", or a date before the column
        # names repeated for the next block ("July 26, 2025  Loan Receivables  Total")
        head = PERIOD_HEAD_RE.match(_plain(ln)) if rows else None
        if not found and (PERIOD_LINE_RE.match(_plain(ln)) or head):
            if YEAR_RE.match(_plain(ln)) and rows and not pending_label and DAY_END_RE.search(rows[-1].label):
                rows[-1].label += " " + _plain(ln)      # "Balance as of December 31," / "2022": the end of a label
                continue
            section = " ".join(w.text for w in pending_label) if pending_label else ""
            pending_label = []
            period_mark = head.group(0) if head else _plain(ln).rstrip(":")
            continue
        label_words = [w for w in ln if w.x1 <= label_limit + 2 or
                       (not is_numeric(w.text) and w.text not in DASH_TOKENS and w.text != "$"
                        and _col_of(w, cols) is None and w.x0 < label_limit + 40)]
        label_words = [w for w in label_words if w.text != "$"]
        # footnote markers glued after the label, e.g. "(1)"
        while label_words and FOOTNOTE_RE.match(label_words[-1].text) and len(label_words) > 1:
            label_words.pop()
        label = " ".join(w.text for w in label_words).strip()
        if not found:
            # A line with a label and no numbers is either a section heading
            # ("Current assets") or the first half of a wrapped label. Which one
            # is only known when the next numeric row arrives.
            if label and not CAPTION_RE.search(label):
                if pending_label:
                    ptext = " ".join(w.text for w in pending_label)
                    if _continues(ptext, label):
                        pending_label = pending_label + label_words     # a label over three lines
                        continue
                    section = ptext
                if label.endswith(":") and label.count("(") <= label.count(")"):
                    section, pending_label = label.rstrip(":"), []
                else:
                    pending_label = label_words
            continue
        if pending_label:
            ptext = " ".join(w.text for w in pending_label)
            continues = _continues(ptext, label)
            if label_words and continues and label_words[0].y0 - pending_label[-1].y1 < 8:
                label = ptext + " " + label
                label_words = pending_label + label_words
            elif not label:
                label = " ".join(w.text for w in pending_label)
                label_words = pending_label
            else:
                section = " ".join(w.text for w in pending_label)
        pending_label = []
        r = Row(idx=len(rows), page=page_no, y=ln[0].yc, label=label,
                indent=label_words[0].x0 if label_words else 0.0,
                label_bbox=_bbox(label_words) if label_words else None, section=section,
                period_mark=period_mark)
        dollar_x = [w.x0 for w in ln if w.text == "$"]
        for ci, (w, v) in found.items():
            r.cells[ci] = Cell(page=page_no, bbox=(w.x0, w.y0, w.x1, w.y1), text=w.text,
                               value=v, col=ci, is_percent=w.text.rstrip(")").endswith("%"),
                               has_dollar=any(0 < w.x0 - dx < 60 for dx in dollar_x))
        rows.append(r)
    if len(rows) < (1 if min_hits == 1 else 2):
        return None
    return Table(doc=doc_name, page=page_no, title=title, caption=caption, scale=scale,
                 share_scale=share_scale, columns=cols, rows=rows, header_text=header_text)


def _plain(ln: list) -> str:
    """The printed text of a line."""
    return " ".join(w.text for w in ln).replace(LABEL_MARK, "").strip()


def is_month(text: str) -> bool:
    """"September", "Sept.", "Dec": a month name with no day ("September" printed over each
    column, the day and year on the line below)."""
    t = text.lower().rstrip(".,")
    return t in MONTHS or t in MONTH_ABBR or t == "sept"


JOIN_TAIL = re.compile(r"(,|;|\band|\bof|\bthe|\bto|\bfor|\bin|\bat|\bor|\bby|\bon)$")


def _continues(above: str, below: str) -> bool:
    """Does the label on the line above run on into the line below?

    Yes if it left a bracket open, ended on a joining word, or the next line
    starts in lower case. "Current assets" above "Cash and cash equivalents"
    is a heading, not a wrapped label.
    """
    return (above.count("(") > above.count(")")
            or bool(JOIN_TAIL.search(above))
            or below[:1].islower())


def _bbox(words: list) -> tuple:
    return (min(w.x0 for w in words), min(w.y0 for w in words),
            max(w.x1 for w in words), max(w.y1 for w in words))


def show(t: Table) -> str:
    out = [f"[{t.title}] page {t.page} scale={t.scale:g} share_scale={t.share_scale:g}"]
    out.append("  cols: " + " | ".join(
        f"{c.idx}:{c.header!r} y={c.year} end={c.end_date} m={c.months}" for c in t.columns))
    for r in t.rows:
        vals = " ".join(f"{v:>14,.2f}" if v is not None else f"{'.':>14}" for v in r.values(t.ncols))
        out.append(f"  {r.idx:3d} {r.indent:5.0f} {r.label[:46]:46s} {vals}   <{r.section[:22]}>")
    return "\n".join(out)
