"""Non-numeric "text facts" from 10-K / 10-Q word boxes, by rule.

Every fact keeps the 1-based PDF page it was read from, the printed words that
support it and their bounding box. Nothing is guessed: a rule either sees the
printed pattern or the attribute is left out. No network, no model.

Two views of a page are used:
  * plain lines (tables.group_lines) for the cover page and signature tables,
    where position matters (labels printed UNDER their values, grids);
  * a reading-order flow for prose, which splits two-column pages at the
    gutter so sentences are not interleaved across columns.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .tables import Word, group_lines

# --------------------------------------------------------------------------- vocab

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}
ABBRS = set(STATES.values())
# Jurisdictions of incorporation outside the US that large filers use.
COUNTRIES = ["Ireland", "Bermuda", "Netherlands", "England and Wales", "United Kingdom",
             "Switzerland", "Cayman Islands", "Canada", "Luxembourg", "Jersey", "Puerto Rico"]
# Longest names first so "New Jersey" wins over "Jersey", "West Virginia" over "Virginia".
JURISDICTIONS = sorted(list(STATES) + COUNTRIES, key=len, reverse=True)
# Case-insensitive: MSFT prints "WASHINGTON" over "(STATE OF INCORPORATION)".
JURIS_RE = re.compile(r"\b(" + "|".join(re.escape(j) for j in JURISDICTIONS) + r")\b", re.I)
_CANON = {j.lower(): j for j in JURISDICTIONS}


def canon_jurisdiction(printed: str) -> str:
    return _CANON[re.sub(r"\s+", " ", printed).lower()]
STATE_TOKEN = r"(?:" + "|".join(re.escape(s) for s in sorted(STATES, key=len, reverse=True)) + \
              r"|" + "|".join(sorted(ABBRS)) + r"|D\.C\.)"

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
MONTH_RE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = MONTH_RE + r"\s+\d{1,2},?\s+\d{4}"
NUM_RE = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

FIRMS = [  # (printed pattern, normalised name)
    (r"Pricewaterhouse\s*Coopers", "PricewaterhouseCoopers"),
    (r"Ernst\s*&\s*Young", "Ernst & Young"),
    (r"Deloitte(?:\s*&\s*Touche)?", "Deloitte"),
    (r"KPMG", "KPMG"),
]
FIRM_RE = "|".join(p for p, _ in FIRMS)
PCAOB_IDS = {238: "PricewaterhouseCoopers", 42: "Ernst & Young", 34: "Deloitte", 185: "KPMG"}


@dataclass
class TextFact:
    """One fact read by rule.

    `share_class` names the class of stock a per-class fact is about ("Class A", "Class B-1",
    "Common"); it is None when the filing has a single class. When a filing prints several
    classes, `primary` marks the one that answers a question that names no class: the class
    traded under the company's first listed symbol. Every other fact is primary."""
    doc: str
    attr: str
    value: str
    page: int
    quote: str
    bbox: Optional[tuple]
    confidence: str
    share_class: Optional[str] = None
    primary: bool = True


# --------------------------------------------------------------------------- page text

@dataclass
class Flow:
    """A page as one string, with every character range mapped back to its Word."""
    page: int                       # 1-based PDF page
    lines: list                     # list of list[Word], reading order
    text: str = ""
    starts: list = field(default_factory=list)   # char start of each word
    words: list = field(default_factory=list)    # Word for each start
    line_of: list = field(default_factory=list)  # line index for each word
    line_start: list = field(default_factory=list)

    def __post_init__(self):
        parts, pos = [], 0
        for li, ln in enumerate(self.lines):
            self.line_start.append(pos)
            for wi, w in enumerate(ln):
                self.starts.append(pos)
                self.words.append(w)
                self.line_of.append(li)
                parts.append(w.text)
                pos += len(w.text)
                last = wi == len(ln) - 1
                # a line ending in a hyphen joins the next line ("full-" + "time")
                sep = "" if last and w.text.endswith("-") and len(w.text) > 1 else " "
                parts.append(sep)
                pos += len(sep)
        self.text = "".join(parts)

    def span_words(self, s: int, e: int) -> list:
        i = max(bisect.bisect_right(self.starts, s) - 1, 0)
        out = []
        while i < len(self.words) and self.starts[i] < e:
            if self.starts[i] + len(self.words[i].text) > s:
                out.append(self.words[i])
            i += 1
        return out

    def line_index(self, pos: int) -> int:
        return max(bisect.bisect_right(self.line_start, pos) - 1, 0)

    def line_text(self, li: int) -> str:
        return " ".join(w.text for w in self.lines[li])


def bbox_of(words: list) -> Optional[tuple]:
    if not words:
        return None
    return (round(min(w.x0 for w in words), 2), round(min(w.y0 for w in words), 2),
            round(max(w.x1 for w in words), 2), round(max(w.y1 for w in words), 2))


def raw_text(page: dict) -> str:
    return " ".join(w.text for w in page["words"])


def find_gutter(words: list, width: float) -> Optional[float]:
    """x of the gap between two prose columns, or None for a one-column page.

    Handles the two-column layout of bank 10-Ks and annual-report sections:
    at the gutter almost no word crosses, and on most lines alphabetic text
    runs up to the gutter from the left and starts again just after it."""
    if len(words) < 150:
        return None
    n_lines = len(group_lines(words))
    xs = [width * 0.38 + 2 * i for i in range(int(width * 0.24 / 2))]
    crossing = {x: sum(1 for w in words if w.x0 < x + 1 and w.x1 > x - 1) for x in xs}
    # the few least-crossed positions, nearest the centre first on ties
    cands = sorted((x for x in xs if crossing[x] <= 0.12 * n_lines),
                   key=lambda x: (crossing[x], abs(x - width / 2)))[:4]
    best, best_score = None, 0.0
    for x in cands:
        left = group_lines([w for w in words if w.x1 <= x])
        right = group_lines([w for w in words if w.x0 >= x])
        if len(left) < 8 or len(right) < 8:
            continue
        # columns are separate lines on each side (baselines need not align), so
        # measure how many left lines reach the gutter and right lines start at it
        reach = sum(1 for ln in left if x - ln[-1].x1 < 45 and ln[-1].text[:1].isalpha()) / len(left)
        start = sum(1 for ln in right if ln[0].x0 - x < 45 and ln[0].text[:1].isalpha()) / len(right)
        score = min(reach, start)
        if score > best_score:
            best, best_score = x, score
    return best if best_score >= 0.5 else None


def make_flow(page_no: int, page: dict, columns: bool = True) -> Flow:
    words = page["words"]
    gutter = find_gutter(words, page["w"]) if columns else None
    if gutter is None:
        return Flow(page_no, group_lines(words))
    left = [w for w in words if (w.x0 + w.x1) / 2 < gutter]
    right = [w for w in words if (w.x0 + w.x1) / 2 >= gutter]
    return Flow(page_no, group_lines(left) + group_lines(right))


# Sentence ends at . ! ? followed by space and a capital, unless the dot ends a
# common abbreviation ("Inc.", "U.S.", "No.", an initial "D.").
_BOUNDARY = re.compile(r"(?:[.!?][\"”’)]?|[☒☐☑])\s+(?=[A-Z0-9(•“\"])")
_ABBREV = re.compile(r"(?:\b(?:Inc|Corp|Co|Ltd|No|Nos|Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|U\.S|D\.C|N\.A|L\.P)|(?<![\w-])[A-Z])\.[\"”’)]?\s+$")


def sentence_bounds(text: str, s: int, e: int) -> tuple:
    start, end = 0, len(text)
    for m in _BOUNDARY.finditer(text):
        if _ABBREV.search(text[max(0, m.start() - 6):m.end()]):
            continue
        if m.end() <= s:
            start = m.end()
        elif m.start() >= e - 1:
            end = m.start() + 1
            break
    return start, end


def clip_quote(text: str, s: int, e: int, limit: int = 300) -> str:
    qs, qe = sentence_bounds(text, s, e)
    if qe - qs > limit:  # keep the match, trim the sentence around it at word edges
        room = max(limit - (e - s), 0)
        cut = max(qs, s - room // 2)
        if cut > qs:
            cut = text.find(" ", cut, s) + 1 or cut
        qs = cut
        qe = min(qe, qs + limit)
        if qe < len(text) and text[qe:qe + 1].strip():
            qe = max(text.rfind(" ", e, qe), e)
    q = text[qs:qe].strip()
    # a checkbox answer printed before the sentence ("☒ No The aggregate ...")
    return re.sub(r"^(?:Yes|No)\s+(?=[A-Z])", "", q)


def to_number(num: str, scale: Optional[str]) -> int:
    v = float(num.replace(",", ""))
    if scale:
        v *= SCALES[scale.lower()]
    return int(round(v))


def to_iso(month: str, day: str, year: str) -> str:
    return f"{int(year):04d}-{MONTHS.index(month.lower()) + 1:02d}-{int(day):02d}"


def title_name(name: str) -> str:
    """'JEN-HSUN HUANG' -> 'Jen-Hsun Huang'; mixed-case names stay as printed."""
    if name.upper() != name:
        return name
    return " ".join("-".join(p[:1] + p[1:].lower() for p in tok.split("-")) for tok in name.split())


class Collector:
    """The facts of one filing, in the order the rules find them."""

    def __init__(self, doc: str):
        self.doc = doc
        self.facts: list = []

    def add(self, attr, value, page, quote, words, confidence, share_class=None, bbox=None, primary=True):
        """Keep one fact; its box is `bbox` if given, else the box around `words`."""
        self.facts.append(TextFact(self.doc, attr, str(value), page, quote[:300],
                                   bbox or bbox_of(words), confidence, share_class, primary))

    def from_flow(self, attr, value, flow: Flow, s, e, confidence, quote=None, prefix="",
                  share_class=None, primary=True):
        q = quote if quote is not None else clip_quote(flow.text, s, e, 300 - len(prefix))
        self.add(attr, value, flow.page, prefix + q, flow.span_words(s, e), confidence,
                 share_class, primary=primary)

    def has(self, attr: str) -> bool:
        return any(f.attr == attr for f in self.facts)

    def ordered(self) -> list:
        """Facts with each primary one after the others of its attribute, so a reader that
        keeps the last fact per attribute gets the primary one."""
        return sorted(self.facts, key=lambda f: f.primary)


# --------------------------------------------------------------------------- cover page

def find_cover(pages: list) -> Optional[int]:
    """0-based index of the 10-K/10-Q cover. Glossy annual reports put pages in
    front of it, so look for the SEC header words rather than assume page 1."""
    for i, p in enumerate(pages):
        t = raw_text(p)
        if (re.search(r"FORM\s+10-?[KQ]\b", t, re.I) and re.search(r"Commission\s+file\b", t, re.I)
                and re.search(r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION", t, re.I)):
            return i
    return None


def cover_flows(pages: list, ci: int) -> list:
    """Cover page plus the part of the next page before the table of contents.
    Long covers (WMT, NVDA) spill the float and share count onto page 2."""
    first = make_flow(ci + 1, pages[ci], columns=False)
    out = [first]
    if re.search(r"DOCUMENTS\s+INCORPORATED\s+BY\s+REFERENCE", first.text, re.I) or ci + 1 >= len(pages):
        return out
    lines = []
    for ln in group_lines(pages[ci + 1]["words"]):
        t = " ".join(w.text for w in ln)
        if re.match(r"Table\s+of\s+Contents", t, re.I) and ln[0].y0 < 60:
            continue  # running page header link, not the contents itself
        if re.match(r"(TABLE OF CONTENTS|INDEX\b|PART I\b|Item 1\.)", t, re.I):
            break
        lines.append(ln)
        if re.search(r"DOCUMENTS\s+INCORPORATED\s+BY\s+REFERENCE", t, re.I):
            break
    out.append(Flow(ci + 2, lines))
    return out


def form_type(cover: Flow) -> str:
    return "10-Q" if re.search(r"FORM\s+10-?Q\b|QUARTERLY\s+REPORT\s+PURSUANT", cover.text, re.I) else "10-K"


def rule_fiscal_year_end(c: Collector, flows: list):
    # "For the fiscal year ended September 27, 2025"; MS prints "For the year ended";
    # JPM prints the date one line lower beside "Commission file number" (gap words).
    rx = re.compile(r"\bfor\s+the\s+(?:fiscal\s+)?year\s+ended:?\s+((?:[A-Za-z]+\s+){0,3}?)(" + MONTH_RE +
                    r")\s+(\d{1,2}),?\s+(\d{4})", re.I)
    for f in flows:
        m = rx.search(f.text)
        if m:
            conf = "low" if m.group(1).strip() else "high"
            c.from_flow("fiscal_year_end", to_iso(m.group(2), m.group(3), m.group(4)), f,
                        m.start(), m.end(), conf, quote=m.group(0))
            return


def _label_span(line: list, k: int) -> tuple:
    """x-range of a parenthesised cover label starting at word k (ends at ')' or ':')."""
    x0, x1, colon = line[k].x0, line[k].x1, False
    for w in line[k + 1:]:
        if w.text.startswith("(") or w.x0 - x1 > 20:
            break
        x1 = w.x1
        if w.text.endswith(")") or w.text.endswith(":"):
            colon = w.text.endswith(":")
            break
    return x0, x1, colon


def _jurisdiction_near(lines: list, li: int, x0: float, x1: float, step: int) -> Optional[tuple]:
    """Look one or two lines above (step=-1) or below (step=+1) the label for a
    state/country printed over the label's x-range."""
    for d in (1, 2):
        j = li + step * d
        if not 0 <= j < len(lines):
            return None
        over = [w for w in lines[j] if w.x1 > x0 - 25 and w.x0 < x1 + 25]
        if not over:
            continue
        m = JURIS_RE.search(" ".join(w.text for w in over))
        if m:
            return canon_jurisdiction(m.group(1)), j, over
    return None


def rule_state_of_incorporation(c: Collector, flow: Flow):
    lines = flow.lines
    for li, ln in enumerate(lines):
        t = " ".join(w.text for w in ln)
        # PG style: "State of Incorporation: Ohio" on one line
        m = re.search(r"State\s+of\s+Incorporation:\s*(.+)", t, re.I)
        if m and JURIS_RE.match(m.group(1)):
            c.add("state_of_incorporation", canon_jurisdiction(JURIS_RE.match(m.group(1)).group(1)),
                  flow.page, t, ln, "high")
            return
        for k, w in enumerate(ln):
            nxt = ln[k + 1].text.lower() if k + 1 < len(ln) else ""
            if w.text.strip("(").lower() == "state" and nxt in ("or", "of"):
                x0, x1, colon = _label_span(ln, k)
                # value printed ABOVE "(State or other jurisdiction ...)"; BAC prints
                # "State or other jurisdiction of incorporation or organization:" with the value BELOW
                hit = _jurisdiction_near(lines, li, x0, x1, +1 if colon else -1)
                if hit:
                    value, j, over = hit
                    quote = " ".join(x.text for x in lines[j]) + " | " + t
                    c.add("state_of_incorporation", value, flow.page, quote, over, "high")
                    return


_CITY_STATE = re.compile(r"(?:^|,\s*|\s)(" + STATE_TOKEN + r")\.?,?(?:\s+(\d{5}(?:-\d{4})?))?\s*$")


def parse_city_state(seg: str) -> Optional[tuple]:
    """'22777 Springwoods Village Parkway, Spring, Texas 77389-1425' -> ('Spring', 'TX').
    Also 'New York NY 10013', 'Rahway New Jersey 07065', 'San Francisco, California'."""
    seg = seg.strip()
    if seg.upper() == seg:  # MSFT prints "ONE MICROSOFT WAY, REDMOND, WASHINGTON 98052-6399"
        seg = " ".join(w if w.strip(",.") in ABBRS else w.title() for w in seg.split())
    m = _CITY_STATE.search(seg)
    if not m:
        return None
    head = seg[:m.start(1)].rstrip(", ")
    city = head.split(",")[-1].strip()
    if not city or re.search(r"\d", city) or len(city.split()) > 4 or not city[0].isupper():
        return None
    st = m.group(1)
    code = "DC" if st == "D.C." else STATES.get(st, st)
    return city, code, bool(m.group(2)) or "," in seg


def _segments(line: list, gap: float = 45.0) -> list:
    """Split a printed line into runs of words separated by wide gaps (cover columns)."""
    segs, cur = [], [line[0]]
    for w in line[1:]:
        if w.x0 - cur[-1].x1 > gap:
            segs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    segs.append(cur)
    return segs


def rule_hq(c: Collector, flow: Flow):
    lines = flow.lines
    for li, ln in enumerate(lines):
        t = " ".join(w.text for w in ln)
        m = re.search(r"\(?Address\s+(?:and\s+telephone\s+number,\s+including\s+area\s+code,\s+)?of\s+(?:registrant.s\s+)?principal\s+executive\s+offices", t, re.I)
        if not m:
            continue
        lab = [w for w in ln if w.text.strip("(").lower().startswith("address")][0]
        x0, x1 = lab.x0, lab.x0 + 200
        below = t.rstrip().endswith(":")
        # address lines sit just ABOVE the "(Address of principal executive offices)"
        # label (BAC: below "Address of principal executive offices:"). Two-column
        # covers (MS) print the city on the label's own line, so it is checked too.
        order = range(li + 1, min(li + 5, len(lines))) if below else range(li, max(li - 5, -1), -1)
        # word gaps: 25pt separates side-by-side cover columns (MS), 45pt keeps a
        # justified "Eden Prairie,   Minnesota 55344" together (UNH)
        for j, gap in ((j, g) for j in order for g in (25.0, 45.0)):
            for seg in _segments(lines[j], gap):
                if seg[-1].x1 < x0 - 40 or seg[0].x0 > x1:
                    continue
                st = " ".join(w.text for w in seg)
                got = parse_city_state(st)
                if got and got[2]:
                    city, code, _ = got
                    q = st + " | " + t
                    c.add("hq_city", city, flow.page, q, seg, "high")
                    c.add("hq_state", code, flow.page, q, seg, "high")
                    return
    # No address label (PG, MRK): one "street, City, State ZIP" line between the
    # registrant name and the Section 12(b) table, only if exactly one exists.
    hits = []
    started = False
    for ln in lines:
        t = " ".join(w.text for w in ln)
        if re.search(r"Commission\s+file", t, re.I):
            started = True
            continue
        if re.search(r"Section\s+12\(b\)", t, re.I):
            break
        if started:
            for seg in _segments(ln):
                st = " ".join(w.text for w in seg)
                got = parse_city_state(st)
                if got and re.search(r"\d{5}(?:-\d{4})?\s*$", st):
                    hits.append((got, st, seg))
    if len(hits) == 1:
        (city, code, _), st, seg = hits[0]
        c.add("hq_city", city, flow.page, st, seg, "low")
        c.add("hq_state", code, flow.page, st, seg, "low")


_EXCH = re.compile(r"\s(?:The\s+)?(New\s+York\s+Stock\s+Exchange|NYSE(?!\s+Arca)|Nasdaq|NASDAQ)\b.*$", re.I)


def class_label(text: str) -> str:
    """The class of stock a printed phrase names: 'Class A', 'Class B-1', or 'Common' when
    no class letter is printed."""
    m = re.search(r"\bClass\s+([A-C](?:-\d)?)\b", text)
    return f"Class {m.group(1)}" if m else "Common"


def _common_rows(flow: Flow) -> list:
    """Common-stock rows of the Section 12(b) table, as (class, symbol, exchange, line text,
    symbol words, line words); 'Common Stock, $0.00001 par value per share  AAPL  The Nasdaq
    Stock Market LLC'. Preferred, depositary, note and warrant rows are skipped."""
    rows, in_table = [], False
    for ln in flow.lines:
        t = " ".join(w.text for w in ln)
        if re.search(r"Section\s+12\(b\)", t, re.I):
            in_table = True
            continue
        if not in_table:
            continue
        if re.search(r"Section\s+12\(g\)|Indicate\s+by\s+check", t, re.I):
            break
        m = _EXCH.search(t)
        if not m:
            continue
        head = t[:m.start()].split()
        cls = " ".join(head[:-1])
        if not head or not re.search(r"\b(common|capital)\s+(stock|shares)\b", cls, re.I):
            continue
        if re.search(r"Depositary|Preferred|Notes|Warrant|Guarantee", cls, re.I):
            continue
        sym = head[-1]
        if not re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", sym):
            continue
        exch = "NYSE" if re.match(r"New|NYSE", m.group(1), re.I) else "NASDAQ"
        if any(r[1] == sym for r in rows):  # the same symbol listed on a second exchange
            continue
        rows.append((class_label(cls), sym, exch, t, [w for w in ln if w.text == sym][:1], ln))
    return rows


def rule_exchange_symbol(c: Collector, flow: Flow):
    """Exchange and trading symbol from the first common row of the Section 12(b) table.

    One trading_symbol per filing. When several common classes are listed (GOOGL and GOOG)
    each symbol is kept with its class and the first row is the primary one. A single listed
    class is named only when it carries a class letter (NKE lists Class B), because that
    letter says which of the cover's share counts the symbol belongs to."""
    rows = _common_rows(flow)
    for k, (cls, sym, exch, t, words, ln) in enumerate(rows):
        named = len(rows) > 1 or cls != "Common"
        c.add("trading_symbol", sym, flow.page, t, words, "high", share_class=cls if named else None,
              primary=k == 0)
    if rows:
        c.add("exchange", rows[0][2], flow.page, rows[0][3], rows[0][5], "high")


def _as_of(text: str) -> Optional[str]:
    m = re.search(r"(?:as\s+of|on|at)\s+(" + MONTH_RE + r")\s+(\d{1,2}),?\s+(\d{4})", text, re.I)
    return to_iso(m.group(1), m.group(2), m.group(3)) if m else None


def rule_shares_outstanding(c: Collector, flows: list):
    cands = []  # (class, iso date or "", value, flow, start, end)
    num = r"(?P<n>" + NUM_RE + r")(?:\s+(?P<sc>million|billion))?"
    # A: "14,532,182,000 shares of common stock were issued and outstanding as of ..."
    #    "3,918,623,149 shares of Common Stock and 70,852,076 shares of Class B Stock"
    rx_a = re.compile(r"(?<![$\d.,])" + num + r"\s+(?:common\s+)?shares\b(?P<tail>[^\d]{0,70})", re.I)
    # B: "Number of shares of common stock outstanding as of January 28, 2026 10,734,920,870"
    #    "... outstanding as of September 29, 2026, was 443,129,349"
    rx_b = re.compile(r"outstanding[^.\d]{0,45}?(?:as\s+of|on|at)\s+" + DATE_RE +
                      r"\s*[,:—–-]?\s*(?:(?:was|were|is)\s+)?(?:approximately\s+)?(?<![$])" + num + r"(?!\s*%)", re.I)
    for f in flows:
        for m in rx_a.finditer(f.text):
            qs, qe = sentence_bounds(f.text, m.start(), m.end())
            sent, before = f.text[qs:qe], f.text[qs:m.start()]
            if "outstanding" not in sent.lower() or re.match(r"\s*held\s+by", m.group("tail"), re.I):
                continue
            # "...excludes 1.0 billion shares held by directors" / float sentences
            if re.search(r"aggregate\s+market\s+value", before, re.I) and not re.search(r"there\s+were|had\s+outstanding", before, re.I):
                continue
            cls = class_label(m.group("tail").split(" and ")[0])
            cands.append((cls, _as_of(sent) or "", to_number(m.group("n"), m.group("sc")), f,
                          m.start("n"), m.end("sc") if m.group("sc") else m.end("n"), None))
        for m in rx_b.finditer(f.text):
            qs, _ = sentence_bounds(f.text, m.start(), m.end())
            pre = f.text[qs:m.end()]
            cands.append((class_label(pre), _as_of(pre) or "", to_number(m.group("n"), m.group("sc")), f,
                          m.start("n"), m.end("sc") if m.group("sc") else m.end("n"), None))
        cands += _shares_table(f)
    if not cands:
        return
    best = {}
    for cand in cands:  # per class keep the latest as-of date (CAT, F print two dates)
        k = cand[0]
        if k not in best or cand[1] > best[k][1]:
            best[k] = cand
    classes = list(best)
    multi = len(classes) > 1
    principal = _principal_class(c, classes)
    for cls, date, value, f, s, e, quote in best.values():
        conflict = any(x[0] == cls and x[1] == date and x[2] != value for x in cands)
        conf = "low" if conflict or not date else "high"
        prefix = f"{cls}: " if multi else ""
        c.from_flow("shares_outstanding", value, f, s, e, conf, quote=quote, prefix=prefix,
                    share_class=cls if multi else None, primary=cls == principal)


def _principal_class(c: Collector, classes: list) -> str:
    """The class a count without a named class means: the first class listed under a trading
    symbol, else the first class printed on the cover. A holding company's total is not
    summed: classes carry different rights (Visa's convert at set ratios), so a sum is not
    a number the filing states."""
    listed = [f.share_class for f in c.facts if f.attr == "trading_symbol" and f.share_class]
    return next((k for k in listed if k in classes), classes[0])


def _shares_table(f: Flow) -> list:
    """XOM/NKE/V print the count as a small table under an 'outstanding' heading:
    'Class A common stock 1,724,473,215'. Each class row becomes a candidate."""
    out = []
    for li, ln in enumerate(f.lines):
        t = f.line_text(li)
        if not re.search(r"outstanding", t, re.I):
            continue
        if not re.search(r"(:|as follows\.?|Outstanding as of " + DATE_RE + r")\s*$", t, re.I):
            continue
        date = _as_of(t) or (_as_of(f.line_text(li - 1)) if li else None) or ""
        for j in range(li + 1, min(li + 9, len(f.lines))):
            row = f.line_text(j)
            if re.fullmatch(r"Class\s+Shares\s+outstanding", row, re.I):
                continue
            # row label: "Class A common stock", "Common stock, without par value", or
            # KO's 10-Q "$0.25 Par Value" under "Class of Common Stock"
            m = re.fullmatch(r"((?:Class\s+[A-C](?:-\d)?\b[^\d$]*)|(?:Common\s+(?:stock|shares)[^\d$]*)|"
                             r"(?:\$[\d.]+\s+Par\s+Value))\s+(" + NUM_RE + r")", row, re.I)
            if not m:
                break
            s = f.line_start[j] + m.start(2)
            out.append((class_label(m.group(1)), date, to_number(m.group(2), None), f, s, s + len(m.group(2)),
                        t + " | " + row))
    return out


def _float_amount(window: str) -> Optional[tuple]:
    """(value, match) of the first dollar amount of $1 million or more in the float sentence,
    else of the first such bare figure (MCD prints 'as of June 30, 2025: 208,464,334,129.').
    Smaller figures are per-share prices, years or days."""
    for dollar in (r"\$\s?", r""):
        for m in re.finditer(dollar + r"(" + NUM_RE + r")(?:\s+(million|billion|trillion))?", window):
            if to_number(m.group(1), m.group(2)) >= 1_000_000:
                return to_number(m.group(1), m.group(2)), m
    return None


def rule_public_float(c: Collector, flows: list):
    found = []
    for f in flows:
        for a in re.finditer(r"aggregate\s+market\s+values?", f.text, re.I):
            qs, qe = sentence_bounds(f.text, a.start(), a.end())
            window = f.text[a.end():qe]
            if re.search(r"were:\s*Class", window):
                found += _float_table(f, a.end())
                continue
            hit = _float_amount(window)
            if hit:
                v, m = hit
                pre = window[max(0, m.start() - 20):m.start()]
                conf = "low" if re.search(r"excess\s+of|more\s+than|less\s+than", pre, re.I) else "high"
                found.append((v, f, a.end() + m.start(), a.end() + m.end(), conf))
    if not found:
        return
    v, f, s, e, conf = found[0]
    if any(x[0] != v for x in found[1:]):
        conf = "low"
    c.from_flow("public_float_usd", v, f, s, e, conf)


def _float_table(f: Flow, start: int) -> list:
    """NKE: 'held by non-affiliates were: Class A $ 4,015,962,912 Class B 76,513,157,723
    $ 80,529,120,635'. Take the last figure only when it is the sum of the others."""
    rest = f.text[start:start + 300]
    stop = re.search(r"\bAs\s+of\b|\bThe\s+number\b|\bNumber\s+of\b", rest)
    rest = rest[:stop.start()] if stop else rest
    nums = [(to_number(m.group(1), None), m) for m in re.finditer(r"(" + NUM_RE + r")", rest)
            if to_number(m.group(1), None) >= 1_000_000]
    if len(nums) >= 3 and nums[-1][0] == sum(n for n, _ in nums[:-1]):
        m = nums[-1][1]
        return [(nums[-1][0], f, start + m.start(), start + m.end(), "high")]
    return []


# --------------------------------------------------------------------------- no cover table

# "common stock is listed on the NYSE under the ticker symbol “C”";
# "is listed and trades on the New York Stock Exchange: WFC."
_LISTED = re.compile(r"\blisted\s+(?:and\s+trades\s+)?on\s+the\s+(New\s+York\s+Stock\s+Exchange|NYSE|Nasdaq[A-Za-z ]*?)\s*"
                     r"(?:under\s+the\s+(?:ticker\s+)?symbol\s*|:\s*)[“\"]?([A-Z]{1,5}(?:\.[A-Z])?)\b")
# "3,092,618,850 common shares outstanding (12/31/25)"
_SHARES_DATED = re.compile(r"(" + NUM_RE + r")\s+(?:common\s+)?shares\s+outstanding\s*\((\d{1,2})/(\d{1,2})/(\d{2,4})\)", re.I)
_REPORT_YEAR = re.compile(r"Form\s+10-K\s+for\s+the\s+(?:fiscal\s+)?year\s+ended\s+(" + MONTH_RE + r")\s+(\d{1,2}),?\s+(\d{4})", re.I)


def rule_listing_sentence(c: Collector, flows: list):
    """Exchange and symbol from a sentence in the stockholder information, for filings whose
    Section 12(b) table sits in an exhibit (C: 'see Exhibit 99.01') or is absent (an annual
    report printed without its 10-K cover, WFC)."""
    for f in flows:
        m = _LISTED.search(f.text)
        if m:
            exch = "NYSE" if re.match(r"New|NYSE", m.group(1), re.I) else "NASDAQ"
            around = " ".join(f.text[max(m.start() - 60, 0):m.end() + 60].split()[1:-1])  # whole words only
            c.from_flow("exchange", exch, f, m.start(), m.end(), "high", quote=around)
            c.from_flow("trading_symbol", m.group(2), f, m.start(2), m.end(2), "high", quote=around)
            return


def rule_shares_dated(c: Collector, flows: list):
    """A share count printed with its date in the stockholder information (no cover)."""
    for f in flows:
        m = _SHARES_DATED.search(f.text)
        if m:
            c.from_flow("shares_outstanding", to_number(m.group(1), None), f, m.start(1), m.end(1), "high")
            return


def rule_fiscal_year_end_report(c: Collector, flows: list):
    """'Annual Report on Form 10-K for the year ended December 31, 2025', read in an annual
    report printed without its cover. It names the report's year, so confidence is low."""
    for f in flows:
        m = _REPORT_YEAR.search(f.text)
        if m:
            c.from_flow("fiscal_year_end", to_iso(m.group(1), m.group(2), m.group(3)), f,
                        m.start(), m.end(), "low", quote=m.group(0))
            return


def rule_hq_back_page(c: Collector, page: dict, page_no: int):
    """The one 'street, City, ST ZIP' line on the last page (an annual report's back cover)."""
    hits = []
    for ln in group_lines(page["words"]):
        for seg in _segments(ln):
            st = " ".join(w.text for w in seg)
            got = parse_city_state(st)
            if got and re.search(r"\d{5}(?:-\d{4})?\s*$", st):
                hits.append((got, st, seg))
    if len(hits) == 1:
        (city, code, _), st, seg = hits[0]
        c.add("hq_city", city, page_no, st, seg, "low")
        c.add("hq_state", code, page_no, st, seg, "low")


# --------------------------------------------------------------------------- auditor

def _firm_name(printed: str) -> str:
    for pat, name in FIRMS:
        if re.match(pat, printed, re.I):
            return name
    return printed.strip()


def _location_lines(f: Flow, li: int) -> Optional[tuple]:
    """City, State printed on its own line under the signature (AAPL 'San Jose,
    California'), within a few lines either side; nearest after the anchor first."""
    order = [li + d for d in range(1, 7)] + [li - d for d in range(1, 7)]
    for j in order:
        if not 0 <= j < len(f.lines):
            continue
        t = f.line_text(j)
        m = re.fullmatch(r"([A-Z][A-Za-z.' ]{1,30}),\s+(" + STATE_TOKEN + r")\.?", t.strip())
        if m:
            st = m.group(2)
            return f"{m.group(1)}, {'DC' if st == 'D.C.' else STATES.get(st, st)}", j
    return None


# "We have served as the Company's auditor since 2009." / "we or our predecessor
# firms have served as the Company's auditor since at least 1942."
SINCE_RE = re.compile(r"\b[Ww]e\b[^.]{0,60}?served\s+as\s+the\s+[\w’'.\s]{0,40}?auditors?\s+since\s+(at\s+least\s+)?(\d{4})\.?")
SIG_RE = re.compile(r"/\s*s\s*/\s*(" + FIRM_RE + r"|[A-Z][A-Za-z&.' ]{2,40}?)\s*,?\s*LLP", re.I)
BARE_FIRM_RE = re.compile(r"^\s*(" + FIRM_RE + r")\s*,?\s*LLP\s*$", re.I)
# "(PCAOB ID 238)", "PCAOB Firm ID No. 00042", "PCAOB ID # 185", "Auditor Firm ID: 185"
PCAOB_RE = re.compile(r"(?:PCAOB|Auditor)\s+(?:Firm\s+)?ID(?:\s+No\.?)?\s*[:#]?\s*(\d{1,5})", re.I)
# Item 14 / index form: "KPMG LLP, Seattle, WA, Auditor Firm ID: 185"
FIRM_CITY_RE = re.compile(r"(" + FIRM_RE + r")\s*,?\s*LLP,\s+([A-Z][A-Za-z.' ]{1,30}),\s+(" + STATE_TOKEN +
                          r")\.?,?\s+(?:\(?PCAOB|Auditor)", re.I)


def rule_auditor(c: Collector, flows: list):
    tenure = [(f, m) for f in flows for m in SINCE_RE.finditer(f.text)]
    ids = [(f, m, PCAOB_IDS.get(int(m.group(1)))) for f in flows for m in PCAOB_RE.finditer(f.text)]
    ids = [x for x in ids if x[2]]
    _auditor_since(c, tenure)
    sig = _auditor_signature(flows, tenure[0]) if tenure else None
    id_firms = Counter(x[2] for x in ids)
    if sig:
        g, s, e, printed = sig
        name = _firm_name(printed)
        agrees = not id_firms or id_firms.most_common(1)[0][0] == name
        c.from_flow("auditor", name, g, s, e, "high" if agrees else "low", quote=g.text[s:e].strip())
    elif ids:  # signature printed as an image: the PCAOB firm ID names the firm (BAC, JPM)
        f, m, name = ids[0]
        c.from_flow("auditor", name, f, m.start(), m.end(), "high" if len(id_firms) == 1 else "low")
    _auditor_location(c, flows, sig, tenure, bool(ids))


def _auditor_since(c: Collector, tenure: list):
    """The tenure sentence appears only in the opinion on the financial statements."""
    if not tenure:
        return
    years = Counter(m.group(2) for _, m in tenure)
    f, m = tenure[0]
    y = m.group(2)
    if 1850 <= int(y) <= 2026:
        conf = "high" if len(years) == 1 and not m.group(1) else "low"  # "since at least 1920" is low
        c.from_flow("auditor_since", y, f, m.start(), m.end(), conf, quote=m.group(0))


def _auditor_signature(flows: list, first_tenure: tuple) -> Optional[tuple]:
    """'/s/ Ernst & Young LLP' nearest the tenure sentence (same page, then the pages
    either side); a bare 'PricewaterhouseCoopers LLP' line if the /s/ is an image (MRK).
    Returns (flow, start, end, printed firm)."""
    tf, tm = first_tenure
    near = sorted((g for g in flows if abs(g.page - tf.page) <= 1), key=lambda g: abs(g.page - tf.page))
    for g in near:
        hits = [(h.start(), h.end(), h.group(1)) for h in SIG_RE.finditer(g.text)]
        if not hits:
            for li in range(len(g.lines)):
                b = BARE_FIRM_RE.match(g.line_text(li))
                if b:
                    s = g.line_start[li]
                    hits.append((s, s + len(b.group(0).strip()), b.group(1)))
        if hits:
            ref = tm.start() if g is tf else (0 if g.page > tf.page else len(g.text))
            s, e, printed = min(hits, key=lambda h: abs(h[0] - ref))
            return g, s, e, printed
    return None


def _auditor_location(c: Collector, flows: list, sig, tenure: list, have_ids: bool):
    """City line under the signature (low confidence when the signature is an image and
    the city is read next to the tenure sentence), else the Item 14 'Firm LLP, City, ST' form."""
    anchor = (sig[0], sig[1]) if sig else ((tenure[0][0], tenure[0][1].start()) if tenure else None)
    if anchor and (sig or have_ids):
        g, pos = anchor
        loc = _location_lines(g, g.line_index(pos))
        if loc:
            value, j = loc
            c.add("auditor_location", value, g.page, g.line_text(j), g.lines[j], "high" if sig else "low")
            return
    for f in flows:
        m = FIRM_CITY_RE.search(f.text)
        if m:
            st = m.group(3)
            value = f"{m.group(2).strip()}, {'DC' if st == 'D.C.' else STATES.get(st, st)}"
            c.from_flow("auditor_location", value, f, m.start(), m.end(), "high")
            return


# --------------------------------------------------------------------------- employees

_QUAL = r"(?:(?:approximately|about|over|more\s+than|nearly|around|roughly|some|a\s+total\s+of)\s+)?"
_NUMX = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s+(million|thousand))?"
_MOD = r"(?:(?:full|part)[- ]time,?|equivalent|regular|permanent|seasonal|and|total|global)\s+"
_NOUN = r"(?:employees|associates|people|persons|team\s+members|colleagues|full-time\s+equivalents?)"
_AND = r"(?:\s+and\s+[\d,.]+(?:\s+(?:million|thousand))?)?"
EMP_STRONG = [
    # "the Company had approximately 166,000 full-time equivalent employees", "we employed about 112,900 full-time persons"
    re.compile(r"\b(?:had|have|has|employed|employ|employs)\s+" + _QUAL + _NUMX + _AND + r"\s+(?:" + _MOD + r"){0,6}" + _NOUN + r"\b", re.I),
    # "a global workforce of 78,865 employees"
    re.compile(r"\bworkforce\s+of\s+" + _QUAL + _NUMX + r"\s+(?:" + _MOD + r"){0,4}" + _NOUN + r"\b", re.I),
    # "a headcount of approximately 213,000", "we had headcount of 47,400"
    # TSLA: "our employee headcount worldwide was 134,785"
    re.compile(r"\bheadcount\s+(?:worldwide\s+|globally\s+)?(?:of|was|totaled)\s+" + _QUAL + _NUMX, re.I),
    # XOM: "The number of regular employees was 58 thousand, 61 thousand, and 62 thousand"
    re.compile(r"\bnumber\s+of\s+(?:[a-z-]+\s+){0,2}employees\s+(?:was|were|totaled)\s+" + _QUAL + _NUMX, re.I),
]
# "With approximately 24,900 employees in more than 30 countries as of ...", "our approximately 2.1 million associates"
EMP_WEAK = re.compile(r"\b(?:with|our)\s+" + _QUAL + _NUMX + r"\s+(?:" + _MOD + r"){0,4}" + _NOUN + r"\b", re.I)
_US_ONLY = re.compile(r"^\W*(?:in|located\s+in|based\s+in|within)\s+(?:the\s+)?(?:U\.S\.|United\s+States|North\s+America)", re.I)
# subsets described by what they did: "over 29,000 employees globally advancing their careers"
_SUBSET = re.compile(r"\b(?:received|participated|have\s+access|engaging|certified|volunteer|hired|advancing|promoted|"
                     r"trained|completed|enrolled|responsible\s+for|dedicated\s+to|to\s+manage)\b", re.I)
_DATEISH = re.compile(r"\bas\s+of\b|\bat\s+the\s+end\s+of\b|\byear[- ]end\b|" + MONTH_RE + r"\s+(?:\d{1,2},?\s+)?\d{4}|\bfiscal\s+\d{4}\b", re.I)
_GLOBAL = re.compile(r"worldwide|globally|global\s+workforce|around\s+the\s+world|\b\d+\s+countries|\btotal\b", re.I)


def _employee_table_rows(f: Flow) -> list:
    """CVX prints a human-capital table: 'Total Employees 12,438 29 % 30,013 70 % 588 1 % 43,039 100 %'.
    The company total is the largest count on the 'Total employees' row."""
    out = []
    for li in range(len(f.lines)):
        t = f.line_text(li)
        if not re.match(r"Total\s+(?:number\s+of\s+)?employees\b", t, re.I):
            continue
        nums = [(int(m.group(0).replace(",", "")), m) for m in re.finditer(r"\d{1,3}(?:,\d{3})+", t)]
        nums = [x for x in nums if 1_000 <= x[0] <= 3_000_000]
        if nums:
            v, m = max(nums, key=lambda x: x[0])
            s = f.line_start[li] + m.start()
            out.append((3, v, f, s, s + len(m.group(0)), t))
    return out


def _text_headcount(flows: list) -> Optional[tuple]:
    """Best headcount sentence as (value, flow, start, end, quote, confidence), or None."""
    cands = []  # (score, value, flow, start, end, quote or None)
    for f in flows:
        page_hc = bool(re.search(r"human\s+capital|our\s+people|workforce", f.text, re.I))
        found = [(m, 3) for rx in EMP_STRONG for m in rx.finditer(f.text)]
        found += [(m, 1) for m in EMP_WEAK.finditer(f.text)]
        for m, base in found:
            v = to_number(m.group(1), m.group(2))
            if not 1_000 <= v <= 3_000_000:
                continue
            qs, qe = sentence_bounds(f.text, m.start(), m.end())
            sent, before, after = f.text[qs:qe], f.text[qs:m.start()], f.text[m.end():m.end() + 40]
            score = base
            if _DATEISH.search(sent):
                score += 2
            if _GLOBAL.search(sent):
                score += 1
            if re.match(r"\W*(?:worldwide|globally|around\s+the\s+world|in\s+(?:more\s+than\s+|over\s+)?\d+\s+countries)", after, re.I):
                score += 1  # F: "With about 169,000 employees worldwide"
            if page_hc:
                score += 1
            if _US_ONLY.search(after) or re.search(r"\bIn\s+the\s+(?:U\.S\.|United\s+States),", before):
                score -= 3
            if _SUBSET.search(after):
                score -= 3
            if score >= 3:
                cands.append((score, v, f, m.start(), m.end(), None))
        cands += [(s + (1 if page_hc else 0), *rest) for s, *rest in _employee_table_rows(f)]
    if not cands:
        return None
    top = max(x[0] for x in cands)
    score, v, f, s, e, quote = min((x for x in cands if x[0] == top), key=lambda x: x[2].page)
    rivals = {x[1] for x in cands if x[1] != v}
    agree = sum(1 for x in cands if x[1] == v)
    # quote the as-of wording: if another sentence prints the same count with a
    # date (WMT p6 '2.1 million associates as of January 31, 2026'), quote that one
    if not _DATEISH.search(clip_quote(f.text, s, e)):
        dated = [x for x in cands if x[1] == v and _DATEISH.search(clip_quote(x[2].text, x[3], x[4]))]
        if dated:
            _, v, f, s, e, quote = max(dated, key=lambda x: x[0])
    conf = "high" if not rivals and (score >= 5 or agree >= 2) else "low"
    return v, f, s, e, quote, conf


# A row about people: 'Total employees', 'Headcount (#)', 'Direct staff (in thousands)'.
_HEADCOUNT_LABEL = re.compile(r"^(?:total\s+)?(?:employees|headcount|direct\s+staff|number\s+of\s+employees)\b", re.I)
_HEADCOUNT_TITLE = re.compile(r"employ|headcount|workforce", re.I)


def _table_headcount(rows: list) -> Optional[dict]:
    """The company headcount from the table layer: count rows labelled as employees, or
    'Total ...' rows of a table titled as employment ('Worldwide employment (in thousands)',
    GM: 'Total Worldwide 155'). Values arrive in full units with the row's own scale. Takes
    the newest period of each row and the largest row (a company total over its segments)."""
    best = None
    for r in rows:
        if r["unit"] != "count" or not r["cells"]:
            continue
        if not (_HEADCOUNT_LABEL.match(r["label"])
                or (_HEADCOUNT_TITLE.search(r["title"]) and r["label"].lower().startswith("total"))):
            continue
        cell = max(r["cells"], key=lambda x: (x["pid"], x["value"]))
        if 1_000 <= cell["value"] <= 3_000_000 and (best is None or cell["value"] > best[0]["value"]):
            best = (cell, r)
    if best is None:
        return None
    cell, r = best
    return {"value": cell["value"], "page": r["page"], "bbox": tuple(cell["bbox"]),
            "quote": f"{r['title']} | {r['label']} {cell['text']}"}


def _mention_headcount(mentions: list) -> Optional[tuple]:
    """First sentence of the sentence layer that gives a count of people ('Over 150,000 people
    are employed by the Company'), as (page, sentence, printed number, value), or None."""
    for m in sorted(mentions, key=lambda x: x["page"]):
        if not re.search(r"\b(?:employ\w*|associates|team\s+members)\b", m["sentence"], re.I):
            continue
        for n in m["numbers"]:
            after = m["sentence"].split(n["text"], 1)[-1][:40]
            if (n["unit"] == "count" and 1_000 <= n["value"] <= 3_000_000
                    and not _SUBSET.search(after) and not _US_ONLY.search(after)):
                return m["page"], m["sentence"], n["text"], n["value"]
    return None


def rule_employees(c: Collector, flows: list, rows: list, mentions: list, pages: list):
    """Employees: the sentence rules first, confirmed or contradicted by the table layer; a
    headcount table alone (GM, WFC) when no sentence states one; the sentence layer last."""
    text, table = _text_headcount(flows), _table_headcount(rows)
    if text:
        v, f, s, e, quote, conf = text
        if table:
            conf = "high" if table["value"] == v else "low"
        c.from_flow("employees", v, f, s, e, conf, quote=quote)
    elif table:
        c.add("employees", table["value"], table["page"], table["quote"], [], "high", bbox=table["bbox"])
    else:
        hit = _mention_headcount(mentions)
        if hit:
            page, sentence, printed, value = hit
            flow = make_flow(page, pages[page - 1])
            at = max(flow.text.find(printed), 0)
            c.from_flow("employees", value, flow, at, at + len(printed), "low", quote=sentence[:300])


# --------------------------------------------------------------------------- officers

_NAME_STOP = re.compile(r"^(?:Chief|President|Chair|Chairman|Chairwoman|Chairperson|Executive|Senior|Vice|Director|"
                        r"Principal|Treasurer|Controller|Managing|Officer|Lead|Corporate|Group|General|"
                        r"Attorney|Title|Date|Dated|By|Signature|and|Corporation|Company|Inc\.?|Incorporated|"
                        r"Signatures?|Board|Directors?|Registrant|Information|Supplemental|Report|Statements?|"
                        r"Notes|Table|Contents|Part|Item|Exhibits?|Index)\b,?", re.I)
_NAME_TOKEN = re.compile(r"^(?:[A-Z][A-Za-z'’\-]*\.?,?|[A-Z]\.(?:[A-Z]\.)?,?|Jr\.?,?|Sr\.?,?|II|III|IV)$")


def _name_run(words: list) -> list:
    """Leading words that read as a person's name: 'Timothy D. Cook', 'JEN-HSUN HUANG',
    'Roger W. Ferguson Jr.'. Stops at a title word, a wide gap or a date ('May 1')."""
    run = []
    for k, w in enumerate(words[:6]):
        if run and w.x0 - run[-1].x1 > 12:
            break
        nxt = words[k + 1].text if k + 1 < len(words) else ""
        if _NAME_STOP.match(w.text) or not _NAME_TOKEN.match(w.text) or w.text.startswith("("):
            break
        if re.fullmatch(MONTH_RE, w.text) and re.match(r"\d", nxt):
            break
        run.append(w)
    return run


def _signatures(line: list) -> list:
    """(name, words) for each '/s/ Name' on a printed line ('/s/Lisa T. Su', '*/s/ X', '/ S / X')."""
    out, i = [], 0
    while i < len(line):
        t = line[i].text
        m = re.match(r"^\*?/[sS]/(.*)$", t)
        if m:
            start, first = i, m.group(1)
        elif t == "/" and i + 2 < len(line) and line[i + 1].text in ("s", "S") and line[i + 2].text == "/":
            start, first, i = i, "", i + 2
        else:
            i += 1
            continue
        run = _name_run(line[i + 1:])
        toks = ([first] if first else []) + [w.text for w in run]
        if len(toks) >= 2:
            out.append((" ".join(toks).strip(" ,"), [line[start]] + run))
        i += 1 + len(run)
    return out


def _typed_name(line: list) -> Optional[tuple]:
    """A typed name opening a line, alone or followed by a title/date ('Michael J.
    Fiddelke'; MRK 'ROBERT M. DAVIS Chairman, Chief Executive Officer and ...').
    Used only on pages that print no /s/ marks at all."""
    words = line[1:] if line[0].text.rstrip(":").lower() == "by" else line
    run = _name_run(words)
    if not 2 <= len(run) <= 5 or not any(len(w.text.strip(".,")) > 2 for w in run):
        return None
    rest = " ".join(w.text for w in words[len(run):])
    if rest and not re.search(r"Officer|President|Director|Chair|" + MONTH_RE, rest, re.I):
        return None
    return " ".join(w.text for w in run).strip(" ,"), run


def _is_attorney(lines: list, y: float, x: float) -> bool:
    """'/s/ JENNIFER ZACHARY' over '(Attorney-in-Fact)' signs for others (MRK, TGT)."""
    return any(re.search(r"Attorney-in-Fact", " ".join(w.text for w in ln), re.I)
               for ln in lines if -3 < ln[0].yc - y < 35 and ln[-1].x1 > x)


def _page_signatures(lines: list) -> list:
    """(yc, x0, name, words) for every officer/director signature on the page."""
    sigs = [(ln[0].yc, ws[0].x0, name, ws) for ln in lines for name, ws in _signatures(ln)]
    sigs = [s for s in sigs if not _is_attorney(lines, s[0], s[1])]
    if not sigs:
        sigs = [(ln[0].yc, ln[0].x0, *_typed_name(ln)) for ln in lines if _typed_name(ln)]
        sigs = [s for s in sigs if not _is_attorney(lines, s[0], s[1])]
    return sigs


def _officer_from_page(page_no: int, page: dict, marker: re.Pattern, block: bool) -> list:
    """Signatures that carry the role marker on this page, as (name, words, line, conf, page).

    Layouts handled:
      * marker on the /s/ line: '/s/ Timothy D. Cook (Principal Executive Officer)' (AAPL);
      * marker inside the signer's block, under the /s/ line: '(Principal Executive Officer)'
        two lines below '/s/ Brian T. Moynihan' (BAC, ADBE, AMD) -> nearest signature above;
      * marker as a grid heading over its column: 'Principal Financial Officer' above
        '/s/ NEIL A. HANSEN' (XOM) -> nearest signature below in the same column.
    `block` forces the block reading (used for bare 'Chief Executive Officer' titles)."""
    lines = group_lines(page["words"])
    sigs = _page_signatures(lines)
    out = []
    for li, ln in enumerate(lines):
        for t, mw, m, mxc in _markers(lines, li, marker):
            y = ln[0].yc
            # a heading opens its line (or ends with ':') and is not parenthesised:
            # XOM 'Principal Financial Officer', NKE 'PRINCIPAL FINANCIAL OFFICER:'
            paren = t[:m.start()].rstrip().endswith("(") or t[m.end():].lstrip().startswith(")")
            heading = not block and not paren and (mw is ln[0] or t.rstrip().endswith(":"))
            same = [s for s in sigs if abs(s[0] - y) < 3 and s[1] < mw.x0]
            above = [s for s in sigs if s[0] < y - 3]
            below = [s for s in sigs if s[0] > y + 3]
            pick, conf = None, "high"
            if same:
                pick = max(same, key=lambda s: s[1])
            elif not heading:
                a = _column_pick(above, mxc, y, -1)
                b = _column_pick(below, mxc, y, +1)
                da, db = (y - a[0] if a else 1e9), (b[0] - y if b else 1e9)
                pick = a if a and da <= 2 * db else b
                # high when nearer, or when the typed name under the /s/ line sits
                # between the signature and the marker (WMT: '/s/ John R. Furner',
                # 'John R. Furner', title, '(Principal Executive Officer)')
                typed = pick is a and any(a[2].lower() in " ".join(w.text for w in lines[k]).lower()
                                          for k in range(len(lines)) if a[0] + 3 < lines[k][0].yc < y + 3)
                conf = "high" if pick is a and (da <= db or typed) else "low"
            else:
                col = [s for s in below if abs(s[1] - mw.x0) < 60]
                pick = min(col, key=lambda s: s[0]) if col else None
            if pick:
                quote = t if same else " ".join(w.text for w in pick[3]) + " | " + t
                out.append((pick[2], pick[3], quote, conf, page_no))
    return out


def _markers(lines: list, li: int, marker: re.Pattern) -> list:
    """(line text, first marker word, match, marker x-centre) for each role marker on line li. A marker
    broken over two lines ('(Principal Executive' / 'Officer)', MS) is joined with the
    first line below that starts within 150pt to the right."""
    ln = lines[li]
    t = " ".join(w.text for w in ln)
    # long lines are prose, and exhibit-index rows ('31.2 Certification of the Chief
    # Financial Officer pursuant to Section 302') are not signature captions
    if len(ln) > 20 or re.search(r"Certification|pursuant|Section\s+\d|Exhibit", t, re.I):
        return []
    starts, pos = [], 0
    for w in ln:
        starts.append(pos)
        pos += len(w.text) + 1
    text = t
    if re.search(r"\b(?:Principal|Chief)(?:\s+(?:Executive|Financial))?$", t, re.I):
        for nxt in lines[li + 1:li + 4]:
            if nxt[0].yc - ln[0].yc < 15 and abs(nxt[0].x0 - ln[-1].x0) < 150 and nxt[0].text.lower().startswith(("officer", "executive", "financial")):
                text = t + " " + " ".join(w.text for w in nxt)
                break
    out = []
    for m in marker.finditer(text):
        if m.start() >= len(t):
            break
        wi = max(i for i, s in enumerate(starts) if s <= m.start())
        span = ln[wi:wi + len(m.group(0).split())]
        out.append((text, ln[wi], m, (span[0].x0 + span[-1].x1) / 2))
    return out


def _column_pick(sigs: list, xc: float, y: float, direction: int) -> Optional[tuple]:
    """Nearest signature above (direction -1) or below (+1) in the signature column
    nearest the marker's x-centre. Columns are signatures within 80pt vertically,
    grouped by x-centre (a 'By: /s/' block high on the page is not a column).
    KO and GS print two signers side by side, each with its marker under it."""
    if not sigs:
        return None
    near = [s for s in sigs if abs(s[0] - y) < 80]
    if not near:
        return (max if direction < 0 else min)(sigs, key=lambda s: s[0])
    centre = {id(s): (s[3][0].x0 + s[3][-1].x1) / 2 for s in near}
    home = min(near, key=lambda s: abs(centre[id(s)] - xc))
    col = [s for s in near if abs(centre[id(s)] - centre[id(home)]) < 80]
    return (max if direction < 0 else min)(col, key=lambda s: s[0])


_SIG_PAGE = re.compile(r"Pursuant\s+to\s+the\s+requirements\s+of\s+(?:Section\s+13|the\s+Securities\s+"
                       r"Exchange\s+Act\s+of\s+1934,?\s+(?:this|the\s+registrant))", re.I)


def rule_officers(c: Collector, pages: list, first: int):
    """CEO / CFO from the 10-K signatures page (and the page after it, where the
    signature table often continues); Exhibit 31 certifications as a fallback."""
    starts = [i for i in range(first, len(pages)) if _SIG_PAGE.search(raw_text(pages[i]))][:2]
    scan = sorted({j for i in starts for j in (i, i + 1) if j < len(pages)})
    roles = {
        "ceo": (re.compile(r"Principal\s+Executive\s+Officer", re.I), re.compile(r"Chief\s+Executive\s+Officer", re.I)),
        "cfo": (re.compile(r"Principal\s+Financial\s+Officer", re.I), re.compile(r"Chief\s+Financial\s+Officer", re.I)),
    }
    certs = _certifications(pages, first)
    for attr, (principal, chief) in roles.items():
        hits = [h for i in scan for h in _officer_from_page(i + 1, pages[i], principal, False)]
        if not hits:
            hits = [h for i in scan for h in _officer_from_page(i + 1, pages[i], chief, True)]
        names = {title_name(h[0]) for h in hits}
        if len(names) == 1:  # two different names (co-CEOs) is not unambiguous: emit nothing
            name, ws, line, conf, page_no = hits[0]
            if any(h[3] == "high" for h in hits):
                conf = "high"
            if certs.get(attr) and _surname(certs[attr][0]) != _surname(title_name(name)):
                # the Exhibit 31 certification names the officer outright; trust it over
                # a signature-grid reading that disagrees, but flag the conflict
                cname, cpage, cquote, cws = certs[attr]
                c.add(attr, cname, cpage, cquote, cws, "low")
                continue
            c.add(attr, title_name(name), page_no, line, ws, conf)
        elif not hits and certs.get(attr):
            name, page_no, quote, ws = certs[attr]
            c.add(attr, name, page_no, quote, ws, "low")
        elif not hits:
            _officer_from_grid(c, attr, pages, first)


# A title under a name: 'Chairman and Chief Executive Officer', 'Chief Financial Officer'.
_GRID_TITLE = {
    "ceo": re.compile(r"(?:(?:Chairman|Chair|President)\s*(?:,|and)\s+)*Chief\s+Executive\s+Officer", re.I),
    "cfo": re.compile(r"Chief\s+Financial\s+Officer", re.I),
}


def _grid_names(page: dict, title: re.Pattern) -> list:
    """(name, words, line) for each name printed in a column with the role's title a few
    lines under it, as in an annual report's list of the operating committee."""
    cols = [_segments(ln, 12.0) for ln in group_lines(page["words"])]
    out = []
    for li, segs in enumerate(cols):
        for seg in segs:
            if not title.fullmatch(" ".join(w.text for w in seg).strip(" ,")):
                continue
            for up in range(li - 1, max(li - 4, -1), -1):
                named = [s for s in cols[up] if abs(s[0].x0 - seg[0].x0) < 12 and len(_name_run(s)) == len(s) >= 2]
                if named:
                    out.append((" ".join(w.text for w in named[0]), named[0], " ".join(w.text for w in seg)))
                    break
    return out


def _officer_from_grid(c: Collector, attr: str, pages: list, first: int):
    """CEO / CFO from a name-over-title grid when the filing has no signature page; one
    distinct name only, as for signatures. Low confidence: a grid is not a signature."""
    hits = [(i, h) for i in range(first, len(pages)) for h in _grid_names(pages[i], _GRID_TITLE[attr])]
    if len({title_name(h[0]) for _, h in hits}) == 1:
        i, (name, words, title) = hits[0]
        c.add(attr, title_name(name), i + 1, name + " | " + title, words, "low")


def _surname(name: str) -> str:
    toks = [t for t in re.split(r"[\s,]+", name.lower()) if t and t not in ("jr.", "jr", "sr.", "ii", "iii", "iv")]
    return toks[-1] if toks else ""


def _certifications(pages: list, first: int) -> dict:
    """Exhibit 31: 'I, Timothy D. Cook, certify that:' on a page that names one role."""
    out = {}
    rx = re.compile(r"\bI,\s+([A-Z][A-Za-z.'’\- ]{3,60}?),\s+certify", re.I)
    for i in range(first, len(pages)):
        t = raw_text(pages[i])
        m = rx.search(t)
        if not m:
            continue
        ceo = re.search(r"Chief\s+Executive\s+Officer|Principal\s+Executive\s+Officer", t[m.end():], re.I)
        cfo = re.search(r"Chief\s+Financial\s+Officer|Principal\s+Financial\s+Officer", t[m.end():], re.I)
        if bool(ceo) == bool(cfo):
            continue
        attr = "ceo" if ceo else "cfo"
        f = make_flow(i + 1, pages[i], columns=False)
        fm = rx.search(f.text)
        ws = f.span_words(fm.start(1), fm.end(1)) if fm else []
        out.setdefault(attr, (title_name(m.group(1).strip()), i + 1, m.group(0), ws))
    return out


# --------------------------------------------------------------------------- driver

def prose_flows(pages: list, first: int, pattern: str) -> list:
    """Column-aware flows for pages (from the cover on) whose raw text matches pattern."""
    rx = re.compile(pattern, re.I)
    return [make_flow(i + 1, pages[i]) for i in range(first, len(pages)) if rx.search(raw_text(pages[i]))]


def extract_text_facts(doc_id: str, pages: list, rows: Optional[list] = None,
                       mentions: Optional[list] = None) -> list:
    """All text facts of one filing. `rows` are its table-layer rows and `mentions` its
    sentence-layer records (both optional); headcounts are read from them as well as the pages."""
    c = Collector(doc_id)
    rows, mentions = rows or [], mentions or []
    ci = find_cover(pages)
    form = "10-K"
    if ci is not None:
        flows = cover_flows(pages, ci)
        form = form_type(flows[0])
        if form == "10-K":
            rule_fiscal_year_end(c, flows)
            rule_public_float(c, flows)
        rule_state_of_incorporation(c, flows[0])
        rule_hq(c, flows[0])
        rule_exchange_symbol(c, flows[0])
        rule_shares_outstanding(c, flows)
    first = ci or 0
    if form == "10-K":
        _stockholder_facts(c, pages, first, ci is None)
        rule_auditor(c, prose_flows(pages, first, r"auditor|LLP|PCAOB|Firm\s+ID"))
        rule_employees(c, prose_flows(pages, first, r"employ|associates|headcount|workforce|team\s+members|colleagues"),
                       rows, mentions, pages)
    rule_officers(c, pages, first)
    return c.ordered()


def _stockholder_facts(c: Collector, pages: list, first: int, no_cover: bool):
    """What the cover table did not give, from the stockholder information: the listing
    sentence, and for an annual report without a cover also the dated share count, the report
    year and the back-cover address."""
    if not c.has("trading_symbol"):
        rule_listing_sentence(c, prose_flows(pages, first, r"\blisted\s+(?:and\s+trades\s+)?on\s+the"))
    if no_cover:
        rule_shares_dated(c, prose_flows(pages, first, r"shares\s+outstanding\s*\("))
        rule_fiscal_year_end_report(c, prose_flows(pages, first, r"Form\s+10-K\s+for\s+the"))
        rule_hq_back_page(c, pages[-1], len(pages))
