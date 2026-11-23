"""The raw layer: every other table of every filing, waiting to be bound.

The base graph binds the three primary statements to shared concepts at build
time. Everything else a filing prints in tables (revenue by product, segment
or region, debt maturities, lease costs, store counts, headcount) is parsed and
checked the same way but left unbound: nobody has asked about it yet. Each row
keeps its page, its table (`table_id`), its table title, the note it sits
under, its printed label, and what the recovered formulas say about it:

    role      total / operand / loose, from the formulas recovered in its table
    parent    the row id of the subtotal whose formula includes this row
    children  on a subtotal: the row ids of its operands (with `signs`)
    verified  the row's numbers sit inside a passing formula
    ties_to   a concept of the base graph that prints the same value for the
              same period (the product table's total is Revenue)
    anchor    the concept its table's total ties to, so an operand of that
              total is known to be a component of that concept

Every value cell is kept, dated or not: `col` is its column header, `ci` its
column index, `pid`/`end` its period when the header is one. `value` is in
full units and `unit` is one of USD, shares, count, percent, ratio, USD/share.
Money versus count is decided from evidence on the page: dollar signs in the
table, its caption and header words. A count (stores, employees, square feet)
is never multiplied by the filing's money scale; it takes a scale only from its
own caption, column header or row label ("(in thousands)").

When a question names a line item that is not a seed concept, the living
layer searches these rows by their wording, lets an LLM pick one from the
labels, and admits the binding only on that structural evidence.
"""
from __future__ import annotations

import calendar
import gzip
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from typing import Optional

from . import pdfio
from .alltables import find_all_tables
from .calc import PER_SHARE_RE, discover, row_pool, row_section, verified_cells
from .concepts import norm_label
from .periods import column_period
from .tables import duration, printed_date

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "graph" / "raw.json.gz"
STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "total", "net", "from", "on", "at", "s", "was", "what"}
UNITS = ("USD", "shares", "count", "percent", "ratio", "USD/share")

SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "000": 1e3}
SCALE_WORD_RE = re.compile(r"\b(?:in|of)\s+(thousands|millions|billions)\b|\b(thousands|millions|billions)\s+of\b|"
                           r"\bin\s+(000)['’]?s\b", re.I)
# "(in millions, except remaining performance obligation, which is presented in billions)"
EXCEPT_RE = re.compile(r"except(?:\s+for)?\s+([a-z][\w ,'’-]{2,70}?),?\s+which\s+(?:is|are)\s+"
                       r"(?:presented|reported|stated|shown|expressed|disclosed)\s+in\s+(thousands|millions|billions)", re.I)
DOLLAR_RE = re.compile(r"\$|\bdollars?\b|\bUSD\b")
RATIO_LABEL_RE = re.compile(r"\bratios?\b|\(to 1\)|\btimes\b|margin|\byields?\b|\brates?\b|return on|leverage|"
                            r"\bbps\b|basis points|\bdays\b|percent|%|multiple|coverage", re.I)
COUNT_RE = re.compile(r"\b(?:stores?|store counts?|warehouses?|clubs|locations|retail units|unit counts?|"
                      # "Wholesale Units (000) (a) 2,862 2,728" (F p53, read as USD billions once the marker chain let
                      # the row be read): units a maker sells or ships are counted
                      r"wholesale units|units (?:sold|shipped|delivered|produced)|"
                      r"employees|employed|employment|"
                      r"headcount|staff|associates|members|cardholders|vehicles|deliveries|square (?:feet|footage)|"
                      r"sq\.?\s*ft|(?:distribution|fulfillment|supply chain|manufacturing) (?:centers|facilities)|"
                      r"branches|atms|restaurants|subscribers|users|barrels|wells|acres|number of)\b", re.I)
# things that are counted, not paid; a row label needs a stronger cue than a table. "#" in a label's
# parenthesis is the filing saying "a number of": "Retail bank branches (#, period-end)" (WFC p26, read as
# $4.09 billion before), "Active digital banking users (in thousands)"
ROW_COUNT_RE = re.compile(r"\bnumber of\b|\bcounts?\b|headcount|\bemployees\b|direct staff|\bstaff \(|\(actual\)|"
                          r"square feet|\bmembers\b|cardholders|\bprocessed transactions\b|\(\s*#\s*[,)]|"
                          r"\b(?:banking|mobile|digital|active) users\b|"
                          r"\bwholesale units\b|\bunits (?:sold|shipped|delivered|produced)\b", re.I)    # F p53
# a table titled with money words holds money even when it prints no dollar sign
MONEY_TITLE_RE = re.compile(r"expenditure|capital|revenue|\bcosts?\b|expense|income|earnings|cash|dollars|price|fees|"
                            r"compensation|amounts?\b", re.I)
PRICE_COL_RE = re.compile(r"price|fair value|per share|grant date", re.I)
SHARES_TABLE_RE =re.compile(r"\bshares\b|common stock|stock units|\bRSUs?\b|stock options|options outstanding", re.I)
MONEY_WORD_RE = re.compile(r"sales|revenue|cost|expense|income|earnings|asset|liabilit|fee|price|value|amount|capital|"
                           r"cash|payment|compensation|\bper\b|debt|loss|gain|tax|dividend|equity|spend|investment|"
                           r"wage|salar(?:y|ies)\b|benefit|obligation|deposit|loan|rent", re.I)


# ------------------------------------------------ scales and units stated in a row label (buildfix 2)
SCALE_NAMES = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6, "billion": 1e9, "billions": 1e9,
               "trillion": 1e12, "trillions": 1e12}
PAREN_RE = re.compile(r"\(([^()]{1,60})\)")
PAREN_SCALE_RE = re.compile(r"\b(thousands|millions|billions|trillions)\b|\b000['’]?s\b|^\s*000\s*$", re.I)    # "(000)"
PAREN_DOLLARS_RE = re.compile(r"^\s*(?:in\s+)?(?:US\s*)?(?:\$|dollars?|usd)\s*$", re.I)
NOT_SCALE_RE = re.compile(r"%|percent|\bper\b|/\s*share|\bbps\b|basis point|\brate\b|\bratio\b", re.I)
DOLLAR_MM_RE = re.compile(r"\$\s?(MM|BN|B|K)\b")             # Chevron's "Earnings $MM"
MONEY_SCALE_RE = re.compile(r"(?:\$|dollars?|usd)\s*(?:amounts\s+)?(?:in\s+)?(?:thousands|millions|billions)|"
                            r"(?:thousands|millions|billions)\s+of\s+(?:u\.?s\.?\s+)?(?:\$|dollars|usd)", re.I)
DOLLAR_WORD_RE = re.compile(r"\$|\bdollars?\b|\bUSD\b", re.I)
# a money line states what it is: these words make a scaled row USD when it prints no dollar sign
MONEY_LABEL_RE = re.compile(MONEY_WORD_RE.pattern + r"|balance|volumes?\b|\baum\b|assets under|spend|billed|"
                            r"premiums?|commitments?|exposure|notional|borrowings|purchases|proceeds", re.I)

# physical units: (pattern, what it measures, its built-in scale). The oil and gas abbreviations
# carry their scale in the prefix: M = thousand, MM = million, B = billion, T = trillion.
_ABBREV = [
    (r"MBOE/?D|MBOED", "barrels of oil-equivalent per day", 1e3), (r"MMBOE/?D", "barrels of oil-equivalent per day", 1e6),
    (r"MMBOE", "barrels of oil-equivalent", 1e6), (r"MBOE", "barrels of oil-equivalent", 1e3),
    (r"BOE/?D|BOED", "barrels of oil-equivalent per day", 1.0), (r"BOE", "barrels of oil-equivalent", 1.0),
    (r"MBD|MB/D|KBD|KB/D|MBBLS?/D|MBPD", "barrels per day", 1e3), (r"MMBBLS?/D", "barrels per day", 1e6),
    (r"MMBBLS?", "barrels", 1e6), (r"MBBLS?", "barrels", 1e3), (r"BBLS?/D|BOPD|BPD", "barrels per day", 1.0),
    (r"BBLS?", "barrels", 1.0),
    (r"MMCFE?/?D|MMCFE?D", "cubic feet per day", 1e6), (r"BCFE?/?D", "cubic feet per day", 1e9),
    (r"MCFE?/?D|MCFE?D", "cubic feet per day", 1e3), (r"MMCFE?", "cubic feet", 1e6), (r"BCFE?", "cubic feet", 1e9),
    (r"TCFE?", "cubic feet", 1e12), (r"MCFE?", "cubic feet", 1e3),
]
ABBREV_RES = [(re.compile(rf"(?<![\w$/]){p}(?![\w/])", re.I), m, s) for p, m, s in _ABBREV]
PHYS_NOUNS = (r"barrels?(?: of oil[- ]equivalent)?(?: (?:per day|daily|a day))?|cubic (?:feet|meters)(?: (?:per day|daily))?|"
              r"megawatts?(?:[- ]hours?)?|gigawatts?(?:[- ]hours?)?|kilowatt[- ]hours?|(?:metric |short )?tons|tonnes|"
              r"square (?:feet|footage)|sq\.? ?ft\.?|vehicles|deliveries|stores|warehouses|restaurants|employees|"
              r"headcount|cards[- ]in[- ]force|cards|cardholders|members|subscribers|branches|ATMs|wells|"
              r"(?:net |gross )?acres|aircraft|units sold|units delivered|housing units|processed transactions|"
              r"transactions|accounts|customers|locations|clubs|pounds|ounces|gallons|passengers|rigs")
PHYS_RE = re.compile(rf"\b(?:(?P<scale>thousands|millions|billions|trillions|thousand|million|billion)\s+(?:of\s+)?"
                     rf"(?:[a-z-]+\s+){{0,2}}?)?(?P<noun>{PHYS_NOUNS})\b", re.I)
# nouns that are a physical unit wherever they are printed; the others ("accounts", "transactions",
# "customers", "members") count only when the row label itself says so, because "accounts payable"
# or "transactions with related parties" are money
STRONG_NOUN_RE = re.compile(r"barrel|cubic|watt|tons|tonnes|square|sq\.? ?ft|deliveries|employees|headcount|"
                            r"cards[- ]in[- ]force|cardholders|acres|units sold|units delivered|housing units|"
                            r"processed transactions|gallons|ounces|rigs|passengers", re.I)
SCALED_ABBREV_RE = re.compile(r"\b(thousands|millions|billions|thousand|million|billion)\s+(?:of\s+)?"
                              r"([A-Za-z]{2,6}(?:/D)?)\b", re.I)
WEAK_NOUN_BLOCK_RE = re.compile(r"accounts? (?:payable|receivable)|receivables?|payables?|"
                                r"transactions? (?:costs?|expenses?|with|related|gains?|losses?)|"
                                r"customers? (?:deposits?|relationships?|contracts?|accounts?|liabilit)", re.I)
PER_PHYS_RE = re.compile(rf"\bper\s+(?:[a-z]+\s+)?(?:barrel|bbl|boe|mcf|cubic|unit|ton|tonne|square foot|sq\.? ?ft|"
                         rf"vehicle|store|employee|card|member|account|customer|gallon|ounce|megawatt|passenger|"
                         rf"day of production)\b|\$\s*/|/\s*(?:bbl|boe|mcf|mmbtu|ton|unit)\b", re.I)


def label_scale(label: str) -> Optional[tuple]:
    """(scale, dollars stated) that a row label prints for its own row: "Cards-in-force (millions)",
    "Network volumes (billions)", "($ in billions)", "Average spending (dollars)", "Earnings $MM",
    "Direct staff (in thousands)". None when the label states no scale or the parenthesis is a rate
    ("(% of sales)", "(per share)")."""
    for m in PAREN_RE.finditer(label or ""):
        inner = m.group(1)
        if NOT_SCALE_RE.search(inner):
            continue
        dollars = bool(DOLLAR_WORD_RE.search(inner))
        s = PAREN_SCALE_RE.search(inner)
        if s:
            return (SCALE_NAMES[s.group(1).lower()] if s.group(1) else 1e3), dollars
        if PAREN_DOLLARS_RE.match(inner):
            return 1.0, True
    m = DOLLAR_MM_RE.search(label or "")
    if m:
        return {"MM": 1e6, "BN": 1e9, "B": 1e9, "K": 1e3}[m.group(1).upper()], True
    s = _scale_of(label)
    if s and not NOT_SCALE_RE.search(label):
        return s, bool(DOLLAR_WORD_RE.search(label))
    return None


def physical(text: str, strong_only: bool = False) -> Optional[tuple]:
    """(what is measured, scale or None) when the text states a physical unit: "MBOED" ->
    ("barrels of oil-equivalent per day", 1e3), "Millions of barrels" -> ("barrels", 1e6),
    "Natural Gas - Billions of cubic feet" -> ("cubic feet", 1e9), "Stores" -> ("stores", None).
    A price per unit ("$/BBL", "per barrel") is not a quantity and returns None."""
    text = text or ""
    if PER_PHYS_RE.search(text):
        return None
    for m in SCALED_ABBREV_RE.finditer(text):       # "(Millions of BOE)"
        for rx, measure, scale in ABBREV_RES:
            if rx.fullmatch(m.group(2)):
                return measure, SCALE_NAMES[m.group(1).lower()] * scale
    for rx, measure, scale in ABBREV_RES:
        if rx.search(text):
            return measure, scale
    for m in PHYS_RE.finditer(text):
        noun = m.group("noun")
        if strong_only and not STRONG_NOUN_RE.search(noun):
            continue
        if WEAK_NOUN_BLOCK_RE.search(text[m.start("noun"):m.start("noun") + 40]) or \
                re.search(r"(?:stock|share|restricted|performance|partnership|LP|common)\s+$", text[:m.start("noun")], re.I):
            continue
        scale = SCALE_NAMES[m.group("scale").lower()] if m.group("scale") else None
        return re.sub(r"\s+", " ", noun.lower()), scale
    return None


def counted_nouns(label: str) -> set:
    """The counted nouns a label names, singular: "Proprietary basic cards-in-force" -> {"card"}."""
    out = set()
    for m in PHYS_RE.finditer(label or ""):
        w = re.split(r"[- ]", m.group("noun").lower())[0]
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _scale_of(text: str) -> Optional[float]:
    m = SCALE_WORD_RE.search(text or "")
    if not m:
        return None
    return SCALES[(m.group(1) or m.group(2) or m.group(3)).lower().rstrip("s")]


def _full(value: float, scale: float):
    """value * scale, as an int when it is a whole number."""
    v = value * scale
    return int(round(v)) if abs(v - round(v)) < 1e-6 * max(1.0, abs(v)) else round(v, 6)


class _Profile:
    """What the page says about the units of one table."""

    def __init__(self, t):
        cells = [c for r in t.rows for c in r.cells.values()]
        text = " ".join([t.title or "", getattr(t, "intro", "") or "", t.caption or "", t.header_text or ""])
        self.dollar = any(c.has_dollar or "$" in c.text for c in cells) or bool(DOLLAR_RE.search(text))
        names = [r.label + " " + r.section for r in t.rows]
        counted = sum(1 for n in names if COUNT_RE.search(n))
        heads = text + " " + " ".join(c.header for c in t.columns)
        self.count = not self.dollar and (bool(COUNT_RE.search(heads)) or counted >= max(1, 0.3 * len(names)))
        # "The following number of shares ...": a table of share counts
        self.shares = self.count and bool(re.search(r"\bshares\b", heads, re.I)) and \
            not COUNT_RE.search(re.sub(r"number of", " ", heads, flags=re.I))
        # a scale printed in a column header (or a caption over some columns) belongs to those columns only
        self.col_scale = {c.idx: getattr(c, "scale", None) or _scale_of(c.header) for c in t.columns
                          if getattr(c, "scale", None) or _scale_of(c.header)}
        own = t.caption + " " + t.header_text + " " + (getattr(t, "intro", "") or "")
        for c in t.columns:
            if c.idx in self.col_scale:
                for piece in (c.header.split("|")[-1].strip(), getattr(c, "scale_text", "")):
                    if piece:
                        own = own.replace(piece, " ")
        m = EXCEPT_RE.search(own + " " + (t.title or ""))
        self.exception = (m.group(1).strip(), SCALES[m.group(2).lower().rstrip("s")]) if m else None
        rest = EXCEPT_RE.sub(" ", own)
        self.own_scale = _scale_of(rest)          # the table's own scale word, outside any exception clause
        self.money_scale = t.scale
        if self.exception and getattr(t, "scale_source", "") != "inherited" and self.own_scale:
            self.money_scale = self.own_scale     # build_table read the exception's scale as the table's
        self.share_scale = t.share_scale
        # the scale a printed caption gives every number of the table, counts included: "(Millions,
        # except percentages and where indicated)"; "Millions of dollars" is said of money only
        cap = (t.caption or "") + " " + (t.header_text or "")
        self.count_scale = t.scale if getattr(t, "scale_source", "") in ("caption", "intro") and \
            not MONEY_SCALE_RE.search(cap + " " + (getattr(t, "intro", "") or "")) else None
        # "(in millions, except employees)": a whole count is one of the exceptions
        self.scale_except = bool(re.search(r"\bexcept", cap, re.I))
        # an award or option activity table: a column of share counts ("Shares", "Number of Options", "RSUs")
        # printed without "$" beside prices ("$ 331.62"), remaining lives and intrinsic values (CAT p74, JNJ
        # p87, XOM p113, BLK p115). The dollar signs of the price column made every cell USD in the money scale
        # (299,523 options read as $299.5 billion)
        # A column counts shares when its header says so ("Shares", "Treasury Common Shares", BLK p115), or, in a
        # table about shares, units or options that also prints a price column, when it prints no "$" and its
        # header names no value (XOM p113: "Shares (thousands)" read as the column "2025")
        self.share_cols = set()
        if AWARD_TABLES and not self.shares:
            about = SHARE_COL_RE.search(" ".join([t.title or "", t.caption or "", t.header_text or "",
                                                  getattr(t, "intro", "") or ""]))
            priced = any(PRICE_COL_RE.search(c.header or "") for c in t.columns)
            for c in t.columns:
                h = c.header or ""
                if VALUE_COL_RE.search(h) or not (SHARE_COL_RE.search(h) or about and priced):
                    continue
                vals = [r.cells[c.idx] for r in t.rows if c.idx in r.cells and r.cells[c.idx].value is not None]
                if vals and not any(v.has_dollar or "$" in v.text or v.is_percent for v in vals):
                    self.share_cols.add(c.idx)
        # a share count takes a scale only when the page says it of shares ("(Shares in Thousands)", "Shares
        # (thousands)"); a money caption ("(Dollars in Millions)", "Amounts are in millions of dollars") is not
        # said about them, nor is the money scale a table inherits from the statements
        words = " ".join([t.title or "", t.caption or "", t.header_text or "", getattr(t, "intro", "") or ""])
        m = SHARE_SCALE_RE.search(words)
        if m:                                   # "(Shares in Thousands)", "Shares (thousands)"
            self.award_share_scale = SCALES[next(g for g in m.groups() if g).lower().rstrip("s")]
        elif getattr(t, "scale_source", "") in ("caption", "intro") and not MONEY_SCALE_RE.search(words):
            self.award_share_scale = t.share_scale      # "(in millions, except per share data)" said of everything
        else:
            self.award_share_scale = 1.0                # a money-only or an inherited scale says nothing of shares


# OFF in handback 4 (23 Nov 00:35): the red team (train_lab/REDTEAM_B5.md, findings 1 to 11) found the share scale lost
# or wrong wherever it comes from a footnote ("(a) In thousands."), from "(in millions)" in a statement of equity or from
# a mixed "(Thousands) ... (Millions)" header (PFE p98 19,621 thousand options read as 19.6 billion), and the linker
# serves a row in the row's unit. A plausible wrong number is worse than the old absurd one, so the rule waits for
# branch build-next, with those findings as its tests.
AWARD_TABLES = False
SHARE_COL_RE = re.compile(r"\bshares\b|\bnumber of (?:shares|options|units|awards|rsus|psus|prsus)\b|\boptions\b|"
                          r"\brsus?\b|\bpsus?\b|\bprsus?\b|\bunits\b|\bawards\b", re.I)
VALUE_COL_RE = re.compile(r"price|value|\$|dollars|per share|intrinsic|fair|cost|amount|\blife\b|years|\bterm\b", re.I)
SHARE_SCALE_RE = re.compile(r"\(\s*shares\s+in\s+(thousands|millions)\s*\)|shares\s+in\s+(thousands|millions)|"
                            r"(thousands|millions)\s+of\s+shares|shares\W+(?:[\w-]+\W+){0,2}\(\s*(thousands|millions)\s*\)|"
                            # a stacked header: "Options Price Term Value" over "(in millions) (in years) (in millions)",
                            # "Units Fair Value Value" over "(in millions) (in millions)" (MA p97, p98)
                            r"\b(?:options|units|awards|rsus|psus|prsus)\b[^()]{0,40}?\(\s*(?:in\s+)?(thousands|millions)\s*\)",
                            re.I)
AGGREGATE_COL_RE = re.compile(r"aggregate|intrinsic|total (?:fair )?value|in millions|in thousands|in billions", re.I)
LIFE_COL_RE = re.compile(r"\blife\b|\(years\)|\bin years\b|\bterm\b", re.I)


def _award_cell(t, ci: int, cell, prof: _Profile):
    """(unit, scale) of one cell of a row that holds a share count, or None for the share count itself."""
    own = t.columns[ci].header or ""
    if ci in prof.share_cols:
        return None
    if LIFE_COL_RE.search(own) and not cell.has_dollar:
        return "ratio", 1.0                                     # "Weighted-Average Remaining Contractual Life (Years)"
    dollar = cell.has_dollar or "$" in cell.text
    if PRICE_COL_RE.search(own) or dollar and "." in cell.text and not AGGREGATE_COL_RE.search(own):
        return "USD/share", 1.0                                 # "Weighted-Average Exercise Price", "$ 331.62"
    # an amount: "Aggregate Intrinsic Value $ 1,063", or the "Amount" beside "Shares" in a statement of equity
    # ("$ 1,031" with no cents is money, not a price: V p64)
    return "USD", prof.col_scale.get(ci, prof.money_scale)


def _award_row(t, r, prof: _Profile) -> bool:
    """A row of an option or award table: it holds a share count, and every other figure is a price, a remaining
    life or an aggregate value. A row with a plain amount beside its shares (a statement of equity: "Shares |
    Amount", GOOGL Q3 p9, V p64) is not one: there the share count is marked on its own cell and the row keeps
    its money unit, so the amounts keep theirs."""
    if not prof.share_cols or MONEY_WORD_RE.search(r.label) or PER_SHARE_RE.search(r.label):
        return False
    share = [ci for ci in r.cells if ci in prof.share_cols and r.cells[ci].value is not None]
    if not share:
        return False
    for ci, cell in r.cells.items():
        if ci in prof.share_cols or cell.value is None:
            continue
        kind = _award_cell(t, ci, cell, prof)[0]
        if kind == "USD" and not AGGREGATE_COL_RE.search(t.columns[ci].header or ""):
            return False
    return True


def row_unit(t, r, prof: _Profile) -> str:
    """One of UNITS for a table row, from its own printing and its table's evidence."""
    cells = list(r.cells.values())
    if _award_row(t, r, prof):
        return "shares"                                         # a row of an award table: its count is shares
    pool = row_pool(t, r)
    row_dollar = any(c.has_dollar or "$" in c.text for c in cells)     # "$90.56" glued is a dollar sign too
    counted = prof.count or (ROW_COUNT_RE.search(r.label) and not MONEY_WORD_RE.search(r.label) and not row_dollar)
    amount = "shares" if prof.shares else ("count" if counted and not row_dollar else "USD")
    if pool == "shares":
        return "shares"
    if pool == "money":
        return amount
    if cells and sum(1 for c in cells if c.is_percent) * 2 >= len(cells):
        return "percent"
    if PER_SHARE_RE.search(r.label) or PER_SHARE_RE.search(row_section(t, r)):
        return "USD/share"
    if RATIO_LABEL_RE.search(r.label):
        return "ratio"
    # printed decimals that are amounts: next to whole amounts in the same row ("13,427 ... (1.0)" with the
    # variance columns in billions), or in a table or row whose own caption gives a scale ("72.4" in billions)
    mixed = any("." not in c.text and not c.is_percent and c.value for c in cells)
    if mixed or _row_scale(t, r, prof) or prof.col_scale or \
            (prof.money_scale >= 1e9 and getattr(t, "scale_source", "") != "inherited"):
        return amount
    if row_dollar:
        return "USD/share"
    return "ratio"


def _row_scale(t, r, prof: _Profile) -> Optional[float]:
    """A scale that belongs to this row alone: "Direct staff (in thousands)", or the
    exception clause of the caption naming it."""
    s = _scale_of(r.label)
    if s:
        return s
    if prof.exception:
        want = set(tokens(prof.exception[0]))
        if want and want <= set(tokens(r.label)):
            return prof.exception[1]
    return None


def _cell_scale(t, r, ci: int, unit: str, prof: _Profile) -> float:
    own = _row_scale(t, r, prof)
    if own:
        return own
    if unit == "count" or unit == "shares" and prof.shares:
        if ci in prof.col_scale:
            return prof.col_scale[ci]
        # a one-line table is all count: its caption ("in millions") is said about that line
        return (prof.own_scale or 1.0) if prof.count or len(t.rows) == 1 else 1.0
    if unit == "shares":
        return prof.share_scale
    if unit == "USD":
        return prof.col_scale.get(ci, prof.money_scale)
    return 1.0


def _fixed_row(t, r) -> bool:
    """A row whose unit no label scale may change: printed percentages, per-share amounts,
    share counts, ratios and rates, prices per physical unit ("$/BBL", "per barrel")."""
    cells = list(r.cells.values())
    if cells and sum(1 for c in cells if c.is_percent) * 2 >= len(cells):
        return True
    if PER_SHARE_RE.search(r.label) or PER_SHARE_RE.search(row_section(t, r)) or row_pool(t, r) == "shares":
        return True
    if PER_PHYS_RE.search(r.label) or re.search(r"\(actual\)", r.label, re.I):
        return True                                     # "Branches (actual)" says it is not scaled
    ph = physical(r.label)
    return bool(RATIO_LABEL_RE.search(r.label)) and not (ph and ph[1] is not None)


def _label_cue(label: str, row_dollar: bool, unit0: str) -> tuple:
    """(unit the label's words state: "USD", "count" or None, physical measure or None).
    A physical unit with its own scale ("MBOED", "millions of barrels") beats money words
    ("Refined Product Sales MBD"); a bare counted noun does not ("Sales at company-operated
    restaurants" is money). A noun that is often money ("Checking accounts", "Deferred
    exploratory wells", "Municipal bond vehicles") counts only when the label also states a
    scale or "number of", or the row was not read as money."""
    ph = physical(label)
    if ph and ph[1] is None and not STRONG_NOUN_RE.search(ph[0]) and unit0 == "USD" and \
            not label_scale(label) and not re.search(r"\bnumber of\b", label, re.I):
        ph = None
    money = row_dollar or bool(DOLLAR_WORD_RE.search(label)) or bool(MONEY_LABEL_RE.search(label))
    if ph and not row_dollar and (ph[1] is not None or not money):
        return "count", ph
    if money:
        return "USD", None
    if ROW_COUNT_RE.search(label) or COUNT_RE.search(label):
        return "count", None
    return None, None


def unit_plans(t, prof: _Profile, base: dict) -> dict:
    """Row idx -> {"unit", "scale", "measure", "from"} for rows whose unit or scale a label states.

    1. A scale in the row label ("Cards-in-force (millions)", "Network volumes (billions)",
       "Average spending (dollars)", "Earnings $MM") sets that row's scale. Its unit is USD when
       the label states dollars or the row prints $ or names money; count when it names a counted
       or physical noun; otherwise the row keeps USD if it already was USD, else count.
    2. That scale carries to the following rows that state no unit of their own ("Proprietary
       cards-in-force" under "Cards-in-force (millions)") until the group ends: another stated
       unit, a fixed row (percent, per share, ratio), another section, an outdent, or a total
       (which still takes it). A USD row already scaled by a scale the table prints is not touched.
    3. A group header (the section) with a scale or physical unit does the same for its rows
       ("Natural Gas - Billions of cubic feet").
    4. A physical unit (barrels, MBOED, cubic feet, vehicles, stores ...) makes a row a count,
       never USD and never multiplied by the money scale; it takes the scale attached to the unit
       ("thousands of barrels per day" x1,000), else a count's own scale (old path).
    """
    plans: dict = {}
    if prof.shares:
        return plans
    cell_dollar = any(c.has_dollar or "$" in c.text for r in t.rows for c in r.cells.values())
    cap = " ".join([t.title or "", t.caption or "", t.header_text or ""])
    caps = ({physical(m.group(0), strong_only=True) for m in PHYS_RE.finditer(cap)} |
            {physical(m.group(0)) for rx, _, _ in ABBREV_RES for m in rx.finditer(cap)})
    caps.discard(None)
    # one physical unit for the whole table; a caption naming two ("thousands of barrels daily",
    # "millions of metric tons per year") is said of different columns and is left to them
    cap_phys = physical(cap, strong_only=True) if not cell_dollar and len(caps) == 1 else None
    carrier = None                                  # (row, plan) whose scale the next rows may take
    for r in t.rows:
        # a dollar sign glued to the figure ("$90.56") is a dollar sign too: without it "Average ticket $90.56"
        # took the "(in millions)" count scale of "Customer transactions" above it (HD p37, red team 12)
        row_dollar = any(c.has_dollar or "$" in c.text for c in r.cells.values())
        if carrier is not None:
            cr = carrier[0]
            if r.section != cr.section or r.indent < cr.indent - 3:
                carrier = None
        if _fixed_row(t, r):
            carrier = None
            continue
        unit0 = base[r.idx]
        cue, ph = _label_cue(r.label, row_dollar, unit0)
        own = label_scale(r.label)
        if own:
            if own[1] or row_dollar or cue == "USD":
                unit = "USD"
            elif cue == "count":
                unit = "count"
            else:
                unit = unit0 if unit0 in ("USD", "count") else "count"
            scale = ph[1] if unit == "count" and ph and ph[1] is not None else own[0]
            if unit == "USD" and own == (1.0, True) and PRICE_COL_RE.search(r.label):
                unit = "USD/share"                      # "Grant price (dollars)"
            plans[r.idx] = {"unit": unit, "scale": scale, "measure": ph[0] if ph and unit == "count" else None,
                            "from": "row label"}
            carrier = (r, plans[r.idx])
            continue
        if ph and ph[1] is not None:
            plans[r.idx] = {"unit": "count", "scale": ph[1], "measure": ph[0], "from": "row label"}
            carrier = (r, plans[r.idx])
            continue
        if carrier is not None and _carries(r, cue, row_dollar, unit0, carrier):
            cp = carrier[1]
            plans[r.idx] = {"unit": cp["unit"], "scale": cp["scale"],
                            "measure": (ph[0] if ph else cp["measure"]) if cp["unit"] == "count" else None,
                            "from": f"carried from {carrier[0].label[:60]!r}"}
            if r.label.lower().startswith("total"):
                carrier = None
            continue
        carrier = None
        if re.search(r"\bprices?\b", r.label, re.I):
            continue                                    # a price beside the counts ("Price 3.46") is no count
        sec = row_section(t, r)
        sec_own, sec_ph = label_scale(sec), physical(sec, strong_only=True)
        # a group header's physical unit makes a count of a row read as money, or one it gives a scale;
        # a decimal under "... employees" with no scale is a share of them, not a number of them
        if sec_ph and cue != "USD" and not row_dollar and (unit0 == "USD" or sec_ph[1] is not None or sec_own):
            plans[r.idx] = {"unit": "count", "scale": sec_ph[1] if sec_ph[1] is not None else
                            (sec_own[0] if sec_own else None), "measure": sec_ph[0], "from": f"group header {sec[:60]!r}"}
            continue
        if sec_own and not (unit0 == "USD" and cue != "count"):    # money rows keep the table's scale
            unit = "USD" if sec_own[1] or row_dollar or cue == "USD" else "count"
            plans[r.idx] = {"unit": unit, "scale": sec_own[0], "measure": ph[0] if ph and unit == "count" else None,
                            "from": f"group header {sec[:60]!r}"}
            continue
        # a counted noun with no scale of its own: a row read as money becomes a count at the count's
        # own scale; a decimal count ("Stores 244.7") takes the caption's scale when it is not a money scale
        decimal = any("." in c.text for c in r.cells.values())
        for found, where in ((ph if cue == "count" else None, "row label"),
                             (cap_phys if cue != "USD" and not row_dollar else None, "table caption")):
            if not found:
                continue
            if found[1] is not None:
                plans[r.idx] = {"unit": "count", "scale": found[1], "measure": found[0], "from": where}
            elif unit0 == "USD":
                plans[r.idx] = {"unit": "count", "scale": None, "measure": found[0], "from": where}
            elif prof.count_scale and (decimal or not prof.scale_except):
                measure = cap_phys[0] if cap_phys and not STRONG_NOUN_RE.search(found[0]) else found[0]
                plans[r.idx] = {"unit": "count", "scale": prof.count_scale, "measure": measure, "from": where}
            break
    for plan in plans.values():
        if plan["measure"] and not STRONG_NOUN_RE.search(plan["measure"]):
            plan["measure"] = None                      # "accounts", "locations": a count, not a unit
    return plans


def _carries(r, cue, row_dollar: bool, unit0: str, carrier: tuple) -> bool:
    """Whether a row that states no scale takes the scale of the labelled row above it."""
    cr, cp = carrier
    if re.search(r"\bprices?\b", r.label, re.I):
        return False
    if cp["unit"] == "count":
        if cue == "USD" or row_dollar or unit0 == "USD" and cue != "count":
            return False
        mine, theirs = counted_nouns(r.label), counted_nouns(cr.label)
        return not mine or bool(mine & theirs)
    # a row already read as money keeps the table's money scale ("Total segment assets" under
    # "Average spending (dollars)"); only rows the table scale did not reach take the label's
    return cue != "count" and unit0 != "USD"


def table_rows(doc: str, ticker: str, k: int, t, periods: list, anchors: dict, cal=None, form: str = "",
               filing_end: Optional[str] = None) -> list:
    """Rows of one parsed table in the raw-layer shape. With the company's calendar, a date line
    inside the table dates the rows below it (_marked_periods). When two or more cells of one row
    share a period id (segment, region, gross and net columns), each of them also carries `member`:
    its column header without the date words, so a reader can tell the cells apart."""
    checks = discover(t)
    ok = verified_cells(t, checks)
    passing = [ch for ch in checks if ch.status != "fail"]
    totals = {ch.total for ch in passing}
    parent = {row: ch.total for ch in passing for row, _ in ch.operands}
    children = {}
    for ch in passing:
        children.setdefault(ch.total, ch.operands)
    prof = _Profile(t)
    table_id = f"{doc}:p{t.page}:x{k}"
    out = {}
    base = {r.idx: row_unit(t, r, prof) for r in t.rows}
    plans = unit_plans(t, prof, base)
    for r in t.rows:
        plan = plans.get(r.idx)
        unit = plan["unit"] if plan else base[r.idx]
        cells, ties = [], defaultdict(int)
        row_periods = _marked_periods(cal, t, r, periods, form, filing_end) if cal is not None else periods
        for ci in sorted(r.cells):
            cell = r.cells[ci]
            if cell.value is None:
                continue
            pid, end = row_periods[ci]
            cunit = "percent" if cell.is_percent else unit
            if unit in ("count", "shares") and (cell.has_dollar or "." in cell.text and
                                                PRICE_COL_RE.search(t.columns[ci].header + " " + t.header_text)):
                cunit = "USD/share"                     # "Weighted-average grant date fair value" beside a count
            award = unit == "shares" and bool(prof.share_cols) and not cell.is_percent
            if award:
                how = _award_cell(t, ci, cell, prof)
                if how is None:                         # the share count: the scale the page prints, else none
                    cunit, value = "shares", _full(cell.value, prof.col_scale.get(ci) or prof.award_share_scale)
                else:
                    cunit = how[0]
                    value = _full(cell.value, how[1]) if cunit == "USD" else cell.value
            elif ci in prof.share_cols and unit == "USD" and not cell.is_percent and not cell.has_dollar:
                # the share count of a statement of equity ("Shares | Amount"): its own cell, its own scale
                cunit, value = "shares", _full(cell.value, prof.col_scale.get(ci) or prof.award_share_scale)
            elif plan and cunit == plan["unit"] and plan["scale"] is not None:
                value = _full(cell.value, plan["scale"])     # the scale the label (or its group) states
            elif plan and cunit == plan["unit"] == "count":
                value = _full(cell.value, _cell_scale(t, r, ci, "count", prof))     # never the money scale
            elif cunit in ("USD", "count", "shares"):
                value = _full(cell.value, _cell_scale(t, r, ci, cunit, prof))
            else:
                value = cell.value
            c = {"pid": pid, "end": end, "value": value, "text": cell.text, "ok": (r.idx, ci) in ok,
                 "col": t.columns[ci].header[:60], "ci": ci, "bbox": [round(x, 1) for x in cell.bbox]}
            if cunit != unit:
                c["unit"] = cunit
            cells.append(c)
            if cunit == "USD" and end and (end, value) in anchors:
                ties[anchors[(end, value)]] += 1
        if not cells:
            continue
        shared = Counter(c["pid"] for c in cells if c["pid"])
        for c in cells:
            if c["pid"] and shared[c["pid"]] > 1:
                c["member"] = member_of(t.columns[c["ci"]].header)
        out[r.idx] = {
            "id": f"{table_id}:r{r.idx}", "doc": doc, "ticker": ticker, "page": r.page, "table": k,
            "table_id": table_id, "title": (t.title or "")[:160], "context": (getattr(t, "context", "") or "")[:120],
            "label": r.label, "section": r.section, "unit": unit,
            "role": "total" if r.idx in totals else ("operand" if r.idx in parent else "loose"),
            "verified": any(c["ok"] for c in cells),
            "ties_to": max(ties, key=ties.get) if ties else None, "anchor": None,
            "parent": None, "cells": cells}
        if plan:
            out[r.idx]["unit_from"] = plan["from"]
            if plan["measure"]:
                out[r.idx]["measure"] = plan["measure"]
    for idx, row in out.items():
        if idx in parent and parent[idx] in out:
            row["parent"] = out[parent[idx]]["id"]
        if idx in children:
            ops = [(i, s) for i, s in children[idx] if i in out]
            row["children"] = [out[i]["id"] for i, _ in ops]
            row["signs"] = [s for _, s in ops]
        # an operand inherits the concept its total (or its total's total) ties to
        p, hops = parent.get(idx), 0
        while p is not None and hops < 4:
            if p in out and out[p]["ties_to"]:
                row["anchor"] = out[p]["ties_to"]
                break
            p, hops = parent.get(p), hops + 1
    return list(out.values())


def build_raw(g, docs: Optional[list] = None, verbose: bool = True) -> list:
    """Raw rows of every readable filing, or only of `docs`."""
    rows = []
    for doc, f in g.filings.items():
        if docs is not None and doc not in docs:
            continue
        if not f.get("readable", True):
            continue
        ticker = f["ticker"]
        cal = g.calendars[ticker]
        pages = pdfio.load_pages(doc)
        skip = {p for t in g.tables.values() if t["doc"] == doc for p in t.get("pages", [t["page"]])}
        # values of bound facts in this filing, to recognise a table that restates one of them
        anchors = {}
        for fact in g.facts.values():
            if fact.ticker != ticker:
                continue
            for rid in fact.readings:
                r = g.readings.get(rid)
                if r is not None and r.doc == doc and r.unit == "USD" and abs(r.value) >= 1_000_000:
                    anchors[(r.period_end, r.value)] = fact.concept
        # a table that prints no unit of its own is read in the unit of the filing's statements
        scales = Counter(t["scale"] for t in g.tables.values() if t["doc"] == doc and t.get("scale"))
        default = scales.most_common(1)[0][0] if scales else None
        n0 = len(rows)
        fiscal_year = int(f["fiscal_year"].lstrip("FY"))
        own = defaultdict(set)                   # printed texts of the primary statements, by page
        for r in g.readings.values():
            if r.doc == doc:
                own[r.page].add(r.text)
        base, extra = 0, 0
        for t in find_all_tables(doc, pages, skip_pages=skip, default_scale=default, statement_cells=own):
            # tables the older reader also found keep their number; a one-line table or a table on a
            # statement page is numbered after the table before it ("12e1"), so no older row id moves
            if getattr(t, "added_by", ""):
                extra += 1
                k = f"{base - 1}e{extra}"
            else:
                k, base, extra = base, base + 1, 0
            buckets = _maturity_columns(t, fiscal_year)
            quarterly = _names_quarters(t)
            periods = [(None, None) if c.idx in buckets or _comparison(c.header)
                       else _period(cal, c, f["form"], f["period_end"], _heading(t), quarterly) for c in t.columns]
            rows.extend(table_rows(doc, ticker, k, t, periods, anchors, cal, f["form"], f["period_end"]))
        rows.extend(_primary_rows(g, doc))
        if verbose:
            print(f"raw    {doc:22s} {len(rows) - n0:5d} rows", flush=True)
    return rows


def _primary_rows(g, doc: str) -> list:
    """Rows of the three primary statements, in the same shape, so that any printed line
    (not only the seed concepts) can be bound on demand. Cells point at the readings the
    base graph already holds."""
    grouped: dict = {}
    for r in g.readings.values():
        if r.doc == doc and r.kind in ("IS", "BS", "CF"):
            grouped.setdefault((r.table, r.row), []).append(r)
    bound = {(b.kind, b.row): b.concept for b in g.bindings if b.doc == doc and b.state == "active"}
    # formula structure from the graph's SUM checks
    key = {r.id: (r.table, r.row) for rs in grouped.values() for r in rs}
    parent, children = {}, {}
    for ch in g.checks.values():
        if ch.type != "SUM" or ch.status == "fail" or ch.doc != doc:
            continue
        tot = [key[m[0]] for m in ch.members if m[1] == "total" and m[0] in key]
        ops = []
        for rid, sign in ch.members:
            if sign != "total" and rid in key and (key[rid], sign) not in ops:
                ops.append((key[rid], sign))
        if tot:
            children.setdefault(tot[0], ops)
            for o, _ in ops:
                parent.setdefault(o, tot[0])
    rid_of = lambda k: f"{k[0]}:r{k[1]}"
    out = []
    for (table, row), rs in grouped.items():
        rs.sort(key=lambda r: r.col)
        roles = set()
        for r in rs:
            for cid in r.checks:
                ch = g.checks.get(cid)
                if ch is not None and ch.type == "SUM" and ch.status != "fail":
                    roles |= {"total" if role == "total" else "operand" for rid, role in ch.members if rid == r.id}
        first = rs[0]
        cols = g.tables[table].get("columns", [])
        rec = {
            "id": rid_of((table, row)), "doc": doc, "ticker": first.ticker, "page": first.page, "table": first.kind,
            "table_id": table, "title": g.tables[table]["title"][:160], "context": "primary financial statement",
            "label": first.label, "section": first.section,
            "unit": first.unit if first.unit in UNITS else "ratio",
            "role": "total" if "total" in roles else ("operand" if roles else "loose"),
            "verified": any(r.status == "verified" for r in rs),
            "ties_to": bound.get((first.kind, row)), "anchor": None,
            "parent": rid_of(parent[(table, row)]) if (table, row) in parent else None,
            "cells": [{"pid": r.period, "end": r.period_end, "value": r.value, "text": r.text,
                       "ok": r.status == "verified", "col": cols[r.col]["header"][:60] if r.col < len(cols) else "",
                       "ci": r.col, "bbox": r.bbox, "rid": r.id} for r in rs]}
        if (table, row) in children:
            rec["children"] = [rid_of(k) for k, _ in children[(table, row)]]
            rec["signs"] = [s for _, s in children[(table, row)]]
        out.append(rec)
    return out


VINTAGE_RE = re.compile(r"\bvintage|origination year|year of origination|by year originated|term loans by", re.I)


def _maturity_columns(t, fiscal_year: int) -> set:
    """Columns whose years are buckets of a schedule, not periods.

    "2026  2027  2028  2029  2030  Thereafter  Total" in a 2025 report says when amounts owed
    at the report date fall due. A filing reports nothing after its own fiscal year, so years
    that run upward past it are buckets of a schedule; "Thereafter" and "Total" beside them are
    buckets too. Every column of such a table keeps its header as the bucket name and gets no
    period id. In a table of loans by vintage the years say when loans were made.
    """
    years = [c.year for c in t.columns if c.year]
    if VINTAGE_RE.search(" ".join([t.title or "", t.caption or "", t.header_text or ""])):
        return {c.idx for c in t.columns if c.year}           # years of origination, not periods
    if len(years) < 2 or max(years) <= fiscal_year or years != sorted(set(years)):
        return set()
    return {c.idx for c in t.columns}


QUARTER_HEAD_RE = re.compile(r"quarter|\bq[1-4]\b|months|weeks|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def _marked_periods(cal, t, r, periods: list, form: str, filing_end: Optional[str] = None) -> list:
    """The periods of one row's columns when a date line sits above the row inside the table.

    "As of December 2024" between two stacked blocks of the same columns dates the second
    block; "2025" over a block of segment columns dates columns the header leaves undated.
    The line replaces the header's period only when the header gives every dated column one
    and the same period, or gives none (and the columns are not quarters or months).
    """
    mark = getattr(r, "period_mark", "")
    years = re.findall(r"\b((?:19|20)\d\d)\b", mark)
    if not years:
        return periods
    dated = {p for p in periods if p[0]}
    if len(dated) > 1 or not dated and any(QUARTER_HEAD_RE.search(c.header) for c in t.columns):
        return periods
    year = int(years[-1])
    out = []
    for c, p in zip(t.columns, periods):
        if dated and not p[0]:
            out.append(p)                       # a column the header leaves undated ("Total") stays so
            continue
        col = replace(c, year=year, end_date=printed_date(mark, year), months=duration(mark) or c.months)
        out.append(_period(cal, col, form, filing_end, _heading(t) + " " + mark))
    return out


DATE_WORD_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?:\s+\d{1,2}\b,?)?"
    r"|\b(?:19|20)\d\d\b|\bq[1-4]\b|\b(?:first|second|third|fourth)\s+quarter\b"
    r"|\b(?:three|six|nine|twelve|3|6|9|12|13|26|39|52|53)\s+(?:months|weeks)\b"
    r"|\b(?:fiscal(?:\s+years?)?|(?:years?|quarters?|months|weeks|periods?)\s+(?:ended|ending)|ended|ending|"
    r"ytd|year-to-date)\b",
    re.I)
EDGE_WORD_RE = re.compile(r"^(?:at|as\s+of|for\s+the|for|and|in|of)\b\s*|\s*\b(?:at|as\s+of|for\s+the|for|and|in|of)$",
                          re.I)


def member_of(header: str) -> str:
    """A column header with its date and year words removed: what the column is a member of
    ("Global Banking | 2025" -> "Global Banking", "September 30, 2025" -> "")."""
    text = DATE_WORD_RE.sub(" ", header or "")
    text = re.sub(r"[|,;:()\[\]⁠]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—.")
    while True:
        cut = EDGE_WORD_RE.sub("", text).strip(" -–—.")
        if cut == text:
            break
        text = cut
    return text if re.search(r"[A-Za-z]", text) else ""


COMPARISON_RE = re.compile(r"(?:19|20)\d\d\s*(?:vs\.?|versus|v\.|compared (?:to|with))(?:\s|$)|"
                           r"(?:19|20)\d\d\s*(?:/|to)\s*'?(?:(?:19|20)\d\d|\d\d)\b", re.I)


def _comparison(header: str) -> bool:
    """A column whose header line nearest the figures compares two years ("Change | 2025 vs. 2024",
    "2025/ 2024", "$ Change 2024 to 2025") holds a difference, not a figure of either year, so it gets no
    period id: a change of 353 must never answer for 2025 (AXP p49), nor WFC's branch change "(2)" (p26).
    Only that last header line is read: words from the lines above drift onto neighbouring columns, and a
    bare "Change" over a real year column (Nike p40 "CHANGE CHANGES" over fiscal 2024) is such a drift."""
    return bool(COMPARISON_RE.search((header or "").split("|")[-1]))


def _period(cal, col, form: str, filing_end: Optional[str] = None, heading: str = "",
            quarterly: bool = False) -> tuple:
    """(period id, ISO end date) of one column: periods.column_period, as a flow; in a table whose columns
    name two or more quarters, a column named by a quarter's ordinal is that quarter (_quarter_word_period)."""
    q = _quarter_word_period(cal, col, filing_end, heading) if quarterly else None
    if q is not None:
        return q
    return column_period(cal, col, form, instant=False, filing_end=filing_end, heading=heading)


def _names_quarters(t) -> bool:
    """The table's columns name at least two different quarters ("Fourth Third Second First"): a quarterly
    table. One column that mentions "fourth quarter" (a buyback table by month, GS p64, MSFT p32) is not."""
    found = set()
    for c in t.columns:
        for m in ORDINAL_QUARTER_RE.finditer(c.header or ""):
            found.add(ORDINALS[m.group(1).lower()] if m.group(1) else int(m.group(2) or m.group(3)))
    return len(found) >= 2


ORDINAL_QUARTER_RE = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\b|\bq([1-4])\b|\b([1-4])q(?:\d\d)?\b",
                                re.I)
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
ORDINAL_NAMES = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}


def _quarter_word_period(cal, col, filing_end: Optional[str], heading: str) -> Optional[tuple]:
    """A column headed by one quarter's ordinal under a year ("2025 Quarters" over "Fourth Third Second First",
    BAC p35; "Fourth Quarter 2024"; "Q3 2025") is that fiscal quarter: three months ending 3, 6, 9 or 12 months
    into the fiscal year labelled by the year. Read only when the header or the table's heading says
    "quarter", the column prints one ordinal and a year, and no full date (a printed date wins). Without this
    rule the year alone dated all four quarters as the full fiscal year. The column name gets the words
    "<ordinal> quarter <year>", so the column reads as a span of months, not a balance at the year end."""
    head = col.header or ""
    if col.end_date or not col.year or not re.search(r"quarter|\bq[1-4]\b|\b[1-4]q", head + " " + heading, re.I):
        return None
    found = set()
    for m in ORDINAL_QUARTER_RE.finditer(head):
        found.add(ORDINALS[m.group(1).lower()] if m.group(1) else int(m.group(2) or m.group(3)))
    if len(found) != 1:
        return None
    q = found.pop()
    fye = cal.end_of_label(col.year)
    month, year = fye.month - 3 * (4 - q), fye.year
    while month <= 0:
        month, year = month + 12, year - 1
    last = calendar.monthrange(year, month)[1]
    end = date(year, month, last if fye.day >= 28 else min(fye.day, last))
    if filing_end and (end - date.fromisoformat(filing_end)).days > 14:
        return None, None                            # a quarter after the filing's own period: nothing to date
    col.header = f"{head} | {ORDINAL_NAMES[q]} quarter {col.year}"[:120]
    return cal.period_id(end, 3, False), end.isoformat()


def _heading(t) -> str:
    """What a table prints above its columns: title, caption, header lines and introducing sentence."""
    return " ".join([t.title or "", t.caption or "", t.header_text or "", getattr(t, "intro", "") or ""])


def save_raw(rows: list, path: pathlib.Path = RAW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f)


def load_raw(path: pathlib.Path = RAW_PATH) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def summary(rows: list) -> dict:
    cells = sum(len(r["cells"]) for r in rows)
    return {"rows": len(rows), "cells": cells, "units": dict(Counter(r["unit"] for r in rows)),
            "roles": dict(Counter(r["role"] for r in rows))}


def tokens(text: str) -> list:
    out = []
    for w in re.findall(r"[a-z0-9&]+", norm_label(text)):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def search(rows: list, ticker: str, doc: str, phrase: str, k: int = 14) -> list:
    """Rows of one filing whose wording overlaps the phrase, best first. Wording only: no values."""
    want = set(tokens(phrase))
    if not want:
        return []
    scored = []
    for r in rows:
        if r["doc"] != doc or not any(c["pid"] for c in r["cells"]):
            continue
        lab = set(tokens(r["label"]))
        ctx = set(tokens(r["section"] + " " + r["title"] + " " + r["context"]))
        hit_l, hit_c = len(want & lab), len(want & ctx)
        if hit_l == 0:
            continue
        score = 3.0 * hit_l + 1.0 * hit_c - 0.15 * len(lab - want) + (0.5 if r["verified"] else 0) \
            + (0.5 if r["anchor"] or r["ties_to"] else 0)
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]
