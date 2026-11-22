"""Linking: tie the words of a question to a place in a filing, once, and keep the tie.

The base graph binds the three primary statements to shared concepts at build
time. Everything else a filing prints (revenue by segment, stores by state,
debt maturities, a figure stated only in a sentence) is already in the graph
as rows of the raw layer and as numeric statements of the text, each with its
page, its value and what the recovered formulas say about it. Nobody has said
what those rows mean. When a plan asks for an item such as "advertising
services revenue":

  1. candidates are found inside that company's filings by wording (row label,
     section heading, table title, note, column headers; or the sentence with
     its numbers masked). Values are never used and never shown;
  2. a language model picks one candidate, and a column or a number slot where
     the choice is not settled by the period;
  3. a gate made of structure, not of another model, records what vouches for
     the number: it sits inside a passing formula, its table's total ties to a
     bound fact, or the same value is printed elsewhere in the filing. A link
     with no such evidence is still served, with a flag on the answer;
  4. the link is written to semantics/learned.jsonl with the question that
     taught it. The same row in the company's other filings is linked too, so
     a later question about another year, or about a restatement, needs no
     model call at all.

`breakdown` does the same for the parts of a total: "revenue by segment" is
linked to a subtotal row, and its parts are the operands of that row's
recovered formula.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import math
import os
import pathlib
import re
import time
from collections import defaultdict
from typing import Optional

from .concepts import norm_label
from .graph import Fact, Graph, Reading
from .llm import DailyLimit, ask_json, link_model
from .witness import WITNESSED, Panel, choose

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEARNED = ROOT / "semantics" / "learned.jsonl"
RAW_PATH = ROOT / "graph" / "raw.json.gz"
MENTIONS_PATH = ROOT / "graph" / "mentions.json.gz"
TIMELESS = "*"                                    # period id of a figure whose source prints no period
MAX_CANDIDATES = 18
STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "total", "net", "from", "on", "at", "s",
        "was", "what", "how", "much", "many", "did", "does", "is", "were", "had", "have", "with", "that", "this"}
SOFT = {"number", "count", "amount", "figure", "level", "value", "reported", "full", "fiscal", "year", "company",
        "overall", "consolidated", "all"}   # words a question uses but a filing's label rarely repeats
COUNTED = re.compile(r"\b(number|how many|count|stores?|warehouses?|restaurants?|branch(es)?|locations?|employees|"
                     r"members?|customers?|subscribers?|vehicles?|units?|deliveries|people|headcount|clubs?|"
                     r"supercenters?|accounts?|users?)\b", re.I)
MONEY = re.compile(r"\b(revenues?|sales|income|expenses?|costs?|assets|liabilit(y|ies)|fees?|dollars?|cash|"
                   r"earnings|profit|loss|spend(ing)?|expenditures?|capex|debt|backlog|obligations?|inflows?|"
                   r"outflows?|money|arr|value)\b|\$", re.I)
RATED = re.compile(r"\b(ratio|percent(age)?|margin|rate|share of|yield)\b|%", re.I)
_ACRONYM = re.compile(r"\b[A-Z][A-Z&]{1,5}s?\b")

SYSTEM = """You match a phrase from a question to one place in a company's SEC filing.
You are shown numbered candidates. A ROW is a line of a table: its printed label, the section heading
above it, the table's title, the note it sits in, its column headers and its role in the table's
arithmetic. A SENTENCE is a statement from the text with every figure replaced by a slot tagged with its unit,
such as <n1 count> or <n2 USD>; dates and years are shown because they say which period a slot is.
You are never shown values. Pick the one candidate that reports the ITEM itself for the company (not a
component of it, not a percentage of it unless the item is a percentage).
The question is given only as context. It may need several figures, other periods or arithmetic; those
are handled elsewhere and are not your concern. A candidate that reports the item is the right pick
even though it does not answer the whole question. Rows marked "columns: periods" hold every period
the filing prints; the period is chosen later.
For a ROW whose columns are not periods, also give the number of the column that holds the item.
For a ROW whose periods each span several columns, give the number of the column that holds the item:
the total of the others, unless the item names one of the columns (a region, "as reported", "net").
For a SENTENCE, give the slot that holds the item's figure for the period asked (a count for "number of",
an amount for revenue or spending, a percentage only when a percentage is asked).
For a figure of the whole company, prefer the consolidated statements' row to a row of a segment
table. When a ROW and a SENTENCE both report the item, pick the ROW: it holds every period, exactly.
A sentence slot that holds a change (an increase or a decrease) is not the item itself.
Answer null only if no candidate reports the item.
Answer as JSON: {"pick": <candidate number or null>, "column": <column number or null>,
"slot": "<slot or null>", "reason": "<one short sentence>"}"""

SYSTEM_BREAKDOWN = """You match a phrase from a question to one subtotal in a company's SEC filing.
Each numbered candidate is a total of a table: its printed label, section, table title, the text printed
above the table, and its parts. The parts are either the rows that add up to it or the columns that add
up to its total column (a segment table prints its segments as columns), and the same total can appear
twice, once for each way. You are never shown values. Pick the candidate whose PARTS are the breakdown
the question asks about (by segment, by product, by region, by type of expense ...): for "profit by
segment" the parts must be the segments, not the lines of an income statement.
If none fits, answer null. Answer as JSON: {"pick": <candidate number or null>, "reason": "<one short sentence>"}"""


# ------------------------------------------------------------------ the store

class Store:
    """Raw-layer rows and text mentions, indexed by company. Large, so loaded once and on demand."""

    def __init__(self, rows: list, mentions: list):
        self.rows_by_ticker: dict = defaultdict(list)
        self.by_id: dict = {}
        for r in rows:
            self.rows_by_ticker[r["ticker"]].append(r)
            self.by_id[r["id"]] = r
        self.mentions_by_ticker: dict = defaultdict(list)
        for m in mentions:
            self.mentions_by_ticker[m["ticker"]].append(m)
            self.by_id[m["id"]] = m
        self.by_heading: dict = defaultdict(list)     # rows under one heading of one table
        for r in rows:
            if r.get("section"):
                self.by_heading[(r["doc"], r["page"], r["table"], r["section"])].append(r)

    def parts(self, total: dict) -> tuple:
        """(lines that make up a subtotal, how they are known).

        The operands of its recovered formula when there is one ("formula"). Otherwise, when the
        subtotal closes a heading of its table (segments listed above a consolidated line, where
        formula recovery failed on a nested subtotal), the other amount lines under that heading
        ("heading"): a weaker grouping, so answers built on it are flagged.
        """
        kids = [self.by_id[i] for i in total.get("children", []) if i in self.by_id]
        if len(kids) >= 2:
            return kids, "formula"
        if not total.get("section"):
            return [], ""
        same = self.by_heading.get((total["doc"], total["page"], total["table"], total["section"]), [])
        kids = [r for r in same if r["id"] != total["id"] and r["unit"] == total["unit"]
                and not norm_label(r["label"]).startswith(("total", "consolidated"))]
        return (kids, "heading") if len(kids) >= 2 else ([], "")


_STORE: Optional[Store] = None


def _read(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store(_read(RAW_PATH), _read(MENTIONS_PATH))
    return _STORE


# ------------------------------------------------------------------ wording

_SUFFIXES = ("ations", "ation", "ating", "ated", "ings", "ing", "ies", "es", "ed", "ate", "s")


def stem(w: str) -> str:
    """A light stem so that operated, operating and operations meet, and stores meets store."""
    if w.isdigit() or len(w) <= 4:
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4 and not w.endswith("ss"):
            w = w[:-len(suf)]
            break
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


_FOOTNOTE_TAIL = re.compile(r"(?<=[a-z]{4})\d{1,2}$")


def tokens(text: str) -> list:
    """Stemmed content words. A footnote marker printed against a word ("Production5", "Data1") is
    dropped, so the word still meets the question's (22 Nov: Chevron's worldwide production total sat
    under "Net Oil-Equivalent Production5" and never reached the shortlist). Short tokens such as
    "q4", "cet1" or "fy2025" keep their digits."""
    words = (_FOOTNOTE_TAIL.sub("", w) for w in re.findall(r"[a-z0-9&]+", norm_label(text)))
    return [stem(w) for w in words if w and w not in STOP]


def _phrase_key(ticker: str, phrase: str) -> tuple:
    return ticker, " ".join(tokens(phrase))


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_label(text)).strip("-")[:n]


def expected_unit(phrase: str) -> Optional[str]:
    """The unit a phrase asks for, when its words make that plain: "count", "USD" or "percent"."""
    if RATED.search(phrase):
        return "percent"
    if COUNTED.search(phrase) and not MONEY.search(phrase):
        return "count"
    if MONEY.search(phrase):
        return "USD"
    return None


def _unit_fit(unit: str, want_unit: Optional[str]) -> float:
    """A bonus for a candidate in the unit the phrase asks for, a penalty for one plainly in another."""
    if not want_unit or not unit:
        return 0.0
    same = {"percent": {"percent", "ratio"}, "count": {"count"}, "USD": {"USD"}}[want_unit]
    if unit in same:
        return 1.5
    if unit == "USD_scale_unknown" and want_unit == "USD":
        return 1.0                                    # an amount whose sentence prints no scale (see apply_link)
    if unit in ("number", "year", "date") or (unit == "shares" and want_unit == "count"):
        return 0.5 if want_unit == "count" and unit in ("number", "shares") else 0.0
    return -3.0 if {unit, want_unit} & {"percent", "ratio"} or want_unit == "count" else -1.0


def _abbreviated(words: list, acronyms: set) -> set:
    """Words of `words` that an acronym stands for: "annualized recurring revenue" for ARR."""
    out: set = set()
    for i in range(len(words)):
        for n in range(2, 6):
            run = words[i:i + n]
            if len(run) == n and "".join(w[0] for w in run) in acronyms:
                out.update(run)
    return out


def _words_in_order(text: str) -> list:
    return re.findall(r"[a-z0-9&]+", norm_label(text))


def _hits(want: set, phrase: str, text: str, text_tokens: set) -> set:
    """The phrase's words a candidate's text covers, counting acronyms either way (ARR, RPO, CET1)."""
    hit = want & text_tokens
    text_acr = {a.rstrip("s").lower() for a in _ACRONYM.findall(text)}
    hit |= want & {stem(w) for w in _abbreviated(_words_in_order(phrase), text_acr)}
    phrase_acr = {a.rstrip("s").lower() for a in _ACRONYM.findall(phrase)}
    if phrase_acr and phrase_acr & {"".join(w[0] for w in run) for run in _runs(_words_in_order(text))}:
        hit |= want & phrase_acr
    return hit


def _runs(words: list) -> list:
    return [words[i:i + n] for i in range(len(words)) for n in range(2, 6) if i + n <= len(words)]


_PID = re.compile(r"^FY(\d{4})(@end)?$")


def _inferred_pid(row: dict, j: int) -> Optional[str]:
    """The period of cell j when its header was not read but its neighbours say it.

    Cells [?, FY2025, FY2024] with a USD row are three years in falling order, so the first is
    FY2026. Only a plain yearly run of at least two printed periods is extended, by one step at
    either end, never past the filing's own fiscal year.
    """
    cells = row["cells"]
    if cells[j].get("pid") or not isinstance(cells[j].get("value"), (int, float)):
        return None
    filed = re.search(r"_FY(\d{4})", str(row.get("doc") or ""))   # doc ids carry the filing's fiscal year
    limit = int(filed.group(1)) if filed else 9999
    if not any(c.get("pid") for c in cells):          # no column was dated: the heading may name the years
        years = _heading_years(row.get("above", ""))
        # two or three reported years, none after the filing's own: a longer run, or later years,
        # is a maturity schedule or loans by year of origination, not reporting periods
        if 2 <= len(years) <= 3 and max(years) <= limit and len(years) <= len(cells):
            return f"FY{years[j]}" if j < len(years) else None
        year = _block_year(row)                       # one year over the whole block: every column is that year
        return f"FY{year}" if year and limit - 6 <= year <= limit else None
    known = [(k, _PID.match(c.get("pid") or "")) for k, c in enumerate(cells)]
    known = [(k, int(m.group(1)), m.group(2) or "") for k, m in known if m]
    if len(known) < 2:
        return None
    steps = {(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(known, known[1:])}
    suffixes = {s for _, _, s in known}
    if len(steps) != 1 or len(suffixes) != 1 or abs(next(iter(steps))) != 1:
        return None
    step, suffix = next(iter(steps)), next(iter(suffixes))
    first, last = known[0], known[-1]
    if j == first[0] - 1:
        year = first[1] - int(step)
    elif j == last[0] + 1:
        year = last[1] + int(step)
    else:
        return None
    return f"FY{year}{suffix}" if year <= limit else None


def _block_year(row: dict) -> Optional[int]:
    """The one year printed directly over a block of a table: "2024" over the second block of a
    segment table whose columns are the segments, "At December 31, 2025" over a schedule. None
    unless the last line above the block that names a year is short and names exactly one."""
    text = row.get("own") or row.get("above") or ""
    for line in reversed(text.split(" | ")):
        years = set(_YEAR_IN_HEADER.findall(line))
        if years:
            return int(next(iter(years))) if len(years) == 1 and len(line.split()) <= 7 else None
    return None


def _heading_years(above: str) -> list:
    """The run of years a table's heading prints for its first columns: "2025 2024 2023" before
    "2025 vs. 2024" gives [2025, 2024, 2023]. Only a run of two or more years that follow one another
    directly and step by one counts; anything else gives []."""
    run: list = []
    for token in above.replace("|", " | ").split():
        bare = token.strip("(),.;:")
        if run and bare.lower() in ("fiscal", "fy"):   # "FISCAL 2026 FISCAL 2025" is one run
            continue
        if len(bare) == 4 and bare.isdigit() and bare[:2] in ("19", "20"):
            year = int(bare)
            if not run or abs(year - run[-1]) == 1 and (len(run) < 2 or year - run[-1] == run[-1] - run[-2]):
                run.append(year)
                continue
        if len(run) >= 2:
            return run
        run = []
    return run if len(run) >= 2 else []


_QUARTERLY_DOC = re.compile(r"_FY(\d{4})_Q(\d)_10Q$")
_QUARTER_PID = re.compile(r"^FY(\d{4})Q(\d)")


_SPAN_WORD = re.compile(r"\b(three|six|nine)\b|\bquarter\b", re.I)
_SPANNED = re.compile(r"\bmonths?\b|\bquarter", re.I)
_FLOW_QUARTER = re.compile(r"^FY(\d{4})Q(\d)(YTD)?$")
_YEAR_IN_HEADER = re.compile(r"\b((?:19|20)\d\d)\b")


def _cell_pid(row: dict, j: int) -> Optional[str]:
    """The period of cell j: read from its header, or inferred.

    In a quarterly report a column cannot be a full fiscal year of flows or a quarter after the
    report's own, so such an id is dropped, and a column headed "Nine Months Ended" is the year
    to date whatever the builder made of its split header (see `_quarterly_pids`). In an annual
    report a column dated as the fourth quarter whose header names no span of months is the
    balance at the year end ("At December 31, 2025").
    """
    cell = row["cells"][j]
    pid = cell.get("pid") or _inferred_pid(row, j)
    filed = _QUARTERLY_DOC.search(str(row.get("doc") or ""))
    if filed:
        pid = _quarterly_pids(row, filed).get(j, pid)
        if pid and not pid.endswith("@end"):
            quarter = _QUARTER_PID.match(pid)
            if quarter is None or (quarter.group(1), quarter.group(2)) > (filed.group(1), filed.group(2)):
                return None
        return pid
    if pid and pid.endswith("Q4") and not _SPANNED.search(f"{cell.get('col') or ''} {cell.get('head') or ''}"):
        return pid[:-2] + "@end"
    return pid


def _header_spans(row: dict) -> list:
    """The spans the last header line over a quarterly table names, left to right:
    "Three Months Ended ... Nine Months Ended" gives ["quarter", "to date"]."""
    text = row.get("own") or row.get("above") or ""
    line = next((ln for ln in reversed(text.split(" | ")) if _SPAN_WORD.search(ln)), "")
    spans: list = []
    for said in _SPAN_WORD.finditer(line):
        span = "quarter" if said.group(0).lower() in ("three", "quarter") else "to date"
        if not spans or spans[-1] != span:
            spans.append(span)
    return spans


def _quarterly_pids(row: dict, filed: "re.Match") -> dict:
    """{cell index: period id} for the cells of a quarterly report's row that the column heads date
    better than the builder did. Empty when the builder's dates are consistent.

    A quarterly table prints "Three Months Ended" and "Nine Months Ended" once each, above the two
    columns they span. The words above each column (`head`, read from the page geometry) say where
    each span starts; a column without a span word belongs to the span that started to its left.
    Used only where the builder left a cell undated or gave two cells of one row the same period,
    and only if the result dates every cell differently.
    """
    if "_quarterly" in row:
        return row["_quarterly"]
    cells, fixed = row["cells"], {}
    built = [c.get("pid") for c in cells]
    flows = [p for p in built if p and _FLOW_QUARTER.match(p)]
    if all(built) and len(flows) == len(set(flows)):
        row["_quarterly"] = fixed
        return fixed
    named = _header_spans(row)
    if named and len(cells) % len(named) == 0:        # the header line names the spans, left to right,
        width = len(cells) // len(named)              # and each covers the same number of columns
        spans = [named[k // width] for k in range(len(cells))]
    else:
        spans, current = [], None
        for c in cells:                               # else the span word printed over each column
            said = _SPAN_WORD.search(c.get("head") or "")
            if said:
                current = "quarter" if said.group(0).lower() in ("three", "quarter") else "to date"
            spans.append(current)
    years = [_YEAR_IN_HEADER.findall(f"{c.get('col') or ''}") for c in cells]
    newest = max((int(y[-1]) for y in years if y), default=None)
    fiscal_year, quarter = int(filed.group(1)), int(filed.group(2))
    for k, c in enumerate(cells):
        if spans[k] is None or not isinstance(c.get("value"), (int, float)):
            continue
        was = _FLOW_QUARTER.match(built[k] or "")
        if was:
            year = int(was.group(1))                  # the builder read the year; only the span was in doubt
        elif built[k] is None and years[k] and newest is not None:
            year = fiscal_year - (newest - int(years[k][-1]))   # the newest column is the report's own period
        else:
            continue
        fixed[k] = f"FY{year}Q{quarter}" + ("YTD" if spans[k] == "to date" and quarter > 1 else "")
    def clashes(pids: list) -> int:
        dated = [p for p in pids if p]
        return len(dated) - len(set(dated))

    after = [fixed.get(k, built[k]) for k in range(len(cells))]
    fewer_clashes = clashes(after) < clashes(built)
    more_dated = sum(1 for p in after if p) > sum(1 for p in built if p) and clashes(after) <= clashes(built)
    if not (fewer_clashes or more_dated):             # the heads explain nothing the builder missed
        fixed = {}
    row["_quarterly"] = fixed
    return fixed


def heading_key(row: dict) -> str:
    """What is printed above a row's table, reduced to its words: the same table in another year's
    filing has the same key, another segment's table with the same row labels does not."""
    words = [w for w in re.findall(r"[a-z&]+", norm_label(row.get("above", ""))) if len(w) > 2]
    return " ".join(words[:14])


def candidates(st: Store, g: Graph, ticker: str, phrase: str, wanted: set, fiscal_year: str) -> list:
    """Rows and sentences of this company whose wording overlaps the phrase, best first.

    `wanted` are the period ids the question asks for. A row that prints periods but none of
    those cannot answer and is left out; a filing of the asked fiscal year is preferred. Words a
    question uses but labels rarely print ("number of", "total") are not required, acronyms
    match their words either way, and a candidate in the unit the phrase plainly asks for (a
    count, an amount, a percentage) is preferred. Wording and units only: no value is consulted.
    """
    want = set(tokens(phrase))
    core = (want - {stem(w) for w in SOFT}) or want
    if not want:
        return []
    want_unit = expected_unit(phrase)
    scored = []
    for r in st.rows_by_ticker.get(ticker, []):
        periods = {_cell_pid(r, j) for j in range(len(r["cells"]))} - {None}
        if periods and not periods & wanted:
            continue                                  # prints other periods only
        label = set(tokens(r["label"]))
        ctx_text = " ".join((r["section"], r["title"], r.get("context", ""), r.get("above", ""),
                             " ".join(c.get("col", "") for c in r["cells"])))
        context = set(tokens(ctx_text))
        hit_label = _hits(core, phrase, r["label"], label)
        hit_context = _hits(core, phrase, ctx_text, context) - hit_label
        if not hit_label and len(hit_context) < min(2, len(core)):
            continue
        covered = len(hit_label | hit_context) == len(core)
        score = (3.0 * len(hit_label) + 1.0 * len(hit_context) - 0.15 * len(label - want)
                 + (1.5 if periods & wanted else 0.0) + (0.5 if r["verified"] else 0.0)
                 + (0.5 if r.get("anchor") or r.get("ties_to") else 0.0)
                 + (0.75 if g.filings[r["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + (1.0 if any(c.get("rid") for c in r["cells"]) else 0.0)     # a line of a primary statement
                 + 2.0 * covered + _unit_fit(r["unit"], want_unit))
        scored.append((score, "row", r, covered))
    for m in st.mentions_by_ticker.get(ticker, []):
        words = set(tokens(m["masked"]))
        hit = _hits(core, phrase, m["masked"], words)
        if len(hit) < max(1, -(-3 * len(core) // 5)):   # at least three fifths of the core words
            continue
        units = {n.get("unit") for n in m["numbers"]}
        fit = max((_unit_fit(u, want_unit) for u in units), default=0.0)
        stated = {stated_period(g, m, n["slot"].strip("<>")) for n in m["numbers"] if not _is_time(n)} - {None}
        timely = 1.5 if stated & wanted else (-1.5 if stated else 0.0)   # states its figures for the period asked
        score = (2.0 * len(hit) - 0.03 * len(words - want) + (0.5 if m.get("ties") else 0.0)
                 + (0.75 if g.filings[m["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + 2.0 * (len(hit) == len(core)) + fit + timely)
        scored.append((score, "sentence", m, len(hit) == len(core)))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for score, kind, ref, covered in scored:         # the same row reprinted in an older filing adds nothing
        sig = (kind, norm_label(ref.get("label") or ref.get("masked", "")), norm_label(ref.get("section", "")),
               ref["doc"] if kind == "sentence" else heading_key(ref))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"kind": kind, "ref": ref, "score": score, "covered": covered})
        if len(out) == MAX_CANDIDATES:
            break
    return out


SYSTEM_LABELS = """You say how an SEC filing would label a financial item. Given an item phrase from a question
and the company, list up to 4 other labels the company's annual or quarterly report might print for the
same item: its own term for it, a synonym, an abbreviation spelled out or the reverse. Never a number.
Answer as JSON: {"labels": ["...", "..."]}"""


def other_labels(g: Graph, ticker: str, phrase: str, model: Optional[str] = None, offline: bool = False) -> list:
    """Other wordings a filing may print for the phrase ("net new money" -> "net inflows"). One cached call."""
    user = f"Company: {g.companies[ticker]['name']} ({ticker})\nItem: {phrase}"
    try:
        ans = ask_json(SYSTEM_LABELS, user, model=link_model(model, offline), tag="labels", offline=offline)
    except DailyLimit:
        raise
    except Exception:  # noqa: BLE001 - no alternatives is a fine answer
        return []
    labels = ans.get("labels") if isinstance(ans, dict) else None
    if not isinstance(labels, list):
        return []
    return [x for x in labels if isinstance(x, str) and x.strip() and not re.search(r"\d{3}", x)][:4]


def widened(st: Store, g: Graph, ticker: str, phrase: str, wanted: set, fiscal_year: str,
            model: Optional[str] = None, offline: bool = False) -> list:
    """Candidates for the phrase; when none of them carries all its words, also for the other
    labels a filing might use, merged best first."""
    cands = candidates(st, g, ticker, phrase, wanted, fiscal_year)
    if any(c["covered"] for c in cands[:5]):
        return cands
    best: dict = {c["ref"]["id"]: c for c in cands}
    for label in other_labels(g, ticker, phrase, model, offline):
        for c in candidates(st, g, ticker, label, wanted, fiscal_year):
            if c["ref"]["id"] not in best or best[c["ref"]["id"]]["score"] < c["score"]:
                best[c["ref"]["id"]] = c
    return sorted(best.values(), key=lambda c: -c["score"])[:MAX_CANDIDATES]


def _columns(row: dict) -> list:
    """Column headers of a row's cells, in print order, without repeats."""
    return list(dict.fromkeys(c.get("col", "") for c in row["cells"]))


def acronym_note(phrase: str, text: str) -> str:
    """" [ARR = annualized recurring revenue]" when the phrase's words abbreviate to an acronym the
    candidate prints (small models miss these), else ""."""
    acronyms = {a.rstrip("s") for a in _ACRONYM.findall(text)}
    words = _words_in_order(phrase)
    for i in range(len(words)):
        for n in range(5, 1, -1):
            run = words[i:i + n]
            if len(run) == n and "".join(w[0] for w in run).upper() in acronyms:
                return f" [{''.join(w[0] for w in run).upper()} = {' '.join(run)}]"
    return ""


def _show(i: int, cand: dict, g: Graph, phrase: str = "") -> str:
    ref = cand["ref"]
    filing = f"{g.filings[ref['doc']]['form']} {g.filings[ref['doc']]['fiscal_year']}"
    if cand["kind"] == "sentence":
        return (f"{i}: SENTENCE ({filing}, page {ref['page']}): {shown_sentence(ref)[:360]}"
                + acronym_note(phrase, ref["masked"]))
    dated = any(_cell_pid(ref, j) for j in range(len(ref["cells"])))
    cols = "periods" if dated else "; ".join(f"[{j}] {h[:40]}" for j, h in enumerate(_columns(ref)))
    spans = members(ref) if dated else []
    if spans:
        cols += "; each period spans the columns " + "; ".join(f"[{k}] {name}" for k, name in enumerate(spans))
    return (f"{i}: ROW ({filing}, page {ref['page']}): label '{ref['label'][:80]}' | section '{ref['section'][:50]}' | "
            f"table '{ref['title'][:90]}' | printed above the table '{ref.get('above', '')[-170:]}' | "
            f"note '{ref.get('context', '')[:50]}' | columns: {cols} | role {ref['role']} | unit {ref['unit']}"
            + acronym_note(phrase, " ".join((ref["label"], ref["section"], ref["title"]))))


def shown_sentence(m: dict) -> str:
    """The sentence as the linker sees it: dates and years printed (they name periods, not
    figures), every other number a slot tagged with its unit, such as <n1 count> or <n2 USD>."""
    text = m["masked"]
    for n in sorted(m["numbers"], key=lambda n: -len(n["slot"])):    # <n10> before <n1>
        inner = n["slot"].strip("<>")
        unit = "USD, scale not printed" if n.get("unit") == "USD_scale_unknown" else (n.get("unit") or "number")
        shown = n["text"] if n.get("unit") in ("date", "year") else f"<{inner} {unit}>"
        text = text.replace(n["slot"], shown)
    return text


def _is_time(n: dict) -> bool:
    return n.get("unit") in ("date", "year")


def stated_period(g: Graph, m: dict, slot: str) -> Optional[str]:
    """The period a sentence states for one of its figures, as a period id; None when it states none.

    "... was $30.0 billion at December 31, 2024 and $27.5 billion at December 31, 2023": each figure
    is followed by its own date. A sentence with one date or year dates every figure in it. A
    figure the builder already paired with a year of a list ("... in 2026, 2025 and 2024") keeps
    that. Anything less clear is left undated."""
    numbers = m["numbers"]
    at = next((i for i, n in enumerate(numbers) if n["slot"].strip("<>") == slot), None)
    if at is None or _is_time(numbers[at]):
        return None
    if numbers[at].get("pid"):
        return numbers[at]["pid"]
    times = [n for n in numbers if _is_time(n)]
    if not times:
        return None
    if len({str(n["value"]) for n in times}) == 1:
        return _time_pid(g, m["doc"], times[0])
    figures = [i for i, n in enumerate(numbers) if not _is_time(n)]
    own = {i: numbers[i + 1] for i in figures if i + 1 < len(numbers) and _is_time(numbers[i + 1])}
    if len(own) == len(figures):                      # every figure is followed by its own date
        return _time_pid(g, m["doc"], own[at])
    if len(times) == len(figures) and max(figures) < min(i for i, n in enumerate(numbers) if _is_time(n)):
        # a list of figures, then the list of their dates in the same order ("939, 914 and 890 ... at
        # August 30, 2026, August 31, 2025 and September 1, 2024")
        return _time_pid(g, m["doc"], times[figures.index(at)])
    return None


def _time_pid(g: Graph, doc: str, n: dict) -> Optional[str]:
    """A date or year printed in a sentence of `doc`, as a period id of that company.

    A bare year is the fiscal year the filing calls by that number. A date that ends one of the
    company's filed periods is that period's end; a date one or more whole years away from a
    fiscal year end is the end of that earlier or later fiscal year."""
    filing = g.filings[doc]
    if n.get("unit") == "year":
        year, filed = int(n["value"]), int(filing["fiscal_year"][2:])
        return f"FY{year}" if filed - 6 <= year <= filed else None
    try:
        day = datetime.date.fromisoformat(str(n["value"]))
    except ValueError:
        return None
    ends = [(d, f) for d, f in g.filings.items() if f["ticker"] == filing["ticker"] and f.get("period_end")]
    for d, f in ends:
        if f["period_end"] == day.isoformat():
            quarter = _QUARTERLY_DOC.search(d)
            return f"{f['fiscal_year']}Q{quarter.group(2)}@end" if quarter else f"{f['fiscal_year']}@end"
    for d, f in ends:
        if f["form"] != "10-K":
            continue
        end = datetime.date.fromisoformat(f["period_end"])
        for shift in (-4, -3, -2, -1, 1):             # a 52/53-week year ends within a few days of the same date
            if abs((day - end).days - 365.25 * shift) <= 8:
                return f"FY{int(f['fiscal_year'][2:]) + shift}@end"
    return None


# ------------------------------------------------------------------ the gate

def gate(cand: dict) -> tuple:
    """(confirmed, evidence, objections): what structure says about the number behind a candidate."""
    ref, evidence, objections = cand["ref"], [], []
    if cand["kind"] == "sentence":
        if ref.get("ties"):
            evidence.append("the same value is printed in a table of the same filing")
        else:
            objections.append("stated in a sentence only; no table prints the same value")
        return bool(evidence), evidence, objections
    if not ref["label"].strip():
        objections.append("row has no printed label")
    if ref["verified"]:
        evidence.append(f"value sits inside a passing formula of its table ({ref['role']})")
    if ref.get("anchor"):
        evidence.append(f"its table's total prints the same value as {ref['anchor']}: a verified component of it")
    if ref.get("ties_to"):
        evidence.append(f"prints the same value as {ref['ties_to']}")
    if not evidence:
        objections.append("no formula covers the row and its table does not tie to any bound fact")
    return bool(evidence) and not objections, evidence, objections


# ------------------------------------------------------------------ learning a link

def aliases(g: Graph) -> dict:
    if not hasattr(g, "item_alias"):
        g.item_alias = {}
    return g.item_alias


def resolve(g: Graph, ticker: str, phrase: str) -> list:
    """The fact keys an item phrase was linked to for this company, newest first ([] if never linked).
    A phrase can hold more than one: the same item may sit in differently laid-out tables in two years' filings."""
    return list(reversed(aliases(g).get(_phrase_key(ticker, phrase), [])))


def _same_figure(x, y) -> bool:
    """Two figures that are one printed fact: equal, or a sentence's rounding of a table's amount."""
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
        return abs(abs(x) - abs(y)) <= 0.006 * max(abs(x), abs(y), 1e-12)
    return x == y


_BASIS = re.compile(r"\b(?:non[- ]?)?(?:u\.?\s?s\.?\s+)?gaap\b", re.I)
_TOTAL = re.compile(r"\btotal\b", re.I)


def named_columns(spans: list, phrase: str) -> list:
    """The columns of a period's group that the phrase names in their own words ("North America
    revenue", "total earnings"). An accounting basis names how a figure is measured, not a column:
    "(U.S. GAAP)" is not the U.S. column. ("Total" is a stop word for row matching, so it is tested
    on its own.) 22 Nov: Exxon's Energy Products earnings "(U.S. GAAP, total ...)" were read from the
    U.S. column although both pickers chose the Total."""
    text = _BASIS.sub(" ", phrase)
    asked = set(tokens(text))
    named = []
    for n, name in enumerate(spans):
        bare = name.replace(" (the sum of the others)", "")
        if _TOTAL.fullmatch(bare.strip()):
            if _TOTAL.search(text):
                named.append(n)
            continue
        words = set(tokens(bare))
        if words and words <= asked:
            named.append(n)
    return named


def pick_yields(g: Graph, cands: list, ans: dict, phrase: str, wanted: set) -> Optional[tuple]:
    """The figures a pick would read for the periods asked, in period order; None when the pick
    names nothing readable. Computed in code from the candidates: no model sees these values. It
    lets two pickers be compared on what they chose rather than on where in the list they found it."""
    i = _as_int(ans.get("pick"))
    if i is None or not 0 <= i < len(cands):
        return None
    ref = cands[i]["ref"]
    if cands[i]["kind"] == "sentence":
        said = re.search(r"n\d+", str(ans.get("slot") or ""))
        fitting = [n["slot"].strip("<>") for n in ref["numbers"]
                   if not _is_time(n) and _unit_fit(n.get("unit"), expected_unit(phrase)) >= 0]
        slot = said.group(0) if said else (fitting[0] if len(fitting) == 1 else "")
        dated = [s for s in fitting if stated_period(g, ref, s) in wanted]
        if len(dated) == 1:
            slot = dated[0]
        number = next((n for n in ref["numbers"] if n["slot"].strip("<>") == slot), None)
        return None if number is None else (number.get("value"),)
    if not any(_cell_pid(ref, j) for j in range(len(ref["cells"]))):
        headers, j = _columns(ref), _as_int(ans.get("column"))
        column = headers[j] if j is not None and 0 <= j < len(headers) else (headers[0] if len(headers) == 1 else None)
        return None if column is None else tuple(c.get("value") for c in ref["cells"] if c.get("col", "") == column)
    spans, member = members(ref), None
    if spans:
        k = _as_int(ans.get("column"))
        named = named_columns(spans, phrase)
        k = named[0] if len(named) == 1 else k
        member = k if k is not None and 0 <= k < len(spans) else None
    figures = tuple(ref["cells"][j].get("value") for pid, j in sorted(period_cells(ref, member).items()) if pid in wanted)
    return figures or None


_QUESTION: dict = {"budget": None, "seen": None}   # no memory of picks until a question starts


def start_question(arbiter_budget: Optional[int] = None) -> None:
    """Begin one question: no memory of another question's picks, and at most `arbiter_budget` calls to
    the arbiter for disputed picks (None: no limit). The strong free models allow about twenty calls a
    day each, and the readings of questions need them first; a dispute past the budget is served from
    the first picker, flagged. (22 Nov: one round-6 question asked the arbiter 76 times.)"""
    _QUESTION["budget"], _QUESTION["seen"] = arbiter_budget, {}


def _pick(system: str, user: str, tag: str, model: Optional[str], offline: bool, yields=None) -> tuple:
    """A pick made once per question: the same prompt asked again (another learning round, another
    reader) gets the same answer without new calls or arbitration."""
    seen = _QUESTION["seen"]
    if seen is None:
        return _pick_once(system, user, tag, model, offline, yields)
    key = (tag, model, os.environ.get("TIEOUT_LINK_CHECKER", ""), os.environ.get("TIEOUT_LINK_ARBITER", ""),
           hashlib.sha1((system + "\n" + user).encode("utf-8")).hexdigest())
    if key not in seen:
        seen[key] = _pick_once(system, user, tag, model, offline, yields)
    return seen[key]


def _pick_once(system: str, user: str, tag: str, model: Optional[str], offline: bool, yields=None) -> tuple:
    """(the pick as the model's JSON answer, or None; the case). One picker by default. With
    TIEOUT_LINK_CHECKER set, the pick is made under the four-eyes rule: a second, independent
    picker, and an arbiter (TIEOUT_LINK_ARBITER, strongest first) when the two differ. `yields(pick)`
    says what a pick would read, so agreement is judged on the figures, not the candidate number."""

    def ask(m: str) -> Optional[dict]:
        try:
            ans = ask_json(system, user, model=m, tag=tag, offline=offline)
        except DailyLimit:
            raise
        except Exception:  # noqa: BLE001 - a picker that fails has no pick
            return None
        return ans if isinstance(ans, dict) and _as_int(ans.get("pick")) is not None else None

    def same(a: dict, b: dict) -> bool:
        """The same choice. Judged on what the picks yield where that is known (two reprints of one
        line give the same figures and are one choice), else on the candidate, column and slot named."""
        if yields is not None:
            ya, yb = yields(a), yields(b)
            if ya is not None and yb is not None:
                return len(ya) == len(yb) and all(_same_figure(x, y) for x, y in zip(ya, yb))
        column_a, column_b = _as_int(a.get("column")), _as_int(b.get("column"))
        slot_a, slot_b = str(a.get("slot") or "").strip("<>"), str(b.get("slot") or "").strip("<>")
        return (_as_int(a.get("pick")) == _as_int(b.get("pick"))
                and (column_a is None or column_b is None or column_a == column_b)
                and (not slot_a or not slot_b or slot_a == slot_b))

    maker, checker = link_model(model, offline), os.environ.get("TIEOUT_LINK_CHECKER")
    if not checker:
        return ask(maker), ""
    arbiters = tuple(m for m in os.environ.get("TIEOUT_LINK_ARBITER", "").split(",") if m)
    budget = _QUESTION["budget"]
    if budget is not None and budget <= 0:
        arbiters = ()                                 # spent for this question: a dispute stays flagged
    ans, case, seen = choose(ask, Panel(maker, checker, arbiters), same)
    if budget is not None and len(seen) > 2:
        _QUESTION["budget"] = budget - 1
    return ans, case


def learn_link(g: Graph, ticker: str, phrase: str, wanted: set, fiscal_year: str, question: str = "",
               question_id: str = "", model: Optional[str] = None, offline: bool = False) -> dict:
    """Try to link (ticker, phrase). Returns the store entry; an admitted or flagged link is applied."""
    st = store()
    entry = {"kind": "item", "ticker": ticker, "phrase": phrase, "taught_by": question_id, "question": question,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "rejected", "item": None, "label": None,
             "evidence": [], "objections": []}
    cands = widened(st, g, ticker, phrase, wanted, fiscal_year, model, offline)
    if not cands:
        entry["objections"] = ["nothing in the company's filings matches the wording"]
        return entry
    ends = sorted({f["period_end"] for f in g.filings.values()
                   if f["ticker"] == ticker and f["fiscal_year"] == fiscal_year and f.get("period_end")})
    period = fiscal_year + (f" (fiscal year ending {ends[-1]})" if ends else "")
    quarter = next((m for m in map(_QUARTER_PID.match, sorted(wanted)) if m), None)
    if quarter:                                       # a quarter is asked: say so, with the date it ends
        report = next((f for d, f in g.filings.items() if f["ticker"] == ticker and f["fiscal_year"] == fiscal_year
                       and d.endswith(f"_Q{quarter.group(2)}_10Q")), None)
        period = (f"{fiscal_year}, quarter {quarter.group(2)}"
                  + (", the year to date" if any(p.endswith("YTD") for p in wanted) else "")
                  + (f" (ending {report['period_end']})" if report and report.get("period_end") else ""))
    # The question comes last and is marked as context: on 22 Nov the picker refused every candidate for
    # the inputs of a derived fourth quarter (J&J Cardiovascular, full year and nine months) because the
    # question asked for a quarter the filing never prints, though the system prompt says that is not its
    # concern. The pick is judged for the item and period of this step only.
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nItem asked for: {phrase}\nPeriod asked: {period}\n"
            f"Question (context only: it may ask for another period, or for arithmetic that is done elsewhere "
            f"with this figure as one input): {question}\n\nCandidates:\n"
            + "\n".join(_show(i, c, g, phrase) for i, c in enumerate(cands))
            + f"\n\nPick the candidate that reports '{phrase}' for {period}.")
    ans, case = _pick(SYSTEM, user, "link", model, offline,
                      yields=lambda pick: pick_yields(g, cands, pick, phrase, wanted))
    pick = _as_int(ans.get("pick")) if ans else None
    if pick is None or not 0 <= pick < len(cands):
        entry["objections"] = ["proposer found no candidate"]
        return entry
    entry["reason"], entry["picked"] = str(ans.get("reason", ""))[:200], case
    cand = cands[pick]
    ref = cand["ref"]
    if cand["kind"] == "sentence":
        said = re.search(r"n\d+", str(ans.get("slot") or ""))                   # "<n1 count>" or "n1" -> "n1"
        fitting = [n["slot"].strip("<>") for n in ref["numbers"]
                   if n.get("unit") not in ("date", "year") and _unit_fit(n.get("unit"), expected_unit(phrase)) >= 0]
        slot = said.group(0) if said else (fitting[0] if len(fitting) == 1 else "")   # one figure of the right unit
        dated = [s for s in fitting if stated_period(g, ref, s) in wanted]
        if len(dated) == 1:                           # the sentence itself says which figure is for the period asked
            slot = dated[0]
        if slot not in {n["slot"].strip("<>") for n in ref["numbers"]}:
            entry["objections"] = ["proposer named no number slot of the sentence"]
            return entry
        unit = next(n.get("unit") for n in ref["numbers"] if n["slot"].strip("<>") == slot)
        if _unit_fit(unit, expected_unit(phrase)) < 0:
            entry["objections"] = [f"the number picked is in {unit}, the item asks for {expected_unit(phrase)}"]
            return entry
        tied = _tied_row(st, ref, slot, wanted)
        if tied is None:
            entry.update(kind="mention", item=f"said:{ref['id']}:{slot}", ref_id=ref["id"], slot=slot,
                         label=ref["masked"][:160], page=ref["page"], doc=ref["doc"],
                         period=stated_period(g, ref, slot))
        else:                                         # the same figure, printed exactly, in a table of the filing
            sentence_page, (ref, j) = ref["page"], tied
            cand = {"kind": "row", "ref": ref}
            key = f"item:{_slug(ref['label'])}@{_slug(ref['section'] or ref['title'])}"
            group = next((js for js in column_groups(ref).values() if j in js), [j])
            member = group.index(j) if len(group) > 1 and j != default_cell(ref, group) else None
            if member is not None:                    # the figure sits in one named column of its period
                key += f"#{_slug(members(ref)[member], 32)}"
            entry.update(item=key, ref_id=ref["id"], row_id=ref["id"], column=None, member=member, label=ref["label"],
                         title=ref["title"], section=ref["section"], page=ref["page"], doc=ref["doc"],
                         tied_from=f"the sentence on page {sentence_page} states this figure")
    else:
        column = None
        if not any(_cell_pid(ref, j) for j in range(len(ref["cells"]))):
            headers = _columns(ref)
            j = _as_int(ans.get("column"))
            column = headers[j] if j is not None and 0 <= j < len(headers) else (headers[0] if len(headers) == 1 else None)
            if column is None:
                entry["objections"] = ["the row's columns are not periods and the proposer named no column"]
                return entry
        key = f"item:{_slug(ref['label'])}@{_slug(ref['section'] or ref['title'])}" + (f"#{_slug(column, 32)}" if column else "")
        spans, member = (members(ref) if column is None else []), None
        if spans:                                     # the proposer may name one column of a period's group
            k = _as_int(ans.get("column"))
            default = [j for js in column_groups(ref).values() if len(js) == len(spans) for j in [js.index(default_cell(ref, js))]]
            named = named_columns(spans, phrase)
            if len(named) == 1:                       # the phrase itself names one column ("North America ...")
                k = named[0]
            if k is not None and 0 <= k < len(spans) and (not default or k != default[0]):
                member, key = k, key + f"#{_slug(spans[k], 32)}"
        entry.update(item=key, ref_id=ref["id"], row_id=ref["id"], column=column, member=member, label=ref["label"],
                     title=ref["title"], section=ref["section"], page=ref["page"], doc=ref["doc"])
    confirmed, entry["evidence"], entry["objections"] = gate(cand)
    if entry.get("tied_from"):
        entry["evidence"].append(entry["tied_from"])
    if case in WITNESSED:
        entry["evidence"].append(f"picked independently by two models ({case})")
    elif case:
        entry["objections"].append(f"the pickers did not agree on this line ({case})")
    twin = lookalike(g, cands, pick, ans, phrase, wanted) if cands[pick]["kind"] == "row" and cand["ref"] is cands[pick]["ref"] else None
    if twin is not None:                              # two models can agree on the wrong one of two namesakes
        entry["second_reading"] = (f"second reading: a line with the same label '{twin['label'][:50]}' is printed on page "
                                   f"{twin['page']} of {twin['doc']} under another heading with a different figure, "
                                   "and the words asked for do not tell the two apart")
        entry["objections"].append(entry["second_reading"])
    entry["state"] = "active" if confirmed and (not case or case in WITNESSED) and twin is None else "flagged"
    apply_link(g, entry)
    return entry


def _context_words(row: dict) -> set:
    """The words printed around a row but not in its label: section, title, the lines above its table, column names."""
    around = " ".join((row["section"], row["title"], row.get("context", ""), row.get("above", ""), row.get("own", ""),
                       " ".join((c.get("head") or c.get("col") or "") for c in row["cells"])))
    return set(tokens(around)) - set(tokens(row["label"]))


def lookalike(g: Graph, cands: list, pick: int, ans: dict, phrase: str, wanted: set) -> Optional[dict]:
    """A candidate row with the picked row's label, printed under another heading, that reads a different
    figure for the period asked and that the phrase's own words fit at least as well. None if there is
    no such row.

    Two segments' tables, or a debt schedule and an amortization schedule, print rows with one
    label. A model tells them apart by the words around the row; when the phrase shares no more
    words with the picked row's surroundings than with its namesake's, nothing but the model's
    judgement separates them, and two models can share a wrong judgement (22 Nov, Oracle: both
    picked "Fiscal 2027" of the amortization schedule for a question about debt maturities)."""
    picked = cands[pick]["ref"]
    mine = pick_yields(g, cands, ans, phrase, wanted)
    if mine is None:
        return None
    asked = set(tokens(phrase)) - {stem(w) for w in SOFT} - set(tokens(picked["label"]))
    fit = len(asked & _context_words(picked))
    for i, c in enumerate(cands):
        row = c["ref"]
        if i == pick or c["kind"] != "row" or norm_label(row["label"]) != norm_label(picked["label"]):
            continue
        if row["doc"] == picked["doc"] and heading_key(row) == heading_key(picked):
            continue                                  # another block of the same table
        theirs = pick_yields(g, cands, {**ans, "pick": i}, phrase, wanted)
        if theirs is None or (len(theirs) == len(mine) and all(_same_figure(x, y) for x, y in zip(theirs, mine))):
            continue                                  # a reprint of the same figures
        if len(asked & _context_words(row)) >= fit:
            return row
    return None


def _same_line(first: Optional[dict], row: dict, namesakes: list, member: Optional[int]) -> bool:
    """Is `row` (same label, same section or title) the line `first` was linked to, printed elsewhere?

    Three witnesses, any one is enough. The same heading above the table (a table repeated in another
    year's filing, or another year's block of one table). A figure both print for a period they share:
    last year's column carried into this year's filing. Or, in another filing, no rival: it prints
    only one table with a line of this name. Two segments' tables in one filing have the same row
    labels and pass none of the three, so they stay apart."""
    if first is None:
        return True
    if member is not None and _member_in(first, row, member) is None:
        return False                                  # the column the link names is not printed in this row
    if heading_key(row) == heading_key(first):
        return True
    if row["doc"] == first["doc"]:
        return False
    mine, theirs = period_cells(first, member), period_cells(row, _member_in(first, row, member))
    for pid in set(mine) & set(theirs):
        a, b = first["cells"][mine[pid]].get("value"), row["cells"][theirs[pid]].get("value")
        if (isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b
                and len(str(abs(int(a))).rstrip("0")) >= 3):          # three significant digits: not a coincidence
            return True
    # No rival: the one line of this name in that filing. When the row prints a period of the linked
    # line, only rows that print one too are rivals: the other year blocks of the same table are not.
    # A restated figure has no shared figure to witness it, so this is how a later filing's comparative
    # column meets the earlier filing's line (22 Nov: Coca-Cola North America 2024 operating income,
    # 4,336 in the 2024 10-K, 4,556 in the 2025 10-K, whose segment columns were reordered and renamed).
    def prints_mine(r: dict) -> bool:
        k = _member_in(first, r, member)
        return (member is None or k is not None) and bool(set(period_cells(r, k)) & set(mine))
    same_doc = [r for r in namesakes if r["doc"] == row["doc"]]
    if prints_mine(row):
        return len({heading_key(r) for r in same_doc if prints_mine(r)}) == 1
    return len({heading_key(r) for r in same_doc}) == 1


def statement_scale(g: Graph, doc: str) -> Optional[float]:
    """The scale a filing's primary statements print in (1e6 for "in millions"): a statement cell's
    value over its printed figure, the commonest power of ten. None if the filing has no statements."""
    cache = g.__dict__.setdefault("_statement_scale", {})
    if doc not in cache:
        seen: dict = defaultdict(int)
        for r in g.readings.values():
            if r.doc != doc or r.kind not in ("IS", "BS", "CF") or not isinstance(r.value, (int, float)) or not r.value:
                continue
            printed = re.sub(r"[^\d.]", "", str(r.text))
            if printed and float(printed):
                ratio = abs(r.value) / float(printed)
                power = round(math.log10(ratio)) if ratio > 0 else None
                if power is not None and abs(ratio - 10 ** power) <= 1e-6 * 10 ** power:
                    seen[power] += 1
        cache[doc] = float(10 ** max(seen, key=seen.get)) if seen else None
    return cache[doc]


def _only_printer(first: dict, row: dict, pool: list, member: Optional[int]) -> bool:
    """Is `row` the one row of its filing, among rows of this label and table title, that prints a
    period the linked line `first` prints (in the column the link names)?"""
    mine = set(period_cells(first, member))

    def prints(r: dict) -> bool:
        k = _member_in(first, r, member)
        return (member is None or k is not None) and bool(set(period_cells(r, k)) & mine)
    if not prints(row):
        return False
    return len({heading_key(r) for r in pool if r["doc"] == row["doc"] and prints(r)}) == 1


def _member_in(first: dict, row: dict, member: Optional[int]) -> Optional[int]:
    """The index in `row` of the column `first`'s link names, found by its printed name: another
    filing may reorder or rename the other columns. None when `row` does not print that column."""
    if member is None or row is first or row["id"] == first["id"]:
        return member
    names = members(first)
    if not 0 <= member < len(names):
        return None
    bare = lambda n: norm_label(n.replace(" (the sum of the others)", ""))  # noqa: E731
    theirs = [bare(n) for n in members(row)]
    want = bare(names[member])
    return theirs.index(want) if want in theirs else None


def shares_words(row: dict, mention: dict) -> bool:
    """Does a table row that prints a sentence's figure also share a content word with the sentence
    (its label, section or title)? A tie by value alone can be a coincidence: on 22 Nov Procter &
    Gamble's "advertising costs ... $10.2 billion" tied to its "Dividends to shareholders" cell of the
    same size, and two pickers who had chosen the right sentence were served the dividends."""
    soft = {stem(w) for w in SOFT}
    words = {w for w in tokens(" ".join((row.get("label", ""), row.get("section", ""), row.get("title", ""))))
             if not w.isdigit() and len(w) > 2 and w not in soft}
    said = {w for w in tokens(mention.get("masked") or mention.get("sentence", "")) if not w.isdigit()}
    return bool(words & said)


def lexical_ties(st: Store, mention: dict) -> list:
    """The sentence's ties to table cells that also share a word with it (see shares_words)."""
    out = []
    for cid in mention.get("ties") or []:
        row = st.by_id.get(str(cid).rpartition(":c")[0])
        if row is not None and shares_words(row, mention):
            out.append(cid)
    return out


def _tied_row(st: Store, mention: dict, slot: str, wanted: set) -> Optional[tuple]:
    """(row, cell index) of the table cell that prints the chosen sentence figure exactly, for the period asked.

    Sentences round ("$16.6 billion"); the table cell the sentence ties to holds the exact amount and
    sits in the table's arithmetic. The match is by value, in code; the model never sees either number.
    """
    number = next((n for n in mention["numbers"] if n["slot"].strip("<>") == slot), None)
    v = number.get("value") if number else None
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not v:
        return None
    for cid in mention.get("ties") or []:
        rid, _, ci = str(cid).rpartition(":c")
        row = st.by_id.get(rid)
        if row is None or not ci.isdigit() or int(ci) >= len(row.get("cells", [])):
            continue
        if not shares_words(row, mention):
            continue                                  # equal by value only: a coincidence, not the same figure
        j = int(ci)
        cell = row["cells"][j]
        if (_cell_pid(row, j) in wanted and isinstance(cell.get("value"), (int, float))
                and abs(abs(cell["value"]) - abs(v)) <= max(1.0, 0.006 * abs(v))):
            return row, j
    return None


def _as_int(x) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    return int(x) if isinstance(x, str) and x.strip().isdigit() else None


def apply_link(g: Graph, entry: dict) -> int:
    """Create the facts a link stands for. Returns the number of readings added."""
    st, ticker, key = store(), entry["ticker"], entry["item"]
    status_cap = None if entry["state"] == "active" else "unverified"
    added = 0
    if entry["kind"] == "mention":
        m = st.by_id.get(entry["ref_id"])
        number = next((n for n in (m or {}).get("numbers", []) if n["slot"].strip("<>") == entry["slot"]), None)
        if number is None:
            return 0
        rid = f"{m['id']}:{entry['slot']}"
        pid = entry.get("period") or TIMELESS         # the period the sentence states for the figure, if it states one
        value, unit, status = number["value"], number["unit"], "verified" if lexical_ties(st, m) else "text"
        if unit == "USD_scale_unknown":
            # "$2,842 and $2,670" in a note that prints the statements' millions without saying so: read in
            # the scale of the filing's own statements, and never trusted beyond FLAGGED
            scale = statement_scale(g, m["doc"])
            if abs(value) >= 1e7:                     # "$346,760,150,000" (a market value) is printed in dollars
                scale = 1.0
            value, unit, status = (value * scale if scale else value), "USD", "unscaled"
        elif number.get("scale_from"):
            status = "scaled_from_table"              # the scale was taken from the table printed above
        g.readings.setdefault(rid, Reading(
            id=rid, doc=m["doc"], ticker=ticker, page=m["page"], bbox=[], table=f"{m['doc']}:text", kind="TEXT",
            row=0, col=0, label=m["sentence"][:200], section="", text=number["text"], value=value,
            unit=unit, period=pid, period_end="", status=status))
        fact = g.facts.setdefault(f"{ticker}|{key}|{pid}", Fact(ticker, key, pid, [], "active"))
        if rid not in fact.readings:
            fact.readings.append(rid)
            added += 1
    else:
        first = st.by_id.get(entry.get("row_id") or entry.get("ref_id"))
        want_label = norm_label(entry["label"])
        want_ctx = norm_label(entry.get("section") or entry.get("title") or "")
        namesakes = [r for r in st.rows_by_ticker.get(ticker, [])
                     if norm_label(r["label"]) == want_label and norm_label(r["section"] or r["title"] or "") == want_ctx]
        # In another filing the same line may carry another section: a later year block of a segment
        # table is filed under the block's first heading. Those rows (same label, same table title) are
        # accepted only as the one row of that filing that prints a period of the linked line.
        want_title = norm_label(entry.get("title") or "")
        loose = [r for r in st.rows_by_ticker.get(ticker, [])
                 if first is not None and want_title and r["doc"] != first["doc"] and r not in namesakes
                 and norm_label(r["label"]) == want_label and norm_label(r["title"] or "") == want_title]
        for row in st.rows_by_ticker.get(ticker, []):
            same_row = first is not None and row["id"] == first["id"]
            if row in loose:
                if not _only_printer(first, row, namesakes + loose, entry.get("member")):
                    continue
            elif not same_row and (row not in namesakes or not _same_line(first, row, namesakes, entry.get("member"))):
                continue                              # the same label under another heading is another table
            # the cell that is the row's figure for each period (the column the link names, if it names one,
            # found by its name in a row of another filing)
            k = entry.get("member") if same_row or first is None else _member_in(first, row, entry.get("member"))
            if entry.get("member") is not None and k is None:
                continue
            stands_for = {j: pid for pid, j in period_cells(row, k).items()}
            for j, c in enumerate(row["cells"]):
                if entry.get("column") is not None:
                    if c.get("col", "") != entry["column"] or not same_row:
                        continue
                    pid = TIMELESS
                elif j in stands_for:
                    pid = stands_for[j]
                else:
                    continue
                rid = c.get("rid") or f"{row['id']}:c{j}"   # a primary-statement cell is already a reading
                if rid not in g.readings:
                    status = "verified" if c.get("ok") else "unverified"
                    g.readings[rid] = Reading(
                        id=rid, doc=row["doc"], ticker=ticker, page=row["page"], bbox=c.get("bbox", []),
                        table=f"{row['doc']}:raw{row['table']}", kind="RAW", row=0, col=j, label=row["label"],
                        section=(row["section"] or row["title"])[:120], text=c["text"], value=c["value"],
                        unit=row["unit"], period=pid, period_end=c.get("end") or "", status=status_cap or status)
                fact = g.facts.setdefault(f"{ticker}|{key}|{pid}", Fact(ticker, key, pid, [], "active"))
                if rid not in fact.readings:
                    fact.readings.append(rid)
                    added += 1
    if entry.get("picked") in WITNESSED and entry["state"] == "active":
        if not hasattr(g, "witnessed_links"):
            g.witnessed_links = set()
        g.witnessed_links.add((ticker, key))          # two models picked this line independently
    if entry.get("second_reading"):
        if not hasattr(g, "second_readings"):
            g.second_readings = {}
        g.second_readings[(ticker, key)] = entry["second_reading"]
    if added or f"{ticker}|{key}|{TIMELESS}" in g.facts or any(k.startswith(f"{ticker}|{key}|") for k in g.facts):
        keys = aliases(g).setdefault(_phrase_key(ticker, entry["phrase"]), [])
        if key in keys:
            keys.remove(key)
        keys.append(key)
    return added


# ------------------------------------------------------------------ the parts of a total

def breakdown(g: Graph, ticker: str, phrase: str) -> Optional[list]:
    """The lines that add up to a linked total, or None if the phrase is not linked yet: the rows
    under a subtotal, or the columns of a row that sum to its total column."""
    total_id, by, witnessed = getattr(g, "breakdown_alias", {}).get(_phrase_key(ticker, phrase), (None, "rows", False))
    total = store().by_id.get(total_id) if total_id else None
    if total is None:
        return None
    if by == "columns":
        return [dict(line, witnessed=witnessed) for line in across(store(), total)] or None
    kids, how = store().parts(total)
    return [dict(_line(k), grouped_by=how, witnessed=witnessed) for k in kids]


_ADJUSTED = re.compile(r"effect|impact|currenc|exchange|adjust|exclud|constant|prior year rates|change|increase|"
                       r"decrease|variance|non-gaap", re.I)
_YEARISH = re.compile(r"\b(19|20)\d\d\b|\b(january|february|march|april|may|june|july|august|september|october|"
                      r"november|december)\b\.?,?\s*\d{0,2},?|\|", re.I)


def column_groups(row: dict) -> dict:
    """{period id: the row's cells for that period, in print order}.

    Usually one cell. Several when the period is printed over a group of columns: U.S., International
    and Total under one year; cost, depreciation and net; as reported and at constant currency. A year
    printed once over three undated columns is such a group too, when one of them sums the others."""
    cells = row["cells"]
    groups: dict = defaultdict(list)
    for j in range(len(cells)):
        pid = _cell_pid(row, j)
        if pid:
            groups[pid].append(j)
    for pid, js in list(groups.items()):
        if len(js) == 1:
            j = js[0]
            around = [w for w in ([j - 1, j, j + 1], [j, j + 1, j + 2], [j - 2, j - 1, j])
                      if w[0] >= 0 and w[-1] < len(cells) and all(k == j or not _cell_pid(row, k) for k in w)]
            groups[pid] = next((w for w in around if _sum_of_others(row, w) is not None), js)
    return dict(groups)


def default_cell(row: dict, js: list) -> int:
    """The cell that stands for the row when nobody names a column of the group `js`.

    A column of adjustments (currency effects, changes) means the first plain column is the figure as
    reported. Otherwise arithmetic decides: the column that is the sum of the others is their total,
    whatever the headers say. Failing that, a column headed "total", else the first column."""
    heads = [(row["cells"][j].get("col") or "") for j in js]
    if any(_ADJUSTED.search(h) for h in heads):
        return next((j for j, h in zip(js, heads) if not _ADJUSTED.search(h)), js[0])
    summed = _sum_of_others(row, js)
    if summed is not None:
        return summed
    return next((j for j, h in zip(js, heads) if "total" in h.lower()), js[0])


def members(row: dict) -> list:
    """Names of the columns one period spans, in print order ([] when every period is a single cell):
    the header without its dates, with the arithmetic total marked."""
    js = max(column_groups(row).values(), key=len, default=[])
    if len(js) < 2:
        return []
    total = _sum_of_others(row, js)
    names = []
    for k, j in enumerate(js):
        names.append(_column_name(row, j, k)[:32] + (" (the sum of the others)" if j == total else ""))
    return names


def _column_name(row: dict, j: int, k: int) -> str:
    """What the page prints over column j: its header words, without dates."""
    cell = row["cells"][j]
    return " ".join(_YEARISH.sub(" ", cell.get("head") or cell.get("col") or "").split()) or f"column {k + 1}"


def period_cells(row: dict, member: Optional[int] = None) -> dict:
    """{period id: the cell index that stands for the row in that period}: the `member`-th column of
    each period's group when a link names one, else the default cell."""
    return {pid: js[member] if member is not None and 0 <= member < len(js) else default_cell(row, js)
            for pid, js in column_groups(row).items()}


def _sum_of_others(row: dict, js: list) -> Optional[int]:
    """The cell among `js` (one period's columns, in print order) that totals the others, by arithmetic.

    Reading left to right, a figure that equals the sum of the two or more figures before it closes
    them into one, and may be summed again further on: regions add up to external sales, external
    plus inter-segment sales add up to the total. The last figure that closed others is the row's
    figure for the period. None when no figure is a sum of others."""
    values = [row["cells"][j].get("value") for j in js]
    if len(js) < 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    totals = _sums(values)
    return js[totals[-1]] if totals else None


def _sums(values: list) -> list:
    """Positions of the figures that are sums of the figures before them, in print order."""
    open_parts: list = []                             # figures not yet folded into a total
    totals: list = []
    for k, v in enumerate(values):
        for n in range(len(open_parts), 1, -1):       # the longest run that adds up to this figure
            if v and abs(v - sum(open_parts[-n:])) <= max(1e-9, 0.004 * abs(v)):
                open_parts = open_parts[:-n]
                totals.append(k)
                break
        open_parts.append(v)
    return totals


def across(st: Store, row: dict) -> list:
    """The parts of a row printed across the page: one line per column that is not a sum of others.

    A segment table prints its segments as columns, and the consolidated column is their sum. The
    same arithmetic that finds a subtotal down a column finds it across a row, so the row is a
    breakdown by column. Blocks of the same table printed for other years are merged in by period.
    Returns [] unless the row has a total found by arithmetic and at least two named parts."""
    groups = column_groups(row)
    js = max(groups.values(), key=len, default=[])
    values = [row["cells"][j].get("value") for j in js]
    if len(js) < 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return []
    totals = set(_sums(values))
    parts = [k for k in range(len(js)) if k not in totals]
    names = {k: _column_name(row, js[k], k) for k in parts}
    if not totals or len(parts) < 2 or sum(1 for n in names.values() if not n.startswith("column ")) < 2:
        return []
    key = (norm_label(row["label"]), norm_label(row["section"] or row["title"] or ""), heading_key(row))
    blocks = [r for r in st.rows_by_ticker.get(row["ticker"], [])
              if (norm_label(r["label"]), norm_label(r["section"] or r["title"] or ""), heading_key(r)) == key]
    lines = []
    for k in parts:
        cells = {}
        for r in blocks:
            for pid, group in column_groups(r).items():
                if len(group) == len(js) and pid not in cells:
                    c = r["cells"][group[k]]
                    cells[pid] = {"id": c.get("rid") or f"{r['id']}:c{group[k]}", "value": c["value"], "text": c["text"],
                                  "ok": c.get("ok")}
        lines.append({"ticker": row["ticker"], "doc": row["doc"], "page": row["page"], "label": names[k],
                      "unit": row["unit"], "cells": cells, "grouped_by": "columns", "of_row": row["id"], "member": k,
                      "width": len(js)})
    return lines


def same_table_figure(line: dict, words: str) -> Optional[dict]:
    """Another figure of a column line, read from the same table: the row whose label carries `words`,
    in the line's own column. None when the table prints no such row."""
    st = store()
    origin = st.by_id.get(line.get("of_row", ""))
    if origin is None:
        return None
    want = set(tokens(words))
    best = None
    for r in st.rows_by_ticker.get(origin["ticker"], []):
        if r["doc"] != origin["doc"] or heading_key(r) != heading_key(origin) or r["id"] == origin["id"]:
            continue
        label = set(tokens(r["label"]))
        if want and want <= label and (best is None or len(label) < len(set(tokens(best["label"])))):
            best = r
    if best is None:
        return None
    return next((ln for ln in across(st, best) if ln["member"] == line["member"] and ln["width"] == line["width"]), None)


def _line(row: dict) -> dict:
    cells = {pid: {"id": row["cells"][j].get("rid") or f"{row['id']}:c{j}", "value": row["cells"][j]["value"],
                   "text": row["cells"][j]["text"], "ok": row["cells"][j].get("ok")}
             for pid, j in period_cells(row).items()}
    return {"ticker": row["ticker"], "doc": row["doc"], "page": row["page"], "label": row["label"],
            "unit": row["unit"], "cells": cells}


def learn_breakdown(g: Graph, ticker: str, phrase: str, wanted: set, fiscal_year: str, question: str = "",
                    question_id: str = "", model: Optional[str] = None, offline: bool = False) -> dict:
    """Link a phrase such as "revenue by segment" to the subtotal row whose operands are its parts."""
    st = store()
    entry = {"kind": "breakdown", "ticker": ticker, "phrase": phrase, "taught_by": question_id, "question": question,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "rejected", "label": None, "evidence": [], "objections": []}
    want = set(tokens(phrase))
    scored = []
    for r in st.rows_by_ticker.get(ticker, []):
        # one line can be broken down two ways: by the rows that add up to it, and by the columns
        # that add up to its total column (segments printed across the page). Both are candidates.
        for kids, how in ((st.parts(r)[0], "rows"), (across(st, r), "columns")):
            if len(kids) < 2:
                continue
            words = set(tokens(" ".join([r["label"], r["section"], r["title"], r.get("context", ""), r.get("above", "")]
                                        + [k["label"] for k in kids])))
            hit = len(want & words)
            if not hit:
                continue
            scored.append((2.0 * hit + (1.5 if any(_cell_pid(r, j) in wanted for j in range(len(r["cells"]))) else 0)
                           + (1.0 if r.get("ties_to") else 0)
                           + (0.75 if g.filings[r["doc"]]["fiscal_year"] == fiscal_year else 0), r, kids, how))
    scored.sort(key=lambda x: -x[0])
    cands, seen = [], set()
    for item in scored:                               # the same subtotal reprinted in another filing adds nothing
        sig = (norm_label(item[1]["label"]), tuple(norm_label(k["label"]) for k in item[2]), item[3])
        if sig not in seen:
            seen.add(sig)
            cands.append(item)
    cands = cands[:16]
    if not cands:
        entry["objections"] = ["no subtotal in the company's filings matches the wording"]
        return entry
    lines = [f"{i}: SUBTOTAL ({g.filings[r['doc']]['form']} {g.filings[r['doc']]['fiscal_year']}, page {r['page']}): "
             f"label '{r['label'][:70]}' | section '{r['section'][:40]}' | table '{r['title'][:80]}' | "
             f"printed above the table '{r.get('above', '')[-150:]}' | "
             f"ties to {r.get('ties_to') or 'nothing bound'} | broken down by its {how}: "
             + "; ".join(k["label"][:40] for k in kids[:14])
             for i, (_s, r, kids, how) in enumerate(cands)]
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nBreakdown asked for: {phrase}\n"
            f"Question: {question}\n\nCandidates:\n" + "\n".join(lines))
    ans, case = _pick(SYSTEM_BREAKDOWN, user, "breakdown", model, offline)
    pick = _as_int(ans.get("pick")) if ans else None
    if pick is None or not 0 <= pick < len(cands):
        entry["objections"] = ["proposer found no subtotal"]
        return entry
    entry["reason"], entry["picked"] = str(ans.get("reason", ""))[:200], case
    total, how = cands[pick][1], cands[pick][3]
    entry.update(state="active" if not case or case in WITNESSED else "flagged", ref_id=total["id"], by=how,
                 label=total["label"], title=total["title"], page=total["page"], doc=total["doc"],
                 evidence=[f"its parts are the {how} that add up to it in the recovered arithmetic"]
                 + ([f"the subtotal prints the same value as {total['ties_to']}"] if total.get("ties_to") else []))
    apply_breakdown(g, entry)
    return entry


def apply_breakdown(g: Graph, entry: dict) -> None:
    if not hasattr(g, "breakdown_alias"):
        g.breakdown_alias = {}
    g.breakdown_alias[_phrase_key(entry["ticker"], entry["phrase"])] = (
        entry["ref_id"], entry.get("by", "rows"), entry.get("picked") in WITNESSED and entry.get("state") == "active")


# ------------------------------------------------------------------ persistence

def save_link(entry: dict, path: pathlib.Path = LEARNED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_links(g: Graph, path: pathlib.Path = LEARNED) -> int:
    """Re-apply every stored link to a freshly loaded graph. Returns how many were applied."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("state") not in ("active", "flagged"):
            continue
        if e.get("kind") in ("item", "mention") and e.get("item"):
            apply_link(g, e)
            n += 1
        elif e.get("kind") == "breakdown" and e.get("ref_id"):
            apply_breakdown(g, e)
            n += 1
    return n
