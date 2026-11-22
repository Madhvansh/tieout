"""Numbers stated in sentences: "Digital Media ARR grew to $19.20 billion".

Tables carry most of a filing's numbers, but some are only ever written out in
running text: a KPI in MD&A, a buyback total, deliveries "approximately 1.64
million", a headcount. Each sentence of running text (outside tables) that
states a number with a unit cue becomes a record:

    id, doc, ticker, page
    sentence   as printed
    masked     the sentence with every number replaced by a slot <n1>, <n2>, ...
    numbers    [{"slot", "text", "value", "unit"}]: value in full units
    ties       ids of statement readings or raw cells of the same filing that
               print the same value (a rounded "$20.5 billion" ties to 20,451)

Unit cues: a dollar sign ($19.20 billion -> USD 19.2e9, "$6.62 per share" ->
USD/share), a percent sign or word (percent), a counted noun after the number
("939, 914 and 890 warehouses", "approximately 2.1 million associates" ->
count; "... shares" -> shares) or a count phrase before it ("store count of
2,359"). Numbers without a cue are still masked, as "year", "date" or
"number" slots, because the masked form is what a language model reads to pick
a sentence and a slot from wording alone: it must never see a number.

Skipped: lines that belong to tables (values set off by wide gaps), page
furniture, tables of contents, and sentences with more than six cued numbers.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Optional

from . import pdfio
from .textfacts import make_flow

ROOT = pathlib.Path(__file__).resolve().parents[1]
MENTIONS_PATH = ROOT / "graph" / "mentions.json.gz"
MAX_NUMBERS = 6
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
NOUNS = (r"employees|associates|team members|people|workers|stores|warehouses|clubs|locations|restaurants|"
         r"vehicles|cars|units|deliveries|customers|members|cardholders|subscribers|users|accounts|"
         r"square feet|countries|markets|facilities|branches|ATMs|patents|holders of record|stockholders of record|"
         r"shareholders of record|barrels|wells|acres|aircraft|properties|offices|sites|households")
TOKEN_RE = re.compile(
    rf"(?P<date>\b(?:{MONTHS})\s+\d{{1,2}},\s+(?:19|20)\d\d\b)"
    r"|(?<![\w.$-])(?P<dollar>\$\s?)?(?P<n>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w]|-[A-Za-z]|\.\d)"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion)\b)?"
    r"(?P<pct>\s?%|\s+percent\b|\s+percentage points?\b|\s+basis points\b)?", re.I)
# possessive (*+): a run of digits is never split again on the way back, which would take
# exponential time on a row of a table that reached the running text
LIST_TAIL = r"(?:\s*,?\s*(?:and|or|to)?\s*\$?\d[\d,.]*+(?:\s?(?:thousand|million|billion))?)*+"
# up to two describing words between the number and its noun ("1.64 million consumer vehicles"),
# never a pronoun, verb or preposition ("in fiscal 2026, we repurchased shares")
GLUE = r"(?!(?:we|our|us|they|it|its|the|a|an|of|in|on|at|to|for|by|from|with|and|or|was|were|is|are|had|has|" \
       r"have|be|been|which|that|than|per)\b)[a-zA-Z][\w’'-]*"
NOUN_AFTER_RE = re.compile(rf"^{LIST_TAIL},?\s+(?:{GLUE}\s+){{0,2}}?(?P<noun>shares|{NOUNS})\b", re.I)
COUNT_BEFORE_RE = re.compile(r"\b(?:count|headcount|number of [a-z ]{3,30}?)\s+(?:of|was|were|is|totaled|"
                             r"totaling|reached|increased to|decreased to)\s+(?:approximately\s+|about\s+)?$", re.I)
PER_SHARE_AFTER_RE = re.compile(r"^\s+per\s+(?:[a-z]+\s+){0,2}share\b", re.I)
BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[A-Z(“\"$•])|\s+•\s+")
ABBREV_RE = re.compile(r"(?:\b(?:Inc|Corp|Co|Ltd|No|Nos|Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|approx|U\.S|D\.C|N\.A|L\.P)|"
                       r"(?<![\w-])[A-Z])\.[\"”’)]?$")
FURNITURE_RE = re.compile(r"^table of contents$|form 10-[kq]\b.*\d{1,3}$|^\d{1,3}$|^page \d+", re.I)


def _is_value(text: str) -> bool:
    return bool(re.fullmatch(r"[$(]*-?[\d,.]+%?\)?|[—–-]|\$", text))


def _table_line(ln: list) -> bool:
    """A line of a table: a value set off from its neighbour by a wide gap."""
    for a, b in zip(ln, ln[1:]):
        gap = b.x0 - a.x1
        if gap > 40 or gap > 14 and (_is_value(a.text) or _is_value(b.text)):
            return True
    return len(ln) <= 6 and sum(_is_value(w.text) for w in ln) >= 2


def paragraphs(page: dict, page_no: int) -> list:
    """Running text of a page as paragraphs, table lines and furniture removed."""
    flow = make_flow(page_no, page)
    lines = [ln for ln in flow.lines if ln]
    ends_in_number = sum(1 for ln in lines if re.fullmatch(r"\d{1,3}", ln[-1].text))
    if lines and ends_in_number >= max(8, 0.4 * len(lines)):
        return []                                         # a table of contents or an index
    out, cur, prev = [], [], None
    for ln in lines:
        text = " ".join(w.text for w in ln)
        broken = prev is not None and (ln[0].y0 - prev[-1].y1 > 1.6 * (prev[0].y1 - prev[0].y0) or ln[0].y0 < prev[0].y0)
        if _table_line(ln) or FURNITURE_RE.search(text.strip()) or broken:
            if cur:
                out.append(cur)
            cur = []
            if _table_line(ln) or FURNITURE_RE.search(text.strip()):
                prev = None
                continue
        cur.append(text)
        prev = ln
    if cur:
        out.append(cur)
    texts = []
    for para in out:
        s = ""
        for t in para:
            s = s[:-1] + t if s.endswith("-") and not s.endswith(" -") else (s + " " + t if s else t)
        texts.append(s)
    return texts


def sentences(text: str) -> list:
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        if ABBREV_RE.search(text[max(0, m.start() - 8):m.start()]):
            continue
        out.append(text[start:m.start()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s.lstrip("• ").strip() for s in out if s.strip()]


def numbers_of(sentence: str) -> tuple:
    """(masked sentence, numbers) for one sentence."""
    nums, parts, last = [], [], 0
    for m in TOKEN_RE.finditer(sentence):
        k = len(nums) + 1
        if m.group("date"):
            text, value, unit = m.group("date"), _iso(m.group("date")), "date"
        else:
            text, n = m.group(0), float(m.group("n").replace(",", ""))
            scale = SCALE[m.group("scale").lower()] if m.group("scale") else 1.0
            after, before = sentence[m.end():m.end() + 160], sentence[max(0, m.start() - 60):m.start()]
            pct = (m.group("pct") or "").strip().lower()
            if pct:
                value, unit = (n / 100 if "basis" in pct else n), "percent"
            elif m.group("dollar"):
                unit = "USD/share" if PER_SHARE_AFTER_RE.match(after) and scale == 1.0 else "USD"
                value = n * scale
            else:
                noun = NOUN_AFTER_RE.match(after)
                yearish = re.fullmatch(r"(19|20)\d\d", m.group("n")) and not m.group("scale")
                if noun and yearish and after[:noun.start("noun")].strip(" ,"):
                    noun = None                           # "fiscal 2026, we repurchased shares": a year
                if noun:
                    unit, value = ("shares" if noun.group("noun").lower() == "shares" else "count"), n * scale
                elif COUNT_BEFORE_RE.search(before):
                    unit, value = "count", n * scale
                else:
                    unit, value = ("year", n) if yearish else ("number", n * scale)
            text = text.strip()
        if isinstance(value, float):
            value = int(round(value)) if abs(value - round(value)) < 1e-6 * max(1.0, abs(value)) else round(value, 6)
        nums.append({"slot": f"<n{k}>", "text": text, "value": value, "unit": unit})
        # only the digits are masked: "$<n1> billion", "<n2>%" keep their unit words
        s, e = m.span("date") if m.group("date") else m.span("n")
        parts.append(sentence[last:s] + f"<n{k}>")
        last = e
    parts.append(sentence[last:])
    return "".join(parts), nums


def _iso(text: str) -> str:
    mon, day, year = re.match(r"(\w+)\s+(\d{1,2}),\s+(\d{4})", text).groups()
    return f"{year}-{MONTHS.split('|').index(mon.capitalize()) + 1:02d}-{int(day):02d}"


CUED = ("USD", "USD/share", "percent", "count", "shares")


def _table_residue(s: str) -> bool:
    """A row of a table that reached the running text: more numbers than words."""
    numbers = len(re.findall(r"\d[\d,.]*", s))
    return numbers > 2 * MAX_NUMBERS or numbers > 4 and numbers > len(re.findall(r"[A-Za-z]{3,}", s))


def page_mentions(doc: str, ticker: str, page_no: int, page: dict) -> list:
    out = []
    for para in paragraphs(page, page_no):
        for s in sentences(para):
            if len(s) < 20 or len(s) > 1500 or not re.search(r"[a-z]{3}", s) or _table_residue(s):
                continue
            masked, nums = numbers_of(s)
            cued = [n for n in nums if n["unit"] in CUED]
            if not cued or len(cued) > MAX_NUMBERS:
                continue
            out.append({"id": f"{doc}:p{page_no}:s{len(out)}", "doc": doc, "ticker": ticker, "page": page_no,
                        "sentence": s, "masked": masked, "numbers": nums, "ties": []})
    return out


def _value_index(g, raw: list, doc: str) -> dict:
    """unit class -> [(value, node id)] for the statement readings and raw cells of one filing."""
    idx = defaultdict(list)
    for r in g.readings.values():
        if r.doc == doc:
            idx[r.unit].append((r.value, r.id))
    for row in raw:
        if row["doc"] != doc:
            continue
        for j, c in enumerate(row["cells"]):
            if c.get("rid") or not isinstance(c.get("value"), (int, float)):
                continue
            idx[c.get("unit", row["unit"])].append((c["value"], f"{row['id']}:c{j}"))
    return idx


def _precision(text: str, scale_word: Optional[str]) -> float:
    """Half a unit of the last printed digit, in full units: "$20.5 billion" -> 0.05e9."""
    digits = re.search(r"\d[\d,]*(?:\.(\d+))?", text)
    decimals = len(digits.group(1) or "") if digits else 0
    return 0.5 * 10 ** -decimals * (SCALE[scale_word] if scale_word else 1.0)


def tie(m: dict, idx: dict, limit: int = 10) -> list:
    ties = []
    for n in m["numbers"]:
        if n["unit"] not in CUED or not isinstance(n["value"], (int, float)):
            continue
        v = n["value"]
        if n["unit"] in ("USD", "count", "shares") and abs(v) < 100:
            continue                                      # small whole numbers match anything
        sw = re.search(r"(thousand|million|billion|trillion)", n["text"], re.I)
        tol = _precision(n["text"], sw.group(1).lower() if sw else None) if abs(v) >= 1e6 else 1e-9
        units = {"percent": ("percent", "ratio"), "USD/share": ("USD/share", "ratio")}.get(n["unit"], (n["unit"],))
        exact, near = [], []
        for u in units:
            for value, nid in idx.get(u, []):
                if not isinstance(value, (int, float)):
                    continue
                d = abs(abs(value) - abs(v))
                if d <= 1e-6 * max(1.0, abs(v)):
                    exact.append(nid)
                elif d <= tol:
                    near.append(nid)
        ties.extend(x for x in exact + near if x not in ties)
    return ties[:limit]


SCALE_WORD_RE = re.compile(r"\b(?:thousands?|millions?|billions?|trillions?)\b|\(in 000", re.I)
BARE_DOLLARS_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})+")
SCALE_UNKNOWN = "USD_scale_unknown"


def usd_values(idx: dict) -> set:
    """Absolute values of the USD cells and readings of one filing (from _value_index)."""
    return {abs(v) for v, _ in idx.get("USD", []) if isinstance(v, (int, float))}


def settle_scale(m: dict, values: set, statement_scale: Optional[float]) -> None:
    """Give a "$N,NNN" amount of a sentence that prints no scale word the scale it was written in.

    On a page printed "in millions", "long-term debt with fixed rates was $6,175" means 6,175
    million. The amount takes the scale under which it equals, exactly, a table cell or statement
    reading of the same filing: first as printed, then the filing's statement scale, then
    thousands, millions, billions. With no such tie its scale is unknown: the number is marked
    (`scale_unknown`, unit USD_scale_unknown) so no reader serves it as a plain dollar amount.
    A rescaled number records the factor in `rescaled`.
    """
    if SCALE_WORD_RE.search(m["sentence"]):
        return
    for n in m["numbers"]:
        if n["unit"] != "USD" or not BARE_DOLLARS_RE.fullmatch(n["text"].strip()):
            continue
        v = abs(float(n["value"]))
        tried = [1.0] + [s for s in (statement_scale, 1e3, 1e6, 1e9) if s and s != 1.0]
        hit = next((s for s in dict.fromkeys(tried) if round(v * s) in values), None)
        if hit is None:
            n["scale_unknown"] = True
            n["unit"] = SCALE_UNKNOWN
        elif hit != 1.0:
            n["value"] = int(round(n["value"] * hit))
            n["rescaled"] = hit


def build_mentions(g, raw: list, docs: Optional[list] = None, verbose: bool = True) -> list:
    out = []
    for doc, f in g.filings.items():
        if docs is not None and doc not in docs or not f.get("readable", True):
            continue
        pages = pdfio.load_pages(doc)
        found = []
        for i, page in enumerate(pages):
            found.extend(page_mentions(doc, f["ticker"], i + 1, page))
        idx = _value_index(g, raw, doc)
        scales = Counter(t["scale"] for t in g.tables.values() if t["doc"] == doc and t.get("scale"))
        statement_scale = scales.most_common(1)[0][0] if scales else None
        values = usd_values(idx)
        for m in found:
            settle_scale(m, values, statement_scale)
            m["ties"] = tie(m, idx)
        out.extend(found)
        if verbose:
            print(f"text   {doc:22s} {len(found):5d} mentions", flush=True)
    return out


def save_mentions(mentions: list, path: pathlib.Path = MENTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(mentions, f)


def load_mentions(path: pathlib.Path = MENTIONS_PATH) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
