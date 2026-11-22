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
from .textfacts import find_gutter, make_flow

ROOT = pathlib.Path(__file__).resolve().parents[1]
MENTIONS_PATH = ROOT / "graph" / "mentions.json.gz"
MAX_NUMBERS = 6
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
NOUNS = (r"employees|associates|team members|people|workers|stores|warehouses|clubs|locations|restaurants|"
         r"vehicles|cars|units|deliveries|customers|members|cardholders|subscribers|users|accounts|"
         r"square feet|countries|markets|facilities|branches|ATMs|patents|holders of record|stockholders of record|"
         r"shareholders of record|barrels|wells|acres|aircraft|properties|offices|sites|households|"
         # counted things a filing states in a sentence or a footnote (22 Nov: "136 of the 184 leases",
         # "1,659 million adjusted scripts" had no record)
         r"leases|scripts|prescriptions|transactions|cards|shipments|franchisees|dealers|dealerships|suppliers|"
         r"patients|policies|loans|centers|plants|rooms|seats|merchants|sellers|developers|positions|engines|"
         r"turbines|installations|orders|packages|flights|beds|hospitals|clinics|pharmacies|physicians|"
         r"participants|beneficiaries|lawsuits|claims|states|cities|memberships|devices|servers|clients|"
         r"investors|advisors|borrowers|passengers|students|hires|jobs|rigs|trucks|vessels|hotels|kiosks")
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
# "136 of the 184 leases": a part of a counted whole
OF_TOTAL_RE = re.compile(rf"^\s+of\s+(?:the|our|its|these|those)\s+\d[\d,.]*\s+(?:{GLUE}\s+){{0,2}}?(?:shares|{NOUNS})\b", re.I)
# a currency sign other than the dollar right before the figure (the PDF text often garbles it)
FOREIGN_MONEY = "€£¥₩₹�"
# "Level 3 loans", "Note 5", "Tier 1": the number names something, it does not count
REFERENCE_BEFORE_RE = re.compile(r"\b(?:asc|asu|ias|ifrs|sfas|fasb|note|notes|item|items|level|tier|table|part|section|topic|rule|form|series|class|"
                                 r"stage|phase|schedule|exhibit|figure|page|chapter|article|title|regulation|scope|"
                                 r"category|group|step|no\.|number)\s*$", re.I)
# "25 million euros": money in another currency is not a count
CURRENCY_AFTER_RE = re.compile(r"^\s+(?:of\s+)?(?:U\.S\.\s+)?(?:dollars|euros?|pounds|yen|yuan|renminbi|francs|pesos|rupees|"
                               r"reais|won|EUR|GBP|JPY|CNY|CAD|CHF|MXN|INR)\b", re.I)
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
    # the last line of a sentence that lists years ("2024 and 2023, respectively.", JPM p70) is short and holds
    # two numbers, but it ends the sentence: dropped as a table line, it took with it the years that date the
    # figures, and all three figures of "$63.4 billion, $47.4 billion and $41.4 billion" read as one year
    if re.search(r"[A-Za-z]{5,}[.;]$", ln[-1].text):
        return False
    return len(ln) <= 6 and sum(_is_value(w.text) for w in ln) >= 2


def paragraphs(page: dict, page_no: int) -> list:
    """Running text of a page as paragraphs, table lines and furniture removed."""
    return [text for text, _ in paragraph_blocks(page, page_no)]


FOOTNOTE_MARK_RE = re.compile(r"^(?:\(?[a-z0-9]{1,2}\)|\*{1,3}|\d{1,2})$")


def _footnote_line(ln: list) -> bool:
    """A footnote under a table: a marker ("(3)", "(a)", "*", a bare "1") set off from a sentence.
    The gap after the marker makes it look like a table line; the words after it are prose."""
    return len(ln) > 5 and bool(FOOTNOTE_MARK_RE.match(ln[0].text)) and not _table_line(ln[1:]) and \
        sum(1 for w in ln[1:] if re.search(r"[a-z]{3}", w.text)) >= 4


def paragraph_blocks(page: dict, page_no: int) -> list:
    """[(paragraph text, y of its first line)]: the y says which table on the page sits above it."""
    return _blocks(page_lines_of(page, page_no), footnotes=True)


def page_lines_of(page: dict, page_no: int) -> list:
    """The text lines of a page, or [] for a table of contents or an index.

    A page whose two columns the reading flow did not split (it finds no gutter when one column is a table
    that runs on below the other: JPM p70, a metrics table on the left and its footnotes on the right) is
    read column by column as the table reader splits it; read across, each footnote line was glued to a
    table line, dropped with it, and the sentence "Firmwide mortgage origination volume was $63.4 billion,
    $47.4 billion and $41.4 billion ..." never became a record."""
    lines = [ln for ln in make_flow(page_no, page).lines if ln]
    if find_gutter(page["words"], page["w"]) is None:
        from .alltables import _side_by_side, _streams          # alltables does not import mentions
        from .statements import page_lines
        raw = page_lines(page)
        streams = [part for s in _streams(raw, page["w"]) for part in _side_by_side(s)]
        if len(streams) >= 2 and _clean_split(raw, streams):
            lines = [ln for s in streams for ln in s if ln]
    ends_in_number = sum(1 for ln in lines if re.fullmatch(r"\d{1,3}", ln[-1].text))
    if lines and ends_in_number >= max(8, 0.4 * len(lines)):
        return []
    return lines


def _clean_split(raw: list, streams: list) -> bool:
    """Every printed line the streams cut in two is cut at a wide gap (a gutter between two columns, JPM p70), never
    between two words of one sentence: a full-width paragraph cut at a false gutter lost "billion" from "$19.9
    billion" (XOM p64, GM p40, AMD p58: red team 15) and glued halves of two sentences (META p122)."""
    where = {}
    for k, s in enumerate(streams):
        for ln in s:
            for w in ln:
                where[(round(w.x0, 1), round(w.y0, 1), w.text)] = k
    for ln in raw:
        words = sorted(ln, key=lambda w: w.x0)
        for a, b in zip(words, words[1:]):
            ka = where.get((round(a.x0, 1), round(a.y0, 1), a.text))
            kb = where.get((round(b.x0, 1), round(b.y0, 1), b.text))
            if ka is not None and kb is not None and ka != kb and b.x0 - a.x1 < 12:
                return False
    return True


def _blocks(lines: list, footnotes: bool) -> list:
    """Paragraphs of the running text. With footnotes=True a footnote line under a table starts a
    paragraph of its own instead of being dropped as a table line (buildfix 2)."""
    out, cur, prev = [], [], None
    for ln in lines:
        text = " ".join(w.text for w in ln)
        note = footnotes and _table_line(ln) and _footnote_line(ln)
        table = _table_line(ln) and not note
        broken = note or prev is not None and (ln[0].y0 - prev[-1].y1 > 1.6 * (prev[0].y1 - prev[0].y0) or
                                                ln[0].y0 < prev[0].y0)
        if table or FURNITURE_RE.search(text.strip()) or broken:
            if cur:
                out.append(cur)
            cur = []
            if table or FURNITURE_RE.search(text.strip()):
                prev = None
                continue
        cur.append((text, ln[0].y0))
        prev = ln
    if cur:
        out.append(cur)
    blocks = []
    for para in out:
        s = ""
        for t, _ in para:
            s = s[:-1] + t if s.endswith("-") and not s.endswith(" -") else (s + " " + t if s else t)
        blocks.append((s, para[0][1]))
    return blocks


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
                if noun and SCALE_BETWEEN_RE.search(after[:noun.start("noun")]):
                    noun = None                           # "151 net thousands acres": a count whose scale is unclear
                yearish = re.fullmatch(r"(19|20)\d\d", m.group("n")) and not m.group("scale")
                if noun and yearish and (after[:noun.start("noun")].strip(" ,") or after.lstrip().startswith(",")):
                    noun = None                           # "fiscal 2026, we repurchased shares": a year
                foreign = before.rstrip()[-1:] in FOREIGN_MONEY       # "€4.3 billion": money, not a count
                labelled = REFERENCE_BEFORE_RE.search(before)        # "Level 3 loans", "Note 5": a name
                if foreign or labelled:
                    noun = None
                if noun:
                    unit, value = ("shares" if noun.group("noun").lower() == "shares" else "count"), n * scale
                elif COUNT_BEFORE_RE.search(before):
                    unit, value = "count", n * scale
                elif not yearish and unit_token(after):
                    unit, value = unit_token(after), n * scale      # "46.7 GWh": the unit as printed
                elif not yearish and not labelled and OF_TOTAL_RE.match(after):
                    unit, value = "count", n * scale                # "136 of the 184 leases": a count of them
                elif m.group("scale") and not foreign and not CURRENCY_AFTER_RE.match(after):
                    # "1,659 million adjusted scripts": a scale word with no dollar sign counts something
                    unit, value = "count", n * scale
                elif not yearish and not labelled and not foreign and _open_noun(after):
                    # any plural noun after a bare number counts it ("12 reportable segments", "1,700 suppliers"
                    # with "suppliers" listed, "214 franchise partners" with "partners" not): NOUNS stays a
                    # gazetteer, and a noun outside it is marked so the answer can say it was not on the list
                    unit, value = "count", n * scale
                else:
                    unit, value = ("year", n) if yearish else ("number", n * scale)
            text = text.strip()
        if isinstance(value, float):
            value = int(round(value)) if abs(value - round(value)) < 1e-6 * max(1.0, abs(value)) else round(value, 6)
        nums.append({"slot": f"<n{k}>", "text": text, "value": value, "unit": unit})
        if unit == "count" and not m.group("date"):
            counted = NOUN_AFTER_RE.match(sentence[m.end():m.end() + 160])
            if counted and counted.group("noun").lower() != "shares":
                nums[-1]["noun"] = counted.group("noun").lower()
            elif _open_noun(sentence[m.end():m.end() + 160]):
                nums[-1]["noun"] = _open_noun(sentence[m.end():m.end() + 160])
                nums[-1]["noun_listed"] = False
        if unit not in CUED and unit not in ("date", "year", "number"):
            nums[-1]["unit_token"] = True
        # only the digits are masked: "$<n1> billion", "<n2>%" keep their unit words
        s, e = m.span("date") if m.group("date") else m.span("n")
        parts.append(sentence[last:s] + f"<n{k}>")
        last = e
    parts.append(sentence[last:])
    return "".join(parts), nums


OPEN_NOUN_RE = re.compile(rf"^{LIST_TAIL},?\s+(?:{GLUE}\s+){{0,2}}?(?P<noun>[a-z][a-z-]{{2,}}s)\b(?![-’'])", re.I)
# plural-looking words that do not count anything after a number: spans of time, money, rates, repetitions, and
# singular words that end in s ("gross", "basis", "status", "various", "across", "less")
NOT_COUNTED = {"seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "years", "decades", "periods",
               "times", "basis", "points", "cents", "dollars", "euros", "pounds", "francs", "pesos", "rupees", "yen",
               "votes", "percentages", "levels", "tiers", "notes", "items", "parts", "sections", "rules", "forms",
               "pages", "exhibits", "schedules", "steps", "stages", "phases", "series", "classes", "grades",
               "ratings", "basis-points", "bps", "x", "fold", "plus", "thus", "always", "versus", "whereas",
               "perhaps", "unless", "across", "afterwards", "towards", "besides", "sometimes", "nevertheless",
               "includes", "represents", "remains", "reflects", "requires", "results", "increases", "decreases",
               "declines", "rises", "grows", "falls", "means", "consists", "contains", "operates", "owns", "has",
               "was", "is", "as", "its", "this", "thus", "us", "lbs", "ms",
               # money is never a count of things (red team 17: JNJ p40 "10.4 non-cash expenses" in billions)
               "expenses", "costs", "charges", "revenues", "sales", "earnings", "losses", "gains", "assets",
               "liabilities", "payments", "proceeds", "fees", "taxes", "savings", "receivables", "payables",
               "dividends", "investments", "securities", "obligations", "commitments", "reserves", "impairments",
               "contributions", "purchases", "repurchases", "credits", "debts", "returns", "borrowings", "deposits",
               "benefits", "premiums", "royalties", "rents", "wages", "salaries", "bonuses", "awards-related",
               # verbs in -s after a standard's number ("ASC 820 establishes", "ASU 2023-09 requires")
               "establishes", "defines", "provides", "applies", "addresses", "amends", "clarifies", "eliminates",
               "specifies", "permits", "allows", "governs", "relates", "describes", "sets", "expands", "modifies",
               "supersedes", "simplifies", "replaces", "introduces", "removes", "improves", "enhances", "changes",
               "affects", "covers", "states", "notes",
               # scale words are not what is counted ("151 net thousands acres")
               "thousands", "millions", "billions", "trillions", "hundreds", "dozens", "tens"}
SCALE_BETWEEN_RE = re.compile(r"\b(?:thousands?|millions?|billions?|trillions?)\b", re.I)


def _open_noun(after: str) -> Optional[str]:
    """The plural noun a bare number counts ("12 reportable segments" -> "segments"), or None: a word that ends
    in s but is no plural ("gross", "basis", "previous"), that measures time, money or a rate, or a verb is not
    one; nor is a noun with a scale word before it ("151 net thousands acres": the count's scale is unclear)."""
    m = OPEN_NOUN_RE.match(after)
    if not m:
        return None
    noun = m.group("noun").lower()
    if noun in NOT_COUNTED or noun.endswith(("ss", "us", "is", "ous", "as")) or \
            SCALE_BETWEEN_RE.search(after[:m.start("noun")]):
        return None
    return noun


def _iso(text: str) -> str:
    mon, day, year = re.match(r"(\w+)\s+(\d{1,2}),\s+(\d{4})", text).groups()
    return f"{year}-{MONTHS.split('|').index(mon.capitalize()) + 1:02d}-{int(day):02d}"


CUED = ("USD", "USD/share", "percent", "count", "shares")

# a unit printed right after a figure ("46.7 GWh", "276 BCF", "12 MMcf"): a closed list of physical
# units. An open pattern (any short token with two capitals) was tried on the 55 filings and dropped: it
# caught "12 CFR", "15 USC", "Scope 1 GHG", "Tier 1 HQ", "24/7 CFE" as often as a real unit.
KNOWN_UNITS = (r"GWh|MWh|kWh|TWh|GW|MW|kW|Bcfe?|BCFE?|Tcfe?|TCFE?|MMcfe?d?|MMCFE?D?|Mcfe?|MCFE?|MMBtu|MMBTU|Btu|"
               r"MMBOE|MBOED|MBOE|BOED|BOE|MBD|MBPD|KBD|MMbbls?|Mbbls?|bbls?|sq\.\s?ft\.?|miles|kilometers|gallons|"
               r"metric tons|tons|tonnes|ounces|acres|hectares|terawatt[- ]hours|gigawatt[- ]hours|"
               r"megawatt[- ]hours|megawatts|gigawatts|petabytes|terabytes")
UNIT_AFTER_RE = re.compile(rf"^\s?({KNOWN_UNITS})(?![\w-])")


def unit_token(after: str) -> Optional[str]:
    """The physical unit printed right after a figure, as printed ("GWh"), or None."""
    m = UNIT_AFTER_RE.match(after)
    return m.group(1) if m else None


def cued(n: dict) -> bool:
    """A number whose unit the sentence states: CUED, or a printed unit token."""
    return n["unit"] in CUED or bool(n.get("unit_token"))


def _table_residue(s: str, years: bool = True) -> bool:
    """A row of a table that reached the running text: more numbers than words. With years=False the
    years are not counted, so a footnote pairing figures with "fiscal 2026, 2025 and 2024" is prose."""
    numbers = len(re.findall(r"\d[\d,.]*", s if years else re.sub(r"\b(?:19|20)\d\d\b", " ", s)))
    return numbers > 2 * MAX_NUMBERS or numbers > 4 and numbers > len(re.findall(r"[A-Za-z]{3,}", s))


def _kept_before(s: str) -> bool:
    """Whether the rule before buildfix 2 made a mention of this sentence (for its id only)."""
    if len(s) < 20 or len(s) > 1500 or not re.search(r"[a-z]{3}", s) or _table_residue(s):
        return False
    old_cued = [n for n in numbers_of(s)[1] if n["unit"] in CUED]
    return bool(old_cued) and len(old_cued) <= MAX_NUMBERS


def page_mentions(doc: str, ticker: str, page_no: int, page: dict, fiscal_year: Optional[int] = None) -> list:
    """Mentions of one page. With `fiscal_year` (a 10-K's), a list of figures paired with a list of
    years gives each figure its own period (pair_years). A sentence kept only because of such a
    list gets an id of its own ("...:p6:y0"), so the ids of the other sentences do not move."""
    lines = page_lines_of(page, page_no)
    # the ids the rules before buildfix 2 gave ("s0", "s1", ... in page order) stay with their sentences
    old_ids: dict = defaultdict(list)
    for para, _ in _blocks(lines, footnotes=False):
        for s in sentences(para):
            if _kept_before(s):
                old_ids[s].append(f"{doc}:p{page_no}:s{sum(len(v) for v in old_ids.values())}")
    out, extra = [], 0
    for para, y in _blocks(lines, footnotes=True):
        for s in sentences(para):
            if len(s) < 20 or len(s) > 1500 or not re.search(r"[a-z]{3}", s) or _table_residue(s, years=False):
                continue
            masked, nums = numbers_of(s)
            paired = pair_years(s, nums, fiscal_year) if fiscal_year else 0
            if old_ids.get(s):
                mid = old_ids[s].pop(0)
            else:
                now = [n for n in nums if cued(n)]
                if not now or len(now) > MAX_NUMBERS + paired:
                    continue
                mid, extra = f"{doc}:p{page_no}:y{extra}", extra + 1
            out.append({"id": mid, "doc": doc, "ticker": ticker, "page": page_no,
                        "sentence": s, "masked": masked, "numbers": nums, "ties": [], "_y": y})
    return out


# ------------------------------------------------ figures listed for several years (buildfix 2)
SEP_RE = re.compile(r"\s*,?\s*(?:and|or)?\s*")
# a list of years may repeat "fiscal" before each: "in fiscal 2025, fiscal 2024, and fiscal 2023 respectively" (WMT
# FY2025 p78: the three fuel-expense figures had no year)
YEAR_SEP_RE = re.compile(r"\s*,?\s*(?:and|or)?\s*(?:fiscal\s+(?:years?\s+)?)?")
YEAR_LEAD_RE = re.compile(r"\b(?:in|for|during|of|at|as of|through|fiscal|years?|end of|ended)\s+$", re.I)
RESPECTIVELY_RE = re.compile(r"^\s*,?\s*respectively\b", re.I)
SUBJECT_AFTER_RE = re.compile(rf"^\s+(?:of\s+)?(?:(?:total|the|our|its)\s+)?(?:{GLUE}\s+){{0,2}}?(?P<noun>shares|{NOUNS})\b",
                              re.I)
SUBJECT_BEFORE_RE = re.compile(rf"\b(?P<noun>shares|{NOUNS})\b", re.I)
PLAIN_INT_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")
WORD_LIST_RE = re.compile(r"[A-Za-z.]+,\s+[^,]{1,40},?\s+(?:and|or)\s+[A-Za-z]")


def _runs(sentence: str, spans: list, nums: list, years: bool) -> list:
    """Lists of two or more tokens of one kind (years, or figures) joined only by commas and
    "and": "42,300, 38,700, and 35,400" -> [[i, i+1, i+2]]."""
    runs, cur = [], []
    sep = YEAR_SEP_RE if years else SEP_RE
    for i, n in enumerate(nums):
        fits = n["unit"] != "date" and (n["unit"] == "year") == years
        joined = bool(cur) and bool(sep.fullmatch(sentence[spans[cur[-1]][1]:spans[i][0]]))
        if not fits or not joined:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
        if fits:
            cur.append(i)
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def pair_years(sentence: str, nums: list, fiscal_year: int) -> int:
    """Give each figure of a list its own period when the sentence pairs it with a list of years
    of the same length: "Executive members represented 42,300, 38,700, and 35,400 of total paid
    members in 2026, 2025, and 2024" -> FY2026, FY2025, FY2024 (key "pid"). "$1.2 billion,
    $1.1 billion and $0.9 billion in 2025, 2024 and 2023, respectively" the same.

    The years must be distinct, none after the filing's fiscal year nor more than four before it
    (so "notes due 2030 and 2032" is not a period list), and be introduced as periods ("in",
    "for", "fiscal", "during" ...) or followed by "respectively". A paired figure printed with no
    unit cue takes the counted noun of its sentence ("... of total paid members") as a count.
    Returns how many figures got a period."""
    spans = [m.span() for m in TOKEN_RE.finditer(sentence)]
    if len(spans) != len(nums):
        return 0
    got = 0
    figures = _runs(sentence, spans, nums, years=False)
    for yr in _runs(sentence, spans, nums, years=True):
        ys = [int(nums[i]["value"]) for i in yr]
        if len(set(ys)) != len(ys) or not all(fiscal_year - 4 <= y <= fiscal_year for y in ys):
            continue
        if not (YEAR_LEAD_RE.search(sentence[:spans[yr[0]][0]]) or RESPECTIVELY_RE.match(sentence[spans[yr[-1]][1]:])):
            continue
        for run in figures:
            if len(run) != len(yr) or any("pid" in nums[i] for i in run):
                continue
            # "At the end of 2026, 2025, and 2024 goodwill in the U.S., Canadian, and Other International
            # operations was $953, $26, and $15": a list of names between them pairs the figures with the names
            between = sentence[spans[yr[-1]][1]:spans[run[0]][0]] if yr[-1] < run[0] else \
                sentence[spans[run[-1]][1]:spans[yr[0]][0]]
            if WORD_LIST_RE.search(between) and not RESPECTIVELY_RE.match(sentence[spans[max(yr[-1], run[-1])][1]:]):
                continue
            unit = None
            if any(nums[i]["unit"] == "number" for i in run):
                # "was 7 million, 18 million and 14 million": the first figure's cue ("number of ...
                # was") is said of the whole list
                mates = {nums[i]["unit"] for i in run} - {"number"}
                unit = mates.pop() if len(mates) == 1 and mates <= {"count", "shares"} else \
                    _subject_unit(sentence, spans[run[0]][0], spans[run[-1]][1])
            for i, y in zip(run, ys):
                nums[i]["pid"] = f"FY{y}"
                if unit and nums[i]["unit"] == "number":
                    nums[i]["unit"] = unit
                got += 1
    return got


def _subject_unit(sentence: str, start: int, end: int) -> Optional[str]:
    """count (or shares) when the list's sentence names what it counts: the noun right after the
    list ("... of total paid members"), else the last counted noun before it in the sentence
    ("Executive members represented ...")."""
    m = SUBJECT_AFTER_RE.match(sentence[end:end + 80])
    if not m:
        before = list(SUBJECT_BEFORE_RE.finditer(sentence[max(0, start - 100):start]))
        m = before[-1] if before else None
    if not m:
        return None
    noun = m.group("noun").lower()
    stock_units = noun == "units" and re.search(r"stock\s+$", m.string[:m.start("noun")], re.I)
    return "shares" if noun == "shares" or stock_units else "count"


def page_tables(raw: list, doc: str) -> dict:
    """page -> [(bottom y, scale, smallest and largest printed value, unit)] for the tables of
    count or share rows of one filing whose numbers were all scaled by one factor: the scale the
    raw layer read from the table's caption ("Membership ... (in thousands)")."""
    by_table = defaultdict(list)
    for row in raw:
        if row["doc"] == doc:
            by_table[row["table_id"]].append(row)
    out = defaultdict(list)
    for rows in by_table.values():
        ratios, printed, units, bottom = set(), [], set(), 0.0
        for row in rows:
            for c in row["cells"]:
                bottom = max(bottom, c["bbox"][3] if c.get("bbox") else 0.0)
                unit = c.get("unit", row["unit"])
                text = (c.get("text") or "").strip("()$ ")
                if unit not in ("count", "shares") or not PLAIN_INT_RE.fullmatch(text) or \
                        not isinstance(c.get("value"), (int, float)):
                    continue
                p = float(text.replace(",", ""))
                if p:
                    ratios.add(round(abs(c["value"]) / p, 6))
                    printed.append(p)
                    units.add(unit)
        if len(ratios) == 1 and len(units) == 1 and next(iter(ratios)) in (1e3, 1e6, 1e9):
            out[rows[0]["page"]].append((bottom, next(iter(ratios)), min(printed), max(printed), units.pop()))
    return out


def table_scale(m: dict, tables: list, y: Optional[float]) -> None:
    """A listed figure printed as a plain whole number, in the same range as the printed values of
    the count table right above its paragraph, is in that table's scale: Costco's "Executive members
    represented 42,300 ..." under the membership table "(in thousands)" is 42.3 million. The figure is
    rescaled and marked `scale_from: "table above"` so a reader can say where the scale came from."""
    if y is None or not tables:
        return
    above = [t for t in tables if t[0] <= y]
    if not above:
        return
    _, scale, lo, hi, unit = max(above, key=lambda t: t[0])
    for n in m["numbers"]:
        if "pid" not in n or n["unit"] not in ("count", "shares", "number") or not PLAIN_INT_RE.fullmatch(n["text"]):
            continue
        if n["unit"] != "number" and n["unit"] != unit:
            continue
        if not lo / 10 <= n["value"] <= hi * 10:
            continue
        n["value"] = int(round(n["value"] * scale))
        n["unit"] = unit
        n["scale_from"] = "table above"


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
        # years printed in a 10-K's sentences are its fiscal-year labels; a 10-Q's "2025 and 2024"
        # usually means the same quarter of two years, which this rule does not date
        fiscal_year = int(f["fiscal_year"].lstrip("FY")) if f.get("form", "10-K") == "10-K" else None
        found = []
        for i, page in enumerate(pages):
            found.extend(page_mentions(doc, f["ticker"], i + 1, page, fiscal_year))
        tables = page_tables(raw, doc)
        for m in found:
            table_scale(m, tables.get(m["page"]), m.pop("_y", None))
        # a sentence kept only for its year-paired list stays only if one of its figures now has a unit
        found = [m for m in found if m["id"].rsplit(":", 1)[-1].startswith("s") or
                 any(cued(n) for n in m["numbers"])]
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
