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

import gzip
import json
import pathlib
import re
import time
from collections import defaultdict
from typing import Optional

from .concepts import norm_label
from .graph import Fact, Graph, Reading
from .llm import DailyLimit, ask_json, link_model

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
For a SENTENCE, give the slot that holds the item's figure for the period asked (a count for "number of",
an amount for revenue or spending, a percentage only when a percentage is asked).
Answer null only if no candidate reports the item.
Answer as JSON: {"pick": <candidate number or null>, "column": <column number or null>,
"slot": "<slot or null>", "reason": "<one short sentence>"}"""

SYSTEM_BREAKDOWN = """You match a phrase from a question to one subtotal in a company's SEC filing.
Each numbered candidate is a subtotal row of a table: its printed label, section, table title, and the
labels of the rows that add up to it. You are never shown values. Pick the subtotal whose parts are
the breakdown the question asks about (by segment, by product, by region, by type of expense ...).
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


def tokens(text: str) -> list:
    return [stem(w) for w in re.findall(r"[a-z0-9&]+", norm_label(text)) if w not in STOP]


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
    m = re.search(r"_FY(\d{4})", str(row.get("doc") or ""))   # doc ids carry the filing's fiscal year
    limit = int(m.group(1)) if m else 9999
    return f"FY{year}{suffix}" if year <= limit else None


def _cell_pid(row: dict, j: int) -> Optional[str]:
    return row["cells"][j].get("pid") or _inferred_pid(row, j)


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
        if any(c.get("pid") for c in r["cells"]) and not any(c.get("pid") in wanted for c in r["cells"]) \
                and not any(_inferred_pid(r, j) in wanted for j in range(len(r["cells"]))):
            continue                                  # prints other periods only
        label = set(tokens(r["label"]))
        ctx_text = " ".join((r["section"], r["title"], r.get("context", ""), " ".join(c.get("col", "") for c in r["cells"])))
        context = set(tokens(ctx_text))
        hit_label = _hits(core, phrase, r["label"], label)
        hit_context = _hits(core, phrase, ctx_text, context) - hit_label
        if not hit_label and len(hit_context) < min(2, len(core)):
            continue
        covered = len(hit_label | hit_context) == len(core)
        score = (3.0 * len(hit_label) + 1.0 * len(hit_context) - 0.15 * len(label - want)
                 + (1.5 if any(c.get("pid") in wanted for c in r["cells"]) else 0.0) + (0.5 if r["verified"] else 0.0)
                 + (0.5 if r.get("anchor") or r.get("ties_to") else 0.0)
                 + (0.75 if g.filings[r["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + 2.0 * covered + _unit_fit(r["unit"], want_unit))
        scored.append((score, "row", r, covered))
    for m in st.mentions_by_ticker.get(ticker, []):
        words = set(tokens(m["masked"]))
        hit = _hits(core, phrase, m["masked"], words)
        if len(hit) < max(1, -(-3 * len(core) // 5)):   # at least three fifths of the core words
            continue
        units = {n.get("unit") for n in m["numbers"]}
        fit = max((_unit_fit(u, want_unit) for u in units), default=0.0)
        score = (2.0 * len(hit) - 0.03 * len(words - want) + (0.5 if m.get("ties") else 0.0)
                 + (0.75 if g.filings[m["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + 2.0 * (len(hit) == len(core)) + fit)
        scored.append((score, "sentence", m, len(hit) == len(core)))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for score, kind, ref, covered in scored:         # the same row reprinted in an older filing adds nothing
        sig = (kind, norm_label(ref.get("label") or ref.get("masked", "")), norm_label(ref.get("section", "")),
               g.filings[ref["doc"]]["fiscal_year"] if kind == "sentence" else "")
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
        ans = ask_json(SYSTEM_LABELS, user, model=link_model(model), tag="labels", offline=offline)
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


def _show(i: int, cand: dict, g: Graph) -> str:
    ref = cand["ref"]
    filing = f"{g.filings[ref['doc']]['form']} {g.filings[ref['doc']]['fiscal_year']}"
    if cand["kind"] == "sentence":
        return f"{i}: SENTENCE ({filing}, page {ref['page']}): {shown_sentence(ref)[:360]}"
    dated = any(c.get("pid") for c in ref["cells"])
    cols = "periods" if dated else "; ".join(f"[{j}] {h[:40]}" for j, h in enumerate(_columns(ref)))
    return (f"{i}: ROW ({filing}, page {ref['page']}): label '{ref['label'][:80]}' | section '{ref['section'][:50]}' | "
            f"table '{ref['title'][:90]}' | note '{ref.get('context', '')[:50]}' | columns: {cols} | "
            f"role {ref['role']} | unit {ref['unit']}")


def shown_sentence(m: dict) -> str:
    """The sentence as the linker sees it: dates and years printed (they name periods, not
    figures), every other number a slot tagged with its unit, such as <n1 count> or <n2 USD>."""
    text = m["masked"]
    for n in sorted(m["numbers"], key=lambda n: -len(n["slot"])):    # <n10> before <n1>
        inner = n["slot"].strip("<>")
        shown = n["text"] if n.get("unit") in ("date", "year") else f"<{inner} {n.get('unit') or 'number'}>"
        text = text.replace(n["slot"], shown)
    return text


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


def resolve(g: Graph, ticker: str, phrase: str) -> Optional[str]:
    """The fact key an item phrase was linked to for this company, if it has been."""
    return aliases(g).get(_phrase_key(ticker, phrase))


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
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nItem asked for: {phrase}\nPeriod asked: {period}\n"
            f"Question: {question}\n\nCandidates:\n" + "\n".join(_show(i, c, g) for i, c in enumerate(cands)))
    try:
        ans = ask_json(SYSTEM, user, model=link_model(model), tag="link", offline=offline)
    except DailyLimit:
        raise
    except Exception as e:  # noqa: BLE001 - a failed proposal is recorded, not fatal
        entry["objections"] = [f"proposer failed: {type(e).__name__}"]
        return entry
    pick = _as_int(ans.get("pick"))
    entry["reason"] = str(ans.get("reason", ""))[:200]
    if pick is None or not 0 <= pick < len(cands):
        entry["objections"] = ["proposer found no candidate"]
        return entry
    cand = cands[pick]
    ref = cand["ref"]
    if cand["kind"] == "sentence":
        said = str(ans.get("slot") or (ref["numbers"][0]["slot"] if len(ref["numbers"]) == 1 else ""))
        slot = (re.search(r"n\d+", said) or re.search(r".*", "")).group(0)      # "<n1 count>" or "n1" -> "n1"
        if slot not in {n["slot"].strip("<>") for n in ref["numbers"]}:
            entry["objections"] = ["proposer named no number slot of the sentence"]
            return entry
        unit = next(n.get("unit") for n in ref["numbers"] if n["slot"].strip("<>") == slot)
        if _unit_fit(unit, expected_unit(phrase)) < 0:
            entry["objections"] = [f"the number picked is in {unit}, the item asks for {expected_unit(phrase)}"]
            return entry
        entry.update(kind="mention", item=f"said:{ref['id']}:{slot}", ref_id=ref["id"], slot=slot,
                     label=ref["masked"][:160], page=ref["page"], doc=ref["doc"])
    else:
        column = None
        if not any(c.get("pid") for c in ref["cells"]):
            headers = _columns(ref)
            j = _as_int(ans.get("column"))
            column = headers[j] if j is not None and 0 <= j < len(headers) else (headers[0] if len(headers) == 1 else None)
            if column is None:
                entry["objections"] = ["the row's columns are not periods and the proposer named no column"]
                return entry
        key = f"item:{_slug(ref['label'])}@{_slug(ref['section'] or ref['title'])}" + (f"#{_slug(column, 32)}" if column else "")
        entry.update(item=key, ref_id=ref["id"], row_id=ref["id"], column=column, label=ref["label"],
                     title=ref["title"], section=ref["section"], page=ref["page"], doc=ref["doc"])
    confirmed, entry["evidence"], entry["objections"] = gate(cand)
    entry["state"] = "active" if confirmed else "flagged"
    apply_link(g, entry)
    return entry


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
        g.readings.setdefault(rid, Reading(
            id=rid, doc=m["doc"], ticker=ticker, page=m["page"], bbox=[], table=f"{m['doc']}:text", kind="TEXT",
            row=0, col=0, label=m["sentence"][:200], section="", text=number["text"], value=number["value"],
            unit=number["unit"], period=TIMELESS, period_end="", status="verified" if m.get("ties") else "text"))
        fact = g.facts.setdefault(f"{ticker}|{key}|{TIMELESS}", Fact(ticker, key, TIMELESS, [], "active"))
        if rid not in fact.readings:
            fact.readings.append(rid)
            added += 1
    else:
        first = st.by_id.get(entry.get("row_id") or entry.get("ref_id"))
        want_label = norm_label(entry["label"])
        want_ctx = norm_label(entry.get("section") or entry.get("title") or "")
        for row in st.rows_by_ticker.get(ticker, []):
            same_row = first is not None and row["id"] == first["id"]
            if not same_row and (norm_label(row["label"]) != want_label
                                 or norm_label(row["section"] or row["title"] or "") != want_ctx):
                continue                              # the same label elsewhere in the filing is another table
            for j, c in enumerate(row["cells"]):
                if entry.get("column") is not None:
                    if c.get("col", "") != entry["column"] or not same_row:
                        continue
                    pid = TIMELESS
                elif _cell_pid(row, j):
                    pid = _cell_pid(row, j)
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
    if added or f"{ticker}|{key}|{TIMELESS}" in g.facts or any(k.startswith(f"{ticker}|{key}|") for k in g.facts):
        aliases(g)[_phrase_key(ticker, entry["phrase"])] = key
    return added


# ------------------------------------------------------------------ the parts of a total

def breakdown(g: Graph, ticker: str, phrase: str) -> Optional[list]:
    """The lines that add up to a linked subtotal, or None if the phrase is not linked yet."""
    total_id = getattr(g, "breakdown_alias", {}).get(_phrase_key(ticker, phrase))
    total = store().by_id.get(total_id) if total_id else None
    if total is None:
        return None
    return [_line(store().by_id[i]) for i in total.get("children", []) if i in store().by_id]


def _line(row: dict) -> dict:
    cells = {_cell_pid(row, j): {"id": c.get("rid") or f"{row['id']}:c{j}", "value": c["value"], "text": c["text"],
                                 "ok": c.get("ok")}
             for j, c in enumerate(row["cells"]) if _cell_pid(row, j)}
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
        kids = [st.by_id[i] for i in r.get("children", []) if i in st.by_id]
        if len(kids) < 2:
            continue
        words = set(tokens(" ".join([r["label"], r["section"], r["title"], r.get("context", "")]
                                    + [k["label"] for k in kids])))
        hit = len(want & words)
        if not hit:
            continue
        scored.append((2.0 * hit + (1.5 if any(c.get("pid") in wanted for c in r["cells"]) else 0)
                       + (1.0 if r.get("ties_to") else 0)
                       + (0.75 if g.filings[r["doc"]]["fiscal_year"] == fiscal_year else 0), r, kids))
    scored.sort(key=lambda x: -x[0])
    cands = scored[:12]
    if not cands:
        entry["objections"] = ["no subtotal in the company's filings matches the wording"]
        return entry
    lines = [f"{i}: SUBTOTAL ({g.filings[r['doc']]['form']} {g.filings[r['doc']]['fiscal_year']}, page {r['page']}): "
             f"label '{r['label'][:70]}' | section '{r['section'][:40]}' | table '{r['title'][:80]}' | "
             f"ties to {r.get('ties_to') or 'nothing bound'} | parts: " + "; ".join(k["label"][:40] for k in kids[:14])
             for i, (_s, r, kids) in enumerate(cands)]
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nBreakdown asked for: {phrase}\n"
            f"Question: {question}\n\nCandidates:\n" + "\n".join(lines))
    try:
        ans = ask_json(SYSTEM_BREAKDOWN, user, model=link_model(model), tag="breakdown", offline=offline)
    except DailyLimit:
        raise
    except Exception as e:  # noqa: BLE001
        entry["objections"] = [f"proposer failed: {type(e).__name__}"]
        return entry
    pick = _as_int(ans.get("pick"))
    entry["reason"] = str(ans.get("reason", ""))[:200]
    if pick is None or not 0 <= pick < len(cands):
        entry["objections"] = ["proposer found no subtotal"]
        return entry
    total = cands[pick][1]
    entry.update(state="active", ref_id=total["id"], label=total["label"], title=total["title"], page=total["page"],
                 doc=total["doc"], evidence=["its parts are the operands of the formula recovered for this subtotal"]
                 + ([f"the subtotal prints the same value as {total['ties_to']}"] if total.get("ties_to") else []))
    apply_breakdown(g, entry)
    return entry


def apply_breakdown(g: Graph, entry: dict) -> None:
    if not hasattr(g, "breakdown_alias"):
        g.breakdown_alias = {}
    g.breakdown_alias[_phrase_key(entry["ticker"], entry["phrase"])] = entry["ref_id"]


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
