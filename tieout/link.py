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
from .llm import ask_json

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEARNED = ROOT / "semantics" / "learned.jsonl"
RAW_PATH = ROOT / "graph" / "raw.json.gz"
MENTIONS_PATH = ROOT / "graph" / "mentions.json.gz"
TIMELESS = "*"                                    # period id of a figure whose source prints no period
MAX_CANDIDATES = 18
STOP = {"the", "of", "and", "in", "for", "to", "a", "an", "by", "its", "total", "net", "from", "on", "at", "s",
        "was", "what", "how", "much", "many", "did", "does", "is", "were", "had", "have", "with", "that", "this"}

SYSTEM = """You match a phrase from a question to one place in a company's SEC filing.
You are shown numbered candidates. A ROW is a line of a table: its printed label, the section heading
above it, the table's title, the note it sits in, its column headers and its role in the table's
arithmetic. A SENTENCE is a statement from the text with every number replaced by a slot such as <n1>.
You are never shown values. Pick the one candidate that reports the ITEM itself for the company (not a
component of it, not a percentage of it unless the item is a percentage).
The question is given only as context. It may need several figures, other periods or arithmetic; those
are handled elsewhere and are not your concern. A candidate that reports the item is the right pick
even though it does not answer the whole question. Rows marked "columns: periods" hold every period
the filing prints; the period is chosen later.
For a ROW whose columns are not periods, also give the number of the column that holds the item.
For a SENTENCE, give the slot that holds the item's figure.
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

def tokens(text: str) -> list:
    out = []
    for w in re.findall(r"[a-z0-9&]+", norm_label(text)):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                              # stores / store
        out.append(w)
    return out


def _phrase_key(ticker: str, phrase: str) -> tuple:
    return ticker, " ".join(tokens(phrase))


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_label(text)).strip("-")[:n]


def candidates(st: Store, g: Graph, ticker: str, phrase: str, wanted: set, fiscal_year: str) -> list:
    """Rows and sentences of this company whose wording overlaps the phrase, best first.

    `wanted` are the period ids the question asks for; a row that prints one of them, or a
    filing of the asked fiscal year, is preferred. Wording only: no value is consulted.
    """
    want = set(tokens(phrase))
    if not want:
        return []
    scored = []
    for r in st.rows_by_ticker.get(ticker, []):
        label = set(tokens(r["label"]))
        context = set(tokens(" ".join((r["section"], r["title"], r.get("context", ""),
                                       " ".join(c.get("col", "") for c in r["cells"])))))
        hit_label, hit_context = len(want & label), len(want & (context - label))
        if hit_label == 0 and hit_context < 2:
            continue
        has_period = any(c.get("pid") in wanted for c in r["cells"])
        score = (3.0 * hit_label + 1.0 * hit_context - 0.15 * len(label - want)
                 + (1.5 if has_period else 0.0) + (0.5 if r["verified"] else 0.0)
                 + (0.5 if r.get("anchor") or r.get("ties_to") else 0.0)
                 + (0.75 if g.filings[r["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + 2.0 * (len(want & (label | context)) == len(want)))
        scored.append((score, "row", r))
    for m in st.mentions_by_ticker.get(ticker, []):
        words = set(tokens(m["masked"]))
        hit = len(want & words)
        if hit < max(2, (len(want) + 1) // 2):
            continue
        score = (2.0 * hit - 0.03 * len(words - want) + (0.5 if m.get("ties") else 0.0)
                 + (0.75 if g.filings[m["doc"]]["fiscal_year"] == fiscal_year else 0.0)
                 + 2.0 * (hit == len(want)))
        scored.append((score, "sentence", m))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _score, kind, ref in scored:                 # the same row reprinted in an older filing adds nothing
        sig = (kind, norm_label(ref.get("label") or ref.get("masked", "")), norm_label(ref.get("section", "")),
               g.filings[ref["doc"]]["fiscal_year"] if kind == "sentence" else "")
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"kind": kind, "ref": ref})
        if len(out) == MAX_CANDIDATES:
            break
    return out


def _columns(row: dict) -> list:
    """Column headers of a row's cells, in print order, without repeats."""
    return list(dict.fromkeys(c.get("col", "") for c in row["cells"]))


def _show(i: int, cand: dict, g: Graph) -> str:
    ref = cand["ref"]
    filing = f"{g.filings[ref['doc']]['form']} {g.filings[ref['doc']]['fiscal_year']}"
    if cand["kind"] == "sentence":
        return f"{i}: SENTENCE ({filing}, page {ref['page']}): {ref['masked'][:320]}"
    dated = any(c.get("pid") for c in ref["cells"])
    cols = "periods" if dated else "; ".join(f"[{j}] {h[:40]}" for j, h in enumerate(_columns(ref)))
    return (f"{i}: ROW ({filing}, page {ref['page']}): label '{ref['label'][:80]}' | section '{ref['section'][:50]}' | "
            f"table '{ref['title'][:90]}' | note '{ref.get('context', '')[:50]}' | columns: {cols} | "
            f"role {ref['role']} | unit {ref['unit']}")


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
    cands = candidates(st, g, ticker, phrase, wanted, fiscal_year)
    if not cands:
        entry["objections"] = ["nothing in the company's filings matches the wording"]
        return entry
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nItem asked for: {phrase}\n"
            f"Question: {question}\n\nCandidates:\n" + "\n".join(_show(i, c, g) for i, c in enumerate(cands)))
    try:
        ans = ask_json(SYSTEM, user, model=model, tag="link", offline=offline)
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
        slot = str(ans.get("slot") or (ref["numbers"][0]["slot"] if len(ref["numbers"]) == 1 else "")).strip("<>")
        if slot not in {n["slot"].strip("<>") for n in ref["numbers"]}:
            entry["objections"] = ["proposer named no number slot of the sentence"]
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
                elif c.get("pid"):
                    pid = c["pid"]
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
    cells = {c["pid"]: {"id": c.get("rid") or f"{row['id']}:c{j}", "value": c["value"], "text": c["text"], "ok": c.get("ok")}
             for j, c in enumerate(row["cells"]) if c.get("pid")}
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
        ans = ask_json(SYSTEM_BREAKDOWN, user, model=model, tag="breakdown", offline=offline)
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
