"""Line items outside the primary statements, bound on demand.

"What were Apple's iPhone net sales in fiscal 2026?" names something that is
no seed concept. It is printed in a note table that the raw layer has parsed
and checked but nobody has bound. When a plan asks for such an item:

  1. the raw rows of that company's latest filing are searched by wording
     (label, section, table title; never values);
  2. an LLM picks the row from the candidates' wording and structural role;
  3. the gate admits the pick only on structural evidence:
       - the row's numbers sit inside a passing formula of its table, or
       - the row's table total prints the same value as a fact of the base
         graph (the product table's total is Revenue), so the row is a
         verified component of that concept;
  4. the binding is stored with the question that taught it, and the same
     label in the same table of the company's other filings is bound too,
     so a later question about another year, or about a restatement, works.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Optional

from .concepts import norm_label
from .graph import Fact, Graph, Reading
from .llm import ask_json
from .living import LEARNED, REVIEW, latest_doc
from .rawlayer import load_raw, search

SYSTEM = """You match a phrase from a question to one row of a table in a company's SEC filing.
You are shown candidate rows: the printed row label, the section heading above it, the table's title,
the note it sits in, and the row's role in the table's arithmetic. You are never shown values.
Pick the single row that reports the item for the company (not a sub-component of it, not a
percentage, not a per-share figure), or answer null if none of the rows reports it.
Answer as JSON: {"row": <candidate number or null>, "reason": "<one short sentence>"}"""


def item_key(row: dict) -> str:
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", norm_label(s)).strip("-")[:48]
    return f"item:{slug(row['label'])}@{slug(row['section'] or row['title'])}"


def _phrase_key(ticker: str, phrase: str) -> tuple:
    return ticker, re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()


def aliases(g: Graph) -> dict:
    if not hasattr(g, "item_alias"):
        g.item_alias = {}
    return g.item_alias


def gate_item(row: dict) -> tuple:
    evidence, objections = [], []
    if not row["label"].strip():
        objections.append("row has no printed label")
    if row["verified"]:
        evidence.append(f"value sits inside a passing formula of its table ({row['role']})")
    if row["anchor"]:
        evidence.append(f"its table's total prints the same value as {row['anchor']}: a verified component of it")
    if row["ties_to"]:
        evidence.append(f"prints the same value as {row['ties_to']}")
    if not (row["verified"] or row["anchor"] or row["ties_to"]):
        objections.append("no formula covers the row and its table does not tie to any bound fact")
    if row["unit"] == "ratio":
        objections.append("row is a percentage or ratio, not an amount")
    return (not objections), evidence, objections


def learn_item(g: Graph, raw: list, ticker: str, phrase: str, question_id: str = "", question: str = "",
               model: Optional[str] = None, use_gate: bool = True, offline: bool = False) -> dict:
    doc = latest_doc(g, ticker)
    entry = {"kind": "item", "ticker": ticker, "phrase": phrase, "doc": doc, "taught_by": question_id,
             "question": question, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "rejected", "item": None,
             "label": None, "title": None, "section": None, "page": None, "evidence": [], "objections": []}
    cands = search(raw, ticker, doc, phrase) if doc else []
    if not cands:
        entry["objections"] = ["no row of the filing matches the wording"]
        return entry
    lines = [f"{i}: label '{r['label'][:80]}' | section '{r['section'][:50]}' | table '{r['title'][:90]}' | "
             f"note '{r['context'][:50]}' | role {r['role']} | unit {r['unit']}" for i, r in enumerate(cands)]
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nItem asked for: {phrase}\n"
            f"Question: {question}\n\nCandidate rows:\n" + "\n".join(lines))
    try:
        ans = ask_json(SYSTEM, user, model=model, tag="item", offline=offline)
    except Exception as e:  # noqa: BLE001
        entry["objections"] = [f"proposer failed: {type(e).__name__}"]
        return entry
    pick = ans.get("row")
    if isinstance(pick, str) and pick.strip().isdigit():
        pick = int(pick)
    entry["reason"] = str(ans.get("reason", ""))[:200]
    if not isinstance(pick, int) or not 0 <= pick < len(cands):
        entry["objections"] = ["proposer found no row"]
        return entry
    row = cands[pick]
    entry.update(item=item_key(row), label=row["label"], title=row["title"], section=row["section"], page=row["page"],
                 row_id=row["id"])
    ok, evidence, objections = gate_item(row)
    entry["evidence"], entry["objections"] = evidence, objections
    if ok or not use_gate:
        entry["state"] = "active"
        apply_item(g, raw, entry)
    else:
        entry["state"] = "proposed"
    return entry


def apply_item(g: Graph, raw: list, entry: dict) -> int:
    """Create facts for the item in every filing of the company that prints the same row."""
    ticker, key = entry["ticker"], entry["item"]
    want_label, want_ctx = norm_label(entry["label"]), norm_label(entry["section"] or entry["title"] or "")
    added = 0
    for row in raw:
        if row["ticker"] != ticker or norm_label(row["label"]) != want_label:
            continue
        ctx = norm_label(row["section"] or row["title"] or "")
        if row["id"] != entry.get("row_id") and ctx != want_ctx:
            continue                                    # same label elsewhere in the filing: a different table
        for j, c in enumerate(row["cells"]):
            if not c["pid"]:
                continue
            rid = c.get("rid") or f"{row['id']}:c{j}"       # a primary-statement cell is already a reading
            if rid not in g.readings:
                g.readings[rid] = Reading(id=rid, doc=row["doc"], ticker=ticker, page=row["page"], bbox=c["bbox"],
                                          table=f"{row['doc']}:raw{row['table']}", kind="RAW", row=0, col=j,
                                          label=row["label"], section=(row["section"] or row["title"])[:120],
                                          text=c["text"], value=c["value"], unit=row["unit"], period=c["pid"],
                                          period_end=c["end"], status="verified" if c["ok"] else "unverified")
            fk = f"{ticker}|{key}|{c['pid']}"
            f = g.facts.setdefault(fk, Fact(ticker, key, c["pid"], [], "active"))
            if rid not in f.readings:
                f.readings.append(rid)
                added += 1
    aliases(g)[_phrase_key(ticker, entry["phrase"])] = key
    return added


def save_item(entry: dict) -> None:
    path = LEARNED if entry["state"] == "active" else REVIEW
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_items(g: Graph, raw: Optional[list] = None, path: pathlib.Path = LEARNED) -> int:
    """Re-apply learned item bindings to a freshly loaded graph."""
    if not path.exists():
        return 0
    entries = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    entries = [e for e in entries if e.get("kind") == "item" and e.get("state") == "active"]
    if not entries:
        return 0
    raw = raw if raw is not None else load_raw()
    return sum(apply_item(g, raw, e) for e in entries)


def resolve(g: Graph, ticker: str, phrase: str) -> Optional[str]:
    return aliases(g).get(_phrase_key(ticker, phrase))
