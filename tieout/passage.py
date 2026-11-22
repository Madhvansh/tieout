"""A reader of the text: a witness and a pointer, never a source of answers.

The brief: "Answers must come from the graph, not from plain vector search over text chunks."

The graph answers from structure: a plan names lines, the executor reads them, arithmetic vouches for
them. That fails in two ways the arithmetic cannot see: the linker never found the line (a fact stated
in a sentence, a row under an unusual label), or the pickers chose a look-alike row from one shortlist.
A different method helps with both: the conventional pipeline this repository also measures as its
baseline (baseline/: passages of the filings, BM25 and dense vectors fused, one hosted model reads
them). Alone it is right about half as often as the graph, on different questions. So it is kept, with
two jobs and one rule.

  The rule     No figure, name or ranking the reader writes is ever served. Every served figure is a
               node of the graph, read by the executor, with the node's own page, scale and period.
  A witness    Its answer is compared with the graph's answer: the same figure corroborates, another
               figure is a dissent (readers.second_road, readers._disputed).
  A pointer    It is asked where one line of a plan is printed. The place it names (filing, page,
               figure) is looked up in the graph (node_for). The node found there is linked to the
               plan's phrase exactly as a picker's choice is (link.apply_link), and the plan runs again
               on the graph (by_pointing). The reader names the row; the column head still decides the
               period, the caption the scale, the executor the arithmetic. No node, no answer.

Every reply is kept under cache/passage/ keyed by the question, so a run replays with no index, no
embedding model and no key.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from typing import Optional

from .execute import Leaf, Result, concept_words, render
from .llm import DailyLimit

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "cache" / "passage"
VERSION = "passage-2"            # the baseline's v2 retriever in hybrid mode and its prompt, with BUDGET characters
BUDGET = 48_000                  # of filing text per question. The baseline keeps its conventional 16,000; three
                                 # times as much made the reader right on 20, 21 and 22 questions of rounds 9, 7
                                 # and 8 instead of 17, 20 and 20 (23 Nov, results/baseline_r<N>_hybrid_b48.json)
TOLERANCE = 0.005                # two methods round differently: figures within half a percent are the same figure
                                 # (the benchmark's own tolerance for a computed answer). One percent was too
                                 # wide: round 8 V07, 182,186 and 183,417 dollars per employee rest on two
                                 # different head-counts and were called the same
ROUNDED_TOLERANCE = 0.01         # ... one percent when one side is a figure of two or three digits (_a_round_figure)
SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
UNITS = {"usd": "USD", "ratio": "ratio", "shares": "shares", "count": "count", "usd/share": "USD/share"}
POINTED = "a reader of the text named this line; no picker chose it from the linker's shortlist"
MAX_LINES = 6                    # a plan that needs more lines pointed at than this is left alone


def _path(question: str, model: Optional[str]) -> pathlib.Path:
    key = hashlib.sha1(f"{VERSION}|{BUDGET}|{model or ''}|{question}".encode("utf-8")).hexdigest()[:24]
    return DIR / f"{key}.json"


def read(question: str, model: Optional[str] = None, offline: bool = False) -> Optional[dict]:
    """What the text reader says: {"abstain", "answer", "pred", "unit", "facts", "facts_ok"}, or None
    when it is not available (offline with no saved reply, no index, no quota)."""
    path = _path(question, model)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):                 # a reply another run is still writing: ask again
            saved = None
        if isinstance(saved, dict) and isinstance(saved.get("pred"), dict):
            _log(path)
            return saved
    if offline:
        return None
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from baseline import answer as ba
        out = ba.answer(question, model=model, retriever="v2", mode="hybrid", budget_chars=BUDGET)
    except DailyLimit:
        return None
    except Exception:  # noqa: BLE001 - no index, no embedding model: the graph still answers
        return None
    if out.get("error"):                              # a failed call is not an abstention: ask again next time
        return None
    shown = list(zip(out.get("retrieved_pages") or [], out.get("retrieved_text") or []))
    texts = {}
    for page, text in shown:
        texts[(page["doc"], page["page"])] = texts.get((page["doc"], page["page"]), "") + "\n" + text
    facts = []
    for f in out.get("facts_used") or []:
        f = dict(f)
        if not (f.get("in_context") and _printed(f.get("value"), texts.get((f.get("doc"), f.get("page")), ""))):
            # the reader sometimes writes a passage's number where the page belongs (6 of 66 answers, 23 Nov):
            # the fact is then looked for in the passages of that filing it was shown
            for page, text in shown:
                if page["doc"] == f.get("doc") and _printed(f.get("value"), text):
                    f.update(page=page["page"], in_context=True, page_found_by_figure=True)
                    break
        f["printed"] = bool(f.get("in_context")) and _printed(f.get("value"), texts.get((f.get("doc"), f.get("page")), ""))
        facts.append(f)
    pred = _numeric(out["pred"])
    reply = {"version": VERSION, "budget": BUDGET, "model": model, "question": question,
             "abstain": pred.get("kind") == "abstain", "answer": out.get("answer", ""), "pred": pred,
             "unit": out.get("unit"), "facts": facts,
             "facts_ok": bool(facts) and all(f.get("in_context") and f["printed"] for f in facts)}
    DIR.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")  # written whole, then moved: two runs may share a question
    scratch.write_text(json.dumps(reply, indent=1, default=str), encoding="utf-8")
    os.replace(scratch, path)
    _log(path)
    return reply


def _numeric(pred: dict) -> dict:
    """A figure the reader wrote as words ("33.8 billion", unit text) is a figure."""
    if pred.get("kind") == "text" and isinstance(pred.get("value"), str):
        try:
            from baseline.answer import to_number
            x = to_number(pred["value"])
        except Exception:  # noqa: BLE001
            x = None
        if x is not None:
            return {"kind": "number", "value": x}
    return pred


def _log(path: pathlib.Path) -> None:
    """Name the saved reply in the run's cache list (TIEOUT_CACHE_LOG), as llm.py does for model calls."""
    log = os.environ.get("TIEOUT_CACHE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"passage/{path.name}\n")


def _printed(value, text: str) -> bool:
    """Are the figure's leading digits printed in the passage it is cited from? The reader gives
    full units ("1.9 trillion" as 1900000000000), so only the significant digits can be compared."""
    if not _is_number(value):
        return False
    digits = re.sub(r"\D", "", f"{abs(value):.6f}".rstrip("0")).lstrip("0").rstrip("0")
    return len(digits) >= 1 and digits[:6] in re.sub(r"[^0-9]", "", text)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ----------------------------------------------------------------- the witness: is it the same answer?

def head(reply: dict, companies: dict) -> Optional[tuple]:
    """What of the reader's answer is compared with the graph's: ("n", figure), or ("s", the names
    of a ranking in order, as tickers where they are companies of the corpus)."""
    pred = reply.get("pred") or {}
    kind, value = pred.get("kind"), pred.get("value")
    if kind == "number" and _is_number(value):
        return ("n", float(value))
    if kind == "companies" and value:
        names = [v[0] if isinstance(v, (list, tuple)) else v for v in value]
        return ("s", tuple(_ticker(str(n), companies) for n in names))
    return None


def _ticker(name: str, companies: dict) -> str:
    """The company's ticker when the name is one of the corpus's companies, else the name as written."""
    from .eval.bench import company_to_ticker
    return company_to_ticker(name, companies) or name.strip()


def same(result: Result, shown: Optional[tuple]) -> bool:
    """Does the graph's result say what the text reader says? For a figure see _same_figures; for a
    choice between two, the same winner; for a ranking of three or more, the same order throughout."""
    if shown is None or not result.ok:
        return False
    if shown[0] == "s":
        if result.kind not in ("companies", "lines") or not result.value:
            return False
        mine = [str(n).strip().lower() for n, _ in result.value]
        theirs = [str(n).strip().lower() for n in shown[1]]
        if len(mine) <= 2:                            # which of two: the winner is the answer
            return bool(theirs) and theirs[0] == mine[0] and theirs == mine[:len(theirs)]
        return theirs == mine                         # a ranking of three or more: the whole order (round 8, W13:
        #                                               the same first name stood for two different rankings)
    figure = result.value if result.kind == "number" else (
        result.value[0][1] if result.kind == "lines" and result.value else None)
    if not _is_number(figure):
        return False
    return _same_figures(float(figure), shown[1], result.unit)


def _same_figures(x: float, y: float, unit: str) -> bool:
    """The comparison behind same() and same_number(), red-teamed on 23 Nov (research/19):
      the sign counts, except for a dollar amount (an outflow is printed with or without its minus);
      a factor of 100 is forgiven only for a rate (a percentage written for a fraction);
      a scale of thousands is forgiven (the reader leaves the page's "in millions" on a figure);
      the tolerance is half a percent, one percent when one side is a figure a sentence would round
      to two or three digits ("$6.7 billion" for 6,661 million), never more."""
    if unit == "USD":
        x, y = abs(x), abs(y)
    elif (x < 0) != (y < 0) and x != 0 and y != 0:
        return False
    factors = SCALES + ((100.0,) if unit in ("ratio", "percent", "percent_points") else ())
    for k in factors:
        for a, b in ((x, y * k), (x * k, y)):
            slack = ROUNDED_TOLERANCE if _a_round_figure(a) or _a_round_figure(b) else TOLERANCE
            if abs(a - b) <= slack * max(abs(a), abs(b), 1e-12):
                return True
    return False


def _a_round_figure(x: float) -> bool:
    """Two or three significant digits, the way a sentence states a table's figure. One digit
    (50, 0.3) says too little to count as a rounding of anything."""
    x = abs(x)
    if x == 0:
        return False
    from math import floor, log10
    top = floor(log10(x))
    three = x / 10 ** (top - 2)                       # the figure as a number between 100 and 999.99
    if abs(three - round(three)) > 1e-6:
        return False                                  # more than three significant digits
    return round(three) % 100 != 0                    # not a single digit followed by zeros


def same_number(x, y) -> bool:
    """Two figures with no unit known: the same sign, a scale of thousands forgiven, no factor of 100."""
    if not (_is_number(x) and _is_number(y)):
        return False
    return _same_figures(float(x), float(y), "")


def cited(reply: dict) -> str:
    """"DOC page N" of the first fact the reader used."""
    for f in reply.get("facts") or []:
        if f.get("doc"):
            return f"{f['doc']} page {f.get('page')}"
    return "its passages"


# ----------------------------------------------------------------- the pointer: from a place to a node
#
# The reader says where a figure is printed: a filing, a page, the figure. That is an address, not an
# answer. node_for looks the address up in the graph. What is served afterwards is the node: its value
# in the scale its table states (the reader often leaves "in millions" on a figure), its unit, its
# page, and for a table row the cell that the column head dates to the period the plan asks for.

_GROUPS = {"USD": "money", "USD_scale_unknown": "money", "USD/share": "money", "percent": "rate", "ratio": "rate",
           "year": None, "date": None}


def _group(unit: str) -> Optional[str]:
    """money, rate or count. None for a year or a date: never the figure of an answer."""
    return _GROUPS.get(unit, "count")


def _digits(x: float) -> int:
    """Significant digits of a figure (the zeros that end a whole number do not count): 25,400 has 3."""
    text = f"{abs(x):.6f}".rstrip("0").rstrip(".").replace(".", "").lstrip("0")
    return len(text.rstrip("0")) or (1 if text else 0)


def _round_to(x: float, digits: int) -> float:
    from math import floor, log10
    return round(x, digits - 1 - floor(log10(abs(x)))) if x else 0.0


def _scale_between(printed: float, said: float, unit: str) -> Optional[tuple]:
    """How a node's figure and the figure the reader wrote relate: (scale, exact) or None.
    scale: 1.0 the same number; 1e3, 1e6, ... the reader left the page's scale on it ("19,918" for a
    row in millions); 100 a rate written as a fraction where the page prints a percentage.
    exact: digit for digit. Otherwise the reader gave the figure to fewer digits than the page prints
    ("3.647 trillion" for 3,647,099,000,000): accepted when it has three digits or more and the node's
    figure rounds to it. 19,919 is not 19,918: stricter than _same_figures, because the node found here
    is what gets served."""
    if unit == "USD":
        printed, said = abs(printed), abs(said)
    elif (printed < 0) != (said < 0) and printed and said:
        return None
    factors = (1.0, 100.0) if unit in ("ratio", "percent") else (1.0,) if unit == "USD/share" else SCALES
    pairs = [(k, a, b) for k in factors for a, b in ((printed, said * k), (printed * k, said))]
    for k, a, b in pairs:
        if abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-12):
            return k, True
    digits = _digits(said)
    if digits >= 3:
        for k, a, b in pairs:
            if a and b and abs(_round_to(a, digits) - b) <= 1e-9 * abs(b):
                return k, False
    return None


def node_for(fact: dict, unit: str = "", g=None) -> Optional[dict]:
    """The node of the graph at the place the reader names: a table cell or a sentence figure of the
    cited filing, on the cited page or the page before or after it, that prints the reader's figure.

    Chosen in this order: the cited page before its neighbours; a figure printed digit for digit before
    one the reader rounded; the node whose label shares most words with what the reader called the
    fact; a table cell before a sentence; the figure as written before one that needs a scale. Refused:
    a node of another kind of unit than the reader's answer (dollars, a rate, a count), and a figure of
    fewer than three digits matched only through a scale and with no word in common ("12" would match
    any row that prints 12 million). None when no node qualifies."""
    from . import link
    doc, said = str(fact.get("doc") or ""), fact.get("value")
    if not doc or not _is_number(said):
        return None
    try:
        page = int(fact.get("page"))
    except (TypeError, ValueError):
        return None
    st, ticker = link.store(), doc.split("_")[0]
    want = _group(UNITS.get(str(unit).lower(), str(unit))) if unit else None
    words = set(link.tokens(str(fact.get("label") or "")))
    few = _digits(float(said)) < 3
    found = []

    def offer(node: dict, text: str, off: int, order: int) -> None:
        if _group(node["unit"]) is None or (want and _group(node["unit"]) != want):
            return
        fit = _scale_between(float(node["value"]), float(said), node["unit"])
        shared = len(words & set(link.tokens(text)))
        if fit is None or (fit[0] != 1.0 and few and not shared):
            return
        found.append(((off, not fit[1], -shared, order, fit[0] != 1.0), node))

    for r in st.rows_by_ticker.get(ticker, []):
        off = abs(int(r["page"]) - page)
        if r["doc"] != doc or off > 1:
            continue
        for j, c in enumerate(r["cells"]):
            if _is_number(c.get("value")):
                offer({"kind": "row", "id": c.get("rid") or f"{r['id']}:c{j}", "row_id": r["id"], "cell": j,
                       "ticker": ticker, "doc": doc, "page": int(r["page"]), "label": r["label"],
                       "printed": str(c.get("text")), "value": float(c["value"]), "unit": r["unit"],
                       "period": link._cell_pid(r, j) or "", "verified": bool(c.get("ok"))},
                      f"{r['label']} {r.get('section') or ''} {r.get('title') or ''}", off, 0)
    for m in st.mentions_by_ticker.get(ticker, []):
        off = abs(int(m["page"]) - page)
        if m["doc"] != doc or off > 1:
            continue
        for n in m.get("numbers") or []:
            value, node_unit = n.get("value"), str(n.get("unit") or "")
            if not _is_number(value):
                continue
            if node_unit == "USD_scale_unknown":      # "$2,842" in a note that prints the statements' millions
                if g is None and abs(value) < 1e7:
                    continue                          # its scale cannot be read here: not served
                scale = link.statement_scale(g, doc) if abs(value) < 1e7 else None
                value, node_unit = value * (scale or 1.0), "USD"
            offer({"kind": "sentence", "id": f"{m['id']}:{str(n.get('slot') or '').strip('<>')}", "ref_id": m["id"],
                   "slot": str(n.get("slot") or "").strip("<>"), "ticker": ticker, "doc": doc, "page": int(m["page"]),
                   "label": str(m.get("sentence") or "")[:200], "printed": str(n.get("text")), "value": float(value),
                   "unit": node_unit, "period": "", "verified": False}, str(m.get("sentence") or ""), off, 1)
    return min(found, key=lambda x: x[0])[1] if found else None


def node_of_answer(reply: dict, g=None) -> Optional[dict]:
    """The node behind a reader's answer that is ONE printed figure: the node of the fact it cites
    whose figure is its answer. None when the answer is the reader's own arithmetic, a ranking or
    words: those are never taken from the reader."""
    pred = reply.get("pred") or {}
    said = pred.get("value")
    if pred.get("kind") != "number" or not _is_number(said):
        return None
    unit = UNITS.get(str(reply.get("unit") or "").lower(), "")
    for f in reply.get("facts") or []:
        if not isinstance(f, dict) or not _is_number(f.get("value")):
            continue
        if _scale_between(float(f["value"]), float(said), unit) is None:
            continue                                  # a fact it used on the way, not the answer itself
        node = node_for(f, unit, g)
        if node is not None:
            return {**node, "cited": cited(reply)}
    return None


ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def fact_question(g, ex, ticker: str, words: str, spec: dict) -> Optional[str]:
    """One line of a plan as a question for the text reader."""
    if ticker not in g.companies or not words or not isinstance(spec, dict):
        return None
    try:
        year = ex._fy(ticker, spec)
        quarter = int(spec["q"]) if spec.get("q") else None
    except Exception:  # noqa: BLE001 - a period the executor cannot resolve, a quarter that is not a number
        return None
    if quarter and spec.get("ytd"):
        when = f"the first {3 * quarter} months of fiscal year {year} (year to date)"
    elif quarter:
        when = f"the {ORDINALS.get(quarter, str(quarter))} quarter of fiscal year {year}"
    else:
        when = f"fiscal year {year} (for a balance: at the end of fiscal year {year})"
    return f"What was {words} of {g.companies[ticker]['name']} for {when}? Give the one figure the filing prints for it."


def point(g, ex, ticker: str, words: str, spec: dict, model: Optional[str] = None, offline: bool = False) -> Optional[dict]:
    """The node for one line of a plan: the reader is asked the line's own small question, and the
    place it names is looked up. None: no answer, a fact not on the page it cites, or no node there."""
    question = fact_question(g, ex, ticker, words, spec)
    if question is None:
        return None
    reply = read(question, model=model, offline=offline)
    if not reply or reply.get("abstain") or not reply.get("facts_ok"):
        return None
    return node_of_answer(reply, g)


def link_entry(g, node: dict, ticker: str, phrase: str) -> Optional[dict]:
    """The linker's entry for a line the reader pointed at: what link.learn_link writes when a picker
    chooses a candidate, with its origin named and its state "flagged" (one model, no second witness).
    A table row is linked as a row, so the executor reads the cell its column head dates to the period
    asked, not the cell the reader happened to quote. Never saved to the store."""
    from . import link
    st = link.store()
    entry = {"kind": "item", "ticker": ticker, "phrase": phrase, "taught_by": "", "question": "",
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "flagged", "picked": "pointed", "pointed": True}
    if node["kind"] == "sentence":
        ref = st.by_id.get(node["ref_id"])
        if ref is None:
            return None
        entry.update(kind="mention", item=f"said:{ref['id']}:{node['slot']}", ref_id=ref["id"], slot=node["slot"],
                     label=ref["masked"][:160], page=ref["page"], doc=ref["doc"],
                     period=link.stated_period(g, ref, node["slot"]))
    else:
        ref, j = st.by_id.get(node["row_id"]), node["cell"]
        if ref is None:
            return None
        column = None
        if not any(link._cell_pid(ref, i) for i in range(len(ref["cells"]))):
            column = ref["cells"][j].get("col", "")   # columns that are not periods: the one the reader quoted
        key = f"item:{link._slug(ref['label'])}@{link._slug(ref['section'] or ref['title'])}" \
            + (f"#{link._slug(column, 32)}" if column else "")
        member = None
        if column is None:                            # the figure may sit in one named column of its period
            group = next((js for js in link.column_groups(ref).values() if j in js), [j])
            if len(group) > 1 and j != link.default_cell(ref, group):
                member = group.index(j)
                key += f"#{link._slug(link.members(ref)[member], 32)}"
        entry.update(item=key, ref_id=ref["id"], row_id=ref["id"], column=column, member=member, label=ref["label"],
                     title=ref["title"], section=ref["section"], page=ref["page"], doc=ref["doc"])
    _, entry["evidence"], objections = link.gate({"kind": node["kind"], "ref": ref})
    entry["objections"] = [POINTED] + objections
    return entry


def lines_of(plan: dict) -> list:
    """(step number, company, name, what the reader is asked for, period spec) for every line a plan
    reads as one printed figure of one company: an item named in words or a concept. A metric, a
    breakdown, a ranking over a set or a line picked by an earlier step is not such a line."""
    out = []
    for i, s in enumerate((plan or {}).get("steps") or []):
        if not isinstance(s, dict) or s.get("op") not in ("get", "change") or s.get("as"):
            continue
        if any(k in s for k in ("metric", "by", "of", "over", "where")):
            continue
        company = s.get("company")
        if not isinstance(company, str) or company.startswith("$"):
            continue
        name = s.get("item") if isinstance(s.get("item"), str) else s.get("concept")
        if not isinstance(name, str):
            continue
        words = name if isinstance(s.get("item"), str) else concept_words(name)
        for spec in ([s.get("period")] if s["op"] == "get" else [s.get("from"), s.get("to")]):
            if isinstance(spec, dict):
                out.append((i, company, name, words, spec))
    return out


def by_pointing(plan: dict, g, ex, model: Optional[str] = None, offline: bool = False,
                missing: Optional[list] = None) -> Optional[tuple]:
    """Run a plan on the graph with lines the text reader pointed at. Two uses:

      missing given   an abstention: each line the executor reported missing is pointed at and linked,
                      and the plan runs again (three rounds at most, as ask._execute does for the linker)
      missing None    a test of the lines the linker chose: every line the plan reads is pointed at
                      afresh, and the plan runs on those lines instead

    A concept line is pointed at by its name in words and read as an item for this run. Returns
    (the executor's result, the nodes pointed at) or None: a line the reader could not place, a place
    with no node, more than MAX_LINES lines, or a plan that still does not run. The links live for this
    call only: the phrase map, the readings and the facts are restored before returning, so no later
    question sees them and a run does not depend on the order of its questions."""
    from . import link
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return None
    alias, before, nodes = link.aliases(g), {}, []
    had: dict = {}                                    # what the graph held before the first pointed link
    plan2 = copy.deepcopy(plan)

    def bind(ticker: str, phrase: str, spec: dict) -> bool:
        node = point(g, ex, ticker, phrase, spec, model, offline)
        entry = link_entry(g, node, ticker, phrase) if node is not None else None
        if entry is None:
            return False
        key = link._phrase_key(ticker, phrase)
        before.setdefault(key, list(alias[key]) if key in alias else None)
        if not had:
            had.update(readings=set(g.readings), facts={k: len(f.readings) for k, f in g.facts.items()})
        link.apply_link(g, entry)
        nodes.append({"phrase": phrase, **{k: node[k] for k in ("id", "kind", "doc", "page", "label", "printed", "cited")
                                           if k in node}})
        return True

    def as_item(ticker: str, concept: str) -> str:
        """Read this company's concept line as an item for this run; returns the item's words."""
        words = concept_words(concept)
        for s in plan2["steps"]:
            if isinstance(s, dict) and s.get("company") == ticker and s.get("concept") == concept \
                    and not any(k in s for k in ("metric", "by", "of", "over", "where")):
                del s["concept"]
                s["item"] = words
        return words

    try:
        if missing is None:
            lines = lines_of(plan)
            if not lines or len(lines) > MAX_LINES:
                return None
            for i, ticker, name, words, spec in lines:
                if not isinstance(plan["steps"][i].get("item"), str):
                    words = as_item(ticker, name)
                if not bind(ticker, words, spec):
                    return None
            res = ex.run(plan2)
        else:
            res, tried = None, set()
            for _ in range(3):
                gaps = []
                for gap in missing or []:
                    if not (isinstance(gap, (list, tuple)) and len(gap) == 3):
                        continue
                    ticker, what, spec = gap
                    mark = (ticker, str(what), json.dumps(spec, sort_keys=True, default=str))
                    if mark not in tried and isinstance(what, str) and isinstance(spec, dict) and not what.startswith("members:"):
                        tried.add(mark)
                        gaps.append((ticker, what, spec))
                if not gaps or len(tried) > MAX_LINES:
                    break
                for ticker, what, spec in gaps:
                    words = what[len("item:"):] if what.startswith("item:") else as_item(ticker, what)
                    if not bind(ticker, words, spec):
                        return None
                res = ex.run(plan2)
                if res.ok:
                    break
                missing = res.missing
        if res is None or not res.ok:
            return None
        if res.tier in ("VERIFIED", "WITNESSED"):
            res.tier = "FLAGGED"
        res.tier_reasons = [POINTED] + [r for r in res.tier_reasons if r != POINTED]
        return res, nodes
    finally:
        # Withdraw everything this call added, so that no later question depends on it: the phrase map, the
        # readings (a pointed link's readings are capped "unverified"; left behind they would lower the tier
        # of a later answer whose pickers link the same row), and the facts
        for key, keys in before.items():
            if keys is None:
                alias.pop(key, None)
            else:
                alias[key] = keys
        if had:
            for rid in [rid for rid in g.readings if rid not in had["readings"]]:
                del g.readings[rid]
            for key in list(g.facts):
                if key not in had["facts"]:
                    del g.facts[key]
                elif len(g.facts[key].readings) > had["facts"][key]:
                    del g.facts[key].readings[had["facts"][key]:]


def as_answer(reply: dict, g=None) -> Optional[dict]:
    """A question no plan could run, whose answer the reader says is ONE printed figure: the graph node
    at the place it names, served FLAGGED, in the shape ask.answer returns. The figure, the unit, the
    page and the label are the node's. None when no node is there, or when the reader's answer is its
    own arithmetic, a ranking or words."""
    node = node_of_answer(reply or {}, g)
    if node is None:
        return None
    value, unit = node["value"], node["unit"]
    if unit == "percent":                             # printed 15.7% enters as 0.157, like every ratio
        value, unit = value / 100.0, "ratio"
    leaf = Leaf(id=str(node["id"]), ticker=node["ticker"], what="pointed:" + node["label"][:60], period=node["period"],
                doc=node["doc"], page=node["page"], label=node["label"], printed=node["printed"],
                status="verified" if node["verified"] else "unverified", value=node["value"])
    res = Result("number", float(value), unit, [leaf],
                 tree={"op": "read", "what": f"{node['ticker']} '{node['label'][:60]}'", "value": float(value),
                       "page": node["page"], "doc": node["doc"]})
    res.tier = "FLAGGED"
    res.tier_reasons = [f"no plan of the planners ran; a reader of the text named one printed figure ({node['cited']}) "
                        f"and the answer is the graph's node there: {node['kind']} '{node['label'][:70]}' on page "
                        f"{node['page']} of {node['doc']}. One model chose it"]
    return {"answer": render(res), "result": res, "learned": [], "seconds": 0.0, "plan_errors": [], "repaired": False,
            "plan": {"steps": [], "answer": None, "lookup": {"node": node["id"], "named_by": "the text reader"}}}
