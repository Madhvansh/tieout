"""The living layer: semantics learned from questions, admitted by structure.

The base graph knows what is printed and which numbers check out. It does not
know that Citigroup prints net income as "Citigroup's net income", or that
UnitedHealth calls operating income "Earnings from operations". When a plan
needs a (company, concept) the graph has no binding for, this module

  1. describes the candidate rows of that company's statement to an LLM:
     label, section and structural role. Never a value;
  2. takes the LLM's pick as a PROPOSAL;
  3. runs the proposal through a gate made of structure, not of another LLM:
       - the unit must fit the concept
       - the row's numbers must sit inside a passing check
       - the concept's own structural rule must hold (a total must be the
         total of a recovered formula; net income must reappear on the
         cash-flow statement; ...)
       - the row's role must agree with the role this concept plays in the
         companies where it is already active
  4. writes an admitted binding to semantics/learned.jsonl with the question
     that taught it, and applies it to the company's other filings and to any
     other company that prints the same label;
  5. sends a proposal the gate could not confirm to
     semantics/review_queue.jsonl for a person to approve or reject.

The store only grows; every entry records where it came from.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import defaultdict
from typing import Optional

from .concepts import Binding, concept_defs, norm_label
from .graph import Fact, Graph
from .llm import ask_json

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEARNED = ROOT / "semantics" / "learned.jsonl"
REVIEW = ROOT / "semantics" / "review_queue.jsonl"

SYSTEM = """You map one financial concept to one row of a company's financial statement.
You are shown the printed row labels of the statement with their section headings and structural
roles. You are never shown values. Pick the single row that reports the concept for the whole
company (consolidated, attributable to the parent company where the statement separates it),
or answer null if no row reports it. Do not pick a component of the concept or a per-share row.
Answer as JSON: {"row": <row number or null>, "reason": "<one short sentence>"}"""


# ------------------------------------------------------------ structure

def row_table(g: Graph, doc: str, kind: str) -> dict:
    """{row index: {"label", "section", "unit", "readings", "role", "verified"}} for one statement."""
    rows: dict = {}
    for r in g.readings.values():
        if r.doc == doc and r.kind == kind:
            d = rows.setdefault(r.row, {"label": r.label, "section": r.section, "unit": r.unit,
                                        "readings": [], "role": "loose", "verified": False})
            d["readings"].append(r)
            d["verified"] = d["verified"] or r.status == "verified"
    for row, d in rows.items():
        roles = set()
        for r in d["readings"]:
            for cid in r.checks:
                ch = g.checks[cid]
                if ch.type != "SUM" or ch.status == "fail":
                    continue
                for rid, role in ch.members:
                    if rid == r.id:
                        roles.add("total" if role == "total" else "operand")
        d["role"] = "total" if "total" in roles else ("operand" if roles else "loose")
        d["n_operands"] = 0
    return rows


def role_prior(g: Graph, concept: str) -> Optional[str]:
    """The role this concept plays where it is already active: 'total', 'operand' or None if mixed."""
    counts = defaultdict(int)
    cache: dict = {}
    for b in g.bindings:
        if b.concept != concept or b.state != "active":
            continue
        key = (b.doc, b.kind)
        if key not in cache:
            cache[key] = row_table(g, *key)
        row = cache[key].get(b.row)
        if row:
            counts["total" if row["role"] == "total" else "other"] += 1
    n = sum(counts.values())
    if n < 5:
        return None
    if counts["total"] / n >= 0.8:
        return "total"
    if counts["other"] / n >= 0.8:
        return "other"
    return None


def gate(g: Graph, doc: str, concept: str, row: int, rows: dict) -> tuple:
    """(admitted, evidence, objections) for binding `row` of `doc` to `concept`."""
    c = concept_defs()[concept]
    d = rows.get(row)
    if d is None:
        return False, [], ["no such row"]
    evidence, objections = [], []
    if not d["label"].strip():
        objections.append("row has no printed label")
    want = "USD/share" if c.unit == "USD/share" else "USD"
    if d["unit"] != want:
        objections.append(f"unit is {d['unit']}, concept needs {want}")
    if d["verified"]:
        evidence.append("value sits inside a passing check")
    else:
        objections.append("value is not covered by any check")
    taken = [b for b in g.bindings if b.doc == doc and b.kind == c.statement and b.row == row
             and b.state == "active" and b.concept != concept]
    if taken:
        objections.append(f"row is already bound to {taken[0].concept}")
    s = c.structure
    if s.get("total_of"):
        if d["role"] == "total":
            evidence.append("is the total of a recovered formula")
        else:
            objections.append("concept must be a total of a recovered formula; this row is not")
    if s.get("operand_of"):
        target = next((b.row for b in g.bindings if b.doc == doc and b.concept == s["operand_of"] and b.state == "active"), None)
        ops, level = set(), {target} if target is not None else set()
        for _ in range(2):                              # directly, or one level down
            nxt = set()
            for top in level:
                for r in rows.get(top, {"readings": []})["readings"]:
                    for cid in r.checks:
                        ch = g.checks[cid]
                        if ch.type == "SUM" and ch.status != "fail" and [r.id, "total"] in ch.members:
                            nxt |= {g.readings[rid].row for rid, role in ch.members if role != "total"}
            ops |= nxt
            level = nxt
        if row in ops:
            evidence.append(f"is an operand of the formula for {s['operand_of']}")
        else:
            objections.append(f"is not an operand of the formula for {s['operand_of']}")
    if s.get("ties_with_cf_first_row"):
        if _reappears(g, doc, d, "CF"):
            evidence.append("same values reappear on the cash-flow statement")
        elif (via := _derived_from_cf_row(g, doc, d, rows)) is not None:
            evidence.append(f"recovered formula derives it from '{via[:40]}', which reappears on the cash-flow statement")
        else:
            objections.append("values neither reappear on the cash-flow statement nor derive from a row that does")
    prior = role_prior(g, concept)
    if prior == "total" and d["role"] != "total":
        objections.append("in other companies this concept is the total of a formula; here it is not")
    elif prior == "other" and d["role"] == "total" and not s.get("total_of"):
        evidence.append("role differs from other companies (total here), accepted because the value is verified")
    elif prior:
        evidence.append(f"role agrees with other companies ({'total' if prior == 'total' else 'line item'})")
    return (not objections), evidence, objections


def _derived_from_cf_row(g: Graph, doc: str, d: dict, rows: dict):
    """Label of an operand of this row's formula whose values reappear on the cash-flow statement."""
    ops = set()
    for r in d["readings"]:
        for cid in r.checks:
            ch = g.checks[cid]
            if ch.type == "SUM" and ch.status != "fail" and [r.id, "total"] in ch.members:
                ops |= {g.readings[rid].row for rid, role in ch.members if role != "total"}
    for row in ops:
        if row in rows and _reappears(g, doc, rows[row], "CF"):
            return rows[row]["label"]
    return None


def _reappears(g: Graph, doc: str, d: dict, other_kind: str) -> bool:
    vals = {(r.period_end, r.value) for r in d["readings"]}
    hits = sum(1 for r in g.readings.values()
               if r.doc == doc and r.kind == other_kind and (r.period_end, r.value) in vals)
    return hits >= min(2, len(vals))


# ----------------------------------------------------------------- learn

def latest_doc(g: Graph, ticker: str) -> Optional[str]:
    return next((d for d, f in g.filings.items() if f["ticker"] == ticker and f["role"] == "latest"
                 and f.get("readable", True)), None)


def describe(rows: dict, bound: dict) -> str:
    lines = []
    for row in sorted(rows):
        d = rows[row]
        if d["unit"] not in ("USD", "USD/share") or not d["label"].strip():
            continue                                  # a row whose label was lost cannot be proposed
        tag = f" [already: {bound[row]}]" if row in bound else ""
        sec = f"  (under: {d['section'][:50]})" if d["section"] else ""
        lines.append(f"{row}: {d['label'][:90]}{sec}  <{d['role']}>{tag}")
    return "\n".join(lines)


def learn(g: Graph, ticker: str, concept: str, question_id: str = "", question: str = "",
          model: Optional[str] = None, use_gate: bool = True, offline: bool = False) -> dict:
    """Try to learn a binding for (ticker, concept). Returns the store entry."""
    c = concept_defs()[concept]
    doc = latest_doc(g, ticker)
    entry = {"ticker": ticker, "concept": concept, "kind": c.statement, "doc": doc, "taught_by": question_id,
             "question": question, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "rejected",
             "label": None, "row": None, "evidence": [], "objections": [], "proposer": model or ""}
    if doc is None:
        entry["objections"] = ["no readable filing"]
        return entry
    rows = row_table(g, doc, c.statement)
    if not rows:
        entry["objections"] = [f"no {c.statement} statement was parsed for this filing"]
        return entry
    bound = {b.row: b.concept for b in g.bindings if b.doc == doc and b.kind == c.statement and b.state == "active"}
    user = (f"Company: {g.companies[ticker]['name']} ({ticker})\nStatement: {c.statement}\n"
            f"Concept: {concept}: {SAYS.get(concept, '')}\n\nRows (number: label (section) <role>):\n"
            + describe(rows, bound))
    try:
        ans = ask_json(SYSTEM, user, schema=None, model=model, tag="bind", offline=offline)
    except Exception as e:  # noqa: BLE001 - a failed proposal is recorded, not fatal
        entry["objections"] = [f"proposer failed: {type(e).__name__}"]
        return entry
    row = ans.get("row")
    if isinstance(row, str) and row.strip().isdigit():
        row = int(row)                                   # small models answer "24" for 24
    entry["reason"] = str(ans.get("reason", ""))[:200]
    if row is None or not isinstance(row, int) or row not in rows:
        entry["objections"] = ["proposer found no row"]
        return entry
    entry["row"], entry["label"] = row, rows[row]["label"]
    ok, evidence, objections = gate(g, doc, concept, row, rows)
    entry["evidence"], entry["objections"] = evidence, objections
    if ok or not use_gate:
        entry["state"] = "active"
        apply_entry(g, entry)
    else:
        entry["state"] = "proposed"
    return entry


def apply_entry(g: Graph, entry: dict) -> int:
    """Bind the learned label in every filing of the company that prints it. Returns facts added."""
    want, kind, concept, ticker = norm_label(entry["label"]), entry["kind"], entry["concept"], entry["ticker"]
    added = 0
    for doc, f in g.filings.items():
        if f["ticker"] != ticker:
            continue
        rows = row_table(g, doc, kind)
        # match by the printed label, not by row number: row numbers move when the parser changes
        match = [r for r, d in rows.items() if norm_label(d["label"]) == want]
        if not match:
            continue
        row = entry["row"] if (doc == entry["doc"] and entry["row"] in match) else match[0]
        if any(b.doc == doc and b.concept == concept and b.state == "active" for b in g.bindings):
            continue
        g.bindings.append(Binding(doc=doc, kind=kind, row=row, label=rows[row]["label"], concept=concept,
                                  origin=f"learned:{entry.get('taught_by', '')}", state="active",
                                  evidence=list(entry.get("evidence", []))))
        for r in rows[row]["readings"]:
            key = f"{ticker}|{concept}|{r.period}"
            fact = g.facts.setdefault(key, Fact(ticker, concept, r.period, [], "active"))
            if r.id not in fact.readings:
                fact.readings.append(r.id)
                added += 1
    return added


def save_entry(entry: dict) -> None:
    path = LEARNED if entry["state"] == "active" else REVIEW
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_learned(g: Graph, path: pathlib.Path = LEARNED) -> int:
    """Re-apply everything learned so far to a freshly built graph."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("state") == "active" and e.get("label"):
                n += apply_entry(g, e)
    return n


SAYS = {
    "Revenue": "total revenue or net sales of the whole company for the period (for a bank: total net revenue)",
    "CostOfRevenue": "total cost of sales / cost of revenue / cost of goods sold",
    "GrossProfit": "revenue minus cost of revenue, if printed as its own line",
    "ResearchAndDevelopment": "research and development expense",
    "SellingGeneralAdmin": "selling, general and administrative expense (one combined line)",
    "OperatingIncome": "operating income / income from operations / earnings from operations",
    "PretaxIncome": "income before income taxes",
    "IncomeTax": "provision for income taxes (income tax expense)",
    "NetIncome": "net income for the period (attributable to the company if the statement separates noncontrolling interests)",
    "EPSDiluted": "diluted earnings per share",
    "SharesDiluted": "diluted weighted-average number of shares used for diluted earnings per share",
    "NetInterestIncome": "net interest income (banks)",
    "ProvisionForCreditLosses": "provision for credit losses (banks)",
    "NoninterestExpense": "total noninterest expense (banks)",
    "Cash": "cash and cash equivalents on the balance sheet",
    "TotalCurrentAssets": "total current assets",
    "TotalAssets": "total assets",
    "TotalCurrentLiabilities": "total current liabilities",
    "TotalLiabilities": "total liabilities (only if printed as a line)",
    "TotalEquity": "total shareholders' equity (total equity)",
    "TotalLiabilitiesAndEquity": "total liabilities and equity (equals total assets)",
    "Inventory": "inventories",
    "Goodwill": "goodwill",
    "LongTermDebt": "long-term debt, non-current portion",
    "Deposits": "total deposits (banks)",
    "OperatingCashFlow": "net cash provided by operating activities",
    "InvestingCashFlow": "net cash used in or provided by investing activities",
    "FinancingCashFlow": "net cash used in or provided by financing activities",
    "CapitalExpenditures": "purchases of property, plant and equipment (capital expenditures)",
    "DepreciationAmortization": "depreciation and amortization on the cash-flow statement",
    "StockBasedCompensation": "stock-based (share-based) compensation expense on the cash-flow statement",
    "DividendsPaid": "cash dividends paid to shareholders",
    "ShareRepurchases": "cash paid to repurchase common stock",
}
