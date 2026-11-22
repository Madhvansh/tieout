"""Run a plan against the graph and return an answer that carries its proof.

The executor is deterministic and does its arithmetic on what the graph holds.
Every value it returns lists the readings it rests on (the leaves of the proof
tree): filing, page, printed label, printed text and whether the reading is
verified. If a needed fact is missing, or its reading sits inside a formula
that does not hold, the executor abstains and names the fact.

A fact that is printed nowhere can still be answered when the graph's
equations determine it: the fourth quarter from the year and the nine months,
total liabilities from the balance-sheet identity, gross profit from revenue
and cost of revenue. `Executor._derive` solves the one equation that has the
wanted fact as its only unknown.

"Retrieved facts" for the benchmark's precision and recall are exactly
`Result.leaves`: every fact the executor read, including every company it
scanned for a ranking.
"""
from __future__ import annotations

import ast
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from .concepts import concept_defs
from .graph import Graph
from .plan import COMPARATORS, NUMERIC_ATTRS, identity_defs, metric_defs, step_measure, validate

OUTFLOW = {"CapitalExpenditures", "DividendsPaid", "ShareRepurchases"}   # printed negative, asked as amounts
# costs whose printed sign is only presentation ("(263)" in a column of deductions): growth, changes and
# ratios use their size, as the SEC's own data and every analyst do
EXPENSES = {"CostOfRevenue", "ResearchAndDevelopment", "SellingGeneralAdmin", "SalesAndMarketing",
            "GeneralAndAdministrative", "InterestExpense", "IncomeTax", "NoninterestExpense",
            "ProvisionForCreditLosses", "DepreciationAmortization", "StockBasedCompensation"}
TIMELESS = "*"                                    # period id of a figure whose source prints no period
_PAYMENT = re.compile(r"^(purchases?|additions?|payments?|repurchases?|repayments?|acquisitions?|capital expenditures|"
                      r"(cash )?dividends? paid|cash paid|expenditures?)\b", re.I)
_CASH_FLOW = re.compile(r"\b(investing|financing)\b|cash flows?", re.I)


def _is_payment(reading) -> bool:
    """A payment line of the cash-flow statement ("Purchases of property and equipment (9,251)"). Its
    minus sign is the statement's way of printing cash going out; a question asks for its size, as
    for the concepts in OUTFLOW."""
    on_cash_flow = reading.kind == "CF" or _CASH_FLOW.search(reading.section or "")
    return bool(on_cash_flow and _PAYMENT.search(reading.label or ""))


@dataclass
class Leaf:
    id: str
    ticker: str
    what: str                 # concept, item key or attribute
    period: str
    doc: str
    page: int
    label: str
    printed: str
    status: str               # verified | unverified | text
    value: Any = None
    witnessed: bool = False   # a linked line that two models picked independently (see witness.py)

    def key(self) -> tuple:
        return (self.ticker, self.what, self.period)


@dataclass
class Result:
    kind: str                 # number | text | companies | lines | series | abstain
    value: Any = None
    unit: str = ""
    leaves: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tree: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)   # (ticker, what, period spec) the graph could not supply
    extra: dict = field(default_factory=dict)
    tier: str = ""            # VERIFIED | FLAGGED | ABSTAIN, set on the final answer (see trust())
    tier_reasons: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind != "abstain"


class PlanTypeError(Exception):
    """A step asked for something its inputs cannot give (a company where a number is needed, ...)."""


class Abstained(Exception):
    """A step used the result of an earlier step that abstained. The earlier abstention, with the
    facts it found missing, becomes the answer, so the learning loop can try to close those gaps."""

    def __init__(self, result: "Result"):
        super().__init__("an earlier step abstained")
        self.result = result


def abstain(note: str, leaves: Optional[list] = None, missing: Optional[list] = None) -> Result:
    return Result("abstain", leaves=leaves or [], notes=[note], missing=missing or [])


class Executor:
    def __init__(self, graph: Graph, use_checks: bool = True, use_derivations: bool = True,
                 require_verified: bool = False, use_solving: bool = True, use_links: bool = True):
        self.g = graph
        self.use_checks = use_checks              # ablation: serve failed readings as if they were fine
        self.require_verified = require_verified  # strict policy: abstain unless every leaf is verified
        self.use_derivations = use_derivations    # ablation: no metric formulas, no equation solving
        self.use_solving = use_solving            # ablation: no fourth quarter from the year, no identities
        self.use_links = use_links                # ablation: no line items, sentences or breakdowns
        self._in_set = 0                          # inside a rank, filter or agg: compare statement lines only
        self.concepts = concept_defs()
        self.metrics = metric_defs()
        self.identities = identity_defs()
        self.latest_fy = {f["ticker"]: int(f["fiscal_year"].lstrip("FY"))
                          for f in graph.filings.values() if f["role"] == "latest"}
        self._text: dict = {}
        latest = {f["ticker"]: d for d, f in graph.filings.items() if f["role"] == "latest"}
        for t in graph.textfacts:
            tk = graph.filings[t["doc"]]["ticker"]
            if t.get("primary") is False and (tk, t["attr"]) in self._text:
                continue                              # a second share class never replaces the primary one
            if latest.get(tk) == t["doc"] or (tk, t["attr"]) not in self._text:
                self._text[(tk, t["attr"])] = t

    # ---------------------------------------------------------------- run

    def run(self, plan: dict) -> Result:
        errs = validate(plan, self.g)
        if errs:
            return abstain("plan rejected: " + "; ".join(errs))
        env: dict = {}
        self._steps = {step["id"]: step for step in plan["steps"]}   # compare names its sides from these
        try:
            for step in plan["steps"]:
                res = getattr(self, "_op_" + step["op"])(step, env)
                ref = step.get("company")
                if isinstance(ref, str) and ref.startswith("$") and ref[1:] in env:
                    # a company or line picked by an earlier step: what that step read is part of the proof
                    prior = env[ref[1:]]
                    res.leaves, res.notes = list(prior.leaves) + res.leaves, list(prior.notes) + res.notes
                env[step["id"]] = res
        except PlanTypeError as e:
            return abstain(f"type error: {e}")
        except Abstained as e:
            r = e.result
            return Result("abstain", leaves=list(r.leaves), notes=list(r.notes), missing=list(r.missing),
                          extra=dict(r.extra))
        except Exception as e:  # noqa: BLE001 - no plan may stop a run; the reason is kept
            # a plan that passed validation but asks for something its steps cannot give
            return abstain(f"plan could not run: {type(e).__name__}: {str(e)[:120]}")
        final = env[plan["answer"]]
        final.leaves = list({lf.id: lf for lf in final.leaves}.values())   # everything read on the way, once
        final.notes = list(dict.fromkeys(final.notes))
        final.tier, final.tier_reasons = trust(final)
        return final

    # ------------------------------------------------------------ periods

    def _fy(self, ticker: str, spec: dict) -> int:
        """The fiscal year a period spec means for this company. "latest" is the year of its
        latest annual report in the corpus, not the latest year that happens to have a fact."""
        fy = spec.get("fy", "latest")
        if fy in ("latest", "prior"):
            return self.latest_fy[ticker] - (fy == "prior")
        return int(fy)

    def _pid(self, ticker: str, spec: dict, nature: str) -> str:
        base = f"FY{self._fy(ticker, spec)}"
        q4_is_year = spec.get("q") and int(spec["q"]) == 4 and (nature == "stock" or spec.get("ytd"))
        if spec.get("q") and not q4_is_year:              # Q4 balance = year end; Q4 year to date = the year
            base += f"Q{int(spec['q'])}"
            if spec.get("ytd") and nature == "flow" and int(spec["q"]) > 1:
                base += "YTD"
        return base + ("@end" if nature == "stock" else "")

    def _year_before(self, ticker: str, spec: dict) -> dict:
        return {**spec, "fy": self._fy(ticker, spec) - 1}

    # ------------------------------------------------------------- facts

    def _fact(self, ticker: str, concept: str, spec: dict, first: bool = False, derive: bool = True) -> Result:
        c = self.concepts[concept]
        pid = self._pid(ticker, spec, c.nature)
        f = self.g.fact(ticker, concept, pid)
        if f is not None:
            return self._from_fact(f, ticker, concept, pid, first, c.unit, concept in OUTFLOW)
        if derive and self.use_derivations and self.use_solving:
            solved = self._derive(ticker, concept, spec, first)
            if solved is not None:
                return solved
        reason = getattr(self.g, "absent", {}).get((ticker, concept))
        if reason and (self._in_set or not self.use_links):
            # in a comparison, a company that does not print the concept is named as such, not as a gap
            return Result("abstain", notes=[f"{ticker} does not report {concept}: {reason}"],
                          extra={"not_reported": True})
        if derive and self.use_links and c.unit == "USD" and not self._in_set:
            # not a line of the statement: a note or a sentence of the filing may still print it
            words = concept_words(concept)
            noted = self._item(ticker, words, spec, first)
            if noted.ok:
                noted.notes.append(f"{ticker} prints no {concept} line on its {c.statement} statement; this is the "
                                   f"line or sentence the linker tied to '{words}'")
                return noted
            return abstain(f"no fact for {ticker} {concept} {pid}", missing=[(ticker, concept, spec)] + noted.missing)
        return abstain(f"no fact for {ticker} {concept} {pid}", missing=[(ticker, concept, spec)])

    def _from_fact(self, f, ticker: str, what: str, pid: str, first: bool, unit: str, outflow: bool) -> Result:
        r = self.g.best_reading(f, as_first_reported=first)
        label = r.label or "(printed without a label; identified by its place in the statement's arithmetic)"
        leaf = Leaf(r.id, ticker, what, r.period, r.doc, r.page, label, r.text, r.status, r.value)
        if self.use_checks and r.status == "failed":
            return abstain(f"{ticker} {what} {pid} is quarantined: its formula does not hold on page {r.page}",
                           [leaf], [(ticker, what, pid)])
        if self.use_checks and self.require_verified and r.status != "verified":
            return abstain(f"{ticker} {what} {pid} is not covered by any passing check (strict policy)", [leaf])
        value = abs(r.value) if outflow else r.value
        if unit == "percent" or r.unit == "percent":       # printed 15.7% enters as 0.157, like every ratio
            value, unit = value / 100.0, "ratio"
        notes = [f"{ticker} {pid} ends {r.period_end}"] if r.period_end else []
        if f.binding_state != "active":
            notes.append(f"{ticker} {what}: label '{r.label[:40]}' matched by wording only, structure did not confirm it")
        if r.status not in ("verified", "text"):
            notes.append(f"{ticker} {what} {pid}: reading is not covered by any check")
        if any(self.g.readings[i].superseded for i in f.readings):
            notes.append(f"{ticker} {what} {pid} was restated; "
                         + ("first-reported" if first else "latest-reported") + " value used")
        other = self._second_reading(ticker, what, pid, value)
        if other:
            notes.append(other)
        return Result("number", value, unit, [leaf], notes,
                      {"op": "read", "what": f"{ticker} {what} {pid}", "value": value, "page": r.page, "doc": r.doc})

    def _second_reading(self, ticker: str, concept: str, pid: str, value) -> str:
        """The other defensible reading of a concept that has two forms (attributable and consolidated
        net income, shareholders' and total equity), when the company prints both and they differ."""
        other = SECOND_READING.get(concept)
        f = self.g.fact(ticker, other, pid) if other else None
        if f is None:
            return ""
        r = self.g.best_reading(f)
        if r.value is None or r.value == value:
            return ""
        return (f"second reading: {other} is {_fmt(r.value, self.concepts[other].unit)} "
                f"({r.doc} page {r.page}, '{(r.label or '')[:50]}')")

    def _derive(self, ticker: str, concept: str, spec: dict, first: bool = False) -> Optional[Result]:
        """Solve for a fact that is printed nowhere, from one equation whose other terms are printed.

        Two kinds of equation: the period lattice (a year is nine months plus a fourth
        quarter) and the accounting identities in semantics/metrics.yaml.
        """
        c = self.concepts[concept]
        if c.nature == "flow" and spec.get("q") == 4 and not spec.get("ytd"):
            year = self._fact(ticker, concept, {"fy": spec.get("fy", "latest")}, first, derive=False)
            ytd = self._fact(ticker, concept, {"fy": spec.get("fy", "latest"), "q": 3, "ytd": True}, first, derive=False)
            if year.ok and ytd.ok:
                value = year.value - ytd.value
                return Result("number", value, c.unit, year.leaves + ytd.leaves,
                              year.notes + ytd.notes + [f"{ticker} {concept} for the fourth quarter is not printed; "
                                                        "derived as the full year minus the nine months"],
                              {"op": "solve", "what": f"{ticker} {concept} Q4 = year - nine months", "value": value,
                               "children": [year.tree, ytd.tree]})
        for identity in self.identities:
            if concept not in identity:
                continue
            terms = {name: self._fact(ticker, name, spec, first, derive=False) for name in identity if name != concept}
            if not all(r.ok for r in terms.values()):
                continue
            value = -sum(identity[name] * r.value for name, r in terms.items()) / identity[concept]
            value = int(value) if float(value).is_integer() else value
            shown = " ".join(f"{'+' if k > 0 else '-'} {n}" for n, k in identity.items()) + " = 0"
            return Result("number", abs(value) if concept in OUTFLOW else value, c.unit,
                          [lf for r in terms.values() for lf in r.leaves],
                          [n for r in terms.values() for n in r.notes]
                          + [f"{ticker} {concept} is not printed; solved from the identity {shown}"],
                          {"op": "solve", "what": f"{ticker} {concept} from {shown}", "value": value,
                           "children": [r.tree for r in terms.values()]})
        return None

    def _item(self, ticker: str, phrase: str, spec: dict, first: bool = False) -> Result:
        """A line of any table, or a figure stated in a sentence, linked on demand by the living layer.

        A fourth quarter that no filing prints is solved the way `_derive` solves it for a concept:
        the line's full year minus its nine months. Each of the two is read (and, if need be,
        linked) for its own period, so the annual report and the quarterly report may print the
        line in differently laid-out tables.
        """
        printed = self._item_printed(ticker, phrase, spec, first)
        fourth_quarter = spec.get("q") and int(spec["q"]) == 4 and not spec.get("ytd")
        if printed.ok or not (fourth_quarter and self.use_derivations and self.use_solving):
            return printed
        year_spec = {k: v for k, v in spec.items() if k != "q"}
        year = self._item_printed(ticker, phrase, year_spec, first)
        ytd = self._item_printed(ticker, phrase, {**year_spec, "q": 3, "ytd": True}, first)
        if not (year.ok and ytd.ok):
            return _merge_abstain(year, ytd)
        value = year.value - ytd.value
        return Result("number", value, year.unit, year.leaves + ytd.leaves,
                      year.notes + ytd.notes + [f"{ticker} '{phrase}' for the fourth quarter is not printed; "
                                                "derived as the full year minus the nine months"],
                      {"op": "solve", "what": f"{ticker} '{phrase}' Q4 = year - nine months", "value": value,
                       "children": [year.tree, ytd.tree]})

    def _item_printed(self, ticker: str, phrase: str, spec: dict, first: bool = False) -> Result:
        """The linked line's figure for one period, as a filing prints it."""
        from .link import resolve
        if not self.use_links:
            return abstain(f"linking is switched off; '{phrase}' is not a shared concept")
        keys = resolve(self.g, ticker, phrase)
        if not keys:
            return abstain(f"no link yet for '{phrase}' at {ticker}", missing=[(ticker, "item:" + phrase, spec)])
        asked = f"FY{self._fy(ticker, spec)}"
        for pid in (self._pid(ticker, spec, "flow"), self._pid(ticker, spec, "stock"), TIMELESS):
            for key in keys:
                f = self.g.fact(ticker, key, pid)
                if f is None:
                    continue
                if pid == TIMELESS:
                    # a figure printed with no period speaks for its own filing's year, never for another
                    said_in = {self.g.filings[self.g.readings[i].doc]["fiscal_year"] for i in f.readings}
                    if asked not in said_in or spec.get("q"):
                        continue
                source = self.g.readings[f.readings[0]]
                res = self._from_fact(f, ticker, key, pid, first, source.unit, _is_payment(source))
                for leaf in res.leaves:
                    leaf.witnessed = (ticker, key) in getattr(self.g, "witnessed_links", ())
                if pid == TIMELESS and res.ok:
                    res.notes.append(f"{ticker} '{phrase}': the source prints no period column for this figure")
                return res
        return abstain(f"'{phrase}' is linked for {ticker} but has no value for {self._pid(ticker, spec, 'flow')}",
                       missing=[(ticker, "item:" + phrase, spec)])

    def _attr(self, ticker: str, attr: str) -> Result:
        if attr == "cluster":
            return Result("text", self.g.companies[ticker]["cluster"], tree={"op": "attr", "what": f"{ticker} cluster"})
        t = self._text.get((ticker, attr))
        if t is None:
            return abstain(f"no text fact {attr} for {ticker}", missing=[(ticker, attr, {})])
        leaf = Leaf(f"{t['doc']}:text:{attr}", ticker, attr, "", t["doc"], t["page"], attr, t["quote"][:160], "text", t["value"])
        numeric = attr in NUMERIC_ATTRS
        value = int(t["value"]) if numeric else t["value"]
        return Result("number" if numeric else "text", value, NUMERIC_ATTRS.get(attr, ""), [leaf], [],
                      {"op": "attr", "what": f"{ticker} {attr}", "value": value, "page": t["page"], "doc": t["doc"]})

    # ----------------------------------------------------------- measures

    def measure(self, ticker: str, m: dict, spec: dict, first: bool = False) -> Result:
        """One measure for one company and period. See plan.py for the forms a measure takes."""
        if "growth" in m or "delta" in m:
            inner = m.get("growth") or m.get("delta")
            return self._between(ticker, inner, self._year_before(ticker, spec), spec, "pct" if "growth" in m else "abs",
                                 first)
        if "ratio" in m:
            top, bottom = (self.measure(ticker, part, spec, first) for part in m["ratio"])
            if not (top.ok and bottom.ok):
                return _merge_abstain(top, bottom)
            if not bottom.value:
                return abstain("division by zero", top.leaves + bottom.leaves)
            return Result("number", top.value / bottom.value, "ratio", top.leaves + bottom.leaves, top.notes + bottom.notes,
                          {"op": "ratio", "value": top.value / bottom.value, "children": [top.tree, bottom.tree]})
        if "attr" in m:
            r = self._attr(ticker, m["attr"])
            if r.ok and r.kind != "number":
                raise PlanTypeError(f"attribute {m['attr']} is not a number")
            return r
        if "item" in m:
            return self._item(ticker, m["item"], spec, first)
        if "metric" in m:
            return self._metric(ticker, m["metric"], spec, first)
        if "concept" not in m:
            raise PlanTypeError(f"{_name(m)} is not a measure of a company")
        return self._fact(ticker, m["concept"], spec, first)

    def _between(self, ticker: str, m: dict, start: dict, end: dict, mode: str, first: bool = False) -> Result:
        a, b = self.measure(ticker, m, start, first), self.measure(ticker, m, end, first)
        if not (a.ok and b.ok):
            return _merge_abstain(a, b)
        if a.leaves and {lf.id for lf in a.leaves} == {lf.id for lf in b.leaves}:
            return abstain(f"{ticker} {_name(m)}: both periods read the same printed figure, which has no period of "
                           "its own, so no change can be computed", a.leaves)
        leaves, notes = a.leaves + b.leaves, a.notes + b.notes
        av, bv = a.value, b.value
        if m.get("concept") in EXPENSES and av < 0 and bv < 0:      # both printed as deductions: use their size
            av, bv = -av, -bv
        a, b = Result(a.kind, av, a.unit), Result(b.kind, bv, b.unit)
        if mode == "abs":
            value, unit = b.value - a.value, a.unit
        elif not a.value:
            return abstain("base value is zero", leaves)
        else:
            value, unit = (b.value - a.value) / abs(a.value), "ratio"
        return Result("number", value, unit, leaves, notes,
                      {"op": "change", "what": f"{ticker} {_name(m)} {mode}", "value": value, "children": [a.tree, b.tree]})

    def _metric(self, ticker: str, name: str, spec: dict, first: bool = False) -> Result:
        m = self.metrics[name]
        if not self.use_derivations:
            return abstain(f"derivations are switched off; {name} is not a printed fact")
        leaves, notes, children, missing = [], [], [], []

        def term(node):
            if isinstance(node, ast.BinOp):
                a, b = term(node.left), term(node.right)
                if a is None or b is None:
                    return None
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b if b else None
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "abs":
                v = term(node.args[0])
                return None if v is None else abs(v)
            if isinstance(node, ast.Name):
                r = self._fact(ticker, node.id, spec, first)
                leaves.extend(r.leaves)
                notes.extend(r.notes)
                if not r.ok:
                    missing.extend(r.missing)
                    return None
                children.append(r.tree)
                return r.value
            if isinstance(node, ast.Constant):
                return node.value
            raise PlanTypeError(f"unsupported element in the formula of {name}")

        value = term(ast.parse(m["formula"], mode="eval").body)
        if value is None:
            return Result("abstain", leaves=leaves, notes=notes or [f"{ticker} {name} cannot be computed"], missing=missing)
        return Result("number", value, m["unit"], leaves, notes,
                      {"op": "metric", "what": f"{ticker} {name} = {m['formula']}", "value": value, "children": children})

    # --------------------------------------------------------------- sets

    def _company(self, ref, env: dict) -> str:
        if isinstance(ref, str) and ref.startswith("$"):
            r = env[ref[1:]]
            if r.kind == "abstain":
                raise Abstained(r)
            if r.kind != "companies" or not r.value:
                raise PlanTypeError(f"{ref} is not a company")
            return r.value[0][0]
        return ref

    def _top_line(self, ref, env: dict) -> Optional[dict]:
        """The line an earlier rank or filter over lines put first, or None if `ref` holds no lines."""
        if not (isinstance(ref, str) and ref.startswith("$")):
            return None
        r = env[ref[1:]]
        if r.kind == "abstain":
            raise Abstained(r)
        if r.kind != "lines" or not r.value:
            return None
        top = r.value[0][0]
        return next((ln for ln in r.extra.get("lines", []) if ln["label"] == top), None)

    def _members(self, over, env: dict) -> tuple:
        """(kind, members, leaves already read). Companies are tickers; lines come from a members step."""
        if over in (None, "all"):
            return "companies", sorted(self.g.companies), []
        if isinstance(over, dict):
            return "companies", sorted(t for t, c in self.g.companies.items() if c["cluster"] == over["cluster"]), []
        if isinstance(over, list):
            return "companies", list(over), []
        r = env[over[1:]]
        if r.kind == "abstain":
            raise Abstained(r)
        if r.kind == "companies":
            return "companies", [t for t, _ in r.value], list(r.leaves)
        if r.kind == "lines":
            return "lines", r.extra["lines"], []
        raise PlanTypeError(f"{over} is not a set")

    def _each(self, step: dict, env: dict, m: dict, spec: dict) -> tuple:
        """Evaluate a measure for every member of the step's set: (kind, rows, leaves, skipped, missing)."""
        kind, members, leaves = self._members(step.get("over"), env)
        rows, skipped, absent, missing = [], [], [], []
        for member in members:
            self._in_set += 1                         # a comparison across a set uses statement lines only
            try:
                r = self.measure(member, m, spec) if kind == "companies" else self._line_measure(member, m, spec)
            finally:
                self._in_set -= 1
            leaves.extend(r.leaves)
            name = member if kind == "companies" else member["label"]
            if r.ok:
                rows.append((name, r.value, r.unit))
            elif r.extra.get("not_reported"):
                absent.append(name)
            else:
                skipped.append(name)
                missing.extend(r.missing)
        if "item" in m:
            rows = _one_scale(rows)
        return kind, rows, leaves, skipped, absent, missing

    def _line_measure(self, line: dict, m: dict, spec: dict) -> Result:
        """A line of a members step: its printed value, or its growth, in the table's own columns."""
        ticker = line["ticker"]
        if "growth" in m or "delta" in m:
            start, end = self._line_value(line, self._year_before(ticker, spec)), self._line_value(line, spec)
            if not (start.ok and end.ok):
                return _merge_abstain(start, end)
            if "delta" in m:
                return Result("number", end.value - start.value, end.unit, start.leaves + end.leaves)
            if not start.value:
                return abstain("base value is zero", start.leaves + end.leaves)
            return Result("number", (end.value - start.value) / abs(start.value), "ratio", start.leaves + end.leaves)
        if "ratio" in m:                              # a segment's margin: its income over its own revenue
            top, bottom = (self._line_measure(line, part, spec) for part in m["ratio"])
            if not (top.ok and bottom.ok):
                return _merge_abstain(top, bottom)
            if not bottom.value:
                return abstain("division by zero", top.leaves + bottom.leaves)
            return Result("number", top.value / bottom.value, "ratio", top.leaves + bottom.leaves, top.notes + bottom.notes)
        if "item" in m:                               # another figure of the same line
            if line.get("grouped_by") == "columns":   # a column of a table: the same column of another row
                from .link import same_table_figure
                beside = same_table_figure(line, m["item"])
                if beside is not None:
                    return self._line_value(beside, spec)
            other = self._item(ticker, f"{line['label']} {m['item']}", spec)
            own, mine = {cell["id"] for cell in line["cells"].values()}, self._line_value(line, spec)
            if other.ok and (any(lf.id in own for lf in other.leaves) or (mine.ok and other.value == mine.value)):
                # the link came back to the line itself (or a reprint of it): this line has no such figure
                return abstain(f"'{line['label']}' prints no separate {m['item']}", other.leaves)
            return other
        if "value" not in m:
            raise PlanTypeError(f"{_name(m)} is a company-wide measure and cannot be read for one line of a "
                                "breakdown; over lines use {\"value\": true}, or a ratio of it with an item "
                                "that names the line's other figure")
        return self._line_value(line, spec)

    def _line_value(self, line: dict, spec: dict) -> Result:
        ticker = line["ticker"]
        for pid in (self._pid(ticker, spec, "flow"), self._pid(ticker, spec, "stock")):
            cell = line["cells"].get(pid)
            if cell:
                leaf = Leaf(cell["id"], ticker, "line:" + line["label"], pid, line["doc"], line["page"], line["label"],
                            cell["text"], "verified" if cell["ok"] else "unverified", cell["value"],
                            witnessed=bool(line.get("witnessed")))
                notes = ([f"the parts of this total were grouped by their heading on page {line['page']}, "
                          "not by a recovered formula"] if line.get("grouped_by") == "heading" else [])
                if line["unit"] == "percent":                # printed percentages enter as fractions
                    return Result("number", cell["value"] / 100.0, "ratio", [leaf], notes)
                return Result("number", cell["value"], line["unit"], [leaf], notes)
        return abstain(f"'{line['label']}' has no value for {self._pid(ticker, spec, 'flow')}")

    # ---------------------------------------------------------------- ops

    def _op_get(self, s: dict, env: dict) -> Result:
        line = self._top_line(s["company"], env)
        if line is not None:                          # "$sN" holds lines: read the top line itself,
            m = step_measure(s) or {}                 # or another figure printed under that line's name
            r = self._line_measure(line, m if ("item" in m or "ratio" in m) else _on_value(m), s["period"])
            prior = env[s["company"][1:]]
            r.leaves, r.notes = list(prior.leaves) + r.leaves, list(prior.notes) + r.notes
            if r.ok:
                r.tree = {"op": "read", "what": f"{line['ticker']} '{line['label']}'", "value": r.value}
                r.notes.append(f"the line is '{line['label']}'")
            return r
        return self.measure(self._company(s["company"], env), step_measure(s), s["period"],
                            first=s.get("as") == "first_reported")

    _op_metric = _op_get

    def _op_change(self, s: dict, env: dict) -> Result:
        return self._between(self._company(s["company"], env), step_measure(s), s["from"], s["to"], s.get("mode", "pct"))

    def _op_calc(self, s: dict, env: dict) -> Result:
        leaves, notes, children = [], [], []

        def term(node):
            if isinstance(node, ast.BinOp):
                (a, ua), (b, ub) = term(node.left), term(node.right)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    if ua and ub and ua != ub:
                        raise PlanTypeError(f"cannot add or subtract {ua} and {ub}")
                    return (a + b if isinstance(node.op, ast.Add) else a - b), ua or ub
                if isinstance(node.op, ast.Mult):
                    return a * b, ua or ub
                if isinstance(node.op, ast.Div):
                    if not b:
                        raise PlanTypeError("division by zero")
                    return a / b, "ratio" if ua == ub or not ua else (ua if not ub else f"{ua} per {ub.rstrip('s')}")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                a, ua = term(node.operand)
                return -a, ua
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "abs":
                a, ua = term(node.args[0])
                return abs(a), ua
            if isinstance(node, ast.Name):
                r = env.get(node.id)
                if r is None or r.kind != "number":
                    raise KeyError(node.id)
                leaves.extend(r.leaves)
                notes.extend(r.notes)
                children.append(r.tree)
                return r.value, r.unit
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value, ""
            raise PlanTypeError(f"unsupported expression: {s['expr']}")

        tree = ast.parse(s["expr"], mode="eval")
        try:
            value, unit = term(tree.body)
        except KeyError as e:
            bad = env.get(str(e).strip("'"))
            if bad is None:
                return abstain(f"step {e} has no number", leaves)
            return Result("abstain", leaves=leaves + bad.leaves, notes=notes + bad.notes, missing=bad.missing)
        used = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in env]
        reads = [frozenset(lf.id for lf in env[i].leaves) for i in dict.fromkeys(used)]
        if len(reads) > 1 and reads[0] and all(r == reads[0] for r in reads):
            # two steps that read the very same printed cells: their difference or ratio says nothing
            line = next((f" ('{lf.label[:60]}', {lf.doc} page {lf.page})" for lf in leaves), "")
            return abstain(f"the steps of '{s['expr']}' all read the same printed figure{line}: one line holds what "
                           "the plan asked for under different names or periods. Read it once; it cannot be added to "
                           "or compared with itself", leaves)
        return Result("number", value, unit, leaves, notes, {"op": "calc", "what": s["expr"], "value": value, "children": children})

    def _op_rank(self, s: dict, env: dict) -> Result:
        m = step_measure(s) or {"value": True}
        kind, rows, leaves, skipped, absent, missing = self._each(s, env, m, s["period"])
        if not rows:
            return Result("abstain", leaves=leaves, notes=["nothing in the set has this fact"], missing=missing)
        if skipped and isinstance(s.get("over"), list):
            # the question named these companies; ranking the ones we happen to have would look complete and not be
            return Result("abstain", leaves=leaves, missing=missing,
                          notes=[f"cannot rank the named companies: no fact for {', '.join(skipped)}"])
        rows.sort(key=lambda row: row[1], reverse=s.get("order", "desc") == "desc")
        top = rows[:int(s.get("k", 1))]
        notes = _coverage_notes(len(rows), skipped, absent)
        extra = {"ranked": len(rows), "lines": [ln for ln in self._members(s.get("over"), env)[1]
                                                if kind == "lines" and ln["label"] in {n for n, _, _ in top}]}
        if len(top) >= 2:                             # "which of the two, and by how much": the gap is part of the answer
            extra["gap"] = abs(top[0][1] - top[1][1])
        if len(top) >= 3:                             # "... how much bigger than the smallest": first against last
            extra["spread"] = abs(top[0][1] - top[-1][1])
        return Result("companies" if kind == "companies" else "lines", [(name, v) for name, v, _ in top], top[0][2],
                      leaves, notes, {"op": "rank", "what": f"{_name(m)} over {len(rows)}", "value": top}, missing, extra)

    def _op_compare(self, s: dict, env: dict) -> Result:
        """Order figures that earlier steps worked out, one per side of the comparison.

        Each side is named by the company its figures were read from. When two sides are the same
        company (two of its segments or products), the plan's own names for them are used, else the
        words of the line each side was worked out from (a fourth quarter computed as the year minus
        nine months of TREMFYA is "TREMFYA"), and the step id only when those words cannot tell the
        sides apart. (22 Nov: sides named "s6" and "s3" could not be scored or read.)"""
        ids = [i.lstrip("$") for i in s["of"]]
        names = s.get("names") if isinstance(s.get("names"), list) and len(s["names"]) == len(ids) else None
        if names is None:
            words = [self._step_words(i) for i in ids]
            names = words if all(words) and len({_norm(w) for w in words}) == len(words) else [None] * len(ids)
        sides, leaves, notes = [], [], []
        for i, name in zip(ids, names):
            r = env[i]
            if r.kind == "abstain":
                raise Abstained(r)
            if r.kind != "number":
                raise PlanTypeError(f"{i} is not a number, so it cannot be compared")
            sides.append({"id": i, "value": r.value, "unit": r.unit, "tickers": sorted({lf.ticker for lf in r.leaves}),
                          "name": str(name) if name else i})
            leaves += r.leaves
            notes += r.notes
        units = {side["unit"] for side in sides if side["unit"]}
        if len(units) > 1:
            raise PlanTypeError("cannot compare " + " with ".join(sorted(units)))
        by_company = (all(len(side["tickers"]) == 1 for side in sides)
                      and len({side["tickers"][0] for side in sides}) == len(sides))
        sides.sort(key=lambda side: side["value"], reverse=s.get("order", "desc") == "desc")
        top = sides[:int(s.get("k", len(sides)))]
        value = [(side["tickers"][0] if by_company else side["name"], side["value"]) for side in top]
        extra = {"ranked": len(sides)}
        if len(top) >= 2:
            extra["gap"] = abs(top[0]["value"] - top[1]["value"])
        if len(top) >= 3:
            extra["spread"] = abs(top[0]["value"] - top[-1]["value"])
        return Result("companies" if by_company else "lines", value, top[0]["unit"], leaves, notes,
                      {"op": "compare", "what": " against ".join(side["name"] for side in sides), "value": value,
                       "children": [env[i].tree for i in ids]}, extra=extra)

    def _step_words(self, sid: str, seen: Optional[set] = None) -> str:
        """The plan's words for what a step reads: its item, concept or metric; for a step that works
        on earlier steps (a calc, a change), the words of the first of them that has any."""
        seen = set() if seen is None else seen
        step = getattr(self, "_steps", {}).get(sid)
        if step is None or sid in seen:
            return ""
        seen.add(sid)
        m = step_measure(step)
        if m and not any(k in m for k in ("growth", "delta", "ratio")):
            return str(next(iter(m.values())))
        if m:
            return _name(m)
        refs = []
        if isinstance(step.get("expr"), str):
            try:
                refs = [n.id for n in ast.walk(ast.parse(step["expr"], mode="eval")) if isinstance(n, ast.Name)]
            except SyntaxError:
                refs = []
        for key in ("of", "over", "from", "company"):
            v = step.get(key)
            refs += [x.lstrip("$") for x in (v if isinstance(v, list) else [v]) if isinstance(x, str)]
        for r in refs:
            words = self._step_words(r, seen)
            if words:
                return words
        return ""

    def _op_filter(self, s: dict, env: dict) -> Result:
        w = s["where"]
        if "attr" in w and "eq" in w:
            kind, members, leaves = self._members(s.get("over"), env)
            keep = []
            for t in members:
                r = self._attr(t, w["attr"])
                leaves.extend(r.leaves)
                if r.ok and same_name(r.value, w["eq"]):
                    keep.append((t, r.value))
            return Result("companies", keep, "", leaves, [], {"op": "filter", "what": str(w), "value": keep})
        spec = w.get("period") or s.get("period") or self._set_period(s.get("over"), env) or {}
        kind, rows, leaves, skipped, absent, missing = self._each(s, env, w["by"], spec)
        test = next(k for k in COMPARATORS if k in w)
        passes = {"gt": lambda v: v > w["gt"], "lt": lambda v: v < w["lt"],
                  "ge": lambda v: v >= w["ge"], "le": lambda v: v <= w["le"]}[test]
        keep = [(name, v) for name, v, _ in rows if passes(v)]
        unit = rows[0][2] if rows else ""
        lines = [ln for ln in self._members(s.get("over"), env)[1]
                 if kind == "lines" and ln["label"] in {n for n, _ in keep}]
        return Result("companies" if kind == "companies" else "lines", keep, unit, leaves,
                      _coverage_notes(len(rows), skipped, absent), {"op": "filter", "what": str(w), "value": keep},
                      missing, {"lines": lines})

    def _op_agg(self, s: dict, env: dict) -> Result:
        m = step_measure(s)
        _kind, rows, leaves, skipped, absent, missing = self._each(s, env, m, s["period"])
        if not rows:
            return Result("abstain", leaves=leaves, notes=["nothing in the set has this fact"], missing=missing)
        if skipped and isinstance(s.get("over"), list):
            return Result("abstain", leaves=leaves, missing=missing,
                          notes=[f"cannot aggregate the named companies: no fact for {', '.join(skipped)}"])
        values = [v for _, v, _ in rows]
        fn = {"sum": sum, "avg": statistics.fmean, "median": statistics.median, "min": min, "max": max}[s["fn"]]
        return Result("number", fn(values), rows[0][2], leaves, _coverage_notes(len(rows), skipped, absent),
                      {"op": "agg", "what": f"{s['fn']} of {_name(m)} over {len(rows)}", "value": fn(values)}, missing)

    def _op_count(self, s: dict, env: dict) -> Result:
        r = env[s["of"][1:]]
        if r.kind == "abstain":
            raise Abstained(r)
        if r.kind not in ("companies", "lines"):
            raise PlanTypeError(f"{s['of']} is not a set")
        return Result("number", len(r.value), "count", list(r.leaves), list(r.notes),
                      {"op": "count", "value": len(r.value)}, extra={"names": [name for name, _ in r.value]})

    def _op_series(self, s: dict, env: dict) -> Result:
        ticker, m = self._company(s["company"], env), step_measure(s)
        latest = self.latest_fy[ticker]
        specs = s.get("periods") or [{"fy": latest - back} for back in range(int(s["last"]) - 1, -1, -1)]
        points, leaves, notes = [], [], []
        for spec in specs:
            r = self.measure(ticker, m, spec)
            leaves.extend(r.leaves)
            notes.extend(r.notes)
            if not r.ok:
                return Result("abstain", leaves=leaves, notes=r.notes, missing=r.missing)
            points.append((f"FY{self._fy(ticker, spec)}" + (f"Q{spec['q']}" if spec.get("q") else ""), r.value, r.unit))
        unit = points[0][2]
        if s.get("pick") in ("max", "min"):
            best = (max if s["pick"] == "max" else min)(points, key=lambda p: p[1])
            return Result("text", best[0], "", leaves, notes + [f"{_name(m)} in {best[0]}: {_fmt(best[1], unit)}"],
                          {"op": "series", "value": best[0]}, extra={"points": [(p, v) for p, v, _ in points]})
        return Result("series", [(p, v) for p, v, _ in points], unit, leaves, notes,
                      {"op": "series", "what": f"{ticker} {_name(m)}", "value": [(p, v) for p, v, _ in points]})

    def _op_members(self, s: dict, env: dict) -> Result:
        from .link import breakdown
        if not self.use_links:
            return abstain("linking is switched off; breakdowns come from the linker")
        ticker = self._company(s["company"], env)
        lines = breakdown(self.g, ticker, s["of"])
        if lines is None:
            return abstain(f"no breakdown linked yet for '{s['of']}' at {ticker}",
                           missing=[(ticker, "members:" + s["of"], s["period"])])
        rows, leaves, notes = [], [], []
        for line in lines:
            r = self._line_value(line, s["period"])
            leaves.extend(r.leaves)
            notes.extend(r.notes)
            if r.ok:
                rows.append((line["label"], r.value))
        if not rows:
            return abstain(f"the parts of '{s['of']}' have no values for {self._pid(ticker, s['period'], 'flow')}", leaves)
        return Result("lines", rows, lines[0]["unit"], leaves, list(dict.fromkeys(notes)),
                      {"op": "members", "what": s["of"], "value": rows}, extra={"lines": lines, "period": s["period"]})

    def _set_period(self, over, env: dict) -> Optional[dict]:
        """The period of the members step a set of lines came from (a filter over lines may omit it)."""
        if isinstance(over, str) and over.startswith("$") and over[1:] in env:
            return env[over[1:]].extra.get("period")
        return None

    def _op_attr(self, s: dict, env: dict) -> Result:
        ref = s["company"]
        r = self._attr(self._company(ref, env), s["attr"])
        if isinstance(ref, str) and ref.startswith("$"):
            prior = env[ref[1:]]
            r.leaves, r.notes = list(prior.leaves) + r.leaves, list(prior.notes) + r.notes
        return r

    def _op_restated(self, s: dict, env: dict) -> Result:
        ticker, m = self._company(s["company"], env), step_measure(s)
        first, last = self.measure(ticker, m, s["period"], first=True), self.measure(ticker, m, s["period"])
        if not (first.ok and last.ok):
            return _merge_abstain(first, last)
        leaves, notes = first.leaves + last.leaves, first.notes + last.notes
        extra = {"first": first.value, "latest": last.value,
                 "first_doc": first.leaves[0].doc, "latest_doc": last.leaves[0].doc}
        if extra["first_doc"] == extra["latest_doc"]:
            notes.append(f"{ticker} {_name(m)}: only one filing reports this period, so there is nothing to compare")
        else:
            why = self._diagnose(ticker, last.leaves[0])
            if why:
                extra["diagnosis"] = why
                leaves = leaves + why.pop("leaves", [])
        return Result("number", last.value - first.value, first.unit, leaves, notes,
                      {"op": "restated", "what": f"{ticker} {_name(m)} latest minus first reported",
                       "value": last.value - first.value, "children": [first.tree, last.tree]}, extra=extra)

    def _diagnose(self, ticker: str, leaf: Leaf) -> Optional[dict]:
        """How the figure changed, read off the two filings' statements (see restate.py)."""
        try:
            from .restate import diagnose
            d = diagnose(self.g, ticker, leaf.period, label=leaf.label)
        except Exception:  # noqa: BLE001 - the restated number stands without its explanation
            return None
        if not d or d.get("kind") in (None, "unchanged", "single_filing"):
            return None
        cited = []
        for c in d.get("counterparts", []):
            for side in ("old_reading", "new_reading"):
                r = c.get(side)
                if r:
                    cited.append(Leaf(r["id"], ticker, c["label"], leaf.period, r["doc"], r["page"], c["label"],
                                      r["printed"], "verified", r.get("value")))
        return {"kind": d["kind"], "summary": d.get("summary", ""), "explanations": d.get("explanations", []),
                "leaves": cited}


# ---------------------------------------------------------------- trust

SECOND_READING = {"NetIncome": "ConsolidatedNetIncome", "ConsolidatedNetIncome": "NetIncome",
                  "ShareholdersEquity": "TotalEquity", "TotalEquity": "ShareholdersEquity"}
_WEAK_NOTES = ("matched by wording only", "not covered by any check", "covered ", "prints no period column",
               "second reading:", "only one filing reports")


def trust(r: Result) -> tuple:
    """The answer's tier and the reasons for it.

    VERIFIED  every number the answer rests on is a statement line bound to its concept by structure
              and sits inside a formula that holds on its page (or is solved from such numbers),
              every text fact is quoted from its page, and nothing was matched by wording alone,
              skipped, or left with a second defensible reading. A line the linker chose by its
              words is never VERIFIED: the arithmetic proves the number, not which line it is.
    WITNESSED the same, except that some numbers are lines the linker chose by their words, and each
              such line was picked independently by two models that agreed (the four-eyes rule of
              witness.py). Arithmetic vouches for the numbers, two witnesses for each choice of line.
    FLAGGED   answered, with each weakness named.
    ABSTAIN   no answer.
    """
    if not r.ok:
        return "ABSTAIN", list(r.notes[:3])
    reasons, linked = [], False
    for lf in r.leaves:
        if str(lf.what).startswith(("item:", "said:", "line:")):
            linked = True
            if lf.witnessed and lf.status == "verified":
                continue
            reasons.append(f"{lf.ticker}: the line was chosen by the linker from its words"
                           + (" and one model alone" if not lf.witnessed else "")
                           + "; the arithmetic confirms the number, not that it is the line asked for "
                           f"({lf.doc} page {lf.page})" if lf.status == "verified" else
                           f"{lf.ticker}: the line was chosen by the linker from its words and its number is not "
                           f"inside a passing formula ({lf.doc} page {lf.page})")
            continue
        if lf.status == "verified":
            continue
        if lf.status == "text" and lf.what not in NUMERIC_ATTRS and not str(lf.what).startswith(("said:", "item:")):
            continue                                      # a name quoted from the cover page or audit report
        reasons.append(f"{lf.ticker} {lf.what}: {'stated in a sentence only' if lf.status == 'text' else 'not inside a passing formula'}"
                       f" ({lf.doc} page {lf.page})")
    reasons += [n for n in r.notes if any(w in n for w in _WEAK_NOTES)]
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return "FLAGGED", reasons[:6]
    return ("WITNESSED", []) if linked else ("VERIFIED", [])


# ---------------------------------------------------------------- rendering

def _merge_abstain(a: Result, b: Result) -> Result:
    return Result("abstain", leaves=a.leaves + b.leaves, notes=[n for r in (a, b) if not r.ok for n in r.notes],
                  missing=a.missing + b.missing, extra={k: v for r in (a, b) for k, v in r.extra.items()})


def _one_scale(rows: list) -> list:
    """A set is compared on one scale. One filing prints a rate as "11.4%" (read as 0.114), another
    prints a bare "14.6" in a table whose heading says percent (read as 14.6). When the same item
    comes back on both scales across a set, the bare figures are percentages too."""
    rates = [abs(v) for _, v, u in rows if u in ("ratio", "percent")]
    if len(rates) < len(rows) or not (any(v < 1 for v in rates) and any(1 <= v <= 100 for v in rates)):
        return rows
    return [(name, v / 100.0 if abs(v) >= 1 else v, u) for name, v, u in rows]


def _coverage_notes(ranked: int, skipped: list, absent: list) -> list:
    notes = []
    if absent:
        notes.append(f"{len(absent)} in the set do not report this and were left out: {', '.join(absent)}")
    if skipped:
        notes.append(f"covered {ranked} of {ranked + len(skipped)}; no fact found for: {', '.join(skipped)}")
    return notes


def concept_words(concept: str) -> str:
    """A concept's name as words a filing might print: "ResearchAndDevelopment" -> "research and development"."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", concept).lower()


def _on_value(m: Optional[dict]) -> dict:
    """The same measure taken on a line's printed value: growth of revenue becomes growth of the value."""
    if not m:
        return {"value": True}
    for wrap in ("growth", "delta"):
        if wrap in m:
            return {wrap: _on_value(m[wrap])}
    return {"value": True}


def _name(m: dict) -> str:
    if "growth" in m:
        return "growth of " + _name(m["growth"])
    if "delta" in m:
        return "change in " + _name(m["delta"])
    if "ratio" in m:
        return f"{_name(m['ratio'][0])} / {_name(m['ratio'][1])}"
    return str(next(iter(m.values())))


def _norm(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


_LEGAL = re.compile(r"\b(llp|llc|inc|corp|corporation|company|co|the|plc|ltd|n a)\b\.?")


def same_name(a, b) -> bool:
    """Two names of the same thing: "KPMG LLP" and "KPMG", "Ernst & Young LLP" and "Ernst and Young"."""
    def key(x):
        return re.sub(r"[^a-z0-9]", "", _LEGAL.sub(" ", str(x).lower().replace("&", " and ")))
    ka, kb = key(a), key(b)
    return bool(ka) and bool(kb) and (ka == kb or ka in kb or kb in ka)


def render(r: Result) -> str:
    """One line a person can read."""
    if r.kind == "abstain":
        return "ABSTAIN: " + ("; ".join(r.notes) or "a needed fact is missing")
    if r.kind == "text":
        return str(r.value)
    if r.extra.get("first") is not None:
        a, b = r.extra["first"], r.extra["latest"]
        if r.extra.get("first_doc") == r.extra.get("latest_doc"):
            return f"Only one filing reports this period: {_fmt(b, r.unit)} ({r.extra['latest_doc']})"
        if a == b:
            return f"Not restated: both filings print {_fmt(a, r.unit)}"
        why = r.extra.get("diagnosis", {}).get("summary")
        return (f"Changed by {_fmt(b - a, r.unit)}: first reported {_fmt(a, r.unit)} ({r.extra['first_doc']}), "
                f"now {_fmt(b, r.unit)} ({r.extra['latest_doc']})" + (f". {why}" if why else ""))
    if r.kind in ("companies", "lines", "series"):
        shown = ", ".join(f"{name} ({v})" if isinstance(v, str) else f"{name} ({_fmt(v, r.unit)})" for name, v in r.value)
        two_sides = r.kind == "companies" or r.tree.get("op") == "compare"
        if two_sides and len(r.value) == 2 and all(not isinstance(v, str) for _, v in r.value):
            shown += f"; difference {_fmt(abs(r.value[0][1] - r.value[1][1]), r.unit)}"
        if r.extra.get("spread") is not None and len(r.value) >= 3:
            shown += f"; first minus last {_fmt(r.extra['spread'], r.unit)}"
        return shown or "none"
    if r.unit == "count" and r.extra.get("names"):
        return f"{r.value}: {', '.join(r.extra['names'])}"
    return _fmt(r.value, r.unit)


def _fmt(v, unit: str) -> str:
    if unit == "USD":
        a = abs(v)
        if a >= 1e9:
            return f"${v / 1e9:,.3f} billion"
        if a >= 1e6:
            return f"${v / 1e6:,.1f} million"
        return f"${v:,.2f}" if isinstance(v, float) and not v.is_integer() else f"${v:,.0f}"
    if unit in ("ratio", "percent"):
        return f"{v * 100:.2f}%" if abs(v) < 10 else f"{v:.2f}x"
    if unit == "USD/share":
        return f"${v:,.2f} per share"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.4f} {unit}".strip()
    return f"{int(v):,} {unit}".strip()
