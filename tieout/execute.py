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
TIMELESS = "*"                                    # period id of a figure whose source prints no period


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

    @property
    def ok(self) -> bool:
        return self.kind != "abstain"


class PlanTypeError(Exception):
    """A step asked for something its inputs cannot give (a company where a number is needed, ...)."""


def abstain(note: str, leaves: Optional[list] = None, missing: Optional[list] = None) -> Result:
    return Result("abstain", leaves=leaves or [], notes=[note], missing=missing or [])


class Executor:
    def __init__(self, graph: Graph, use_checks: bool = True, use_derivations: bool = True,
                 require_verified: bool = False):
        self.g = graph
        self.use_checks = use_checks              # ablation: serve failed readings as if they were fine
        self.require_verified = require_verified  # strict policy: abstain unless every leaf is verified
        self.use_derivations = use_derivations    # ablation: no metric formulas, no equation solving
        self.concepts = concept_defs()
        self.metrics = metric_defs()
        self.identities = identity_defs()
        self.latest_fy = {f["ticker"]: int(f["fiscal_year"].lstrip("FY"))
                          for f in graph.filings.values() if f["role"] == "latest"}
        self._text: dict = {}
        latest = {f["ticker"]: d for d, f in graph.filings.items() if f["role"] == "latest"}
        for t in graph.textfacts:
            tk = graph.filings[t["doc"]]["ticker"]
            if latest.get(tk) == t["doc"] or (tk, t["attr"]) not in self._text:
                self._text[(tk, t["attr"])] = t

    # ---------------------------------------------------------------- run

    def run(self, plan: dict) -> Result:
        errs = validate(plan, self.g)
        if errs:
            return abstain("plan rejected: " + "; ".join(errs))
        env: dict = {}
        try:
            for step in plan["steps"]:
                env[step["id"]] = getattr(self, "_op_" + step["op"])(step, env)
        except PlanTypeError as e:
            return abstain(f"type error: {e}")
        final = env[plan["answer"]]
        final.leaves = list({lf.id: lf for lf in final.leaves}.values())   # everything read on the way, once
        final.notes = list(dict.fromkeys(final.notes))
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
        if spec.get("q"):
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
        if derive and self.use_derivations:
            solved = self._derive(ticker, concept, spec)
            if solved is not None:
                return solved
        reason = getattr(self.g, "absent", {}).get((ticker, concept))
        if reason:
            return Result("abstain", notes=[f"{ticker} does not report {concept}: {reason}"],
                          extra={"not_reported": True})
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
        notes = [f"{ticker} {pid} ends {r.period_end}"] if r.period_end else []
        if f.binding_state != "active":
            notes.append(f"{ticker} {what}: label '{r.label[:40]}' matched by wording only, structure did not confirm it")
        if r.status not in ("verified", "text"):
            notes.append(f"{ticker} {what} {pid}: reading is not covered by any check")
        if any(self.g.readings[i].superseded for i in f.readings):
            notes.append(f"{ticker} {what} {pid} was restated; "
                         + ("first-reported" if first else "latest-reported") + " value used")
        return Result("number", value, unit, [leaf], notes,
                      {"op": "read", "what": f"{ticker} {what} {pid}", "value": value, "page": r.page, "doc": r.doc})

    def _derive(self, ticker: str, concept: str, spec: dict) -> Optional[Result]:
        """Solve for a fact that is printed nowhere, from one equation whose other terms are printed.

        Two kinds of equation: the period lattice (a year is nine months plus a fourth
        quarter) and the accounting identities in semantics/metrics.yaml.
        """
        c = self.concepts[concept]
        if c.nature == "flow" and spec.get("q") == 4 and not spec.get("ytd"):
            year = self._fact(ticker, concept, {"fy": spec.get("fy", "latest")}, derive=False)
            ytd = self._fact(ticker, concept, {"fy": spec.get("fy", "latest"), "q": 3, "ytd": True}, derive=False)
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
            terms = {name: self._fact(ticker, name, spec, derive=False) for name in identity if name != concept}
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
        """A line of any table, or a figure stated in a sentence, linked on demand by the living layer."""
        from .link import resolve
        key = resolve(self.g, ticker, phrase)
        if key is None:
            return abstain(f"no link yet for '{phrase}' at {ticker}", missing=[(ticker, "item:" + phrase, spec)])
        for pid in (self._pid(ticker, spec, "flow"), self._pid(ticker, spec, "stock"), TIMELESS):
            f = self.g.fact(ticker, key, pid)
            if f is not None:
                res = self._from_fact(f, ticker, key, pid, first, self.g.readings[f.readings[0]].unit, False)
                if pid == TIMELESS and res.ok:
                    res.notes.append(f"{ticker} '{phrase}': the source prints no period column for this figure")
                return res
        return abstain(f"'{phrase}' is linked for {ticker} but has no value for {self._pid(ticker, spec, 'flow')}")

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
            return self._between(ticker, inner, self._year_before(ticker, spec), spec, "pct" if "growth" in m else "abs")
        if "ratio" in m:
            top, bottom = (self.measure(ticker, part, spec) for part in m["ratio"])
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
            return self._metric(ticker, m["metric"], spec)
        return self._fact(ticker, m["concept"], spec, first)

    def _between(self, ticker: str, m: dict, start: dict, end: dict, mode: str) -> Result:
        a, b = self.measure(ticker, m, start), self.measure(ticker, m, end)
        if not (a.ok and b.ok):
            return _merge_abstain(a, b)
        leaves, notes = a.leaves + b.leaves, a.notes + b.notes
        if mode == "abs":
            value, unit = b.value - a.value, a.unit
        elif not a.value:
            return abstain("base value is zero", leaves)
        else:
            value, unit = (b.value - a.value) / abs(a.value), "ratio"
        return Result("number", value, unit, leaves, notes,
                      {"op": "change", "what": f"{ticker} {_name(m)} {mode}", "value": value, "children": [a.tree, b.tree]})

    def _metric(self, ticker: str, name: str, spec: dict) -> Result:
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
                r = self._fact(ticker, node.id, spec)
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
            if r.kind != "companies" or not r.value:
                raise PlanTypeError(f"{ref} is not a company")
            return r.value[0][0]
        return ref

    def _members(self, over, env: dict) -> tuple:
        """(kind, members, leaves already read). Companies are tickers; lines come from a members step."""
        if over in (None, "all"):
            return "companies", sorted(self.g.companies), []
        if isinstance(over, dict):
            return "companies", sorted(t for t, c in self.g.companies.items() if c["cluster"] == over["cluster"]), []
        if isinstance(over, list):
            return "companies", list(over), []
        r = env[over[1:]]
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
            r = self.measure(member, m, spec) if kind == "companies" else self._line_measure(member, m, spec)
            leaves.extend(r.leaves)
            name = member if kind == "companies" else member["label"]
            if r.ok:
                rows.append((name, r.value, r.unit))
            elif r.extra.get("not_reported"):
                absent.append(name)
            else:
                skipped.append(name)
                missing.extend(r.missing)
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
        return self._line_value(line, spec)

    def _line_value(self, line: dict, spec: dict) -> Result:
        ticker = line["ticker"]
        for pid in (self._pid(ticker, spec, "flow"), self._pid(ticker, spec, "stock")):
            cell = line["cells"].get(pid)
            if cell:
                leaf = Leaf(cell["id"], ticker, line["label"], pid, line["doc"], line["page"], line["label"],
                            cell["text"], "verified" if cell["ok"] else "unverified", cell["value"])
                return Result("number", cell["value"], line["unit"], [leaf])
        return abstain(f"'{line['label']}' has no value for {self._pid(ticker, spec, 'flow')}")

    # ---------------------------------------------------------------- ops

    def _op_get(self, s: dict, env: dict) -> Result:
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

        try:
            value, unit = term(ast.parse(s["expr"], mode="eval").body)
        except KeyError as e:
            bad = env.get(str(e).strip("'"))
            if bad is None:
                return abstain(f"step {e} has no number", leaves)
            return Result("abstain", leaves=leaves + bad.leaves, notes=notes + bad.notes, missing=bad.missing)
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
        return Result("companies" if kind == "companies" else "lines", [(name, v) for name, v, _ in top], top[0][2],
                      leaves, notes, {"op": "rank", "what": f"{_name(m)} over {len(rows)}", "value": top}, missing,
                      {"ranked": len(rows), "lines": [ln for ln in self._members(s.get("over"), env)[1]
                                                       if kind == "lines" and ln["label"] in {n for n, _, _ in top}]})

    def _op_filter(self, s: dict, env: dict) -> Result:
        w = s["where"]
        if "attr" in w and "eq" in w:
            kind, members, leaves = self._members(s.get("over"), env)
            keep = []
            for t in members:
                r = self._attr(t, w["attr"])
                leaves.extend(r.leaves)
                if r.ok and _norm(r.value) == _norm(w["eq"]):
                    keep.append((t, r.value))
            return Result("companies", keep, "", leaves, [], {"op": "filter", "what": str(w), "value": keep})
        kind, rows, leaves, skipped, absent, missing = self._each(s, env, w["by"], w.get("period") or s.get("period") or {})
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
        ticker = self._company(s["company"], env)
        lines = breakdown(self.g, ticker, s["of"])
        if lines is None:
            return abstain(f"no breakdown linked yet for '{s['of']}' at {ticker}",
                           missing=[(ticker, "members:" + s["of"], s["period"])])
        rows, leaves = [], []
        for line in lines:
            r = self._line_value(line, s["period"])
            leaves.extend(r.leaves)
            if r.ok:
                rows.append((line["label"], r.value))
        if not rows:
            return abstain(f"the parts of '{s['of']}' have no values for {self._pid(ticker, s['period'], 'flow')}", leaves)
        return Result("lines", rows, lines[0]["unit"], leaves, [], {"op": "members", "what": s["of"], "value": rows},
                      extra={"lines": lines})

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
        except ImportError:
            return None
        d = diagnose(self.g, ticker, leaf.period, label=leaf.label)
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


# ---------------------------------------------------------------- rendering

def _merge_abstain(a: Result, b: Result) -> Result:
    return Result("abstain", leaves=a.leaves + b.leaves, notes=[n for r in (a, b) if not r.ok for n in r.notes],
                  missing=a.missing + b.missing, extra={k: v for r in (a, b) for k, v in r.extra.items()})


def _coverage_notes(ranked: int, skipped: list, absent: list) -> list:
    notes = []
    if absent:
        notes.append(f"{len(absent)} in the set do not report this and were left out: {', '.join(absent)}")
    if skipped:
        notes.append(f"covered {ranked} of {ranked + len(skipped)}; no fact found for: {', '.join(skipped)}")
    return notes


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


def render(r: Result) -> str:
    """One line a person can read."""
    if r.kind == "abstain":
        return "ABSTAIN: " + ("; ".join(r.notes) or "a needed fact is missing")
    if r.kind == "text":
        return str(r.value)
    if r.extra.get("first") is not None:
        a, b = r.extra["first"], r.extra["latest"]
        if a == b:
            return f"Not restated: both filings print {_fmt(a, r.unit)}"
        why = r.extra.get("diagnosis", {}).get("summary")
        return (f"Changed by {_fmt(b - a, r.unit)}: first reported {_fmt(a, r.unit)} ({r.extra['first_doc']}), "
                f"now {_fmt(b, r.unit)} ({r.extra['latest_doc']})" + (f". {why}" if why else ""))
    if r.kind in ("companies", "lines", "series"):
        shown = ", ".join(f"{name} ({v})" if isinstance(v, str) else f"{name} ({_fmt(v, r.unit)})" for name, v in r.value)
        if r.kind == "companies" and len(r.value) == 2 and all(not isinstance(v, str) for _, v in r.value):
            shown += f"; difference {_fmt(abs(r.value[0][1] - r.value[1][1]), r.unit)}"
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
