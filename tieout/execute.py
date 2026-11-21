"""Run a plan against the graph and return an answer that carries its proof.

The executor is deterministic and does integer arithmetic on what the graph
holds. Every value it returns lists the readings it rests on (the leaves of
the proof tree): filing, page, printed label, printed text and whether the
reading is verified. If a needed fact is missing, or its reading sits inside
a formula that does not hold, the executor abstains and names the fact.

"Retrieved facts" for the benchmark's precision and recall are exactly
`Result.leaves`: every fact the executor read, including every company it
scanned for a ranking.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .concepts import concept_defs
from .graph import Graph
from .plan import CLUSTERS, NUMERIC_ATTRS, metric_defs, validate

OUTFLOW = {"CapitalExpenditures", "DividendsPaid", "ShareRepurchases"}   # printed negative, asked as amounts


@dataclass
class Leaf:
    id: str
    ticker: str
    what: str                 # concept or attribute
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
    kind: str                 # number | text | companies | abstain
    value: Any = None
    unit: str = ""
    leaves: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tree: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)     # restated: {"first": v, "latest": v}

    @property
    def ok(self) -> bool:
        return self.kind != "abstain"


class TypeErrorInPlan(Exception):
    pass


class Executor:
    def __init__(self, graph: Graph, use_checks: bool = True, use_derivations: bool = True,
                 require_verified: bool = False):
        self.g = graph
        self.use_checks = use_checks              # ablation: serve failed readings as if they were fine
        self.require_verified = require_verified  # strict policy: abstain unless every leaf is verified
        self.use_derivations = use_derivations    # ablation: no metric formulas, no period lattice
        self.concepts = concept_defs()
        self.metrics = metric_defs()
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
            return Result("abstain", notes=["plan rejected: " + "; ".join(errs)])
        env: dict = {}
        try:
            for step in plan["steps"]:
                env[step["id"]] = getattr(self, "_op_" + step["op"])(step, env)
        except TypeErrorInPlan as e:
            return Result("abstain", notes=[f"type error: {e}"])
        final = env[plan["answer"]]
        # the answer rests on everything that was read on the way to it
        seen, leaves = set(), []
        for lf in final.leaves:
            if lf.id not in seen:
                seen.add(lf.id)
                leaves.append(lf)
        final.leaves = leaves
        return final

    # ------------------------------------------------------------ periods

    def _pid(self, ticker: str, spec: dict, nature: str, what: str) -> Optional[str]:
        fy = spec.get("fy")
        if fy in ("latest", "prior"):
            pat = re.compile(r"^FY(\d{4})@end$" if nature == "stock" else r"^FY(\d{4})$")
            years = sorted(int(m.group(1)) for p in self.g.periods_of(ticker, what) if (m := pat.match(p)))
            if not years:
                return None
            fy = years[-1] if fy == "latest" else years[-1] - 1
        base = f"FY{int(fy)}"
        q = spec.get("q")
        if q:
            base += f"Q{int(q)}"
            if spec.get("ytd") and nature == "flow" and int(q) > 1:
                base += "YTD"
        return base + ("@end" if nature == "stock" else "")

    # -------------------------------------------------------------- facts

    def _fact(self, ticker: str, concept: str, spec: dict, first: bool = False) -> Result:
        c = self.concepts[concept]
        pid = self._pid(ticker, spec, c.nature, concept)
        f = self.g.fact(ticker, concept, pid) if pid else None
        if f is None:
            derived = self._derive_q4(ticker, concept, spec) if self.use_derivations else None
            if derived is not None:
                return derived
            return Result("abstain", missing=[(ticker, concept, pid or str(spec))],
                          notes=[f"no fact for {ticker} {concept} {pid or spec}"])
        return self._from_fact(f, ticker, concept, pid, first, c.unit, concept in OUTFLOW)

    def _item(self, ticker: str, phrase: str, spec: dict, first: bool = False) -> Result:
        """A line item outside the seed concepts, bound on demand by the living layer."""
        from .items import resolve
        key = resolve(self.g, ticker, phrase)
        if key is None:
            return Result("abstain", missing=[(ticker, "item:" + phrase, str(spec))],
                          notes=[f"no binding yet for '{phrase}' at {ticker}"])
        f = pid = None
        for nature in ("flow", "stock"):
            pid = self._pid(ticker, spec, nature, key)
            f = self.g.fact(ticker, key, pid) if pid else None
            if f is not None:
                break
        if f is None:
            return Result("abstain", missing=[],
                          notes=[f"'{phrase}' is bound for {ticker} but has no value for {pid or spec}"])
        unit = self.g.readings[f.readings[0]].unit
        return self._from_fact(f, ticker, key, pid, first, unit, False)

    def _from_fact(self, f, ticker: str, concept: str, pid: str, first: bool, unit: str, outflow: bool) -> Result:
        r = self.g.best_reading(f, as_first_reported=first)
        leaf = Leaf(r.id, ticker, concept, r.period, r.doc, r.page, r.label, r.text, r.status, r.value)
        if self.use_checks and r.status == "failed":
            return Result("abstain", missing=[(ticker, concept, pid)], leaves=[leaf],
                          notes=[f"{ticker} {concept} {pid} is quarantined: its formula does not hold on page {r.page}"])
        if self.use_checks and self.require_verified and r.status != "verified":
            return Result("abstain", missing=[], leaves=[leaf],
                          notes=[f"{ticker} {concept} {pid} is not covered by any passing check (strict policy)"])
        value = abs(r.value) if outflow else r.value
        notes = [f"{ticker} {pid} ends {r.period_end}"]
        if f.binding_state != "active":
            notes.append(f"{ticker} {concept}: label '{r.label[:40]}' matched by wording only, structure did not confirm it")
        if r.status != "verified":
            notes.append(f"{ticker} {concept} {pid}: reading is not covered by any check")
        if any(self.g.readings[i].superseded for i in f.readings):
            notes.append(f"{ticker} {concept} {pid} was restated; "
                         + ("first-reported" if first else "latest-reported") + " value used")
        return Result("number", value, unit, [leaf], notes,
                      {"op": "get", "what": f"{ticker} {concept} {pid}", "value": value, "page": r.page, "doc": r.doc})

    def _derive_q4(self, ticker: str, concept: str, spec: dict) -> Optional[Result]:
        """Period lattice: the fourth quarter is the year minus the nine months to date."""
        c = self.concepts[concept]
        if c.nature != "flow" or spec.get("q") != 4 or spec.get("ytd"):
            return None
        year = self._fact_plain(ticker, concept, {"fy": spec["fy"]})
        ytd = self._fact_plain(ticker, concept, {"fy": spec["fy"], "q": 3, "ytd": True})
        if year is None or ytd is None:
            return None
        value = year.value - ytd.value
        return Result("number", value, c.unit, year.leaves + ytd.leaves,
                      year.notes + ytd.notes + [f"{ticker} {concept} Q4 is not printed; derived as full year minus nine months"],
                      {"op": "derive", "what": f"{ticker} {concept} FY{spec['fy']}Q4 = FY - Q3YTD", "value": value,
                       "children": [year.tree, ytd.tree]})

    def _fact_plain(self, ticker, concept, spec) -> Optional[Result]:
        saved, self.use_derivations = self.use_derivations, False
        try:
            r = self._fact(ticker, concept, spec)
        finally:
            self.use_derivations = saved
        return r if r.ok else None

    # ------------------------------------------------------------ metrics

    def _metric(self, ticker: str, name: str, spec: dict) -> Result:
        m = self.metrics[name]
        if not self.use_derivations:
            return Result("abstain", notes=[f"derivations are switched off; {name} is not a printed fact"])
        last = None
        for formula in (m["formula"], m.get("fallback")):
            if not formula:
                continue
            res = self._eval_formula(ticker, formula, spec, m["unit"], name)
            if res.ok:
                if formula != m["formula"]:
                    res.notes.append(f"{ticker} {name}: computed as {formula} because a direct line is not printed")
                return res
            last = res
        return last

    def _eval_formula(self, ticker, formula, spec, unit, name) -> Result:
        leaves, notes, children, missing = [], [], [], []

        def ev(node):
            if isinstance(node, ast.BinOp):
                a, b = ev(node.left), ev(node.right)
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
                v = ev(node.args[0])
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
            raise TypeErrorInPlan(f"unsupported formula element in {formula}")

        value = ev(ast.parse(formula, mode="eval").body)
        if value is None:
            return Result("abstain", leaves=leaves, notes=notes, missing=missing)
        return Result("number", value, unit, leaves, notes,
                      {"op": "metric", "what": f"{ticker} {name} = {formula}", "value": value, "children": children})

    def _measure(self, ticker: str, by: dict, spec: dict) -> Result:
        if by.get("attr"):
            r = self._attr(ticker, by["attr"])
            if r.ok and r.kind != "number":
                raise TypeErrorInPlan(f"attribute {by['attr']} is not a number")
            return r
        if by.get("item"):
            return self._item(ticker, by["item"], spec)
        if by.get("metric"):
            return self._metric(ticker, by["metric"], spec)
        return self._fact(ticker, by["concept"], spec)

    # ---------------------------------------------------------------- ops

    def _company(self, ref, env) -> str:
        if isinstance(ref, str) and ref.startswith("$"):
            r = env[ref[1:]]
            if r.kind != "companies" or not r.value:
                raise TypeErrorInPlan(f"{ref} is not a company")
            return r.value[0][0]
        return ref

    def _set(self, over, env) -> tuple:
        if over in (None, "all"):
            return sorted(self.g.companies), []
        if isinstance(over, dict):
            return sorted(t for t, c in self.g.companies.items() if c["cluster"] == over["cluster"]), []
        if isinstance(over, list):
            return list(over), []
        r = env[over[1:]]
        if r.kind != "companies":
            raise TypeErrorInPlan(f"{over} is not a set of companies")
        return [t for t, _ in r.value], list(r.leaves)

    def _op_get(self, s, env) -> Result:
        first = s.get("as") == "first_reported"
        if s.get("item"):
            return self._item(self._company(s["company"], env), s["item"], s["period"], first=first)
        return self._fact(self._company(s["company"], env), s["concept"], s["period"], first=first)

    def _op_metric(self, s, env) -> Result:
        return self._metric(self._company(s["company"], env), s["metric"], s["period"])

    def _op_calc(self, s, env) -> Result:
        leaves, notes, children = [], [], []
        units = []

        def ev(node):
            if isinstance(node, ast.BinOp):
                (a, ua), (b, ub) = ev(node.left), ev(node.right)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    if ua and ub and ua != ub:
                        raise TypeErrorInPlan(f"cannot add or subtract {ua} and {ub}")
                    return (a + b if isinstance(node.op, ast.Add) else a - b), ua or ub
                if isinstance(node.op, ast.Mult):
                    return a * b, ua or ub
                if isinstance(node.op, ast.Div):
                    if not b:
                        raise TypeErrorInPlan("division by zero")
                    unit = "ratio" if ua == ub or not ua else (ua if not ub else f"{ua}/{ub.rstrip('s')}")
                    return a / b, unit
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                a, ua = ev(node.operand)
                return -a, ua
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "abs":
                a, ua = ev(node.args[0])
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
            raise TypeErrorInPlan(f"unsupported expression: {s['expr']}")

        try:
            value, unit = ev(ast.parse(s["expr"], mode="eval").body)
        except KeyError as e:
            bad = env.get(str(e).strip("'"))
            return Result("abstain", leaves=leaves + (bad.leaves if bad else []),
                          notes=notes + (bad.notes if bad else [f"step {e} has no number"]),
                          missing=bad.missing if bad else [])
        return Result("number", value, unit, leaves, notes,
                      {"op": "calc", "what": s["expr"], "value": value, "children": children})

    def _op_change(self, s, env) -> Result:
        t = self._company(s["company"], env)
        by = ({"metric": s["metric"]} if s.get("metric") else
              {"item": s["item"]} if s.get("item") else {"concept": s["concept"]})
        a, b = self._measure(t, by, s["from"]), self._measure(t, by, s["to"])
        leaves, notes = a.leaves + b.leaves, a.notes + b.notes
        if not (a.ok and b.ok):
            return Result("abstain", leaves=leaves, notes=notes, missing=a.missing + b.missing)
        if s.get("mode", "pct") == "abs":
            value, unit = b.value - a.value, a.unit
        else:
            if not a.value:
                return Result("abstain", leaves=leaves, notes=notes + ["base value is zero"])
            value, unit = (b.value - a.value) / abs(a.value), "ratio"
        return Result("number", value, unit, leaves, notes,
                      {"op": "change", "what": f"{t} {list(by.values())[0]} {s['from']} -> {s['to']}",
                       "value": value, "children": [a.tree, b.tree]})

    def _op_rank(self, s, env) -> Result:
        tickers, leaves = self._set(s.get("over"), env)
        rows, notes, skipped, missing = [], [], [], []
        for t in tickers:
            r = self._measure(t, s["by"], s["period"])
            leaves.extend(r.leaves)
            if r.ok:
                rows.append((t, r.value))
            else:
                skipped.append(t)
                missing.extend(r.missing)
        if not rows:
            return Result("abstain", leaves=leaves, notes=["no company in the set has this fact"], missing=missing)
        if skipped and isinstance(s.get("over"), list):
            # the question named these companies; ranking the ones we happen to have would look complete and not be
            return Result("abstain", leaves=leaves, missing=missing,
                          notes=[f"cannot rank the named companies: no fact for {', '.join(skipped)}"])
        rows.sort(key=lambda x: x[1], reverse=s.get("order", "desc") == "desc")
        if skipped:
            notes.append(f"ranked {len(rows)} of {len(tickers)} companies; no fact for: {', '.join(skipped)}")
        k = int(s.get("k", 1))
        by = s["by"]
        unit = (self.metrics[by["metric"]]["unit"] if by.get("metric") else
                self.concepts[by["concept"]].unit if by.get("concept") else
                NUMERIC_ATTRS.get(by.get("attr"), "USD"))
        return Result("companies", rows[:k], unit, leaves, notes,
                      {"op": "rank", "what": f"{list(by.values())[0]} over {len(rows)} companies", "value": rows[:k]},
                      missing)

    def _op_filter(self, s, env) -> Result:
        tickers, leaves = self._set(s.get("over"), env)
        w = s["where"]
        keep, notes = [], []
        for t in tickers:
            if "attr" in w:
                r = self._attr(t, w["attr"])
                leaves.extend(r.leaves)
                if r.ok and _norm(r.value) == _norm(w.get("eq")):
                    keep.append((t, r.value))
                continue
            r = self._measure(t, w["by"], w["period"])
            leaves.extend(r.leaves)
            if not r.ok:
                continue
            v = r.value
            if ("gt" in w and v > w["gt"]) or ("lt" in w and v < w["lt"]) \
                    or ("ge" in w and v >= w["ge"]) or ("le" in w and v <= w["le"]):
                keep.append((t, v))
        return Result("companies", keep, "", leaves, notes, {"op": "filter", "what": str(w), "value": keep})

    def _attr(self, ticker: str, attr: str) -> Result:
        if attr == "cluster":
            return Result("text", self.g.companies[ticker]["cluster"], "", [], [], {"op": "attr", "what": f"{ticker} cluster"})
        t = self._text.get((ticker, attr))
        if t is None:
            return Result("abstain", missing=[(ticker, attr, "")], notes=[f"no text fact {attr} for {ticker}"])
        leaf = Leaf(f"{t['doc']}:text:{attr}", ticker, attr, "", t["doc"], t["page"], attr, t["quote"][:160], "text", t["value"])
        if attr in NUMERIC_ATTRS:
            return Result("number", int(t["value"]), NUMERIC_ATTRS[attr], [leaf], [],
                          {"op": "attr", "what": f"{ticker} {attr}", "value": int(t["value"]), "page": t["page"], "doc": t["doc"]})
        return Result("text", t["value"], "", [leaf], [],
                      {"op": "attr", "what": f"{ticker} {attr}", "value": t["value"], "page": t["page"], "doc": t["doc"]})

    def _op_attr(self, s, env) -> Result:
        ref = s["company"]
        if isinstance(ref, str) and ref.startswith("$"):
            prior = env[ref[1:]]
            r = self._attr(self._company(ref, env), s["attr"])
            r.leaves = list(prior.leaves) + r.leaves
            r.notes = list(prior.notes) + r.notes
            return r
        return self._attr(ref, s["attr"])

    def _op_count(self, s, env) -> Result:
        r = env[s["of"][1:]]
        if r.kind != "companies":
            raise TypeErrorInPlan(f"{s['of']} is not a set of companies")
        return Result("number", len(r.value), "count", list(r.leaves), list(r.notes),
                      {"op": "count", "value": len(r.value)})

    def _op_restated(self, s, env) -> Result:
        t = self._company(s["company"], env)
        if s.get("item"):
            first, last = self._item(t, s["item"], s["period"], first=True), self._item(t, s["item"], s["period"])
        else:
            first = self._fact(t, s["concept"], s["period"], first=True)
            last = self._fact(t, s["concept"], s["period"], first=False)
        leaves, notes = first.leaves + last.leaves, first.notes + last.notes
        if not (first.ok and last.ok):
            return Result("abstain", leaves=leaves, notes=notes, missing=first.missing + last.missing)
        if first.leaves[0].doc == last.leaves[0].doc:
            notes.append(f"{t} {s.get('concept') or s.get('item')}: only one filing reports this period, so nothing to compare")
        return Result("number", last.value - first.value, first.unit, leaves, notes,
                      {"op": "restated", "what": f"{t} {s.get('concept') or s.get('item')} latest minus first reported",
                       "value": last.value - first.value, "children": [first.tree, last.tree]},
                      extra={"first": first.value, "latest": last.value,
                             "first_doc": first.leaves[0].doc, "latest_doc": last.leaves[0].doc})


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
        return (f"Restated by {_fmt(b - a, r.unit)}: first reported {_fmt(a, r.unit)} ({r.extra['first_doc']}), "
                f"latest {_fmt(b, r.unit)} ({r.extra['latest_doc']})")
    if r.kind == "companies":
        return ", ".join(f"{t} ({_fmt(v, r.unit)})" if not isinstance(v, str) else f"{t} ({v})" for t, v in r.value) or "none"
    return _fmt(r.value, r.unit)


def _fmt(v, unit: str) -> str:
    if unit == "USD":
        a = abs(v)
        if a >= 1e9:
            return f"${v / 1e9:,.3f} billion"
        if a >= 1e6:
            return f"${v / 1e6:,.1f} million"
        return f"${v:,.0f}"
    if unit == "ratio":
        return f"{v * 100:.2f}%" if abs(v) < 10 else f"{v:.2f}x"
    if unit == "USD/share":
        return f"${v:,.2f} per share"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.4f} {unit}".strip()
    return f"{int(v):,} {unit}".strip()
