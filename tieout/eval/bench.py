"""Score a system on the benchmark: accuracy, precision and recall, by question type.

Definitions (fixed before any system was run on the questions)

Accuracy     correct answers / questions.
               numeric      within the question's relative tolerance of the gold value
               text         same after normalisation (case, punctuation, "LLP")
               company      the same company; for a ranked list, the same companies in the same order
               unanswerable correct only if the system abstains
               comparison   "which of A or B ..., and by how much" is answered with both values, the
                            winner and the gap. Against a company gold the winner must match; against
                            a numeric gold the gap or the winner's value must match. (Added 21 Nov
                            before the first round-2 run: two blind authors recorded different halves
                            of such two-part questions. On 22 Nov, before the held-out run, the same
                            rule was applied to two lines of one company, such as two products: it
                            had read only the winner's value. Also on 22 Nov: a comparison of three
                            or more prints the first minus the last as well ("how much bigger is the
                            largest than the smallest"), and that figure counts as one of its
                            stated figures.)

Retrieved    every fact the system read to produce its answer: for the graph system, every
facts        leaf of the proof tree, including every company scanned for a ranking. Nothing
             is trimmed after the fact.

Gold facts   written by hand from the PDF pages as (document, page, printed value), before the
             system was run. They do not refer to graph ids.

A retrieved fact matches a gold fact when it is the same company and the same value in full
units (so the same number read from another page or another filing still counts, and is
reported separately as not page-exact).

Precision    matched retrieved facts / retrieved facts.   Abstention with nothing retrieved: 1.
Recall       matched gold facts / gold facts.             Unanswerable questions have no gold facts
                                                          and are left out of recall.
Both are reported macro (mean over questions) and micro (pooled counts).

Trust       every answer carries a tier (tieout.execute.trust): VERIFIED, FLAGGED with its named
            weaknesses, or ABSTAIN. Accuracy is reported per tier, and "silent wrong" counts the
            answers that were wrong while marked VERIFIED: the number a bank cares about.

On 22 Nov at 03:00, before any held-out run, the code was brought in line with the rules above
in three places where it had drifted: unanswerable questions are left out of recall even when
their notes list facts; a retrieved fact matches every gold fact with the same company and value;
a ranked list of segments, products or regions is scored like a ranked list of companies. Every
system, the baseline included, is scored by this one function.

Also on 22 Nov, before the held-out run: a gold in percentage points (a difference of two rates, 3.72)
is matched by the same difference stated as a fraction (0.0372), the unit every rate has in the graph;
the fresh rounds 5 and 6 had such golds and no answer could match them.

Also on 22 Nov, before any held-out run: a gold value read off one printed figure is matched by any
value that rounds to that figure as printed. A page that says "$14.0 trillion" and a table that
says 14,041,518 (millions) state the same fact; the first fresh development batch had a gold taken
from the sentence while the graph answered from the table.

TWO SCORERS (23 Nov, before the freeze and before any held-out run). Every rule relaxed on 22 Nov
was found from a miss, and the red team (research/17 Part 1b) gave a wrong answer that passes each.
So every headline number is reported under both:
  final   (default)  the 22 Nov rules, each tightened against its counter-example:
            printed slack only for a figure printed with at least two significant digits
              ("$1 billion" no longer admits 0.62 billion);
            "by how much" is scored on the gap: a winner's value equal to the gold counts only when
              the gap the answer states is the gap the key's own facts give;
            names of lines compared as sets of words, not substrings ("U.S." is not "Non-U.S.",
              "Other" is not "Other Bets");
            a fall asked as a size is read from the gold's own text, not the question's words
              ("did the margin decline or improve" no longer turns +0.4 into a match for -0.4);
            a count kept in the page's scale only when the gold equals one of its facts' values;
            a dollar amount of the wrong sign is wrong when the question asks for a change or a
              difference (an outflow printed negative and asked as an amount still matches).
  strict  (TIEOUT_SCORER=strict)  every 22 Nov relaxation switched off (the scorer of 21 Nov), with
            the gap and sign rules above kept, so strict never passes an answer final fails.
Each row records `correct` (the scorer selected by TIEOUT_SCORER, final by default) and
`correct_strict` and `correct_final`, and the prediction itself (`pred`) for rescoring.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
import re
from collections import defaultdict
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark" / "questions.jsonl"
TYPES = ("single_fact", "multi_hop", "cross_filing")


def load_benchmark(path: pathlib.Path = BENCH) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_text(x) -> str:
    s = str(x).lower()
    s = re.sub(r"\b(llp|inc|corp|corporation|company|co|the|plc|n\.a)\b\.?", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def company_to_ticker(x, companies: dict) -> Optional[str]:
    """Gold answers name companies as people do; map them to tickers."""
    if x in companies:
        return x
    n = _norm_text(x)
    alias = {"google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "jj": "JNJ", "johnsonjohnson": "JNJ",
             "jpmorgan": "JPM", "jpmorganchase": "JPM", "bankofamerica": "BAC", "bofa": "BAC", "exxon": "XOM",
             "exxonmobil": "XOM", "cocacola": "KO", "coke": "KO", "pg": "PG", "proctergamble": "PG",
             "unitedhealth": "UNH", "unitedhealthgroup": "UNH", "wellsfargo": "WFC", "goldmansachs": "GS",
             "goldman": "GS", "morganstanley": "MS", "americanexpress": "AXP", "amex": "AXP", "homedepot": "HD",
             "generalmotors": "GM", "gm": "GM", "citigroup": "C", "citi": "C", "merck": "MRK", "pfizer": "PFE"}
    if n in alias:
        return alias[n]
    for t, c in companies.items():
        cn = _norm_text(c["name"])
        if n and (n == cn or n in cn or cn in n):
            return t
    return None


_DECLINE = re.compile(r"\b(fall|fell|fallen|declin\w*|decreas\w*|drop\w*|shr[aiu]nk|reduc\w*|lower|narrow\w*)\b", re.I)
_RISE = re.compile(r"\b(rose|risen|rise|increas\w*|grew|grow\w*|improv\w*|widen\w*|higher|up)\b", re.I)
# a question that asks for a change or a direction: the sign of a dollar answer is then part of it. "How much
# do the two differ", "the gap", "the difference" ask for a size and are left out (23 Nov: round 5 Q17, "how
# much does the figure in the 2025 10-K differ from the 2026 one", gold +4,720 for an answer of -4,720 million)
_CHANGE = re.compile(r"\b(chang\w*|increas\w*|decreas\w*|grow\w*|grew|growth|rose|fell|fall|declin\w*|drop\w*|"
                     r"more than|less than|fewer than|how much (more|less|higher|lower|bigger|larger|smaller|greater))\b",
                     re.I)
# a question that asks how far apart two figures are: the gap is the answer
_BY_HOW_MUCH = re.compile(r"\bby how much\b|\bhow much (more|less|higher|lower|bigger|larger|smaller|greater|bigger)\b"
                          r"|\bhow many (more|fewer|percentage points?|points?|basis points)\b|\b(difference|gap)\b"
                          r"|\bby what (amount|margin)\b|\bby how many\b", re.I)
_SOFT_NAME = {"the", "and", "of", "segment", "segments", "total", "net", "revenue", "revenues", "sales", "business",
              "group", "division", "inc", "co", "company", "operating"}


def scorer_mode() -> str:
    """'final' (default) or 'strict' (TIEOUT_SCORER=strict): see the module docstring."""
    return "strict" if os.environ.get("TIEOUT_SCORER", "").strip().lower() == "strict" else "final"


def _name_words(x) -> frozenset:
    """A line's name as a set of words: "U.S." -> {us}, "Non-U.S." -> {non, us}, "Other Bets" -> {other, bets}."""
    s = re.sub(r"\bu\.\s?s\.?", "us", str(x).lower())
    return frozenset(w for w in re.findall(r"[a-z0-9]+", s) if w not in _SOFT_NAME)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fact_values(q: dict) -> list:
    return [sf["value"] for sf in q.get("supporting_facts") or [] if _num(sf.get("value"))]


def _comparison(q: dict, gap, spread, winner, companies: dict, mode: str) -> bool:
    """A comparison answered with both sides: the winner's figure and the gap (and, for three or more,
    the first minus the last). Matched on the gap; on the winner's figure only when the question does
    not ask "by how much", or when it does and the gap stated is the gap the key's own facts give
    (research/17 Part 1b: gold = winner's 10, the other side misread as 3 for 8, gap 7, was right)."""
    one = lambda v: _num(v) and is_correct(q, {"kind": "number", "value": v}, companies, mode)  # noqa: E731
    if one(gap) or (mode == "final" and one(spread)):
        return True
    if not one(winner):
        return False
    if not _BY_HOW_MUCH.search(q.get("question", "")):
        return True
    gold = q.get("answer_value")
    for other in _fact_values(q):
        if abs(other - gold) > 1e-9 * max(1.0, abs(gold)) and _num(gap):
            target = abs(abs(gold) - abs(other))
            tol = max(q.get("tolerance", 0.005) or 0.0, 1e-9) * max(target, 1e-12)
            if abs(abs(gap) - target) <= tol:
                return True
    return False


def is_correct(q: dict, pred: dict, companies: dict, mode: Optional[str] = None) -> bool:
    """pred = {"kind": number|text|companies|abstain, "value": ...}. `mode`: "final" or "strict"
    (default: TIEOUT_SCORER, else final)."""
    mode = mode or scorer_mode()
    strict = mode == "strict"
    gold = q.get("answer_value")
    if gold is None or "unanswerable" in q.get("tags", []):
        return pred["kind"] == "abstain"
    if pred["kind"] == "abstain":
        return False
    unit = q.get("answer_unit", "")
    if pred["kind"] == "lines" and pred["value"] and not isinstance(gold, list):
        # the parts of a total (segments, products): the answer is the top line, by name or by value;
        # two lines compared ("which of the two products, and by how much") also carry the gap.
        # (A gold that is itself a list is compared line by line further down: until 22 Nov 22:10 this
        # branch reduced the answer to its first line first, so a correct ranking of six lines was
        # scored wrong; found on round 7, T11, corrected before the held-out run.)
        name, value = pred["value"][0]
        extra = pred.get("extra") or {}
        if _num(gold) and extra.get("gap") is not None and not strict:   # strict: the winner's value only (21 Nov)
            return _comparison(q, extra["gap"], extra.get("spread"), value, companies, mode)
        pred = {"kind": "number", "value": value} if isinstance(gold, (int, float)) else {"kind": "text", "value": name}
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        alt = pred.get("extra") or {}
        if pred["kind"] == "companies" and alt.get("gap") is not None and pred["value"]:
            return _comparison(q, alt["gap"], alt.get("spread"), pred["value"][0][1], companies, mode)
        if pred["kind"] != "number":
            return False
        tol = q.get("tolerance", 0.005) or 0.0
        v = pred["value"]
        if not _num(v):
            return False
        if alt.get("first") is not None and v == 0 and gold != 0:
            v = alt["latest"]                         # "not restated, both filings print X": X is the answer
        if abs(gold) < 1e-12:
            return abs(v) < 1e-9
        slack = max(tol, 1e-9) * abs(gold) + (0.0 if strict else printed_slack(q))
        if not strict and unit == "percent_points" and abs(100 * v - gold) <= slack:
            return True                               # 3.72 points stated as the fraction 0.0372
        if abs(v - gold) <= slack:
            return True
        if unit == "USD" and abs(abs(v) - abs(gold)) <= slack and not _CHANGE.search(q.get("question", "")):
            return True                               # an outflow printed negative, asked as an amount
        if strict:
            return False
        # a count the key keeps in the page's own scale ("1,927" in a table of thousands of barrels a day)
        # is the same quantity as the graph's full-unit figure (1,927,000): matched when every supporting
        # fact states that scale and is recorded as printed (dated 22 Nov 22:25, round 8, W17; corrected
        # before the held-out run), and (23 Nov) only when the gold itself is in the page's scale, of the
        # size of its facts (a fact, or a sum or gap of them: round 8 W17, 1,927 - 1,038 = 889 thousand
        # barrels a day): a key that converted its gold to full units but left its facts as printed does
        # not admit a double-scaled answer
        scale = as_printed_scale(q) if unit == "count" else None
        facts = [abs(f) for f in _fact_values(q)]
        if scale and facts and abs(gold) <= 10 * max(facts) and abs(v - gold * scale) <= slack * scale:
            return True
        # a fall asked for as a size ("by how many turns did coverage fall": 3.53) is answered by a change
        # of minus that size (dated 22 Nov 22:10, round 7, U11; corrected before the held-out run). Since
        # 23 Nov the fall is read from the gold's own words, not the question's: "did it decline or
        # improve" names both.
        said = str(q.get("answer") or "")
        return (gold > 0 > v and abs(-v - gold) <= slack and bool(_DECLINE.search(said))
                and not _RISE.search(said))
    if isinstance(gold, list):
        if pred["kind"] == "lines":                   # a ranked list of segments, products or regions
            if strict:
                return False                          # 21 Nov: a ranked list of lines was not compared line by line
            want, got = [_name_words(x) for x in gold], [_name_words(n) for n, _ in pred["value"]]
            return len(got) >= len(want) and all(w and w == g for w, g in zip(want, got))
        want = [company_to_ticker(x, companies) or _norm_text(x) for x in gold]
        if pred["kind"] != "companies":
            return False
        got = [t for t, _ in pred["value"]]
        return got[:len(want)] == want
    if unit == "company" and company_to_ticker(gold, companies) is not None:
        want = company_to_ticker(gold, companies)
        if pred["kind"] == "companies":
            return bool(pred["value"]) and pred["value"][0][0] == want
        return company_to_ticker(pred["value"], companies) == want
    g, p = _norm_text(gold), _norm_text(pred["value"])
    return bool(g) and bool(p) and (g == p or g in p or p in g)


_SCALE_WORD = re.compile(r"\b(thousand|million|billion)s?\b", re.I)


def as_printed_scale(q: dict) -> Optional[float]:
    """The scale a count's answer key was left in, when it was left as printed: 1000.0 if every
    supporting fact says "thousands ..." and records the printed digits as its value. None when the
    key is already in full units, states no scale, or mixes scales."""
    scales = set()
    for sf in q.get("supporting_facts") or []:
        word = _SCALE_WORD.search(str(sf.get("scale") or ""))
        digits = re.search(r"\d[\d,]*\.?\d*", str(sf.get("printed") or ""))
        if not word or not digits or not isinstance(sf.get("value"), (int, float)):
            return None
        if abs(float(digits.group(0).replace(",", "")) - abs(sf["value"])) > 1e-9 * max(1.0, abs(sf["value"])):
            return None                               # the key already converted this figure to full units
        scales.add({"thousand": 1e3, "million": 1e6, "billion": 1e9}[word.group(1).lower()])
    return scales.pop() if len(scales) == 1 else None


def printed_slack(q: dict) -> float:
    """Half a unit of the last digit of the gold figure as its page prints it: "$14.0 trillion" stands
    for every amount from 13.95 to 14.05 trillion. 0 unless the gold is one printed figure, and (23 Nov)
    0 for a figure printed with fewer than two significant digits: "$1 billion" would otherwise admit
    anything from 0.5 to 1.5 billion (research/17 Part 1b)."""
    gold = q.get("answer_value")
    for sf in q.get("supporting_facts") or []:
        printed = sf.get("printed")
        if sf.get("value") != gold or not isinstance(printed, str):
            continue
        m = re.search(r"\d[\d,]*\.?\d*", printed)
        text = m.group(0).replace(",", "") if m else ""
        if text and float(text):
            if len(text.replace(".", "").lstrip("0").rstrip("0" if "." not in text else "")) < 2:
                return 0.0
            decimals = len(text.split(".")[1]) if "." in text else 0
            return 0.5 * 10 ** -decimals * abs(gold / float(text))
    return 0.0


def fact_match(q: dict, retrieved: list, companies: dict) -> dict:
    """retrieved = [{"ticker", "value", "doc", "page"}, ...] -> counts for precision and recall."""
    gold = q.get("supporting_facts", []) or []
    if q.get("answer_value") is None or "unanswerable" in q.get("tags", []):
        gold = []                                     # unanswerable questions are left out of recall
    gold_keys = []
    for gf in gold:
        ticker = gf["doc"].split("_")[0]
        v = gf.get("value")
        gold_keys.append((ticker, _vkey(v if v is not None else gf.get("printed")), gf["doc"], gf.get("page")))
    used = set()
    matched_ret = page_exact = 0
    for rf in retrieved:
        key = (rf["ticker"], _vkey(rf["value"]))
        hits = [i for i, gk in enumerate(gold_keys) if gk[0] == key[0] and _same(gk[1], key[1])]
        if hits:                                      # the same company and value matches every such gold fact
            matched_ret += 1
            used.update(hits)
            if any(gold_keys[i][2] == rf["doc"] and gold_keys[i][3] == rf["page"] for i in hits):
                page_exact += 1
    return {"retrieved": len(retrieved), "matched_retrieved": matched_ret, "gold": len(gold_keys),
            "matched_gold": len(used), "page_exact": page_exact}


def _same(a, b) -> bool:
    """Numbers must be equal; text matches when one normalised string contains the other
    ("/s/ KPMG LLP" on the page and "KPMG" in the graph are the same fact)."""
    if a == b:
        return True
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] == "t":
        return bool(a[1]) and bool(b[1]) and (a[1] in b[1] or b[1] in a[1])
    return False


def _vkey(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return ("n", round(abs(float(v)), 4))        # outflows are printed negative and asked as amounts
    s = str(v).strip()
    if re.fullmatch(r"\(?-?[\d,]+(\.\d+)?\)?", s):
        return ("n", round(abs(float(s.strip("()").replace(",", ""))), 4))
    return ("t", _norm_text(s))


def score(rows: list) -> dict:
    """rows: per-question dicts with type, correct, match (from fact_match), abstained."""
    out = {}
    for name, subset in [("overall", rows)] + [(t, [r for r in rows if r["type"] == t]) for t in TYPES]:
        n = len(subset)
        if not n:
            continue
        k = sum(1 for r in subset if r["correct"])
        ks = sum(1 for r in subset if r.get("correct_strict", r["correct"]))
        kf = sum(1 for r in subset if r.get("correct_final", r["correct"]))
        prec = [(r["match"]["matched_retrieved"] / r["match"]["retrieved"]) if r["match"]["retrieved"] else
                (1.0 if (r["abstained"] or r["match"]["gold"] == 0) else 0.0) for r in subset]
        rec = [r["match"]["matched_gold"] / r["match"]["gold"] for r in subset if r["match"]["gold"]]
        tr = sum(r["match"]["retrieved"] for r in subset)
        tg = sum(r["match"]["gold"] for r in subset)
        out[name] = {
            "n": n, "correct": k, "accuracy": k / n, "accuracy_ci95": wilson(k, n),
            "correct_final": kf, "correct_strict": ks, "accuracy_strict_ci95": wilson(ks, n),
            "precision_macro": sum(prec) / len(prec), "recall_macro": (sum(rec) / len(rec)) if rec else None,
            "precision_micro": (sum(r["match"]["matched_retrieved"] for r in subset) / tr) if tr else None,
            "recall_micro": (sum(r["match"]["matched_gold"] for r in subset) / tg) if tg else None,
            "page_exact_share": (sum(r["match"]["page_exact"] for r in subset) /
                                 max(1, sum(r["match"]["matched_retrieved"] for r in subset))),
            "abstained": sum(1 for r in subset if r["abstained"]),
            "by_tier": {t: {"n": sum(1 for r in subset if r.get("tier") == t),
                            "correct": sum(1 for r in subset if r.get("tier") == t and r["correct"])}
                        for t in ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED", "ABSTAIN")},
            "silent_wrong": sum(1 for r in subset if r.get("tier") == "VERIFIED" and not r["correct"]),
            "witnessed_wrong": sum(1 for r in subset if r.get("tier") == "WITNESSED" and not r["correct"]),
            # the same questions as the plan road alone served them, before the second road (readers.second_road)
            "plan_road_correct": (sum(1 for r in subset if (r.get("plan_road") or {}).get("correct"))
                                  if any(r.get("plan_road") for r in subset) else None),
            "answered_accuracy": (sum(1 for r in subset if r["correct"] and not r["abstained"]) /
                                  max(1, sum(1 for r in subset if not r["abstained"]))),
        }
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def paired_bootstrap(a: list, b: list, n_boot: int = 10000, seed: int = 11) -> dict:
    """a, b: 0/1 correctness of two systems on the same questions. Difference a - b."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return {"diff": (sum(a) - sum(b)) / n, "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]],
            "a_only": sum(1 for x, y in zip(a, b) if x and not y), "b_only": sum(1 for x, y in zip(a, b) if y and not x),
            "n": n, "n_boot": n_boot, "seed": seed}


def _jsonable(x):
    """A prediction as JSON keeps it rescorable: tuples become lists, anything else its string."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def rescore(row: dict, q: dict, companies: dict, mode: str) -> Optional[bool]:
    """Score a stored row again under `mode` from its recorded prediction; None if it has none."""
    pred = row.get("pred")
    if not isinstance(pred, dict):
        return None
    value = pred.get("value")
    if pred.get("kind") in ("companies", "lines", "series") and isinstance(value, list):
        value = [tuple(v) if isinstance(v, list) else v for v in value]
    return is_correct(q, {**pred, "value": value}, companies, mode)


def run_graph_system(questions: list, g, answer_fn: Callable, label: str = "tieout") -> list:
    """answer_fn(question dict) -> the dict returned by tieout.ask.answer."""
    rows = []
    for q in questions:
        out = answer_fn(q)
        res = out["result"]
        retrieved = [{"ticker": lf.ticker, "value": lf.value, "doc": lf.doc, "page": lf.page} for lf in res.leaves]
        pred = {"kind": res.kind, "value": res.value, "extra": res.extra}
        both = {m: is_correct(q, pred, g.companies, m) for m in ("final", "strict")}
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "gold": q.get("answer"),
            "answer": out["answer"], "correct": both[scorer_mode()],
            "correct_final": both["final"], "correct_strict": both["strict"], "pred": _jsonable(pred),
            "abstained": res.kind == "abstain", "match": fact_match(q, retrieved, g.companies),
            "plan": out.get("plan"), "notes": list(dict.fromkeys(res.notes))[:6],
            # each link with how its pick was made (two pickers agreed, an arbiter settled it, one picker)
            "learned": [{k: e.get(k) for k in ("kind", "ticker", "concept", "phrase", "state", "label", "doc", "page",
                                               "picked")} for e in out.get("learned", [])],
            "seconds": round(out.get("seconds", 0.0), 2), "system": label, "tags": q.get("tags", []),
            "repaired": out.get("repaired", False), "plan_errors": out.get("plan_errors", []),
            "tier": res.tier or ("ABSTAIN" if res.kind == "abstain" else "FLAGGED"), "tier_reasons": res.tier_reasons[:4],
        })
        if out.get("reading"):                        # two readers and an arbiter: what each one answered
            rows[-1]["reading"] = out["reading"]["case"]
            rows[-1]["readers"] = [
                {"role": s["role"], "model": s["model"], "answer": s["answer"], "tier": s["tier"], "plan": s["plan"],
                 "correct": is_correct(q, {"kind": s["result"].kind, "value": s["result"].value,
                                           "extra": s["result"].extra}, g.companies)}
                for s in out["reading"]["readers"]]
            road = out["reading"].get("plan_road")
            if road:                                  # what the plan road alone served, before the second road
                rows[-1]["plan_road"] = {"answer": road["answer"], "tier": road["tier"],
                                         "correct": is_correct(q, {"kind": road["kind"], "value": road["value"],
                                                                   "extra": road["extra"]}, g.companies)}
            if out["reading"].get("passage"):
                rows[-1]["passage"] = _jsonable(out["reading"]["passage"])
    return rows
