"""Rescore old result files under the `final` and `strict` scorers of tieout/eval/bench.py.

Old result rows store the RENDERED answer string, not the structured prediction. This script parses the
string back into a pred (inverse of tieout.execute.render/_fmt), or uses the row's own "pred" when present,
and scores it with is_correct(q, pred, companies, mode) for mode in ("final", "strict").

Run from repo_a:
  PYTHONPATH=. PYTHONIOENCODING=utf-8 ..\\repo\\.venv\\Scripts\\python.exe scripts\\rescore_rendered.py \\
      --out ..\\dev_scripts\\a\\rescore_first_runs.txt --pairs RESULT.json=BENCH.jsonl ...
No LLM or network call is made.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.eval.bench import is_correct  # noqa: E402
from tieout.graph import Graph  # noqa: E402

NUM = r"-?[\d,]+(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def parse_val(s: str):
    """One rendered figure -> number, or None. Inverse of _fmt."""
    s = s.strip()
    m = re.fullmatch(r"(-?)\$(-?)(" + NUM + r")( billion| million)?( per share)?", s)
    if m:
        v = _f(m.group(3))
        v *= {" billion": 1e9, " million": 1e6}.get(m.group(4), 1.0)
        return -v if (m.group(1) or m.group(2)) else v
    m = re.fullmatch(r"(" + NUM + r")%", s)
    if m:
        return _f(m.group(1)) / 100
    m = re.fullmatch(r"(" + NUM + r")x", s)
    if m:
        return _f(m.group(1))
    m = re.fullmatch(r"(" + NUM + r")(?: [^()]+)?", s)      # "1,234 count", "322,847,710.3301 USD per count"
    if m:
        return _f(m.group(1))
    return None


_MONEY = r"-?\$-?[\d,]+(?:\.\d+)?(?: billion| million)?(?: per share)?"
_FIG = r"(?:" + _MONEY + r"|" + NUM + r"%|" + NUM + r"x|" + NUM + r"(?: [^()]+)?)"


def parse_answer(ans: str, companies: dict):
    """Rendered answer -> pred dict, or None when it falls back to text."""
    a = (ans or "").strip()
    if a.startswith("ABSTAIN"):
        return {"kind": "abstain", "value": None}
    # restatement forms
    m = re.fullmatch(r"Only one filing reports this period: (.+?) \(([^()]*)\)", a)
    if m:
        x = parse_val(m.group(1))
        if x is not None:
            return {"kind": "number", "value": 0.0, "extra": {"first": x, "latest": x}}
    m = re.fullmatch(r"Not restated: both filings print (.+)", a)
    if m:
        x = parse_val(m.group(1))
        if x is not None:
            return {"kind": "number", "value": 0.0, "extra": {"first": x, "latest": x}}
    m = re.match(r"Changed by (.+?): first reported (.+?) \(([^()]*)\), now (.+?) \(([^()]*)\)(?:\. .*)?$", a, re.S)
    if m:
        d, x, y = parse_val(m.group(1)), parse_val(m.group(2)), parse_val(m.group(4))
        if None not in (d, x, y):
            return {"kind": "number", "value": d, "extra": {"first": x, "latest": y}}
    # lists: "A (v), B (v); difference D; first minus last S"
    body, gap, spread = a, None, None
    ms = re.search(r"; first minus last (.+)$", body)
    if ms:
        spread = parse_val(ms.group(1))
        body = body[:ms.start()]
    mg = re.search(r"; difference (.+)$", body)
    if mg:
        gap = parse_val(mg.group(1))
        body = body[:mg.start()]
    items = []
    pos = 0
    pat = re.compile(r"(.+?) \((" + _FIG + r")\)(?=, |$)")
    while pos < len(body):
        m = pat.match(body, pos)
        if not m:
            items = None
            break
        v = parse_val(m.group(2))
        if v is None:
            items = None
            break
        items.append((m.group(1), v))
        pos = m.end()
        if body[pos:pos + 2] == ", ":
            pos += 2
    if items and (ms is None or spread is not None) and (mg is None or gap is not None):
        extra = {}
        nums = [v for _n, v in items if isinstance(v, (int, float))]
        if gap is None and len(nums) >= 2 and len(nums) == len(items):
            gap = abs(nums[0] - nums[1])          # the executor's extra["gap"] (rank, compare), printed only for two
        if gap is not None:
            extra["gap"] = gap
        if spread is not None:
            extra["spread"] = spread
        kind = "companies" if all(n in companies for n, _ in items) else "lines"
        pred = {"kind": kind, "value": items}
        if extra:
            pred["extra"] = extra
        return pred
    # "3: A, B"  (a count with its names)
    m = re.fullmatch(r"(-?\d+): .+", a, re.S)
    if m:
        return {"kind": "number", "value": float(m.group(1))}
    v = parse_val(a)
    if v is not None:
        return {"kind": "number", "value": v}
    return None


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()


def load_rows(path: pathlib.Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) else d


def resolve(p: str) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else ROOT / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="RESULT.json=BENCH.jsonl ...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Tee(args.out)
    companies = Graph.load().companies
    parsed_pool, total_fallback, total_rows = [], 0, 0
    fallback_examples = []
    out("file | n | recorded | final | strict   (parse: own_pred / parsed / text-fallback)")
    for pair in args.pairs:
        rp, bp = pair.split("=", 1)
        rpath, bpath = resolve(rp), resolve(bp)
        bench = {}
        for line in bpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                bench[q["id"]] = q
        rows = load_rows(rpath)
        n = rec = fin = stc = 0
        own = prs = fb = missing = 0
        diff_final, diff_strict = [], []
        for r in rows:
            q = bench.get(r["id"])
            if q is None:
                missing += 1
                continue
            n += 1
            if isinstance(r.get("pred"), dict):
                pred = r["pred"]
                own += 1
            else:
                pred = parse_answer(r.get("answer", ""), companies)
                if pred is None:
                    pred = {"kind": "text", "value": r.get("answer", "")}
                    fb += 1
                    g = q.get("answer_value")
                    fallback_examples.append((rpath.name, r["id"], str(r.get("answer", ""))[:110],
                                              "gold=" + str(g)[:40]))
                else:
                    prs += 1
                    parsed_pool.append((rpath.name, r["id"], str(r.get("answer", "")), pred))
            c = bool(r.get("correct"))
            f = is_correct(q, pred, companies, "final")
            s = is_correct(q, pred, companies, "strict")
            rec += c
            fin += f
            stc += s
            if f != c:
                diff_final.append(f"{r['id']}({'T' if c else 'F'}->{'T' if f else 'F'})")
            if s != c:
                diff_strict.append(f"{r['id']}({'T' if c else 'F'}->{'T' if s else 'F'})")
        total_fallback += fb
        total_rows += n
        out(f"\n== {rpath.name}  [bench {bpath.relative_to(ROOT) if bpath.is_relative_to(ROOT) else bpath}]")
        out(f"n={n}  recorded={rec}  final={fin}  strict={stc}   parse: own_pred={own} parsed={prs} "
            f"text_fallback={fb}" + (f"  ROWS_NOT_IN_BENCH={missing}" if missing else ""))
        out("  final differs from recorded:  " + (", ".join(diff_final) or "none"))
        out("  strict differs from recorded: " + (", ".join(diff_strict) or "none"))
    out(f"\nTOTAL rows={total_rows}  text_fallback={total_fallback}")
    out("\nText-fallback rows (answer string -> gold):")
    for fn, i, a, g in fallback_examples:
        out(f"  {fn} {i}: {a!r}  {g}")
    out("\nSpot check of 10 parsed rows (answer string -> pred), stratified by kind:")
    rng = random.Random(435)
    by = {}
    for t in parsed_pool:
        by.setdefault(t[3]["kind"] + ("+extra" if t[3].get("extra") else ""), []).append(t)
    pick = []
    for k, lst in sorted(by.items()):
        pick += rng.sample(lst, min(2, len(lst)))
    rest = [t for t in parsed_pool if t not in pick]
    pick += rng.sample(rest, max(0, 10 - len(pick)))
    for fn, i, a, p in pick[:10]:
        out(f"  {fn} {i}: {a!r}\n      -> {p}")


if __name__ == "__main__":
    main()
