"""How much of what the page prints exists as a raw row with the same figures.

Measured against the page text (the text blind authors read, `bench2/pdf_text/<DOC>/pNNN.txt`), not
against our own parser, so it can show what the build misses. A *data line* is a text line that ends in
two or more figures (not a line of years). It is

  full      one raw row on the same page holds every figure of the line,
  change    one row holds every figure except a trailing run of change columns (small integers or
            percents, "8 6", "7 % 5 %"), which the build drops on purpose,
  partial   the best row holds at least half of the figures (a lost or shifted cell),
  in_label  the figures sit inside a row's label or a table's title (a row glued to its label, a data
            row taken as a title),
  missing   no row holds half of them.

    python scripts/raw_coverage.py [--raw graph/raw.json.gz] [--text ../bench2/pdf_text] [--out FILE]
                                   [--worst 30] [--show N]
"""
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FIG = re.compile(r"^\(?[-+]?\$?\d[\d,]*(?:\.\d+)?\)?%?\)?$")
FILLER = {"$", "%", "*", "†"}
DASHES = {"—", "–", "-", "$—", "$-", "$–", "—%", "-%"}
MARKER = re.compile(r"^\((?:[a-z]{1,2}|\d)\)$")
YEAR = re.compile(r"^(?:19[89]\d|20[0-3]\d)$")
LABEL_YEAR = re.compile(r"^(?:19[89]\d|20[0-7]\d)$")


def norm(tok):
    """'(1,234.5)' -> '1234.5', any dash -> '—'; the sign and the decoration are not compared."""
    if tok.strip("$%") in ("—", "–", "-", ""):
        return "—"
    return tok.strip("$()%+,").replace(",", "").lstrip("-")


def tail(line):
    """Figures at the end of a line, left to right, and the text before them. A dash is a printed nil."""
    toks = line.split()
    figs, i = [], len(toks) - 1
    while i >= 0:
        t = toks[i]
        if t in FILLER:
            i -= 1
            continue
        if t in DASHES:
            figs.append((t, False))
            i -= 1
            continue
        # "4)" closes a parenthetical of the label ("(Note 4)"), it is not a value
        if FIG.match(t) and any(ch.isdigit() for ch in t) and not (t.endswith(")") and not t.startswith("(")):
            figs.append((t, i + 1 < len(toks) and toks[i + 1].startswith("%") or t.endswith("%")))
            i -= 1
            continue
        break
    figs.reverse()
    label = " ".join(toks[: i + 1])
    # a footnote marker "(1)" between the label and the figures is not a value
    while len(figs) > 2 and MARKER.match(figs[0][0]):
        figs = figs[1:]
    # a year that opens the run belongs to the label ("notes due 2031 1,991 1,989", "2022 1,223 3,284")
    while len(figs) > 2 and LABEL_YEAR.match(norm(figs[0][0])) and figs[0][0] == norm(figs[0][0]):
        label += " " + figs[0][0]
        figs = figs[1:]
    if all(MARKER.match(t) or norm(t) == "—" for t, _ in figs):
        figs = []
    return figs, label


KINDS = ("full", "change", "partial", "in_label", "missing")
MONTHS = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b\.?", re.I)


def is_header(figs, line=""):
    """A line of years, or of dates ("At December 31, 2025 At December 31, 2024")."""
    years = [YEAR.match(norm(t)) and "," not in t.rstrip(",") and "." not in t for t, _ in figs]
    if all(years):
        return True
    days = [not y and norm(t).isdigit() and int(norm(t)) <= 31 for (t, _), y in zip(figs, years)]
    return bool(MONTHS.search(line)) and all(y or d for y, d in zip(years, days))


def big(tok):
    v = norm(tok)
    return "," in tok or (v.replace(".", "").isdigit() and float(v or 0) >= 100)


def rate_like(tok, pct):
    """A figure the build may drop on purpose: a percent, a change or a ratio column."""
    v = norm(tok)
    try:
        x = float(v)
    except ValueError:
        return False
    return pct or ("." in v and x < 100 and "," not in tok)


def load_rows(path):
    rows = json.load(gzip.open(path, "rt", encoding="utf-8"))
    by_page = defaultdict(list)
    for r in rows:
        r["_have"] = Counter(norm(c.get("text") or "") for c in r["cells"])
        by_page[(r["doc"], r["page"])].append(r)
    return rows, by_page


def best_row(figs, rows):
    want = Counter(norm(t) for t, _ in figs)
    best, best_n = None, 0
    for r in rows:
        have = r["_have"]
        n = sum(min(k, have[f]) for f, k in want.items() if f in have)
        if n > best_n:
            best, best_n = r, n
    return best, best_n


def classify(figs, rows):
    n = len(figs)
    row, hit = best_row(figs, rows)
    if hit == n:
        return "full", row, hit
    if row is not None and hit >= 1:
        have = Counter(row["_have"])
        missing, matched = [], []
        for j, (t, pct) in enumerate(figs):
            if have[norm(t)] > 0:
                have[norm(t)] -= 1
                matched.append(j)
            else:
                missing.append(j)
        # the row lacks only percent, ratio or change columns (dropped by design) while it holds the amounts;
        # small integers count as change columns only when they trail the line ("8 6" after the amounts)
        big_hit = sum(1 for j in matched if big(figs[j][0]))
        trail = len(figs)
        while trail > 0 and (trail - 1) in missing:
            trail -= 1
        def tolerated(j):
            v = norm(figs[j][0])
            return (rate_like(*figs[j]) or v == "—"
                    or (j >= trail and v.isdigit() and len(v) <= 2)       # trailing change column "8 6"
                    or (j == 0 and v.isdigit() and len(v) <= 2 and len(figs) >= 3))  # "Level 2", "1 to 4"
        if all(tolerated(j) for j in missing) and (big_hit >= 1 or hit >= 2):
            return "change", row, hit
    if hit * 2 >= n:
        return "partial", row, hit
    nums = {norm(t) for t, _ in figs}
    for r in rows:
        text = (r.get("label") or "") + " " + (r.get("title") or "")
        found = {norm(t) for t in text.split() if FIG.match(t) and any(ch.isdigit() for ch in t)}
        if len(nums & found) >= max(2, (len(nums) + 1) // 2):
            return "in_label", r, hit
    return "missing", row, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="graph/raw.json.gz")
    ap.add_argument("--text", default="../bench2/pdf_text")
    ap.add_argument("--out", default="")
    ap.add_argument("--worst", type=int, default=30)
    ap.add_argument("--show", type=int, default=0, help="print N bad lines per worst page")
    a = ap.parse_args()
    _, by_page = load_rows(a.raw)
    per_doc = defaultdict(Counter)
    pages = []
    for ddir in sorted(Path(a.text).iterdir()):
        doc = ddir.name
        for pf in sorted(ddir.glob("p*.txt")):
            page = int(pf.stem[1:])
            rows = by_page.get((doc, page), [])
            cnt, bad = Counter(), []
            for line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
                figs, label = tail(line)
                if len(figs) < 2:
                    continue
                if is_header(figs, line):
                    cnt["header"] += 1
                    continue
                kind, row, hit = classify(figs, rows)
                cnt[kind] += 1
                if kind in ("full", "change") and not any(c.get("pid") for c in row["cells"]):
                    cnt["undated"] += 1
                if kind not in ("full", "change"):
                    cnt["no_rows_on_page" if not rows else "rows_on_page"] += 1
                    bad.append({"kind": kind, "line": line.strip()[:160], "figs": len(figs), "hit": hit,
                                "row": row["id"] if row is not None else None,
                                "row_label": (row["label"][:60] if row is not None else None)})
            per_doc[doc].update(cnt)
            data = sum(cnt[k] for k in KINDS)
            if data:
                pages.append({"doc": doc, "page": page, "data_lines": data, "ok": cnt["full"] + cnt["change"],
                              "counts": dict(cnt), "rows_on_page": len(rows), "bad": bad})
    tot = Counter()
    for c in per_doc.values():
        tot.update(c)
    data = sum(tot[k] for k in KINDS)
    ok = tot["full"] + tot["change"]
    print(f"data lines {data}: full {tot['full']}, change-columns only {tot['change']}, partial {tot['partial']}, "
          f"in label/title {tot['in_label']}, missing {tot['missing']} | covered {ok / max(data, 1):.1%} "
          f"| header lines {tot['header']}")
    print(f"not covered: on pages with no raw row {tot['no_rows_on_page']}, on pages with rows {tot['rows_on_page']}; "
          f"covered lines whose row has no period on any cell: {tot['undated']} of {ok}")
    docs = []
    for doc, c in sorted(per_doc.items()):
        d = sum(c[k] for k in KINDS)
        docs.append({"doc": doc, "data_lines": d, "covered": (c["full"] + c["change"]) / max(d, 1), "counts": dict(c)})
    print("\nworst filings by share covered:")
    for d in sorted(docs, key=lambda d: d["covered"])[:12]:
        print(f"  {d['doc']:22s} {d['covered']:.1%} of {d['data_lines']}  {d['counts']}")
    worst = sorted(pages, key=lambda p: (-(p["data_lines"] - p["ok"]), p["ok"] / p["data_lines"]))[: a.worst]
    print(f"\n{a.worst} worst pages (lines not covered):")
    for p in worst:
        print(f"  {p['doc']:22s} p{p['page']:<4d} {p['data_lines'] - p['ok']:3d} of {p['data_lines']:3d} not covered "
              f"| rows on page {p['rows_on_page']:3d} | {p['counts']}")
        for b in p["bad"][: a.show]:
            print(f"        {b['kind']:8s} {b['hit']}/{b['figs']}  {b['line'][:110]}  <- {b['row_label']}")
    undated = sorted((p for p in pages if p["counts"].get("undated")), key=lambda p: -p["counts"]["undated"])
    print(f"\npages whose covered rows carry no period (header lines on page in brackets), top {a.worst // 2}:")
    for p in undated[: a.worst // 2]:
        print(f"  {p['doc']:22s} p{p['page']:<4d} {p['counts']['undated']:3d} undated of {p['ok']:3d} covered "
              f"[{p['counts'].get('header', 0)} header lines]")
    if a.out:
        Path(a.out).write_text(json.dumps({"totals": dict(tot), "data_lines": data, "covered": ok / max(data, 1),
                                           "docs": docs, "worst": worst,
                                           "undated_pages": [{k: p[k] for k in ("doc", "page", "ok", "counts")}
                                                             for p in undated[:200]]}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
