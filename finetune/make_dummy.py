"""Write a synthetic question -> plan dataset (DUMMY data, for pipeline proofing only).

Usage: python make_dummy.py [--n 600] [--val-frac 0.1] [--seed 435] [--out data]
Writes <out>/train.jsonl and <out>/val.jsonl, one {"question": ..., "plan": {...}} per line.
Key order inside each step follows system_prompt.txt and plan_schema.json.
"""
import argparse
import json
import os
import random

COMPANIES = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
    ("META", "Meta Platforms"), ("NVDA", "Nvidia"), ("TSLA", "Tesla"), ("BRK.B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("MA", "Mastercard"), ("JNJ", "Johnson & Johnson"),
    ("WMT", "Walmart"), ("PG", "Procter & Gamble"), ("XOM", "Exxon Mobil"), ("CVX", "Chevron"),
    ("HD", "Home Depot"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("COST", "Costco"),
    ("ORCL", "Oracle"), ("CRM", "Salesforce"), ("ADBE", "Adobe"), ("INTC", "Intel"),
    ("AMD", "AMD"), ("CSCO", "Cisco"), ("NFLX", "Netflix"), ("DIS", "Disney"),
    ("NKE", "Nike"), ("MCD", "McDonald's"), ("BAC", "Bank of America"), ("WFC", "Wells Fargo"),
    ("GS", "Goldman Sachs"), ("PFE", "Pfizer"), ("MRK", "Merck"), ("ABBV", "AbbVie"),
    ("UNH", "UnitedHealth"), ("LLY", "Eli Lilly"), ("IBM", "IBM"), ("QCOM", "Qualcomm"),
]

CONCEPTS = {
    "Revenue": ["revenue", "total revenue", "sales", "net sales", "top line"],
    "NetIncome": ["net income", "profit", "net earnings", "bottom line"],
    "OperatingIncome": ["operating income", "operating profit", "EBIT"],
    "GrossProfit": ["gross profit"],
    "EPSDiluted": ["diluted EPS", "earnings per share", "diluted earnings per share"],
    "TotalAssets": ["total assets", "assets"],
    "TotalLiabilities": ["total liabilities", "liabilities"],
    "StockholdersEquity": ["shareholders' equity", "stockholders' equity", "book equity"],
    "OperatingCashFlow": ["operating cash flow", "cash from operations"],
    "CapitalExpenditure": ["capex", "capital expenditure", "capital spending"],
    "ResearchAndDevelopment": ["R&D spending", "research and development expense", "R&D"],
    "LongTermDebt": ["long-term debt", "long term borrowings"],
}

ATTR_Q = {
    "sector": ["What sector is {c} in?", "Which industry sector does {c} belong to?"],
    "ceo": ["Who is the CEO of {c}?", "Who runs {c}?", "Name {c}'s chief executive."],
    "hq": ["Where is {c} headquartered?", "Where are {c}'s headquarters?"],
    "fiscal_year_end": ["When does {c}'s fiscal year end?", "What is the fiscal year end of {c}?"],
    "exchange": ["Which exchange is {c} listed on?", "Where does {c} stock trade?"],
}

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def pick_company(r):
    t, n = r.choice(COMPANIES)
    return t, r.choice([t, n, n])


def pick_concept(r):
    k = r.choice(list(CONCEPTS))
    return k, r.choice(CONCEPTS[k])


def pick_period(r):
    y = r.choice(YEARS)
    if r.random() < 0.25:
        q = r.randint(1, 4)
        return {"fy": y, "fq": q}, r.choice([f"Q{q} {y}", f"Q{q} FY{y}", f"the Q{q} {y} quarter"])
    return {"fy": y}, r.choice([f"FY{y}", f"fiscal {y}", f"{y}", f"fiscal year {y}"])


def get(i, t, k, p):
    return {"id": f"s{i}", "op": "get", "company": t, "concept": k, "period": p}


def g_get(r):
    t, cn = pick_company(r); k, kn = pick_concept(r); p, pn = pick_period(r)
    q = r.choice([f"What was {cn}'s {kn} in {pn}?", f"Show {cn} {kn} for {pn}.",
                  f"{kn[0].upper() + kn[1:]} of {cn} in {pn}?", f"How much {kn} did {cn} report in {pn}?"])
    return q, {"steps": [get(1, t, k, p)], "answer": "s1"}


def g_diff(r):
    (t1, c1), (t2, c2) = pick_company(r), pick_company(r)
    if t1 == t2:
        return None
    k, kn = pick_concept(r); p, pn = pick_period(r)
    op, q = r.choice([
        ("-", f"How much higher was {c1}'s {kn} than {c2}'s in {pn}?"),
        ("-", f"What is the gap between {c1} and {c2} {kn} in {pn}?"),
        ("+", f"What was the combined {kn} of {c1} and {c2} in {pn}?"),
        ("/", f"How many times larger was {c1}'s {kn} than {c2}'s in {pn}?"),
    ])
    steps = [get(1, t1, k, p), get(2, t2, k, p), {"id": "s3", "op": "calc", "expr": f"s1 {op} s2"}]
    return q, {"steps": steps, "answer": "s3"}


def g_ratio(r):
    t, cn = pick_company(r); p, pn = pick_period(r)
    num, den, name = r.choice([
        ("NetIncome", "Revenue", "net margin"), ("OperatingIncome", "Revenue", "operating margin"),
        ("GrossProfit", "Revenue", "gross margin"), ("TotalLiabilities", "StockholdersEquity", "debt-to-equity ratio"),
        ("ResearchAndDevelopment", "Revenue", "R&D intensity"), ("CapitalExpenditure", "OperatingCashFlow", "capex share of operating cash flow"),
    ])
    pct = "margin" in name or "intensity" in name or "share" in name
    q = r.choice([f"What was {cn}'s {name} in {pn}?", f"Compute the {name} for {cn} in {pn}."])
    expr = "s1 / s2 * 100" if pct else "s1 / s2"
    steps = [get(1, t, num, p), get(2, t, den, p), {"id": "s3", "op": "calc", "expr": expr}]
    return q, {"steps": steps, "answer": "s3"}


def g_change(r):
    t, cn = pick_company(r); k, kn = pick_concept(r)
    y1, y2 = sorted(r.sample(YEARS, 2))
    kind, q = r.choice([
        ("pct", f"By what percent did {cn}'s {kn} change from {y1} to {y2}?"),
        ("pct", f"What was {cn}'s {kn} growth rate between FY{y1} and FY{y2}?"),
        ("abs", f"How much did {cn}'s {kn} increase from {y1} to {y2}?"),
        ("abs", f"What is the change in {cn} {kn} between fiscal {y1} and fiscal {y2}?"),
    ])
    step = {"id": "s1", "op": "change", "company": t, "concept": k, "from": {"fy": y1}, "to": {"fy": y2}, "kind": kind}
    return q, {"steps": [step], "answer": "s1"}


def g_change_cmp(r):
    (t1, c1), (t2, c2) = pick_company(r), pick_company(r)
    if t1 == t2:
        return None
    k, kn = pick_concept(r); y1, y2 = sorted(r.sample(YEARS, 2))
    q = r.choice([f"How much faster did {c1}'s {kn} grow than {c2}'s from {y1} to {y2}, in percentage points?",
                  f"Compare {kn} growth of {c1} versus {c2} between {y1} and {y2}: what is the difference in percent growth?"])
    mk = lambda i, t: {"id": f"s{i}", "op": "change", "company": t, "concept": k, "from": {"fy": y1}, "to": {"fy": y2}, "kind": "pct"}
    return q, {"steps": [mk(1, t1), mk(2, t2), {"id": "s3", "op": "calc", "expr": "s1 - s2"}], "answer": "s3"}


def g_rank(r):
    n = r.randint(3, 6)
    picks = r.sample(COMPANIES, n)
    names = [r.choice([t, nm]) for t, nm in picks]
    k, kn = pick_concept(r); p, pn = pick_period(r)
    lst = ", ".join(names[:-1]) + " and " + names[-1]
    order, kk, q = r.choice([
        ("desc", 1, f"Which of {lst} had the highest {kn} in {pn}?"),
        ("asc", 1, f"Which of {lst} had the lowest {kn} in {pn}?"),
        ("desc", 2, f"Name the top 2 of {lst} by {kn} in {pn}."),
        ("desc", 3, f"Rank {lst} by {kn} in {pn} and give the top 3."),
        ("asc", 2, f"Which two of {lst} reported the smallest {kn} in {pn}?"),
    ])
    if kk > n:
        return None
    step = {"id": "s1", "op": "rank", "companies": [t for t, _ in picks], "concept": k, "period": p, "order": order, "k": kk}
    return q, {"steps": [step], "answer": "s1"}


def g_attr(r):
    t, cn = pick_company(r); f = r.choice(list(ATTR_Q))
    q = r.choice(ATTR_Q[f]).format(c=cn)
    return q, {"steps": [{"id": "s1", "op": "attr", "company": t, "field": f}], "answer": "s1"}


GENS = [(g_get, 3), (g_diff, 3), (g_ratio, 2), (g_change, 3), (g_change_cmp, 1), (g_rank, 2), (g_attr, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=435)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    r = random.Random(a.seed)
    fns = [f for f, w in GENS for _ in range(w)]
    seen, rows = set(), []
    while len(rows) < a.n:
        res = r.choice(fns)(r)
        if res is None or res[0] in seen:
            continue
        seen.add(res[0]); rows.append({"question": res[0], "plan": res[1]})
    r.shuffle(rows)
    nv = max(1, int(round(a.n * a.val_frac)))
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("val", rows[:nv]), ("train", rows[nv:])):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"{name}: {len(part)} rows -> {os.path.join(a.out, name + '.jsonl')}")


if __name__ == "__main__":
    main()
