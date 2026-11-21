"""Dev check: run a handful of hand-written plans end to end and print answers with their proofs."""
import json

from tieout.execute import Executor, render
from tieout.graph import Graph

PLANS = {
    "Apple revenue FY2026": {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026}}], "answer": "s1"},
    "Apple gross margin FY2026": {"steps": [{"id": "s1", "op": "metric", "company": "AAPL", "metric": "gross_margin", "period": {"fy": 2026}}], "answer": "s1"},
    "Amazon gross margin FY2025 (no gross profit line)": {"steps": [{"id": "s1", "op": "metric", "company": "AMZN", "metric": "gross_margin", "period": {"fy": 2025}}], "answer": "s1"},
    "Revenue growth Nvidia latest": {"steps": [{"id": "s1", "op": "change", "company": "NVDA", "concept": "Revenue", "from": {"fy": "prior"}, "to": {"fy": "latest"}, "mode": "pct"}], "answer": "s1"},
    "Top 3 by net income, latest FY, all": {"steps": [{"id": "s1", "op": "rank", "over": "all", "by": {"concept": "NetIncome"}, "period": {"fy": "latest"}, "k": 3}], "answer": "s1"},
    "Highest operating margin among big tech": {"steps": [{"id": "s1", "op": "rank", "over": {"cluster": "big_tech"}, "by": {"metric": "operating_margin"}, "period": {"fy": "latest"}, "k": 3}], "answer": "s1"},
    "JPM vs BAC total assets FY2025": {"steps": [{"id": "s1", "op": "get", "company": "JPM", "concept": "TotalAssets", "period": {"fy": 2025}}, {"id": "s2", "op": "get", "company": "BAC", "concept": "TotalAssets", "period": {"fy": 2025}}, {"id": "s3", "op": "calc", "expr": "s1 - s2"}], "answer": "s3"},
    "Apple Q4 FY2026 revenue (not printed)": {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026, "q": 4}}], "answer": "s1"},
    "Alphabet Q4 FY2025 revenue (not printed)": {"steps": [{"id": "s1", "op": "get", "company": "GOOGL", "concept": "Revenue", "period": {"fy": 2025, "q": 4}}], "answer": "s1"},
    "Type error: revenue plus margin": {"steps": [{"id": "s1", "op": "get", "company": "AAPL", "concept": "Revenue", "period": {"fy": 2026}}, {"id": "s2", "op": "metric", "company": "AAPL", "metric": "net_margin", "period": {"fy": 2026}}, {"id": "s3", "op": "calc", "expr": "s1 + s2"}], "answer": "s3"},
    "Missing: bank gross profit": {"steps": [{"id": "s1", "op": "get", "company": "JPM", "concept": "GrossProfit", "period": {"fy": 2025}}], "answer": "s1"},
}


def main():
    g = Graph.load()
    ex = Executor(g)
    for name, plan in PLANS.items():
        r = ex.run(plan)
        print(f"\n# {name}\n  -> {render(r)}")
        for lf in r.leaves[:4]:
            print(f"     leaf {lf.ticker} {lf.what} {lf.period} = {lf.printed!r} [{lf.status}] {lf.doc} p{lf.page} '{lf.label[:40]}'")
        if len(r.leaves) > 4:
            print(f"     ... {len(r.leaves)} leaves in all")
        for n in r.notes[:3]:
            print("     note:", n)


if __name__ == "__main__":
    main()
