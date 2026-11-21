"""One-off: build data/manifest.csv from the research notes (latest 10-Ks plus prior-year 10-Ks)."""
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "research" / "05_corpus_manifest.csv"
OUT = ROOT / "data" / "manifest.csv"

PRIOR = [
    ("AAPL", "FY2025", "2025-09-27", "https://s2.q4cdn.com/470004039/files/doc_earnings/2025/q4/filing/10K-Q4-2025-as-filed.pdf"),
    ("MSFT", "FY2025", "2025-06-30", "https://microsoft.gcs-web.com/static-files/9640e86d-a99f-4502-934d-1d3c883fa786"),
    ("GOOGL", "FY2024", "2024-12-31", "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/goog-10-k-2024.pdf"),
    ("BAC", "FY2024", "2024-12-31", "https://investor.bankofamerica.com/regulatory-and-other-filings/annual-reports/content/0000070858-25-000139/0000070858-25-000139.pdf"),
    ("GS", "FY2024", "2024-12-31", "https://www.goldmansachs.com/investor-relations/financials/10k/2024/2024-10-k.pdf"),
    ("WMT", "FY2025", "2025-01-31", "https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-25-000021/0000104169-25-000021.pdf"),
    ("KO", "FY2024", "2024-12-31", "https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-25-000011/0000021344-25-000011.pdf"),
    ("JNJ", "FY2024", "2024-12-29", "https://s203.q4cdn.com/636242992/files/doc_financials/2024/q4/Form-10-K-2024-Final.pdf"),
    ("XOM", "FY2024", "2024-12-31", "https://investor.exxonmobil.com/sec-filings/annual-reports/content/0000034088-25-000010/0000034088-25-000010.pdf"),
    ("CAT", "FY2024", "2024-12-31", "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000018230/d10913c3-95ae-450c-9432-f55a05fa18c0.pdf"),
]


def main():
    rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
    by_ticker = {r["ticker"]: r for r in rows}
    out = []
    for r in rows:
        out.append({
            "doc_id": f"{r['ticker']}_{r['fiscal_year']}_10K", "ticker": r["ticker"], "company": r["company"],
            "cik": r.get("cik", ""), "cluster": r["cluster"], "form": "10-K",
            "fiscal_year": r["fiscal_year"], "period_end": r["fiscal_year_end_date"],
            "pdf_url": r["pdf_url"], "source_domain": r["source_domain"], "role": "latest",
            "notes": r.get("notes", "")[:200],
        })
    for t, fy, end, url in PRIOR:
        b = by_ticker[t]
        out.append({
            "doc_id": f"{t}_{fy}_10K", "ticker": t, "company": b["company"], "cik": b.get("cik", ""),
            "cluster": b["cluster"], "form": "10-K", "fiscal_year": fy, "period_end": end,
            "pdf_url": url, "source_domain": url.split("/")[2], "role": "prior_year", "notes": "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows ->", OUT)


if __name__ == "__main__":
    main()
