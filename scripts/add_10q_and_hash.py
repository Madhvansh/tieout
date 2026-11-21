"""One-off: append the 10-Q rows from the research notes to data/manifest.csv,
then record size and sha256 for every PDF already on disk."""
import csv
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
SRC = ROOT.parent / "research" / "06_10q_manifest.csv"


def main():
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    for extra in ("bytes", "sha256", "status"):
        if extra not in fields:
            fields.append(extra)
    have = {r["doc_id"] for r in rows}
    by_ticker = {r["ticker"]: r for r in rows if r["role"] == "latest"}
    if SRC.exists():
        for q in csv.DictReader(open(SRC, newline="", encoding="utf-8")):
            if not q.get("pdf_url") or q["ticker"] not in by_ticker:
                continue
            if q.get("verified", "").lower().startswith("partial") and q["ticker"] == "MSFT":
                continue                                    # could not be confirmed as a PDF
            base = by_ticker[q["ticker"]]
            doc = f"{q['ticker']}_{q['fiscal_year']}_Q3_10Q"
            if doc in have:
                continue
            rows.append({
                "doc_id": doc, "ticker": q["ticker"], "company": base["company"], "cik": base["cik"],
                "cluster": base["cluster"], "form": "10-Q", "fiscal_year": q["fiscal_year"],
                "period_end": q["period_end"], "pdf_url": q["pdf_url"],
                "source_domain": q["pdf_url"].split("/")[2], "role": "q3", "notes": "",
            })
            have.add(doc)
    for r in rows:
        p = ROOT / "data" / "pdf" / f"{r['doc_id']}.pdf"
        if p.exists():
            data = p.read_bytes()
            r["bytes"], r["sha256"], r["status"] = str(len(data)), hashlib.sha256(data).hexdigest(), "ok"
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows;", sum(1 for r in rows if r.get("status") == "ok"), "on disk")


if __name__ == "__main__":
    main()
