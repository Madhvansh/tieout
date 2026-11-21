"""Retrieval sanity check: hand-picked questions whose page was found by searching page texts.

    python -m baseline.sanity

"page" is the primary statement or report page; "any" is the first retrieved page of the right
filing whose text contains the printed value (the same figure is often repeated in MD&A).
"""
from __future__ import annotations

from baseline import index

CASES = [
    ("What were Apple's total net sales in fiscal 2026?", "AAPL_FY2026_10K", 35, "478,834"),
    ("How much did NVIDIA spend on research and development in fiscal year 2026?", "NVDA_FY2026_10K", 139, "18,497"),
    ("Which firm is Coca-Cola's independent registered public accounting firm?", "KO_FY2025_10K", 124, "Ernst & Young"),
    ("What was JPMorgan Chase's total net revenue in 2025?", "JPM_FY2025_10K", 167, "182,447"),
    ("How much cash and cash equivalents did Tesla have at December 31, 2025?", "TSLA_FY2025_10K", 52, "16,513"),
    ("What was Microsoft's total revenue in fiscal year 2026?", "MSFT_FY2026_10K", 50, "331,839"),
]


def main(depth: int = 50) -> None:
    import time
    t0 = time.time()
    index.load()
    print(f"index loaded in {time.time() - t0:.1f} s")
    for q, doc, page, printed in CASES:
        hits = index.retrieve(q, depth)
        rank = next((i + 1 for i, h in enumerate(hits) if (h["doc"], h["page"]) == (doc, page)), None)
        anyr = next((i + 1 for i, h in enumerate(hits) if h["doc"] == doc and printed in h["text"]), None)
        top = f"{hits[0]['doc']} p{hits[0]['page']}" if hits else "-"
        print(f"page rank {str(rank or '>' + str(depth)):>4s}  any rank {str(anyr or '>' + str(depth)):>4s}  "
              f"{doc} p{page}  top1 {top}  | {q}")


if __name__ == "__main__":
    main()
