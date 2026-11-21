"""Dev helper: show every line in a filing that looks like a statement title, and why it was kept or dropped."""
import re
import sys

from tieout import pdfio
from tieout.statements import KINDS, NOT_PRIMARY, OTHER_TITLE, page_lines, split_regions

doc = sys.argv[1]
want = re.compile(sys.argv[2], re.I) if len(sys.argv) > 2 else re.compile(
    r"results of operations|financial position|cash flow|statements? of (earnings|income|operations)|balance sheet", re.I)
pages = pdfio.load_pages(doc)
for i, page in enumerate(pages):
    lines = page_lines(page)
    regions = split_regions(lines)
    for ln in lines:
        text = " ".join(w.text for w in ln)
        if want.search(text) and len(text) < 160:
            kinds = [k for k, p in KINDS.items() if p.search(text)]
            print(f"p{i + 1:<4d} len={len(text):<4d} kinds={kinds} notprim={bool(NOT_PRIMARY.search(text))} "
                  f"other={bool(OTHER_TITLE.search(text))} regions={[(k, len(r)) for k, _, r in regions]} | {text[:110]}")
