"""Give every table of the raw layer the lines printed just above it.

Two tables of one filing can have the same row labels: "Total revenues net of interest expense"
is a row of every segment's table. What tells them apart is printed above the table, not in it:
the segment's name, the table's caption, the units note. This pass reads the page's words (the
same word boxes the build used; no PDF is opened), takes the few text lines above each table's
first row, and stores them on the table's rows as `above`. The linker shows that text beside a
candidate row and matches a question's words against it.

It also names the columns. A segment table prints its segments across the page, and the header
words sit above the numbers they belong to: each header word in the few lines above a table is
given to the column whose numbers it stands over, and the name is stored on that column's cells
as `head`. With it, a row of such a table is a breakdown by column, not six anonymous figures.

    python scripts/add_headings.py            # rewrites graph/raw.json.gz in place

Run after scripts/build_raw.py. Needs data/words/<doc>.json.gz (made by `python -m tieout build`).
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tieout.rawlayer import RAW_PATH, load_raw, save_raw            # noqa: E402

WORDS = ROOT / "data" / "words"
LINES_ABOVE = 5          # caption, heading, units note and the column header lines
REACH = 130.0            # points above the first row that still belong to the table
MAX_CHARS = 260
HEAD_LINES = 4           # header rows directly above the first data row
HEAD_CHARS = 60


def page_lines(words: list) -> list:
    """[(top, bottom, text)] for the text lines of a page, top to bottom."""
    lines: dict = defaultdict(list)
    for x0, top, _x1, bottom, text in words:
        lines[round((top + bottom) / 2 / 3)].append((x0, top, bottom, text))      # 3-point bands
    out = []
    for band in sorted(lines):
        parts = sorted(lines[band])
        out.append((min(p[1] for p in parts), max(p[2] for p in parts), " ".join(p[3] for p in parts)))
    return out


def word_lines(words: list) -> list:
    """[(top, bottom, [(x0, x1, text), ...])] for the text lines of a page, top to bottom."""
    lines: dict = defaultdict(list)
    for x0, top, x1, bottom, text in words:
        lines[round((top + bottom) / 2 / 3)].append((x0, x1, top, bottom, text))
    out = []
    for band in sorted(lines):
        parts = sorted(lines[band])
        out.append((min(q[2] for q in parts), max(q[3] for q in parts), [(q[0], q[1], q[4]) for q in parts]))
    return out


def column_heads(lines: list, table_rows: list, top: float, others: list) -> dict:
    """{column index: the header words printed above that column's numbers}.

    Numbers are set flush right, so a column owns the strip from the previous column's right edge
    to its own. A header word in the lines just above the table goes to the strip its centre falls
    in; words over the label column, and figures, are left out."""
    rights: dict = defaultdict(list)
    for r in table_rows:
        for c in r["cells"]:
            if len(c.get("bbox") or []) == 4 and c.get("ci") is not None:
                rights[c["ci"]].append(c["bbox"][2])
    if len(rights) < 2:
        return {}
    edges = sorted((sorted(xs)[len(xs) // 2], ci) for ci, xs in rights.items())      # (right edge, column)
    first_left = min(c["bbox"][0] for r in table_rows for c in r["cells"] if len(c.get("bbox") or []) == 4)
    strips = [(edges[k - 1][0] + 1 if k else first_left - 45, edge + 8, ci) for k, (edge, ci) in enumerate(edges)]
    header = [ln for ln in lines if ln[1] <= top + 1
              and not any(o_top - 1 <= ln[0] and ln[1] <= o_bottom + 1 for o_top, o_bottom in others)][-HEAD_LINES:]
    names: dict = defaultdict(list)
    for _t, _b, parts in header:
        wording = [text for _x0, _x1, text in parts if any(ch.isalpha() for ch in text)]
        over_labels = sum(1 for x0, x1, text in parts if (x0 + x1) / 2 <= strips[0][0] and any(ch.isalpha() for ch in text))
        if over_labels > 6 or (len(wording) >= 8 and over_labels >= 3) or (len(wording) >= 6 and wording[-1][-1] in ":."):
            continue                                  # a sentence or a caption, not a header row
        for x0, x1, text in parts:
            if not any(ch.isalpha() for ch in text):
                continue
            centre = (x0 + x1) / 2
            for left, right, ci in strips:
                if left < centre <= right:
                    names[ci].append(text)
                    break
    return {ci: " ".join(ws)[:HEAD_CHARS] for ci, ws in names.items()}


def table_tops(rows: list) -> dict:
    """{table_id: (page, top of its first row, bottom of its last row)} from the cells' boxes."""
    box: dict = {}
    for r in rows:
        ys = [(c["bbox"][1], c["bbox"][3]) for c in r["cells"] if len(c.get("bbox") or []) == 4]
        if not ys:
            continue
        top, bottom = min(y[0] for y in ys), max(y[1] for y in ys)
        page, t0, b0 = box.get(r["table_id"], (r["page"], top, bottom))
        box[r["table_id"]] = (page, min(t0, top), max(b0, bottom))
    return box


CONTINUES_WITHIN = 70.0  # points between a table and an undated fragment below it
ALIGNED = 6.0            # points two columns' right edges may differ and still be one column


def column_edges(table_rows: list) -> dict:
    """{column index: the median right edge of its numbers} for one table."""
    rights: dict = defaultdict(list)
    for r in table_rows:
        for c in r["cells"]:
            if len(c.get("bbox") or []) == 4 and c.get("ci") is not None:
                rights[c["ci"]].append(c["bbox"][2])
    return {ci: sorted(xs)[len(xs) // 2] for ci, xs in rights.items()}


def continue_tables(tables: list, rows_of: dict) -> int:
    """Date the undated fragments of tables that the builder cut in two.

    A long table is sometimes split where a row could not be read; the lower piece then has no
    header and its columns no period. If it starts just below a dated table of the same page and
    its columns end at the same x positions, it is that table continued: each of its columns takes
    the period of the column above it. `tables` is [(table id, top, bottom)] of one page.
    Returns the number of cells dated."""
    dated = 0
    for tid, top, _bottom in sorted(tables, key=lambda t: t[1]):
        mine = rows_of[tid]
        if any(c.get("pid") for r in mine for c in r["cells"]):
            continue
        upper = [(b, other) for other, _t, b in tables if other != tid and b <= top + 1 and top - b <= CONTINUES_WITHIN
                 and any(c.get("pid") for r in rows_of[other] for c in r["cells"])]
        if not upper:
            continue
        source = rows_of[max(upper)[1]]
        periods: dict = defaultdict(list)             # column of the table above -> the periods its cells carry
        for r in source:
            for c in r["cells"]:
                if c.get("pid") and c.get("ci") is not None:
                    periods[c["ci"]].append((c["pid"], c.get("end")))
        above_edges = {ci: x for ci, x in column_edges(source).items() if periods.get(ci)}
        my_edges = column_edges(mine)
        match = {}
        for ci, x in my_edges.items():
            near = [(abs(x - ux), uci) for uci, ux in above_edges.items() if abs(x - ux) <= ALIGNED]
            if near:
                match[ci] = max(set(periods[min(near)[1]]), key=periods[min(near)[1]].count)
        if len(match) < len(my_edges) or len({pid for pid, _ in match.values()}) < len(match):
            continue                                  # every column must sit under a column of its own period
        for r in mine:
            for c in r["cells"]:
                if c.get("ci") in match:
                    c["pid"], end = match[c["ci"]]
                    c["pid_from"] = "the table it continues"
                    if end:
                        c["end"] = end
                    dated += 1
    return dated


def above(lines: list, top: float, others: list) -> str:
    """The last few text lines that end above `top` and are not rows of another table."""
    kept = [text for t, b, text in lines
            if top - REACH <= t and b <= top + 1 and not any(o_top - 1 <= t and b <= o_bottom + 1 for o_top, o_bottom in others)]
    text = " | ".join(words_only(line) for line in kept[-LINES_ABOVE:])
    return " | ".join(part for part in text.split(" | ") if part)[-MAX_CHARS:]


def words_only(line: str) -> str:
    """A line without its figures: words and years stay, amounts go (no value is ever shown to a model)."""
    keep = []
    for token in line.split():
        bare = token.strip("()$%,.;:")
        if any(ch.isalpha() for ch in token) or (len(bare) == 4 and bare.isdigit() and bare[:2] in ("19", "20")):
            keep.append(token)
    return " ".join(keep)


def main() -> None:
    rows = load_raw()
    by_doc: dict = defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
        for c in r["cells"]:                          # a rerun starts from the builder's own dates
            if c.pop("pid_from", None):
                c["pid"] = None
    done = continued = 0
    for doc, doc_rows in sorted(by_doc.items()):
        path = WORDS / f"{doc}.json.gz"
        if not path.exists():
            print("no word boxes for", doc)
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            pages = json.load(f)
        boxes = table_tops(doc_rows)
        title_of = {r["table_id"]: r["title"] for r in doc_rows if r["title"]}
        on_page: dict = defaultdict(list)
        for tid, (page, top, bottom) in boxes.items():
            on_page[page].append((tid, top, bottom))
        heading: dict = {}
        heads: dict = {}
        own: dict = {}
        rows_of: dict = defaultdict(list)
        for r in doc_rows:
            rows_of[r["table_id"]].append(r)
        for page, tables in on_page.items():
            continued += continue_tables(tables, rows_of)
            if not 1 <= page <= len(pages):
                continue
            lines = page_lines(pages[page - 1]["words"])
            wlines = word_lines(pages[page - 1]["words"])
            first_of: dict = {}                       # title -> heading of the topmost table with that title
            first_heads: dict = {}
            for tid, top, _bottom in sorted(tables, key=lambda t: t[1]):
                others = [(t, b) for other, t, b in tables if other != tid and b <= top + 1]
                # blocks of one table stacked on a page (one per year) share the caption of the first;
                # what is printed directly over a later block (its year) is kept beside it as `own`
                mine = above(lines, top, others)
                heading[tid] = first_of.setdefault(title_of.get(tid, tid), mine)
                if mine != heading[tid]:
                    own[tid] = mine
                heads[tid] = first_heads.setdefault(title_of.get(tid, tid), column_heads(wlines, rows_of[tid], top, others))
        for r in doc_rows:
            if r.get("context") != "primary financial statement":
                r["above"] = heading.get(r["table_id"], "")
                r.pop("own", None)
                if own.get(r["table_id"]):
                    r["own"] = own[r["table_id"]]
                done += bool(r["above"])
                for c in r["cells"]:
                    name = heads.get(r["table_id"], {}).get(c.get("ci"))
                    if name:
                        c["head"] = name
                    else:
                        c.pop("head", None)
    tmp = RAW_PATH.with_suffix(".tmp")
    save_raw(rows, tmp)
    tmp.replace(RAW_PATH)                             # readers never see a half-written file
    print(f"{done} of {len(rows)} rows now carry the lines printed above their table; "
          f"{continued} cells of table fragments dated from the table they continue")


if __name__ == "__main__":
    main()
