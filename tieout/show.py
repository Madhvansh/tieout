"""Draw a statement page with the formulas the graph recovered on it.

A 10-K is a spreadsheet whose formulas were erased, and the graph puts them back:
tieout/calc.py finds each subtotal as a signed sum of rows above it that holds in
every period column. This module draws that structure on the page itself.

    bracket   one per formula and column, just right of the column: a tick at each
              operand, a dot at the total, a minus beside an operand that is
              subtracted. Green where the formula holds exactly in that column,
              amber where it holds only within rounding, red where it fails. A
              formula that adds up other formulas' totals sits one lane further
              out than they do (its nesting depth), so brackets never overlap.
    tie       a dashed connector between two cells that print the same value
    cell      filled by the reading's state: verified green, unverified grey,
              failed red
    answer    its source cells outlined in blue with numbered badges, and a
              caption with the question, the answer and its formula; every number
              in the caption is read from the graph

Geometry is in PDF points with the origin at the top left of the page, the frame
of the graph's bounding boxes. pdfplumber renders the page (through pypdfium2) and
matplotlib draws over it, so a PDF output keeps the drawing as vectors.

    python scripts/show.py --doc DOC --page N --out page.png
    python scripts/show.py --reading ID [ID ...] --out cells.png
    python scripts/show.py --plan PLAN --question "..." --out answer.pdf
    python scripts/show.py --question "..." --out answer.png      (cached plans only)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import matplotlib
import pdfplumber
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath

from .execute import Executor, Leaf, Result, render
from .graph import Check, Graph, Reading

PDF_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "pdf"
Box = tuple            # (x0, top, x1, bottom) in PDF points, origin at the top left of the page
Run = tuple            # (style, text) of a piece of caption; style: "", "bold", "muted" or "badge"
Paths = Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]]

PASS, ROUNDED, FAIL = "#15934a", "#e09a00", "#d93025"
BLUE, INK, MUTED = "#1a56db", "#1b1b1b", "#5f6368"
STATE = {"pass": PASS, "rounded": ROUNDED, "fail": FAIL}
FILL = {"verified": "#9fdcb1", "unverified": "#c9c9c9", "failed": "#f3a49b"}
FILL_ALPHA = 0.5

# On the page, in PDF points: these scale with the page.
PAD = 1.6                      # from a column's numbers to its innermost bracket
MAX_STEP, MIN_STEP = 3.6, 1.6  # between neighbouring bracket lanes
END = 2.4                      # kept free before whatever is printed right of a column
HEADING_GAP = 20.0             # a wider blank above the statement ends its heading block
HEADING_MAX = 130.0            # the heading block is never taller than this
MARGIN = 5.0                   # around the region shown
EPS = 0.5                      # brackets that only touch at an end do not overlap

# On paper, in typographic points (and inches): these do not scale with the page.
LINE, DOT, MINUS_SIZE = 0.6, 2.2, 5.0
FONT, SMALL, BADGE = 6.5, 5.5, 5.0
LEADING = 1.38                 # line height as a multiple of the font size
FIGURE_MARGIN, PANEL_GAP, CAPTION_GAP = 0.04, 0.14, 0.05
MAX_PANELS, MAX_SOURCES = 3, 6


# ------------------------------------------------------------------ geometry (PDF points)

def middle(box: Box) -> float:
    """The vertical centre of a box."""
    return (box[1] + box[3]) / 2


def union(boxes: Sequence[Box]) -> Optional[Box]:
    """The smallest box holding all the given boxes, or None for none."""
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def lanes(spans: Sequence[tuple], uses: Sequence[Sequence[int]]) -> list:
    """The lane of each bracket: 0 sits next to the column, higher lanes further right.

    `spans[i]` is bracket i's vertical extent (top, bottom); `uses[i]` lists the
    brackets whose totals bracket i adds up. A bracket goes one lane outside the
    outermost bracket it uses (its nesting depth), then further out while it would
    overlap a bracket already placed in that lane; touching at an end is allowed.
    Brackets are placed in the order given, so each must come after those it uses.
    """
    lane: list = [0] * len(spans)
    placed: list = []
    for i, (top, bottom) in enumerate(spans):
        k = max((lane[j] + 1 for j in uses[i]), default=0)
        while any(other == k and top < b - EPS and t < bottom - EPS for t, b, other in placed):
            k += 1
        lane[i] = k
        placed.append((top, bottom, k))
    return lane


def lane_step(room: float, n_lanes: int) -> float:
    """Spacing between lanes so that `n_lanes` lanes fit in the `room` points right of a column."""
    if n_lanes <= 1:
        return MAX_STEP
    return max(MIN_STEP, min(MAX_STEP, (room - PAD - END) / (n_lanes - 1)))


def bracket_x(column_right: float, lane: int, step: float) -> float:
    """Where the vertical line of a bracket in `lane` stands."""
    return column_right + PAD + lane * step


def room_right(column_right: float, words: Sequence[Box], top: float, bottom: float, page_right: float) -> float:
    """Blank points right of a column, up to the nearest word printed between `top` and `bottom`."""
    gaps = [w[0] - column_right for w in words if w[0] > column_right + EPS and w[1] < bottom and w[3] > top]
    return min(gaps, default=page_right - column_right)


def heading_top(top: float, words: Sequence[Box]) -> float:
    """Walk up from `top` through the lines printed just above it (column dates, the title) and
    stop at the first blank taller than HEADING_GAP or after HEADING_MAX points."""
    start = top
    for w in sorted((w for w in words if w[3] <= top + EPS), key=lambda w: -w[3]):
        if top - w[3] > HEADING_GAP or start - w[1] > HEADING_MAX:
            break
        top = min(top, w[1])
    return top


def crop_region(core: Box, words: Sequence[Box], page_box: Box) -> Box:
    """The part of the page to show: the drawn region, widened to everything printed on its rows
    (the row labels on the left) and raised through the heading block just above it."""
    x0, top, x1, bottom = core
    rows = [w for w in words if w[1] < bottom and w[3] > top]
    x0 = min([x0] + [w[0] for w in rows])
    x1 = max([x1] + [w[2] for w in rows])
    top = heading_top(top, words)
    return (max(page_box[0], x0 - MARGIN), max(page_box[1], top - MARGIN),
            min(page_box[2], x1 + MARGIN), min(page_box[3], bottom + MARGIN))


def layout(sizes: Sequence[tuple], width_in: Optional[float]) -> tuple:
    """(inches per PDF point, size in inches of each panel) for panels placed side by side.

    With no width the pages keep their printed size; with one, every panel is scaled
    by the same factor so that the row of panels fills the width between the margins.
    """
    if width_in is None:
        scale = 1 / 72
    else:
        free = width_in - 2 * FIGURE_MARGIN - PANEL_GAP * (len(sizes) - 1)
        scale = free / sum(w for w, _ in sizes)
    return scale, [(w * scale, h * scale) for w, h in sizes]


# ------------------------------------------------------------------ what the graph says about a page

@dataclass
class Bracket:
    """One formula in one column: a vertical line with a tick to each member cell."""
    x: float
    operands: list               # (right edge of the cell, y, sign)
    total: Optional[tuple]       # (right edge of the cell, y); None when the total is on another page
    state: str                   # pass | rounded | fail


@dataclass
class TieLine:
    """Two cells that print the same value, joined by a dashed line through x."""
    x: float
    upper: tuple                 # (right edge of the cell, y)
    lower: tuple


@dataclass
class Marks:
    """Everything drawn on one page, and the region it covers."""
    cells: list = field(default_factory=list)       # (box, reading state)
    brackets: list = field(default_factory=list)
    ties: list = field(default_factory=list)
    extent: Optional[Box] = None
    formulas: int = 0


def statement_readings(g: Graph, doc: str, page: int) -> list:
    """Readings of the primary statements printed on this page."""
    return [r for r in g.readings.values() if r.doc == doc and r.page == page and r.table in g.tables and r.bbox]


def checks_on_page(g: Graph, kind: str, here: dict) -> list:
    """(check, [(reading, role)]) for each check of this type with two or more members on the page."""
    out = []
    for c in g.checks.values():
        if c.type == kind:
            members = [(here[rid], role) for rid, role in c.members if rid in here]
            if len(members) >= 2:
                out.append((c, members))
    return out


def column_states(g: Graph, check: Check) -> dict:
    """The state of a SUM check in each of its columns.

    The graph keeps one state per formula. In a column where the total equals the
    signed sum to the unit the formula holds exactly; a "rounded" formula is off by
    rounding in its other columns; a failing formula misses in exactly one column
    (tieout/calc.py), the one with the largest residual.
    """
    residual: dict = defaultdict(float)
    for rid, role in check.members:
        r = g.readings[rid]
        residual[r.col] += r.value if role == "total" else -float(role) * r.value
    if check.status == "pass":
        return {c: "pass" for c in residual}
    worst = max(residual, key=lambda c: abs(residual[c]))
    return {c: "fail" if check.status == "fail" and c == worst else ("pass" if residual[c] == 0 else "rounded")
            for c in residual}


def tie_pairs(cells: Sequence[Reading]) -> list:
    """Cells of one tie that print the same value, each joined to the next such cell down the page.

    Inside one statement a tie pairs two rows column by column, so cells pair only within a column.
    """
    ordered = sorted(cells, key=lambda r: middle(r.bbox))
    pairs = []
    for i, a in enumerate(ordered):
        b = next((b for b in ordered[i + 1:] if b.value == a.value and (b.table != a.table or b.col == a.col)), None)
        if b is not None:
            pairs.append((a, b))
    return pairs


def page_marks(g: Graph, doc: str, page: int, words: Sequence[Box], page_right: float) -> Marks:
    """The brackets, ties and cell states of every statement printed on this page."""
    readings = statement_readings(g, doc, page)
    here = {r.id: r for r in readings}
    sums = sorted(checks_on_page(g, "SUM", here), key=lambda cm: max(middle(r.bbox) for r, _ in cm[1]))
    ties = [pair for _, members in checks_on_page(g, "TIE", here) for pair in tie_pairs([r for r, _ in members])]
    ties.sort(key=lambda ab: middle(ab[1].bbox))
    spans = [(min(middle(r.bbox) for r, _ in m), max(middle(r.bbox) for r, _ in m)) for _, m in sums]
    spans += [(middle(a.bbox), middle(b.bbox)) for a, b in ties]
    lane = lanes(spans, [formula_inputs(i, sums) for i in range(len(sums))] + [[] for _ in ties])
    right = column_rights(readings)
    tables = [m[0][0].table for _, m in sums] + [a.table for a, _ in ties]
    step = table_steps(readings, right, words, page_right, tables, lane)
    marks = Marks(cells=[(tuple(r.bbox), r.status) for r in readings], formulas=len(sums))
    for (check, members), k in zip(sums, lane):
        marks.brackets += formula_brackets(g, check, members, right, k, step)
    for (a, b), k in zip(ties, lane[len(sums):]):
        x = max(right[(a.table, a.col)], right[(b.table, b.col)]) + PAD + k * step[a.table]
        marks.ties.append(TieLine(x, (a.bbox[2], middle(a.bbox)), (b.bbox[2], middle(b.bbox))))
    drawn = [tuple(r.bbox) for _, m in sums for r, _ in m] + [tuple(r.bbox) for ab in ties for r in ab]
    reach = [(x, y, x + END, y) for x in [b.x for b in marks.brackets] + [t.x for t in marks.ties] for y in (0,)]
    core = union(drawn)
    marks.extent = None if core is None else union([core] + [(x0, core[1], x1, core[3]) for x0, _, x1, _ in reach])
    return marks


def formula_inputs(i: int, sums: Sequence[tuple]) -> list:
    """Indices of the formulas in the same statement whose totals formula i adds up."""
    check, members = sums[i]
    table = members[0][0].table
    rows = {r.row for r, role in members if role != "total"}
    return [j for j, (other, m) in enumerate(sums) if j != i and m[0][0].table == table
            and any(role == "total" and r.row in rows for r, role in m)]


def column_rights(readings: Sequence[Reading]) -> dict:
    """The right edge of each (statement, column): its numbers are right-aligned there."""
    right: dict = {}
    for r in readings:
        key = (r.table, r.col)
        right[key] = max(right.get(key, r.bbox[2]), r.bbox[2])
    return right


def table_steps(readings: Sequence[Reading], right: dict, words: Sequence[Box], page_right: float,
                tables: Sequence[str], lane: Sequence[int]) -> dict:
    """The lane spacing of each statement: as wide as fits right of its tightest column."""
    rows = defaultdict(list)
    for r in readings:
        rows[r.table].append(r.bbox)
    deepest: dict = defaultdict(int)
    for t, k in zip(tables, lane):
        deepest[t] = max(deepest[t], k)
    step = {}
    for t, boxes in rows.items():
        top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
        room = min(room_right(x, words, top, bottom, page_right) for (table, _), x in right.items() if table == t)
        step[t] = lane_step(room, deepest[t] + 1)
    return step


def formula_brackets(g: Graph, check: Check, members: Sequence[tuple], right: dict, lane: int,
                     step: dict) -> list:
    """One bracket per column in which the formula has two or more cells on this page."""
    states = column_states(g, check)
    by_col = defaultdict(list)
    for r, role in members:
        by_col[r.col].append((r, role))
    out = []
    for col, cells in by_col.items():
        if len(cells) < 2:
            continue
        table = cells[0][0].table
        operands = [(r.bbox[2], middle(r.bbox), int(role)) for r, role in cells if role != "total"]
        total = next(((r.bbox[2], middle(r.bbox)) for r, role in cells if role == "total"), None)
        out.append(Bracket(bracket_x(right[(table, col)], lane, step[table]), operands, total,
                           states.get(col, check.status)))
    return out


# ------------------------------------------------------------------ captions: words and numbers from the graph

def concept_words(name: str) -> str:
    """'TotalLiabilitiesAndEquity' -> 'total liabilities and equity'."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def period_words(pid: str) -> str:
    """A period id in words: FY2026 -> fiscal 2026, FY2026Q3YTD -> nine months of fiscal 2026."""
    m = re.fullmatch(r"FY(\d{4})(?:Q(\d))?(YTD)?(@end)?", pid)
    if m is None:
        return pid
    fy, q, ytd, end = m.groups()
    if end:
        return f"end of Q{q} of fiscal {fy}" if q else f"end of fiscal {fy}"
    if ytd:
        months = {1: "three", 2: "six", 3: "nine"}.get(int(q), str(3 * int(q)))
        return f"{months} months of fiscal {fy}"
    return f"Q{q} of fiscal {fy}" if q else f"fiscal {fy}"


def units_of(value: float, printed: str) -> float:
    """The power of ten from a printed number to the graph's full-unit value (1e6 for "in millions")."""
    digits = re.sub(r"[^\d.]", "", printed)
    try:
        shown = float(digits)
    except ValueError:
        return 1.0
    if not shown or not value:
        return 1.0
    return 10.0 ** round(math.log10(abs(value) / shown))


def number(v: float) -> str:
    """A number as a statement prints it: thousands separators, decimals only when it has them."""
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"


OPERATIONS = (
    ("{a} − {b}", lambda a, b: a - b),
    ("{a} + {b}", lambda a, b: a + b),
    ("{a} / {b}", lambda a, b: a / b),
    ("{a} × {b}", lambda a, b: a * b),
    ("{b} − {a}", lambda a, b: b - a),
    ("({b} − {a}) / {a}", lambda a, b: (b - a) / abs(a)),
)


def joining_operation(value: float, a: float, b: float) -> Optional[str]:
    """The template, such as "{a} − {b}", of the arithmetic that gives `value` from `a` and `b`."""
    for template, fn in OPERATIONS:
        try:
            if math.isclose(fn(a, b), value, rel_tol=1e-9, abs_tol=1e-9):
                return template
        except ZeroDivisionError:
            continue
    return None


def leaf_for(node: dict, leaves: Sequence[Leaf]) -> Optional[Leaf]:
    """The source cell a read step of the proof tree took its value from."""
    named = [lf for lf in leaves if f"{lf.ticker} {lf.what} {lf.period}" == node.get("what")
             and lf.doc == node.get("doc") and lf.page == node.get("page")]
    if named:
        return named[0]
    return next((lf for lf in leaves if lf.value is not None and isinstance(node.get("value"), (int, float))
                 and abs(abs(lf.value) - abs(node["value"])) < 1e-9), None)


def formula_runs(tree: dict, leaves: Sequence[Leaf], numbers: dict) -> Optional[list]:
    """The arithmetic of an answer over its numbered sources, or None when it is not one formula.

    The proof tree keeps every step's value and inputs. A step with two inputs is written
    with the operation that reproduces its value from them, read back from the numbers, so
    the caption never depends on how a step describes itself.
    """
    kids = tree.get("children", [])
    if tree.get("op") == "read":
        lf = leaf_for(tree, leaves)
        if lf is None:
            return [("", number(tree["value"]))]
        shown = number(tree["value"] / units_of(lf.value, lf.printed))
        return [("badge", str(numbers[lf.id])), ("", " " + shown)] if lf.id in numbers else [("", shown)]
    if len(kids) == 1:
        return formula_runs(kids[0], leaves, numbers)
    if len(kids) != 2 or not all(isinstance(k.get("value"), (int, float)) for k in kids):
        return None
    template = joining_operation(tree["value"], kids[0]["value"], kids[1]["value"])
    parts = [formula_runs(k, leaves, numbers) for k in kids]
    if template is None or None in parts:
        return None
    a, b = (p if len(k.get("children", [])) < 2 else [("", "(")] + p + [("", ")")] for p, k in zip(parts, kids))
    runs: list = []
    for piece in re.split(r"(\{a\}|\{b\})", template):
        runs += a if piece == "{a}" else b if piece == "{b}" else [("", piece)] if piece else []
    return runs


def unit_note(leaves: Sequence[Leaf], g: Graph) -> str:
    """'($ millions)' when every source is a dollar amount printed in the same units, else ''."""
    scales = {units_of(lf.value, lf.printed) for lf in leaves if lf.value is not None}
    units = {g.readings[lf.id].unit for lf in leaves if lf.id in g.readings}
    names = {1e3: "thousands", 1e6: "millions", 1e9: "billions"}
    if len(scales) == 1 and units == {"USD"} and next(iter(scales)) in names:
        return f"   ($ {names[next(iter(scales))]}, as printed)"
    return ""


def source_runs(g: Graph, lf: Leaf, n: int, with_value: bool) -> list:
    """One source line: badge, printed label, period, filing and page, and its state if not verified."""
    r = g.readings.get(lf.id)
    label = (r.label if r is not None else lf.label) or f"{concept_words(lf.what)} (printed with no label)"
    form = g.filings[lf.doc]["form"] if lf.doc in g.filings else lf.doc
    text = f" {label} · {period_words(lf.period)} · {form} page {lf.page}"
    runs = [("badge", str(n)), ("", text + (f" · {lf.printed}" if with_value else ""))]
    if r is None or not r.bbox:
        runs.append(("muted", " · not in a table, not drawn"))
    elif lf.status != "verified":
        runs.append(("muted", f" · {lf.status}"))
    return runs


def answer_caption(g: Graph, result: Result, numbers: dict, question: Optional[str], hidden: int) -> list:
    """Lines of runs: the question, the answer with its formula, then the numbered sources."""
    lines: list = [[("bold", question)]] if question else []
    formula = formula_runs(result.tree, result.leaves, numbers) if result.ok else None
    answer = [("bold", render(result))]
    if formula:
        answer += [("", "  =  ")] + formula + [("muted", unit_note(result.leaves, g))]
    lines.append(answer)
    for lf in result.leaves[:MAX_SOURCES]:
        lines.append(source_runs(g, lf, numbers[lf.id], with_value=not formula))
    if len(result.leaves) > MAX_SOURCES:
        lines.append([("muted", f"and {len(result.leaves) - MAX_SOURCES} more sources")])
    if hidden:
        lines.append([("muted", f"sources on {hidden} more pages are listed but not drawn")])
    return lines


def page_caption(g: Graph, doc: str, page: int, marks: Marks) -> list:
    """What the graph recovered on a page, in one line."""
    states = defaultdict(int)
    for _, s in marks.cells:
        states[s] += 1
    title = panel_title(g, doc, page)
    if not marks.cells:
        return [[("bold", title), ("", ": no statement the graph reads is printed on this page")]]
    return [[("bold", title), ("", f": {marks.formulas} formulas recovered; {states['verified']} of "
                                    f"{len(marks.cells)} statement numbers verified, {states['failed']} failed")]]


def panel_title(g: Graph, doc: str, page: int) -> str:
    """'AAPL 10-K FY2026, page 35' (a 10-Q also names the end of its period)."""
    f = g.filings.get(doc)
    if f is None:
        return f"{doc}, page {page}"
    period = f", period ended {f['period_end']}" if f["form"] != "10-K" else ""
    return f"{f['ticker']} {f['form']} {f['fiscal_year']}{period}, page {page}"


# ------------------------------------------------------------------ drawing

@dataclass
class Panel:
    """One page region: the rendered image, what is drawn on it and the answer cells it outlines."""
    title: str
    crop: Box
    image: object
    marks: Marks
    outlined: list               # (box, badge number)


def pdf_path(doc: str) -> pathlib.Path:
    path = PDF_DIR / f"{doc}.pdf"
    if not path.exists():
        raise FileNotFoundError(f"{path} is not here; fetch the corpus first (scripts/fetch_corpus.py)")
    return path


def load_panel(g: Graph, doc: str, page: int, numbers: dict, resolution: int) -> Panel:
    """Open the page once: its words place the drawing, and its pixels are what the drawing sits on."""
    with pdfplumber.open(pdf_path(doc)) as pdf:
        p = pdf.pages[page - 1]
        words = [(w["x0"], w["top"], w["x1"], w["bottom"]) for w in p.extract_words()]
        page_box = tuple(float(v) for v in p.bbox)
        marks = page_marks(g, doc, page, words, page_box[2])
        outlined = [(tuple(g.readings[rid].bbox), n) for rid, n in numbers.items()
                    if rid in g.readings and g.readings[rid].doc == doc and g.readings[rid].page == page
                    and g.readings[rid].bbox]
        core = union([b for b in [marks.extent] if b] + [b for b, _ in outlined])
        crop = page_box if core is None else crop_region(core, words, page_box)
        image = p.crop(crop).to_image(resolution=resolution).original
    return Panel(panel_title(g, doc, page), crop, image, marks, outlined)


def draw_panel(ax, p: Panel) -> None:
    x0, top, x1, bottom = p.crop
    ax.imshow(p.image, extent=(x0, x1, bottom, top), interpolation="antialiased", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(bottom, top)
    ax.set_aspect("auto")
    ax.set_axis_off()
    for box, state in p.marks.cells:
        ax.add_patch(Rectangle((box[0] - 0.8, box[1] - 0.6), box[2] - box[0] + 1.6, box[3] - box[1] + 1.2,
                               facecolor=FILL.get(state, FILL["unverified"]), edgecolor="none",
                               alpha=FILL_ALPHA, zorder=1))
    for b in p.marks.brackets:
        draw_bracket(ax, b)
    for t in p.marks.ties:
        ax.plot([t.upper[0] + EPS, t.x, t.x, t.lower[0] + EPS], [t.upper[1], t.upper[1], t.lower[1], t.lower[1]],
                color=PASS, lw=LINE, ls=(0, (2.2, 1.4)), zorder=3)
    for box, n in p.outlined:
        ax.add_patch(Rectangle((box[0] - 1.6, box[1] - 1.2), box[2] - box[0] + 3.2, box[3] - box[1] + 2.4,
                               fill=False, edgecolor=BLUE, lw=1.1, zorder=5))
        ax.annotate(str(n), (box[0] - 1.6, middle(box)), xytext=(-2.2, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=BADGE, fontweight="bold", color="white", zorder=6,
                    bbox={"boxstyle": "circle,pad=0.22", "facecolor": BLUE, "edgecolor": "none"})


def draw_bracket(ax, b: Bracket) -> None:
    """The vertical line, a tick to each operand, a dot at the total and a minus beside subtracted operands."""
    colour = STATE[b.state]
    ys = [y for _, y, _ in b.operands] + ([b.total[1]] if b.total else [])
    ends = b.operands + ([(b.total[0], b.total[1], 1)] if b.total else [])
    segments = [[(b.x, min(ys)), (b.x, max(ys))]] + [[(x + EPS, y), (b.x, y)] for x, y, _ in ends]
    ax.add_collection(LineCollection(segments, colors=colour, linewidths=LINE, capstyle="butt", zorder=3))
    if b.total:
        ax.plot([b.x], [b.total[1]], marker="o", ms=DOT, mew=0, color=colour, zorder=4)
    for _, y, sign in b.operands:
        if sign < 0:
            ax.annotate("−", (b.x, y), xytext=(0.7, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=MINUS_SIZE, fontweight="bold", color=colour, zorder=4,
                        path_effects=[patheffects.withStroke(linewidth=1.4, foreground="white")])


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Width in points of a string set in the figure's font."""
    prop = FontProperties(size=size, weight="bold" if bold else "normal")
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0]


def run_width(run: Run, size: float) -> float:
    style, text = run
    return size * 1.5 if style == "badge" else text_width(text, size, style == "bold")


def wrap(line: Sequence[Run], width: float, size: float) -> list:
    """Break one line of runs at spaces into lines no wider than `width` points."""
    pieces = [(s, t) if s == "badge" else (s, w) for s, t in line for w in ([t] if s == "badge" else
                                                                           re.findall(r"\S+\s*|\s+", t))]
    lines: list = [[]]
    used = 0.0
    for piece in pieces:
        w = run_width(piece, size)
        if lines[-1] and used + run_width((piece[0], piece[1].rstrip()), size) > width:
            lines.append([])
            used = 0.0
            if piece[0] != "badge" and not piece[1].strip():
                continue
        lines[-1].append(piece)
        used += w
    return lines


def legend_runs() -> list:
    """The key, as (swatch, label) pairs."""
    return [(("fill", "verified"), "verified"), (("fill", "unverified"), "unverified"), (("fill", "failed"), "failed"),
            (("bracket", "pass"), "formula holds"), (("bracket", "rounded"), "within rounding"),
            (("bracket", "fail"), "fails"), (("tie", ""), "same value printed twice"), (("outline", ""), "answer source")]


def caption_height(lines: Sequence[Sequence[Run]], width: float) -> tuple:
    """(wrapped caption lines, legend rows, height in points) for a caption `width` points wide."""
    wrapped = [w for line in lines for w in wrap(line, width, FONT)]
    rows: list = [[]]
    used = 0.0
    for swatch, label in legend_runs():
        w = 12 + text_width(label, SMALL) + 8
        if rows[-1] and used + w > width:
            rows.append([])
            used = 0.0
        rows[-1].append((swatch, label))
        used += w
    return wrapped, rows, len(wrapped) * FONT * LEADING + len(rows) * SMALL * LEADING + 2


def draw_caption(ax, wrapped: Sequence[Sequence[Run]], rows: Sequence, width: float, height: float) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    y = 0.0
    for line in wrapped:
        x = 0.0
        for style, text in line:
            draw_run(ax, x, y + FONT * LEADING / 2, style, text)
            x += run_width((style, text), FONT)
        y += FONT * LEADING
    for row in rows:
        x = 0.0
        y += SMALL * LEADING
        for swatch, label in row:
            draw_swatch(ax, x, y - SMALL * LEADING / 2, swatch)
            ax.text(x + 12, y - SMALL * LEADING / 2, label, fontsize=SMALL, color=MUTED, va="center",
                    parse_math=False)
            x += 12 + text_width(label, SMALL) + 8


def draw_run(ax, x: float, y: float, style: str, text: str) -> None:
    if style == "badge":
        ax.text(x + FONT * 0.72, y, text, fontsize=BADGE, fontweight="bold", color="white", ha="center",
                va="center", bbox={"boxstyle": "circle,pad=0.22", "facecolor": BLUE, "edgecolor": "none"})
        return
    ax.text(x, y, text, fontsize=FONT, va="center", ha="left", color=MUTED if style == "muted" else INK,
            fontweight="bold" if style == "bold" else "normal", parse_math=False)


def draw_swatch(ax, x: float, y: float, swatch: tuple) -> None:
    kind, state = swatch
    if kind == "fill":
        ax.add_patch(Rectangle((x, y - 2.2), 9, 4.4, facecolor=FILL[state], alpha=FILL_ALPHA, edgecolor="none"))
    elif kind == "bracket":
        ax.plot([x + 1, x + 6, x + 6, x + 1], [y - 2.4, y - 2.4, y + 2.4, y + 2.4], color=STATE[state], lw=LINE)
        ax.plot([x + 6], [y + 2.4], marker="o", ms=DOT, mew=0, color=STATE[state])
    elif kind == "tie":
        ax.plot([x, x + 9], [y, y], color=PASS, lw=LINE, ls=(0, (2.2, 1.4)))
    else:
        ax.add_patch(Rectangle((x, y - 2.2), 9, 4.4, fill=False, edgecolor=BLUE, lw=1.1))


def compose(panels: Sequence[Panel], caption: Sequence[Sequence[Run]], out: Paths, width_cm: Optional[float],
            resolution: int) -> list:
    """Place the panels side by side above the caption and the key; write every output path."""
    scale, sizes = layout([(p.crop[2] - p.crop[0], p.crop[3] - p.crop[1]) for p in panels],
                          width_cm / 2.54 if width_cm else None)
    width = 2 * FIGURE_MARGIN + sum(w for w, _ in sizes) + PANEL_GAP * (len(sizes) - 1)
    title = SMALL * LEADING / 72
    text_pt = (width - 2 * FIGURE_MARGIN) * 72
    wrapped, rows, cap_pt = caption_height(caption, text_pt)
    height = 2 * FIGURE_MARGIN + title + max(h for _, h in sizes) + CAPTION_GAP + cap_pt / 72
    fig = Figure(figsize=(width, height))
    x = FIGURE_MARGIN
    for p, (w, h) in zip(panels, sizes):
        fig.text(x / width, 1 - FIGURE_MARGIN / height, p.title, fontsize=SMALL, color=MUTED, va="top",
                 parse_math=False)
        draw_panel(fig.add_axes((x / width, 1 - (FIGURE_MARGIN + title + h) / height, w / width, h / height)), p)
        x += w + PANEL_GAP
    cap = fig.add_axes((FIGURE_MARGIN / width, FIGURE_MARGIN / height, text_pt / 72 / width, cap_pt / 72 / height))
    draw_caption(cap, wrapped, rows, text_pt, cap_pt)
    paths = [pathlib.Path(out)] if isinstance(out, (str, pathlib.Path)) else [pathlib.Path(o) for o in out]
    with matplotlib.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=resolution, facecolor="white")
    return paths


# ------------------------------------------------------------------ entry points

def render_page(g: Graph, doc: str, page: int, out_path: Paths, highlight: Sequence[str] = (),
                caption: Optional[str] = None, resolution: int = 150, *, width_cm: Optional[float] = None) -> list:
    """Draw one page with its formulas; `highlight` reading ids are outlined and numbered in order."""
    panel = load_panel(g, doc, page, {rid: i + 1 for i, rid in enumerate(highlight)}, resolution)
    lines = [[("", caption)]] if caption else page_caption(g, doc, page, panel.marks)
    return compose([panel], lines, out_path, width_cm, resolution)


def render_answer(g: Graph, result: Result, out_path: Paths, question: Optional[str] = None, *,
                  width_cm: Optional[float] = None, resolution: int = 150) -> list:
    """Draw the pages an answer's sources are printed on, side by side, under its caption."""
    numbers = {lf.id: i + 1 for i, lf in enumerate(result.leaves)}
    pages = source_pages(g, result.leaves)
    if not pages:
        raise ValueError("none of this answer's sources is a cell the graph can place on a page")
    panels = [load_panel(g, doc, page, numbers, resolution) for doc, page in pages[:MAX_PANELS]]
    caption = answer_caption(g, result, numbers, question, len(pages) - len(panels))
    return compose(panels, caption, out_path, width_cm, resolution)


def render_readings(g: Graph, ids: Sequence[str], out_path: Paths, *, width_cm: Optional[float] = None,
                    resolution: int = 150) -> list:
    """Draw the pages of the given readings, each outlined and numbered, with their values listed."""
    leaves = [Leaf(r.id, r.ticker, r.label, r.period, r.doc, r.page, r.label, r.text, r.status, r.value)
              for r in (g.readings[i] for i in ids)]
    numbers = {lf.id: i + 1 for i, lf in enumerate(leaves)}
    pages = source_pages(g, leaves)
    panels = [load_panel(g, doc, page, numbers, resolution) for doc, page in pages[:MAX_PANELS]]
    caption = [source_runs(g, lf, numbers[lf.id], with_value=True) for lf in leaves]
    return compose(panels, caption, out_path, width_cm, resolution)


def source_pages(g: Graph, leaves: Sequence[Leaf]) -> list:
    """(doc, page) of every source the graph can place, in the order the sources come."""
    pages = [(lf.doc, lf.page) for lf in leaves if lf.id in g.readings and g.readings[lf.id].bbox]
    return list(dict.fromkeys(pages))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tieout show", description="Draw the formulas the graph recovered on a "
                                "statement page, with an answer's source cells outlined.")
    p.add_argument("--doc", help="filing id, with --page")
    p.add_argument("--page", type=int, help="page number in the PDF (1 is the first page)")
    p.add_argument("--reading", nargs="+", metavar="ID", help="reading ids to outline")
    p.add_argument("--plan", help="a plan as JSON text or a path to a JSON file; run directly, no model")
    p.add_argument("--question", help="the question: shown in the caption; alone, answered from cached plans")
    p.add_argument("--out", action="append", help="output .png or .pdf (repeat for both); default show.png")
    p.add_argument("--width", type=float, help="figure width in cm (default: the page's printed size)")
    p.add_argument("--resolution", type=int, default=150, help="dots per inch of the page image and a PNG")
    args = p.parse_args(argv)
    if not (args.doc and args.page) and not args.reading and not args.plan and not args.question:
        p.error("give --doc and --page, --reading, --plan or --question")
    return args


def read_plan(text: str) -> dict:
    """A plan given inline as JSON or as the path of a JSON file."""
    return json.loads(text if text.lstrip().startswith("{") else pathlib.Path(text).read_text(encoding="utf-8"))


def answer_offline(question: str, g: Graph) -> Optional[Result]:
    """Answer from the cached plans only; None (with the reason printed) when no plan is cached."""
    from .ask import answer
    out = answer(question, g, offline=True, persist=False)
    if not out["plan"].get("steps"):
        print(f"no plan for this question is available offline ({out['plan'].get('cannot', 'no steps')}).\n"
              "Write the plan yourself and pass it with --plan (see benchmark/plan_language.txt).")
        return None
    print("PLAN:  ", json.dumps(out["plan"]))
    return out["result"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = args.out or ["show.png"]
    size = {"width_cm": args.width, "resolution": args.resolution}
    try:
        if args.doc:
            paths = render_page(Graph.load(), args.doc, args.page, out, resolution=args.resolution,
                                width_cm=args.width)
        else:
            from .__main__ import load_graph
            g = load_graph()
            if args.reading:
                unknown = [i for i in args.reading if i not in g.readings]
                if unknown:
                    print("not a reading of the graph:", " ".join(unknown))
                    return 1
                paths = render_readings(g, args.reading, out, **size)
            else:
                result = Executor(g).run(read_plan(args.plan)) if args.plan else answer_offline(args.question, g)
                if result is None:
                    return 1
                print("ANSWER:", render(result))
                paths = render_answer(g, result, out, args.question, **size)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1
    for path in paths:
        print("wrote", path)
    return 0
