"""Tests that need no PDFs: formula recovery, fiscal calendars and the executor's type checks."""
from tieout.calc import coverage, discover, ties, verified_cells
from tieout.periods import FiscalCalendar
from tieout.tables import Cell, Column, Row, Table
from datetime import date


def make_table(rows, ncols=3):
    """rows: [(label, [values...])] -> Table with money cells."""
    t = Table(doc="T", page=1, title="test", caption="(in millions)", scale=1e6, share_scale=1e6,
              columns=[Column(i, 100.0 + 50 * i, 80.0 + 50 * i) for i in range(ncols)], rows=[])
    for i, (label, vals) in enumerate(rows):
        r = Row(idx=i, page=1, y=10.0 * i, label=label, indent=0.0, label_bbox=None)
        for c, v in enumerate(vals):
            if v is not None:
                r.cells[c] = Cell(page=1, bbox=(0, 0, 1, 1), text=f"{v:,}", value=float(v), col=c)
        t.rows.append(r)
    return t


INCOME = [("Products", [294866, 298085, 316199]), ("Services", [96169, 85200, 78129]),
          ("Total net sales", [391035, 383285, 394328]),
          ("Cost of products", [185233, 189282, 201471]), ("Cost of services", [25119, 24855, 22075]),
          ("Total cost of sales", [210352, 214137, 223546]),
          ("Gross margin", [180683, 169148, 170782]),
          ("Research and development", [31370, 29915, 26251]), ("Selling, general and administrative", [26097, 24932, 25094]),
          ("Total operating expenses", [57467, 54847, 51345]),
          ("Operating income", [123216, 114301, 119437]),
          ("Other income, net", [269, -565, -334]),
          ("Income before taxes", [123485, 113736, 119103]),
          ("Provision for income taxes", [29749, 16741, 19300]),
          ("Net income", [93736, 96995, 99803])]


def test_formulas_are_recovered_from_arithmetic_alone():
    t = make_table(INCOME)
    checks = discover(t)
    by_total = {t.rows[c.total].label: c for c in checks}
    assert set(by_total) == {"Total net sales", "Total cost of sales", "Gross margin", "Total operating expenses",
                             "Operating income", "Income before taxes", "Net income"}
    gm = by_total["Gross margin"]
    assert gm.form == "base_minus"
    assert [(t.rows[i].label, s) for i, s in gm.operands] == [("Total net sales", 1), ("Total cost of sales", -1)]
    assert coverage(t, checks) == (45, 45)


def test_a_sign_flip_is_not_verified():
    rows = [(label, list(vals)) for label, vals in INCOME]
    rows[11] = ("Other income, net", [269, 565, -334])          # the parenthesis of (565) was lost
    t = make_table(rows)
    ok = verified_cells(t, discover(t))
    assert (11, 1) not in ok                                     # the damaged cell is no longer vouched for
    assert (11, 0) in ok and (11, 2) in ok                       # its other columns still foot


def test_an_identity_rearranged_around_a_leaf_is_refused():
    # AOCI = assets - liabilities - stock - retained earnings is true arithmetic and the wrong tree;
    # the leaf belongs to "Total equity" two rows below.
    bs = [("Total assets", [364980, 352583]), ("Total liabilities", [308030, 290437]),
          ("Common stock", [83276, 73812]), ("Retained earnings", [-19154, -214]), ("Accumulated OCI", [-7172, -11452]),
          ("Total shareholders' equity", [56950, 62146]), ("Total liabilities and equity", [364980, 352583])]
    t = make_table(bs, ncols=2)
    checks = discover(t)
    totals = {t.rows[c.total].label for c in checks}
    assert "Accumulated OCI" not in totals
    assert {"Total shareholders' equity", "Total liabilities and equity"} <= totals
    assert ties(t) == [(0, 6)]                                   # assets == liabilities and equity


def test_fiscal_calendars_differ_by_company():
    apple = FiscalCalendar.from_latest("AAPL", "FY2026", "2026-09-26")
    target = FiscalCalendar.from_latest("TGT", "FY2025", "2026-01-31")
    assert apple.period_id(date(2025, 9, 27), 12, False) == "FY2025"        # 52/53-week drift
    assert apple.period_id(date(2026, 6, 27), 9, False) == "FY2026Q3YTD"
    assert apple.period_id(date(2026, 6, 27), 3, False) == "FY2026Q3"
    assert apple.period_id(date(2026, 9, 26), None, True) == "FY2026@end"
    assert target.period_id(date(2026, 1, 31), 12, False) == "FY2025"       # Target names the year it starts in
    assert target.end_of_label(2024) == date(2025, 1, 31)
