"""Periods of table columns and rows, on synthetic pages: no PDF needed."""
from tieout.alltables import page_tables
from tieout.periods import FiscalCalendar
from tieout.rawlayer import _maturity_columns, _period, table_rows
from tieout.statements import _lengths, _percent_columns, _side_by_side
from tieout.tables import Cell, Column, Row, Table, Word, printed_date

CAL = FiscalCalendar("T", 12, 31, 0)          # fiscal years end on December 31


def _line(y, *parts, cw=5.0):
    """parts: (x0, text) pairs; each word `cw` points wide per character."""
    return [Word(x, y, x + cw * len(s), y + 8, s) for x, s in parts]


def _page(lines):
    return {"w": 612, "h": 792, "words": [w for ln in lines for w in ln]}


def _values(y, label, rights, texts):
    """A table row: the label at x=60, each value right-aligned on its column edge."""
    return _line(y, (60, label), *[(r - 5 * len(t), t) for r, t in zip(rights, texts)])


def _tables(lines, form="10-K"):
    """(table, periods of its columns) for every table on the page."""
    out = []
    for t in page_tables("T_FY2025_10K", 5, _page(lines), 1e6):
        out.append((t, [_period(CAL, c, form)[0] for c in t.columns]))
    return out


def test_year_over_each_group_of_category_columns():
    rights = [300, 350, 400, 450, 500, 550]
    lines = [_line(100, (330, "2025"), (480, "2024")),
             _line(124, (280, "U.S."), (310, "Non-U.S."), (375, "Total"), (430, "U.S."), (460, "Non-U.S."), (525, "Total"))]
    for i, label in enumerate(["Upstream", "Energy Products", "Chemical Products", "Total Capex"]):
        lines.append(_values(140 + 12 * i, label, rights, ["1,100", "2,200", "3,300", "1,000", "2,000", "3,000"]))
    (t, pids), = _tables(lines)
    assert pids == ["FY2025"] * 3 + ["FY2024"] * 3


def test_each_fiscal_year_label_is_its_own_column():
    rights = [302, 347, 475]
    lines = [_line(100, (267, "FISCAL"), (290, "2026"), (312, "FISCAL"), (335, "2025"), (440, "FISCAL"), (463, "2024"),
                   cw=3.5)]
    for i, label in enumerate(["Footwear", "Apparel", "Equipment", "Total revenues"]):
        lines.append(_values(116 + 12 * i, label, rights, ["4,188", "4,805", "5,552"]))
    (t, pids), = _tables(lines)
    assert pids == ["FY2026", "FY2025", "FY2024"]


def test_nine_months_columns_are_year_to_date():
    rights = [376, 446, 516, 586]
    lines = [_line(100, (336, "Three"), (356, "Months"), (379, "Ended"), (399, "September"), (433, "30"),
                   (478, "Nine"), (494, "Months"), (517, "Ended"), (537, "September"), (571, "30"), cw=3.4),
             _line(110, (360, "2025"), (430, "2024"), (500, "2025"), (570, "2024"), cw=4.0)]
    for i, label in enumerate(["Loans and leases", "Debt securities", "Trading account assets", "Total interest"]):
        lines.append(_values(126 + 12 * i, label, rights, ["16,191", "15,725", "47,065", "46,303"]))
    (t, pids), = _tables(lines, form="10-Q")
    assert pids == ["FY2025Q3", "FY2024Q3", "FY2025Q3YTD", "FY2024Q3YTD"]


def test_stacked_blocks_take_the_date_line_above_them():
    rights = [300, 360]
    lines = [_line(100, (250, "Repurchase"), (320, "Securities")), _line(112, (250, "agreements"), (330, "loaned")),
             _line(124, (60, "As"), (73, "of"), (86, "December"), (132, "2025"))]
    y = 136
    for label in ("Gross carrying value", "Counterparty netting", "Total"):
        lines.append(_values(y, label, rights, ["404,771", "71,581"]))
        y += 12
    lines.append(_line(y, (60, "As"), (73, "of"), (86, "December"), (132, "2024")))
    y += 12
    for label in ("Gross carrying value", "Counterparty netting", "Total"):
        lines.append(_values(y, label, rights, ["313,924", "66,674"]))
        y += 12
    (t, pids), = _tables(lines)
    rows = table_rows("T", "TT", 0, t, [_period(CAL, c, "10-K") for c in t.columns], {}, CAL, "10-K")
    assert [{c["pid"] for c in r["cells"]} for r in rows] == [{"FY2025"}] * 3 + [{"FY2024"}] * 3
    assert {c["end"] for c in rows[-1]["cells"]} == {"2024-12-31"}


def test_month_without_day_is_the_month_end():
    assert printed_date("As of December 2025") == "2025-12-31"
    assert printed_date("December 31, 2025") == "2025-12-31"
    assert printed_date("September 28, 2025") == "2025-09-28"
    assert printed_date("June 2024", year=2025) == "2025-06-30"
    assert printed_date("Total") is None


def _table(headers, years):
    cols = [Column(i, 300.0 + 50 * i, 280.0 + 50 * i, header=h, year=y) for i, (h, y) in enumerate(zip(headers, years))]
    return Table(doc="T", page=1, title="", caption="", scale=1e6, share_scale=1e6, columns=cols, rows=[])


def test_maturity_schedule_columns_are_not_periods():
    t = _table(["2026", "2027", "2028", "Thereafter", "Total"], [2026, 2027, 2028, None, None])
    assert _maturity_columns(t, 2025) == {0, 1, 2, 3, 4}
    assert _maturity_columns(_table(["2025", "2024", "2023"], [2025, 2024, 2023]), 2025) == set()
    assert _maturity_columns(_table(["2023", "2024", "2025"], [2023, 2024, 2025]), 2025) == set()


def _statement_table(page, labels, cols):
    t = Table(doc="T", page=page, title="", caption="", scale=1e6, share_scale=1e6,
              columns=[Column(i, 300.0 + 60 * i, 270.0 + 60 * i, header=h, year=y, months=m)
                       for i, (h, y, m) in enumerate(cols)], rows=[])
    for i, (label, vals) in enumerate(labels):
        r = Row(idx=i, page=page, y=10.0 * i, label=label, indent=0.0, label_bbox=None)
        for c, v in enumerate(vals):
            if v is not None:
                r.cells[c] = Cell(page=page, bbox=(0, 0, 1, 1), text=str(v), value=float(v), col=c)
        t.rows.append(r)
    return t


def test_quarter_and_year_to_date_pages_join_side_by_side():
    quarter = _statement_table(7, [("Sales", [23993, 22471]), ("Cost", [7303, 6963]), ("Gross profit", [16690, 15508])],
                               [("2025", 2025, 3), ("2024", 2024, 3)])
    ytd = _statement_table(8, [("Sales", [69629, 66301]), ("Cost", [22288, 20343]), ("Impairment", [0, 194]),
                               ("Gross profit", [47341, 45958])], [("2025", 2025, 9), ("2024", 2024, 9)])
    t = _side_by_side(quarter, ytd)
    assert [c.months for c in t.columns] == [3, 3, 9, 9] and _lengths(t) == {3, 9}
    assert [r.label for r in t.rows] == ["Sales", "Cost", "Impairment", "Gross profit"]
    assert sorted(t.rows[0].cells) == [0, 1, 2, 3] and t.rows[0].cells[2].page == 8
    assert sorted(t.rows[2].cells) == [2, 3]                      # printed on the year-to-date page only


def test_percent_of_sales_columns_are_found():
    t = _statement_table(7, [("Cost", [7303, "30.4", 6963, "31.0"]), ("Gross profit", [16690, "69.6", 15508, "69.0"]),
                             ("Selling", [5922, "24.7", 5478, "24.3"])],
                         [("2025", 2025, 3), ("Percent to Sales", None, 3), ("2024", 2024, 3), ("to Sales", None, 3)])
    t.header_text = "2025 Percent to Sales 2024 Percent to Sales"
    assert _percent_columns(t) == {1, 3}
