"""Which cell stands for a row in a period: arithmetic across columns, headings above the table,
and what a quarterly report cannot print (tieout/link.py)."""
from tieout.link import _heading_years, column_groups, default_cell, heading_key, members, period_cells


def row(cells, doc="ACME_FY2025_10K", above=""):
    """cells: (period id or None, column header, value)."""
    return {"doc": doc, "above": above, "label": "Line",
            "cells": [{"pid": pid, "col": col, "value": value, "text": str(value)} for pid, col, value in cells]}


def test_the_column_that_sums_the_others_is_the_total():
    r = row([("FY2025", "U.S.", 11252), ("FY2025", "Non-U.S.", 10596), ("FY2025", "", 21848)])
    assert period_cells(r) == {"FY2025": 2}
    assert members(r)[2].endswith("(the sum of the others)")


def test_nested_sums_end_at_the_grand_total():
    # four regions add up to external sales; external plus inter-segment sales add up to the total
    r = row([("FY2024", h, v) for h, v in [("North America", 4597), ("Latin America", 2079), ("EAME", 1809),
                                           ("Asia/Pacific", 3615), ("", 12100), ("", 371), ("", 12471)]])
    assert period_cells(r) == {"FY2024": 6}


def test_a_link_can_name_one_column_of_the_group():
    r = row([("FY2025", "U.S.", 2641), ("FY2025", "Int'l", 2592), ("FY2025", "Total", 5233)])
    assert period_cells(r, member=0) == {"FY2025": 0}
    assert period_cells(r, member=9) == {"FY2025": 2}            # no such column: the default cell


def test_a_year_printed_once_over_three_columns_is_one_group():
    r = row([(None, "U.S.", 2425), ("FY2024", "Int'l | 2024", 6158), (None, "Total", 8583)])
    assert column_groups(r) == {"FY2024": [0, 1, 2]}
    assert period_cells(r) == {"FY2024": 2}


def test_an_adjusted_column_is_never_the_default_even_when_it_is_a_sum():
    # as reported + currency effect = at constant currency: the figure as reported is the row's figure
    r = row([("FY2025", "As Reported", 716924), ("FY2025", "Exchange Rate Effect", -4409),
             ("FY2025", "At Prior Year Rates", 712515)])
    assert default_cell(r, [0, 1, 2]) == 0


def test_without_arithmetic_the_headed_total_then_the_first_column():
    assert period_cells(row([("FY2025", "Stores", 939), ("FY2025", "Square feet", 137000)])) == {"FY2025": 0}
    assert period_cells(row([("FY2025", "Owned", 10), ("FY2025", "Total", 25)])) == {"FY2025": 1}


def test_undated_columns_take_the_years_the_heading_prints():
    above = "TABLE SELECTED INCOME STATEMENT DATA | (Millions, except percentages) 2025 2024 2023 2025 vs. 2024"
    r = row([(None, "", 34814), (None, "", 31427), (None, "", 28116), (None, "", 3387)], above=above)
    assert period_cells(r) == {"FY2025": 0, "FY2024": 1, "FY2023": 2}


def test_heading_years_are_a_short_run_of_reported_years():
    assert _heading_years("(Dollars in millions) FISCAL 2026 FISCAL 2025 CHANGE") == [2026, 2025]
    assert _heading_years("2025 vs. 2024") == []
    # a maturity schedule runs past the filing's own year, a vintage table is too long: neither dates a row
    schedule = row([(None, "", 1), (None, "", 2), (None, "", 3)], above="(Millions) 2026 2027 2028")
    vintage = row([(None, "", v) for v in range(5)], above="2025 2024 2023 2022 2021 Prior")
    assert period_cells(schedule) == {} and period_cells(vintage) == {}


def test_a_quarterly_report_prints_no_full_year_and_no_later_quarter():
    r = row([("FY2025", "2025", 100), ("FY2025Q3", "Three months", 30), ("FY2025Q4", "", 40),
             ("FY2024@end", "December 31, 2024", 7)], doc="ACME_FY2025_Q3_10Q")
    assert set(period_cells(r)) == {"FY2025Q3", "FY2024@end"}


def test_the_same_table_in_two_filings_shares_a_heading_key_and_another_segment_does_not():
    a = {"above": "U.S. CONSUMER SERVICES | TABLE USCS SELECTED INCOME STATEMENT DATA | (Millions) 2025 2024 2023"}
    b = {"above": "U.S. CONSUMER SERVICES | TABLE USCS SELECTED INCOME STATEMENT DATA | (Millions) 2024 2023 2022"}
    c = {"above": "COMMERCIAL SERVICES | TABLE CS SELECTED INCOME STATEMENT DATA | (Millions) 2025 2024 2023"}
    assert heading_key(a) == heading_key(b) != heading_key(c)


# ---------------------------------------------------------------- breakdowns across columns

class FakeStore:
    def __init__(self, rows):
        self.rows_by_ticker = {"ACME": rows}
        self.by_id = {r["id"]: r for r in rows}


def segment_row(rid, label, values, pid="FY2025"):
    heads = ["Retail", "Wholesale", "Online", "Consolidated"]
    return {"id": rid, "doc": "ACME_FY2025_10K", "ticker": "ACME", "page": 70, "label": label, "section": "",
            "title": "Segment information", "above": "Segment information | (in millions)", "unit": "USD",
            "cells": [{"pid": pid, "head": h, "col": "", "value": v, "text": str(v), "ok": True} for h, v in zip(heads, values)]}


def test_segments_printed_as_columns_are_a_breakdown():
    from tieout.link import across
    st = FakeStore([segment_row("r1", "Operating income", [50, -10, 30, 70])])
    lines = across(st, st.by_id["r1"])
    assert [ln["label"] for ln in lines] == ["Retail", "Wholesale", "Online"]          # the sum column is the total
    assert lines[1]["cells"]["FY2025"]["value"] == -10


def test_a_row_without_an_arithmetic_total_is_not_a_breakdown():
    from tieout.link import across
    st = FakeStore([segment_row("r1", "Headcount", [5, 6, 7, 99])])
    assert across(st, st.by_id["r1"]) == []


def test_the_other_figure_of_a_segment_is_the_same_column_of_another_row(monkeypatch):
    from tieout import link
    st = FakeStore([segment_row("r1", "Operating income", [50, -10, 30, 70]),
                    segment_row("r2", "Total revenues", [500, 200, 300, 1000])])
    monkeypatch.setattr(link, "store", lambda: st)
    wholesale = link.across(st, st.by_id["r1"])[1]
    beside = link.same_table_figure(wholesale, "revenues")
    assert beside["label"] == "Wholesale" and beside["cells"]["FY2025"]["value"] == 200
    assert link.same_table_figure(wholesale, "headcount") is None
