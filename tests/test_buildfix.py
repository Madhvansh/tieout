"""Build-side fixes: share-count rows, wrapped statement titles, 10-Q year columns, label objections,
unscaled sentence dollars and per-cell members of category columns. Synthetic inputs only."""
from types import SimpleNamespace as NS

from tieout.calc import row_pool, row_section
from tieout.concepts import label_objections, label_unrelated
from tieout.mentions import SCALE_UNKNOWN, settle_scale
from tieout.periods import FiscalCalendar, column_period
from tieout.rawlayer import member_of, table_rows
from tieout.statements import split_regions
from tieout.tables import Cell, Column, Row, Table, Word

CAL = FiscalCalendar("T", 12, 31, 0)


def _table(rows, ncols=2, caption="(in millions, except per share amounts)"):
    t = Table(doc="T", page=1, title="Consolidated Statements of Income", caption=caption, scale=1e6,
              share_scale=1e6, columns=[Column(i, 300.0 + 60 * i, 260.0 + 60 * i, header=h)
                                        for i, h in enumerate(["2025", "2024", "2023"][:ncols])], rows=[])
    for i, (label, texts, section) in enumerate(rows):
        r = Row(idx=i, page=1, y=10.0 * i, label=label, indent=0.0, label_bbox=None, section=section)
        for c, s in enumerate(texts):
            if s is not None:
                v = float(s.replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
                r.cells[c] = Cell(page=1, bbox=(0, 0, 1, 1), text=s, value=v, col=c, has_dollar="$" in s)
        t.rows.append(r)
    return t


# ---------------------------------------------------------------- A: share counts and EPS

def test_decimal_share_counts_are_shares_not_per_share_amounts():
    t = _table([("Weighted-average basic shares", ["2,776.5", "2,797.6"], ""),
                ("Weighted-average diluted shares", ["2,781.5", "2,801.6"], ""),
                ("Average diluted common shares issued and outstanding", ["7,680.9", "7,935.8"], ""),
                ("Diluted earnings per share", ["$19.75", "$17.98"], "")])
    assert [row_pool(t, r) for r in t.rows] == ["shares", "shares", "shares", "ratio"]


def test_shares_used_to_compute_per_share_is_a_share_count():
    t = _table([("Shares used to compute basic net income per share", ["426", "449"], ""),
                ("Basic net income per share", ["$16.70", "$12.43"], ""),
                ("Average price paid per share", ["412.50", "398.10"], "")])
    assert [row_pool(t, r) for r in t.rows] == ["shares", "ratio", "ratio"]


def test_bare_diluted_under_empty_section_inherits_the_row_above():
    t = _table([("Net earnings per share (Note Basic", ["$2.13", "$2.06"], ""),
                ("Diluted", ["$2.12", "$2.05"], ""),
                ("Average shares outstanding (Notes 1 and Basic", ["2,407.4", "2,410.6"], ""),
                ("Diluted", ["2,429.4", "2,434.2"], "")])
    assert row_section(t, t.rows[3]).startswith("Average shares outstanding")
    assert [row_pool(t, r) for r in t.rows] == ["ratio", "ratio", "shares", "shares"]


def test_kind_row_under_a_folded_heading_ignores_the_stale_section():
    sec = "Net earnings per share (Notes 1 and 15)"
    t = _table([("Total net earnings per share - diluted", ["$5.79", "$13.72"], sec),
                ("Average shares outstanding (Notes 1 and Basic", ["2,407.3", "2,426.5"], sec),
                ("Diluted", ["2,429.4", "2,663.9"], sec)])
    assert [row_pool(t, r) for r in t.rows] == ["ratio", "shares", "shares"]


def test_equity_lines_mentioning_shares_stay_money():
    t = _table([("Common stock, par value $1.00 per share (authorized 4,320,000,000 shares)", ["3,120", "3,120"], ""),
                ("Shares repurchased for tax withholdings", ["(1,234)", "(987)"], "Financing activities"),
                ("Shares authorized: 3,500,000,000; Shares issued: 2,038,893,979; Shares outstanding: 1,589",
                 ["20", "20"], "")])
    assert [row_pool(t, r) for r in t.rows] == ["money", "money", "money"]


def test_kind_row_with_a_parenthetical_inherits_its_heading():
    t = _table([("Basic", ["155.0", "153.4"], "Weighted-average common shares outstanding"),
                ("Diluted (including Subco Units)", ["160.9", "155.9"], "Weighted-average common shares outstanding")])
    assert [row_pool(t, r) for r in t.rows] == ["shares", "shares"]


# ---------------------------------------------------------------- B: wrapped statement titles

def _lines(*texts):
    return [[Word(60 + 40 * j, 20.0 * i, 95 + 40 * j, 20.0 * i + 8, w) for j, w in enumerate(s.split())]
            for i, s in enumerate(texts)]


def test_title_wrapped_one_word_early_is_recognised():
    lines = _lines("Johnson & Johnson and subsidiaries consolidated statements", "of earnings",
                   "Sales to customers 88,821 85,159", "Cost of products sold 27,471 26,553")
    regions = split_regions(lines)
    assert [k for k, _title, _ in regions] == ["IS"]


def test_wrapped_comprehensive_income_title_is_not_an_income_statement():
    lines = _lines("Consolidated statements", "of comprehensive income", "Net earnings 14,066 35,153")
    assert [k for k, _t, _ in split_regions(lines)] == []


def test_line_with_values_is_never_the_second_half_of_a_title():
    lines = _lines("Notes to the consolidated statements", "of earnings 1,234 5,678")
    assert split_regions(lines) == []


# ---------------------------------------------------------------- C: 10-Q columns headed by a year

def test_10q_bare_year_takes_the_filing_period_end_and_the_title_length():
    col = NS(end_date=None, year=2025, months=None)
    got = column_period(CAL, col, "10-Q", filing_end="2025-09-30", heading="Three Months Ended September 30")
    assert got == ("FY2025Q3", "2025-09-30")
    prior = column_period(CAL, NS(end_date=None, year=2024, months=None), "10-Q", filing_end="2025-09-30",
                          heading="Nine Months Ended September 30")
    assert prior == ("FY2024Q3YTD", "2024-09-30")


def test_10q_bare_year_with_no_printed_length_gets_no_period():
    col = NS(end_date=None, year=2025, months=None)
    assert column_period(CAL, col, "10-Q", filing_end="2025-09-30", heading="Revenues by geography") == (None, None)
    assert column_period(CAL, col, "10-Q", filing_end="2025-09-30",
                         heading="Three and Nine Months Ended September 30") == (None, None)


def test_no_period_ends_after_the_filing():
    col = NS(end_date=None, year=2026, months=None)
    assert column_period(CAL, col, "10-K", filing_end="2025-12-31") == (None, None)
    dated = NS(end_date="2026-03-31", year=2026, months=3)
    assert column_period(CAL, dated, "10-Q", filing_end="2025-09-30") == (None, None)


def test_10k_year_columns_are_unchanged():
    col = NS(end_date=None, year=2024, months=None)
    assert column_period(CAL, col, "10-K", filing_end="2025-12-31") == ("FY2024", "2024-12-31")
    assert column_period(CAL, col, "10-K") == ("FY2024", "2024-12-31")


# ---------------------------------------------------------------- D: label objections

def test_other_lines_are_not_named_concepts():
    assert label_objections("IntangibleAssets", "Other non-current assets")
    assert label_objections("IntangibleAssets", "Other assets, including intangibles - net")
    assert not label_objections("IntangibleAssets", "Other intangible assets, net")


def test_partial_totals_are_not_overall_totals():
    assert label_objections("TotalLiabilities", "Total non-current liabilities")
    assert not label_objections("TotalLiabilities", "Total liabilities")
    assert not label_objections("TotalCurrentLiabilities", "Total current liabilities")
    assert label_objections("OperatingIncome", "Income from continuing operations")
    assert not label_objections("OperatingIncome", "Operating income from continuing operations")


def test_line_at_cost_is_not_the_net_concept_beside_a_net_line():
    sibs = ["Less accumulated depreciation", "Property, plant and equipment, net"]
    assert label_objections("PropertyPlantEquipment", "Property, plant and equipment at cost", sibs)
    assert not label_objections("PropertyPlantEquipment", "Property, plant and equipment at cost", [])
    assert not label_objections("PropertyPlantEquipment",
                                "Property, plant and equipment, at cost, less accumulated depreciation", sibs)


def test_label_with_no_seed_word_is_unrelated_but_no_label_is_not():
    assert label_unrelated("IntangibleAssets", "Deferred tax assets on leases") is False   # shares "assets"
    assert label_unrelated("Goodwill", "Prepaid expenses")
    assert not label_unrelated("Goodwill", "")
    assert not label_unrelated("NetIncome", "Net income attributable to the company")


# ---------------------------------------------------------------- E: sentence dollars without a scale word

def _mention(sentence, text, value):
    return {"sentence": sentence, "numbers": [{"slot": "<n1>", "text": text, "value": value, "unit": "USD"}]}


def test_unscaled_dollars_take_the_scale_that_ties_to_a_table():
    m = _mention("Long-term debt with fixed interest rates was $6,175.", "$6,175", 6175)
    settle_scale(m, {6_175_000_000, 12}, 1e6)
    assert m["numbers"][0]["value"] == 6_175_000_000 and m["numbers"][0]["rescaled"] == 1e6


def test_unscaled_dollars_with_no_tie_are_marked():
    m = _mention("The fee was $8,511 for the year.", "$8,511", 8511)
    settle_scale(m, {1}, 1e6)
    assert m["numbers"][0]["unit"] == SCALE_UNKNOWN and m["numbers"][0]["scale_unknown"]


def test_dollars_with_a_scale_word_or_a_whole_dollar_tie_are_left_alone():
    m = _mention("Revenue was $6,175 million.", "$6,175", 6175)
    settle_scale(m, set(), 1e6)
    assert m["numbers"][0]["unit"] == "USD" and m["numbers"][0]["value"] == 6175
    m = _mention("The retainer is $25,000 a year.", "$25,000", 25000)
    settle_scale(m, {25000}, 1e6)
    assert m["numbers"][0]["unit"] == "USD" and "rescaled" not in m["numbers"][0]


# ---------------------------------------------------------------- H: one-line tables

def _pline(y, *parts, cw=5.0):
    return [Word(x, y, x + cw * len(s), y + 8, s) for x, s in parts]


def _page(lines):
    return {"w": 612, "h": 792, "words": [w for ln in lines for w in ln]}


def test_one_line_under_a_year_header_is_a_table():
    from tieout.alltables import page_tables
    lines = [_pline(100, (60, "The"), (80, "following"), (130, "table"), (160, "presents"), (205, "transactions:")),
             _pline(130, (300, "2026"), (360, "2025"), (420, "2024")),
             _pline(142, (60, "(in"), (75, "millions)")),
             _pline(154, (60, "Processed"), (110, "transactions"), (290, "281,331"), (350, "257,545"), (410, "233,758")),
             _pline(220, (60, "Figures"), (100, "may"), (120, "not"), (140, "recalculate."))]
    ts = page_tables("T_FY2026_10K", 3, _page(lines), 1e6)
    assert len(ts) == 1 and [r.label for r in ts[0].rows] == ["Processed transactions"]
    assert [c.year for c in ts[0].columns] == [2026, 2025, 2024]


def test_one_line_with_no_year_header_is_not_a_table():
    from tieout.alltables import page_tables
    lines = [_pline(100, (60, "Interest"), (100, "paid"), (290, "1,748"), (350, "1,585"), (410, "1,401")),
             _pline(160, (60, "Some"), (90, "sentence"), (140, "follows"), (180, "here."))]
    assert page_tables("T_FY2026_10K", 3, _page(lines), 1e6) == []


# ---------------------------------------------------------------- G: members of category columns

def test_member_is_the_header_without_date_words():
    assert member_of("Global Banking | 2025") == "Global Banking"
    assert member_of("September 30, 2025") == ""
    assert member_of("Three Months Ended | 2025") == ""
    assert member_of("At Prior Year Rates | 2025") == "Prior Year Rates"


def test_cells_sharing_a_period_carry_their_member():
    t = _table([("Net sales", ["1,000", "2,000", "3,000"], "")], ncols=3)
    for c, h in zip(t.columns, ["United States | 2025", "International | 2025", "Total | 2025"]):
        c.header = h
    periods = [("FY2025", "2025-12-31")] * 3
    row, = table_rows("T", "TT", 0, t, periods, {}, CAL, "10-K")
    assert [c["member"] for c in row["cells"]] == ["United States", "International", "Total"]
    assert [c["pid"] for c in row["cells"]] == ["FY2025"] * 3
    t2 = _table([("Net sales", ["1,000", "2,000"], "")])
    row2, = table_rows("T", "TT", 0, t2, [("FY2025", "2025-12-31"), ("FY2024", "2024-12-31")], {}, CAL, "10-K")
    assert all("member" not in c for c in row2["cells"])
