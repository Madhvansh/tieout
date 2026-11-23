"""Build fixes, handback 4 (Session B, phase 5, 22 Nov night). The three build cases left open by the loss funnel
(HD p37, BAC p35, JPM p70), each in its smallest form, plus the class each one belongs to."""
import calendar as _cal

from tieout.alltables import (INTRODUCES_RE, Line, Region, _bind_label_numbers, _new_columns, _quarter_header,
                              _value_run)
from tieout.mentions import _table_line
from tieout.periods import FiscalCalendar
from tieout.rawlayer import _quarter_word_period
from tieout.tables import Column, Word


def _line(spec, y=0.0):
    """[(x0, x1, text), ...] -> words on one line."""
    return [Word(x0, y, x1, y + 8, t) for x0, x1, t in spec]


def test_a_chain_of_footnote_markers_belongs_to_the_label():
    """HD FY2025 p37: "Average ticket (2) (3)   $90.56 $89.31 $90.07": only the first marker was passed over, the
    second hid the gap before the values, the line read as prose, and the table lost it and the rows below."""
    ln = _bind_label_numbers(_line([(52.9, 89.9, "Average"), (92.7, 116.0, "ticket"), (118.8, 126.7, "(2)"),
                                    (128.5, 136.5, "(3)"), (303.6, 334.1, "$90.56"), (353.8, 384.4, "$89.31"),
                                    (404.1, 434.6, "$90.07")]))
    assert _value_run(ln) == 4
    three = _bind_label_numbers(_line([(52.9, 107.3, "Comparable"), (110.1, 146.2, "average"), (149.0, 172.3, "ticket"),
                                       (175.1, 187.3, "(%"), (190.1, 226.2, "change)"),
                                       (229.0, 236.9, "(1)"), (238.7, 246.7, "(2)"), (248.5, 256.4, "(3)"),
                                       (309.9, 323.8, "1.4"), (356.8, 377.4, "(0.9)"), (407.1, 427.6, "(0.3)")]))
    assert _value_run(three) == 8
    # markers set off by a wide gap are values ("(1)" = -1 in a column), and a lone marker starts no label
    apart = _line([(50, 80, "Other"), (300, 312, "(1)"), (350, 362, "(2)")])
    assert _value_run(apart) == 1


def test_amounts_under_rate_lines_continue_the_table_only_when_adjacent():
    """HD p37: three "% change" lines whose "%" signs end where the amounts below end, then "Customer transactions
    (in millions) 1,601.5": one table. PEP p82: amounts under the tax-rate table after a sentence: the next table."""
    reg = Region(start=0, nnum=3, edges=[324, 375, 425], alt=[336, 386, 436])
    amounts = Line(words=[], kind="num", text="", edges=[334, 384, 435])
    assert not _new_columns(reg, amounts, adjacent=True)
    assert _new_columns(reg, amounts, adjacent=False)


def test_a_sentence_that_introduces_a_table_ends_the_one_above():
    """JNJ p87, PEP p82: the option-activity and taxes-paid tables were glued to the rate tables above them."""
    for s in ("A summary of option activity under the Plan as of December 28, 2025, is presented below:",
              "A summary of income taxes paid in 2025 is as follows:", "The components are shown below:"):
        assert INTRODUCES_RE.search(s), s
    for s in ("Net sales:", "Adjustments to reconcile net income to net cash provided by operating activities:",
              "Amounts recognized in the balance sheet consist of:", "Total revenue"):
        assert not INTRODUCES_RE.search(s), s


def test_a_line_of_quarter_names_is_a_column_header():
    """BAC FY2025 p35: "(In millions, except per share information) Fourth Third Second First Fourth ..." read as
    prose, so the quarterly table lost its header and none of its 400 figures had a period."""
    words = "(In millions, except per share information) Fourth Third Second First Fourth Third Second First".split()
    ln = _line([(20 + 12 * k, 30 + 12 * k, t) for k, t in enumerate(words)])
    assert _quarter_header(ln)
    assert _quarter_header(_line([(200, 230, "First"), (232, 260, "Quarter"), (300, 330, "Second"), (332, 360, "Quarter")]))
    prose = "funding and pre-funding residual risk through our GCLA. First Second Third Fourth".split()
    assert not _quarter_header(_line([(20 + 12 * k, 30 + 12 * k, t) for k, t in enumerate(prose)]))


def test_a_quarter_column_is_dated_as_that_quarter():
    """A column "Fourth" under "2025 Quarters" in a calendar-year 10-K is the fourth quarter of fiscal 2025 (three
    months), not the fiscal year; the column name says "quarter" so the linker keeps it a span of months."""
    cal = FiscalCalendar("BAC", 12, 31, 0)
    heading = "Table 7 Selected Quarterly Financial Data 2025 Quarters 2024 Quarters"
    got = {}
    for name, year in (("Fourth", 2025), ("Third | 2025 Quarters", 2025), ("First", 2024)):
        col = Column(0, 0, 0, header=name, year=year)
        got[name] = _quarter_word_period(cal, col, "2025-12-31", heading)
        assert "quarter" in col.header
    assert got["Fourth"] == ("FY2025Q4", "2025-12-31")
    assert got["Third | 2025 Quarters"] == ("FY2025Q3", "2025-09-30")
    assert got["First"] == ("FY2024Q1", "2024-03-31")
    # a fiscal year ending in June (MSFT-like): the second quarter of fiscal 2026 ends in December 2025
    msft = FiscalCalendar("MSFT", 6, 30, 0)
    col = Column(0, 0, 0, header="Second Quarter", year=2026)
    assert _quarter_word_period(msft, col, "2026-06-30", "") == ("FY2026Q2", "2025-12-31")
    # no "quarter" anywhere, two ordinals, or a printed date: not this rule
    assert _quarter_word_period(cal, Column(0, 0, 0, header="First Lien", year=2025), "2025-12-31", "") is None
    assert _quarter_word_period(cal, Column(0, 0, 0, header="First Second", year=2025), "2025-12-31", "quarter") is None
    assert _quarter_word_period(cal, Column(0, 0, 0, header="Fourth Quarter", year=2025, end_date="2025-12-31"),
                                "2025-12-31", "") is None
    assert _cal.monthrange(2025, 2)[1] == 28                       # (calendar is used for month ends)


def test_the_last_line_of_a_sentence_is_not_a_table_line():
    """JPM p70 footnote (e): "... for the years ended December 31, 2025, / 2024 and 2023, respectively." The short
    last line holds two numbers; dropped as a table line, it took the years that date the figures."""
    assert not _table_line(_line([(320, 340, "2024"), (342, 360, "and"), (362, 385, "2023,"), (387, 440, "respectively.")]))
    assert _table_line(_line([(50, 80, "Total"), (300, 330, "1,234"), (360, 390, "5,678")]))
    assert _table_line(_line([(50, 60, "2025"), (62, 72, "2024")]))


def test_only_a_table_naming_two_quarters_is_quarterly():
    """GS p64, MSFT p32: a buyback table by month whose columns mention "fourth quarter" once is not a quarterly
    table; its columns keep what the column reader gave them."""
    from types import SimpleNamespace
    from tieout.rawlayer import _names_quarters
    quarterly = SimpleNamespace(columns=[Column(i, 0, 0, header=h) for i, h in enumerate(["Fourth", "Third", "Second"])])
    monthly = SimpleNamespace(columns=[Column(0, 0, 0, header="Total Number of Shares Purchased in the Fourth Quarter"),
                                       Column(1, 0, 0, header="Average Price Paid per Share")])
    assert _names_quarters(quarterly)
    assert not _names_quarters(monthly)


def test_share_scale_is_read_only_from_words_about_shares():
    """JNJ p87 "(Shares in Thousands) ... (Dollars in Millions)", XOM p113 "Shares (thousands)": the share count
    takes the share words' scale; CAT p74 "Amounts are in millions of dollars" says nothing about shares."""
    from tieout.rawlayer import SHARE_SCALE_RE
    assert SHARE_SCALE_RE.search("(Shares in Thousands) Shares Price (Dollars in Millions)").group(1) == "Thousands"
    m = SHARE_SCALE_RE.search("Restricted stock and units outstanding Shares Grant-Date (thousands) Fair Value")
    assert m and "thousands" in m.groups()
    assert not SHARE_SCALE_RE.search("Amounts are in millions of dollars.")
    assert not SHARE_SCALE_RE.search("(In millions, except per share data)")
    m = SHARE_SCALE_RE.search("Options Price Term Value (in millions) (in years) (in millions)")    # MA p97
    assert m and "millions" in m.groups()


def test_a_year_list_may_repeat_fiscal():
    """WMT FY2025 p78: "... $9.9 billion, $10.6 billion, and $12.1 billion in fiscal 2025, fiscal 2024, and fiscal
    2023 respectively": the figures had no year. A list of notes due in two years is still no period list."""
    from tieout.mentions import numbers_of, pair_years
    s = ("Total fuel-related expenses were $9.9 billion, $10.6 billion, and $12.1 billion in fiscal 2025, fiscal 2024, "
         "and fiscal 2023 respectively.")
    _, nums = numbers_of(s)
    assert pair_years(s, nums, 2025) == 3
    assert [n.get("pid") for n in nums if n["unit"] == "USD"] == ["FY2025", "FY2024", "FY2023"]
    s2 = "We issued notes due 2030 and 2032 for $500 million and $700 million."
    _, nums2 = numbers_of(s2)
    assert pair_years(s2, nums2, 2025) == 0


def test_any_plural_noun_after_a_bare_number_counts_it():
    """MA FY2025 p17 "we employed approximately 39,800 persons" had no record ("persons" was not in the list);
    a span of time, money or a singular word ending in s is not a count."""
    from tieout.mentions import numbers_of
    _, nums = numbers_of("As of December 31, 2025, we employed approximately 39,800 persons globally.")
    n = [x for x in nums if x["text"] == "39,800"][0]
    assert n["unit"] == "count" and n["noun"] == "persons" and n["noun_listed"] is False
    _, nums = numbers_of("The plan runs for 5 years and 3 months, with 2 gross adjustments.")
    assert all(x["unit"] != "count" for x in nums)
    # red team 17: money nouns, verbs after a standard's number, and a scale word before the noun are no counts
    for s in ("This includes 10.4 non-cash expenses.", "ASC 820 establishes a framework.",
              "We hold 151 net thousands acres."):
        _, nums = numbers_of(s)
        assert all(x["unit"] != "count" for x in nums), s


def test_red_team_b5_build_cases():
    """Red team of handback 4 (train_lab/REDTEAM_B5.md): a "$" glued to a figure is a dollar sign (HD p37 Average
    ticket took the "(in millions)" count scale of the row above); "(000)" in a label is thousands and wholesale units
    are counted (F p53, read as USD billions); a one-line table under a caption that prints the years is a table
    (AXP p123 maturities of certificates of deposit were lost)."""
    from tieout.rawlayer import ROW_COUNT_RE, label_scale
    assert label_scale("Wholesale Units (000)") == (1000.0, False)
    assert ROW_COUNT_RE.search("Wholesale Units (000)")
    assert not ROW_COUNT_RE.search("Wholesale revenue")


def test_streams_never_cut_a_sentence_at_a_word_gap():
    """Red team 15: a full-width paragraph cut at a false gutter lost "billion" from "$19.9 billion" (XOM p64). A line
    may be split into streams only at a wide gap (JPM p70: table on the left, footnotes on the right)."""
    from tieout.mentions import _clean_split
    words = _line([(50, 80, "netted"), (82, 92, "to"), (94, 120, "$19.9"), (123, 150, "billion"), (152, 160, "in")])
    left, right = [words[:3]], [words[3:]]
    assert not _clean_split([words], [left, right])
    wide = _line([(50, 90, "Number"), (92, 140, "branches"), (230, 260, "5,083"), (320, 340, "(e)"), (342, 400, "Firmwide")])
    assert _clean_split([wide], [[wide[:3]], [wide[3:]]])
