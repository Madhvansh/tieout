"""Build fixes, lane 3 (Session B, 22 Nov): classes found by measuring the raw layer against the page text
(`scripts/raw_coverage.py`), not by reading code. Each test is the smallest form of a page that failed."""
from tieout.alltables import _bind_label_numbers, _date_header, _value_run
from tieout.tables import LABEL_MARK, Word, find_columns


def _line(spec, y=0.0):
    """[(x0, x1, text), ...] -> words on one line."""
    return [Word(x0, y, x1, y + 8, t) for x0, x1, t in spec]


def test_change_header_between_years_is_a_header():
    """AXP FY2025 p51 and p56: "Years Ended December 31, 2025 2024 2023 2025 vs. 2024 2024 vs. 2023" read as
    prose, so the card tables lost their years (and p56 its "(Millions ...)" scale)."""
    ln = _line([(19, 35, "Years"), (37, 55, "Ended"), (57, 87, "December"), (89, 98, "31,"), (325, 340, "2025"),
                (386, 401, "2024"), (447, 462, "2023"), (478, 492, "2025"), (494, 502, "vs."), (504, 518, "2024"),
                (543, 557, "2024"), (559, 567, "vs."), (569, 584, "2023")])
    bound = _bind_label_numbers(ln)
    assert not any(LABEL_MARK in w.text for w in bound)            # the years stay years
    assert _date_header(bound)
    # a sentence is still not a header
    prose = _line([(19, 40, "during"), (42, 60, "2025"), (62, 90, "compared"), (92, 100, "to"), (102, 120, "2024")])
    assert not _date_header(_bind_label_numbers(prose))


def test_footnote_marker_glued_to_a_label_is_not_a_value():
    """TGT FY2025 p57: "Beauty (b)   13,214 13,173 12,538" read as prose and the net sales table was cut there."""
    ln = _line([(52.1, 83.2, "Beauty"), (86.0, 94.0, "(b)"), (395.9, 426.4, "13,214"), (461.9, 492.4, "13,173"),
                (527.9, 558.4, "12,538")])
    assert _value_run(ln) == 2
    lone = _line([(50, 58, "(a)"), (300, 320, "1,234")])
    assert _value_run(lone) == 0                                   # a marker that starts a line is unchanged


def test_dash_between_label_words_is_label_text():
    """AXP FY2025 p56: "Net write-off rate — principal only (a) 2.0 % ..." once read as a row, its em dash
    lined up six times and made a column of dashes."""
    ln = _line([(25, 40, "Net"), (42, 70, "write-off"), (72, 85, "rate"), (87, 95, "—"), (97, 130, "principal"),
                (132, 150, "only"), (152, 160, "(a)"), (312, 323, "2.0"), (378, 389, "2.1")])
    bound = _bind_label_numbers(ln)
    assert bound[3].text == "—" + LABEL_MARK
    nil = _line([(25, 60, "Goodwill"), (318, 327, "—"), (380, 394, "1,234")])
    assert _bind_label_numbers(nil)[1].text == "—"                  # a dash in a value column is still a nil


def test_a_column_comparing_two_years_has_no_period():
    """AXP FY2025 p49: once its header line was read, "Change | 2025 vs. 2024" (a change of 353) would have been
    dated 2025. Only the header line nearest the figures is read: a drifted "Change" over a real year column
    (Nike p40, Merck p49, Apple p28) keeps its year."""
    from tieout.rawlayer import _comparison
    for h in ("Change | 2025 vs. 2024", "2025/ | 2025/ 2024", "$ Change 2024 to 2025", "Change | Change 2025 vs.",
              "Increase (Decrease) Due to Change in: Rate | 2024 versus 202"):
        assert _comparison(h), h
    for h in ("$ Change % Change 2025/ 2025/ | 2023", "CHANGE CHANGES", "% CHANGE FISCAL 2024", "Change 2025",
              "At | December 31, December 31, 2024 Change", "2025", "Exchange rates"):
        assert not _comparison(h), h


def test_a_hash_in_the_label_means_a_count():
    """WFC FY2025 p26: "Retail bank branches (#, period-end) 4,090" was read as $4.09 billion."""
    from tieout.rawlayer import ROW_COUNT_RE
    assert ROW_COUNT_RE.search("Retail bank branches (#, period-end)")
    assert ROW_COUNT_RE.search("Active digital banking users (in thousands)")
    assert not ROW_COUNT_RE.search("Deposits (# of days outstanding are not shown)")


def test_rates_printed_with_percent_join_the_amount_column():
    """AXP FY2025 p56: "2.5 %" stops one character short of "$ 114,368" above it. Two clusters a few points
    apart that never meet on one line are one column (only outside the primary statements)."""
    lines = [_line([(25, 60, "Loans"), (307, 327, "114,368"), (374, 394, "107,051")], y=0),
             _line([(25, 60, "Average"), (307, 327, "106,377"), (374, 394, "98,928")], y=10),
             _line([(25, 60, "Rate"), (312, 323, "2.5"), (379, 390, "2.5")], y=20),
             _line([(25, 60, "Rate"), (312, 323, "2.0"), (379, 390, "2.1")], y=30),
             _line([(25, 60, "Due"), (312, 323, "1.3"), (379, 390, "1.3")], y=40),
             _line([(25, 60, "Total"), (307, 327, "100,171"), (374, 394, "92,632")], y=50)]
    assert len(find_columns(lines, min_hits=2)) == 4
    assert len(find_columns(lines, min_hits=2, merge_ragged=True)) == 2
