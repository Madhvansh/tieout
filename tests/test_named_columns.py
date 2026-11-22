"""A phrase names a column of a period's group only in the column's own words; an accounting basis
does not. Round 5 Q12 (22 Nov): Exxon's Energy Products earnings "(U.S. GAAP, total, before excluding
identified items)" were read from the U.S. column (2,992 / 2,099 = +42.5%) although both pickers had
chosen the Total (7,423 / 4,033 = +84.1%)."""
from tieout.link import named_columns

SPANS = ["U.S.", "Non-U.S.", "Total (the sum of the others)"]


def test_an_accounting_basis_is_not_a_column():
    assert named_columns(SPANS, "Energy Products earnings (U.S. GAAP, total, before excluding identified items)") == [2]
    assert named_columns(SPANS, "earnings excluding identified items (non-GAAP)") == []


def test_a_region_or_the_total_named_in_the_phrase():
    assert named_columns(SPANS, "U.S. revenue") == [0]
    assert named_columns(SPANS, "total earnings") == [2]
    assert named_columns(["North America", "International", "Total (the sum of the others)"], "North America net revenue") == [0]


def test_nothing_named_leaves_the_choice_to_the_pickers():
    assert named_columns(SPANS, "earnings") == []
    assert len(named_columns(SPANS, "Non-U.S. earnings")) == 2     # ambiguous: the pickers' column decides


def test_sentence_amounts_with_no_printed_scale_are_read_in_the_statements_scale_and_flagged():
    # B's census, 22 Nov: 199 sentence amounts print the statements' millions without saying so
    from tieout.execute import Leaf, Result, trust
    from tieout.link import _unit_fit
    assert _unit_fit("USD_scale_unknown", "USD") > 0
    leaf = Leaf("COST:p26:s2:n1", "COST", "said:gross margin", "FY2026", "COST_FY2026_10K", 26, "s", "$3,501", "unscaled",
                3501e6, witnessed=True)
    tier, reasons = trust(Result("number", 3501e6, "USD", [leaf], [], {}))
    assert tier == "FLAGGED" and "prints no scale" in reasons[0]
