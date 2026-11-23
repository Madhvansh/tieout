"""Text-fact rules on synthetic word boxes: no PDF needed. Filing names appear only in comments."""
from tieout.tables import Word, group_lines
from tieout.textfacts import (Collector, Flow, find_gutter, rule_employees, rule_exchange_symbol,
                              rule_fiscal_year_end_report, rule_hq_back_page, rule_listing_sentence,
                              rule_officers, rule_public_float, rule_shares_dated, rule_shares_outstanding)


def _words(rows) -> list:
    """Word boxes for rows of text: a string starts at x=50; a list of (x, string) places
    each string at its own x (columns)."""
    words = []
    for i, row in enumerate(rows):
        y = 100 + 12 * i
        for x, text in ([(50.0, row)] if isinstance(row, str) else row):
            for tok in text.split(" "):
                words.append(Word(x, y, x + 5 * len(tok), y + 8, tok))
                x += 5 * len(tok) + 3
    return words


def _flow(rows, page=1) -> Flow:
    return Flow(page, group_lines(_words(rows)))


def _facts(c: Collector, attr: str) -> list:
    return [f for f in c.ordered() if f.attr == attr]


# ---------------------------------------------------------------- share classes

COVER_META = [
    "Securities registered pursuant to Section 12(b) of the Act:",
    "Class A Common Stock, $0.000006 par value META Nasdaq Global Select Market",
    "Indicate by check mark whether the registrant has filed all reports",
    "On January 23, 2026, the registrant had 2,187,177,748 shares of Class A common stock "
    "and 342,377,716 shares of Class B common stock outstanding.",
]


def test_each_cover_class_is_a_fact_with_its_class():
    c = Collector("D")
    flow = _flow(COVER_META)
    rule_exchange_symbol(c, flow)
    rule_shares_outstanding(c, [flow])
    got = {f.share_class: (int(f.value), f.primary) for f in _facts(c, "shares_outstanding")}
    assert got == {"Class A": (2_187_177_748, False), "Class B": (342_377_716, False),
                   "All classes": (2_529_555_464, True)}


def test_plain_shares_outstanding_is_all_classes_and_comes_last():
    c = Collector("D")
    flow = _flow(COVER_META)
    rule_exchange_symbol(c, flow)
    rule_shares_outstanding(c, [flow])
    last = _facts(c, "shares_outstanding")[-1]      # a reader keeping the last fact gets all classes
    assert (last.share_class, last.value, last.primary, last.derived) == ("All classes", "2529555464", True, True)
    assert last.quote.startswith("All classes (sum of Class A, Class B): ")


def test_printed_total_under_class_rows_is_the_all_classes_fact():
    rows = ["As of July 8, 2026, the number of shares of the Registrant's Common Stock outstanding were:",
            "Class A 281,387,752", "Class B 1,202,110,951", "1,483,498,703",
            "DOCUMENTS INCORPORATED BY REFERENCE:"]
    c = Collector("D")
    rule_shares_outstanding(c, [_flow(rows)])
    primary = [f for f in _facts(c, "shares_outstanding") if f.primary]
    assert [(f.share_class, f.value, f.derived) for f in primary] == [("All classes", "1483498703", False)]
    assert {f.share_class for f in _facts(c, "shares_outstanding")} == {"Class A", "Class B", "All classes"}


def test_the_listed_class_stays_selectable_by_class():
    # a company whose Class B is the traded one: symbol row names Class B
    rows = ["Securities registered pursuant to Section 12(b) of the Act:",
            "Class B Common Stock NKE New York Stock Exchange",
            "Indicate by check mark whether the registrant has filed all reports",
            "As of July 8, 2026, there were 281,387,752 shares of Class A common stock and "
            "1,202,110,951 shares of Class B common stock outstanding."]
    c = Collector("D")
    flow = _flow(rows)
    rule_exchange_symbol(c, flow)
    rule_shares_outstanding(c, [flow])
    facts = _facts(c, "shares_outstanding")
    primary = [f for f in facts if f.primary]
    assert [(f.share_class, f.value) for f in primary] == [("All classes", "1483498703")]
    assert ("Class B", "1202110951") in [(f.share_class, f.value) for f in facts if not f.primary]
    assert _facts(c, "trading_symbol")[0].share_class == "Class B"


def test_several_listed_classes_keep_every_symbol():
    rows = ["Securities registered pursuant to Section 12(b) of the Act:",
            "Class A Common Stock, $0.001 par value GOOGL Nasdaq Stock Market LLC",
            "Class C Capital Stock, $0.001 par value GOOG Nasdaq Stock Market LLC",
            "Indicate by check mark whether the registrant has filed all reports"]
    c = Collector("D")
    rule_exchange_symbol(c, _flow(rows))
    syms = _facts(c, "trading_symbol")
    assert [(f.value, f.share_class, f.primary) for f in syms] == [("GOOG", "Class C", False), ("GOOGL", "Class A", True)]
    assert [f.value for f in _facts(c, "exchange")] == ["NASDAQ"]


def test_single_unlettered_class_carries_no_class():
    rows = ["Securities registered pursuant to Section 12(b) of the Act:",
            "Common Stock, $0.01 par value MCD New York Stock Exchange",
            "Indicate by check mark whether the registrant has filed all reports",
            "The number of shares outstanding of the registrant's common stock as of January 31, 2026: 710,398,642."]
    c = Collector("D")
    flow = _flow(rows)
    rule_exchange_symbol(c, flow)
    rule_shares_outstanding(c, [flow])
    (sym,), (shares,) = _facts(c, "trading_symbol"), _facts(c, "shares_outstanding")
    assert sym.share_class is None and shares.share_class is None and shares.primary
    assert shares.value == "710398642" and not shares.quote.startswith("Common")


def test_a_second_exchange_for_the_same_symbol_is_not_a_second_class():
    rows = ["Securities registered pursuant to Section 12(b) of the Act:",
            "Common Stock, $1 par value XYZ New York Stock Exchange",
            "Common Stock, $1 par value XYZ NYSE Chicago",
            "Indicate by check mark whether the registrant has filed all reports"]
    c = Collector("D")
    rule_exchange_symbol(c, _flow(rows))
    assert [(f.value, f.share_class) for f in _facts(c, "trading_symbol")] == [("XYZ", None)]


# ---------------------------------------------------------------- headcount

def _row(label, cells, title="", unit="count", page=10):
    return {"unit": unit, "label": label, "title": title, "page": page,
            "cells": [{"pid": p, "value": v, "text": t, "bbox": [1.0, 2.0, 3.0, 4.0]} for p, v, t in cells]}


def test_headcount_table_in_thousands_gives_full_units():
    # printed "155" under "Worldwide employment (in thousands)"; the table layer scales it
    rows = [_row("GMNA", [("FY2025", 119000, "119")], "Worldwide employment"),
            _row("Total Worldwide", [("FY2025", 155000, "155")], "Worldwide employment")]
    c = Collector("D")
    rule_employees(c, [], rows, [], [])
    (f,) = _facts(c, "employees")
    assert (f.value, f.page, f.confidence, f.bbox) == ("155000", 10, "high", (1.0, 2.0, 3.0, 4.0))
    assert "Total Worldwide 155" in f.quote


def test_headcount_table_takes_newest_period_of_the_company_row():
    rows = [_row("Total Company", [("FY2024", 171000, "171"), ("FY2025", 169000, "169")],
                 "The approximate number of individuals employed by us as of December 31"),
            _row("United States", [("FY2025", 87000, "87")], "The approximate number of individuals employed by us")]
    c = Collector("D")
    rule_employees(c, [], rows, [], [])
    assert [f.value for f in _facts(c, "employees")] == ["169000"]


def test_money_rows_about_employees_are_not_a_headcount():
    rows = [_row("Employee benefit plans", [("FY2025", 1_500_000, "1,500")], unit="USD"),
            _row("Employees", [("FY2025", 12, "12")])]          # below any real company
    c = Collector("D")
    rule_employees(c, [], rows, [], [])
    assert _facts(c, "employees") == []


SENTENCE = ["As of December 31, 2025, we had approximately 169,000 employees worldwide."]


def test_table_confirming_the_sentence_makes_it_high_and_a_conflict_makes_it_low():
    agree = [_row("Total employees", [("FY2025", 169000, "169")])]
    clash = [_row("Total employees", [("FY2025", 150000, "150")])]
    c1, c2 = Collector("D"), Collector("D")
    rule_employees(c1, [_flow(SENTENCE)], agree, [], [])
    rule_employees(c2, [_flow(SENTENCE)], clash, [], [])
    assert [(f.value, f.confidence) for f in _facts(c1, "employees")] == [("169000", "high")]
    assert [(f.value, f.confidence) for f in _facts(c2, "employees")] == [("169000", "low")]


def test_headcount_falls_back_to_the_sentence_layer():
    page = {"w": 612.0, "h": 792.0, "words": _words(["Employees Over 150,000 people are employed by the Company."])}
    sentence = "Employees Over 150,000 people are employed by the Company."
    mentions = [{"page": 1, "sentence": sentence,
                 "numbers": [{"text": "150,000", "value": 150000, "unit": "count"}]}]
    c = Collector("D")
    rule_employees(c, [], [], mentions, [page])
    (f,) = _facts(c, "employees")
    assert (f.value, f.confidence, f.page) == ("150000", "low", 1) and f.bbox is not None


def test_sentence_layer_skips_subsets_of_the_workforce():
    page = {"w": 612.0, "h": 792.0, "words": _words(["x"])}
    mentions = [{"page": 1, "sentence": "Nearly 90,000 employees received an award.",
                 "numbers": [{"text": "90,000", "value": 90000, "unit": "count"}]}]
    c = Collector("D")
    rule_employees(c, [], [], mentions, [page])
    assert _facts(c, "employees") == []


# ---------------------------------------------------------------- filings without a cover table

def test_listing_sentence_gives_exchange_and_symbol():
    c = Collector("D")
    rule_listing_sentence(c, [_flow(["Note: Citigroup's common stock is listed on the NYSE under the ticker symbol “C” "
                                     "and held by 55,540 registered common stockholders of record."])])
    assert [(f.attr, f.value) for f in c.facts] == [("exchange", "NYSE"), ("trading_symbol", "C")]
    assert "listed on the NYSE" in c.facts[0].quote and c.facts[0].quote.startswith("Citigroup")


def test_listing_sentence_with_colon_form():
    c = Collector("D")
    rule_listing_sentence(c, [_flow(["Wells is listed and trades on the New York Stock Exchange: WFC. At February 13, 2026"])])
    assert [(f.attr, f.value) for f in c.facts] == [("exchange", "NYSE"), ("trading_symbol", "WFC")]


def test_dated_share_count_in_stockholder_information():
    c = Collector("D")
    rule_shares_dated(c, [_flow(["3,092,618,850 common shares outstanding (12/31/25)"])])
    (f,) = c.facts
    assert (f.attr, f.value, f.confidence) == ("shares_outstanding", "3092618850", "high")


def test_report_year_from_the_form_10k_mention():
    c = Collector("D")
    rule_fiscal_year_end_report(c, [_flow(["in our Annual Report on Form 10-K for the year ended December 31, 2025 (2025 Form 10-K)."])])
    (f,) = c.facts
    assert (f.value, f.confidence) == ("2025-12-31", "low")


def test_back_cover_address_needs_exactly_one_address_line():
    one = {"w": 612.0, "h": 792.0, "words": _words(["Wells Fargo & Company", "333 Market Street, San Francisco, CA 94105"])}
    two = {"w": 612.0, "h": 792.0, "words": _words(["333 Market Street, San Francisco, CA 94105", "30 Hudson Yards, New York, NY 10001"])}
    c, d = Collector("D"), Collector("D")
    rule_hq_back_page(c, one, 200)
    rule_hq_back_page(d, two, 200)
    assert [(f.attr, f.value, f.confidence) for f in c.facts] == [("hq_city", "San Francisco", "low"), ("hq_state", "CA", "low")]
    assert d.facts == []


def _grid_page():
    return {"w": 612.0, "h": 792.0, "words": _words([
        [(63, "Bridget Engle"), (247, "Scott E. Powell"), (421, "Charles W. Scharf")],
        [(63, "Senior EVP"), (247, "Senior EVP"), (421, "Chairman and Chief Executive Officer")],
        [(63, "Head of Technology"), (247, "Chief Operating Officer")],
        [(247, "Michael P. Santomassimo")],
        [(247, "Senior EVP")],
        [(247, "Chief Financial Officer")]])}


def test_officers_from_a_name_over_title_grid_when_there_is_no_signature_page():
    c = Collector("D")
    rule_officers(c, [_grid_page()], 0)
    got = {f.attr: (f.value, f.confidence) for f in c.facts}
    assert got == {"ceo": ("Charles W. Scharf", "low"), "cfo": ("Michael P. Santomassimo", "low")}


# ---------------------------------------------------------------- float and columns

def test_public_float_printed_without_a_dollar_sign():
    c = Collector("D")
    rule_public_float(c, [_flow(["The aggregate market value of the registrant's common stock held by "
                                 "non-affiliates as of June 30, 2025: 208,464,334,129."])])
    (f,) = c.facts
    assert (f.value, f.confidence) == ("208464334129", "high")


def test_dollar_amount_wins_over_a_bare_figure():
    c = Collector("D")
    rule_public_float(c, [_flow(["The aggregate market value held by non-affiliates on June 30, 2025, "
                                 "using 1,500,000,000 shares, was approximately $156.2 billion."])])
    assert [f.value for f in c.facts] == ["156200000000"]


def test_two_columns_split_even_with_a_few_full_width_footer_lines():
    rows = []
    for i in range(40):
        y = 100 + 12 * i
        rows += [Word(60 + 30 * k, y, 85 + 30 * k, y + 8, "word") for k in range(8)]
        rows += [Word(332 + 30 * k, y, 357 + 30 * k, y + 8, "text") for k in range(8)]
    for i in range(5):                                  # footer words that run across the gutter
        y = 600 + 12 * i
        rows.append(Word(234, y, 380, y + 8, "Shareholder-services-footer"))
    assert find_gutter(rows, 612.0) is not None
