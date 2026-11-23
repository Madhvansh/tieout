"""Mention tests that need no PDF: synthetic sentences and word lines."""
from tieout.mentions import numbers_of, page_mentions, sentences, tie
from tieout.tables import Word


def units(sentence):
    return [(n["text"], n["value"], n["unit"]) for n in numbers_of(sentence)[1]]


def test_dollar_amount_with_scale_word():
    masked, nums = numbers_of("Digital Media ARR grew to $19.20 billion at the end of fiscal 2025, "
                              "representing 11.5% year-over-year growth.")
    assert [(n["value"], n["unit"]) for n in nums] == [(19_200_000_000, "USD"), (2025, "year"), (11.5, "percent")]
    assert masked == ("Digital Media ARR grew to $<n1> billion at the end of fiscal <n2>, "
                      "representing <n3>% year-over-year growth.")


def test_masked_form_has_no_digits():
    masked, _ = numbers_of("On January 23, 2026, the registrant had 2,187,177,748 shares of Class A common stock "
                           "and 342,377,716 shares of Class B common stock outstanding.")
    assert not any(ch.isdigit() for ch in masked.replace("<n1>", "").replace("<n2>", "").replace("<n3>", ""))


def test_counted_nouns():
    assert units("In 2025, we delivered approximately 1.64 million consumer vehicles.")[1] == \
        ("1.64 million", 1_640_000, "count")
    got = units("Costco operated 939, 914, and 890 warehouses worldwide at August 30, 2026.")
    assert [g[2] for g in got[:3]] == ["count"] * 3 and got[3] == ("August 30, 2026", "2026-08-30", "date")
    assert units("resulting in a total store count of 2,359 at February 1, 2026.")[0] == ("2,359", 2359, "count")
    assert units("the registrant had 2,187,177,748 shares of Class A common stock")[0][2] == "shares"


def test_year_is_not_a_count():
    got = units("During fiscal 2026, we repurchased shares of our class A common stock for $20.5 billion.")
    assert got == [("2026", 2026, "year"), ("$20.5 billion", 20_500_000_000, "USD")]


def test_per_share_and_basis_points():
    assert units("Diluted earnings per share were $6.62 per diluted share.")[0][2] == "USD/share"
    assert units("Net interest margin rose 25 basis points.")[0] == ("25 basis points", 0.25, "percent")


def test_identifiers_are_not_numbers():
    assert units("See Item 7 of this Form 10-K and COVID-19 matters.") == [("7", 7, "number")]


def test_sentence_split_keeps_abbreviations():
    s = sentences("Revenue grew 5% in the U.S. market. Costs fell $2 million. Apple Inc. paid $1 billion.")
    assert s == ["Revenue grew 5% in the U.S. market.", "Costs fell $2 million.", "Apple Inc. paid $1 billion."]


def test_ties_exact_and_rounded():
    m = {"numbers": numbers_of("we repurchased shares for $20.5 billion and paid $1,234 million")[1]}
    idx = {"USD": [(20_451_000_000, "cf:r1:c0"), (1_234_000_000, "raw:r2:c0"), (99_000_000_000, "x")]}
    assert tie(m, idx) == ["cf:r1:c0", "raw:r2:c0"]


def _page(lines):
    words = []
    for i, parts in enumerate(lines):
        x = 50.0
        for text in parts.split(" ") if isinstance(parts, str) else []:
            words.append(Word(x, 100 + 12 * i, x + 5 * len(text), 108 + 12 * i, text))
            x += 5 * len(text) + 3
        if isinstance(parts, tuple):
            for x0, text in parts:
                words.append(Word(x0, 100 + 12 * i, x0 + 5 * len(text), 108 + 12 * i, text))
    return {"w": 612.0, "h": 792.0, "words": words}


def test_page_mentions_skip_table_lines():
    page = _page(["Total revenues were $41.5 billion in fiscal 2026, up 10% year over year.",
                  ((50, "Subscription"), (300, "$"), (310, "39,388"), (400, "35,679")),
                  ((50, "Services"), (310, "2,137"), (400, "2,216"))])
    ms = page_mentions("D", "T", 5, page)
    assert len(ms) == 1 and ms[0]["id"] == "D:p5:s0" and ms[0]["page"] == 5
    assert [n["unit"] for n in ms[0]["numbers"]] == ["USD", "year", "percent"]
