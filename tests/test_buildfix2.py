"""Buildfix 2: scales stated in row labels, physical units, and figures listed for several years."""
from tests.test_raw import make_table, rows_of
from tieout.mentions import numbers_of, page_tables, pair_years, table_scale
from tieout.rawlayer import label_scale, physical


def _values(row):
    return [c["value"] for c in row["cells"]]


# ---------------------------------------------------------------- rule 1: a scale in a row label

def test_label_scale_forms():
    assert label_scale("Cards-in-force (millions)") == (1e6, False)
    assert label_scale("Network volumes (billions)") == (1e9, False)
    assert label_scale("Payments Volume ($B)") == (1e9, True)
    assert label_scale("Total volume ($ in billions)") == (1e9, True)
    assert label_scale("Average Card Member spending (dollars)") == (1.0, True)
    assert label_scale("Earnings $MM") == (1e6, True)
    assert label_scale("Net interest margin (% of assets)") is None
    assert label_scale("Proprietary cards-in-force") is None


def test_axp_card_table_label_scale_carries_to_the_rows_below():
    """AXP FY2025 p51, Table 5: "Proprietary cards-in-force 86.6" is 86.6 million cards, not 86.6x."""
    t = make_table([("Network volumes (billions)", ["1,897.0", "1,764.8"]),
                    ("Billed business", ["1,669.8", "1,550.9"]),
                    ("Cards-in-force (millions)", ["152.8", "146.5"]),
                    ("Proprietary cards-in-force", ["86.6", "83.6"]),
                    ("Basic cards-in-force (millions)", ["128.9", "123.3"]),
                    ("Proprietary basic cards-in-force", ["66.7", "64.3"]),
                    ("Average proprietary basic Card Member spending (dollars)", ["25,453", "24,608"])],
                   dollar_rows=("Network volumes (billions)", "Billed business",
                                "Average proprietary basic Card Member spending (dollars)"))
    out = {r["label"]: r for r in rows_of(t)}
    assert out["Network volumes (billions)"]["unit"] == "USD"
    assert _values(out["Network volumes (billions)"]) == [1_897_000_000_000, 1_764_800_000_000]
    assert out["Billed business"]["unit"] == "USD" and _values(out["Billed business"])[0] == 1_669_800_000_000
    assert out["Proprietary cards-in-force"]["unit"] == "count"
    assert _values(out["Proprietary cards-in-force"]) == [86_600_000, 83_600_000]
    assert out["Proprietary basic cards-in-force"]["unit"] == "count"
    assert _values(out["Proprietary basic cards-in-force"])[0] == 66_700_000
    spend = out["Average proprietary basic Card Member spending (dollars)"]
    assert spend["unit"] == "USD" and _values(spend) == [25_453, 24_608]   # not times the money scale


def test_label_scale_never_touches_percent_or_per_share_rows():
    t = make_table([("Cards-in-force (millions)", ["152.8", "146.5"]),
                    ("Growth", ["4%", "4%"]),
                    ("Earnings per share", ["11.95", "10.10"])])
    out = {r["label"]: r for r in rows_of(t)}
    assert out["Growth"]["unit"] == "percent" and _values(out["Growth"]) == [4.0, 4.0]
    assert out["Earnings per share"]["unit"] == "USD/share" and _values(out["Earnings per share"]) == [11.95, 10.1]


def test_money_row_under_a_dollars_row_keeps_the_table_scale():
    t = make_table([("Average Card Member spending (dollars)", ["21,215", "20,707"]),
                    ("Total segment assets", ["122,968", "114,228"])],
                   dollar_rows=("Average Card Member spending (dollars)", "Total segment assets"))
    out = {r["label"]: r for r in rows_of(t)}
    assert _values(out["Total segment assets"])[0] == 122_968_000_000


# ---------------------------------------------------------------- rule 2: physical units

def test_physical_forms():
    assert physical("Net Oil-Equivalent Production MBOED") == ("barrels of oil-equivalent per day", 1e3)
    assert physical("Natural Gas Production MMCFD") == ("cubic feet per day", 1e6)
    assert physical("Natural Gas - Billions of cubic feet") == ("cubic feet", 1e9)
    assert physical("Proved Undeveloped Reserves (Millions of BOE)") == ("barrels of oil-equivalent", 1e6)
    assert physical("Liquids Realization $/BBL") is None              # a price, not a quantity
    assert physical("Natural gas, per thousand cubic feet") is None


def test_chevron_production_is_a_count_not_money():
    """CVX FY2025 p43: "Net Oil-Equivalent Production MBOED 1,858" was read as $1.858 billion."""
    t = make_table([("Earnings $MM", ["5,815", "7,602"]),
                    ("Net Oil-Equivalent Production MBOED", ["1,858", "1,599"]),
                    ("Natural Gas Production MMCFD", ["3,099", "2,684"]),
                    ("Liquids Realization $/BBL", ["48.13", "56.24"])], scale_source="caption")
    out = {r["label"]: r for r in rows_of(t)}
    assert out["Earnings $MM"]["unit"] == "USD" and _values(out["Earnings $MM"])[0] == 5_815_000_000
    boe = out["Net Oil-Equivalent Production MBOED"]
    assert boe["unit"] == "count" and _values(boe) == [1_858_000, 1_599_000]
    assert boe["measure"] == "barrels of oil-equivalent per day"
    assert _values(out["Natural Gas Production MMCFD"])[0] == 3_099_000_000
    assert out["Liquids Realization $/BBL"]["unit"] != "count"


def test_group_header_with_a_physical_unit_scales_its_rows():
    t = make_table([("Consolidated Companies", ["27,642", "26,526"]), ("Total Natural Gas", ["29,245", "28,375"])])
    for r in t.rows:
        r.section = "Natural Gas - Billions of cubic feet"
    out = rows_of(t)
    assert all(r["unit"] == "count" for r in out)
    assert _values(out[0])[0] == 27_642_000_000_000


def test_money_rows_with_count_nouns_stay_money():
    t = make_table([("Total deposits", ["9,000", "8,000"]), ("Checking accounts", ["2,536", "41"]),
                    ("Municipal bond vehicles", ["3,367", "1,200"])],
                   caption="(in millions)", scale_source="caption", dollar_rows=("Total deposits",))
    out = rows_of(t)
    assert all(r["unit"] == "USD" for r in out)
    assert _values(out[1])[0] == 2_536_000_000


# ---------------------------------------------------------------- rule 3: figures listed for several years

COSTCO = ("Executive members represented 42,300, 38,700, and 35,400 of total paid members in 2026, 2025, "
          "and 2024.")


def test_year_list_gives_each_figure_its_period_and_counted_noun():
    _, nums = numbers_of(COSTCO)
    assert pair_years(COSTCO, nums, 2026) == 3
    figs = [n for n in nums if n["unit"] != "year"]
    assert [(n["value"], n["unit"], n["pid"]) for n in figs] == \
        [(42300, "count", "FY2026"), (38700, "count", "FY2025"), (35400, "count", "FY2024")]


def test_cued_list_with_respectively():
    s = "Repurchases were $1.2 billion, $1.1 billion and $0.9 billion in 2025, 2024 and 2023, respectively."
    _, nums = numbers_of(s)
    assert pair_years(s, nums, 2025) == 3
    assert [n.get("pid") for n in nums if n["unit"] == "USD"] == ["FY2025", "FY2024", "FY2023"]


def test_maturity_years_are_not_periods():
    s = "The notes of $500 million and $750 million mature in 2030 and 2032."
    _, nums = numbers_of(s)
    assert pair_years(s, nums, 2025) == 0


def test_list_of_names_between_years_and_figures_is_not_a_year_pairing():
    s = ("At the end of 2026, 2025, and 2024 goodwill balances in the Company's U.S., Canadian, and Other "
         "International operations were $953, $26, and $15.")
    _, nums = numbers_of(s)
    assert pair_years(s, nums, 2026) == 0


def test_nike_footnote_with_years_is_running_text_and_dated():
    """NKE FY2026 p35: the footnote has 13 numbers counting the years, which read as a table row before."""
    from tieout.mentions import _table_residue
    s = ("(3) Included in NIKE Brand revenues are sales of Jordan Brand products of $7,034 million, $7,270 million "
         "and $8,701 million in fiscal 2026, 2025 and 2024, respectively, decreasing 3% and 16% on a reported "
         "basis and decreasing 5% and 16% on a currency-neutral basis, for fiscal 2026 and 2025, respectively.")
    assert _table_residue(s) and not _table_residue(s, years=False)
    _, nums = numbers_of(s)
    pair_years(s, nums, 2026)
    assert [n["pid"] for n in nums if n["unit"] == "USD"] == ["FY2026", "FY2025", "FY2024"]


def test_printed_unit_after_a_figure_is_a_cue():
    nums = numbers_of("In 2025, we deployed 46.7 GWh of energy storage products.")[1]
    assert (nums[1]["unit"], nums[1]["value"], nums[1].get("unit_token")) == ("GWh", 46.7, True)
    nums = numbers_of("as described in 12 CFR 217 and Scope 1 GHG emissions")[1]
    assert all(n["unit"] == "number" for n in nums)


def test_listed_figures_take_the_scale_of_the_count_table_above():
    raw = [{"doc": "D", "page": 6, "table_id": "D:p6:x1", "unit": "count",
            "cells": [{"text": "84,100", "value": 84_100_000, "bbox": [0, 310, 1, 322]},
                      {"text": "150,400", "value": 150_400_000, "bbox": [0, 340, 1, 347]}]}]
    _, nums = numbers_of(COSTCO)
    pair_years(COSTCO, nums, 2026)
    m = {"numbers": nums}
    table_scale(m, page_tables(raw, "D")[6], y=401.0)
    first = [n for n in nums if n.get("pid") == "FY2026"][0]
    assert first["value"] == 42_300_000 and first["scale_from"] == "table above"
    m2 = {"numbers": numbers_of(COSTCO)[1]}
    pair_years(COSTCO, m2["numbers"], 2026)
    table_scale(m2, page_tables(raw, "D")[6], y=200.0)              # the table is below the sentence
    assert all("scale_from" not in n for n in m2["numbers"])
