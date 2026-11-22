"""Tests of the scale test's key and its judging. No PDFs, no graph: a made-up filer and made-up answers."""
from types import SimpleNamespace

from tieout.eval import scale
from tieout.execute import Leaf, Result
from tieout.periods import FiscalCalendar

JUNE = FiscalCalendar("X", 6, 30, 0)          # fiscal 2025 ends 2025-06-30
JANUARY = FiscalCalendar("Y", 1, 31, -1)      # the year ending 2025-02-01 is fiscal 2024
NO_TABLES = SimpleNamespace(readings={}, tables={})


# ---------------------------------------------------------------- the SEC's dates on the graph's calendar

def test_sec_periods_map_to_the_graphs_period_ids():
    assert scale.period_of(JUNE, "2024-07-01", "2025-06-30") == "FY2025"
    assert scale.period_of(JUNE, None, "2025-06-30") == "FY2025@end"
    assert scale.period_of(JUNE, "2025-01-01", "2025-03-31") == "FY2025Q3"
    assert scale.period_of(JUNE, "2024-07-01", "2025-03-31") == "FY2025Q3YTD"
    assert scale.period_of(JUNE, None, "2025-03-31") == "FY2025Q3@end"
    assert scale.period_of(JUNE, "2025-04-01", "2025-06-30") == "FY2025Q4"
    assert scale.period_of(JUNE, "2024-07-01", "2024-12-31") is None          # six months: no key period


def test_a_52_week_year_ending_in_february_takes_the_calendars_label():
    assert scale.period_of(JANUARY, "2024-02-04", "2025-02-01") == "FY2024"
    assert scale.period_of(JANUARY, None, "2025-02-01") == "FY2024@end"


def test_plan_period_specs_and_ids():
    ytd = scale.Period(2025, 3, True)
    assert ytd.spec() == {"fy": 2025, "q": 3, "ytd": True}
    assert (ytd.pid("flow"), ytd.pid("stock")) == ("FY2025Q3YTD", "FY2025Q3@end")
    assert (scale.Period(2025).pid("flow"), scale.Period(2025).pid("stock")) == ("FY2025", "FY2025@end")


def test_a_third_quarter_report_prints_the_quarter_the_nine_months_and_one_balance():
    report = scale.Report("X_FY2025_Q3_10Q", "10-Q", 2025, "2025-03-31")
    ids = scale.period_ids(report, JUNE)
    assert {"FY2025Q3", "FY2024Q3", "FY2025Q3YTD", "FY2024Q3YTD", "FY2025Q3@end"} <= ids
    assert "FY2024Q3@end" not in ids


# ---------------------------------------------------------------- latest and first reported

def _filer() -> scale.CompanyKey:
    """Two annual reports. The older prints fiscal 2023's balance as 100; the newer restated it to 90, but
    only inside its statement of equity, not on a balance sheet. Fiscal 2024's balance was restated 50 to 55."""
    def rec(end, val, accn, filed, start=None):
        return {"end": end, "val": val, "accn": accn, "form": "10-K", "filed": filed, **({"start": start} if start else {})}
    equity = [rec("2023-06-30", 100, "OLD", "2024-08-01"), rec("2023-06-30", 90, "NEW", "2025-08-01"),
              rec("2024-06-30", 50, "OLD", "2024-08-01"), rec("2024-06-30", 55, "NEW", "2025-08-01")]
    durations = [rec("2024-06-30", 1, "OLD", "2024-08-01", "2023-07-01"), rec("2025-06-30", 1, "NEW", "2025-08-01", "2024-07-01")]
    facts = {"facts": {"us-gaap": {"StockholdersEquity": {"units": {"USD": equity}},
                                   "Revenues": {"units": {"USD": durations}}}}}
    reports = [scale.Report("X_FY2025_10K", "10-K", 2025, "2025-06-30"), scale.Report("X_FY2024_10K", "10-K", 2024, "2024-06-30")]
    return scale.CompanyKey(facts, JUNE, reports)


def test_latest_value_comes_from_the_latest_report_that_prints_the_period():
    key, tags = _filer(), ["StockholdersEquity"]
    assert key.latest(tags, "FY2024@end").value == 55       # printed by both: the newer
    assert key.latest(tags, "FY2023@end").value == 100      # printed by the older only; the 90 is in no balance sheet


def test_first_value_comes_from_the_earliest_report_that_prints_the_period():
    assert _filer().first(["StockholdersEquity"], "FY2024@end").value == 50


def test_a_blocked_concept_has_no_key():
    key = _filer()
    key.blocked = frozenset({"ShareholdersEquity"})
    assert scale.Reader(key)("ShareholdersEquity", "FY2024@end") is None
    assert scale.Reader(key)("TotalAssets", "FY2024@end") is None      # no such element in this filer


def test_second_definitions_become_alternative_keys():
    key = _filer()
    value, alts, used = scale.keyed(lambda read: read("ShareholdersEquity", "FY2024@end"), key, ["ShareholdersEquity"])
    assert (value, alts) == (55, []) and used[0]["tag"] == "StockholdersEquity"


# ---------------------------------------------------------------- the formula and the growth key

def test_formula_concepts_and_evaluation():
    values = {"NetIncome": 20.0, "Revenue": 100.0, "Zero": 0.0}
    assert scale.formula_concepts("NetIncome / Revenue") == ["NetIncome", "Revenue"]
    assert scale.evaluate("NetIncome / Revenue", values.get) == 0.2
    assert scale.evaluate("NetIncome / Zero", values.get) is None
    assert scale.evaluate("NetIncome / Missing", values.get) is None
    assert scale.evaluate("abs(NetIncome - Revenue)", values.get) == 80.0


def test_growth_is_relative_to_the_size_of_the_earlier_value():
    assert scale._growth(100, 150) == 0.5
    assert scale._growth(-100, -150) == -0.5          # a loss that deepens
    assert scale._growth(0, 5) is None and scale._growth(None, 5) is None


# ---------------------------------------------------------------- judging

def _question(family="fact", key=12_443_000_000, unit="USD", alts=(), members=None):
    return scale.Question("q", family, "X", "NetIncome", "FY2025", unit, {}, key, list(alts), members=members or {})


def _answer(value, status="verified", printed="12,443", kind="number"):
    leaf = Leaf("r1", "X", "NetIncome", "FY2025", "X_FY2025_10K", 1, "Net income", printed, status, value)
    return Result(kind, value, "USD", [leaf])


def _judged(question, result):
    return scale.judge(question, result, NO_TABLES)


def test_the_value_printed_is_correct():
    assert _judged(_question(), _answer(12_443_000_000))["class"] == "correct"


def test_half_a_unit_of_the_printed_digit_is_the_tolerance():
    assert _judged(_question(key=12_443_400_000), _answer(12_443_000_000))["class"] == "correct"
    assert _judged(_question(key=12_444_000_000), _answer(12_443_000_000))["class"] == "wrong_silent"


def test_an_outflow_printed_negative_is_correct_by_magnitude_and_counted_as_such():
    row = _judged(_question(key=92_000_000), _answer(-92_000_000, printed="(92)"))
    assert row["class"] == "correct" and row["sign_differs"]


def test_a_ratio_with_the_wrong_sign_is_wrong():
    result = Result("number", -0.42, "ratio", [_answer(0).leaves[0]])
    assert _judged(_question("growth", 0.42, "ratio"), result)["class"] == "wrong_silent"


def test_wrong_with_every_cell_verified_is_silent_and_otherwise_unverified():
    assert _judged(_question(), _answer(1_000_000_000))["class"] == "wrong_silent"
    assert _judged(_question(), _answer(1_000_000_000, status="unverified"))["class"] == "wrong_unverified"


def test_a_wrong_answer_equal_to_a_second_definition_stays_wrong_and_is_marked():
    row = _judged(_question(alts=[12_000_000_000]), _answer(12_000_000_000, printed="12,000"))
    assert row["class"] == "wrong_silent" and row["alt_definition"]
    assert not _judged(_question(), _answer(12_000_000_000, printed="12,000"))["alt_definition"]


def test_an_abstention_is_not_wrong():
    row = _judged(_question(), Result("abstain", notes=["no fact"]))
    assert row["class"] == "abstained" and row["why"] == "no fact"


def test_a_text_answer_to_a_number_question_is_wrong():
    assert _judged(_question(), Result("text", "n/a"))["class"] == "wrong_silent"


def test_a_ranking_is_correct_in_the_keys_order_with_ties_either_way():
    members = {"A": 9, "B": 5, "C": 5, "D": 1}
    question = _question("rank", ["A", "B", "C", "D"], "USD", members=members)
    ranked = lambda *names: Result("companies", [(n, members[n]) for n in names], leaves=[_answer(0).leaves[0]])
    assert _judged(question, ranked("A", "B", "C", "D"))["class"] == "correct"
    assert _judged(question, ranked("A", "C", "B", "D"))["class"] == "correct"
    assert _judged(question, ranked("B", "A", "C", "D"))["class"] == "wrong_silent"
    assert _judged(question, ranked("A", "B", "C"))["class"] == "wrong_silent"          # one company missing


def test_a_crash_is_recorded_as_abstained_with_its_error():
    class Crashing:
        def run(self, plan):
            raise KeyError("boom")
    row = scale.run([_question()], Crashing(), NO_TABLES)[0]
    assert row["class"] == "abstained" and "KeyError" in row["error"]


def test_summary_counts_each_class_once():
    rows = [{"class": "correct"}, {"class": "abstained"}, {"class": "wrong_silent", "alt_definition": True},
            {"class": "wrong_unverified", "alt_definition": False}]
    s = scale.stats(rows)
    assert (s["n"], s["correct"], s["abstained"], s["wrong_silent"], s["wrong_unverified"]) == (4, 1, 1, 1, 1)
    assert s["silent_wrong_equal_to_second_definition"] == 1 and s["accuracy"] == 0.25
