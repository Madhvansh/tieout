"""The second road (tieout/passage.py, readers.second_road): a passage reader as a witness of another kind.
It may fill an abstention, break a tie between unwitnessed readings, or corroborate a FLAGGED answer.
It never overrides an answer the plan road served, and never touches VERIFIED or WITNESSED."""
import pytest

from tieout import passage, readers
from tieout.execute import Result
from tieout.witness import AGREE, ARBITER_ALONE, Panel

COMPANIES = {"GM": {"name": "General Motors Company"}, "F": {"name": "Ford Motor Company"}}


class G:
    companies = COMPANIES


def reply(value, unit="USD", abstain=False, facts_ok=True):
    kind = "abstain" if abstain else ("companies" if isinstance(value, list) else "number")
    return {"abstain": abstain, "answer": str(value), "unit": unit, "facts_ok": facts_ok,
            "pred": {"kind": kind, "value": None if abstain else value},
            "facts": [{"ticker": "GM", "doc": "GM_FY2025_10K", "page": 5, "value": value, "label": "a line", "in_context": True}]}


def seen(role, value, tier="FLAGGED", declined=False):
    res = Result("abstain") if value is None else Result("number", value, "USD")
    res.tier = "" if value is None else tier
    out = {"answer": "ABSTAIN" if value is None else str(value), "result": res, "plan": {"steps": []}, "learned": []}
    return {"role": role, "model": role, "answer": out["answer"], "tier": res.tier or "ABSTAIN", "plan": out["plan"],
            "declined": declined, "result": res, "out": out}


def served(case, who, readers_seen):
    s = next(x for x in readers_seen if x["role"] == who)
    return {**s["out"], "reading": {"case": case, "witnessed": case == AGREE, "readers": readers_seen}}


@pytest.fixture
def road(monkeypatch):
    calls = []

    def set_reply(r):
        def fake(question, model=None, offline=False):
            calls.append(question)
            return r
        monkeypatch.setattr(passage, "read", fake)
        return calls
    return set_reply


PANEL = Panel("maker", "checker", ("arbiter",), passage="reader")


def test_an_abstention_no_planner_declined_is_filled_and_flagged(road):
    road(reply(25400.0, unit="count"))
    rs = [seen("maker", None), seen("checker", None), seen("arbiter", None)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {})
    assert out["result"].ok and out["result"].value == 25400.0 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.PASSAGE_ALONE
    assert "not tied by arithmetic" in out["result"].tier_reasons[0]
    assert out["reading"]["plan_road"]["tier"] == "ABSTAIN"       # what the first road served is kept for the record


def test_an_abstention_the_arbiter_declined_stays_an_abstention(road):
    road(reply(25400.0, unit="count"))
    rs = [seen("maker", None), seen("checker", None), seen("arbiter", None, declined=True)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {})
    assert not out["result"].ok


def test_a_passage_answer_whose_facts_are_not_on_its_pages_is_not_used(road):
    road(reply(25400.0, unit="count", facts_ok=False))
    rs = [seen("maker", None), seen("checker", None)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "maker", rs), PANEL, {})
    assert not out["result"].ok


def test_an_unwitnessed_reading_gives_way_to_the_reader_the_passage_road_repeats(road):
    road(reply(-0.1577, unit="ratio"))
    rs = [seen("maker", None), seen("checker", 0.7768), seen("arbiter", -0.1577)]
    for s in rs:
        s["result"].unit = "ratio"
    out = readers.second_road("q", G, served(ARBITER_ALONE, "checker", rs), PANEL, {})
    assert out["result"].value == -0.1577 and out["result"].tier == "CORROBORATED"
    assert out["reading"]["case"] == readers.READER_AND_PASSAGE


def test_a_flagged_answer_the_passage_reader_repeats_is_corroborated(road):
    road(reply(23545.0))                                # the reader left the page's millions on the figure
    rs = [seen("maker", 23.545e9), seen("checker", 23.545e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {})
    assert out["result"].value == 23.545e9 and out["result"].tier == "CORROBORATED"


def test_a_different_passage_answer_never_replaces_a_served_one(road):
    road(reply(54e6))
    rs = [seen("maker", 14e6), seen("checker", 14e6)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {})
    assert out["result"].value == 14e6 and out["result"].tier == "FLAGGED"
    assert any("second reading" in r for r in out["result"].tier_reasons)


def _arbiter_says(monkeypatch, value):
    def fake(question, g, arbiters, kw):
        s = seen("arbiter", value, tier="FLAGGED")
        return s["out"], "arbiter"
    monkeypatch.setattr(readers, "_arbitrate", fake)


def test_a_verified_answer_the_passage_reader_repeats_or_cannot_read_stands(road, monkeypatch):
    monkeypatch.setattr(readers, "_arbitrate", lambda *a: pytest.fail("no dissent: the arbiter must not be called"))
    for r in (reply(5e9), reply(None, abstain=True), reply(7e9, facts_ok=False)):
        road(r)
        for tier in ("VERIFIED", "WITNESSED"):
            rs = [seen("maker", 5e9, tier=tier), seen("checker", 5e9, tier=tier)]
            out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {})
            assert out["result"].tier == tier and out["result"].value == 5e9


def test_a_dissent_against_a_verified_answer_goes_to_the_arbiter(road, monkeypatch):
    # round 8, W11: both cheap planners read "gross margin on subscription revenue" as the company's gross margin
    road(reply(0.9113, unit="ratio"))

    def case(arbiter_value):
        _arbiter_says(monkeypatch, arbiter_value)
        rs = [seen("maker", 0.8927, tier="VERIFIED"), seen("checker", 0.8927, tier="VERIFIED")]
        return readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {})

    stands = case(0.8927)                               # the arbiter repeats the planners
    assert stands["result"].value == 0.8927 and stands["result"].tier == "VERIFIED"
    turned = case(0.9115)                               # the arbiter repeats the passage reader
    assert turned["result"].value == 0.9115 and turned["result"].tier == "CORROBORATED"
    assert turned["reading"]["case"] == readers.ARBITER_AND_PASSAGE
    split = case(0.7373)                                # three readings
    assert split["result"].value == 0.8927 and split["result"].tier == "FLAGGED"
    assert any("second reading" in r for r in split["result"].tier_reasons)


def test_a_dissent_after_the_arbiter_sided_with_a_planner_changes_nothing(road, monkeypatch):
    road(reply(0.9113, unit="ratio"))
    monkeypatch.setattr(readers, "_arbitrate", lambda *a: pytest.fail("the arbiter has already read this question"))
    rs = [seen("maker", 0.8927, tier="WITNESSED"), seen("checker", 0.5), seen("arbiter", 0.8927, tier="WITNESSED")]
    out = readers.second_road("q", G, served(readers.ARBITER_AND_READER, "maker", rs), PANEL, {})
    assert out["result"].value == 0.8927 and out["result"].tier == "WITNESSED"


def test_two_passage_readings_overrule_the_graphs_line(road, monkeypatch):
    # JPMorgan, round 8 W05: the graph read a per-share row ($52.80); the passage reader read $63.4 billion,
    # and the served plan run on facts read from passages gives $63.4 billion too
    road(reply(63.4e9))
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: {"value": 63.4e9, "leaves": []})
    rs = [seen("maker", 52.8), seen("checker", 52.8)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 63.4e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.PASSAGES_OVERRULE
    assert out["reading"]["plan_road"]["value"] == 52.8


def test_one_passage_reading_alone_does_not_overrule(road, monkeypatch):
    road(reply(54e6))
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: None)
    rs = [seen("maker", 14e6), seen("checker", 14e6)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 14e6 and out["result"].tier == "FLAGGED"


def test_the_plan_on_passage_facts_corroborates_the_lines_of_a_witnessed_reading(road, monkeypatch):
    road(reply(None, abstain=True))                     # the passage reader has no whole answer (two filings, arithmetic)
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: {"value": 3.4e9, "leaves": []})
    rs = [seen("maker", 3.4e9), seen("checker", 3.4e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].tier == "CORROBORATED"
    rs = [seen("maker", None), seen("checker", 1.0), seen("arbiter", 3.4e9)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})
    assert out["result"].tier == "FLAGGED"              # the lines agree, but the reading has one planner only


def test_a_plan_on_passage_facts_fills_an_abstention_the_reader_could_not(road, monkeypatch):
    road(reply(None, abstain=True))
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: {"value": 3.4e9, "leaves": []})
    rs = [seen("maker", None), seen("checker", None)]
    for s in rs:
        s["plan"] = {"steps": [{"id": "s1", "op": "calc", "expr": "1 - 1"}], "answer": "s1"}
    out = readers.second_road("q", G, served(ARBITER_ALONE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 3.4e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.PLAN_ON_PASSAGES


def test_run_plan_does_the_plans_arithmetic_on_passage_facts(monkeypatch):
    figures = {("WMT", 2023): 12.1e9, ("WMT", 2026): 8.7e9}
    monkeypatch.setattr(passage, "read_fact",
                        lambda g, ex, ticker, words, spec, model, offline: (figures[(ticker, spec["fy"])], {"facts": []}))
    change = {"steps": [{"id": "s1", "op": "change", "company": "WMT", "item": "fuel-related expenses",
                         "from": {"fy": 2023}, "to": {"fy": 2026}, "mode": "abs"}], "answer": "s1"}
    assert passage.run_plan(change, G, None)["value"] == pytest.approx(-3.4e9)
    calc = {"steps": [{"id": "s1", "op": "get", "company": "WMT", "item": "x", "period": {"fy": 2023}},
                      {"id": "s2", "op": "get", "company": "WMT", "item": "x", "period": {"fy": 2026}},
                      {"id": "s3", "op": "calc", "expr": "(s2 - s1) / abs(s1)"}], "answer": "s3"}
    assert passage.run_plan(calc, G, None)["value"] == pytest.approx(-3.4 / 12.1)
    ranking = {"steps": [{"id": "s1", "op": "rank", "over": ["WMT"], "by": "Revenue", "period": {"fy": 2026}}], "answer": "s1"}
    assert passage.run_plan(ranking, G, None) is None                       # not one printed line: not run
    metric = {"steps": [{"id": "s1", "op": "get", "company": "WMT", "metric": "gross_margin", "period": {"fy": 2026}}], "answer": "s1"}
    assert passage.run_plan(metric, G, None) is None


def test_a_witnessed_line_two_passage_readings_dispute_is_overruled(road, monkeypatch):
    # round 8, W02: three planners' plans all reached a reserves row in millions of barrels; the passage
    # reader and the same plan on passage facts both read 19.3 billion
    road(reply(19.3e9, unit="count"))
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: {"value": 19.3e9, "leaves": []})
    monkeypatch.setattr(readers, "_arbitrate", lambda *a: pytest.fail("settled by the vote: no arbiter call"))
    rs = [seen("maker", 36.408e9, tier="WITNESSED"), seen("checker", 36.408e9, tier="WITNESSED")]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 19.3e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.PASSAGES_OVERRULE


def test_the_plan_on_passage_facts_does_not_corroborate_against_the_readers_own_dissent(road, monkeypatch):
    # round 8, W05: the plan dropped "firmwide"; the graph and the plan on passage facts both read the
    # channel total (52.8), the passage reader, given the whole question, read the firmwide figure (63.4)
    road(reply(63.4e9))
    monkeypatch.setattr(passage, "run_plan", lambda plan, g, ex, model=None, offline=False: {"value": 52.8e9, "leaves": []})
    rs = [seen("maker", 52.8e9), seen("checker", 52.8e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 52.8e9 and out["result"].tier == "FLAGGED"
    assert any("second reading" in r for r in out["result"].tier_reasons)


def test_a_sentence_rounding_of_a_table_figure_is_the_same_figure_not_a_dissent(road, monkeypatch):
    # round 7, U02: the table prints 6,661 million, a sentence says "$6.7 billion"
    road(reply(6.7e9))
    monkeypatch.setattr(passage, "run_plan", lambda *a, **k: pytest.fail("no dissent: the plan is not rerun"))
    rs = [seen("maker", 6.661e9), seen("checker", 6.661e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 6.661e9 and out["result"].tier == "CORROBORATED"
    assert passage.same(Result("number", 25412.0, "count"), ("n", 25400.0))
    assert not passage.same(Result("number", 182186.0, ""), ("n", 183417.0))     # two head-counts, not a rounding
    assert not passage.same(Result("number", 52.0, ""), ("n", 50.0))             # one significant digit says too little


def test_a_ranking_is_repeated_only_in_full():
    three = Result("companies", [("AXP", 1.0), ("KO", 2.0), ("MA", 3.0)])
    assert not passage.same(three, passage.head(reply(["American Express", "Coca-Cola"]), {**COMPANIES, "AXP": {"name": "American Express"}, "KO": {"name": "Coca-Cola"}}))
    two = Result("companies", [("GM", 1.0), ("F", 2.0)])
    assert passage.same(two, passage.head(reply(["General Motors"]), COMPANIES))  # which of two: the winner


def test_same_figure_across_roads():
    assert passage.same(Result("number", 0.9115, "ratio"), ("n", 0.9113))        # rounded differently
    assert passage.same(Result("number", 0.1332, "ratio"), ("n", 13.32))         # a percentage for a fraction
    assert passage.same(Result("number", 6.429e9, "USD"), ("n", 6429.0))          # the page's scale left on
    assert not passage.same(Result("number", 14e6, "USD"), ("n", 54e6))
    assert not passage.same(Result("number", 1.4742, "ratio"), ("n", 1.4158))
    assert passage.same(Result("companies", [("GM", 1.0), ("F", 2.0)]), passage.head(reply(["General Motors", "Ford"]), COMPANIES))
    assert not passage.same(Result("companies", [("F", 1.0), ("GM", 2.0)]), passage.head(reply(["General Motors", "Ford"]), COMPANIES))
