"""A reader of the text as witness and pointer (tieout/passage.py, readers.second_road, readers._disputed).
The brief: "Answers must come from the graph, not from plain vector search over text chunks." So the reader
may corroborate or dispute an answer of the graph, and may point at a place whose graph node then answers.
Nothing it writes is ever served: no figure, no name, no ranking."""
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


NODE = {"kind": "sentence", "id": "GM_FY2025_10K:p5:s0:n1", "ticker": "GM", "doc": "GM_FY2025_10K", "page": 5,
        "label": "We employed approximately 25,400 people.", "printed": "25,400", "value": 25400.0, "unit": "count",
        "period": "", "verified": False, "cited": "GM_FY2025_10K page 5"}


def test_an_abstention_is_filled_from_the_node_the_reader_points_at(road, monkeypatch):
    road(reply(25400.0, unit="count"))
    monkeypatch.setattr(passage, "node_of_answer", lambda reply, g=None: NODE)
    rs = [seen("maker", None), seen("checker", None), seen("arbiter", None)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {})
    assert out["result"].ok and out["result"].value == 25400.0 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.POINTED_NODE
    assert out["result"].leaves[0].id == NODE["id"]               # the proof leaf is the graph's node
    assert "the graph's node" in out["result"].tier_reasons[0]
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


def ran(value, unit="USD"):
    """What passage.by_pointing returns: the executor's result on lines the reader pointed at, and the nodes."""
    res = Result("number", value, unit)
    res.tier = "FLAGGED"
    return res, [{"phrase": "a line", "id": "GM_FY2025_10K:p5:x0:r3:c0", "doc": "GM_FY2025_10K", "page": 5}]


def pointing(monkeypatch, result):
    monkeypatch.setattr(passage, "by_pointing", lambda plan, g, ex, model=None, offline=False, missing=None: result)


def test_the_readers_lines_overrule_the_linkers_when_they_give_the_readers_own_figure(road, monkeypatch):
    # JPMorgan, round 8 W05: the linker read a per-share row ($52.80); the text reader read $63.4 billion, and
    # the served plan run on the graph lines the reader points at gives $63.4 billion too. Served from those nodes
    road(reply(63.4e9))
    pointing(monkeypatch, ran(63.4e9))
    rs = [seen("maker", 52.8), seen("checker", 52.8)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 63.4e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.LINES_OVERRULED
    assert out["reading"]["passage"]["pointed"][0]["id"].startswith("GM_FY2025_10K")   # the nodes are on record
    assert out["reading"]["plan_road"]["value"] == 52.8


def test_the_readers_own_figure_alone_does_not_overrule(road, monkeypatch):
    road(reply(54e6))
    pointing(monkeypatch, None)
    rs = [seen("maker", 14e6), seen("checker", 14e6)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 14e6 and out["result"].tier == "FLAGGED"


def test_pointed_lines_corroborate_the_lines_of_a_witnessed_reading(road, monkeypatch):
    road(reply(None, abstain=True))                     # the reader has no whole answer (two filings, arithmetic)
    pointing(monkeypatch, ran(3.4e9))
    rs = [seen("maker", 3.4e9), seen("checker", 3.4e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].tier == "CORROBORATED"
    rs = [seen("maker", None), seen("checker", 1.0), seen("arbiter", 3.4e9)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})
    assert out["result"].tier == "FLAGGED"              # the lines agree, but the reading has one planner only


def test_a_plan_on_pointed_lines_fills_an_abstention(road, monkeypatch):
    road(reply(None, abstain=True))
    pointing(monkeypatch, ran(3.4e9))
    rs = [seen("maker", None), seen("checker", None)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 3.4e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.POINTED_LINES


def test_no_figure_name_or_ranking_of_the_reader_is_ever_served(road, monkeypatch):
    """The brief: "Answers must come from the graph, not from plain vector search over text chunks." With no
    node at the place the reader names, and no plan that runs on pointed lines, an abstention stays one."""
    monkeypatch.setattr(passage, "node_of_answer", lambda reply, g=None: None)
    pointing(monkeypatch, None)
    rs = [seen("maker", None), seen("checker", None), seen("arbiter", None)]
    for said in (25400.0, ["General Motors", "Ford"]):
        road(reply(said))
        out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})
        assert not out["result"].ok
    text = {"abstain": False, "answer": "Deloitte", "unit": "", "facts_ok": True, "pred": {"kind": "text", "value": "Deloitte"},
            "facts": [{"ticker": "GM", "doc": "GM_FY2025_10K", "page": 5, "value": "Deloitte", "in_context": True}]}
    road(text)
    assert not readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})["result"].ok


class FakeStore:
    def __init__(self, rows=(), mentions=()):
        self.rows_by_ticker, self.mentions_by_ticker = {"GM": list(rows)}, {"GM": list(mentions)}
        self.by_id = {x["id"]: x for x in list(rows) + list(mentions)}


def row(label, values, page=5, unit="USD", rid="r0"):
    return {"id": f"GM_FY2025_10K:p{page}:x0:{rid}", "doc": "GM_FY2025_10K", "ticker": "GM", "page": page, "table": 0,
            "title": "", "section": "", "label": label, "unit": unit, "role": "loose", "verified": False,
            "cells": [{"pid": pid, "value": v, "text": f"{v:,}", "ok": False, "col": pid} for pid, v in values]}


def test_the_place_the_reader_names_is_looked_up_in_the_graph(monkeypatch):
    from tieout import link
    rows = [row("Automotive revenue", [("FY2025", 19_918_000_000.0), ("FY2024", 17_000_000_000.0)], rid="r1"),
            row("Shares repurchased", [("FY2025", 12_000_000.0)], unit="shares", rid="r2"),
            row("Operating margin", [("FY2025", 29.8)], unit="percent", rid="r3"),
            row("Automotive revenue", [("FY2025", 19_918_000_000.0)], page=9, rid="r4")]
    sentence = {"id": "GM_FY2025_10K:p5:s0", "doc": "GM_FY2025_10K", "ticker": "GM", "page": 5,
                "sentence": "We employed approximately 25,400 people.", "masked": "We employed approximately <n1> people.",
                "numbers": [{"slot": "<n1>", "text": "25,400", "value": 25400, "unit": "count"}], "ties": []}
    monkeypatch.setattr(link, "store", lambda: FakeStore(rows, [sentence]))
    fact = {"doc": "GM_FY2025_10K", "page": 5, "value": 19918.0, "label": "Automotive revenue"}
    node = passage.node_for(fact, "USD")
    assert node["id"].endswith("r1:c0") and node["value"] == 19_918_000_000.0     # the node's scale, not the reader's
    assert passage.node_for({**fact, "page": 8}, "USD")["page"] == 9             # the page after the cited one
    assert passage.node_for({**fact, "page": 30}, "USD") is None                 # no node near that page: no answer
    assert passage.node_for({**fact, "value": 19919.0}, "USD") is None           # 19,919 is not 19,918
    assert passage.node_for({**fact, "value": 19.9e9}, "USD")["id"].endswith("r1:c0")   # "19.9 billion": it rounds to it
    assert passage.node_for({**fact, "value": 12.0, "label": "employees"}, "count") is None   # 12 against 12 million shares
    assert passage.node_for({**fact, "value": 12.0, "label": "shares repurchased"}, "shares")["id"].endswith("r2:c0")
    assert passage.node_for({**fact, "value": 0.298, "label": "margin"}, "ratio")["id"].endswith("r3:c0")   # 29.8% printed
    assert passage.node_for({**fact, "value": 29.8, "label": "margin"}, "USD") is None   # a rate is not a dollar amount
    people = passage.node_for({**fact, "value": 25400.0, "label": "employees"}, "count")
    assert people["kind"] == "sentence" and people["id"] == "GM_FY2025_10K:p5:s0:n1"
    # an answer that is one printed figure is served as its node; the reader's own arithmetic is not
    one = {"pred": {"kind": "number", "value": 19918000000.0}, "unit": "USD", "facts": [fact]}
    out = passage.as_answer(one)
    assert out["result"].value == 19_918_000_000.0 and out["result"].leaves[0].id.endswith("r1:c0")
    assert out["result"].tier == "FLAGGED" and out["plan"]["lookup"]["node"].endswith("r1:c0")
    summed = {"pred": {"kind": "number", "value": 36918000000.0}, "unit": "USD", "facts": [fact]}
    assert passage.as_answer(summed) is None
    assert passage.as_answer({"pred": {"kind": "companies", "value": ["General Motors"]}, "facts": [fact]}) is None


def test_the_lines_of_a_plan_are_its_single_printed_figures():
    change = {"steps": [{"id": "s1", "op": "change", "company": "WMT", "item": "fuel-related expenses",
                         "from": {"fy": 2023}, "to": {"fy": 2026}, "mode": "abs"}], "answer": "s1"}
    assert [(c, w, s) for _, c, _, w, s in passage.lines_of(change)] == [
        ("WMT", "fuel-related expenses", {"fy": 2023}), ("WMT", "fuel-related expenses", {"fy": 2026})]
    concept = {"steps": [{"id": "s1", "op": "get", "company": "WMT", "concept": "ResearchAndDevelopment", "period": {"fy": 2026}}]}
    assert passage.lines_of(concept)[0][3] == "research and development"       # a concept is asked for by its name in words
    ranking = {"steps": [{"id": "s1", "op": "rank", "over": ["WMT"], "by": "Revenue", "period": {"fy": 2026}}]}
    metric = {"steps": [{"id": "s1", "op": "get", "company": "WMT", "metric": "gross_margin", "period": {"fy": 2026}}]}
    picked = {"steps": [{"id": "s2", "op": "get", "company": "$s1", "item": "x", "period": {"fy": 2026}}]}
    assert passage.lines_of(ranking) == passage.lines_of(metric) == passage.lines_of(picked) == []
    assert passage.by_pointing(ranking, G, None) is None                       # nothing to point at: not run


def test_a_witnessed_line_the_readers_lines_dispute_is_overruled(road, monkeypatch):
    # round 8, W02: three planners' plans all reached a reserves row in millions of barrels; the text reader read
    # 19.3 billion, and the plan on the lines it points at gives 19.3 billion too
    road(reply(19.3e9, unit="count"))
    pointing(monkeypatch, ran(19.3e9, "count"))
    monkeypatch.setattr(readers, "_arbitrate", lambda *a: pytest.fail("settled by the lines: no arbiter call"))
    rs = [seen("maker", 36.408e9, tier="WITNESSED"), seen("checker", 36.408e9, tier="WITNESSED")]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 19.3e9 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.LINES_OVERRULED


def test_pointed_lines_do_not_corroborate_against_the_readers_own_dissent(road, monkeypatch):
    # round 8, W05: the plan dropped "firmwide"; the graph and the plan on pointed lines both read the channel
    # total (52.8), the reader, given the whole question, read the firmwide figure (63.4)
    road(reply(63.4e9))
    pointing(monkeypatch, ran(52.8e9))
    rs = [seen("maker", 52.8e9), seen("checker", 52.8e9)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 52.8e9 and out["result"].tier == "FLAGGED"
    assert any("second reading" in r for r in out["result"].tier_reasons)


def test_a_sentence_rounding_of_a_table_figure_is_the_same_figure_not_a_dissent(road, monkeypatch):
    # round 7, U02: the table prints 6,661 million, a sentence says "$6.7 billion"
    road(reply(6.7e9))
    monkeypatch.setattr(passage, "by_pointing", lambda *a, **k: pytest.fail("no dissent: the plan is not rerun"))
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


def test_the_agreement_test_is_not_looser_than_it_says():
    # each of these was called "the same" before the red team of 23 Nov (research/19_redteam_second_road.md)
    assert not passage.same(Result("number", 0.1149, "ratio"), ("n", 0.11))       # "about 11%" is not 11.49%
    assert not passage.same(Result("number", -0.05, "ratio"), ("n", 0.05))        # a fall is not a rise
    assert not passage.same(Result("number", 12.0, "count"), ("n", 1200.0))       # a count is not a percentage
    assert not passage.same_number(5.0, 500.0)
    assert not passage.same_number(-0.2, 0.2)
    assert passage.same(Result("number", -9.941e9, "USD"), ("n", 9941.0))         # an outflow, the page's millions left on
    assert passage.same_number(0.4894, 0.4885)                                    # two sums of the same printed figures
    assert passage.as_answer({"pred": {"kind": "number", "value": None}}) is None
    assert passage.as_answer({}) is None


def test_same_figure_across_roads():
    assert passage.same(Result("number", 0.9115, "ratio"), ("n", 0.9113))        # rounded differently
    assert passage.same(Result("number", 0.1332, "ratio"), ("n", 13.32))         # a percentage for a fraction
    assert passage.same(Result("number", 6.429e9, "USD"), ("n", 6429.0))          # the page's scale left on
    assert not passage.same(Result("number", 14e6, "USD"), ("n", 54e6))
    assert not passage.same(Result("number", 1.4742, "ratio"), ("n", 1.4158))
    assert passage.same(Result("companies", [("GM", 1.0), ("F", 2.0)]), passage.head(reply(["General Motors", "Ford"]), COMPANIES))
    assert not passage.same(Result("companies", [("F", 1.0), ("GM", 2.0)]), passage.head(reply(["General Motors", "Ford"]), COMPANIES))


def test_a_node_at_a_named_place_outranks_a_planners_belief_that_filings_lack_the_fact(road, monkeypatch):
    # round 10, N05: two planners said no filing reports an employee turnover rate; the reader named the page,
    # and the graph holds the figure there. The node is served, FLAGGED, with the planners' decline on record
    road(reply(0.037, unit="ratio"))
    monkeypatch.setattr(passage, "node_of_answer", lambda reply, g=None: {**NODE, "value": 3.7, "unit": "percent", "printed": "3.7%"})
    rs = [seen("maker", None, declined=True), seen("checker", None), seen("arbiter", None, declined=True)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {})
    assert out["result"].ok and out["result"].value == pytest.approx(0.037) and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.POINTED_NODE
    assert any("planners said the filings do not state this" in r for r in out["result"].tier_reasons)
    monkeypatch.setattr(passage, "node_of_answer", lambda reply, g=None: None)   # no node there: the decline stands
    assert not readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {})["result"].ok


def test_another_planners_plan_is_completed_by_pointing_when_it_gives_the_readers_figure(road, monkeypatch):
    # round 10, N13: the served reading has one planner only; another plan stopped at a line with no value for
    # the year asked; with that line pointed at it gives 13,200, which the reader gives for the whole question
    road(reply(13200.0, unit="count"))
    rs = [seen("maker", None), seen("checker", 65900.0), seen("arbiter", None)]
    rs[0]["result"].missing = [("KO", "item:number of employees", {"fy": 2023})]
    pointing(monkeypatch, ran(13200.0, "count"))
    out = readers.second_road("q", G, served(readers.ONE_READER, "checker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 13200.0 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.POINTED_LINES
    pointing(monkeypatch, ran(70000.0, "count"))        # the completed plan says something else: nothing changes
    out = readers.second_road("q", G, served(readers.ONE_READER, "checker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 65900.0


# ----------------------------------------------- the places the reader cited for the whole question (by_citing)

TWO_LINES = {"steps": [{"op": "get", "company": "GM", "item": "number of dealers", "period": {"fy": 2025}},
                       {"op": "get", "company": "F", "item": "number of dealers", "period": {"fy": 2025}},
                       {"op": "calc", "expr": "s1 + s2"}]}


def cited_node(nid, ticker, value, label="dealers"):
    return {"kind": "row", "id": nid, "row_id": nid, "ticker": ticker, "doc": f"{ticker}_FY2025_10K", "page": 7,
            "label": label, "printed": str(value), "value": float(value), "unit": "count", "period": "FY2025",
            "cited": f"{ticker}_FY2025_10K page 7", "said": label}


def summing(binds_seen):
    """A fake executor run: the plan adds the figures of the nodes bound to its lines."""
    def run_bound(plan, g, ex, binds):
        binds_seen.append([n["id"] for _, n in binds])
        res = Result("number", sum(n["value"] for _, n in binds), "count")
        res.tier = "FLAGGED"
        return res, [{"phrase": s[3], "id": n["id"]} for s, n in binds]
    return run_bound


def test_by_citing_keeps_only_a_binding_that_gives_the_readers_own_figure(monkeypatch):
    nodes = [cited_node("gm-a", "GM", 4566), cited_node("gm-b", "GM", 6276), cited_node("f-a", "F", 2900)]
    monkeypatch.setattr(passage, "cited_nodes", lambda reply, g=None: nodes)
    tried = []
    monkeypatch.setattr(passage, "_run_bound", summing(tried))
    got = passage.by_citing(TWO_LINES, G, object(), {}, ("n", 6276.0 + 2900.0))
    assert got is not None and got[0].value == 9176.0
    assert [n["id"] for n in got[1]] == ["gm-b", "f-a"]            # each line took a node of its own company
    assert all(len(ids) == 2 for ids in tried)                     # every line was bound in every run
    assert passage.by_citing(TWO_LINES, G, object(), {}, ("n", 1234.0)) is None   # a run that merely runs is not served
    assert passage.by_citing(TWO_LINES, G, object(), {}, None) is None             # no figure of the reader: nothing


def test_by_citing_never_binds_two_lines_to_one_node(monkeypatch):
    plan = {"steps": [{"op": "get", "company": "GM", "item": "dealers in GMNA", "period": {"fy": 2025}},
                      {"op": "get", "company": "GM", "item": "dealers in GMI", "period": {"fy": 2025}},
                      {"op": "calc", "expr": "s1 + s2"}]}
    monkeypatch.setattr(passage, "cited_nodes", lambda reply, g=None: [cited_node("gm-a", "GM", 4566)])
    monkeypatch.setattr(passage, "_run_bound", summing([]))
    assert passage.by_citing(plan, G, object(), {}, ("n", 9132.0)) is None


def citing(monkeypatch, result):
    monkeypatch.setattr(passage, "by_citing", lambda plan, g, ex, reply, shown: result)


def test_an_abstention_is_filled_by_a_plan_on_the_places_the_reader_cited(road, monkeypatch):
    road(reply(10842.0, unit="count"))
    monkeypatch.setattr(passage, "node_of_answer", lambda reply, g=None: None)   # the answer is arithmetic, not one figure
    monkeypatch.setattr(passage, "by_pointing", lambda *a, **k: pytest.fail("the cited places settle it: no new call"))
    citing(monkeypatch, ran(10842.0, "count"))
    rs = [seen("maker", None), seen("checker", None), seen("arbiter", None)]
    out = readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})
    assert out["result"].value == 10842.0 and out["result"].tier == "FLAGGED"
    assert out["reading"]["case"] == readers.CITED_LINES
    citing(monkeypatch, None)                           # no binding gives the reader's figure: the old pointer is tried
    pointing(monkeypatch, None)
    assert not readers.second_road("q", G, served(ARBITER_ALONE, "arbiter", rs), PANEL, {"executor": object()})["result"].ok


def test_a_served_answer_is_overruled_by_the_cited_places_only_when_switched_on(road, monkeypatch):
    road(reply(63.4e9))
    pointing(monkeypatch, None)
    citing(monkeypatch, ran(63.4e9))
    rs = [seen("maker", 52.8), seen("checker", 52.8)]
    out = readers.second_road("q", G, served(AGREE, "maker", rs), PANEL, {"executor": object()})
    assert out["result"].value == 52.8                  # off by default: the served answer stands ...
    assert out["reading"]["passage"]["cited_alt"]["value"] == 63.4e9   # ... and what the overrule would serve is on record
    monkeypatch.setattr(readers, "CITED_OVERRULE", True)
    out = readers.second_road("q", G, served(AGREE, "maker", [seen("maker", 52.8), seen("checker", 52.8)]), PANEL,
                              {"executor": object()})
    assert out["result"].value == 63.4e9 and out["reading"]["case"] == readers.CITED_OVERRULED
