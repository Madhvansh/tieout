"""Question side: the plan validator, the answer-type check and the planner's example pool.

No language model is called; every test runs on plans written here.
"""
from tieout.ask import shape_misfit
from tieout.execute import Result
from tieout.graph import Graph
from tieout.plan import validate
from tieout.planner import EXAMPLES, pick_examples

G = Graph.load()


def _plan(*steps):
    return {"steps": [dict(id=f"s{i + 1}", **s) for i, s in enumerate(steps)], "answer": f"s{len(steps)}"}


def test_every_planner_example_is_a_valid_plan():
    for question, plan in EXAMPLES:
        if plan.get("cannot"):
            continue
        assert validate(plan, G) == [], question


def test_examples_are_picked_by_shape_not_by_company():
    shots = pick_examples("How many stores did Target operate in Ohio at the end of fiscal 2025?")
    assert any("restaurants" in q or "stores" in q for q, _ in shots)


def test_a_list_of_companies_is_not_a_company():
    plan = _plan({"op": "change", "company": ["KO", "PEP"], "concept": "NetIncome",
                  "from": {"fy": 2024}, "to": {"fy": 2025}, "mode": "abs"})
    assert any("never a list" in e for e in validate(plan, G))


def test_a_line_chosen_by_an_earlier_step_can_be_read():
    plan = _plan({"op": "members", "company": "MSFT", "of": "revenue by segment", "period": {"fy": 2026}},
                 {"op": "rank", "over": "$s1", "by": {"growth": {"value": True}}, "period": {"fy": 2026}, "k": 1},
                 {"op": "get", "company": "$s2", "period": {"fy": 2026}})
    assert validate(plan, G) == []


def test_how_many_answered_with_a_list_is_a_misfit():
    plan = _plan({"op": "filter", "over": "all", "where": {"by": {"concept": "Revenue"}, "period": {"fy": 2025},
                                                           "gt": 1e11}})
    res = Result("companies", [("AAPL", 1.0)])
    assert "count" in shape_misfit("How many companies had revenue above $100 billion in 2025?", plan, res)


def test_a_count_where_none_was_asked_is_a_misfit():
    plan = _plan({"op": "members", "company": "UNH", "of": "operating income by segment", "period": {"fy": 2025}},
                 {"op": "filter", "over": "$s1", "where": {"by": {"value": True}, "lt": 0}},
                 {"op": "count", "of": "$s2"})
    res = Result("number", 1, "count")
    assert shape_misfit("Which business posted an operating loss in 2025, and how big was it?", plan, res)


def test_by_how_many_is_a_difference_not_a_count():
    plan = _plan({"op": "rank", "over": ["F", "GM"], "by": {"attr": "employees"}, "period": {"fy": "latest"}, "k": 2})
    res = Result("companies", [("F", 2), ("GM", 1)])
    assert shape_misfit("Which employs more people, Ford or GM, and by roughly how many?", plan, res) == ""


def test_an_amount_where_a_share_was_asked_is_a_misfit():
    plan = _plan({"op": "get", "company": "MRK", "item": "Keytruda sales", "period": {"fy": 2025}})
    res = Result("number", 1.0, "USD")
    assert shape_misfit("What share of Merck's 2025 sales came from Keytruda?", plan, res)


def test_which_of_two_answered_by_one_difference_is_a_misfit():
    plan = _plan({"op": "get", "company": "KO", "concept": "OperatingIncome", "period": {"fy": 2025}},
                 {"op": "get", "company": "PEP", "concept": "OperatingIncome", "period": {"fy": 2025}},
                 {"op": "calc", "expr": "s1 - s2"})
    res = Result("number", 1.0, "USD")
    assert shape_misfit("Which earned more operating income in 2025, Coca-Cola or PepsiCo, and by how much?", plan, res)
    assert shape_misfit("How much more operating income did PepsiCo earn than Coca-Cola in 2025?", plan, res) == ""


def test_normalize_fixes_loose_measures_without_changing_the_question():
    from tieout.plan import normalize
    plan = _plan({"op": "filter", "over": ["KO", "PEP"], "where": {"by": {"delta": "net_income"}, "period": {"fy": 2025}, "lt": 0}},
                 {"op": "count", "of": "$s1"})
    fixed = normalize(plan)
    assert fixed["steps"][0]["where"]["by"] == {"delta": {"concept": "NetIncome"}}
    assert validate(fixed, G) == []
    plan = _plan({"op": "rank", "over": ["WMT", "TGT"], "by": {"metric": "inventory"}, "period": {"fy": 2025}, "k": 2})
    assert normalize(plan)["steps"][0]["by"] == {"concept": "Inventory"}


def test_normalize_gives_a_line_read_the_period_of_the_step_it_refers_to():
    from tieout.plan import normalize
    plan = _plan({"op": "members", "company": "CRM", "of": "revenue by offering", "period": {"fy": 2026}},
                 {"op": "rank", "over": "$s1", "by": {"value": True}, "period": {"fy": 2026}, "k": 1},
                 {"op": "get", "company": "$s2", "item": "revenue"})
    assert normalize(plan)["steps"][2]["period"] == {"fy": 2026}


def test_a_quarter_answered_with_the_year_or_the_year_to_date_is_a_misfit():
    res = Result("number", 1.0, "USD")
    year = _plan({"op": "get", "company": "JNJ", "concept": "Revenue", "period": {"fy": 2025}})
    ytd = _plan({"op": "get", "company": "KO", "concept": "OperatingIncome", "period": {"fy": 2025, "q": 3, "ytd": True}})
    q3 = _plan({"op": "get", "company": "KO", "concept": "OperatingIncome", "period": {"fy": 2025, "q": 3}})
    assert shape_misfit("What were Johnson & Johnson's sales in the fourth quarter of fiscal 2025?", year, res)
    assert shape_misfit("What operating income did Coca-Cola report for the third quarter of 2025?", ytd, res)
    assert shape_misfit("What operating income did Coca-Cola report for the third quarter of 2025?", q3, res) == ""
    assert shape_misfit("How much cash did Target generate in the first nine months of fiscal 2025?", year, res)


def test_periods_named_in_words_are_set_without_a_model():
    from tieout.ask import fix_periods
    year = _plan({"op": "get", "company": "JNJ", "concept": "Revenue", "period": {"fy": 2025}})
    assert fix_periods("What were J&J's sales in the fourth quarter of fiscal 2025?", year)["steps"][0]["period"] == {"fy": 2025, "q": 4}
    ytd = _plan({"op": "get", "company": "KO", "concept": "OperatingIncome", "period": {"fy": 2025, "q": 3, "ytd": True}})
    assert fix_periods("What operating income did Coca-Cola report for the third quarter of 2025?", ytd)["steps"][0]["period"] == {"fy": 2025, "q": 3}
    assert fix_periods("How much cash in the first nine months of fiscal 2025?", year)["steps"][0]["period"] == {"fy": 2025, "q": 3, "ytd": True}


def test_the_binding_audit_drops_only_rows_the_arithmetic_contradicts():
    from tieout.concepts import audit_bindings
    g = Graph.load()
    dropped = audit_bindings(g)
    assert dropped, "the audit should find the known contradictions"
    for d in dropped:
        assert d["why"]
        assert not any(b.state == "active" and b.doc == d["doc"] and b.concept == d["concept"] and b.label == d["label"]
                       for b in g.bindings)
    # a net line is never dropped as the gross one
    assert not any(d["concept"] == "PropertyPlantEquipment" and ", net" in d["label"].lower() for d in dropped)
