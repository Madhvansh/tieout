"""A question that asks how much something GREW, and names no unit, asks for the growth rate.
Round 9 (23 Nov), X15 and X18: "By how much did Apple's Services gross profit grow from fiscal 2023 to
fiscal 2026?" was answered with the dollar change by two readers and with the rate by the third; the
key, written by a blind author, wants the rate. Over all development rounds: "grow" with no unit named
wants the rate in 3 of 3 cases; increase, fall and change want the amount in 10 of 10
(dev_scripts/fable/grow_words.py), so only the verb "grow" is rewritten."""
from tieout.ask import growth_is_a_rate

GETS = [{"id": "s1", "op": "get", "company": "AAPL", "item": "Services gross profit", "period": {"fy": 2023}},
        {"id": "s2", "op": "get", "company": "AAPL", "item": "Services gross profit", "period": {"fy": 2026}}]
CHANGE = {"steps": [{"id": "s1", "op": "change", "company": "AAPL", "item": "Services gross profit",
                     "from": {"fy": 2023}, "to": {"fy": 2026}, "mode": "abs"}], "answer": "s1"}
DIFF = {"steps": GETS + [{"id": "s3", "op": "calc", "expr": "s2 - s1"}], "answer": "s3"}
GROW = "By how much did Apple's Services gross profit grow from fiscal 2023 to fiscal 2026?"


def test_grew_with_no_unit_is_the_rate():
    assert growth_is_a_rate(GROW, CHANGE)["steps"][0]["mode"] == "pct"
    assert growth_is_a_rate(GROW, DIFF)["steps"][-1]["expr"] == "(s2 - s1) / s1"
    assert growth_is_a_rate("How much did Goldman Sachs grow the client assets it supervises?", CHANGE)["steps"][0]["mode"] == "pct"


def test_a_named_unit_or_another_verb_is_left_alone():
    for q in ("By how many dollars did Apple's Services gross profit grow from fiscal 2023 to fiscal 2026?",
              "How much did Pfizer's COVID franchise revenue fall in 2025?",
              "By how much did Alphabet's revenue backlog increase between September 30 and December 31, 2025?",
              "How much of Adobe's fiscal 2025 revenue growth came from its Digital Media business?"):
        assert growth_is_a_rate(q, CHANGE) == CHANGE
        assert growth_is_a_rate(q, DIFF) == DIFF


def test_a_plan_that_already_gives_a_rate_or_answers_otherwise_is_left_alone():
    rate = {"steps": [{**CHANGE["steps"][0], "mode": "pct"}], "answer": "s1"}
    assert growth_is_a_rate(GROW, rate) == rate
    other = {"steps": GETS + [{"id": "s3", "op": "calc", "expr": "s2 / s1"}], "answer": "s3"}
    assert growth_is_a_rate(GROW, other) == other
    unfinished = {"steps": [], "answer": None, "cannot": "not stated"}
    assert growth_is_a_rate(GROW, unfinished) == unfinished
