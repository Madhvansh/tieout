"""The two scorers (tieout/eval/bench.py, 23 Nov): each 22 Nov relaxation tightened against the red
team's counter-example (research/17 Part 1b), and strict = every 22 Nov relaxation switched off."""
from tieout.eval.bench import is_correct, rescore


def num(v):
    return {"kind": "number", "value": v, "extra": {}}


def q(value, unit="USD", question="", answer="", facts=(), tol=0.005):
    return {"id": "X", "answer_value": value, "answer_unit": unit, "tolerance": tol, "tags": [],
            "question": question, "answer": answer, "supporting_facts": list(facts)}


def test_printed_slack_needs_two_significant_digits():
    one_digit = q(1e9, facts=[{"value": 1e9, "printed": "$1 billion"}])
    assert not is_correct(one_digit, num(0.62e9), {}, "final")
    two_digits = q(2.1e9, facts=[{"value": 2.1e9, "printed": "$2.1 billion"}])
    assert is_correct(two_digits, num(2.14e9), {}, "final")
    assert not is_correct(two_digits, num(2.14e9), {}, "strict")         # no printed slack on 21 Nov


def test_by_how_much_is_scored_on_the_gap():
    facts = [{"value": 10.0}, {"value": 8.0}]
    asked = q(10.0, unit="count", question="Which segment sold more, A or B, and by how much?", facts=facts)
    misread = {"kind": "lines", "value": [("A", 10.0), ("B", 3.0)], "extra": {"gap": 7.0}}
    right = {"kind": "lines", "value": [("A", 10.0), ("B", 8.0)], "extra": {"gap": 2.0}}
    assert not is_correct(asked, misread, {}, "final")                  # the winner's 10 no longer carries a gap of 7
    assert is_correct(asked, right, {}, "final")
    which = q(10.0, unit="count", question="Which segment sold more, A or B?", facts=facts)
    assert is_correct(which, misread, {}, "final")                      # no gap was asked


def test_names_of_lines_are_compared_as_sets_of_words():
    gold = q(["U.S.", "Other"], unit="list")
    assert not is_correct(gold, {"kind": "lines", "value": [("Non-U.S.", 5.0), ("Other Bets", 3.0)]}, {}, "final")
    assert is_correct(gold, {"kind": "lines", "value": [("U.S. segment", 5.0), ("Other", 3.0)]}, {}, "final")
    assert not is_correct(gold, {"kind": "lines", "value": [("U.S.", 5.0), ("Other", 3.0)]}, {}, "strict")


def test_a_fall_is_read_from_the_golds_own_words():
    both = "Did the operating margin decline or improve in 2025, and by how much?"
    improved = q(0.004, unit="ratio", question=both, answer="It improved by 0.4 points.")
    assert not is_correct(improved, num(-0.004), {}, "final")
    fell = q(3.53, unit="ratio", question="By how many turns did coverage fall?", answer="Coverage fell 3.53 turns.")
    assert is_correct(fell, num(-3.53), {}, "final")
    assert not is_correct(fell, num(-3.53), {}, "strict")


def test_a_count_in_the_pages_scale_needs_the_gold_to_be_one_of_its_facts():
    facts = [{"value": 1927.0, "printed": "1,927", "scale": "thousands of barrels per day"}]
    as_printed = q(1927.0, unit="count", facts=facts)
    assert is_correct(as_printed, num(1927000.0), {}, "final")
    converted = q(1927000.0, unit="count", facts=facts)                 # the key converted its gold, not its facts
    assert not is_correct(converted, num(1927000000.0), {}, "final")
    gap = q(889.0, unit="count", facts=facts + [{"value": 1038.0, "printed": "1,038", "scale": "thousands"}])
    assert is_correct(gap, num(889000.0), {}, "final")                  # round 8 W17: a gap of two printed counts


def test_a_dollar_change_of_the_wrong_sign_is_wrong():
    change = q(5e9, question="How much did revenue change from 2024 to 2025?")
    assert not is_correct(change, num(-5e9), {}, "final") and not is_correct(change, num(-5e9), {}, "strict")
    amount = q(5e9, question="How much did the company pay in dividends in 2025?")
    assert is_correct(amount, num(-5e9), {}, "final")                   # an outflow printed negative
    differ = q(4.72e9, question="How much does the figure in the 2025 10-K differ from the figure in the 2026 10-K?")
    assert is_correct(differ, num(-4.72e9), {}, "final")                # "differ" asks for a size (round 5 Q17)


def test_strict_switches_the_percentage_point_rule_off():
    pp = q(3.72, unit="percent_points")
    assert is_correct(pp, num(0.0372), {}, "final") and not is_correct(pp, num(0.0372), {}, "strict")


def test_a_stored_prediction_can_be_rescored():
    row = {"pred": {"kind": "companies", "value": [["MSFT", 137791e6], ["AMZN", 128725e6]], "extra": {"gap": 9066e6}}}
    assert rescore(row, q(9066e6), {}, "final") is True
    assert rescore({}, q(1.0), {}, "final") is None
