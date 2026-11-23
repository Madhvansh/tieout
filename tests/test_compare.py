"""compare between two lines of one company: sides named by their item words, scored on the gap.

Round 6 R12 (22 Nov): J&J's TREMFYA against STELARA in a fourth quarter worked out as the year minus
nine months. The figures were right ($1,589 million against $1,230 million, gap $359 million), but the
sides were named by step ids ("s6", "s3") and the scorer read only the winner's value."""
from types import SimpleNamespace

from tieout.eval.bench import is_correct
from tieout.execute import Executor

PLAN = [
    {"id": "s1", "op": "get", "company": "JNJ", "item": "STELARA worldwide sales", "period": {"fy": 2025}},
    {"id": "s2", "op": "get", "company": "JNJ", "item": "STELARA worldwide sales", "period": {"fy": 2025, "q": 3, "ytd": True}},
    {"id": "s3", "op": "calc", "expr": "s1 - s2"},
    {"id": "s4", "op": "get", "company": "JNJ", "item": "TREMFYA", "period": {"fy": 2025}},
    {"id": "s5", "op": "get", "company": "JNJ", "item": "TREMFYA", "period": {"fy": 2025, "q": 3, "ytd": True}},
    {"id": "s6", "op": "calc", "expr": "s4 - s5"},
    {"id": "s7", "op": "compare", "of": ["s3", "s6"], "order": "desc"},
]


def _words(sid):
    fake = SimpleNamespace(_steps={s["id"]: s for s in PLAN})
    fake._step_words = lambda i, seen=None: Executor._step_words(fake, i, seen)
    return Executor._step_words(fake, sid)


def test_sides_take_the_words_of_the_line_they_were_worked_out_from():
    assert _words("s6") == "TREMFYA"
    assert _words("s3") == "STELARA worldwide sales"
    assert _words("s1") == "STELARA worldwide sales"


def test_a_step_with_no_words_behind_it_gives_none():
    assert _words("s99") == ""


Q = {"id": "R12", "answer_value": 359000000.0, "answer_unit": "USD", "tolerance": 0.005, "tags": []}


def test_two_lines_compared_are_scored_on_the_gap_or_the_winner():
    pred = {"kind": "lines", "value": [("TREMFYA", 1589e6), ("STELARA", 1230e6)], "extra": {"gap": 359e6}}
    assert is_correct(Q, pred, {})
    winner_gold = dict(Q, answer_value=1589000000.0)
    assert is_correct(winner_gold, pred, {})


def test_three_compared_are_scored_on_first_minus_last_too():
    # round 6 R16: "which is largest, and how much bigger is it than the smallest of the three?"
    q = {"id": "R16", "answer_value": 79086000000.0, "answer_unit": "USD", "tolerance": 0.005, "tags": []}
    value = [("MSFT", 137791e6), ("AMZN", 128725e6), ("GOOGL", 58705e6)]
    assert is_correct(q, {"kind": "companies", "value": value, "extra": {"gap": 9066e6, "spread": 79086e6}}, {})
    assert not is_correct(q, {"kind": "companies", "value": value, "extra": {"gap": 9066e6}}, {})


def test_lines_without_a_gap_still_read_the_top_line_only():
    pred = {"kind": "lines", "value": [("TREMFYA", 1589e6), ("STELARA", 1230e6)], "extra": {}}
    assert not is_correct(Q, pred, {})
    assert is_correct(dict(Q, answer_value=1589000000.0), pred, {})
