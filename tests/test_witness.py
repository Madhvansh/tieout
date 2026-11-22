"""The four-eyes rule (tieout/witness.py) and its use for row picks (tieout/link._pick)."""
from tieout import link
from tieout.llm import DailyLimit
from tieout.witness import AGREE, ARBITER_ALONE, ARBITER_AND_READER, ONE_READER, Panel, choose

PANEL = Panel("maker", "checker", ("strong-1", "strong-2"))


def asker(by_model: dict, asked: list):
    def ask(model):
        asked.append(model)
        got = by_model[model]
        if isinstance(got, Exception):
            raise got
        return got
    return ask


def test_two_equal_choices_need_no_arbiter():
    asked = []
    assert choose(asker({"maker": 3, "checker": 3}, asked), PANEL)[:2] == (3, AGREE)
    assert asked == ["maker", "checker"]


def test_a_disagreement_is_settled_by_the_arbiter():
    choice, case, seen = choose(asker({"maker": 3, "checker": 5, "strong-1": 5}, []), PANEL)
    assert (choice, case) == (5, ARBITER_AND_READER) and seen[-1] == ("strong-1", 5)
    assert choose(asker({"maker": 3, "checker": 5, "strong-1": 7}, []), PANEL)[:2] == (7, ARBITER_ALONE)


def test_the_next_arbiter_is_tried_when_a_quota_is_spent():
    asked = []
    choice, case, _ = choose(asker({"maker": 1, "checker": 2, "strong-1": DailyLimit("spent"), "strong-2": 2}, asked), PANEL)
    assert (choice, case) == (2, ARBITER_AND_READER) and asked[-2:] == ["strong-1", "strong-2"]


def test_with_no_arbiter_left_one_reader_is_served_as_such():
    spent = {"maker": 1, "checker": None, "strong-1": DailyLimit("spent"), "strong-2": DailyLimit("spent")}
    assert choose(asker(spent, []), PANEL)[:2] == (1, ONE_READER)


def test_row_picks_are_witnessed_when_a_second_picker_is_set(monkeypatch):
    answers = {"maker-picker": {"pick": 2, "column": None}, "second-picker": {"pick": 2, "column": 1}}
    monkeypatch.setattr(link, "link_model", lambda model, offline: "maker-picker")
    monkeypatch.setattr(link, "ask_json", lambda system, user, model=None, tag="", offline=False: answers[model])
    monkeypatch.delenv("TIEOUT_LINK_CHECKER", raising=False)
    assert link._pick("s", "u", "link", None, False) == ({"pick": 2, "column": None}, "")       # one picker: no case
    monkeypatch.setenv("TIEOUT_LINK_CHECKER", "second-picker")
    assert link._pick("s", "u", "link", None, False)[1] == AGREE                                 # same candidate
    answers["second-picker"] = {"pick": 4}
    monkeypatch.setenv("TIEOUT_LINK_ARBITER", "strong")
    answers["strong"] = {"pick": 4}
    assert link._pick("s", "u", "link", None, False) == ({"pick": 4}, ARBITER_AND_READER)
