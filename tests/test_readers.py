"""Two readers and an arbiter (tieout/readers.py): which answer is served in each case."""
import pytest

from tieout import readers
from tieout.execute import Result, abstain
from tieout.llm import DailyLimit
from tieout.readers import AGREE, AGREE_ON_ORDER, ARBITER_ALONE, ARBITER_AND_READER, ONE_READER, Readers, same_answer

PANEL = Readers("maker-model", "checker-model", ("strong-1", "strong-2"))


def out(result: Result, cannot: str = "") -> dict:
    plan = {"steps": [], "answer": None, "cannot": cannot} if cannot else {"steps": [{"id": "s1"}], "answer": "s1"}
    return {"answer": str(result.value), "result": result, "plan": plan, "learned": []}


def number(v, unit="USD", tier="VERIFIED") -> dict:
    return out(Result("number", v, unit, tier=tier))


def panel(monkeypatch, by_model: dict) -> list:
    """Answer with canned results per model; returns the list of models that were asked."""
    asked = []

    def fake(question, g, model=None, **kw):
        asked.append(model)
        got = by_model[model]
        if isinstance(got, Exception):
            raise got
        return got

    monkeypatch.setattr(readers.ask, "answer", fake)
    return asked


def test_same_answer_compares_what_was_computed():
    assert same_answer(Result("number", 1e9, "USD"), Result("number", 1e9 + 0.1, "USD"))
    assert same_answer(Result("number", -5e6, "USD"), Result("number", 5e6, "USD"))        # an outflow, either sign
    assert not same_answer(Result("number", 0.146, "ratio"), Result("number", 14.6, "ratio"))
    assert same_answer(Result("text", "KPMG LLP"), Result("text", "KPMG"))
    assert same_answer(Result("companies", [("JPM", 2.0), ("C", 1.0)]), Result("companies", [("JPM", 2.0), ("C", 1.0)]))
    assert not same_answer(Result("companies", [("JPM", 2.0), ("C", 1.0)]), Result("companies", [("C", 1.0), ("JPM", 2.0)]))
    assert not same_answer(Result("number", 1.0), abstain("nothing"))
    assert not same_answer(abstain("a"), abstain("b"))                                    # two failures are not a reading


def test_two_readers_agree_and_the_arbiter_is_not_called(monkeypatch):
    asked = panel(monkeypatch, {"maker-model": number(7.0, tier="FLAGGED"), "checker-model": number(7.0)})
    served = readers.answer("q", None, PANEL)
    assert served["reading"]["case"] == AGREE and asked == ["maker-model", "checker-model"]
    assert served["result"].tier == "VERIFIED"            # of two equal answers, the one whose numbers all tie out


def test_both_decline_is_an_agreed_abstention(monkeypatch):
    panel(monkeypatch, {"maker-model": out(abstain("planner: no"), cannot="stock prices are not in filings"),
                        "checker-model": out(abstain("planner: no"), cannot="not printed")})
    assert readers.answer("q", None, PANEL)["reading"]["case"] == AGREE


def test_disagreement_goes_to_the_arbiter(monkeypatch):
    asked = panel(monkeypatch, {"maker-model": number(7.0), "checker-model": number(9.0), "strong-1": number(9.0)})
    served = readers.answer("q", None, PANEL)
    assert asked[-1] == "strong-1" and served["reading"]["case"] == ARBITER_AND_READER
    assert served["result"].value == 9.0 and served["result"].tier == "VERIFIED"


def test_an_unwitnessed_reading_is_never_verified(monkeypatch):
    panel(monkeypatch, {"maker-model": number(7.0), "checker-model": number(9.0), "strong-1": number(8.0)})
    served = readers.answer("q", None, PANEL)
    assert served["reading"]["case"] == ARBITER_ALONE and served["result"].tier == "FLAGGED"
    assert served["result"].tier_reasons[0] == readers.UNWITNESSED


def test_next_arbiter_when_the_daily_quota_is_spent(monkeypatch):
    asked = panel(monkeypatch, {"maker-model": number(7.0), "checker-model": out(abstain("plan rejected")),
                                "strong-1": DailyLimit("spent"), "strong-2": number(7.0)})
    served = readers.answer("q", None, PANEL)
    assert asked == ["maker-model", "checker-model", "strong-1", "strong-2"]
    assert served["reading"]["case"] == ARBITER_AND_READER


def test_one_reader_only_when_no_arbiter_can_answer(monkeypatch):
    panel(monkeypatch, {"maker-model": number(7.0, tier="FLAGGED"), "checker-model": out(abstain("plan rejected")),
                        "strong-1": DailyLimit("spent"), "strong-2": DailyLimit("spent")})
    served = readers.answer("q", None, PANEL)
    assert served["reading"]["case"] == ONE_READER and served["result"].value == 7.0
    assert served["result"].tier == "FLAGGED"


def test_a_failed_arbiter_plan_falls_back_to_a_reader(monkeypatch):
    panel(monkeypatch, {"maker-model": number(7.0), "checker-model": number(9.0, tier="FLAGGED"),
                        "strong-1": out(abstain("no link yet"))})
    served = readers.answer("q", None, PANEL)
    assert served["reading"]["case"] == ONE_READER and served["result"].value == 7.0


def test_the_same_ranking_with_different_figures_is_witnessed_but_flagged(monkeypatch):
    one = out(Result("companies", [("CAT", 0.165), ("WMT", 0.0431)], tier="VERIFIED"))
    two = out(Result("companies", [("CAT", 0.165), ("WMT", 0.0418)], tier="VERIFIED"))
    asked = panel(monkeypatch, {"maker-model": one, "checker-model": two})
    served = readers.answer("q", None, PANEL)
    assert served["reading"]["case"] == AGREE_ON_ORDER and asked == ["maker-model", "checker-model"]
    assert served["result"].tier == "FLAGGED" and served["result"].tier_reasons[0] == readers.FIGURES_DIFFER


@pytest.mark.parametrize("why,expected", [("stock prices are not in filings", True), ("planner failed (ValueError)", False)])
def test_declined_means_a_deliberate_cannot(why, expected):
    assert readers.declined(out(abstain("planner: x"), cannot=why)) is expected
