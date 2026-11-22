"""The four-eyes rule, in one place.

Arithmetic can prove a number. It cannot prove a choice a language model makes from words:
which reading of a question, which printed row a phrase means. So no such choice rests on one
model. Two models choose, each without seeing the other's choice:

    they choose the same      accepted: the choice has two witnesses
    they differ               one call to the strongest free model, the arbiter, which chooses
                              without seeing either choice; its choice is accepted, and counts
                              as witnessed when it equals one of the first two

The two are not independent, and nothing here claims they are. Both row pickers are Gemini
flash-lite models; the local planner was trained on plans written with flash-lite's help; and the
word rules (ask.year_end_not_quarter, fix_periods, fix_named_set, link.named_columns, the noun and
period lists) rewrite what both readers see in the same way, so an error those rules make is made
for both. Models of one family share about 60 percent of their errors (Kim et al., ICML 2025). Two
witnesses lower the rate of a wrong choice; they do not remove it, and the measured rate is reported
(WITNESSED answers on fresh first runs: 20 of 25 right).

The strong free models allow about twenty calls a day, so they are spent only on disagreements.
`tieout/readers.py` applies the rule to plans (agreement is judged on what the plans compute);
`tieout/link.py` applies it to the pick of a row or sentence among masked candidates, and
`tieout/living.py` to the row a concept is bound to while answering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .llm import DailyLimit, impatient

AGREE = "two readers agree"
AGREE_ON_ORDER = "two readers agree on the order"
AGREE_OF_THREE = "two of three readers agree"
ARBITER_AND_READER = "arbiter and one reader agree"
ARBITER_ALONE = "arbiter alone"
ONE_READER = "one reader only"
WITNESSED = (AGREE, AGREE_ON_ORDER, AGREE_OF_THREE, ARBITER_AND_READER)


@dataclass
class Panel:
    maker: str                       # e.g. "gemini:gemini-3.5-flash-lite"
    checker: str                     # a second model
    arbiters: tuple = ()             # strongest first; the next is tried when one's daily quota is spent
    third: str = ""                  # item 8 (23 Nov): a third cheap reader of another family, asked only when
                                     # the first two differ, before any arbiter call (readers.answer)


def choose(ask: Callable, panel: Panel, same: Callable = lambda a, b: a == b) -> tuple:
    """(choice, case, [(model, choice), ...]) for one choice made under the rule. `ask(model)` returns
    that model's choice, or None when it has none; `same(a, b)` says whether two choices are the same."""
    first, second = ask(panel.maker), ask(panel.checker)
    seen = [(panel.maker, first), (panel.checker, second)]
    if first is not None and second is not None and same(first, second):
        return first, AGREE, seen
    for model in panel.arbiters:
        try:
            with impatient(model):                   # an overloaded arbiter is skipped for the next in the chain
                third = ask(model)
        except DailyLimit:
            continue
        seen.append((model, third))
        if third is None:
            break
        agreed = any(other is not None and same(third, other) for other in (first, second))
        return third, (ARBITER_AND_READER if agreed else ARBITER_ALONE), seen
    return (first if first is not None else second), ONE_READER, seen
