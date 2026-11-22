"""The namesake audit of a row pick, and the power operator of calc.

Two tables can print a row with one label (a debt schedule and an amortization schedule both have
"Fiscal 2027"). Two models that agree on one of them are trusted only if the words asked for fit the
picked row's surroundings better than its namesake's.
"""
from tieout import link
from tieout.plan import calc_errors


def row(label: str, above: str, value: float, page: int) -> dict:
    return {"id": f"X_FY2026_10K:p{page}:r0", "doc": "X_FY2026_10K", "ticker": "X", "page": page, "label": label,
            "section": "", "title": "", "context": "", "above": above, "unit": "USD", "role": "loose",
            "verified": False, "cells": [{"pid": "FY2026", "value": value, "text": str(value), "col": "2026"}]}


def test_a_pick_against_the_words_has_a_second_reading():
    amortization = row("Fiscal 2027", "estimated future amortization of intangible assets", 731, 83)
    maturities = row("Fiscal 2027", "future principal payments for all of our borrowings", 7210, 85)
    cands = [{"kind": "row", "ref": amortization}, {"kind": "row", "ref": maturities}]
    phrase, wanted = "borrowings principal due in fiscal 2027", {"FY2026"}
    assert link.lookalike(None, cands, 0, {"pick": 0}, phrase, wanted) is maturities     # the words fit the other row
    assert link.lookalike(None, cands, 1, {"pick": 1}, phrase, wanted) is None           # the words single this one out


def test_a_reprint_of_the_same_figure_is_not_a_second_reading():
    note = row("Fiscal 2027", "future principal payments for all of our borrowings", 7210, 85)
    again = row("Fiscal 2027", "contractual obligations and borrowings", 7210, 47)
    cands = [{"kind": "row", "ref": note}, {"kind": "row", "ref": again}]
    assert link.lookalike(None, cands, 0, {"pick": 0}, "borrowings due in fiscal 2027", {"FY2026"}) is None


def test_calc_accepts_a_power_for_compound_growth():
    assert calc_errors("(s2 / s1) ** (1 / 3) - 1", {"s1", "s2"}, "s3") == []
    assert calc_errors("s1 ** s9", {"s1"}, "s3")                                           # s9 is not defined
