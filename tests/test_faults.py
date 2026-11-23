"""Fault study helpers on small synthetic tables: no PDF, no graph file."""
import random
from types import SimpleNamespace

from tieout.eval.faults import (Case, Evidence, SortedValues, changed_cells, classify, damage_statements,
                                inject, pins_of_table, significant_digits, study, wilson)
from tieout.tables import Cell, Column, Row, Table


def make_table(rows, scale=1_000_000.0):
    """rows: [(label, v0, v1)] printed in millions."""
    t_rows = []
    for i, (label, *vals) in enumerate(rows):
        cells = {c: Cell(1, (0, 0, 1, 1), f"{v:,}", float(v), c) for c, v in enumerate(vals)}
        t_rows.append(Row(i, 1, float(i), label, 0.0, None, cells))
    cols = [Column(0, 100, 50), Column(1, 200, 150)]
    return Table("D", 1, "balance", "", scale, 1.0, cols, t_rows)


ROWS = [("Cash", 1000, 900), ("Receivables", 2000, 1800), ("Inventory", 3000, 2700),
        ("Other assets", 4000, 3600), ("Total assets", 10000, 9000),
        ("Payables", 2500, 2200), ("Debt", 3500, 3300), ("Equity", 4000, 3500), ("Total liabilities and equity", 10000, 9000)]


def test_wilson_interval_orders_and_handles_empty():
    lo, hi = wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert wilson(0, 0) == (None, None)


def test_significant_digits_distinguish_round_and_precise_figures():
    assert significant_digits("$416.2 billion") == 4
    assert significant_digits("$5 billion") == 1
    assert significant_digits("$0.05 million") == 1


def test_sorted_values_tie_within_tolerance():
    sv = SortedValues([-416_161_000_000, 12])
    assert sv.ties(416.2e9, 0.05e9)
    assert not sv.ties(416.2e9, 1e6)


def test_pin_follows_the_table_scale():
    t = make_table(ROWS)
    figures = [(10_000_000_000.0, 0.05e9)]                     # "$10.0 billion" ties to Total assets
    pins = pins_of_table({(r.idx, c): int(cell.value * t.scale) for r in t.rows for c, cell in r.cells.items()}, figures)
    assert len(pins) == 1 and {cell[0] for cell in pins[0].cells} == {4, 8}
    thousand = make_table(ROWS, scale=1_000_000_000.0)
    assert pins_of_table({(r.idx, c): int(cell.value * thousand.scale) for r in thousand.rows
                          for c, cell in r.cells.items()}, figures) == []


def test_changed_cells_sees_a_row_swap_and_a_table_scale():
    t = make_table(ROWS)
    bad, _ = inject(t, "row_swap", 0, 0, random.Random(1))
    assert {(r, c) for r, c, _ in changed_cells(t, bad)} >= {(0, 0), (1, 0)}
    big, _ = inject(t, "table_scale", 0, 0, random.Random(1))
    assert len(changed_cells(t, big)) == 18


def fake_graph(partner_value):
    a = SimpleNamespace(id="D:p1:BS:r0:c0", value=1_000_000_000)
    b = SimpleNamespace(id="E:p1:BS:r0:c1", value=partner_value)
    check = SimpleNamespace(type="CARRY", status="pass", members=[[a.id, "FY1"], [b.id, "FY2"]])
    return SimpleNamespace(readings={a.id: a, b.id: b}, checks={"c": check})


def test_carry_catches_a_changed_cell_only_where_a_partner_exists():
    t = make_table(ROWS)
    ev = Evidence(fake_graph(1_000_000_000), [])
    case = Case("D", "BS", t)
    bad, _ = inject(t, "sign_flip", 0, 0, random.Random(1))
    from tieout.eval.faults import carry_verdict
    assert carry_verdict(case, ev, bad, changed_cells(t, bad)) == (True, True)
    bad, _ = inject(t, "sign_flip", 1, 0, random.Random(1))       # row 1 has no carried copy
    assert carry_verdict(case, ev, bad, changed_cells(t, bad)) == (False, False)


def test_study_reports_before_and_after_keys():
    out = study([make_table(ROWS)])
    assert set(out["sign_flip"]) >= {"injected", "caught", "caught_after", "pin", "carry", "ci95_after"}
    assert out["sign_flip"]["caught_after"] == out["sign_flip"]["caught"]       # no evidence given


def test_damage_curve_counts_silent_wrong_only_when_checks_are_off_or_blind():
    def reading(i, value, status):
        return SimpleNamespace(id=i, value=value, status=status)
    fact = SimpleNamespace(readings=["a"])
    clean = SimpleNamespace(facts={"k": fact}, best_reading=lambda f: reading("a", 5, "verified"))
    for status, on, off in (("verified", "silent_wrong", "silent_wrong"), ("unverified", "flagged", "silent_wrong")):
        g = SimpleNamespace(facts={"k": fact}, best_reading=lambda f, s=status: reading("a", 6, s))
        assert classify(clean, g)[on] == 1
        assert classify(clean, g, checks=False)[off] == 1
    g = SimpleNamespace(facts={"k": fact}, best_reading=lambda f: reading("a", 6, "verified"))
    assert classify(clean, g, flagged={"a"})["flagged"] == 1          # a contradicting channel turns it into a warning
    g = SimpleNamespace(facts={}, best_reading=lambda f: None)
    assert classify(clean, g)["missing"] == 1


def test_damage_statements_changes_cells_in_proportion_to_the_rate():
    t = make_table(ROWS * 6)
    st = SimpleNamespace(table=t, checks=[], ties=[], verified=0, cells=0)
    assert damage_statements({"BS": st}, "D", 0.0, 1) == 0
    assert damage_statements({"BS": st}, "D", 0.5, 1) > 5
