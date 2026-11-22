"""23 Nov red team (research/17 Part 1c): every path by which an answer reached VERIFIED or WITNESSED
without a second model's check, and the rule that now closes it (execute.trust, link.witness_of)."""
from tieout.concepts import Binding
from tieout.execute import Executor, Leaf, Result, trust
from tieout.graph import Fact, Graph, Reading
from tieout.link import _seen_by_pickers, column_choice, label_names_phrase, witness_of, _phrase_key

DOC = "ACME_FY2025_10K"


def graph(origin: str) -> Graph:
    g = Graph()
    g.filings = {DOC: {"ticker": "ACME", "fiscal_year": "FY2025", "role": "latest", "period_end": "2025-12-31",
                       "form": "10-K"}}
    g.companies = {"ACME": {"name": "Acme Corp", "cluster": "industrial"}}
    g.readings = {"r1": Reading(id="r1", doc=DOC, ticker="ACME", page=50, bbox=[], table="t", kind="IS", row=7, col=1,
                                label="Earnings from operations", section="", text="1,234", value=1.234e9, unit="USD",
                                period="FY2025", period_end="2025-12-31", status="verified")}
    g.facts = {"ACME|OperatingIncome|FY2025": Fact("ACME", "OperatingIncome", "FY2025", ["r1"], "active")}
    g.bindings = [Binding(doc=DOC, kind="IS", row=7, label="Earnings from operations", concept="OperatingIncome",
                          origin=origin, state="active")]
    return g


def served(g: Graph) -> tuple:
    ex = Executor(g)
    res = ex._from_fact(g.facts["ACME|OperatingIncome|FY2025"], "ACME", "OperatingIncome", "FY2025", False, "USD", False)
    return trust(res)


# 1. a concept row bound while answering is one model's choice
def test_a_seed_binding_is_verified():
    assert served(graph("seed"))[0] == "VERIFIED"


def test_a_binding_one_model_learned_while_answering_is_flagged_with_its_reason():
    tier, reasons = served(graph("learned:T16"))
    assert tier == "FLAGGED" and "one model's proposal (learned:T16)" in reasons[0]


def test_a_binding_two_models_made_is_witnessed_never_verified():
    g = graph("learned:T16")
    g.witnessed_bindings = {(DOC, "IS", 7, "OperatingIncome")}
    assert served(g) == ("WITNESSED", [])


# 2. a stored link reused for rows the pickers never saw
def row(rid, doc, above, values, label="Total revenues"):
    return {"id": rid, "doc": doc, "label": label, "above": above, "section": "", "title": "",
            "cells": [{"pid": pid, "col": "", "value": v, "text": str(v)} for pid, v in values]}


def test_only_rows_the_pickers_saw_or_the_arithmetic_ties_keep_their_witness():
    first = row("a", DOC, "Segment results | (in millions)", [("FY2025", 4336), ("FY2024", 4100)])
    block = row("b", DOC, "Segment results | (in millions)", [("FY2023", 3900)])          # same table, same filing
    carried = row("c", "ACME_FY2026_10K", "Results by segment", [("FY2026", 4600), ("FY2025", 4336)])
    restated = row("d", "ACME_FY2026_10K", "Segment results | (in millions)", [("FY2026", 4600), ("FY2025", 4556)])
    entry = {"doc": DOC, "member": None}
    assert _seen_by_pickers(first, first, entry) and _seen_by_pickers(first, block, entry)
    assert _seen_by_pickers(first, carried, entry)                 # prints 4,336 for FY2025 too: arithmetic
    assert not _seen_by_pickers(first, restated, entry)            # only the heading says it is the same line


def test_the_one_line_of_its_name_in_another_filing_is_the_same_line_but_not_witnessed():
    from tieout.link import _same_line
    first = row("a", DOC, "Segment results | (in millions)", [("FY2025", 4336), ("FY2024", 4100)])
    later = row("e", "ACME_FY2026_10K", "Operating segments", [("FY2026", 4700), ("FY2025", 4556)])
    assert _same_line(first, later, [first, later], None)          # no rival in that filing: the same line
    assert not _seen_by_pickers(first, later, {"doc": DOC, "member": None})


def test_a_witnessed_line_covers_its_own_phrase_and_readings_only():
    g = Graph()
    g.witnessed_links = {("ACME", "item:k")}
    g.witnessed_phrases = {(_phrase_key("ACME", "segment revenue"), "item:k")}
    g.witnessed_readings = {"a:c0"}
    assert witness_of(g, "ACME", "segment revenue", "item:k", "a:c0") == (True, "")
    ok, why = witness_of(g, "ACME", "segment revenue", "item:k", "d:c1")
    assert not ok and "no picker saw" in why
    ok, why = witness_of(g, "ACME", "sales of the segment", "item:k", "a:c0")
    assert not ok and "another wording" in why


def test_an_unwitnessed_reading_of_a_witnessed_line_is_flagged_with_its_reason():
    leaf = Leaf("d:c1", "ACME", "item:k", "FY2025", "ACME_FY2026_10K", 60, "Total revenues", "4,556", "verified", 4556,
                witnessed=False, unwitnessed="two models picked this line in another filing; ... no picker saw")
    tier, reasons = trust(Result("number", 4556, "USD", [leaf]))
    assert tier == "FLAGGED" and "no picker saw" in reasons[0]


# 3. a column the phrase names, chosen by a word rule for both pickers
def test_a_named_column_is_a_rules_choice_and_a_dispute_when_the_pickers_differ():
    spans = ["U.S.", "Non-U.S.", "Total (the sum of the others)"]
    dispute, by_rule = column_choice([("m", {"pick": 3, "column": 0}), ("c", {"pick": 3, "column": 2})], 3, spans, 0)
    assert dispute and "different columns" in dispute and not by_rule
    dispute, by_rule = column_choice([("m", {"pick": 3}), ("c", {"pick": 3})], 3, spans, 0)
    assert not dispute and "word rule" in by_rule
    assert column_choice([("m", {"pick": 3, "column": 0}), ("c", {"pick": 3, "column": 0})], 3, spans, 0) == ("", "")


# 4. a sentence swapped for the table row it ties to
def test_a_tied_row_keeps_the_witness_only_if_its_label_names_the_item():
    assert not label_names_phrase("Dividends to shareholders", "advertising costs")      # P&G, 22 Nov
    assert label_names_phrase("Advertising expense", "advertising costs")
