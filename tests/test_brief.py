"""The brief's rule as a test: "Answers must come from the graph, not from plain vector search over text chunks."

tieout/eval/brief.py decides whether a proof leaf is a node of the graph; scripts/check_brief.py applies it
to every answer of a result file. These tests pin what counts as a node, and that an answer found through
the text reader carries graph nodes as its leaves."""
from types import SimpleNamespace

from tieout import link, passage
from tieout.eval.brief import from_graph, node_of

DOC = "GM_FY2025_10K"
ROW = {"id": f"{DOC}:p5:x0:r1", "doc": DOC, "ticker": "GM", "page": 5, "table": 0, "title": "", "section": "",
       "label": "Automotive revenue", "unit": "USD", "role": "loose", "verified": False,
       "cells": [{"pid": "FY2025", "value": 19_918_000_000.0, "text": "19,918", "ok": False, "col": "FY2025"}]}
SENTENCE = {"id": f"{DOC}:p7:s0", "doc": DOC, "ticker": "GM", "page": 7, "sentence": "We employed approximately 25,400 people.",
            "masked": "We employed approximately <n1> people.", "ties": [],
            "numbers": [{"slot": "<n1>", "text": "25,400", "value": 25400, "unit": "count"}]}


class Store:
    rows_by_ticker = {"GM": [ROW]}
    mentions_by_ticker = {"GM": [SENTENCE]}
    by_id = {ROW["id"]: ROW, SENTENCE["id"]: SENTENCE}


def graph():
    return SimpleNamespace(readings={f"{DOC}:p80:t0:r3:c1": SimpleNamespace(doc=DOC, page=80)},
                           textfacts=[{"doc": DOC, "attr": "auditor", "page": 60, "value": "Deloitte"}])


def test_what_counts_as_a_node_of_the_graph(monkeypatch):
    monkeypatch.setattr(link, "store", lambda: Store)
    g = graph()
    assert node_of(f"{DOC}:p80:t0:r3:c1", g)["kind"] == "reading"         # a cell of a statement, inside the arithmetic
    assert node_of(f"{DOC}:p5:x0:r1:c0", g)["kind"] == "cell"             # a cell of any other table row
    assert node_of(f"{DOC}:p7:s0:n1", g)["kind"] == "sentence"            # a figure stated in a sentence
    assert node_of(f"{DOC}:text:auditor", g)["kind"] == "record"          # a fact of the company's record
    assert node_of(f"{DOC}:text:founder", g) is None
    assert node_of(f"{DOC}:p5:passage0", g) is None                       # a passage is not a node
    assert node_of(f"{DOC}:p5:x0:r1:c9", g) is None                       # nor is a cell the row does not have
    assert node_of(f"{DOC}:p7:s0:n4", g) is None


def test_a_leaf_outside_the_graph_is_reported(monkeypatch):
    monkeypatch.setattr(link, "store", lambda: Store)
    good = {"id": f"{DOC}:p5:x0:r1:c0", "doc": DOC, "page": 5}
    assert from_graph([good], graph())["outside"] == []
    invented = {"id": f"{DOC}:p5:passage0", "doc": DOC, "page": 5}
    moved = {"id": f"{DOC}:p5:x0:r1:c0", "doc": DOC, "page": 6}            # the node exists, on another page
    found = from_graph([good, invented, moved], graph())
    assert found["leaves"] == 3 and found["nodes"] == 1 and [x["id"] for x in found["outside"]] == [invented["id"], moved["id"]]


def test_an_answer_found_through_the_text_reader_has_graph_nodes_as_leaves(monkeypatch):
    monkeypatch.setattr(link, "store", lambda: Store)
    reply = {"pred": {"kind": "number", "value": 25400.0}, "unit": "count",
             "facts": [{"doc": DOC, "page": 7, "value": 25400.0, "label": "employees"}]}
    out = passage.as_answer(reply)
    leaves = [{"id": lf.id, "doc": lf.doc, "page": lf.page} for lf in out["result"].leaves]
    assert leaves and from_graph(leaves, graph())["outside"] == []
    assert out["result"].value == 25400.0 and out["result"].leaves[0].printed == "25,400"   # the node's own print
