"""The brief's rule, checked on every served answer.

"Answers must come from the graph, not from plain vector search over text chunks."

An answer is a figure the executor computed from proof leaves, or a name read from a company's record.
The rule holds for an answer when every one of its leaves is a node of the graph: a reading of the
built graph, a fact of a company's record, a cell of a table row of the raw layer, or a figure of a
sentence record, at the filing and page the leaf states. A leaf that names anything else (a passage, a model's reply) breaks the
rule. `from_graph` checks one answer; scripts/check_brief.py checks a whole result file and fails when
any answer breaks it.
"""
from __future__ import annotations

from typing import Optional


def node_of(leaf_id: str, g) -> Optional[dict]:
    """The graph node a proof leaf names: {"kind", "doc", "page"}, or None when there is none."""
    from .. import link
    reading = g.readings.get(leaf_id)
    if reading is not None:
        return {"kind": "reading", "doc": reading.doc, "page": reading.page}
    doc, marker, attr = str(leaf_id).partition(":text:")
    if marker:                                        # a fact of a company's record (auditor, employees), read at build time
        fact = next((t for t in getattr(g, "textfacts", []) if t["doc"] == doc and t["attr"] == attr), None)
        return {"kind": "record", "doc": fact["doc"], "page": fact["page"]} if fact else None
    st = link.store()
    row_id, _, cell = str(leaf_id).rpartition(":c")
    row = st.by_id.get(row_id)
    if row is not None and "cells" in row and cell.isdigit() and int(cell) < len(row["cells"]):
        return {"kind": "cell", "doc": row["doc"], "page": row["page"]}
    sentence_id, _, slot = str(leaf_id).rpartition(":")
    sentence = st.by_id.get(sentence_id)
    if sentence is not None and any(str(n.get("slot") or "").strip("<>") == slot for n in sentence.get("numbers") or []):
        return {"kind": "sentence", "doc": sentence["doc"], "page": sentence["page"]}
    return None


def from_graph(leaves: list, g) -> dict:
    """{"leaves", "nodes", "outside"} for one answer's proof leaves, each {"id", "doc", "page"}.
    outside lists every leaf that is not a node of the graph at the filing and page it states."""
    outside = []
    for leaf in leaves:
        node = node_of(str(leaf.get("id")), g)
        if node is None:
            outside.append({**leaf, "why": "no node of the graph has this id"})
        elif leaf.get("doc") and (node["doc"] != leaf["doc"] or int(node["page"]) != int(leaf.get("page") or 0)):
            outside.append({**leaf, "why": f"the node is on page {node['page']} of {node['doc']}"})
    return {"leaves": len(leaves), "nodes": len(leaves) - len(outside), "outside": outside}
