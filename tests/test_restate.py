"""tieout.restate on a small synthetic graph: no PDFs, no graph file."""
from tieout import restate
from tieout.graph import Check, Fact, Graph, Reading

SCALE = 1_000_000


def add_statement(g: Graph, doc: str, end: str, kind: str, period: str, rows: list, sums: list) -> dict:
    """rows: [(label, printed value)]; sums: [(total label, [(operand label, sign)])].

    Returns {label: reading id}. One column, money in millions.
    """
    ticker = doc.split("_")[0]
    g.filings[doc] = {"ticker": ticker, "form": "10-K", "fiscal_year": end[:4], "period_end": end, "role": ""}
    table = f"{doc}:p1:{kind}"
    ids = {}
    for i, (label, v) in enumerate(rows):
        rid = f"{table}:r{i}:c0"
        text = f"({-v:,})" if v < 0 else f"{v:,}"
        g.readings[rid] = Reading(id=rid, doc=doc, ticker=ticker, page=1, bbox=[0, 0, 1, 1], table=table,
                                  kind=kind, row=i, col=0, label=label, section="", text=text,
                                  value=v * SCALE, unit="USD", period=period, period_end=end)
        ids[label] = rid
    for n, (total, ops) in enumerate(sums):
        cid = f"{table}:sum{n}"
        members = [[ids[total], "total"]] + [[ids[lab], s] for lab, s in ops]
        g.checks[cid] = Check(id=cid, type="SUM", status="pass", doc=doc, members=members)
        for rid, _ in members:
            g.readings[rid].checks.append(cid)
    return ids


def balance_sheets() -> Graph:
    """Other assets lose 30 to a new line Intangibles; Total assets stay at 150."""
    g = Graph()
    add_statement(g, "X_FY1_10K", "2024-12-31", "BS", "FY1@end",
                  [("Cash", 50), ("Other assets", 100), ("Total assets", 150)],
                  [("Total assets", [("Cash", 1), ("Other assets", 1)])])
    add_statement(g, "X_FY2_10K", "2025-12-31", "BS", "FY1@end",
                  [("Cash", 50), ("Intangible assets, net", 30), ("Other assets", 70), ("Total assets", 150)],
                  [("Total assets", [("Cash", 1), ("Intangible assets, net", 1), ("Other assets", 1)])])
    return g


def income_statements() -> Graph:
    """Revenue restated from 500 to 480, so net income falls from 200 to 180."""
    g = Graph()
    sums = [("Net income", [("Revenue", 1), ("Costs", -1)])]
    old = add_statement(g, "Y_FY1_10K", "2024-12-31", "IS", "FY1",
                        [("Revenue", 500), ("Costs", 300), ("Net income", 200)], sums)
    new = add_statement(g, "Y_FY2_10K", "2025-12-31", "IS", "FY1",
                        [("Revenue", 480), ("Costs", 300), ("Net income", 180)], sums)
    g.facts["Y|NetIncome|FY1"] = Fact("Y", "NetIncome", "FY1", [new["Net income"], old["Net income"]])
    return g


def test_reclassification():
    d = restate.diagnose(balance_sheets(), "X", "FY1@end", label="Other assets", explain=False)
    assert d["kind"] == "reclassification"
    assert d["delta"] == -30 * SCALE
    assert d["line"]["old"]["printed"] == "100" and d["line"]["new"]["printed"] == "70"
    assert [(c["label"], c["new"], c["new_line"]) for c in d["counterparts"]] == \
        [("Intangible assets, net", 30 * SCALE, True)]
    assert d["unchanged_total"]["label"] == "Total assets" and d["unchanged_total"]["value"] == 150 * SCALE
    for ref in (d["line"]["old"], d["line"]["new"], *d["counterparts"][0]["readings"]):
        assert {"id", "doc", "page", "label", "printed"} <= set(ref)
    assert "reclassification" in d["summary"]


def test_restatement():
    g = income_statements()
    d = restate.diagnose(g, "Y", "FY1", concept="NetIncome", explain=False)
    assert d["kind"] == "restatement"
    assert d["delta"] == -20 * SCALE
    assert d["unchanged_total"] is None
    assert {(c["label"], c["delta"]) for c in d["changes"]} == {("Revenue", -20 * SCALE), ("Net income", -20 * SCALE)}
    assert d["changed_lines"] == 2
    assert [e["kind"] for e in restate.events(g)] == ["restatement"]


def test_unchanged():
    d = restate.diagnose(balance_sheets(), "X", "FY1@end", label="Cash", explain=False)
    assert d["kind"] == "unchanged" and d["delta"] == 0
    assert d["explanations"] == []


def test_renamed_line_is_aligned_by_value():
    old = [Reading("a", "D1", "Z", 1, [], "t", "BS", 0, 0, "Accumulated deficit", "", "(5)", -5, "USD", "P", "")]
    new = [Reading("b", "D2", "Z", 1, [], "t", "BS", 0, 0, "Retained earnings/(Accumulated deficit)", "", "(5)",
                   -5, "USD", "P", "")]
    pairs = restate.align(old, new)
    assert len(pairs) == 1 and not pairs[0].changed


def test_single_filing():
    g = balance_sheets()
    for rid in [r for r in g.readings if r.startswith("X_FY1")]:
        del g.readings[rid]
    d = restate.diagnose(g, "X", "FY1@end", label="Other assets", explain=False)
    assert d["kind"] == "single_filing"
