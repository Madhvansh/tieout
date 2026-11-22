"""The outcome gate at fact level (scripts/run_benchmark.keep_links, 23 Nov): a right answer keeps a
learned line only when the figure that line supplied is one of the gold's facts."""
import importlib.util
import pathlib

from tieout.execute import Leaf, Result

spec = importlib.util.spec_from_file_location(
    "run_benchmark", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def test_only_lines_that_supplied_a_gold_fact_are_kept(monkeypatch):
    saved = []
    monkeypatch.setattr("tieout.link.save_link", lambda e, path=None: saved.append(e["item"]))
    monkeypatch.setattr("tieout.living.save_entry", lambda e: saved.append(e["concept"]))
    leaves = [Leaf("a", "AAA", "item:aws", "FY2025", "D", 1, "AWS", "107,556", "verified", 107556e6),
              Leaf("b", "BBB", "item:azure", "FY2025", "D", 2, "Azure", "99,000", "verified", 99e9),   # a ranking's loser
              Leaf("c", "CCC", "OperatingIncome", "FY2025", "D", 3, "Earnings", "5", "verified", 5e9)]
    out = {"result": Result("companies", [("AAA", 107556e6), ("BBB", 99e9)], "USD", leaves),
           "learned": [{"kind": "item", "ticker": "AAA", "item": "item:aws", "state": "active"},
                       {"kind": "item", "ticker": "BBB", "item": "item:azure", "state": "active"},
                       {"kind": "IS", "ticker": "CCC", "concept": "OperatingIncome", "state": "active"},
                       {"kind": "item", "ticker": "AAA", "item": "item:never", "state": "rejected"}]}
    q = {"id": "Q1", "supporting_facts": [{"value": 107556e6}, {"value": -5e9}]}
    assert rb.keep_links(out, q) == (2, 1)
    assert saved == ["item:aws", "OperatingIncome"]
