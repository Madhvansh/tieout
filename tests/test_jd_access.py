"""Access control with provenance (scripts/jd_access.py): an answer is served only if the asker may read
every leaf of its proof; a derived figure carries the labels of all its leaves."""
import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "jd_access.py"
_spec = importlib.util.spec_from_file_location("jd_access", _PATH)
jd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jd)

FILINGS = {"AAPL_FY2026_10K": {"ticker": "AAPL"}, "AAPL_FY2025_10K": {"ticker": "AAPL"},
           "MSFT_FY2026_10K": {"ticker": "MSFT"}}
APPLE = {"doc": "AAPL_FY2026_10K", "page": 28, "printed": "478,834"}
MICROSOFT = {"doc": "MSFT_FY2026_10K", "page": 40, "printed": "245,122"}


def test_everything_allowed_serves_the_answer_unchanged():
    assert jd.gate("478,834", [APPLE, MICROSOFT], None, FILINGS) == (True, "478,834")


def test_a_derived_figure_inherits_the_labels_of_every_leaf():
    served, text = jd.gate("Apple's revenue is 1.95 times Microsoft's", [APPLE, MICROSOFT], {"AAPL"}, FILINGS)
    assert not served
    assert "MSFT_FY2026_10K page 40" in text
    assert "245,122" not in text                      # the hidden figure itself never leaks


def test_both_companies_allowed_serves_the_comparison():
    assert jd.gate("x", [APPLE, MICROSOFT], {"AAPL", "MSFT"}, FILINGS)[0]


def test_a_filing_label_allows_only_that_filing():
    assert jd.gate("x", [APPLE], {"AAPL_FY2026_10K"}, FILINGS)[0]
    assert not jd.gate("x", [APPLE], {"AAPL_FY2025_10K"}, FILINGS)[0]


def test_an_answer_with_no_source_is_withheld_under_a_policy():
    served, text = jd.gate("x", [], {"AAPL"}, FILINGS)
    assert not served and "no source" in text
