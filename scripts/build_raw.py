"""Build the raw layer (every non-primary table of every filing) and the numeric
mentions of running text; write graph/raw.json.gz and graph/mentions.json.gz.

Each file is written under a temporary name and then moved over the old one, so a
process reading the graph files meanwhile never sees half a file.

    python scripts/build_raw.py                 # every filing
    python scripts/build_raw.py DOC [DOC ...]   # only these filings, merged into the saved files
"""
import logging
import os
import sys
import time
from collections import Counter

from tieout.graph import Graph
from tieout.mentions import MENTIONS_PATH, build_mentions, load_mentions, save_mentions
from tieout.rawlayer import RAW_PATH, build_raw, load_raw, save_raw, search, summary
from tieout.restate import SENTENCES_PATH, save_sentences

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _atomic(save, data, path):
    """save(data, tmp) and move tmp over path; returns what save returned."""
    tmp = path.with_name(path.name.replace(".json.gz", ".tmp.json.gz"))
    out = save(data, tmp)
    os.replace(tmp, path)
    return out


def main():
    docs = sys.argv[1:] or None
    t0 = time.time()
    g = Graph.load()
    rows = build_raw(g, docs=docs)
    mentions = build_mentions(g, rows, docs=docs)
    if docs:                                    # keep the other filings as they were saved
        rows = [r for r in load_raw() if r["doc"] not in docs] + rows
        mentions = [m for m in load_mentions() if m["doc"] not in docs] + mentions
    _atomic(save_raw, rows, RAW_PATH)
    _atomic(save_mentions, mentions, MENTIONS_PATH)
    print("sentences that explain changed figures, saved for a copy without page caches:",
          _atomic(save_sentences, g, SENTENCES_PATH))
    s = summary(rows)
    ok = sum(1 for r in rows for c in r["cells"] if c["ok"])
    print(f"rows {s['rows']}  cells {s['cells']}  verified cells {ok} ({ok / max(1, s['cells']):.1%})")
    print("units", s["units"])
    print("roles", s["roles"])
    print("rows that tie to a bound concept:", sum(1 for r in rows if r["ties_to"]),
          "| rows anchored through their total:", sum(1 for r in rows if r["anchor"]))
    print(f"mentions {len(mentions)}  numbers {sum(len(m['numbers']) for m in mentions)}  "
          f"units {dict(Counter(n['unit'] for m in mentions for n in m['numbers']))}  "
          f"tied {sum(1 for m in mentions if m['ties'])}")
    print(f"built in {time.time() - t0:.0f} s")
    if docs:
        return
    for ticker, doc, phrase in (("AAPL", "AAPL_FY2026_10K", "iPhone net sales"),
                                ("GOOGL", "GOOGL_FY2025_10K", "Google Cloud revenue"),
                                ("WMT", "WMT_FY2026_10K", "Sam's Club net sales")):
        print(f"\nsearch {ticker}: {phrase!r}")
        for r in search(rows, ticker, doc, phrase, k=4):
            print(f"   p{r['page']:<3d} {r['label'][:34]:34s} | {r['title'][:46]:46s} | {r['role']:7s} "
                  f"ok={r['verified']} anchor={r['anchor']} ties={r['ties_to']} | {[c['text'] for c in r['cells'][:3]]}")


if __name__ == "__main__":
    main()
