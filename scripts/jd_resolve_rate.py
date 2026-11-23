"""How often is the place the text reader names a node of the graph? (JD direction 1: graph coverage
against what can be answered from the graph.)

    python scripts/jd_resolve_rate.py --files results/jd_resolve_rate_files.txt   # the reported set (below)
    python scripts/jd_resolve_rate.py --out results/jd_resolve_rate_all.json       # every saved reply now
    python scripts/jd_resolve_rate.py --before 2026-11-23T03:29:10 --save-list results/jd_resolve_rate_files.txt

The reported set is the 367 replies saved before 03:29:10 IST on 23 Nov, the moment the figure was
first measured (546 of 579 cited figures are nodes). Their file names are listed in
results/jd_resolve_rate_files.txt, so the number replays in a fresh clone, where file times differ.

Every reply of the text reader is saved under cache/passage/ with the figures it cites (filing, page,
printed figure). For each cited figure, `passage.node_for` looks for the graph node at that place: a
table cell or a sentence figure on the cited page or a neighbour, digit for digit. No model is called.
Replies are split by what was asked: a whole benchmark question, or one line of a plan (the pointer's
own small question, "What was <line> of <company> for <period>? Give the one figure ...").

Writes results/jd_resolve_rate.json. Prints counts only (no question text, no answers).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections import Counter

from tieout import passage
from tieout.graph import Graph

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINE_ASK = "Give the one figure the filing prints for it."


def count(paths: list, g: Graph, shift: int = 7) -> dict:
    """Counts over the replies; the control asks the same of each cited figure with its page moved by
    `shift` pages (a match there is a coincidence of digits, the share a match by value alone would give)."""
    c, kinds = Counter(), Counter()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        c["replies"] += 1
        if d.get("abstain") or not d.get("facts_ok"):
            c["abstains or facts not on their pages"] += 1
            continue
        c["answers"] += 1
        for f in d.get("facts") or []:
            if not passage._is_number(f.get("value")):
                continue
            c["cited figures"] += 1
            node = passage.node_for(f, "", g)
            if node is not None:
                c["cited figures that are nodes"] += 1
                kinds[node["kind"]] += 1
            if shift and isinstance(f.get("page"), int):
                c["control: figures with a page to move"] += 1
                if passage.node_for({**f, "page": f["page"] + shift}, "", g) is not None:
                    c["control: nodes found on the moved page"] += 1
        pred = d.get("pred") or {}
        if pred.get("kind") == "number" and passage._is_number(pred.get("value")):
            c["numeric answers"] += 1
            if passage.node_of_answer(d, g) is not None:
                c["numeric answers that are one node"] += 1
        else:
            c["answers that are a ranking or words"] += 1
    out = dict(c)
    out["node_kinds"] = dict(kinds)
    n, k = c.get("cited figures", 0), c.get("cited figures that are nodes", 0)
    out["share_of_cited_figures_that_are_nodes"] = round(k / n, 4) if n else None
    m, j = c.get("control: figures with a page to move", 0), c.get("control: nodes found on the moved page", 0)
    out["control_share_on_a_page_moved_by_%d" % shift] = round(j / m, 4) if m else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None, help="local time, ISO format: only replies saved before it")
    ap.add_argument("--files", default=None, help="a text file of reply file names (one per line) to count")
    ap.add_argument("--save-list", default=None, help="write the names of the counted replies to this file")
    ap.add_argument("--out", default=str(ROOT / "results" / "jd_resolve_rate.json"))
    a = ap.parse_args()
    paths = sorted((ROOT / "cache" / "passage").glob("*.json"))
    if a.before:
        cut = dt.datetime.fromisoformat(a.before).timestamp()
        paths = [p for p in paths if p.stat().st_mtime < cut]
    if a.files:
        names = {x.strip() for x in pathlib.Path(a.files).read_text(encoding="utf-8").splitlines() if x.strip()}
        missing = names - {p.name for p in paths}
        if missing:
            raise SystemExit(f"{len(missing)} listed replies are not in cache/passage/")
        paths = [p for p in paths if p.name in names]
    if a.save_list:
        pathlib.Path(a.save_list).write_text("".join(p.name + "\n" for p in paths), encoding="utf-8")
    g = Graph.load()
    whole, line = [], []
    for p in paths:
        q = str(json.loads(p.read_text(encoding="utf-8")).get("question") or "")
        (line if q.endswith(LINE_ASK) else whole).append(p)
    res = {"what": "figures the text reader cites that are nodes of the graph (passage.node_for), over saved replies",
           "command": "python scripts/jd_resolve_rate.py" + (f" --before {a.before}" if a.before else "")
                      + (f" --files {a.files}" if a.files else ""),
           "before": a.before, "file_list": a.files, "files": len(paths),
           "all_replies": count(paths, g),
           "whole_questions": count(whole, g),
           "one_line_of_a_plan": count(line, g)}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for part in ("all_replies", "whole_questions", "one_line_of_a_plan"):
        r = res[part]
        print(f"{part}: replies {r.get('replies', 0)}, answers {r.get('answers', 0)}, cited figures "
              f"{r.get('cited figures', 0)}, nodes {r.get('cited figures that are nodes', 0)} "
              f"({r['share_of_cited_figures_that_are_nodes']}), kinds {r['node_kinds']}, numeric answers "
              f"{r.get('numeric answers', 0)}, one node {r.get('numeric answers that are one node', 0)}")


if __name__ == "__main__":
    main()
