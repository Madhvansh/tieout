"""JD direction 3, access control with provenance: an answer is served only if the asker may read
every leaf of its proof.

Every served answer is a proof tree whose leaves carry the filing and page they were read from
(scripts/check_brief.py shows that each leaf is a node of the graph). So access control is a filter
on leaves, applied after the answer is computed and before it is shown: a leaf is labelled with its
filing and its company; a derived figure (a ratio, a difference, a ranking) carries the labels of
all its leaves; an answer with one leaf the asker may not read is withheld, and the reply names that
leaf. Nothing about the hidden figure leaks: not its value, only where it sits.

    python scripts/jd_access.py measure results/h3_r7.json results/h3_r8.json results/h3_r9.json results/h3_r10.json
    python scripts/jd_access.py ask --allow AAPL --offline "What were Apple's net sales in the fourth quarter of fiscal 2026?"

`measure` replays the gate over saved runs under three policies (the asker may read everything; one
company, the first the question's gold names; every company the question's gold names) and writes
results/jd_access.json. `ask` answers one question (as `python -m tieout ask`) and gates it.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def leaf_labels(leaf: dict, filings: dict) -> set:
    """A leaf's labels: its filing and its company."""
    doc = str(leaf.get("doc") or "")
    ticker = (filings.get(doc) or {}).get("ticker") or doc.split("_")[0]
    return {doc, ticker}


def first_unreadable(leaves: list, allow: Optional[set], filings: dict) -> Optional[dict]:
    """The first proof leaf the asker may not read, or None. allow=None means everything."""
    if allow is None:
        return None
    for lf in leaves:
        if not (leaf_labels(lf, filings) & allow):
            return lf
    return None


def gate(answer: str, leaves: list, allow: Optional[set], filings: dict) -> tuple:
    """(served, what the asker sees). A derived figure carries every leaf's labels, so one unreadable
    leaf withholds it. Under a policy, an answer that shows no source cannot be checked and is withheld."""
    if allow is not None and not leaves:
        return False, "withheld: this answer shows no source, so the asker's access to it cannot be checked"
    lf = first_unreadable(leaves, allow, filings)
    if lf is None:
        return True, answer
    return False, (f"withheld: the answer rests on {lf.get('doc')} page {lf.get('page')}, which this asker may not "
                   "read")


def _filings() -> dict:
    import gzip
    with gzip.open(ROOT / "graph" / "graph.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["filings"]


def _named(bench: pathlib.Path, filings: dict) -> dict:
    """question id -> the companies its gold supporting facts come from, in order of first mention."""
    out = {}
    for line in bench.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        names = []
        for f in q.get("supporting_facts") or []:
            t = leaf_labels({"doc": f.get("doc")}, filings) - {f.get("doc")}
            for x in sorted(t):
                if x and x not in names:
                    names.append(x)
        out[q["id"]] = names
    return out


def _most_cited(files: list, filings: dict) -> str:
    """The company whose filings the served answers cite most often (the hardest one to withhold)."""
    c: collections.Counter = collections.Counter()
    for path in files:
        for r in json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["rows"]:
            if r.get("abstained") or r.get("error"):
                continue
            for t in {x for lf in r.get("leaves") or [] for x in leaf_labels(lf, filings) - {lf.get("doc")}}:
                c[t] += 1
    return c.most_common(1)[0][0] if c else ""


def measure(files: list) -> dict:
    filings = _filings()
    withheld_co = _most_cited(files, filings)
    all_cos = {(v or {}).get("ticker") or k.split("_")[0] for k, v in filings.items()}
    pol = ("everything", "one company", "the companies the question names", f"every company but {withheld_co}")
    stats = {p: collections.Counter() for p in pol}
    examples: dict = {p: [] for p in pol}
    for path in files:
        run = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        bench = pathlib.Path(run["args"]["bench"])
        named = _named(bench if bench.is_absolute() else ROOT / bench, filings)
        for r in run["rows"]:
            if "leaves" not in r:
                stats["everything"]["rows without recorded leaves"] += 1
                continue
            served = not r.get("abstained") and not r.get("error")
            leaves = r.get("leaves") or []
            for p in pol:
                s = stats[p]
                s["questions"] += 1
                if not served:
                    continue
                s["served before the gate"] += 1
                s["right before the gate"] += bool(r.get("correct"))
                allow = (None if p == "everything" else
                         set(named.get(r["id"], [])[:1]) if p == "one company" else
                         all_cos - {withheld_co} if p.startswith("every company but") else
                         set(named.get(r["id"], [])))
                ok, text = gate(r.get("answer", ""), leaves, allow, filings)
                if ok:
                    s["served after the gate"] += 1
                    s["right after the gate"] += bool(r.get("correct"))
                    if first_unreadable(leaves, allow, filings) is not None:
                        s["VIOLATION: served with an unreadable leaf"] += 1
                    continue
                s["withheld"] += 1
                s["withheld and was right"] += bool(r.get("correct"))
                s["withheld and was wrong"] += not r.get("correct")
                if not leaves:
                    s["withheld: no source shown"] += 1
                elif len({lf.get("doc") for lf in leaves}) >= 2:
                    s["withheld: a figure derived from leaves of two or more filings"] += 1
                if len(examples[p]) < 3:
                    examples[p].append(f"{r['id']}: {text}")
    out = {"what": "the gate replayed over saved runs: an answer is served only if the asker may read every proof leaf",
           "command": "python scripts/jd_access.py measure " + " ".join(str(pathlib.Path(f).as_posix()) for f in files),
           "withheld_company": withheld_co, "companies": len(all_cos),
           "policies": {p: dict(stats[p]) for p in pol}, "examples": examples}
    (RES / "jd_access.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for p in pol:
        s = stats[p]
        print(f"{p}: served {s['served before the gate']} -> {s['served after the gate']} (right {s['right before the gate']} -> "
              f"{s['right after the gate']}); withheld {s['withheld']} (right {s['withheld and was right']}, wrong "
              f"{s['withheld and was wrong']}); violations {s['VIOLATION: served with an unreadable leaf']}")
    return out


def ask(question: str, allow: set, offline: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from tieout.__main__ import load_graph
    from tieout.ask import answer
    g = load_graph()
    out = answer(question, g, question_id="cli", offline=offline)
    res = out["result"]
    leaves = [{"doc": lf.doc, "page": lf.page, "printed": lf.printed} for lf in res.leaves]
    ok, text = gate(out["answer"], leaves, allow or None, g.filings)
    print("ANSWER:" if ok else "WITHHELD:", text)
    if ok:
        for lf in leaves:
            print(f"  source: {lf['doc']} page {lf['page']} | printed {lf['printed']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("files", nargs="+")
    k = sub.add_parser("ask")
    k.add_argument("--allow", default="", help="comma-separated tickers or filing ids the asker may read")
    k.add_argument("--offline", action="store_true")
    k.add_argument("question", nargs="+")
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.files)
    else:
        ask(" ".join(a.question), {x.strip() for x in a.allow.split(",") if x.strip()}, a.offline)


if __name__ == "__main__":
    main()
