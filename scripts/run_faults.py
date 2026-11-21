"""Fault-injection study over every parsed primary statement. Writes results/faults.json."""
import json
import logging
import pathlib

from tieout import pdfio
from tieout.eval.faults import study
from tieout.graph import load_manifest
from tieout.statements import find_statements, readable

logging.getLogger("pdfminer").setLevel(logging.ERROR)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    tables, by_kind = [], {"IS": [], "BS": [], "CF": []}
    for m in load_manifest():
        if not pdfio.cache_path(m["doc_id"]).exists():
            continue
        pages = pdfio.load_pages(m["doc_id"])
        if not readable(pages):
            continue
        for kind, st in find_statements(m["doc_id"], pages).items():
            tables.append(st.table)
            by_kind[kind].append(st.table)
    print(len(tables), "statements")
    res = {"all": study(tables)}
    for kind, ts in by_kind.items():
        res[kind] = study(ts)
    out = ROOT / "results" / "faults.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'fault':14s} {'injected':>8s} {'caught':>7s} {'silent':>7s}  catch rate (95% CI)   | IS    BS    CF")
    for f, d in res["all"].items():
        per = "  ".join(f"{res[k][f]['catch_rate']:.2f}" if res[k][f]["catch_rate"] is not None else " n/a" for k in ("IS", "BS", "CF"))
        print(f"{f:14s} {d['injected']:8d} {d['caught']:7d} {d['silent']:7d}  {d['catch_rate']:.3f} {d['ci95']}   | {per}")


if __name__ == "__main__":
    main()
