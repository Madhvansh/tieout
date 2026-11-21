"""Write the two sheets a person signs off:

benchmark/CHECK_SHEET_30.md   the 30 benchmark questions with every gold supporting fact and its PDF page
semantics/REVIEW_SHEET.md     every binding the living layer admitted or held, with its evidence, for
                              approve / reject. Bindings the SEC audit disagrees with are listed first.
"""
import json
import pathlib

from tieout.graph import Graph
from tieout.items import load_items
from tieout.living import load_learned

ROOT = pathlib.Path(__file__).resolve().parents[1]


def gold_sheet():
    rows = [json.loads(x) for x in (ROOT / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = ["# Gold check sheet: the 30 benchmark questions", "",
           "For each question open the PDF at the page given and confirm the printed value. Tick or correct.",
           "PDFs are in `data/pdf/<document>.pdf`. Page numbers are PDF page numbers (what the viewer shows), not printed ones.", ""]
    for q in rows:
        out += [f"## {q['id']} ({q['type']})", "", f"Question: {q['question']}", "", f"Gold answer: {q['answer']}", ""]
        for f in q.get("supporting_facts", []):
            out.append(f"- [ ] `{f['doc']}.pdf` page {f['page']}: \"{f.get('label', '')}\" = {f.get('printed', '')}"
                       f" ({f.get('period', '')}{', in ' + f['scale'] if f.get('scale') else ''})")
        if q.get("derivation"):
            out += ["", f"Derivation: {q['derivation']}"]
        if q.get("notes"):
            out += ["", f"Note: {q['notes']}"]
        out.append("")
    (ROOT / "benchmark" / "CHECK_SHEET_30.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote benchmark/CHECK_SHEET_30.md", len(rows), "questions")


def binding_sheet():
    g = Graph.load()
    load_learned(g)
    load_items(g)
    log = [json.loads(x) for x in (ROOT / "results" / "teaching_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    audit = json.loads((ROOT / "results" / "xbrl_audit.json").read_text(encoding="utf-8")) if (ROOT / "results" / "xbrl_audit.json").exists() else {"disagreements": []}
    flagged = {(d["ticker"], d["concept"]) for d in audit["disagreements"]}

    def latest_value(ticker, concept):
        ps = [p for p in g.periods_of(ticker, concept) if "Q" not in p]
        if not ps:
            return ""
        f = g.fact(ticker, concept, ps[-1])
        r = g.best_reading(f)
        return f"{ps[-1]}: {r.text} (page {r.page} of {r.doc})"

    out = ["# Binding review sheet", "",
           "Each line is a binding the living layer proposed from a question. The structural gate judged it;",
           "a person has the last word. Mark each `approve` or `reject` (with a reason) in",
           "`semantics/review_decisions.yaml`. A rejected binding is removed and the count is reported.", "",
           "## A. Admitted by the gate, but the SEC audit disagrees (look at these first)", ""]
    admitted = [e for e in log if e["state"] == "active"]
    held = [e for e in log if e["state"] == "proposed"]
    first = [e for e in admitted if (e["ticker"], e["concept"]) in flagged]
    rest = [e for e in admitted if (e["ticker"], e["concept"]) not in flagged]
    for title, group in (("", first), ("## B. Admitted by the gate", rest), ("## C. Held by the gate (not served until approved)", held)):
        if title:
            out += [title, ""]
        for e in group:
            out.append(f"- [ ] {e['ticker']} {e['concept']}: printed as \"{e['label']}\"")
            out.append(f"  - latest value: {latest_value(e['ticker'], e['concept'])}")
            for x in e["evidence"]:
                out.append(f"  - for: {x}")
            for x in e["objections"]:
                out.append(f"  - against: {x}")
        out.append("")
    (ROOT / "semantics" / "REVIEW_SHEET.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote semantics/REVIEW_SHEET.md:", len(first), "flagged,", len(rest), "admitted,", len(held), "held")


if __name__ == "__main__":
    gold_sheet()
    binding_sheet()
