"""Write the numbers of README.md and the PDF's held-out table from the result files, never by hand.

    python scripts/readme_numbers.py [--dev-prefix h3]

Rewrites every README block between `<!-- gen:NAME ... -->` and `<!-- /gen:NAME -->` and writes
report/tables.tex (macros HeldTypeRows, HeldNotes, DevLine, RoundsRows). A number whose result file
does not exist yet is written as [[...]], so the README is complete at every moment and shows what is
still missing. Every figure is read from a file in results/, named beside it.

Blocks:
  heldout   accuracy, precision and recall by question type on the 30 held-out questions
            (results/heldout_v2.json), then the baseline, the plans alone, the tiers, the strict scorer,
            the brief check and the paid side experiment on the same questions
  devline   the four latest development rounds on the current code (results/<prefix>_r7..r10.json)
  rounds    every fresh round: its first run (results/first_runs/) against the current code
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = ROOT / "README.md"
GAP = "[[...]]"
TYPES = (("single_fact", "Single fact"), ("multi_hop", "Multi-hop"), ("cross_filing", "Cross-filing comparison"),
         ("overall", "**Overall**"))
TIERS = ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED", "ABSTAIN")


def load(name: str):
    path = RES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x) -> str:
    return GAP if x is None else f"{100 * x:.0f}%"


def score(d) -> dict:
    return (d or {}).get("score") or (d or {}).get("score_fact_level") or {}


def brief_check(files: list) -> dict:
    """{file name: (served, leaves, nodes, outside)} from scripts/check_brief.py, run on the files that exist."""
    have = [f for f in files if (RES / f).exists()]
    if not have:
        return {}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_brief.py"), *[str(RES / f) for f in have]],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"})
    found = {}
    for line in out.stdout.splitlines():
        m = re.match(r"(\S+?\.json): (\d+) answers served, (\d+) proof leaves, (\d+) are graph nodes, (\d+) outside", line)
        if m:
            found[m.group(1)] = tuple(int(m.group(i)) for i in range(2, 6))
    return found


def heldout_block() -> tuple:
    held, base = load("heldout_v2.json"), load("heldout_baseline_hybrid.json")
    sc, bsc = score(held), score(base)
    lines = ["| Question type | Questions | Accuracy | Precision | Recall |", "|---|---|---|---|---|"]
    tex = []
    for key, label in TYPES:
        s, b = sc.get(key), bsc.get(key)
        if s:
            acc = f"{s['correct']}/{s['n']} ({pct(s['accuracy'])})"
            row = [str(s["n"]), acc, pct(s.get("precision_macro")), pct(s.get("recall_macro"))]
        else:
            row = [GAP] * 4
        lines.append(f"| {label} | " + " | ".join(row) + " |")
        base_acc = f"{b['correct']}/{b['n']}" if b else GAP
        tex.append(f"{label.strip('*')} & {row[0]} & {row[1].replace('%', chr(92) + '%')} & "
                   f"{row[2].replace('%', chr(92) + '%')} & {row[3].replace('%', chr(92) + '%')} & {base_acc} \\\\")
    notes = []
    o = sc.get("overall")
    bo = bsc.get("overall")
    plans = o.get("plan_road_correct") if o else None
    notes.append(f"On the same 30 questions: the retrieval baseline (passages, the same free model reads) "
                 f"{bo['correct']}/{bo['n'] if bo else GAP}" if bo else
                 "On the same 30 questions: the retrieval baseline (passages, the same free model reads) " + GAP)
    notes[-1] += (f"; TieOut's plans alone, without the text reader, {plans}/{o['n']}" if plans is not None
                  else f"; TieOut's plans alone, without the text reader, {GAP}")
    notes[-1] += " (`results/heldout_v2.json`, `results/heldout_baseline_hybrid.json`)."
    if o:
        bt = o.get("by_tier", {})
        tiers = ", ".join(f"{t} {bt[t]['correct']}/{bt[t]['n']}" for t in TIERS if bt.get(t, {}).get("n"))
        notes.append(f"Right by tier: {tiers}. Wrong while VERIFIED: {o.get('silent_wrong', 0)}.")
        lo, hi = o.get("accuracy_ci95") or (None, None)
        notes.append(f"Under the strict scorer (the rules of 21 November, before six were relaxed after misses were "
                     f"read): {o.get('correct_strict', o['correct'])}/{o['n']}. "
                     f"95% interval of the accuracy: {pct(lo)} to {pct(hi)}.")
    else:
        notes.append(f"Right by tier: {GAP}. Under the strict scorer (the rules of 21 November, before six were "
                     f"relaxed after misses were read): {GAP}.")
    chk = brief_check(["heldout_v2.json"]).get("heldout_v2.json")
    notes.append(f"`python scripts/check_brief.py results/heldout_v2.json`: {chk[0]} answers served, {chk[1]} proof "
                 f"leaves, {chk[2]} of them graph nodes." if chk else
                 f"`python scripts/check_brief.py results/heldout_v2.json`: {GAP}.")
    scaled = score(load("heldout_v2_scaled.json")).get("overall")
    notes.append("Paid side experiment, is the model the bottleneck? (Claude Sonnet 5.5 plans and picks, "
                 "gemini-3.1-pro arbitrates, labelled and never mixed into the headline): "
                 + (f"{scaled['correct']}/{scaled['n']} (`results/heldout_v2_scaled.json`)." if scaled else GAP + "."))
    md = "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes)
    return md, "\n".join(tex), " ".join(notes)


def dev_block(prefix: str) -> str:
    files = [f"{prefix}_r{n}.json" for n in (7, 8, 9, 10)]
    rows = [score(load(f)).get("overall") for f in files]
    if not all(rows):
        return f"On the four latest development rounds on the current code: {GAP}."
    k = sum(r["correct"] for r in rows)
    n = sum(r["n"] for r in rows)
    plans = sum(r.get("plan_road_correct") or 0 for r in rows)
    chk = brief_check(files)
    passed = sum(1 for f in files if f in chk and chk[f][3] == 0)
    last = chk.get(files[-1])
    text = (f"On the four latest development rounds ({n} questions, `results/{files[0]}` to `{files[-1]}`), the plans "
            f"alone were right on {plans} and the whole system on {k}. `check_brief` passes on {passed} of the 4 files")
    text += (f" (round 10: {last[0]} answers served, {last[1]} proof leaves, {last[2]} of them graph nodes)." if last else ".")
    return text


def rounds_block(prefix: str) -> tuple:
    """First runs of rounds 3 to 9 as rescored under both scorers (results/pooled_first_runs.json); round 10 from
    the first run of the corrected design (results/first_runs/round10_pointer.json); the current code's runs beside."""
    lines = ["| Fresh round | First run, final scorer | First run, strict scorer | Current code |", "|---|---|---|---|"]
    tex = []
    tot = [0, 0, 0, 0, 0]
    pooled = {r["round"]: r for r in (load("pooled_first_runs.json") or {}).get("rounds", [])}
    for r in range(3, 12):
        p = pooled.get(r) if r != 10 else None
        if p:
            n, fin, strict = p["n"], p["first_run_final"], p["first_run_strict"]
        else:
            fo = score(load(f"first_runs/round{r}_pointer.json") or load(f"first_runs/round{r}.json")).get("overall")
            if not fo:
                continue
            n, fin, strict = fo["n"], fo.get("correct_final", fo["correct"]), fo.get("correct_strict", fo["correct"])
        now = score(load(f"{prefix}_r{r}.json")).get("overall")
        cur = f"{now['correct']}/{now['n']}" if now else GAP
        lines.append(f"| {r} | {fin}/{n} | {strict}/{n} | {cur} |")
        tex.append(f"{r} & {fin}/{n} & {strict}/{n} & {cur} \\\\")
        tot[0] += n
        tot[1] += fin
        tot[2] += strict
        if now:
            tot[3] += now["correct"]
            tot[4] += now["n"]
    if tot[0]:
        lines.append(f"| **All** | {tot[1]}/{tot[0]} ({pct(tot[1] / tot[0])}) | {tot[2]}/{tot[0]} ({pct(tot[2] / tot[0])}) | "
                     f"{tot[3]}/{tot[4]} ({pct(tot[3] / tot[4]) if tot[4] else GAP}) |")
        tex.append(f"All & {tot[1]}/{tot[0]} & {tot[2]}/{tot[0]} & {tot[3]}/{tot[4]} \\\\")
    note = (f"Files: `results/pooled_first_runs.json` (first runs rescored under both scorers), "
            f"`results/first_runs/`, `results/{prefix}_r<N>.json` (current code). Round 10's row is the first run of "
            "the corrected design, made before anyone read the round's answers; a version withdrawn the same night "
            "(it served answers the text reader wrote) had run it once before.")
    judged = score(load("first_runs/round11_judge.json")).get("overall")
    if judged:                                        # round 11 was answered twice before anyone read an answer
        note += (f" Round 11's row is its first run, on the code before the judge between two derivations was merged; "
                 f"run once more with the judge, still before anyone had read an answer, it scored "
                 f"{judged['correct']}/{judged['n']} (`results/first_runs/round11_judge.json`).")
    return "\n".join(lines) + "\n\n" + note, "\n".join(tex)


def other_rows() -> str:
    """The other systems and ablations on the held-out questions, as scripts/report_numbers.py lists them, keeping
    only the runs whose result file exists (so the PDF shows no row of n/a)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import report_numbers as rn                       # the one list of systems and their files
    rows = []
    for label, name, kind in rn.SYSTEMS:
        d = rn.load(name)
        if d is None:
            continue
        sc = rn.score_of(d)
        o = sc.get("overall", {})
        rows.append(f"{rn.tex(label)} & {rn.cell(sc, 'single_fact')} & {rn.cell(sc, 'multi_hop')} & "
                    f"{rn.cell(sc, 'cross_filing')} & \\textbf{{{rn.cell(sc, 'overall')}}} & "
                    f"{rn.pct(o.get('precision_macro'))} & {rn.pct(o.get('recall_macro'))} \\\\")
        if name == "heldout_v2.json" and any(r.get("plan_road") for r in d.get("rows", [])):
            by = {t: [0, 0] for t in ("single_fact", "multi_hop", "cross_filing", "overall")}
            for r in d["rows"]:
                for t in (r.get("type"), "overall"):
                    if t in by:
                        by[t][0] += bool((r.get("plan_road") or {}).get("correct"))
                        by[t][1] += 1
            s = {t: f"{k}/{n}" for t, (k, n) in by.items()}
            rows.append(f"TieOut, its plans alone (no text reader) & {s['single_fact']} & {s['multi_hop']} & "
                        f"{s['cross_filing']} & \\textbf{{{s['overall']}}} & -- & -- \\\\")
    return "\n".join(rows) or "[[...]] (run after the freeze) & & & & & & \\\\"


def put(text: str, name: str, body: str) -> str:
    pat = re.compile(r"(<!-- gen:" + re.escape(name) + r"\b[^>]*-->\n)(.*?)(<!-- /gen:" + re.escape(name) + r" -->)",
                     re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no block gen:{name}")
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(3), text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-prefix", default="h3", help="result files of the current code: results/<prefix>_r<N>.json")
    args = ap.parse_args()
    held_md, held_tex, held_notes = heldout_block()
    dev = dev_block(args.dev_prefix)
    rounds_md, rounds_tex = rounds_block(args.dev_prefix)
    text = README.read_text(encoding="utf-8")
    for name, body in (("heldout", held_md), ("devline", dev), ("rounds", rounds_md)):
        text = put(text, name, body)
    README.write_text(text, encoding="utf-8")
    esc = (lambda s: s.replace("%", "\\%").replace("_", "\\_").replace("`", ""))
    macros = {"HeldTypeRows": held_tex, "HeldNotes": esc(held_notes), "DevLine": esc(dev), "RoundsRows": rounds_tex,
              "OtherRows": other_rows()}
    (ROOT / "report" / "tables.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")
    gaps = text.count(GAP)
    print(f"README.md: blocks written; {gaps} gaps [[...]] left; report/tables.tex written; "
          f"README lines {len(text.splitlines())}")


if __name__ == "__main__":
    main()
