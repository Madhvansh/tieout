"""Collect every number the README and the results PDF quote, from the result files, in one place.

    python scripts/report_numbers.py        # writes results/summary.json and report/numbers.tex

Nothing here is typed by hand: each figure is read from a file in results/ (named beside it in
summary.json). A file that does not exist yet gives "n/a", so the report builds at any time and
shows what is still missing.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TYPES = ("single_fact", "multi_hop", "cross_filing", "overall")
# (row label in the table, result file, what was switched off)
SYSTEMS = [
    ("TieOut, declared configuration", "heldout_v2.json", ""),
    ("Baseline: dense retrieval, same model reads", "heldout_baseline_dense.json", "baseline"),
    ("TieOut, one reader (no second reader, no arbiter)", "heldout_abl_onereader.json", "ablation"),
    ("TieOut, one picker (no second picker)", "heldout_abl_onepicker.json", "ablation"),
    ("TieOut, no arbiter", "heldout_abl_noarbiter.json", "ablation"),
    ("TieOut, empty link store", "heldout_abl_store.json", "ablation"),
    ("TieOut, no equation solving", "heldout_abl_solve.json", "ablation"),
    ("TieOut, no linking (statement concepts only)", "heldout_abl_link.json", "ablation"),
    ("TieOut, checks off", "heldout_abl_checks.json", "ablation"),
    ("TieOut, no checks read from the question's words", "heldout_abl_words.json", "ablation"),
]


def load(name: str):
    path = RES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def score_of(d: dict) -> dict:
    """The per-type score table of a result file (the graph system's or the baseline's)."""
    return d.get("score") or d.get("score_fact_level") or {}


def cell(score: dict, kind: str) -> str:
    s = score.get(kind)
    return f"{s['correct']}/{s['n']}" if s else "n/a"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}\\%"


def tex(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_") if "\\" not in s else s


def calls(d: dict) -> dict:
    rows = d.get("rows", [])
    n = max(1, len(rows))
    llm = d.get("llm") or {}
    strong = sum(v for r in rows for m, v in (r.get("calls_by_model") or {}).items()
                 if "lite" not in m and m.startswith("gemini"))
    return {"calls_per_question": round((llm.get("live", 0) + llm.get("cached", 0)) / n, 1),
            "seconds_per_question": round(sum(r.get("seconds", 0.0) for r in rows) / n, 1),
            "strong_model_calls": strong}


def main() -> None:
    summary: dict = {"systems": [], "files_missing": []}
    rows_tex = []
    for label, name, kind in SYSTEMS:
        d = load(name)
        if d is None:
            summary["files_missing"].append(name)
            rows_tex.append(f"{tex(label)} & n/a & n/a & n/a & n/a & n/a & n/a \\\\")
            continue
        sc = score_of(d)
        o = sc.get("overall", {})
        summary["systems"].append({"label": label, "file": name, "kind": kind,
                                   **{t: cell(sc, t) for t in TYPES},
                                   "precision_macro": o.get("precision_macro"), "recall_macro": o.get("recall_macro"),
                                   "ci95": o.get("accuracy_ci95")})
        rows_tex.append(f"{tex(label)} & {cell(sc, 'single_fact')} & {cell(sc, 'multi_hop')} & {cell(sc, 'cross_filing')} & "
                        f"\\textbf{{{cell(sc, 'overall')}}} & {pct(o.get('precision_macro'))} & {pct(o.get('recall_macro'))} \\\\")
        if kind == "baseline":
            rows_tex.append("\\midrule")
    held = load("heldout_v2.json")
    tier_line = cost_line = case_line = "n/a"
    if held:
        o = held["score"]["overall"]
        bt = o["by_tier"]
        tier_line = ", ".join(f"{t} {bt[t]['correct']}/{bt[t]['n']}" for t in ("VERIFIED", "WITNESSED", "FLAGGED", "ABSTAIN") if t in bt)
        tier_line += f"; wrong while VERIFIED: {o.get('silent_wrong', 0)}; wrong while WITNESSED: {o.get('witnessed_wrong', 0)}"
        c = calls(held)
        cost_line = (f"{c['calls_per_question']} model calls and {c['seconds_per_question']} s per question; "
                     f"{c['strong_model_calls']} strong-model calls in the run")
        cases: dict = {}
        for r in held["rows"]:
            reading = r.get("reading")
            name = reading.get("case") if isinstance(reading, dict) else reading
            cases.setdefault(name, [0, 0])
            cases[name][0] += bool(r.get("correct"))
            cases[name][1] += 1
        case_line = "; ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(cases.items(), key=lambda kv: -kv[1][1]))
        summary["heldout"] = {"tiers": tier_line, "cost": c, "reading_cases": cases, "ci95": o.get("accuracy_ci95"),
                              "answered_accuracy": o.get("answered_accuracy")}
    rounds = load("rounds.json") or []
    round_rows = []
    for t in rounds:
        first, final = t.get("first_run"), t.get("final_code")
        round_rows.append(f"{t['round']} & {first['right']}/{first['n'] if first else ''} & "
                          f"{(str(final['right']) + '/' + str(final['n'])) if final else '--'} \\\\" if first else
                          f"{t['round']} & n/a & n/a \\\\")
    summary["rounds"] = rounds
    store = {}
    for key, name in (("empty", "first_runs/round8.json"), ("learned", "store_round8.json")):
        d = load(name)
        if d:
            store[key] = {"right": d["score"]["overall"]["correct"], "n": d["score"]["overall"]["n"], **calls(d),
                          "live_calls": (d.get("llm") or {}).get("live")}
    summary["store_experiment"] = store
    store_line = "n/a"
    if len(store) == 2:
        e, l = store["empty"], store["learned"]
        store_line = (f"empty store {e['right']}/{e['n']}, {e['calls_per_question']} calls and {e['seconds_per_question']} s per "
                      f"question; with links learned on the earlier rounds {l['right']}/{l['n']}, "
                      f"{l['calls_per_question']} calls and {l['seconds_per_question']} s")
    scale = load("scale_test.json")
    scale_line = "n/a"
    if scale:
        o = (scale.get("summary") or scale).get("overall") if isinstance(scale, dict) else None
        if o:
            scale_line = f"{o.get('n')} questions: {o.get('correct')} correct, {o.get('abstained')} abstained, {o.get('silent')} silent wrong"
            summary["scale_test"] = o
    macros = {"HeldRows": "\n".join(rows_tex), "TierLine": tex(tier_line), "CostLine": tex(cost_line),
              "CaseLine": tex(case_line), "RoundRows": "\n".join(round_rows), "StoreLine": tex(store_line),
              "ScaleLine": tex(scale_line)}
    (ROOT / "report" / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rounds"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
