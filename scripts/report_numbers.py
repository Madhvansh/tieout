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
    ("TieOut, the declared configuration on one free key (the headline)", "heldout_v2.json", ""),
    ("Baseline: hybrid retrieval, the same model reads 16,000 characters", "heldout_baseline_hybrid.json", ""),
    ("Baseline: the same with 48,000 characters, as the text reader inside TieOut gets", "heldout_baseline_hybrid_b48.json",
     "baseline"),
    ("Baseline: dense retrieval, the same model reads", "heldout_baseline_dense.json", "baseline"),
    ("Paid side run: Claude Sonnet 5.5 plans and picks, gemini-3.1-pro arbitrates", "heldout_v2_scaled.json", "scaled"),
    # The ablations ran on the development key, where the stronger arbiter models answer. On the free key they
    # answered three calls and stopped, so the like-for-like reference of the rows below is this run, not the headline
    ("Reference for the rows below: TieOut on a key without the free tier's daily limit", "heldout_abl_none.json", "reference"),
    ("... one planner (no second planner, no arbiter)", "heldout_abl_onereader.json", "ablation"),
    ("... no arbiter", "heldout_abl_noarbiter.json", "ablation"),
    ("... no linking (statement concepts only)", "heldout_abl_link.json", "ablation"),
    ("... one picker (no second picker)", "heldout_abl_onepicker.json", "ablation"),
    ("... empty link store", "heldout_abl_store.json", "ablation"),
    ("... no equation solving", "heldout_abl_solve.json", "ablation"),
    ("... checks off", "heldout_abl_checks.json", "ablation"),
    ("... no checks read from the question's words", "heldout_abl_words.json", "ablation"),
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


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: the discordant pairs b and c under Binomial(b + c, 1/2)."""
    import math
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


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
        if name == "heldout_v2.json" and any(r.get("plan_road") for r in d.get("rows", [])):
            # the same run with the second road taken away: what the plan road alone served, question by question
            by = {t: [0, 0] for t in TYPES}
            for r in d["rows"]:
                for t in (r.get("type"), "overall"):
                    if t in by:
                        by[t][0] += bool((r.get("plan_road") or {}).get("correct"))
                        by[t][1] += 1
            show = {t: f"{k}/{n}" for t, (k, n) in by.items()}
            rows_tex.append(f"TieOut, plans alone (no text reader) & {show['single_fact']} & {show['multi_hop']} & "
                            f"{show['cross_filing']} & \\textbf{{{show['overall']}}} & -- & -- \\\\")
            summary["systems"].append({"label": "TieOut, plans alone (no text reader)", "file": name,
                                       "kind": "ablation", **show})
        if kind == "baseline":
            rows_tex.append("\\midrule")
    held = load("heldout_v2.json")
    tier_line = cost_line = case_line = "n/a"
    if held:
        o = held["score"]["overall"]
        bt = o["by_tier"]
        tier_line = ", ".join(f"{t} {bt[t]['correct']}/{bt[t]['n']}"
                              for t in ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED", "ABSTAIN") if t in bt)
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
    # round 8 on one code version (d776eb9), with the seed store and with the links learned on rounds 1 to 7
    for key, name in (("empty", "nolinks_round8.json"), ("learned", "store_round8.json")):
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
    # 23 Nov (Session A): both scorers, the exact McNemar test against the baseline, the scaled side
    # experiment and the store's transfer split
    strict_line = mcnemar_line = scaled_line = transfer_line = road_line = "n/a"
    if held:
        o = held["score"]["overall"]
        if o.get("plan_road_correct") is not None:      # 23 Nov: the second road; every row keeps the plan road's answer
            used = {}
            for r in held["rows"]:
                how = (r.get("passage") or {}).get("used")
                if how:
                    used[how] = used.get(how, 0) + 1
            road_line = (f"the plan road alone {o['plan_road_correct']}/{o['n']}; with the text reader as witness and pointer "
                         f"{o['correct']}/{o['n']}")
            summary["second_road"] = {"plan_road_alone": o["plan_road_correct"], "with_second_road": o["correct"],
                                      "n": o["n"], "how_the_passage_reader_was_used": used}
        strict_line = (f"final scorer {o.get('correct_final', o['correct'])}/{o['n']}, strict scorer (the 21 Nov rules) "
                       f"{o.get('correct_strict', o['correct'])}/{o['n']}, 95\\% interval (final) "
                       f"{pct(o['accuracy_ci95'][0])} to {pct(o['accuracy_ci95'][1])}")
        base = load("heldout_baseline_hybrid.json") or load("heldout_baseline_dense.json")
        if base:
            mine = {r["id"]: bool(r.get("correct")) for r in held["rows"]}
            theirs = {r["id"]: bool(r.get("correct")) for r in base.get("rows", []) if "id" in r}
            ids = sorted(set(mine) & set(theirs))
            b = sum(1 for i in ids if mine[i] and not theirs[i])
            c = sum(1 for i in ids if theirs[i] and not mine[i])
            p = mcnemar_exact(b, c)
            summary["mcnemar_vs_dense_baseline"] = {"n": len(ids), "only_tieout": b, "only_baseline": c, "p_two_sided": p}
            mcnemar_line = (f"on the {len(ids)} held-out questions TieOut alone is right on {b}, the retrieval baseline alone on "
                            f"{c}; exact McNemar two-sided p = {p:.4f}")
    scaled = load("heldout_v2_scaled.json")
    if scaled:
        o = scaled["score"]["overall"]
        scaled_line = (f"labelled side experiment, paid models (Claude Sonnet 5.5 reads and picks, gemini-3.1-pro "
                       f"arbitrates): {o['correct']}/{o['n']} (strict {o.get('correct_strict', o['correct'])}/{o['n']})")
        summary["heldout_scaled"] = {"correct": o["correct"], "n": o["n"], "correct_strict": o.get("correct_strict")}
    transfer = load("transfer_heldout.json")
    if transfer:
        gr = transfer["groups"]
        t, u = gr.get("touches the store", {}), gr.get("does not touch it", {})
        transfer_line = (f"with the learned store {gr['all']['with_store']}/{gr['all']['n']}, without it "
                         f"{gr['all']['without_store']}/{gr['all']['n']}; on the {t.get('n', 0)} questions that touch a "
                         f"stored pair {t.get('with_store')} vs {t.get('without_store')}, on the other {u.get('n', 0)} "
                         f"{u.get('with_store')} vs {u.get('without_store')}")
        summary["transfer"] = transfer
    round10_line = "n/a"
    # Round 10 was run twice before anyone read its answers: first by a version that served some answers the
    # text reader had written (first_runs/round10.json; withdrawn the same night, it broke the brief), then
    # by the design in which the reader can only point at a graph node (first_runs/round10_pointer.json).
    r10, withdrawn = load("first_runs/round10_pointer.json"), load("first_runs/round10.json")
    if r10:
        o = r10["score"]["overall"]
        bt = o["by_tier"]
        round10_line = (f"Round 10 (35 questions nobody had seen; first run of the design in which the text reader can "
                        f"only point at a graph node): the plans alone {o.get('plan_road_correct')}/{o['n']}, with the "
                        f"text reader {o['correct']}/{o['n']}; "
                        + ", ".join(f"{t} {bt[t]['correct']}/{bt[t]['n']}"
                                    for t in ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED", "ABSTAIN") if t in bt) + "."
                        + (f" A version withdrawn the same night, which served answers the reader had written, scored "
                           f"{withdrawn['score']['overall']['correct']}/{o['n']}." if withdrawn else ""))
        summary["round10_first_run"] = {"plan_road_alone": o.get("plan_road_correct"), "with_text_reader": o["correct"],
                                        "n": o["n"], "by_tier": bt, "correct_strict": o.get("correct_strict"),
                                        "withdrawn_version": withdrawn["score"]["overall"]["correct"] if withdrawn else None}
    macros = {"HeldRows": "\n".join(rows_tex), "TierLine": tex(tier_line), "CostLine": tex(cost_line),
              "RoundTenLine": tex(round10_line),
              "CaseLine": tex(case_line), "RoundRows": "\n".join(round_rows), "StoreLine": tex(store_line),
              "ScaleLine": tex(scale_line), "StrictLine": strict_line, "McNemarLine": tex(mcnemar_line),
              "ScaledLine": tex(scaled_line), "TransferLine": tex(transfer_line), "RoadLine": tex(road_line)}
    (ROOT / "report" / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rounds"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
