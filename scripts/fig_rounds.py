"""Figure: accuracy on each round of blind questions, the first time the system saw it and after the
misses were fixed by class, with the held-out round beside them.

    python scripts/fig_rounds.py            # writes report/fig_rounds.png and results/rounds.json

Every bar is read from a result file in results/ (named in ROUNDS below). A first run is the one run
made before any fix for that round; "after" is the same round on the frozen code (results/c2_final_r<N>.json,
the last pass over every round in order), when it had become a development set. Round 9's misses were read
only after the frozen code was complete, so no fix was aimed at it. The held-out round has a first run
only: nothing was fixed against it.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
# (label, first-run result file, final-code result file or None, note)
ROUNDS = [
    ("round 3", "first_runs/round3.json", "c2_final_r3.json", ""),
    ("round 4", "first_runs/round4.json", "c2_final_r4.json", ""),
    ("round 5", "first_runs/round5.json", "c2_final_r5.json", "arbiter quota spent"),
    ("round 6", "first_runs/round6.json", "c2_final_r6.json", "27 of 35 run"),
    ("round 7", "first_runs/round7.json", "c2_final_r7.json", ""),
    ("round 8", "first_runs/round8.json", "c2_final_r8.json", ""),
    ("round 9", "first_runs/round9.json", "c2_final_r9.json", "no fix aimed at it"),
    ("round 10", "first_runs/round10_pointer.json", None, "text reader as pointer"),
    ("held-out", "heldout_v2.json", None, "run once"),
]
TIERS = ("VERIFIED", "WITNESSED", "CORROBORATED", "FLAGGED")
SHADE = {"VERIFIED": "#1b5e20", "WITNESSED": "#43a047", "CORROBORATED": "#81c784", "FLAGGED": "#dcedc8"}


def read(name: str):
    path = ROOT / "results" / name
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    right = [r for r in rows if r.get("correct")]
    by_tier = {t: sum(1 for r in right if r.get("tier") == t) for t in TIERS}
    by_tier["ABSTAIN"] = sum(1 for r in right if r.get("tier") == "ABSTAIN")   # a right refusal of an unanswerable question
    return {"n": len(rows), "right": len(right), "right_by_tier": by_tier,
            "wrong_by_tier": {t: sum(1 for r in rows if not r.get("correct") and r.get("tier") == t)
                              for t in TIERS + ("ABSTAIN",)}}


def main() -> None:
    table = []
    for label, first, final, note in ROUNDS:
        table.append({"round": label, "note": note, "first_run": read(first), "final_code": read(final) if final else None,
                      "first_file": first, "final_file": final})
    (ROOT / "results" / "rounds.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    shown = [t for t in table if t["first_run"]]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    width = 0.38
    for i, t in enumerate(shown):
        first = t["first_run"]
        bottom = 0.0
        for tier in TIERS + ("ABSTAIN",):               # the first run, split by the tier of each right answer
            share = first["right_by_tier"].get(tier, 0) / first["n"]
            ax.bar(i - width / 2, share, width, bottom=bottom, color=SHADE.get(tier, "#eeeeee"), edgecolor="#2e7d32",
                   linewidth=0.4, label=tier.lower() if i == 0 and tier in SHADE else None)
            bottom += share
        ax.text(i - width / 2, bottom + 0.015, f"{first['right']}/{first['n']}", ha="center", va="bottom", fontsize=6.5)
        if t["final_code"]:
            final = t["final_code"]
            share = final["right"] / final["n"]
            ax.bar(i + width / 2, share, width, color="#90a4ae", edgecolor="#546e7a", linewidth=0.4,
                   label="same round, final code" if i == 0 else None)
            ax.text(i + width / 2, share + 0.015, f"{final['right']}/{final['n']}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([t["round"] + (f"\n({t['note']})" if t["note"] else "") for t in shown], fontsize=6.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of questions right", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2),
              title="first run on unseen questions, right answers by tier", title_fontsize=6.5)
    fig.savefig(ROOT / "report" / "fig_rounds.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ROOT / "report" / "fig_rounds.pdf", bbox_inches="tight", pad_inches=0.03)
    for t in table:
        first, final = t["first_run"], t["final_code"]
        print(t["round"], "first", f"{first['right']}/{first['n']}" if first else "-", "| final",
              f"{final['right']}/{final['n']}" if final else "-")


if __name__ == "__main__":
    main()
