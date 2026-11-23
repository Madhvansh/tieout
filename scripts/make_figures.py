"""Draw the figure for the results PDF from the result files. Writes report/fig_results.pdf and .png.

Panel A  correct answers out of 30 for the system, its ablations and the baseline (one series, one hue)
Panel B  correct answers as a share of statement cells is damaged, under the two answering policies
Panel C  wrong answers in the same experiment
Colours are the two first categorical slots of a validated palette (blue, orange); text stays in ink.
"""
import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"


def correct(name: str) -> int:
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))["score"]["overall"]["correct"]


def baseline_correct() -> int:
    b = json.loads((ROOT / "results" / "baseline.json").read_text(encoding="utf-8"))
    return sum(1 for r in b["rows"] if r["correct"])


def noise(name: str) -> dict:
    rows = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy"], r["rate"])].append(r)
    out = defaultdict(dict)
    for (policy, rate), rs in agg.items():
        out[policy][rate] = {k: sum(r[k] for r in rs) / len(rs) for k in ("correct", "wrong", "abstain")}
    return out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=6.5, length=2, width=0.5)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.3, 2.25), gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.5})

    # ---- A: one series, sorted, value at the end of each bar
    bars = [("Hand-written plans (blind)", correct("benchmark_oracle")),
            ("TieOut, frozen first run", correct("benchmark_tieout")),
            ("Living layer off", correct("tieout_no-living_seed_only")),
            ("Derivations off", correct("tieout_no-derivations")),
            ("Concept layer off", correct("tieout_no-concepts")),
            ("Chunk-retrieval baseline", baseline_correct())]
    ys = list(range(len(bars)))[::-1]
    a.barh(ys, [v for _, v in bars], height=0.56, color=BLUE, edgecolor="white", linewidth=1.2)
    for y, (label, v) in zip(ys, bars):
        a.text(v + 0.5, y, str(v), va="center", ha="left", color=INK, fontsize=7,
               fontweight="bold" if label.startswith("TieOut") else "normal")
    a.set_yticks(ys)
    a.set_yticklabels([label for label, _ in bars], color=INK2, fontsize=6.8)
    a.set_xlim(0, 32)
    a.set_xticks([0, 10, 20, 30])
    a.xaxis.grid(True, color=GRID, linewidth=0.5)
    a.set_title("Correct answers out of 30 held-out questions", loc="left", color=INK, fontsize=7.5, pad=6)
    style(a)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0, labelcolor=INK2)

    # ---- B, C: two series (answering policy); legend plus a direct label at each line's end
    data = noise("noise_frozen_v1")
    rates = sorted(data["serve_all"])
    xs = [r * 100 for r in rates]
    for ax, key, title, ymax in ((b, "correct", "Correct answers, damaged graph", 30),
                                 (c, "wrong", "Wrong answers, damaged graph", 8)):
        for policy, colour, name in (("serve_all", BLUE, "serve everything"), ("verified_only", ORANGE, "verified only")):
            vals = [data[policy][r][key] for r in rates]
            ax.plot(xs, vals, color=colour, linewidth=1.6, marker="o", markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.8, label=name, clip_on=False)
            ax.text(xs[-1] + 0.35, vals[-1], f"{vals[-1]:.1f}", va="center", ha="left", color=INK, fontsize=6.5)
        ax.set_xlim(0, 11.6)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 1, 3, 5, 10])
        ax.set_xticklabels(["0", "1", "3", "5", "10%"])
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_title(title, loc="left", color=INK, fontsize=7.5, pad=6)
        ax.set_xlabel("share of statement cells damaged", color=MUTED, fontsize=6.5, labelpad=2)
        style(ax)
    b.legend(loc="lower left", frameon=False, fontsize=6.5, labelcolor=INK2, handlelength=1.6, borderaxespad=0.2)

    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_results.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "fig_results.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", out / "fig_results.pdf")


if __name__ == "__main__":
    main()
