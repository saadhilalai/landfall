"""
make_charts.py — draws the README charts from saved results (never re-runs anything).

Inputs:  runs/final/results.json (final test), runs/<agent>/seed*/evaluations.npz (learning curves)
Outputs: charts/1_learning_curves.png, 2_landing_success.png,
         3_more_practice_more_fragile.png, 4_memory_in_wind.png

Style: one fixed color per agent across every chart, thin marks, light gridlines,
text in ink colors (never the series color), values labelled directly.
The palette was checked with a colorblind-safety validator for every pair shown together.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import config  # noqa: E402

OUT = Path("charts")
RESULTS = json.load(open("runs/final/results.json"))

# ---- Style tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"

AGENT_STYLE = {   # name shown on charts, fixed color (follows the agent everywhere)
    "baseline":                ("Baseline (official reward)", "#2a78d6"),
    "shaped_v1":               ("Custom reward v1", "#eb6834"),
    "shaped_v2":               ("Custom reward v2, 1M steps", "#1baf7a"),
    "randomized":              ("Randomized, 1M steps", "#e87ba4"),
    "shaped_v2_2m":            ("Custom reward v2, 2M steps", "#eda100"),
    "randomized_2m":           ("Randomized, 2M steps", "#008300"),
    "randomized_memory":       ("Randomized + memory", "#6250d6"),
    "randomized_memory_tuned": ("Final: randomized + memory + tuned", "#e34948"),
}
SCENARIO_LABEL = {
    "normal": "Normal", "heavy_gravity": "Heavy\ngravity", "weak_engine": "Weak\nengine",
    "strong_wind": "Strong\nwind", "harsh": "Harsh\n(all at once)",
    "strong_wind_extreme": "Strong wind\n(extreme)", "harsh_extreme": "Harsh\n(extreme)",
}
SUBTITLE_TEST = "Unseen test landings 10000-10099, 3 trained seeds per agent = 300 landings per {}"


def pct(v):
    """Whole percent, except just below 100 (99.7% must not read as a perfect 100%)."""
    return f"{v:.1f}%" if 99.5 <= v < 100 else f"{v:.0f}%"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 10,
})


def title(fig, main, sub):
    fig.text(0.01, 0.975, main, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, sub, fontsize=10.5, color=INK_2, va="top")


def cell(agent, scenario):
    return RESULTS["cells"][scenario][agent]


def p_value(agent, reference, scenario):
    for t in RESULTS["paired_tests"]:
        if (t["agent"], t["reference"], t["scenario"]) == (agent, reference, scenario):
            return t["p_value"]
    return None


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---- 1. Learning curves (small multiples) --------------------------------------
def learning_curves():
    order = list(AGENT_STYLE)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True)
    for ax, agent in zip(axes.flat, order):
        label, color = AGENT_STYLE[agent]
        runs = []
        for seed in config.TRAIN_SEEDS:
            d = np.load(f"runs/{agent}/seed{seed}/evaluations.npz")
            runs.append((d["timesteps"], d["results"].mean(axis=1)))
        n = min(len(t) for t, _ in runs)
        steps = runs[0][0][:n] / 1e6
        scores = np.array([r[:n] for _, r in runs])
        smooth = lambda y: np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        mean, lo, hi = smooth(scores.mean(0)), smooth(scores.min(0)), smooth(scores.max(0))
        ax.axhline(200, color=INK_2, linewidth=0.9)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, solid_capstyle="round")
        ax.plot(steps[-1], mean[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{mean[-1]:.0f}", (steps[-1], mean[-1]), xytext=(6, -14), textcoords="offset points",
                    fontsize=10, color=INK)
        ax.set_title(label, fontsize=10.5, loc="left", color=INK, pad=6)
        ax.set_xlim(0, 2.1)
        ax.set_ylim(-650, 330)
    axes[0, 0].text(0.05, 207, "solved (200)", fontsize=9, color=INK_2)
    for ax in axes[1]:
        ax.set_xlabel("training steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("official score")
    title(fig, "Learning curves: every agent teaches itself to land",
          "Average official score in the normal game over 10 check-up landings, every ~33k steps. "
          "Line = mean of 3 seeds (smoothed over 3 check-ups), band = best to worst seed.\n"
          "The y-axis stops at -650: a few very early, untrained check-ups score far lower (down to about -4,300).")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "1_learning_curves.png", dpi=170)
    plt.close(fig)


# ---- 2. Landing success across all scenarios (heatmap) ----------------------------
def landing_success():
    agents, scenarios = RESULTS["agents"], list(RESULTS["scenarios"])
    grid = np.array([[cell(a, sc)["landed_rate"] * 100 for sc in scenarios] for a in agents])
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.imshow(grid, cmap=blues, vmin=0, vmax=100, aspect="auto")
    for i in range(len(agents)):
        for j in range(len(scenarios)):
            v = grid[i, j]
            ax.text(j, i, pct(v), ha="center", va="center", fontsize=11.5,
                    color="white" if v >= 55 else INK, fontweight="bold" if i == len(agents) - 1 else "normal")
    ax.set_xticks(range(len(scenarios)), [SCENARIO_LABEL[s] for s in scenarios], fontsize=10.5, color=INK)
    ax.set_yticks(range(len(agents)), [AGENT_STYLE[a][0] for a in agents], fontsize=10.5, color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(scenarios)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(agents)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.axvline(4.5, color=INK_2, linewidth=1.2)
    ax.text(5.5, len(agents) - 0.35, "limits: beyond what any agent reliably lands", ha="center", va="top",
            fontsize=9.5, color=INK_2)
    title(fig, "Landing safely, by test condition",
          "% of landings that came to rest on the pad. " + SUBTITLE_TEST.format("cell") + ".")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "2_landing_success.png", dpi=170)
    plt.close(fig)


# ---- Grouped bars with 95% intervals (charts 3 and 4) ----------------------------
def grouped_bars(path, agents, scenarios, main, sub, notes):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    width, gap = 0.17, 0.03
    x = np.arange(len(scenarios))
    for k, agent in enumerate(agents):
        label, color = AGENT_STYLE[agent]
        offset = (k - (len(agents) - 1) / 2) * (width + gap)
        rates = [cell(agent, sc)["landed_rate"] * 100 for sc in scenarios]
        cis = [cell(agent, sc)["landed_ci95"] for sc in scenarios]
        err = [[max(0.0, r - lo * 100) for r, (lo, _) in zip(rates, cis)],
               [max(0.0, hi * 100 - r) for r, (_, hi) in zip(rates, cis)]]
        ax.bar(x + offset, rates, width, color=color, label=label, zorder=2)
        ax.errorbar(x + offset, rates, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=0, zorder=3)
        for xi, r, (_, hi) in zip(x + offset, rates, cis):
            ax.text(xi, hi * 100 + 1.5, pct(r), ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_xticks(x, [SCENARIO_LABEL[s].replace("\n", " ") for s in scenarios], fontsize=11, color=INK)
    for xi, note in zip(x, notes):
        ax.text(xi, -15, note, ha="center", va="top", fontsize=10, color=INK_2)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_ylabel("landed safely")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(agents), handlelength=1.2)
    title(fig, main, sub)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(OUT / path, dpi=170)
    plt.close(fig)


def more_practice_more_fragile():
    scenarios = ["normal", "heavy_gravity", "weak_engine"]
    notes = ["both near 100%"] + [fmt_p(p_value("shaped_v2_2m", "shaped_v2", sc)) for sc in scenarios[1:]]
    grouped_bars("3_more_practice_more_fragile.png", ["shaped_v2", "shaped_v2_2m"], scenarios,
                 "More practice in one world made the agent more fragile",
                 "Same agent and reward, trained only in the normal game: 1M vs 2M steps. "
                 "Lines = 95% intervals. p = paired test on the same landings.\n"
                 + SUBTITLE_TEST.format("bar") + ".", notes)


def memory_in_wind():
    scenarios = ["strong_wind", "strong_wind_extreme", "harsh", "harsh_extreme"]
    notes = [fmt_p(p_value("randomized_memory", "randomized_2m", sc)) for sc in scenarios]
    grouped_bars("4_memory_in_wind.png", ["randomized_2m", "randomized_memory"], scenarios,
                 "Memory vs no memory in wind (same 2M-step budget)",
                 "Memory = the agent sees its last 4 moments, so it can feel the wind. It lands more often in all four,\n"
                 "but with 300 landings per bar the strong-wind gaps could still be chance (p ≈ 0.2). "
                 "Lines = 95% intervals. p = paired test on the same landings.", notes)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    learning_curves()
    landing_success()
    more_practice_more_fragile()
    memory_in_wind()
    print("Saved:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
