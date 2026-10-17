"""
hard_mode/report.py — the Hard Mode curriculum at a glance: one chart of validation landing rate vs practice
steps for every level and seed (charts/hard_mode_curriculum.png), and a markdown summary of the training runs
and, once it has run, the hidden test (runs_hard/report.md).

Usage:
    python -m hard_mode.report
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hard_mode.levels import BY_KEY, PASS_RATE  # noqa: E402

RUNS = Path("runs_hard")
SEED_COLORS = {42: "#2a78d6", 142: "#e8693d", 242: "#1aa37a"}   # same hues as the main project's charts


def panel(info):
    """Chart panel for a run: its level, plus the tag for runs outside the rules (e.g. '4a overtime')."""
    return f"{info['level']} {info['tag']}" if info.get("tag") else str(info["level"])


def runs():
    order = {k: i for i, k in enumerate(BY_KEY)}
    out = [json.loads(f.read_text()) for f in RUNS.glob("level*/seed*/run_info.json")]
    return sorted(out, key=lambda i: (order[str(i["level"])], i.get("tag", ""), i["seed"]))


def chart(infos):
    keys = list(dict.fromkeys(panel(i) for i in infos))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("#fbfbf9")
    for ax, n in zip(axes, keys):
        level = BY_KEY[n.split()[0]]
        ax.set_facecolor("#fbfbf9")
        ax.axhline(PASS_RATE * 100, color="#333", lw=1)
        ax.text(0, PASS_RATE * 100 + 1.5, "mastered (90%)", fontsize=8, color="#555")
        for info in [i for i in infos if panel(i) == n]:
            h = info["history"]
            xs = [r["steps"] / 1e6 for r in h]
            ys = [r["landed_rate"] * 100 for r in h]
            c = SEED_COLORS.get(info["seed"], "#777")
            ax.plot(xs, ys, color=c, lw=2, label=f"seed {info['seed']}")
            ax.scatter(xs[-1:], ys[-1:], color=c, s=28, zorder=3)
        extra = n.split()[1:]
        label = {"overtime": "OVERTIME, round 1 (outside the rules)", "overtime2": "OVERTIME, round 2 (outside the rules)"}
        ax.set_title(f"Level {n.split()[0]}\n{level.name}" + (f"\n{label.get(extra[0], extra[0])}" if extra else ""),
                     fontsize=9.5, loc="left")
        ax.set_xlabel("practice steps (millions)")
        ax.set_ylim(0, 100)
        ax.grid(color="#ddd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("validation landings that landed safely (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Hard Mode curriculum: each level starts from the brain that mastered the one before", x=0.01,
                 ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path("charts").mkdir(exist_ok=True)
    fig.savefig("charts/hard_mode_curriculum.png", dpi=150)
    print("Saved charts/hard_mode_curriculum.png")


def origin(text):
    """'brain models_hard/level3a_seed42_best.zip' -> 'Level 3a brain'; 'Final AI (...)' -> 'Final AI'."""
    m = re.search(r"level(\w+?)_seed", text)
    return f"Level {m.group(1)} brain" if m else text.split(" (")[0]


def markdown(infos):
    lines = ["## Training runs", "",
             "| Level | Seed | Started from | Validation at start | Best validation | Mastered? | Practice steps | Minutes |",
             "|---|---|---|---|---|---|---|---|"]
    for i in infos:
        start = i["history"][0]["landed_rate"]
        lines.append(f"| {panel(i)} {i['name']} | {i['seed']} | {origin(i['start'])} | {start:.1%} | "
                     f"{i['best']['landed_rate']:.1%} | {'yes' if i['mastered'] else 'no'} | {i['steps']:,} | {i['minutes']} |")
    test = RUNS / "final_test" / "results.json"
    if test.exists():
        rows = json.loads(test.read_text())["rows"]
        lines += ["", "## Hidden test (300 landings per row, seeds 40000-40299, run once)", "",
                  "| Brain | Seed | Tested on level | Landed safely (95% CI) | Score | Fuel per landing | Ran dry | Outcomes |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fuel = f"{r['fuel_per_safe_landing']:.0f}" if r["fuel_per_safe_landing"] is not None else "-"
            lo, hi = r["landed_ci95"]
            lines.append(f"| {r['brain']} | {r['seed']} | {r['level']} | {r['landed_rate']:.1%} ({lo:.0%}–{hi:.0%}) | "
                         f"{r['mean_score']:.1f} | {fuel} | {r['ran_out_of_fuel']:.1%} | {r['outcomes']} |")
    (RUNS / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RUNS}/report.md")


if __name__ == "__main__":
    infos = runs()
    if not infos:   # on main the training logs aren't included, so keep the committed chart and table
        raise SystemExit("No training logs in runs_hard/ (they're on the hard-mode branch); nothing redrawn.")
    chart(infos)
    markdown(infos)
