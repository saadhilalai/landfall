"""
final_eval.py — Step 6: the ONE-TIME final evaluation on the untouched test landings.

Run it exactly once. It refuses to run if results already exist, so the numbers in
the README are the first and only result: no re-running, no changes after seeing them.
This file was committed to git BEFORE it was run, so the plan below is on record.

The plan:
  * Agents: baseline, shaped_v1, shaped_v2, randomized (the story so far), shaped_v2_2m and
    randomized_2m (the equal-budget controls), randomized_memory, randomized_memory_tuned (final).
  * Every trained seed of each agent (42, 142, 242).
  * Test landings: seeds 10000-10099. Never used for training, tuning or any decision.
  * Scenarios: normal, heavy_gravity, weak_engine, strong_wind, harsh, plus the
    "limits" rows strong_wind_extreme and harsh_extreme.
  * Graded on the official score and the "landed safely" rate (came to rest on the pad);
    fuel and landing time are recorded too.
  * Paired comparisons (McNemar exact test on failures, same landings, same training seeds):
      - shaped_v2 vs baseline ............. normal (does the custom reward help?)
      - shaped_v2_2m vs shaped_v2 ......... heavy_gravity, weak_engine (more practice -> more fragile?)
      - randomized_2m vs shaped_v2_2m ..... every scenario (randomization, equal budget)
      - randomized_memory vs randomized_2m  every scenario (memory, equal budget)
      - randomized_memory_tuned vs randomized_memory  every scenario (tuning)
      - randomized_memory_tuned vs baseline .......... every scenario (start vs finish)
  * Video landings are picked from these results by fixed rules (see make_videos.py):
    the seed-42 model of each agent, the FIRST test landing in seed order that matches.

Outputs: runs/final/episodes.csv (every landing), runs/final/results.json (summary + tests),
runs/final/results.md (readable tables).

Usage:
    python final_eval.py
"""

import csv
import json
import platform
import subprocess
import time
from collections import Counter
from math import sqrt
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config
from compare import mcnemar_p

OUT = Path("runs") / "final"
AGENTS = ["baseline", "shaped_v1", "shaped_v2", "randomized", "shaped_v2_2m", "randomized_2m",
          "randomized_memory", "randomized_memory_tuned"]
SCENARIOS = ["normal", "heavy_gravity", "weak_engine", "strong_wind", "harsh",
             "strong_wind_extreme", "harsh_extreme"]
N_LANDINGS = 100
ALL = SCENARIOS
PAIRED_TESTS = [
    ("shaped_v2", "baseline", ["normal"]),
    ("shaped_v2_2m", "shaped_v2", ["heavy_gravity", "weak_engine"]),
    ("randomized_2m", "shaped_v2_2m", ALL),
    ("randomized_memory", "randomized_2m", ALL),
    ("randomized_memory_tuned", "randomized_memory", ALL),
    ("randomized_memory_tuned", "baseline", ALL),
]


def fly(job):
    """One model, one scenario, all test landings. Runs in its own process."""
    import torch
    from stable_baselines3 import PPO
    from evaluate import run_episodes

    agent, seed, scenario = job
    torch.set_num_threads(1)
    model = PPO.load(f"models/{agent}_seed{seed}.zip", device="cpu")
    episodes = run_episodes(model, N_LANDINGS, config.TEST_SEED, conditions=config.TEST_SCENARIOS[scenario])
    return [dict(agent=agent, train_seed=seed, scenario=scenario, landing_seed=e["seed"],
                 score=e["score"], outcome=e["outcome"], fuel_used=e["fuel_used"], steps=e["steps"])
            for e in episodes]


def wilson(k, n, z=1.96):
    """95% confidence interval for a success rate (Wilson score interval)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_cell(rows):
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    landed = [r for r in rows if r["outcome"] == "landed"]
    per_seed_score = [np.mean([r["score"] for r in rows if r["train_seed"] == s]) for s in config.TRAIN_SEEDS]
    per_seed_landed = [np.mean([r["outcome"] == "landed" for r in rows if r["train_seed"] == s])
                       for s in config.TRAIN_SEEDS]
    lo, hi = wilson(outcomes["landed"], n)
    return dict(
        landings=n,
        landed_rate=outcomes["landed"] / n,
        landed_ci95=[lo, hi],
        landed_rate_per_seed=[float(x) for x in per_seed_landed],
        crash_rate=outcomes["crashed"] / n,
        never_settled_rate=outcomes["never_settled"] / n,
        mean_score=float(np.mean([r["score"] for r in rows])),
        score_std_across_seeds=float(np.std(per_seed_score)),
        fuel_per_safe_landing=float(np.mean([r["fuel_used"] for r in landed])) if landed else None,
        seconds_per_safe_landing=float(np.mean([r["steps"] for r in landed]) / 50) if landed else None,
        outcomes=dict(outcomes),
    )


def main():
    if (OUT / "results.json").exists() or (OUT / "episodes.csv").exists():
        raise SystemExit("Final results already exist. The final evaluation runs exactly once.")
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    jobs = [(a, s, sc) for a in AGENTS for s in config.TRAIN_SEEDS for sc in SCENARIOS]
    print(f"Final evaluation: {len(AGENTS)} agents x {len(config.TRAIN_SEEDS)} seeds x {len(SCENARIOS)} scenarios "
          f"x {N_LANDINGS} test landings = {len(jobs) * N_LANDINGS:,} landings", flush=True)
    start = time.time()
    with Pool() as pool:
        rows = [r for batch in pool.map(fly, jobs) for r in batch]
    print(f"Done in {(time.time() - start) / 60:.1f} min", flush=True)

    with open(OUT / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cells = {sc: {a: summarize_cell([r for r in rows if r["agent"] == a and r["scenario"] == sc])
                  for a in AGENTS} for sc in SCENARIOS}

    index = {(r["agent"], r["train_seed"], r["scenario"], r["landing_seed"]): r for r in rows}
    tests = []
    for a, b, scenarios in PAIRED_TESTS:
        for sc in scenarios:
            only_a = only_b = both = 0
            for s in config.TRAIN_SEEDS:
                for i in range(N_LANDINGS):
                    land = config.TEST_SEED + i
                    fail_a = index[(a, s, sc, land)]["outcome"] != "landed"
                    fail_b = index[(b, s, sc, land)]["outcome"] != "landed"
                    only_a += fail_a and not fail_b
                    only_b += fail_b and not fail_a
                    both += fail_a and fail_b
            tests.append(dict(agent=a, reference=b, scenario=sc, failed_only_by_agent=only_a,
                              failed_only_by_reference=only_b, failed_by_both=both,
                              landed_agent=cells[sc][a]["landed_rate"], landed_reference=cells[sc][b]["landed_rate"],
                              p_value=mcnemar_p(only_a, only_b)))

    results = dict(
        run=dict(git_commit=commit, date=time.strftime("%Y-%m-%d %H:%M"), test_seeds=[config.TEST_SEED,
                 config.TEST_SEED + N_LANDINGS - 1], train_seeds=list(config.TRAIN_SEEDS),
                 python=platform.python_version()),
        agents=AGENTS, scenarios={sc: config.TEST_SCENARIOS[sc] for sc in SCENARIOS},
        cells=cells, paired_tests=tests,
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Final results (test landings 10000-10099, 3 seeds per agent = 300 landings per cell)\n",
             "## Landed safely\n", "| Agent | " + " | ".join(SCENARIOS) + " |",
             "|---|" + "---|" * len(SCENARIOS)]
    for a in AGENTS:
        lines.append(f"| {a} | " + " | ".join(f"{cells[sc][a]['landed_rate']:.1%}" for sc in SCENARIOS) + " |")
    lines += ["", "## Official score (mean)\n", "| Agent | " + " | ".join(SCENARIOS) + " |",
              "|---|" + "---|" * len(SCENARIOS)]
    for a in AGENTS:
        lines.append(f"| {a} | " + " | ".join(f"{cells[sc][a]['mean_score']:.1f}" for sc in SCENARIOS) + " |")
    lines += ["", "## Paired tests (failures on the same landings)\n",
              "| Agent | vs | Scenario | Landed (agent / ref) | Failed only by agent / ref | p |",
              "|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['agent']} | {t['reference']} | {t['scenario']} | {t['landed_agent']:.1%} / "
                     f"{t['landed_reference']:.1%} | {t['failed_only_by_agent']} / {t['failed_only_by_reference']} | "
                     f"{t['p_value']:.4f} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUT}/episodes.csv, results.json, results.md")


if __name__ == "__main__":
    main()
