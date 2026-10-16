"""
compare.py — compares agent types on the exact same landings, for every training seed.

Usage:
    python compare.py --agents baseline shaped_v1 shaped_v2
    python compare.py --agents baseline shaped_v2 randomized --scenarios normal harsh --episodes 200
    python compare.py --agents baseline shaped_v2 randomized --reference baseline shaped_v2 --scenarios harsh

Every model is graded on the OFFICIAL game score and the "landed safely" rate
(came to rest on the pad). Uses the analysis landings (config.ANALYSIS_SEED),
so the final test landings stay untouched until Step 6.

For each agent vs each reference agent (same training seed, same landings) it reports:
  * a paired significance test on failures (McNemar's exact test): landings that
    only one of the two failed are the evidence; if those split 50/50 there's no
    real difference, and the p-value says how unlikely the actual split is by luck
  * on the landings where the reference never settled, what the other agent did
"""

import argparse
import json
from collections import Counter
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config
from evaluate import run_episodes, summarize


def evaluate_model(job):
    """Runs in its own process, so all models are tested in parallel."""
    import torch
    from stable_baselines3 import PPO

    agent, seed, scenario, n_episodes = job
    torch.set_num_threads(1)
    model = PPO.load(f"models/{agent}_seed{seed}.zip", device="cpu")
    conditions = config.TEST_SCENARIOS[scenario]
    return job, run_episodes(model, n_episodes, config.ANALYSIS_SEED, conditions=conditions)


def mcnemar_p(only_a, only_b):
    """Exact two-sided McNemar test: chance of a split at least this lopsided if both were equal."""
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def across_seeds(per_seed, key):
    values = [s[key] for s in per_seed.values()]
    return dict(mean=float(np.mean(values)), std=float(np.std(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=["baseline", "shaped_v1", "shaped_v2"])
    parser.add_argument("--reference", nargs="+", default=["baseline"],
                        help="agent(s) the others are compared against")
    parser.add_argument("--scenarios", nargs="+", default=["normal"], choices=list(config.TEST_SCENARIOS))
    parser.add_argument("--episodes", type=int, default=config.N_ANALYSIS_EPISODES)
    parser.add_argument("--out", default="runs/compare.json")
    args = parser.parse_args()
    seeds = config.TRAIN_SEEDS

    jobs = [(a, s, sc, args.episodes) for sc in args.scenarios for a in args.agents for s in seeds]
    with Pool() as pool:
        results = {(a, s, sc): eps for (a, s, sc, _), eps in pool.map(evaluate_model, jobs)}

    report = {}
    for scenario in args.scenarios:
        print(f"\n=== {scenario}: {config.TEST_SCENARIOS[scenario] or 'default game'} | "
              f"{args.episodes:,} landings x {len(seeds)} seeds per agent ===")
        print(f"  {'agent':11s} {'official score':>16s} {'landed':>15s} {'crashed':>8s} "
              f"{'never settled':>14s} {'fuel':>6s} {'landing time':>13s}")
        report[scenario] = {}
        for agent in args.agents:
            per_seed = {s: summarize(results[(agent, s, scenario)]) for s in seeds}
            stats = {k: across_seeds(per_seed, k) for k in
                     ("mean_score", "success_rate", "crash_rate", "never_settled_rate", "mean_fuel")}
            landed = [e for s in seeds for e in results[(agent, s, scenario)] if e["outcome"] == "landed"]
            stats["landing_seconds"] = float(np.mean([e["steps"] for e in landed]) / 50) if landed else None
            report[scenario][agent] = dict(per_seed=per_seed, across_seeds=stats)
            secs = f"{stats['landing_seconds']:.1f} s" if landed else "-"
            print(f"  {agent:11s} {stats['mean_score']['mean']:7.1f} ± {stats['mean_score']['std']:5.1f}   "
                  f"{stats['success_rate']['mean']:6.1%} ± {stats['success_rate']['std']:4.1%} "
                  f"{stats['crash_rate']['mean']:8.1%} {stats['never_settled_rate']['mean']:14.1%} "
                  f"{stats['mean_fuel']['mean']:6.1f} {secs:>13s}")

        pairs = [(ref, other) for ref in args.reference if ref in args.agents
                 for other in args.agents[args.agents.index(ref) + 1:]]
        for ref, other in pairs:
            only_ref = only_other = both = 0
            ref_never_settled = Counter()
            for s in seeds:
                for r, o in zip(results[(ref, s, scenario)], results[(other, s, scenario)]):
                    r_fail, o_fail = r["outcome"] != "landed", o["outcome"] != "landed"
                    only_ref += r_fail and not o_fail
                    only_other += o_fail and not r_fail
                    both += r_fail and o_fail
                    if r["outcome"] == "never_settled":
                        ref_never_settled[o["outcome"]] += 1
            p = mcnemar_p(only_ref, only_other)
            print(f"  {other} vs {ref}: failed only by {ref} {only_ref}, "
                  f"only by {other} {only_other}, both {both} -> p = {p:.4f}")
            if ref_never_settled:
                print(f"    where {ref} never settled ({sum(ref_never_settled.values())}), "
                      f"{other} did: {dict(ref_never_settled)}")
            report[scenario][f"{other}_vs_{ref}"] = dict(
                failed_only_by_reference=only_ref, failed_only_by_other=only_other, failed_by_both=both,
                mcnemar_p=p, reference_never_settled=dict(ref_never_settled))

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
