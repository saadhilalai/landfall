"""
hard_mode/evaluate.py — flies a brain through a fixed list of Hard Mode landings and summarises them.

Landings run 20 at a time so the network answers for all of them in one go (much faster than one by one).
Each landing number fixes the terrain, the pad spot(s), the obstacles, the push at the start and the
random weather and engine power, so every brain faces exactly the same landings.

Usage:
    python -m hard_mode.evaluate models_hard/level1_seed42_best.zip --level 1     # validation landings
"""

import argparse
import math
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from hard_mode.levels import LEVELS, VALIDATION_SEEDS
from hard_mode.worlds import make_env


def fly_many(model, level, seeds, parallel=20, deterministic=True):
    """One result dict per landing, in the order of `seeds`."""
    seeds = list(seeds)
    envs = [make_env(level) for _ in range(min(parallel, len(seeds)))]
    results = {}
    queue = iter(enumerate(seeds))
    slots = []                               # per env: [index, seed, obs, score] or None

    def start(env):
        nxt = next(queue, None)
        if nxt is None:
            return None
        i, seed = nxt
        obs, info = env.reset(seed=seed)
        return [i, seed, obs, 0.0]

    slots = [start(env) for env in envs]
    while any(slots):
        active = [k for k, s in enumerate(slots) if s]
        actions, _ = model.predict(np.stack([slots[k][2] for k in active]), deterministic=deterministic)
        for k, action in zip(active, actions):
            slot, env = slots[k], envs[k]
            obs, reward, terminated, truncated, info = env.step(int(action))
            slot[2], slot[3] = obs, slot[3] + reward
            if terminated or truncated:
                game = env.unwrapped
                results[slot[0]] = dict(seed=slot[1], outcome=info["outcome"], score=round(slot[3], 2),
                                        fuel_used=round(info["fuel_used"], 1), steps=game.steps,
                                        pads=len(game.pads), obstacles=len(game.obstacles),
                                        out_of_fuel=game.fuel_left <= 0)
                slots[k] = start(env)
    for env in envs:
        env.close()
    return [results[i] for i in range(len(seeds))]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(results):
    n = len(results)
    landed = [r for r in results if r["outcome"] == "landed"]
    lo, hi = wilson(len(landed), n)
    return dict(landings=n, landed_rate=len(landed) / n, landed_ci95=[lo, hi],
                mean_score=float(np.mean([r["score"] for r in results])),
                fuel_per_safe_landing=float(np.mean([r["fuel_used"] for r in landed])) if landed else None,
                seconds_per_safe_landing=float(np.mean([r["steps"] for r in landed]) / 50) if landed else None,
                ran_out_of_fuel=sum(r["out_of_fuel"] for r in results) / n,
                outcomes=dict(Counter(r["outcome"] for r in results)))


def line(s):
    fuel = f"{s['fuel_per_safe_landing']:.0f}" if s["fuel_per_safe_landing"] is not None else "-"
    return (f"landed {s['landed_rate']:6.1%}  score {s['mean_score']:7.1f}  fuel/landing {fuel:>4}  "
            f"ran dry {s['ran_out_of_fuel']:5.1%}  {s['outcomes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--level", type=int, required=True)
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    print(line(summarize(fly_many(model, LEVELS[args.level], VALIDATION_SEEDS))))
