"""
evaluate.py — puts a trained agent through a fixed list of test landings.

Every agent is graded on the OFFICIAL game score (never a shaped reward),
using the same list of seeds, so they all face the exact same landings.

Usage:
    python evaluate.py models/baseline_seed42.zip                  # 100 test landings
    python evaluate.py models/baseline_seed42.zip --episodes 20
"""

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

import config
from envs import make_test_env


def run_episodes(model, n_episodes, seed, conditions=None, deterministic=True):
    """Fly `n_episodes` landings. Landing i uses seed `seed + i` (same terrain + start every time)."""
    env = make_test_env(conditions)
    episodes = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        score, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            steps += 1
            done = terminated or truncated
        episodes.append(dict(seed=seed + i, score=score, outcome=info["outcome"],
                             fuel_used=info["fuel_used"], steps=steps))
    env.close()
    return episodes


def summarize(episodes):
    """Boil a list of landings down to the headline numbers."""
    scores = np.array([e["score"] for e in episodes])
    outcomes = Counter(e["outcome"] for e in episodes)
    n = len(episodes)
    return dict(
        episodes=n,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        success_rate=outcomes["landed"] / n,     # came to rest on the pad, between the flags
        crash_rate=outcomes["crashed"] / n,
        never_settled_rate=outcomes["never_settled"] / n,
        mean_fuel=float(np.mean([e["fuel_used"] for e in episodes])),
        outcomes=dict(outcomes),
    )


def format_summary(s):
    return (
        f"score {s['mean_score']:6.1f} ± {s['std_score']:5.1f} | "
        f"landed {s['success_rate']:6.1%} | crashed {s['crash_rate']:5.1%} | "
        f"never settled {s['never_settled_rate']:5.1%} | fuel {s['mean_fuel']:5.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="path to a saved model .zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=config.TEST_SEED)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    summary = summarize(run_episodes(model, args.episodes, args.seed))
    print(format_summary(summary))
    print("outcomes:", summary["outcomes"])


if __name__ == "__main__":
    main()
