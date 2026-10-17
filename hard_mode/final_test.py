"""
hard_mode/final_test.py — the ONE-TIME hidden test for Hard Mode. Committed before any Hard Mode training
ran, like final_eval.py in the main project. It refuses to run twice.

What it tests, on the 300 hidden test landings (seeds 40000-40299, never flown before), each with Step 4's
random weather and engine power:
  * every level's best brain, for every seed that trained it, on its own level
  * the same brain on every EARLIER level too (did learning the new level make it forget the old ones?)
  * the starting point: the Final AI (widened, untrained on Hard Mode) on every level, for comparison
  * added overnight, before this test ever ran: each seed's best brain from the obstacle-free mission stage (4a),
    tested on 4a and on the real Level 4; and any "overtime" brain (training past the 12-million-step fair try),
    reported separately and labelled as overtime

Reported per test: safe-landing rate with a Wilson 95% interval, mean official score, fuel per safe
landing, seconds per safe landing, share of landings that ran the tank dry, and how each landing ended.

Usage:
    python -m hard_mode.final_test
"""

import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config
from hard_mode.brain import load_start, model_path
from hard_mode.evaluate import fly_many, line, summarize
from hard_mode.levels import BY_KEY, HIDDEN_TEST_SEEDS, LEVELS
from hard_mode.worlds import make_env

torch.set_num_threads(1)
OUT = Path("runs_hard") / "final_test"
SEEDS = config.TRAIN_SEEDS


def final_ai(seed):
    model = PPO(env=make_vec_env(lambda: make_env(LEVELS[1]), n_envs=1), seed=0, device="cpu", **config.TUNED_PPO_KWARGS)
    load_start(1, seed, model)
    return model


def main():
    if (OUT / "results.json").exists():
        raise SystemExit(f"{OUT}/results.json exists: the hidden test has already run. It only runs once.")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def test(brain_name, seed, model, level):
        s = summarize(fly_many(model, BY_KEY[str(level)], HIDDEN_TEST_SEEDS))
        rows.append(dict(brain=brain_name, seed=seed, level=level, **s))
        print(f"{brain_name:>16s} seed {seed:>3}  on level {level}: {line(s)}", flush=True)

    for seed in SEEDS:
        trained = [n for n in LEVELS if model_path(n, seed).exists()]
        if not trained:
            continue                                                  # (every seed has at least Level 1)
        model = final_ai(seed)
        for level in [*LEVELS, "4a"]:
            test("Final AI (start)", seed, model, level)
        for n in trained:
            model = PPO.load(model_path(n, seed), device="cpu")
            for level in range(1, n + 1):
                test(f"Level {n} brain", seed, model, level)
        for tag, label in (("", "Mission 4a brain"), ("overtime", "Mission 4a brain, overtime"),
                           ("overtime2", "Mission 4a brain, overtime round 2")):
            if model_path("4a", seed, tag=tag).exists():
                model = PPO.load(model_path("4a", seed, tag=tag), device="cpu")
                for level in ("4a", 4):
                    test(label, seed, model, level)
        for tag in ("overtime", "overtime2"):
            if model_path(4, seed, tag=tag).exists():
                model = PPO.load(model_path(4, seed, tag=tag), device="cpu")
                test(f"Level 4 brain, {tag}", seed, model, 4)

    (OUT / "results.json").write_text(json.dumps(dict(test_seeds=[HIDDEN_TEST_SEEDS.start, HIDDEN_TEST_SEEDS.stop - 1],
                                                      rows=rows), indent=1))
    print(f"Wrote {OUT}/results.json")


if __name__ == "__main__":
    main()
