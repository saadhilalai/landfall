"""
train.py — trains one type of agent with PPO, once per seed, and logs everything to TensorBoard.

Usage:
    python train.py --agent baseline              # all 3 seeds from config.TRAIN_SEEDS
    python train.py --agent shaped_v2 --seeds 42  # just one seed

What happens for each seed:
  1. 16 copies of the simulator start up, each in its own process.
  2. PPO practises in them for 1M steps. Every ~33k steps it pauses for a
     10-landing check-up, which becomes a point on the learning curve.
  3. At 1M steps: a 100-landing validation test. If the average official
     score is below 200, training continues in 250k-step chunks (testing
     after each) until it passes or reaches 2M steps.
  4. Saves models/<agent>_seed<N>.zip and runs/<agent>/seed<N>/run_info.json
     with every setting, seed, library version and result.

Watch it live:  tensorboard --logdir runs/tensorboard
"""

import argparse
import json
from dataclasses import asdict
import platform
import shutil
import time
import warnings
from collections import Counter
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from envs import AGENT_TYPES, AGENTS, make_test_env, make_vec_env
from evaluate import format_summary, run_episodes, summarize

# Harmless: the check-up world is a single simulator, the practice worlds are 16 in parallel.
warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")


class LandingStatsCallback(BaseCallback):
    """After every practice round, logs to TensorBoard how the landers did:
    landing / crash / never-settled rates, fuel, and (shaped agent) each reward rule."""

    def _on_rollout_start(self):
        self.outcomes, self.fuel, self.shaped, self.shaping = [], [], [], []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "outcome" in info:
                self.outcomes.append(info["outcome"])
                self.fuel.append(info["fuel_used"])
                if "shaping" in info:
                    self.shaped.append(info["shaped_return"])
                    self.shaping.append(info["shaping"])
        return True

    def _on_rollout_end(self):
        if not self.outcomes:
            return
        counts, n = Counter(self.outcomes), len(self.outcomes)
        self.logger.record("rollout/landing_rate", counts["landed"] / n)
        self.logger.record("rollout/crash_rate", counts["crashed"] / n)
        self.logger.record("rollout/never_settled_rate", counts["never_settled"] / n)
        self.logger.record("rollout/fuel_used", float(np.mean(self.fuel)))
        if self.shaping:
            self.logger.record("rollout/shaped_return", float(np.mean(self.shaped)))
            for term in self.shaping[0]:
                self.logger.record(f"shaping/{term}", float(np.mean([s[term] for s in self.shaping])))


def validate(model, conditions=None):
    """The 100-landing 'is it solved?' test (validation seeds, official score)."""
    summary = summarize(run_episodes(model, config.N_VALIDATION_EPISODES, config.VALIDATION_SEED,
                                     conditions=conditions))
    for key in ("mean_score", "success_rate", "crash_rate", "never_settled_rate", "mean_fuel"):
        model.logger.record(f"validation/{key}", summary[key])
    model.logger.dump(model.num_timesteps)
    return summary


def train_one(agent, seed):
    """Train one agent with one seed. Returns its run_info dict."""
    run_name = f"{agent}_seed{seed}"
    recipe = AGENTS[agent]
    validation_conditions = config.TEST_SCENARIOS[recipe["validate_in"]]
    run_dir = Path("runs") / agent / f"seed{seed}"
    tb_dir = Path("runs") / "tensorboard"

    # Fresh folders for this run (re-running replaces the old one).
    shutil.rmtree(run_dir, ignore_errors=True)
    for old in tb_dir.glob(f"{run_name}_*"):
        shutil.rmtree(old)
    Path("models").mkdir(exist_ok=True)

    # Practice worlds (16 in parallel) and a separate world for learning-curve check-ups.
    train_env = make_vec_env(agent, monitor_dir=run_dir / "monitor")
    curve_env = DummyVecEnv([lambda: Monitor(make_test_env(validation_conditions, memory=recipe["memory"]))])
    curve_env.seed(config.CURVE_EVAL_SEED)

    # Shared PPO settings, unless this agent type has its own (e.g. found by tune.py).
    ppo_kwargs = {**config.PPO_KWARGS, **(recipe.get("ppo") or {})}
    model = PPO(
        env=train_env,
        seed=seed,
        device="cpu",
        tensorboard_log=str(tb_dir),
        verbose=0,
        **ppo_kwargs,
    )

    callbacks = CallbackList([
        LandingStatsCallback(),
        EvalCallback(
            curve_env,
            eval_freq=config.CURVE_EVAL_EVERY // config.N_ENVS,
            n_eval_episodes=config.CURVE_EVAL_EPISODES,
            log_path=str(run_dir),          # saves evaluations.npz for the charts
            deterministic=True,
            verbose=0,
        ),
    ])

    def train_for(steps, first):
        model.learn(steps, callback=callbacks, tb_log_name=run_name,
                    reset_num_timesteps=first, progress_bar=False)

    start = time.time()
    history = []

    budget = f"{config.MAX_STEPS:,} steps (full budget)" if recipe["full_budget"] else \
        f"{config.FIRST_CHECK_STEPS:,} steps, more if not solved"
    print(f"\n[{run_name}] training for {budget}...", flush=True)
    train_for(config.FIRST_CHECK_STEPS, first=True)
    summary = validate(model, validation_conditions)
    history.append(dict(steps=model.num_timesteps, minutes=(time.time() - start) / 60, **summary))
    print(f"  {model.num_timesteps:>9,} steps | {format_summary(summary)}", flush=True)

    # Keep going if not solved yet, or if this agent always gets the full budget.
    while model.num_timesteps < config.MAX_STEPS and (
            recipe["full_budget"] or summary["mean_score"] < config.SOLVED_SCORE):
        chunk = min(config.EXTRA_CHUNK_STEPS, config.MAX_STEPS - model.num_timesteps)
        train_for(chunk, first=False)
        summary = validate(model, validation_conditions)
        history.append(dict(steps=model.num_timesteps, minutes=(time.time() - start) / 60, **summary))
        print(f"  {model.num_timesteps:>9,} steps | {format_summary(summary)}", flush=True)

    minutes = (time.time() - start) / 60
    solved = summary["mean_score"] >= config.SOLVED_SCORE
    model.save(f"models/{run_name}.zip")
    train_env.close()
    curve_env.close()

    # Write down exactly how this agent was made.
    run_info = dict(
        agent=agent,
        seed=seed,
        solved=solved,
        total_steps=model.num_timesteps,
        train_minutes=round(minutes, 2),
        final_validation=summary,
        validation_history=history,
        settings=dict(
            env_id=config.ENV_ID,
            n_envs=config.N_ENVS,
            train_seed=seed,
            curve_eval_seed=config.CURVE_EVAL_SEED,
            validation_seed=config.VALIDATION_SEED,
            first_check_steps=config.FIRST_CHECK_STEPS,
            extra_chunk_steps=config.EXTRA_CHUNK_STEPS,
            max_steps=config.MAX_STEPS,
            solved_score=config.SOLVED_SCORE,
            ppo=ppo_kwargs,
            reward_shaping=asdict(recipe["reward"]) if recipe["reward"] else None,
            training_conditions=recipe["conditions"],
            memory=recipe["memory"],
            full_budget=recipe["full_budget"],
            validated_in=recipe["validate_in"],
        ),
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            stable_baselines3=stable_baselines3.__version__,
            gymnasium=gymnasium.__version__,
            machine=f"{platform.system()} {platform.machine()}",
            torch_threads=torch.get_num_threads(),
        ),
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"  done in {minutes:.1f} min -> {'SOLVED' if solved else 'not solved'}, "
          f"saved models/{run_name}.zip", flush=True)
    return run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=AGENT_TYPES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.TRAIN_SEEDS))
    args = parser.parse_args()

    # One CPU thread: fastest for this small network, and keeps results reproducible.
    torch.set_num_threads(1)

    runs = [train_one(args.agent, seed) for seed in args.seeds]

    print(f"\n=== {args.agent}: {len(runs)} seeds ===")
    for r in runs:
        v = r["final_validation"]
        print(f"  seed {r['seed']:>3} | {r['total_steps']:>9,} steps | {format_summary(v)}")
    scores = [r["final_validation"]["mean_score"] for r in runs]
    landed = [r["final_validation"]["success_rate"] for r in runs]
    print(f"  mean across seeds: score {np.mean(scores):.1f} ± {np.std(scores):.1f}, "
          f"landed {np.mean(landed):.1%} ± {np.std(landed):.1%}")


if __name__ == "__main__":
    main()
