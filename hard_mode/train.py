"""
hard_mode/train.py — trains one Hard Mode level for one seed, starting from the previous level's brain.

Every `--check-every` steps it flies the 200 validation landings. It keeps the best brain so far, and stops
as soon as one reaches 90% safe landings ("mastered": move on to the next level). If the budget runs out
first, the level counts as not mastered and the best brain is kept.

Shares the Mac: one process, one CPU thread, the 16 practice landers stepped one after another.

Usage:
    python -m hard_mode.train --level 1 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from hard_mode.brain import HARD_MODELS, load_start, model_path
from hard_mode.evaluate import fly_many, line, summarize
from hard_mode.levels import BY_KEY, PASS_RATE, VALIDATION_SEEDS
from hard_mode.worlds import make_env

torch.set_num_threads(1)
RUNS = Path("runs_hard")


class Checkup(BaseCallback):
    def __init__(self, key, level, seed, every, log, tag=""):
        super().__init__()
        self.key, self.level, self.seed, self.every, self.log, self.tag = key, level, seed, every, log, tag
        self.next_check, self.best, self.history, self.passed = 0, None, [], False

    def check(self):
        s = summarize(fly_many(self.model, self.level, VALIDATION_SEEDS))
        s["steps"] = self.num_timesteps
        s["minutes"] = round((time.time() - self.t0) / 60, 1)
        self.history.append(s)
        better = self.best is None or (s["landed_rate"], s["mean_score"]) > (self.best["landed_rate"], self.best["mean_score"])
        if better:
            self.best = s
            self.model.save(model_path(self.key, self.seed, tag=self.tag))
        self.log(f"  {self.num_timesteps:>10,} steps {s['minutes']:6.1f} min | {line(s)}{'  <- best' if better else ''}")
        self.passed = s["landed_rate"] >= PASS_RATE
        return not self.passed

    def _on_training_start(self):
        self.t0 = time.time()
        self.next_check = self.every
        self.check()                        # the starting brain, before any practice
        if self.passed:
            self.model.stop_now = True

    def _on_step(self):
        if getattr(self.model, "stop_now", False):
            return False
        if self.num_timesteps >= self.next_check:
            self.next_check += self.every
            return self.check()
        return True


def stamped(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_level(level_number, seed, budget, check_every, log=stamped, learning_rate=None, start=None, tag="",
                n_steps=None, batch_size=None, lr_decay=False, ent_coef=None):
    level_number = str(level_number)
    level = BY_KEY[level_number]
    run_name = f"level{level_number}{'_' + tag if tag else ''}"
    run_dir = RUNS / run_name / f"seed{seed}"
    (run_dir / "monitor").mkdir(parents=True, exist_ok=True)
    HARD_MODELS.mkdir(exist_ok=True)

    def builder(i):
        return lambda: make_env(level, practice=True, monitor_file=str(run_dir / "monitor" / f"env{i}"))

    env = DummyVecEnv([builder(i) for i in range(config.N_ENVS)])
    ppo_kwargs = dict(config.TUNED_PPO_KWARGS)
    if learning_rate is not None:          # fine-tuning a good brain: smaller steps (see PROGRESS.md)
        ppo_kwargs["learning_rate"] = learning_rate
    if n_steps:
        ppo_kwargs["n_steps"] = n_steps
    if batch_size:
        ppo_kwargs["batch_size"] = batch_size
    if ent_coef is not None:                # more exploration (see PROGRESS.md, mission stage 4a)
        ppo_kwargs["ent_coef"] = ent_coef
    lr = ppo_kwargs["learning_rate"]
    if lr_decay:                            # shrink the learning rate linearly to 0 over the budget
        # SB3's own class, not a lambda: a saved brain stores a lambda's code with this file's full path on the Mac
        ppo_kwargs["learning_rate"] = LinearSchedule(lr, 0.0, 1.0)
    model = PPO(env=env, seed=seed, device="cpu", verbose=0, tensorboard_log=str(RUNS / "tensorboard"), **ppo_kwargs)
    origin = load_start(level_number, seed, model, start)
    log(f"Level {level_number}{' [' + tag + ']' if tag else ''} ({level.name}), seed {seed}. Start: {origin}. "
        f"Budget {budget:,} steps. Learning rate {lr:.3g}{' decaying to 0' if lr_decay else ''}, "
        f"rounds of {ppo_kwargs['n_steps']} x {config.N_ENVS}, batches of {ppo_kwargs['batch_size']}, "
        f"exploration bonus {ppo_kwargs['ent_coef']:.2g}.")
    cb = Checkup(level_number, level, seed, check_every, log, tag)
    t0 = time.time()
    model.learn(total_timesteps=budget, callback=cb, tb_log_name=f"{run_name}_seed{seed}")
    model.save(model_path(level_number, seed, "last", tag))
    info = dict(level=level_number, tag=tag, name=level.name, seed=seed, start=origin, budget=budget,
                steps=model.num_timesteps, minutes=round((time.time() - t0) / 60, 1), mastered=cb.passed,
                best=cb.best, history=cb.history, lr_decay=lr_decay,
                ppo=dict(ppo_kwargs, learning_rate=lr, policy_kwargs=str(ppo_kwargs["policy_kwargs"])))
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    verdict = "MASTERED" if cb.passed else "not mastered"
    log(f"Level {level_number}{' [' + tag + ']' if tag else ''}, seed {seed}: {verdict} after {model.num_timesteps:,} steps ({info['minutes']} min). "
        f"Best validation: {cb.best['landed_rate']:.1%}")
    env.close()
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, help="1, 2, 3a, 3b, 3 or 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=12_000_000)
    parser.add_argument("--check-every", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=None, help="default: the Final AI's tuned value")
    parser.add_argument("--start", default=None, help="starting brain (default: the previous level's best)")
    parser.add_argument("--tag", default="", help="name for a training variant (kept apart from the main run)")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-decay", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=None, help="exploration bonus (default: the tuned value)")
    args = parser.parse_args()
    train_level(args.level, args.seed, args.budget, args.check_every, learning_rate=args.learning_rate, start=args.start,
                tag=args.tag, n_steps=args.n_steps, batch_size=args.batch_size, lr_decay=args.lr_decay,
                ent_coef=args.ent_coef)
