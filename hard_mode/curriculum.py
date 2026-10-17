"""
hard_mode/curriculum.py — the whole curriculum for one seed: Level 1, then 2, 3 and 4, each starting from
the brain that mastered the level before. Stops at the first level that isn't mastered within its budget.
Each level runs as its own process (hard_mode/train.py), so it always uses the latest code.

Usage (low priority, so other work on the Mac keeps its speed):
    nice -n 19 python -m hard_mode.curriculum --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import time

from hard_mode.brain import model_path
from hard_mode.train import RUNS

# From Level 2 on: a quarter of the Final AI's tuned learning rate. The tuned rate was found for learning from
# scratch; fine-tuning an already good brain with it made validation swing between 75% and 88% (PROGRESS.md).
FINE_TUNE_LR = 1.2e-4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--route", default="1,2,3,4", help="levels in order, e.g. 3a,3b,3,4")
    parser.add_argument("--budget", type=int, default=12_000_000)
    parser.add_argument("--check-every", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=FINE_TUNE_LR)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    log_path = RUNS / f"curriculum_seed{args.seed}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    route = args.route.split(",")
    for i, n in enumerate(route):
        start = ["--start", str(model_path(route[i - 1], args.seed))] if i > 0 else []
        with open(log_path, "a") as log:
            subprocess.run([sys.executable, "-m", "hard_mode.train", "--level", n, "--seed", str(args.seed),
                            "--budget", str(args.budget), "--check-every", str(args.check_every),
                            "--learning-rate", str(args.learning_rate), *start],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        info = json.loads((RUNS / f"level{n}" / f"seed{args.seed}" / "run_info.json").read_text())
        if not info["mastered"]:
            with open(log_path, "a") as log:
                log.write(f"[{time.strftime('%H:%M:%S')}] Stopping the curriculum for seed {args.seed} at level {n}.\n")
            break
