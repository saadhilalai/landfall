"""
hard_mode/calibrate.py — sizes the fuel tank from how much fuel the Final AI actually uses.

Flies the (widened) Final AI with an endless tank on 500 calibration landings (seeds 8000-8499, kept
apart from the validation and hidden test landings) in the random training weather, and reports the
fuel it burns. Also works out the least fuel physics allows in the hardest training conditions.
"""

from dataclasses import replace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config
from hard_mode.brain import load_start
from hard_mode.evaluate import fly_many, summarize
from hard_mode.levels import L1
from hard_mode.worlds import make_env

torch.set_num_threads(1)
CALIBRATION_SEEDS = range(8000, 8500)

if __name__ == "__main__":
    endless = replace(L1, fuel_tank=1e9)
    model = PPO(env=make_vec_env(lambda: make_env(endless), n_envs=1), seed=0, device="cpu", **config.TUNED_PPO_KWARGS)
    load_start(1, 42, model)
    res = fly_many(model, endless, CALIBRATION_SEEDS)
    s = summarize(res)
    fuel = np.array([r["fuel_used"] for r in res if r["outcome"] == "landed"])
    print(f"Final AI, endless tank: landed {s['landed_rate']:.1%}; fuel per safe landing: "
          f"mean {fuel.mean():.0f}, median {np.median(fuel):.0f}, 75th {np.percentile(fuel, 75):.0f}, "
          f"90th {np.percentile(fuel, 90):.0f}, max {fuel.max():.0f}")
    for tank in (80, 100, 110, 120, 130, 150):
        print(f"  tank {tank}: {np.mean(fuel > tank):5.1%} of its safe landings needed more")
    # Least fuel to stop a fall from the start height: free fall, then full main engine to touch down at 0.
    drop = 13.33 - (13.33 / 4 + 18 / 30)
    for g, engine in ((10, 1.0), (11, 0.85), (8, 1.15)):
        a = 17.24 * engine
        v = (2 * drop / (1 / g + 1 / (a - g))) ** 0.5
        print(f"  gravity {g}, engine {engine:.0%}: perfect 'suicide burn' needs {v / (a - g) * 50:.0f} fuel")
