"""
hard_mode/brain.py — carries a trained brain into the next level instead of starting over.

Level 1 starts from the Final AI (models/randomized_memory_tuned_seed<N>.zip). The Final AI saw 8 numbers
per moment; Hard Mode shows 19 (env.py). So its first layer gets new input connections for the 11 new
numbers, all starting at ZERO: on day one it ignores the fuel gauge, sensors and phase, and flies exactly
like the Final AI. Training then learns what the new numbers mean. Every other weight is copied as is.

Levels 2-4 see the same 19 numbers, so they start from the previous level's brain unchanged.
"""

from pathlib import Path

import torch
from stable_baselines3 import PPO

from hard_mode.env import OBS_SIZE
from hard_mode.worlds import MEMORY

MAIN_MODELS = Path("models")
HARD_MODELS = Path("models_hard")


def model_path(level, seed, kind="best", tag=""):
    """models_hard/level<key>[_<tag>]_seed<N>_<kind>.zip; a tag marks a training variant."""
    return HARD_MODELS / f"level{level}{'_' + tag if tag else ''}_seed{seed}_{kind}.zip"


def widen(old_state, new_state):
    """Copy every weight; first layers (64 x 4*8) are widened to (64 x 4*19) with zeros for the new inputs."""
    out = {}
    for key, new in new_state.items():
        old = old_state[key]
        if old.shape == new.shape:
            out[key] = old.clone()
            continue
        assert key.endswith(".0.weight"), key
        old_n, new_n = old.shape[1] // MEMORY, new.shape[1] // MEMORY
        w = torch.zeros_like(new)
        for frame in range(MEMORY):
            w[:, frame * new_n: frame * new_n + old_n] = old[:, frame * old_n:(frame + 1) * old_n]
        out[key] = w
    return out


def load_start(level, seed, model, start=None):
    """Put the starting brain for `level` (a key like "1", "3a") into `model`, a fresh PPO with the Hard Mode
    input size. `start` overrides the default (the previous level's best brain)."""
    from hard_mode.levels import PREVIOUS
    level = str(level)
    if start is None and level == "1":
        source = PPO.load(MAIN_MODELS / f"randomized_memory_tuned_seed{seed}.zip", device="cpu")
        state = widen(source.policy.state_dict(), model.policy.state_dict())
        origin = f"Final AI (randomized_memory_tuned_seed{seed}), widened from 8 to {OBS_SIZE} inputs per moment"
    else:
        path = Path(start) if start else model_path(PREVIOUS[level], seed)
        state = PPO.load(path, device="cpu").policy.state_dict()
        origin = f"brain {path}"
    model.policy.load_state_dict(state)
    return origin
