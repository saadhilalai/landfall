"""
envs.py — builds the simulated worlds the agents practise and get tested in.

Each practice world is a stack of layers ("wrappers") around LunarLander-v3:

    LunarLander-v3          the physics simulation
      -> PhysicsConditions  (some agents) sets gravity/wind/engine, or re-rolls them every episode
      -> LandingStats       notes how each episode ended (landed / crashed / ...) and fuel used
      -> Monitor            records the OFFICIAL game score of every episode
      -> RewardShaping      (shaped agents) adds our own reward rules on top
      -> FrameStack         (memory agents) shows the agent its last few moments, not just now

Monitor sits *below* any reward changes, so the score we log is always the
official one. That keeps the comparison between agents fair.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config
from domain_randomization import PhysicsConditions
from landing_stats import LandingStats
from reward_shaping import V1, V2, LandingRewardShaping


def recipe(reward=None, conditions=None, memory=1, full_budget=False, validate_in="normal"):
    """What makes an agent type different. Everything else (PPO settings, seeds, tests) is shared.

    reward       custom reward settings (None = the official reward only)
    conditions   physics, fixed or as ranges (None = the normal game)
    memory       how many recent moments the agent sees (1 = just the current one)
    full_budget  always train the full config.MAX_STEPS, instead of stopping as soon
                 as the validation score passes 200
    validate_in  the test scenario used for validation and learning-curve check-ups
    """
    return dict(reward=reward, conditions=conditions, memory=memory,
                full_budget=full_budget, validate_in=validate_in)


AGENTS = {
    # Steps 2-4: train until the validation score passes 200 (at most 2M steps).
    "baseline":   recipe(),
    "shaped_v1":  recipe(reward=V1),
    "shaped_v2":  recipe(reward=V2),
    "randomized": recipe(reward=V2, conditions=config.TRAIN_CONDITIONS),

    # Step 4 fix: the same full 2M budget for all three, so the only differences
    # are randomization (2nd vs 1st) and memory (3rd vs 2nd).
    "shaped_v2_2m":      recipe(reward=V2, full_budget=True),
    "randomized_2m":     recipe(reward=V2, conditions=config.TRAIN_CONDITIONS, full_budget=True),
    "randomized_memory": recipe(reward=V2, conditions=config.TRAIN_CONDITIONS,
                                memory=config.MEMORY, full_budget=True),

}

# Landability check: a specialist per test scenario, trained directly IN it, to find
# out whether it can be landed at all. Not part of the comparison.
for _name, _conditions in config.TEST_SCENARIOS.items():
    if _name != "normal":
        AGENTS[f"specialist_{_name}"] = recipe(reward=V2, conditions=_conditions, memory=config.MEMORY,
                                               full_budget=True, validate_in=_name)
AGENT_TYPES = tuple(AGENTS)


def make_env(agent_type="baseline", monitor_file=None, render_mode=None):
    """One practice world for the given agent type."""
    if agent_type not in AGENTS:
        raise ValueError(f"Unknown agent type {agent_type!r}; choose from {AGENT_TYPES}")
    r = AGENTS[agent_type]
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    if r["conditions"] is not None:
        env = PhysicsConditions(env, r["conditions"])
    env = LandingStats(env)
    env = Monitor(env, filename=monitor_file, info_keywords=("outcome", "fuel_used"))
    if r["reward"] is not None:
        env = LandingRewardShaping(env, r["reward"])
    if r["memory"] > 1:
        env = FrameStackObservation(env, r["memory"])
    return env


def make_vec_env(agent_type, n_envs=config.N_ENVS, monitor_dir=None, subproc=True):
    """Many practice worlds running side by side (one per CPU process if subproc=True)."""
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def builder(rank):
        def _init():
            monitor_file = str(Path(monitor_dir) / f"env{rank}") if monitor_dir else None
            return make_env(agent_type, monitor_file=monitor_file)
        return _init

    fns = [builder(i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)


def make_test_env(conditions=None, render_mode=None, memory=1):
    """A test world with the OFFICIAL reward. `conditions` sets gravity/wind/etc.
    (fixed numbers, or (low, high) ranges re-drawn every episode). `memory` must
    match the agent being tested."""
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    if conditions:
        env = PhysicsConditions(env, conditions)
    env = LandingStats(env)
    if memory > 1:
        env = FrameStackObservation(env, memory)
    return env


def model_memory(model):
    """How many moments a trained model expects to see: (4, 8) inputs -> 4, (8,) -> 1."""
    shape = model.observation_space.shape
    return shape[0] if len(shape) == 2 else 1
