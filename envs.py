"""
envs.py — builds the simulated worlds the agents practise and get tested in.

Each practice world is a stack of layers ("wrappers") around LunarLander-v3:

    LunarLander-v3          the physics simulation
      -> PhysicsConditions  (randomized agent only) re-rolls gravity/wind/engine every episode
      -> LandingStats       notes how each episode ended (landed / crashed / ...) and fuel used
      -> Monitor            records the OFFICIAL game score of every episode
      -> RewardShaping      (shaped + randomized agents) adds our own reward rules on top

Monitor sits *below* any reward changes, so the score we log is always the
official one. That keeps the comparison between agents fair.
"""

from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config
from domain_randomization import PhysicsConditions
from landing_stats import LandingStats
from reward_shaping import VERSIONS, LandingRewardShaping

# What makes each agent different. Everything else (PPO settings, seeds,
# training budget, tests) is shared by all of them.
AGENTS = {
    "baseline":   dict(reward=None,           conditions=None),
    "shaped_v1":  dict(reward=VERSIONS["v1"], conditions=None),
    "shaped_v2":  dict(reward=VERSIONS["v2"], conditions=None),
    "randomized": dict(reward=VERSIONS["v2"], conditions=config.TRAIN_CONDITIONS),
}
AGENT_TYPES = tuple(AGENTS)


def make_env(agent_type="baseline", monitor_file=None, render_mode=None):
    """One practice world for the given agent type."""
    if agent_type not in AGENTS:
        raise ValueError(f"Unknown agent type {agent_type!r}; choose from {AGENT_TYPES}")
    recipe = AGENTS[agent_type]
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    if recipe["conditions"] is not None:
        env = PhysicsConditions(env, recipe["conditions"])
    env = LandingStats(env)
    env = Monitor(env, filename=monitor_file, info_keywords=("outcome", "fuel_used"))
    if recipe["reward"] is not None:
        env = LandingRewardShaping(env, recipe["reward"])
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


def make_test_env(conditions=None, render_mode=None):
    """A test world with the OFFICIAL reward. `conditions` sets gravity/wind/etc.
    (fixed numbers, or (low, high) ranges re-drawn every episode)."""
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    if conditions:
        env = PhysicsConditions(env, conditions)
    return LandingStats(env)
