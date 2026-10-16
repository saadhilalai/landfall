"""
domain_randomization.py — changes the physics of the world: gravity, wind, turbulence, engine power.

The idea (used by robotics labs to move robots from simulation to the real world):
an agent that only ever practises in one world learns tricks that only work there.
If every practice episode re-rolls the physics within a range, the agent can't
rely on any one set of conditions. It has to learn a strategy that works in all of them.

Each condition is either a fixed number (for tests) or a (low, high) range that is
re-drawn at the start of every episode (for training). The agent is never told
what the conditions are; it only sees its usual 8 numbers.
"""

import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module
import numpy as np

# The game's built-in engine strengths (main engine, side thrusters).
DEFAULT_MAIN_ENGINE = lunar_lander_module.MAIN_ENGINE_POWER   # 13.0
DEFAULT_SIDE_ENGINE = lunar_lander_module.SIDE_ENGINE_POWER   # 0.6

# The ordinary game: Earth-like gravity, no wind, normal engines.
NORMAL = dict(gravity=-10.0, wind_power=0.0, turbulence_power=0.0, engine_power=1.0)


class PhysicsConditions(gym.Wrapper):
    """Sets gravity, wind, turbulence and engine power. Must wrap the game directly."""

    def __init__(self, env, conditions):
        super().__init__(env)
        unknown = set(conditions) - set(NORMAL)
        if unknown:
            raise ValueError(f"Unknown conditions: {unknown}")
        self.conditions = {**NORMAL, **conditions}
        self.rng = np.random.default_rng()
        self.current = dict(NORMAL)

    def _draw(self, value):
        return float(self.rng.uniform(*value)) if isinstance(value, tuple) else float(value)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Its own random stream, separate from the one that builds the terrain.
            self.rng = np.random.default_rng([seed, 1])
        self.current = {name: self._draw(value) for name, value in self.conditions.items()}

        game = self.env.unwrapped
        game.gravity = self.current["gravity"]          # the game rebuilds its world with this on reset
        game.wind_power = self.current["wind_power"]
        game.turbulence_power = self.current["turbulence_power"]
        game.enable_wind = game.wind_power > 0 or game.turbulence_power > 0

        obs, info = self.env.reset(seed=seed, options=options)
        info["conditions"] = dict(self.current)
        return obs, info

    def step(self, action):
        # The game hard-codes engine strength as a global constant, so we swap in
        # this episode's strength for the duration of the step, then put it back.
        # (Safe because each simulator runs one step at a time in its own process.)
        scale = self.current["engine_power"]
        lunar_lander_module.MAIN_ENGINE_POWER = DEFAULT_MAIN_ENGINE * scale
        lunar_lander_module.SIDE_ENGINE_POWER = DEFAULT_SIDE_ENGINE * scale
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        finally:
            lunar_lander_module.MAIN_ENGINE_POWER = DEFAULT_MAIN_ENGINE
            lunar_lander_module.SIDE_ENGINE_POWER = DEFAULT_SIDE_ENGINE
        info["conditions"] = dict(self.current)
        return obs, reward, terminated, truncated, info
