"""
hard_mode/worlds.py — builds Hard Mode practice and test worlds, with the same layers as the main project:

    HardLander          the game with the Hard Mode additions (env.py)
      -> PhysicsConditions  Step 4's random gravity, wind, turbulence and engine power, re-rolled every landing
      -> TimeLimit          the level's time limit
      -> HardStats          how the landing ended (landed / crashed / hit an obstacle / ...) and fuel used
      -> Monitor            records the official score (practice worlds only)
      -> HardShaping        the v2 reward rules from Step 3 (practice worlds only)
      -> FrameStack         the last 4 moments (memory), like the Final AI
"""

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TimeLimit
from stable_baselines3.common.monitor import Monitor

import config
from domain_randomization import PhysicsConditions
from hard_mode.env import HardLander
from reward_shaping import V2

MEMORY = config.MEMORY
# Mission: on pad A the Level 3 brain fires the side thruster toward pad B 99% of the time and the main engine
# 0.1%, so it never discovers lift-off by chance. A small bonus for trying the main engine while waiting on pad A
# makes it experiment; capped at 20 steps (20 points), it can't be farmed (never leaving still scores -100).
TAKEOFF_TRY_BONUS, TAKEOFF_TRIES = 1.0, 20


class HardStats(gym.Wrapper):
    """Adds outcome, is_success and fuel_used to `info` when a landing ends."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            game = self.env.unwrapped
            info["outcome"] = game.outcome(obs)
            info["is_success"] = info["outcome"] == "landed"
            info["fuel_used"] = game.fuel_used
        return obs, reward, terminated, truncated, info


class HardShaping(gym.Wrapper):
    """Step 3's v2 reward rules (reward_shaping.py), measured against the target pad. On a mission, between
    landing on pad A and taking off again, firing the engines on the ground is the point: the hover and
    "not settling" penalties are paused until it is airborne, and trying the main engine earns a small bonus."""

    def __init__(self, env, level, settings=V2):
        super().__init__(env)
        self.s, self.max_steps = settings, level.max_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps, self.prev_obs, self.prev_legs, self.takeoff_tries = 0, obs, 0, 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        s, game = self.s, self.env.unwrapped
        self.steps += 1
        legs = game.legs_on_ground()
        height, down_speed, tilt = obs[1], -obs[3], abs(obs[4])
        leaving_pad = game.level.mission and game.phase == 1 and not game.taken_off
        extra = 0.0
        if legs == 0:
            extra -= s.tilt_cost * tilt
        if leaving_pad and int(action) == 2 and self.takeoff_tries < TAKEOFF_TRIES:
            extra += TAKEOFF_TRY_BONUS      # nudge: try the main engine on pad A (at most 20 points per mission)
            self.takeoff_tries += 1
        if not leaving_pad:
            if legs == 0 and height < s.hover_height and down_speed < s.hover_speed:
                extra -= s.hover_cost
            if legs == 1 or (legs == 2 and int(action) != 0):
                extra -= s.not_settled_cost
        if legs > 0 and self.prev_legs == 0:
            too_fast = max(0.0, -self.prev_obs[3] - s.safe_touchdown_speed)
            extra -= min(s.hard_landing_cost * too_fast, s.hard_landing_max)
        if terminated and game.outcome(obs) == "landed":
            extra += s.settle_bonus * (1.0 - self.steps / self.max_steps)
            extra += s.fuel_bonus * max(0.0, s.fuel_budget - game.fuel_used)
        self.prev_obs, self.prev_legs = obs, legs
        return obs, reward + extra, terminated, truncated, info


def make_env(level, practice=False, monitor_file=None, render_mode=None, conditions=config.TRAIN_CONDITIONS):
    """A Hard Mode world. practice=True adds the score log and the training reward rules."""
    env = HardLander(level, render_mode=render_mode, practice=practice)
    env = PhysicsConditions(env, conditions)
    env = TimeLimit(env, level.max_steps)
    env = HardStats(env)
    if practice:
        env = Monitor(env, filename=monitor_file, info_keywords=("outcome", "fuel_used"))
        env = HardShaping(env, level)
    return FrameStackObservation(env, MEMORY)
