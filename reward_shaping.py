"""
reward_shaping.py — our own reward rules, added ON TOP of the game's official reward.

The official reward already pays for getting close to the pad, slowing down and
touching legs down, and charges a little for fuel. On top of that we add:

  PENALTIES, every step
    1. Tilt          lose points for leaning while flying: stay upright.
    2. Hover         lose points for floating low over the ground without coming down.
    3. Not settling  lose points for balancing on one leg, or for still firing
                     engines when both legs are already down.
  PENALTY, each time the legs touch down
    4. Hard landing  lose points for touching down faster than a safe speed (capped).
  BONUSES, only when the lander comes to a complete rest on the pad
    5. Settle bonus  worth more the sooner it settles.
    6. Fuel bonus    points for every unit of fuel left in a 200-unit budget.

How we stop the AI from "gaming" these points:
  * Bonuses are only paid for a real landing on the pad. Otherwise the cheapest
    way to "save fuel" would be to never fire the engines and just crash.
  * Once the lander is on the ground, the only per-step penalty is "not settling",
    at most 0.1 per step (at most ~100 per episode). So crashing on purpose (-100)
    is never a shortcut to escape it. That's also why tilt only counts while
    flying: a lander parked crooked on a slope would otherwise pile up penalties.
  * Leg contact comes straight from the physics engine (see landing_stats.py),
    so a faulty leg sensor can't trigger a penalty.
  * Every agent is GRADED on the official score, never on these points.

Two versions:
  v1  first design; numbers picked by measuring how the baseline agent flies.
  v2  one tuning round. v1 fixed hovering but burned MORE fuel: its "safe" touchdown
      speed (the baseline's median) was so strict that it crept down slowly, burning
      fuel all the way. v2 relaxes the safe speed and doubles the fuel bonus.
"""

from dataclasses import dataclass, replace

import gymnasium as gym

from landing_stats import FUEL_COST, classify_outcome, legs_touching


@dataclass(frozen=True)
class RewardSettings:
    tilt_cost: float = 1.0              # points per step, per radian of lean (baseline leans ~0.07 rad)
    hover_height: float = 0.2           # "low" = below this height (pad = 0, start = ~1.4)
    hover_speed: float = 0.05           # "hovering" = coming down slower than this (normal final approach ~0.15)
    hover_cost: float = 0.1             # points per step spent hovering low
    not_settled_cost: float = 0.1       # points per step on one leg, or firing engines with both legs down
    safe_touchdown_speed: float = 0.15  # downward speed at touchdown that costs nothing
    hard_landing_cost: float = 200.0    # points per unit of speed above the safe speed
    hard_landing_max: float = 30.0      # cap: much faster is basically a crash, which already costs -100.
                                        # Without the cap the ground becomes scarier than flying
                                        # off-screen, and the AI could learn to avoid landing at all.
    settle_bonus: float = 100.0         # full bonus for settling instantly, shrinking to 0 at the time limit
    fuel_budget: float = 200.0          # fuel units (the baseline burns ~136 per landing)
    fuel_bonus: float = 0.25            # points per unit of fuel left in the budget


V1 = RewardSettings()
V2 = replace(
    V1,
    safe_touchdown_speed=0.20,   # was 0.15 (the baseline's median); its safe landings rarely exceed 0.20
    fuel_bonus=0.5,              # was 0.25
)
VERSIONS = {"v1": V1, "v2": V2}


class LandingRewardShaping(gym.Wrapper):
    """Adds the six rules above to the official reward."""

    TERMS = ("tilt", "hover", "not_settled", "hard_landing", "settle_bonus", "fuel_bonus")

    def __init__(self, env, settings=V2):
        super().__init__(env)
        self.s = settings
        self.max_steps = env.spec.max_episode_steps   # 1000 for LunarLander

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps, self.fuel_used = 0, 0.0
        self.prev_obs, self.prev_legs = obs, 0
        self.totals = dict.fromkeys(self.TERMS, 0.0)
        self.shaped_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        s = self.s
        game = self.env.unwrapped
        action = int(action)
        self.steps += 1
        self.fuel_used += FUEL_COST[action]
        legs = legs_touching(game)
        height, down_speed, tilt = obs[1], -obs[3], abs(obs[4])
        extra = dict.fromkeys(self.TERMS, 0.0)

        # 1. Tilt, while flying: lean angle in radians (0 = perfectly upright).
        if legs == 0:
            extra["tilt"] = -s.tilt_cost * tilt

        # 2. Hover: in the air, close to the ground, and not really coming down.
        if legs == 0 and height < s.hover_height and down_speed < s.hover_speed:
            extra["hover"] = -s.hover_cost

        # 3. Not settling: balancing on one leg, or fidgeting with the engines on two.
        if legs == 1 or (legs == 2 and action != 0):
            extra["not_settled"] = -s.not_settled_cost

        # 4. Hard landing: legs just touched down from the air, so check the speed of
        #    the impact (measured one step earlier, before the ground slowed us down).
        if legs > 0 and self.prev_legs == 0:
            impact_speed = -self.prev_obs[3]
            too_fast = max(0.0, impact_speed - s.safe_touchdown_speed)
            extra["hard_landing"] = -min(s.hard_landing_cost * too_fast, s.hard_landing_max)

        # 5 + 6. Bonuses, only for coming to a complete rest on the pad.
        if terminated and classify_outcome(game, obs) == "landed":
            time_left = 1.0 - self.steps / self.max_steps
            extra["settle_bonus"] = s.settle_bonus * time_left
            extra["fuel_bonus"] = s.fuel_bonus * max(0.0, s.fuel_budget - self.fuel_used)

        shaped_reward = reward + sum(extra.values())

        # Bookkeeping, so TensorBoard can show which rules are doing what.
        for term, value in extra.items():
            self.totals[term] += value
        self.shaped_return += shaped_reward
        self.prev_obs, self.prev_legs = obs, legs
        if terminated or truncated:
            info["shaped_return"] = self.shaped_return
            info["shaping"] = dict(self.totals)

        return obs, shaped_reward, terminated, truncated, info
