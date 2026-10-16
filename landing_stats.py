"""
landing_stats.py — how we judge a landing, by asking the physics engine directly.

Why not trust the game's own leg sensors? While building this project I found a
bug in LunarLander: the ground is made of separate straight segments, and when a
leg slides from one segment onto the next, the game switches that leg's
"touching the ground" flag OFF, even though the leg is still on the ground.
So a lander resting on both legs can look like it's balancing on one.
Everything in this file asks Box2D (the physics engine) instead.
"""

import gymnasium as gym

PAD_HALF_WIDTH = 0.2   # the landing pad spans x = -0.2 ... +0.2 in the lander's coordinates

# Fuel units per step for each action: 0 nothing, 1 left thruster, 2 main engine, 3 right thruster.
# Main engine = 1 unit, side thruster = 0.1 unit: the same 10:1 ratio the game uses for its fuel penalty.
FUEL_COST = {0: 0.0, 1: 0.1, 2: 1.0, 3: 0.1}


def legs_touching(game):
    """How many legs (0, 1 or 2) are PHYSICALLY touching the ground right now.
    `game` is the raw LunarLander (env.unwrapped)."""
    return sum(
        any(edge.contact.touching for edge in leg.contacts)
        for leg in game.legs
    )


def classify_outcome(game, obs):
    """Turn the final moment of an episode into a plain-English outcome."""
    x = obs[0]
    if game.game_over:            # the body (not the legs) hit the ground
        return "crashed"
    if abs(x) >= 1.0:             # drifted off the side of the screen
        return "flew_away"
    if not game.lander.awake:     # the physics engine says: completely at rest
        return "landed" if abs(x) <= PAD_HALF_WIDTH else "landed_off_pad"
    return "never_settled"        # still hovering or fidgeting when time ran out


class LandingStats(gym.Wrapper):
    """Adds `outcome`, `is_success` and `fuel_used` to `info` when an episode ends."""

    def reset(self, **kwargs):
        self.fuel_used = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.fuel_used += FUEL_COST[int(action)]
        if terminated or truncated:
            outcome = classify_outcome(self.env.unwrapped, obs)
            info["outcome"] = outcome
            info["is_success"] = outcome == "landed"   # landed = came to rest on the pad
            info["fuel_used"] = self.fuel_used
        return obs, reward, terminated, truncated, info
