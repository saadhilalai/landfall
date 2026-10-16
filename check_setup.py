"""
check_setup.py — Step 1 sanity check.

Proves that:
  1. The physics simulator (Box2D) loads.
  2. LunarLander-v3 runs a full episode.
  3. We can render frames (needed later for videos and the web demo).
  4. We can change gravity and wind (needed later for domain randomization).

Run it with:  .venv/bin/python check_setup.py
"""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image


def run_random_episode(env, seed=0):
    """Fly the lander with random button-presses until it crashes or lands."""
    obs, info = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    done = False
    while not done:
        action = env.action_space.sample()  # pick a random engine command
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return total_reward, steps


def main():
    print(f"PyTorch {torch.__version__} | device: cpu | threads: {torch.get_num_threads()}")
    print(f"Gymnasium {gym.__version__}\n")

    # --- 1 & 2: default environment ---------------------------------------
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    print("Observation (what the AI sees):", env.observation_space.shape, "numbers")
    print("  -> x, y, x-speed, y-speed, angle, spin, left-leg-touching, right-leg-touching")
    print("Actions (what the AI can do):", env.action_space.n, "choices")
    print("  -> 0 do nothing, 1 fire left engine, 2 fire main engine, 3 fire right engine\n")

    score, steps = run_random_episode(env)
    print(f"Random pilot: score {score:.1f} after {steps} steps (expect a big negative = crash)")

    # --- 3: rendering -------------------------------------------------------
    env.reset(seed=0)
    frame = env.render()
    Image.fromarray(frame).save("setup_check_frame.png")
    print(f"Rendered a frame of size {frame.shape} -> saved setup_check_frame.png")
    env.close()

    # --- 4: physics knobs we'll randomize later -----------------------------
    harsh = gym.make(
        "LunarLander-v3",
        gravity=-11.5,          # stronger pull than default (-10)
        enable_wind=True,
        wind_power=18.0,        # sideways gusts (default 15)
        turbulence_power=1.8,   # random spinning push (default 1.5)
    )
    score, steps = run_random_episode(harsh, seed=1)
    print(f"Harsh-weather env works too: random pilot scored {score:.1f}")
    harsh.close()

    print("\nAll checks passed. The simulator is ready.")


if __name__ == "__main__":
    main()
