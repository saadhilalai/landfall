"""
hard_mode/video.py — a short video for each mastered Hard Mode level (videos/hard_mode/level<N>.mp4).

Same look as the main project's videos (render.py: the physics redrawn as sharp shapes at 1920x1280, 50 fps,
real time), plus what's new in Hard Mode: the fuel gauge, the rocks and walls, the 8 obstacle sensors (thin
lines; teal where a beam hits something), both pads of a mission, and the random weather of that landing.

Which landing (a fixed rule, not cherry-picked): the seed-42 brain that mastered the level (or, if seed 42
didn't master it, the first seed that did), and the FIRST
validation landing (seeds 7000-7199, in order) that lands safely AND shows the level's new challenge:
    level 1  the tank ends below 25% (the fuel limit mattered)
    level 2  the pad is at least 4 m from the middle
    level 3  there's a wall between the start and the pad
    level 4  the whole mission is done (pad A, then pad B)
The video is checked to show exactly the outcome and score that the same landing gets in evaluate.py.

Usage:
    python -m hard_mode.video --level 1
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from domain_randomization import PhysicsConditions
from hard_mode.brain import model_path
from hard_mode.env import H, PAD_HALF, RAY_ANGLES, RAY_LENGTH, W
from hard_mode.evaluate import fly_many
from hard_mode.levels import BY_KEY, VALIDATION_SEEDS
from hard_mode.worlds import make_env
from render import current_wind, draw_wind_arrow, font, render_hires

OUT = Path("videos") / "hard_mode"
K = 3.2
S = 30.0 * K                          # pixels per world unit
FPS = 50
HOLD_START, HOLD_END = 0.6, 1.8
TEAL, RED, WHITE, GREY = (41, 199, 172), (255, 95, 87), (255, 255, 255), (150, 150, 150)


def px(x, y):
    return x * S, (H - y) * S


def shows_challenge(level, game, result):
    if result["outcome"] != "landed":
        return False
    if level.number == 1:
        return game.fuel_left < 0.25 * level.fuel_tank
    if level.number == 2:
        return abs(game.pads[0] - W / 2) >= 4
    if level.number == 3:
        return any(len(p) == 4 for p in game.obstacles)      # walls have 4 corners, rocks 6
    return True                                               # level 4: "landed" = whole mission done


TITLE_NOTE = ""      # optional note after the level name (e.g. "best attempt, not mastered")


def overlay(frame, level, game, obs, cond, ending=None):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    for poly in getattr(game, "obstacles", []):
        pts = [px(x, y) for x, y in poly]
        d.polygon(pts, fill=(200, 200, 200, 255), outline=WHITE)
    # the other pad of a mission (the game draws flags only at the current target)
    for i, pad in enumerate(game.pads):
        y1 = (H - game.helipad_y) * S
        for x in (pad - PAD_HALF, pad + PAD_HALF):
            if pad != game.target_x:
                d.line([(x * S, y1), (x * S, y1 - 50 * K)], fill=WHITE, width=max(1, round(K)))
                d.polygon([(x * S, y1 - 50 * K), (x * S, y1 - 40 * K), (x * S + 25 * K, y1 - 45 * K)], fill=(204, 204, 0))
        if level.mission:
            done = i == 0 and game.phase == 1
            d.text((pad * S, y1 - 80 * K), ("A" if i == 0 else "B") + (" done" if done else ""), fill=WHITE,
                   font=font(round(30 * K / 2)), anchor="mm")
    # obstacle sensors
    if level.obstacles:
        lx, ly = game.lander.position
        for deg, closeness in zip(RAY_ANGLES, obs[10:18]):
            frac = 1.0 - closeness
            r = math.radians(deg)
            end = (lx + frac * RAY_LENGTH * math.sin(r), ly - frac * RAY_LENGTH * math.cos(r))
            hit = frac < 1.0
            d.line([px(lx, ly), px(*end)], fill=(*TEAL, 230) if hit else (255, 255, 255, 45), width=2)
            if hit:
                cx, cy = px(*end)
                d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=TEAL)
    # fuel gauge and title (top left, clear of the wind arrow in the middle)
    d.text((48, 34), f"LEVEL {level.number}  ·  {level.name.upper()}", fill=WHITE, font=font(34))
    y = 92
    if TITLE_NOTE:                                     # e.g. "best attempt, not mastered", on its own line
        d.text((48, 78), TITLE_NOTE, fill=GREY, font=font(24))
        y = 118
    share = game.fuel_left / level.fuel_tank
    d.text((48, y), "FUEL", fill=WHITE, font=font(28))
    d.rectangle([128, y + 4, 128 + 300, y + 26], outline=WHITE, width=2)
    d.rectangle([131, y + 7, 131 + round(294 * share), y + 23], fill=RED if share < 0.2 else WHITE)
    d.text((446, y), f"{game.fuel_left:.0f} / {level.fuel_tank:.0f}", fill=WHITE, font=font(28))
    weather = (f"gravity {-cond['gravity']:.1f}   wind {cond['wind_power']:.1f}   "
               f"turbulence {cond['turbulence_power']:.2f}   engine {cond['engine_power'] * 100:.0f}%")
    d.text((img.width - 48, 40), weather, fill=GREY, font=font(30), anchor="ra")
    if ending:
        d.text((img.width / 2, img.height * 0.42), ending, fill=WHITE, font=font(96), anchor="mm")
    frame = np.array(img)
    if game.wind_power > 0:
        frame = draw_wind_arrow(frame, current_wind(game), max(game.wind_power, 1e-6))
    return frame


def conditions_of(env):
    """This landing's random weather and engine power (kept by the PhysicsConditions layer)."""
    while not isinstance(env, PhysicsConditions):
        env = env.env
    return env.current


def fly(model, level, seed):
    env = make_env(level, render_mode="rgb_array")
    game = env.unwrapped
    obs, info = env.reset(seed=seed)
    cond = conditions_of(env)
    frames, score, done = [], 0.0, False
    frames.append(overlay(render_hires(game, K), level, game, obs[-1], cond))
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        score += reward
        done = terminated or truncated
        frames.append(overlay(render_hires(game, K), level, game, obs[-1], cond))
    ending = {"landed": "MISSION COMPLETE" if level.mission else "LANDED", "crashed": "CRASHED",
              "hit_obstacle": "HIT AN OBSTACLE", "flew_away": "FLEW AWAY", "out_of_time": "OUT OF TIME",
              "landed_off_pad": "LANDED OFF THE PAD", "stuck_on_first_pad": "NEVER TOOK OFF"}[info["outcome"]]
    last = overlay(render_hires(game, K), level, game, obs[-1], cond, ending)
    env.close()
    frames = [frames[0]] * int(HOLD_START * FPS) + frames + [last] * int(HOLD_END * FPS)
    return frames, info["outcome"], score


def pick(model, level):
    """The first validation landing that matches the rule above, plus its evaluate.py result."""
    results = fly_many(model, level, VALIDATION_SEEDS)
    for r in results:
        if r["outcome"] != "landed":
            continue
        env = make_env(level)
        obs, _ = env.reset(seed=r["seed"])
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if shows_challenge(level, env.unwrapped, r):
            return r
    return next(r for r in results if r["outcome"] == "landed")


def write(path, frames):
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=16,
                                output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  {path}  {frames[0].shape[1]}x{frames[0].shape[0]}, {len(frames) / FPS:.1f} s, "
          f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, help="1, 2, 3, 3a, 3b, 4a or 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="default: the level's best brain for --seed")
    parser.add_argument("--out", default=None)
    parser.add_argument("--note", default="", help="shown after the level name, e.g. 'best attempt, not mastered'")
    args = parser.parse_args()
    torch.set_num_threads(1)
    level = BY_KEY[args.level]
    TITLE_NOTE = args.note
    model = PPO.load(args.model or model_path(args.level, args.seed), device="cpu")
    chosen = pick(model, level)
    frames, outcome, score = fly(model, level, chosen["seed"])
    assert outcome == chosen["outcome"] and abs(score - chosen["score"]) < 0.01, (chosen, outcome, score)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Level {args.level}: validation landing {chosen['seed']} ({outcome}, score {score:.1f}, fuel {chosen['fuel_used']})")
    write(Path(args.out) if args.out else OUT / f"level{args.level}.mp4", frames)
