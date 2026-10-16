"""
make_videos.py — renders the promo/README videos as high-resolution MP4s.

Resolution: the simulator itself only draws at 600x400, and its physics is tied to
that drawing scale, so we can't just turn the resolution up. Instead, render_hires()
redraws the same physics objects (terrain, lander, legs, flags, engine flames) as
vector shapes at 1920x1280 (3.2x). Real detail, not blurry upscaling, and the same
look and colors as the game. 50 frames per second = the simulator's real time.

Which landings (fixed rules, written down before the final evaluation ran; see final_eval.py):
the seed-42 model of each agent, and the FIRST test landing (seeds 10000-10099, in order)
that matches. Every recorded landing is checked against runs/final/episodes.csv to make
sure the video shows exactly what the evaluation measured.

    untrained_crash.mp4             a brand-new untrained network, first test landing that crashes
    baseline_landing.mp4            baseline, normal game, first safe landing
    final_strong_wind.mp4           final agent, strong wind, first safe landing
    final_strong_wind_arrow.mp4     the same landing with a small wind indicator
    side_by_side_heavy_gravity.mp4  no-randomization agent (2M steps) vs final agent, heavy gravity,
                                    first landing where the first fails and the final lands
                                    (+ each half as its own clean clip)
    broll_1.mp4 ... broll_5.mp4     final agent, normal game, first 5 safe landings

Usage:
    python make_videos.py
"""

import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pygame
import torch
from gymnasium.envs.box2d import lunar_lander as ll
from PIL import Image, ImageDraw, ImageFont
from pygame import gfxdraw
from stable_baselines3 import PPO

import config
from envs import make_test_env, make_vec_env, model_memory

OUT = Path("videos")
K = 3.2                       # 600x400 -> 1920x1280
FPS = ll.FPS                  # 50
HOLD_START, HOLD_END = 0.5, 1.5   # seconds of still frame before and after each landing
FINAL = "randomized_memory_tuned"
NO_RANDOMIZATION = "shaped_v2_2m"
SEED = 42
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


# ---- Rendering ------------------------------------------------------------------
def render_hires(game, k=K):
    """LunarLander's own render(), redrawn at k times the size. Like the original, it also
    ages the engine-flame particles (they never touch the lander, so physics is unaffected)."""
    S = ll.SCALE * k
    surf = pygame.Surface((int(ll.VIEWPORT_W * k), int(ll.VIEWPORT_H * k)))
    surf.fill((255, 255, 255))

    for obj in game.particles:
        obj.ttl -= 0.15
        color = (int(max(0.2, 0.15 + obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255),
                 int(max(0.2, 0.5 * obj.ttl) * 255))
        obj.color1 = obj.color2 = color
    game._clean_particles(False)

    for poly in game.sky_polys:
        points = [(x * S, y * S) for x, y in poly]
        pygame.draw.polygon(surf, (0, 0, 0), points)
        gfxdraw.aapolygon(surf, points, (0, 0, 0))

    for obj in game.particles + game.drawlist:
        for f in obj.fixtures:
            trans = f.body.transform
            if isinstance(f.shape, ll.circleShape):
                c = trans * f.shape.pos * S
                r = f.shape.radius * S
                pygame.draw.circle(surf, obj.color1, (c[0], c[1]), r)
            else:
                path = [tuple(trans * v * S) for v in f.shape.vertices]
                pygame.draw.polygon(surf, obj.color1, path)
                gfxdraw.aapolygon(surf, path, obj.color1)
                pygame.draw.aalines(surf, obj.color2, True, path)

    for x in (game.helipad_x1, game.helipad_x2):
        x *= S
        y1 = game.helipad_y * S
        y2 = y1 + 50 * k
        pygame.draw.line(surf, (255, 255, 255), (x, y1), (x, y2), max(1, round(k)))
        flag = [(x, y2), (x, y2 - 10 * k), (x + 25 * k, y2 - 5 * k)]
        pygame.draw.polygon(surf, (204, 204, 0), flag)
        gfxdraw.aapolygon(surf, flag, (204, 204, 0))

    surf = pygame.transform.flip(surf, False, True)
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)).copy()


def current_wind(game):
    """The sideways wind force the game applied on the last step (0 when landed or wind off)."""
    if not game.enable_wind or game.legs[0].ground_contact or game.legs[1].ground_contact:
        return 0.0
    i = game.wind_idx - 1
    return math.tanh(math.sin(0.02 * i) + math.sin(math.pi * 0.01 * i)) * game.wind_power


def draw_wind_arrow(frame, wind, wind_power):
    """Small white arrow at the top centre: direction and strength of the wind right now."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    w = img.width
    cx, cy = w // 2, 120
    font = ImageFont.truetype(FONT, 40)
    d.text((cx, cy - 70), "WIND", fill=(255, 255, 255), font=font, anchor="mm")
    length = 260 * wind / max(wind_power, 1e-6)
    if abs(length) > 8:
        tip = cx + length
        d.line([(cx - length * 0.0, cy), (tip, cy)], fill=(255, 255, 255), width=8)
        head = 26 if length > 0 else -26
        d.polygon([(tip + head, cy), (tip - head * 0.2, cy - 20), (tip - head * 0.2, cy + 20)], fill=(255, 255, 255))
    else:
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 255))
    return np.array(img)


def label(frame, text, size=48, xy=(40, 36)):
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text(xy, text, fill=(255, 255, 255), font=ImageFont.truetype(FONT, size))
    return np.array(img)


# ---- Flying ---------------------------------------------------------------------
def untrained_model():
    env = make_vec_env("baseline", n_envs=1, subproc=False)
    return PPO(env=env, seed=SEED, device="cpu", verbose=0, **config.PPO_KWARGS)


def fly(model, scenario, landing_seed, deterministic=True, wind_arrow=False):
    """Fly one landing and return (frames, outcome, score). Frames include the start/end holds."""
    conditions = config.TEST_SCENARIOS[scenario]
    # render_mode turns on the game's engine-flame particles (pure decoration: the thrust on the
    # lander is the same either way, and check() confirms every landing matches the evaluation).
    env = make_test_env(conditions, render_mode="rgb_array", memory=model_memory(model))
    game = env.unwrapped
    torch.manual_seed(landing_seed)            # only matters for the untrained (random) pilot
    obs, _ = env.reset(seed=landing_seed)
    frames, score, done = [render_hires(game)], 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        frame = render_hires(game)
        if wind_arrow:
            frame = draw_wind_arrow(frame, current_wind(game), game.wind_power)
        frames.append(frame)
    env.close()
    frames = [frames[0]] * int(HOLD_START * FPS) + frames + [frames[-1]] * int(HOLD_END * FPS)
    return frames, info["outcome"], score


def write(path, frames):
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                                macro_block_size=16,
                                output_params=["-crf", "16", "-preset", "slow", "-movflags", "+faststart"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    h, w = frames[0].shape[:2]
    print(f"  {path}  {w}x{h}, {len(frames) / FPS:.1f} s, {path.stat().st_size / 1e6:.1f} MB", flush=True)


# ---- Picking landings from the final results ------------------------------------
EPISODES = list(csv.DictReader(open("runs/final/episodes.csv")))


def recorded(agent, scenario):
    rows = [r for r in EPISODES if r["agent"] == agent and r["scenario"] == scenario and int(r["train_seed"]) == SEED]
    return {int(r["landing_seed"]): r for r in rows}


def first_landings(agent, scenario, n=1):
    rows = recorded(agent, scenario)
    return [s for s in sorted(rows) if rows[s]["outcome"] == "landed"][:n]


def check(agent, scenario, landing_seed, outcome, score):
    """The video must show exactly what the final evaluation recorded."""
    r = recorded(agent, scenario)[landing_seed]
    assert r["outcome"] == outcome and abs(float(r["score"]) - score) < 1e-6, (agent, scenario, landing_seed, r, outcome, score)


def load(agent):
    return PPO.load(f"models/{agent}_seed{SEED}.zip", device="cpu")


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    pygame.init()

    print("Untrained AI crashing")
    untrained = untrained_model()
    for seed in range(config.TEST_SEED, config.TEST_SEED + 100):
        frames, outcome, _ = fly(untrained, "normal", seed, deterministic=False)
        if outcome == "crashed":
            write(OUT / "untrained_crash.mp4", frames)
            break

    print("Baseline landing")
    seed = first_landings("baseline", "normal")[0]
    frames, outcome, score = fly(load("baseline"), "normal", seed)
    check("baseline", "normal", seed, outcome, score)
    write(OUT / "baseline_landing.mp4", frames)

    print("Final AI in strong wind")
    final = load(FINAL)
    seed = first_landings(FINAL, "strong_wind")[0]
    for arrow, name in ((False, "final_strong_wind.mp4"), (True, "final_strong_wind_arrow.mp4")):
        frames, outcome, score = fly(final, "strong_wind", seed, wind_arrow=arrow)
        check(FINAL, "strong_wind", seed, outcome, score)
        write(OUT / name, frames)

    print("Side by side in heavy gravity")
    a, b = recorded(NO_RANDOMIZATION, "heavy_gravity"), recorded(FINAL, "heavy_gravity")
    seed = next(s for s in sorted(a) if a[s]["outcome"] != "landed" and b[s]["outcome"] == "landed")
    left, out_l, score_l = fly(load(NO_RANDOMIZATION), "heavy_gravity", seed)
    right, out_r, score_r = fly(final, "heavy_gravity", seed)
    check(NO_RANDOMIZATION, "heavy_gravity", seed, out_l, score_l)
    check(FINAL, "heavy_gravity", seed, out_r, score_r)
    write(OUT / "heavy_gravity_no_randomization.mp4", left)
    write(OUT / "heavy_gravity_final.mp4", right)
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    combined = []
    for l_frame, r_frame in zip(left, right):
        halves = []
        for frame, name in ((l_frame, "Trained in normal conditions only"),
                            (r_frame, "Final AI: randomized + memory + tuned")):
            frame = label(frame, name)                       # labels sit in the black sky
            halves.append(label(frame, "Same landing  |  gravity -12, never seen in training", size=34, xy=(40, 100)))
        frame = np.concatenate(halves, axis=1)
        mid = l_frame.shape[1]
        frame[:, mid - 4:mid + 4] = 255                      # thin white divider
        combined.append(frame)
    write(OUT / "side_by_side_heavy_gravity.mp4", combined)
    print(f"    landing {seed}: no randomization -> {out_l}, final -> {out_r}")

    print("B-roll: final AI, normal game")
    for i, seed in enumerate(first_landings(FINAL, "normal", 5), start=1):
        frames, outcome, score = fly(final, "normal", seed)
        check(FINAL, "normal", seed, outcome, score)
        write(OUT / f"broll_{i}.mp4", frames)


if __name__ == "__main__":
    main()
