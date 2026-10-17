"""
app.py — Step 7: LANDFALL, the live demo website (local server + Python flight reference).

How the site works, in plain words: the web page flies every landing itself, in the
visitor's browser (web/sim: the game rewritten in JavaScript, plus the trained networks
exported by export_browser_pilots.py), and draws it live on a canvas with launch-webcast
telemetry. So the public site is just files, hosted free as a static Hugging Face Space
(make_space.py).

This server is for running it on your own computer: it serves the page and the video
clips, and keeps the original Python flight at POST /api/fly. That Python version is the
reference: check_browser_sim.py flies the same landings both ways and compares them.
Same landing number + same conditions + same pilot = exactly the same flight, every time.
Landings 10000-10099 are the official test landings from Step 6.

Usage:
    python app.py          then open http://127.0.0.1:7860
"""

import json
import math
import os
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

import config
from envs import make_test_env, make_vec_env, model_memory
from landing_stats import FUEL_COST, legs_touching
from render import current_wind

torch.set_num_threads(1)
SITE = json.load(open("web/data/site.json"))   # pilots, slider ranges (made by export_site_data.py)
SLIDERS = SITE["sliders"]


def load_pilot(model_name):
    if model_name is None:   # a fresh, untrained network with the same shape as the baseline's
        return PPO(env=make_vec_env("baseline", n_envs=1, subproc=False), seed=42, device="cpu", verbose=0,
                   **config.PPO_KWARGS)
    return PPO.load(f"models/{model_name}_seed42.zip", device="cpu")


MODELS = {p["id"]: load_pilot(p["model"]) for p in SITE["pilots"]}
TRAINED = {p["id"]: p["model"] is not None for p in SITE["pilots"]}


# ---- Flying ---------------------------------------------------------------------
FLIGHT_LOCK = threading.Lock()   # engine power is set by swapping a global constant: one flight at a time


def to_conditions(s):
    """Slider values -> the PhysicsConditions format (gravity points down, engine as a multiplier)."""
    return dict(gravity=-s["gravity"], wind_power=s["wind"], turbulence_power=s["turbulence"],
                engine_power=s["engine"] / 100)


def record_flight(pilot, conditions, landing):
    """Fly one landing and record it, step by step."""
    model = MODELS[pilot]
    env = make_test_env(conditions, memory=model_memory(model))   # exactly the Step 6 test setup
    game = env.unwrapped
    torch.manual_seed(landing)              # only the untrained pilot uses randomness; this makes it repeatable
    obs, _ = env.reset(seed=landing)

    keys = ("x", "y", "a", "vx", "vy", "l0x", "l0y", "l0a", "l1x", "l1y", "l1a")
    rec = {k: [] for k in keys + ("act", "wind", "fuel", "legs")}
    fuel, contact_speed = 0.0, None

    def snapshot(action):
        body, (leg0, leg1) = game.lander, game.legs
        values = (body.position.x, body.position.y, body.angle, body.linearVelocity.x, body.linearVelocity.y,
                  leg0.position.x, leg0.position.y, leg0.angle, leg1.position.x, leg1.position.y, leg1.angle)
        for k, v in zip(keys, values):
            rec[k].append(round(float(v), 3))
        rec["act"].append(int(action))
        rec["wind"].append(round(current_wind(game), 2))
        rec["fuel"].append(round(fuel, 1))
        rec["legs"].append(legs_touching(game))

    snapshot(0)
    score, done = 0.0, False
    while not done:
        before = math.hypot(*game.lander.linearVelocity)
        action, _ = model.predict(obs, deterministic=TRAINED[pilot])
        obs, reward, terminated, truncated, info = env.step(action)
        fuel += FUEL_COST[int(action)]
        score += reward
        done = terminated or truncated
        if contact_speed is None and (legs_touching(game) or game.game_over):
            contact_speed = before          # speed just before the first touch of the ground
        snapshot(action)
    env.close()

    terrain = [list(poly[0]) for poly in game.sky_polys] + [list(game.sky_polys[-1][1])]
    return dict(pilot=pilot, outcome=info["outcome"], seconds=(len(rec["x"]) - 1) / 50,
                fuel=round(fuel, 1), contact_speed=None if contact_speed is None else round(contact_speed, 2),
                score=round(score, 1), terrain=[[round(x, 3), round(y, 3)] for x, y in terrain],
                pad=[round(game.helipad_x1, 3), round(game.helipad_x2, 3), round(game.helipad_y, 3)], frames=rec)


# ---- Web server -----------------------------------------------------------------
class FlightRequest(BaseModel):
    pilots: list[str] = Field(min_length=1, max_length=2)
    gravity: float
    wind: float
    turbulence: float
    engine: float
    landing: int = Field(ge=0, le=999_999)


app = FastAPI(title="LANDFALL", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.post("/api/fly")
def fly(req: FlightRequest):
    if any(p not in MODELS for p in req.pilots):
        raise HTTPException(400, f"Unknown pilot. Choose from {list(MODELS)}")
    sliders = {name: min(max(getattr(req, name), s["min"]), s["max"]) for name, s in SLIDERS.items()}
    conditions = to_conditions(sliders)
    with FLIGHT_LOCK:
        flights = [record_flight(p, conditions, req.landing) for p in req.pilots]
    return dict(landing=req.landing, sliders=sliders, flights=flights)


app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/", StaticFiles(directory="web", html=True), name="site")

if __name__ == "__main__":
    # Locally: http://127.0.0.1:7860. On Hugging Face Spaces the Dockerfile sets HOST=0.0.0.0.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 7860)))
