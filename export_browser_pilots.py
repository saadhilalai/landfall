"""
export_browser_pilots.py — saves each pilot's neural network for the website (web/data/pilots.json).

The website flies every landing inside the visitor's browser, so it needs the trained
networks in a form JavaScript can read. Each pilot's "brain" is three layers of numbers
(weights and biases); this file writes them out as 32-bit floats, exactly as trained.
Only the part that picks actions is saved: the critic (the part that guessed scores
during training) isn't needed to fly.

Usage:
    python export_browser_pilots.py
"""

import base64
import json

import numpy as np
from stable_baselines3 import PPO

import config
from envs import make_vec_env, model_memory

SITE = json.load(open("web/data/site.json"))
OUT = "web/data/pilots.json"
LAYERS = ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net")


def load(model_name):
    if model_name is None:   # the untrained pilot: a fresh network, the same one app.py builds
        return PPO(env=make_vec_env("baseline", n_envs=1, subproc=False), seed=42, device="cpu", verbose=0,
                   **config.PPO_KWARGS)
    return PPO.load(f"models/{model_name}_seed42.zip", device="cpu")


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()


def export():
    pilots = {}
    for p in SITE["pilots"]:
        model = load(p["model"])
        weights = model.policy.state_dict()
        layers = []
        for name in LAYERS:
            w, b = weights[f"{name}.weight"].numpy(), weights[f"{name}.bias"].numpy()
            layers.append(dict(inputs=w.shape[1], outputs=w.shape[0], weight=b64(w), bias=b64(b)))
        pilots[p["id"]] = dict(model=p["model"], memory=model_memory(model),
                               deterministic=p["model"] is not None, layers=layers)
        print(f"{p['id']:10s} memory {model_memory(model)}  layers {[(l['inputs'], l['outputs']) for l in layers]}")
    with open(OUT, "w") as f:
        json.dump(dict(activation="tanh", note="weights are little-endian float32, row-major (outputs x inputs)",
                       pilots=pilots), f, separators=(",", ":"))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    export()
