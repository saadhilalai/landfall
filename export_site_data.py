"""
export_site_data.py — gathers everything the LANDFALL website shows into one file,
web/data/site.json. It only reads saved results and measures the simulator; it never
trains or re-tests anything.

What goes in:
  * the pilots, and the slider ranges (with the range the AI practised in)
  * physics constants MEASURED in the simulator, used for the "is a landing even
    possible?" check: how hard the main engine and side thrusters push, and how hard
    wind and turbulence push back
  * the Step 6 test results (landing rates, paired tests), for the expected-landing-rate
    readout, the big numbers and the charts
  * the learning curves (Step 2-5 check-ups), for the learning-curve chart

Usage:
    python export_site_data.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

import config
from envs import make_test_env

OUT = Path("web") / "data" / "site.json"
RESULTS = json.load(open("runs/final/results.json"))

PILOTS = [   # id, name, model in models/ (seed 42), what it is
    dict(id="final", name="Final AI", model="randomized_memory_tuned",
         about="Practised in random conditions, remembers its last 4 moments, auto-tuned settings. "
               "Fast and fuel-efficient, best in wind."),
    dict(id="careful", name="Careful", model="randomized_2m",
         about="Practised 2M steps in random gravity, wind, turbulence and engine power. "
               "Slow and steady, with a big safety margin."),
    dict(id="baseline", name="Baseline", model="baseline",
         about="Practised 1M steps in the normal game only, with the game's own scoring."),
    dict(id="untrained", name="Untrained", model=None,
         about="A brand-new network that hasn't practised at all. It presses buttons at random."),
]

SLIDERS = {   # min, max, step, start value, the range the randomized pilots practised in
    "gravity":    dict(label="Gravity", unit="m/s²", min=4, max=16, step=0.25, value=10,
                       trained=[-config.TRAIN_CONDITIONS["gravity"][1], -config.TRAIN_CONDITIONS["gravity"][0]]),
    "wind":       dict(label="Wind", unit="", min=0, max=25, step=0.25, value=0,
                       trained=list(config.TRAIN_CONDITIONS["wind_power"])),
    "turbulence": dict(label="Turbulence", unit="", min=0, max=3, step=0.025, value=0,
                       trained=list(config.TRAIN_CONDITIONS["turbulence_power"])),
    "engine":     dict(label="Engine power", unit="%", min=50, max=150, step=0.25, value=100,
                       trained=[v * 100 for v in config.TRAIN_CONDITIONS["engine_power"]]),
}

SCENARIO_NAMES = {
    "normal": "Normal", "heavy_gravity": "Heavy gravity", "weak_engine": "Weak engine",
    "strong_wind": "Strong wind", "harsh": "Harsh (all at once)",
    "strong_wind_extreme": "Strong wind, extreme", "harsh_extreme": "Harsh, extreme",
}
AGENT_NAMES = {
    "baseline": "Baseline", "shaped_v1": "Custom reward v1", "shaped_v2": "Custom reward v2",
    "randomized": "Randomized, 1M steps", "shaped_v2_2m": "Custom reward v2, 2M steps",
    "randomized_2m": "Careful (randomized, 2M)", "randomized_memory": "Randomized + memory",
    "randomized_memory_tuned": "Final AI",
}


def as_sliders(conditions):
    c = {**dict(gravity=-10.0, wind_power=0.0, turbulence_power=0.0, engine_power=1.0), **conditions}
    return dict(gravity=-c["gravity"], wind=c["wind_power"], turbulence=c["turbulence_power"],
                engine=c["engine_power"] * 100)


def fresh_lander(seed=0):
    """A lander hanging still and upright in mid-air, in the normal game."""
    env = make_test_env(None)
    env.reset(seed=seed)
    game = env.unwrapped
    for body in [game.lander] + game.legs:
        body.linearVelocity, body.angularVelocity = (0, 0), 0
    game.lander.angle = 0
    return env, game


def measure_physics(trials=40, steps=10):
    """Push the lander with each engine (and with a known torque) and measure how fast it speeds up."""
    dt = steps / 50

    def accel(action, torque=0.0):
        lin, ang = [], []
        for seed in range(trials):
            env, game = fresh_lander(seed)
            for _ in range(steps):
                if torque:
                    game.lander.ApplyTorque(torque, True)
                env.step(action)
            lin.append(np.array(game.lander.linearVelocity) / dt)
            ang.append(game.lander.angularVelocity / dt)
        return np.mean(lin, axis=0), float(np.mean(ang))

    coast, _ = accel(0)
    main, _ = accel(2)
    side, side_spin = accel(3)
    _, torque_spin = accel(0, torque=1.0)
    env, game = fresh_lander()
    return dict(
        main_thrust_accel=float(main[1] - coast[1]),        # m/s² of push from the main engine at 100%
        side_thrust_accel=float(abs(side[0] - coast[0])),   # sideways push from one side thruster
        side_spin_accel=float(abs(side_spin)),              # how fast a side thruster can turn the lander
        spin_per_torque=float(abs(torque_spin)),            # turning from 1 unit of turbulence torque
        lander_mass=float(game.lander.mass),                # the wind pushes the main body only
        total_mass=float(game.lander.mass + sum(leg.mass for leg in game.legs)),
        gust_peak=math.tanh(2.0),                           # strongest gust = 96% of the wind setting
        rest_height=0.595,                                  # lander centre above the pad when parked (measured)
        world=dict(w=20.0, h=40 / 3),
        lander_shape=[list(v) for v in game.lander.fixtures[0].shape.vertices],
        leg_shape=[list(v) for v in game.legs[0].fixtures[0].shape.vertices],
    )


def tests():
    """The Step 6 test conditions with each pilot's landing rate (300 unseen landings each)."""
    out = []
    for sc, conditions in RESULTS["scenarios"].items():
        rates = {p["id"]: RESULTS["cells"][sc][p["model"]]["landed_rate"] for p in PILOTS if p["model"]}
        out.append(dict(id=sc, name=SCENARIO_NAMES[sc], sliders=as_sliders(conditions), landed=rates,
                        landings=RESULTS["cells"][sc]["baseline"]["landings"]))
    return out


def learning_curves():
    curves = {}
    for agent in AGENT_NAMES:
        runs = [np.load(f"runs/{agent}/seed{s}/evaluations.npz") for s in config.TRAIN_SEEDS]
        n = min(len(r["timesteps"]) for r in runs)
        scores = np.array([r["results"][:n].mean(axis=1) for r in runs])
        smooth = lambda y: np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        curves[agent] = dict(steps=[round(float(s) / 1e6, 4) for s in runs[0]["timesteps"][:n]],
                             mean=[round(float(v), 1) for v in smooth(scores.mean(0))],
                             low=[round(float(v), 1) for v in smooth(scores.min(0))],
                             high=[round(float(v), 1) for v in smooth(scores.max(0))])
    return curves


def first_safe_landing(agent, scenario):
    rows = [r for r in csv.DictReader(open("runs/final/episodes.csv"))
            if r["agent"] == agent and r["scenario"] == scenario and r["train_seed"] == "42" and r["outcome"] == "landed"]
    return min(int(r["landing_seed"]) for r in rows)


def main():
    cells = RESULTS["cells"]
    data = dict(
        pilots=PILOTS,
        sliders=SLIDERS,
        physics=measure_physics(),
        tests=tests(),
        first_flight=dict(pilot="final", sliders=as_sliders(config.TEST_SCENARIOS["strong_wind"]),
                          landing=first_safe_landing("randomized_memory_tuned", "strong_wind")),
        results=dict(
            agents=[dict(id=a, name=AGENT_NAMES[a]) for a in RESULTS["agents"]],
            scenarios=[dict(id=s, name=SCENARIO_NAMES[s]) for s in RESULTS["scenarios"]],
            landed={s: {a: cells[s][a]["landed_rate"] for a in RESULTS["agents"]} for s in RESULTS["scenarios"]},
            ci95={s: {a: cells[s][a]["landed_ci95"] for a in RESULTS["agents"]} for s in RESULTS["scenarios"]},
            mean_score={s: {a: round(cells[s][a]["mean_score"], 1) for a in RESULTS["agents"]} for s in RESULTS["scenarios"]},
            seconds={a: cells["normal"][a]["seconds_per_safe_landing"] for a in RESULTS["agents"]},
            fuel={a: cells["normal"][a]["fuel_per_safe_landing"] for a in RESULTS["agents"]},
            paired=[dict(agent=t["agent"], reference=t["reference"], scenario=t["scenario"], p=t["p_value"])
                    for t in RESULTS["paired_tests"]],
            landings_per_cell=cells["normal"]["baseline"]["landings"],
        ),
        curves=learning_curves(),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    p = data["physics"]
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")
    print(f"  main engine {p['main_thrust_accel']:.2f} m/s² (thrust/weight at gravity 10: "
          f"{p['main_thrust_accel'] / 10:.2f}), side thruster {p['side_thrust_accel']:.2f} m/s², "
          f"spin {p['side_spin_accel']:.2f} rad/s² vs {p['spin_per_torque']:.2f} per unit of turbulence")
    print(f"  first flight: strong wind, landing {data['first_flight']['landing']}")


if __name__ == "__main__":
    main()
