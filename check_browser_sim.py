"""
check_browser_sim.py — checks that the browser version of the game flies like the Python one.

The website flies landings in the visitor's browser (web/sim: the game rewritten in
JavaScript, the trained networks exported by export_browser_pilots.py). This script flies
the same pilots on the same official test landings in both versions, and compares:
  - did each landing end the same way (landed / crashed / ...)?
  - the landing rate of each pilot in each test scenario
  - how long the two flights stay identical before rounding differences make them drift

Needs Node.js. Writes runs/browser_check.json.

Usage:
    python check_browser_sim.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np

import app
import config

PILOTS = ("final", "careful", "baseline")
LANDINGS = range(config.TEST_SEED, config.TEST_SEED + 100)   # the official test landings
SCENARIOS = ("normal", "strong_wind", "heavy_gravity", "weak_engine", "harsh")
OUT = Path("runs") / "browser_check.json"


def sliders(scenario):
    c = config.TEST_SCENARIOS[scenario]
    return dict(gravity=-c.get("gravity", -10.0), wind=c.get("wind_power", 0.0),
                turbulence=c.get("turbulence_power", 0.0), engine=100 * c.get("engine_power", 1.0))


def same_until(a, b):
    """Seconds until the two flights' positions differ by more than 1 cm, or None if they never do."""
    n = min(len(a["x"]), len(b["x"]))
    for i in range(n):
        if abs(a["x"][i] - b["x"][i]) > 0.01 or abs(a["y"][i] - b["y"][i]) > 0.01:
            return i / 50
    return None if len(a["x"]) == len(b["x"]) else n / 50


def main():
    report = dict(landings=f"{LANDINGS.start}-{LANDINGS.stop - 1}", cells=[])
    agree_all, total, drift = 0, 0, []
    for scenario in SCENARIOS:
        s = sliders(scenario)
        jobs = [dict(pilot=p, sliders=s, landing=l) for p in PILOTS for l in LANDINGS]
        js = json.loads(subprocess.run(["node", "check_browser_sim.mjs"], input=json.dumps(jobs),
                                       capture_output=True, text=True, check=True).stdout)
        k = 0
        for p in PILOTS:
            py_out, js_out = [], []
            for landing in LANDINGS:
                py = app.record_flight(p, app.to_conditions(s), landing)
                jf = js["flights"][k]
                k += 1
                py_out.append(py["outcome"])
                js_out.append(jf["outcome"])
                drift.append(same_until(dict(x=py["frames"]["x"], y=py["frames"]["y"]), jf))
            agree = sum(a == b for a, b in zip(py_out, js_out))
            agree_all += agree
            total += len(py_out)
            cell = dict(scenario=scenario, pilot=p, same_outcome=agree / len(py_out),
                        python_landed=py_out.count("landed") / len(py_out),
                        browser_landed=js_out.count("landed") / len(js_out))
            report["cells"].append(cell)
            print(f"{scenario:14s} {p:9s} same outcome {cell['same_outcome']:6.1%}   landed: "
                  f"python {cell['python_landed']:6.1%}  browser {cell['browser_landed']:6.1%}")
        print(f"  (browser: {js['ms_per_flight']:.1f} ms per flight in Node)")
    report["same_outcome"] = agree_all / total
    report["identical_whole_flight"] = float(np.mean([d is None for d in drift]))
    report["median_seconds_before_drift"] = float(np.median([d for d in drift if d is not None]))
    gaps = [abs(c["python_landed"] - c["browser_landed"]) for c in report["cells"]]
    report["max_rate_gap"] = max(gaps)
    report["mean_rate_gap"] = float(np.mean(gaps))
    print(f"\nSame outcome on {report['same_outcome']:.1%} of {total} landings. Landing rates differ by "
          f"{report['mean_rate_gap']:.1%} on average (at most {report['max_rate_gap']:.1%}).")
    print(f"{report['identical_whole_flight']:.1%} of flights are identical (within 1 cm) from start to finish; "
          f"the others drift apart after a median of {report['median_seconds_before_drift']:.1f} s.")
    OUT.write_text(json.dumps(report, indent=1))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
