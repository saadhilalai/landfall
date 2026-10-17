"""
make_space.py — packs the LANDFALL website into a folder that runs as a free
Hugging Face Space, and (optionally) uploads it.

The site needs no server: every landing is flown in the visitor's browser (web/sim), so
the Space is a free "static" Space, just files. It holds the web page, the physics and the
four pilots' networks (web/data/pilots.json, from export_browser_pilots.py), and the four
video clips on the page. Training logs, models and test data stay out.

Usage:
    python make_space.py                          # build build/hf_space/ only
    python make_space.py --push USERNAME/landfall # build, then upload (run `hf auth login` first)
"""

import argparse
import shutil
from pathlib import Path

OUT = Path("build") / "hf_space"
CLIPS = ["side_by_side_heavy_gravity.mp4", "untrained_crash.mp4", "baseline_landing.mp4", "final_strong_wind_arrow.mp4"]

SPACE_CARD = """\
---
title: LANDFALL
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: A rocket lander AI that taught itself, flying live
---

# LANDFALL

A reinforcement learning agent that taught itself to land a rocket in randomized gravity,
wind, turbulence and engine power. Set the conditions, press Launch, and your browser flies
the landing live with the trained network; the page draws it with webcast-style telemetry.

Designed & built by Saad Hilal · S4D.

Simulator: Gymnasium LunarLander-v3, rewritten in JavaScript on planck.js (MIT, see
vendor/planck-LICENSE.txt). Learning: PPO (Stable-Baselines3). Tuning: Optuna.
Typeface: D-DIN by Datto (SIL Open Font License, see fonts/OFL.txt).
"""


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree("web", OUT, ignore=shutil.ignore_patterns(".DS_Store"))
    (OUT / "videos").mkdir()
    for f in CLIPS:
        shutil.copy(Path("videos") / f, OUT / "videos" / f)
    (OUT / "README.md").write_text(SPACE_CARD)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"Built {OUT}/ ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")


def push(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy LANDFALL", delete_patterns="*")
    user, name = repo_id.split("/")
    print(f"Uploaded. Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"Live site: https://{user.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", metavar="USERNAME/SPACE", help="upload to this Hugging Face Space")
    args = parser.parse_args()
    build()
    if args.push:
        push(args.push)
