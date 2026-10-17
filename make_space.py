"""
make_space.py — packs the LANDFALL website into a folder that runs as a free
Hugging Face Space, and (optionally) uploads it.

The Space is a "Docker" Space on the free CPU hardware. It only needs the code that
flies a landing, the three trained pilots it shows, the web page, and the four video
clips on the page. Training logs, the other models and the test data stay out.

Usage:
    python make_space.py                          # build build/hf_space/ only
    python make_space.py --push USERNAME/landfall # build, then upload (run `hf auth login` first)
"""

import argparse
import shutil
from pathlib import Path

OUT = Path("build") / "hf_space"
CODE = ["app.py", "config.py", "envs.py", "domain_randomization.py", "landing_stats.py", "reward_shaping.py", "render.py"]
MODELS = ["baseline_seed42.zip", "randomized_2m_seed42.zip", "randomized_memory_tuned_seed42.zip"]
CLIPS = ["side_by_side_heavy_gravity.mp4", "untrained_crash.mp4", "baseline_landing.mp4", "final_strong_wind_arrow.mp4"]

# Same versions as requirements.txt, but the small CPU-only build of PyTorch (no graphics-card code).
REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.1+cpu
gymnasium[box2d]==1.4.0
Box2D==2.3.10
stable-baselines3==2.9.0
numpy==2.4.6
pillow==11.3.0
fastapi==0.143.2
uvicorn==0.54.0
"""

DOCKERFILE = """\
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
EXPOSE 7860
CMD ["python", "app.py"]
"""

SPACE_CARD = """\
---
title: LANDFALL
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: A rocket lander AI that taught itself, flying live
---

# LANDFALL

A reinforcement learning agent that taught itself to land a rocket in randomized gravity,
wind, turbulence and engine power. Set the conditions, press Launch, and the server flies
the landing live with the trained network; the page draws it with webcast-style telemetry.

Simulator: Gymnasium LunarLander-v3. Learning: PPO (Stable-Baselines3). Tuning: Optuna.
Typeface: D-DIN by Datto (SIL Open Font License, see web/fonts/OFL.txt).
"""


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    (OUT / "videos").mkdir()
    for f in CODE:
        shutil.copy(f, OUT / f)
    for f in MODELS:
        shutil.copy(Path("models") / f, OUT / "models" / f)
    for f in CLIPS:
        shutil.copy(Path("videos") / f, OUT / "videos" / f)
    shutil.copytree("web", OUT / "web")
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / "Dockerfile").write_text(DOCKERFILE)
    (OUT / "README.md").write_text(SPACE_CARD)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Built {OUT}/ ({sum(1 for p in OUT.rglob('*') if p.is_file())} files, {size / 1e6:.1f} MB)")


def push(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy LANDFALL")
    user, name = repo_id.split("/")
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{repo_id}")
    print(f"Live site (after the build, ~5 min): https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", metavar="USERNAME/SPACE", help="upload to this Hugging Face Space")
    args = parser.parse_args()
    build()
    if args.push:
        push(args.push)
