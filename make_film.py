"""
make_film.py — the launch film at the top of the LANDFALL website, cut for the web from
brag-output/brag.mp4 (the 23-second trailer, 1920x1080).

    web/media/film-1080p.mp4        computers: full HD, about 2.1 MB
    web/media/film-720p.mp4         phones: 1280x720, about 1.2 MB, so the page loads fast
    web/media/film-poster-1080.jpg  the first frame, shown while the film loads
    web/media/film-poster-720.jpg

Both cuts keep the music (AAC) because the page has an unmute button. "faststart" puts the index
at the front of the file, so playback starts before the whole file has downloaded.

Usage:
    python make_film.py
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg

SRC = Path("brag-output")
OUT = Path("web") / "media"
CUTS = [   # width, quality (lower CRF = sharper, bigger), audio bitrate
    (1920, 24, "128k"),
    (1280, 27, "96k"),
]


def ffmpeg(*args):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", *args], check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for width, crf, audio in CUTS:
        height = width * 9 // 16
        film = OUT / f"film-{height}p.mp4"
        ffmpeg("-i", str(SRC / "brag.mp4"), "-vf", f"scale={width}:-2:flags=lanczos",
               "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
               "-g", "60", "-c:a", "aac", "-b:a", audio, "-movflags", "+faststart", str(film))
        poster = OUT / f"film-poster-{height}.jpg"   # brag.jpg is the trailer's first frame
        ffmpeg("-i", str(SRC / "brag.jpg"), "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "4", str(poster))
        print(f"  {film}  {film.stat().st_size / 1e6:.1f} MB   {poster}  {poster.stat().st_size / 1e3:.0f} KB")
