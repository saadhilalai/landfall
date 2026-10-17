"""
render.py — draws the game sharply at any size, plus the on-screen wind arrow and labels.

The simulator itself only draws at 600x400, and its physics is tied to that drawing
scale, so we can't just turn the resolution up. Instead, render_hires() redraws the
same physics objects (terrain, lander, legs, flags, engine flames) as vector shapes
at k times the size: real detail, not blurry upscaling, and the same look and colors
as the game. Used by make_videos.py (promo videos) and app.py (the live demo).
"""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # draw off-screen, no window (needed on servers)

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from gymnasium.envs.box2d import lunar_lander as ll  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pygame import gfxdraw  # noqa: E402

FONTS = ("/System/Library/Fonts/HelveticaNeue.ttc",                 # macOS
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # most Linux servers
         "DejaVuSans.ttf")
_font_cache = {}


def font(size):
    """Helvetica Neue on a Mac, DejaVu Sans on Linux, Pillow's built-in font otherwise."""
    if size not in _font_cache:
        for path in FONTS:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default(size)
    return _font_cache[size]


def render_hires(game, k):
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

    # Box2D's y axis points up, the image's points down: read the rows out upside down.
    rgb = pygame.image.tobytes(surf, "RGB", True)
    return np.frombuffer(rgb, np.uint8).reshape(surf.get_height(), surf.get_width(), 3).copy()


def current_wind(game):
    """The sideways wind force the game applied on the last step (0 when landed or wind off)."""
    if not game.enable_wind or game.legs[0].ground_contact or game.legs[1].ground_contact:
        return 0.0
    i = game.wind_idx - 1
    return math.tanh(math.sin(0.02 * i) + math.sin(math.pi * 0.01 * i)) * game.wind_power


def draw_wind_arrow(frame, wind, full_scale, scale=None):
    """White arrow at the top centre: direction and strength of the wind right now.
    The arrow is full length when |wind| == full_scale. Sized for a 1920-wide frame and
    scaled to the frame's width, unless `scale` is given."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    s = img.width / 1920 if scale is None else scale
    cx, cy = img.width // 2, round(120 * s)
    d.text((cx, cy - round(70 * s)), "WIND", fill=(255, 255, 255), font=font(round(40 * s)), anchor="mm")
    length = 260 * s * wind / max(full_scale, 1e-6)
    if abs(length) > 8 * s:
        tip = cx + length
        d.line([(cx - length * 0.0, cy), (tip, cy)], fill=(255, 255, 255), width=max(2, round(8 * s)))
        head = (26 if length > 0 else -26) * s
        d.polygon([(tip + head, cy), (tip - head * 0.2, cy - 20 * s), (tip - head * 0.2, cy + 20 * s)],
                  fill=(255, 255, 255))
    else:
        r = 6 * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
    return np.array(img)


def label(frame, text, size=48, xy=(40, 36)):
    """White text in the black sky (sizes and positions in pixels of this frame)."""
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text(xy, text, fill=(255, 255, 255), font=font(size))
    return np.array(img)
