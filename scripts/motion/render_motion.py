#!/usr/bin/env python3
"""Render a clip as a filmstrip of stick-figure frames (input for the Qwen3-VL captioner, scripts/captioning/qwen_caption.py).

  python3 scripts/motion/render_motion.py Backflip out.png          # one clip by name (data/clips_report.csv)

Each of the 8 panels follows the character (hips x/z fixed, height kept), seen from the front-right at 15 degrees up,
with the floor line and 0.5 m ticks that scroll with the travel, so walking, turning and jumping show up. Limbs are
coloured by side where the rig has labelled roles (left blue, right orange), spine and the rest grey.
No mesh and no GPU: the clip GLBs hold the skeleton only, and a few PIL lines per frame are fast (~20 ms a clip).
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]

PANEL = (256, 256)
COLS, ROWS = 4, 2
AZ, EL = math.radians(35), math.radians(15)
BG, FLOOR, TICK = (20, 22, 26), (90, 98, 120), (60, 66, 80)
LEFT, RIGHT, MID = (110, 168, 255), (255, 159, 90), (200, 204, 212)


def side_colours(sk, roles):
    """joint index -> colour: left-limb joints blue, right-limb orange, everything else grey."""
    col = [MID] * len(sk["names"])
    if roles:
        par = sk["parents"]
        for key, c in (("leg_L", LEFT), ("arm_L", LEFT), ("leg_R", RIGHT), ("arm_R", RIGHT)):
            ch = roles.get(key)
            if not ch:
                continue
            for j in range(len(col)):                        # the chain, and everything below its end (foot, hand, digits)
                a = j
                while a >= 0 and a != ch[0]:
                    a = par[a]
                if a == ch[0]:
                    col[j] = c
    return col


def project(P):
    """World (..., 3) -> screen (x right, y up) for the fixed camera: turned by AZ about the vertical, tilted down by EL."""
    x = P[..., 0] * math.cos(AZ) - P[..., 2] * math.sin(AZ)
    depth = P[..., 0] * math.sin(AZ) + P[..., 2] * math.cos(AZ)
    return np.stack([x, P[..., 1] * math.cos(EL) + depth * math.sin(EL)], -1)


def render(sk, n_frames=8, panel=PANEL, label=True):
    Wp, fps = sk["Wp"], sk["fps"]
    T = len(Wp)
    idx = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    while len(idx) < n_frames:                                    # very short clip: repeat the last frame
        idx = np.append(idx, idx[-1])
    hips = Wp[:, sk.get("hips", 0)]
    rel = Wp - np.stack([hips[:, 0], np.zeros(T), hips[:, 2]], -1)[:, None]      # follow x/z, keep height
    xy = project(rel[idx])
    half = max(float(np.abs(xy[..., 0]).max()), 0.3) * 1.15
    top = float(xy[..., 1].max())
    bot = min(float(xy[..., 1].min()), 0.0)
    span = max(2 * half, (top - bot) * 1.1, 0.6)
    scale = (panel[0] - 16) / span
    cx = panel[0] / 2
    cols = sk.get("colours") or [MID] * len(sk["names"])
    par = sk["parents"]
    img = Image.new("RGB", (panel[0] * COLS, panel[1] * ROWS), BG)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        font = ImageFont.load_default()
    for k, t in enumerate(idx):
        ox, oy = (k % COLS) * panel[0], (k // COLS) * panel[1]
        d.rectangle([ox, oy, ox + panel[0] - 1, oy + panel[1] - 1], outline=(40, 44, 52))
        floor_y = oy + panel[1] - 22 + bot * scale                 # screen row of y = 0
        d.line([ox + 4, floor_y, ox + panel[0] - 4, floor_y], fill=FLOOR, width=1)
        for m in range(-40, 41):                                   # 0.5 m ticks along world Z; they slide as the character travels
            tick = np.array([-hips[t, 0], 0.0, m * 0.5 - hips[t, 2]])
            px = cx + project(tick)[0] * scale
            if 4 < px < panel[0] - 4:
                d.line([ox + px, floor_y, ox + px, floor_y + 5], fill=TICK, width=1)
        pts = project(rel[t])
        sx, sy = ox + cx + pts[:, 0] * scale, floor_y - pts[:, 1] * scale
        for j, p in enumerate(par):
            if p >= 0:
                d.line([sx[p], sy[p], sx[j], sy[j]], fill=cols[j], width=3)
        for j in range(len(par)):
            d.ellipse([sx[j] - 2, sy[j] - 2, sx[j] + 2, sy[j] + 2], fill=cols[j])
        if label:
            d.text((ox + 6, oy + 4), f"{t / fps:.1f}s", fill=(160, 168, 184), font=font)
    return img


def load_for_render(clip):
    """Skeleton + clip FK for a clip name, with side colours from the dataset's joint labels (grey when unlabelled)."""
    import retarget as R
    rep, inv = R.clip_index()
    sk = R.load_clip(R.ROOT / rep[clip] if not Path(rep[clip]).is_absolute() else Path(rep[clip]))
    roles = None
    try:
        r = inv[clip]
        roles = R.label_roles(sk, R.clean_names(r["dataset"], r["object_type"]))
        sk["hips"] = roles["hips"]
    except Exception:                                              # Skip or anything odd: draw uncoloured
        pass
    sk["colours"] = side_colours(sk, roles)
    sk["roles"] = roles
    return sk


if __name__ == "__main__":
    sk = load_for_render(sys.argv[1])
    render(sk).save(sys.argv[2] if len(sys.argv) > 2 else "clip.png")
