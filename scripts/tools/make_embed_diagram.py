#!/usr/bin/env python3
"""Draw docs/embed_pipeline.svg (the diagram at the top of docs/EMBED.md).

  python3 scripts/tools/make_embed_diagram.py

Plain SVG with fixed light boxes and dark text on a white panel, so it reads the same on a light or a dark page. Edit the
numbers here (they mirror docs/EMBED.md) and rerun.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[2] / "docs" / "embed_pipeline.svg"
W, H = 1260, 800
STYLE = {  # fill, stroke
    "data": ("#e8f0fb", "#4a78b8"), "feat": ("#e3f4f1", "#2f8f83"), "model": ("#fdeede", "#d9822b"),
    "out": ("#e6f5e9", "#3a9a54"), "frozen": ("#efe8fa", "#7a56b8"), "todo": ("#f4f4f4", "#8a8a8a"),
}
parts = []


def box(x, y, w, kind, title, lines, dashed=False, sub=None):
    """Draws a box sized to its text; returns its height."""
    h = 41 + (17 if sub else 0) + 17 * len(lines) + 10
    fill, stroke = STYLE[kind]
    d = ' stroke-dasharray="6 4"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.8"{d}/>')
    parts.append(f'<text x="{x + 12}" y="{y + 22}" font-size="14.5" font-weight="700" fill="#1d2430">{escape(title)}</text>')
    yy = y + 41
    if sub:
        parts.append(f'<text x="{x + 12}" y="{yy}" font-size="11.5" fill="#5b6575" font-family="ui-monospace,Menlo,monospace">{escape(sub)}</text>')
        yy += 17
    for ln in lines:
        parts.append(f'<text x="{x + 12}" y="{yy}" font-size="12.5" fill="#2b3442">{escape(ln)}</text>')
        yy += 17
    return h


def arrow(pts, label=None, lx=None, ly=None, dashed=False, color="#4b5563"):
    d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#ah)"/>')
    if label:
        parts.append(f'<text x="{lx}" y="{ly}" font-size="11.5" fill="#4b5563" font-style="italic">{escape(label)}</text>')


def line(pts, color="#4b5563", dashed=False):
    d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')


parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
parts.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563"/></marker></defs>')
parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
parts.append(f'<rect x="4" y="4" width="{W - 8}" height="{H - 8}" rx="12" fill="#ffffff" stroke="#d5d9e0"/>')
parts.append('<text x="28" y="38" font-size="20" font-weight="700" fill="#1d2430">Caption and motion embeddings: from clips and captions to training pairs</text>')
parts.append('<text x="28" y="60" font-size="13" fill="#5b6575">Both motion models are trained to land near the frozen CLIP text embedding of the clip\'s caption. Numbers are held-out results (rigs never seen in training); chance in brackets.</text>')

# lane headers
for x, w, t in ((30, 250, "DATA"), (340, 290, "FEATURES"), (700, 290, "MODELS"), (1050, 190, "OUTPUTS")):
    parts.append(f'<text x="{x}" y="92" font-size="12" font-weight="700" letter-spacing="1.5" fill="#7b8494">{t}</text>')
    parts.append(f'<line x1="{x}" y1="98" x2="{x + w}" y2="98" stroke="#d5d9e0"/>')

# ---- data (a column of boxes, 16 px apart)
y = 110
h1 = box(30, y, 250, "data", "UniML3D clips", ["12,638 clips, raw skeleton NPZ", "biped, quadruped, insect, bird,", "fish, snake, rigid"]); d1 = (y, y + h1); y += h1 + 16
h2 = box(30, y, 250, "data", "Template rigs (free)", ["65 clips: spider 9, fox 13, horse 13,", "kaiju 9, snake 7, fish 6, bird 4, dragon 4"]); d2 = (y, y + h2); y += h2 + 16
h3 = box(30, y, 250, "data", "Captions", ["3 styles per clip (dataset, 17,384", "distinct); template captions are written", "from clip names, not checked"]); d3 = (y, y + h3); y += h3 + 34
h4 = box(30, y, 250, "todo", "Truebones ZOO: not added", ["paid pack (about $99+), 1,097 clips", "on 74 animals; not redistributable"], dashed=True); y += h4 + 16
h5 = box(30, y, 250, "todo", "Qwen3-VL-2B captions: tried", ["from stick-figure renders: R@5 8-17%", "(chance 8%), so not used"], dashed=True); d_end = y + h5

# ---- features
y = 110
f2h = box(340, y, 290, "feat", "Role features", ["root velocity, yaw rate, hips height and tilt,", "head, knee/elbow and foot/hand of 4 limbs,", "all relative to the hips, divided by hips height", "6,055 clips (biped and quadruped only)"], sub="motion_features.py"); f2 = (y, y + f2h); y += f2h + 18
f1h = box(340, y, 290, "feat", "Joint-set features", ["each joint is a token: height, distance from", "centroid, vertical and horizontal speed over", "128 frames, plus rest-pose structure (depth,", "leaf, subtree size). No names, no joint order", "12,703 clips, any body"], sub="motion_sem_features.py"); f1 = (y, y + f1h); y += f1h + 18
f3h = box(340, y, 290, "frozen", "CLIP ViT-B/32 text encoder", ["frozen; every caption becomes 512 numbers,", "centred by a fixed mean, so the shared", '"An object ..." wording drops out', "17,413 captions"], sub="embed.py, template_clips.py"); f3 = (y, y + f3h)

# ---- models
y = 110
m2h = box(700, y, 290, "model", "Role model (variational autoencoder)", ["transformer encoder gives z (256) and", "sem (512); decoder rebuilds the role features", "from z. Held-out text to motion: median rank", "11 of 555 [278], R@5 36.4%. Poses rebuild well,", "root velocity and turning barely"], sub="motion_ae.py"); m2 = (y, y + m2h); y += m2h + 18
m1h = box(700, y, 290, "model", "Semantic model (any body)", ["small conv per joint, then a transformer over", "the SET of joints and a whole-body token (2.6M", "parameters) gives sem (512). Bodies sampled", "evenly. Held-out: median rank 36 of 1,262", "[631], R@5 18.5%. A body never seen transfers", "only weakly"], sub="motion_sem.py"); m1 = (y, y + m1h)

# ---- outputs
y = 110
o2h = box(1050, y, 190, "out", "motion.npz", ["6,055 clips: caption,", "CLIP text embedding,", "z (256), sem (512)", "-> train your diffusion", "model (prompt to z)"]); o2 = (y, y + o2h); y += o2h + 16
o3h = box(1050, y, 190, "out", "Onto a character", ["decode z -> IK on a human", "rig -> retarget.py onto any", "biped or quadruped"]); o3 = (y, y + o3h); y += o3h + 16
y = max(y, m1[0] + 70)
o1h = box(1050, y, 190, "out", "motion_sem.npz", ["12,703 clips: sem (512)", "-> find motions by text,", "same word on 2, 4, 8 legs"]); o1 = (y, y + o1h)
y += o1h + 16
o4h = box(1050, y, 190, "out", "Diffusion test (quick)", ["prompt -> z, held-out:", "R@5 33.5% (direct 36.4%).", "Speed-aware sampling: bad", "walks 22% -> 10% (real 7%)"]); o4 = (y, y + o4h)

mid = lambda r: (r[0] + r[1]) // 2
# ---- arrows: data -> features
arrow([(280, 140), (340, 140)])                                      # UniML3D -> role features
bus_x, to_y = 310, f1[0] + 36
line([(280, d1[0] + 70), (bus_x, d1[0] + 70), (bus_x, to_y)])        # UniML3D -> joint-set (bus)
line([(280, mid(d2)), (bus_x, mid(d2))])                             # templates join the bus
arrow([(bus_x, to_y), (340, to_y)])
arrow([(280, mid(d3)), (320, mid(d3)), (320, f3[0] + 34), (340, f3[0] + 34)])      # captions -> CLIP
# ---- features -> models
arrow([(630, 160), (700, 160)])                                      # role features -> VAE
yj = mid(f1)
arrow([(630, yj), (700, yj)])                                        # joint-set -> semantic
# text embedding to both models (the trunk is broken where it passes the joint-set arrow)
tx, ty_vae, ty_sem = 668, m2[0] + 130, m1[0] + 110
line([(630, mid(f3)), (tx, mid(f3)), (tx, yj + 14)])
line([(tx, yj - 14), (tx, ty_vae)])
arrow([(tx, ty_vae), (700, ty_vae)])
arrow([(tx, ty_sem), (700, ty_sem)])
parts.append(f'<circle cx="{tx}" cy="{ty_sem}" r="3" fill="#4b5563"/>')
parts.append(f'<text x="636" y="{f3[1] + 16}" font-size="11.5" fill="#6a3fb0" font-style="italic">InfoNCE: each motion is pulled</text>')
parts.append(f'<text x="636" y="{f3[1] + 31}" font-size="11.5" fill="#6a3fb0" font-style="italic">toward its caption (both models)</text>')
# ---- models -> outputs
arrow([(990, 150), (1050, 150)])
arrow([(990, m2[1] - 10), (1020, m2[1] - 10), (1020, mid(o3)), (1050, mid(o3))])
arrow([(990, mid(o1)), (1050, mid(o1))])

# ---- legend and notes
ly = max(d_end, f3[1], m1[1], o4[1]) + 40
parts.append(f'<line x1="30" y1="{ly - 22}" x2="{W - 30}" y2="{ly - 22}" stroke="#d5d9e0"/>')
x = 30
for kind, label in (("data", "data"), ("feat", "features"), ("frozen", "frozen pretrained model"), ("model", "trained here"), ("out", "saved output"), ("todo", "not built / not used")):
    fill, stroke = STYLE[kind]
    d = ' stroke-dasharray="4 3"' if kind == "todo" else ""
    parts.append(f'<rect x="{x}" y="{ly - 10}" width="22" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>')
    parts.append(f'<text x="{x + 29}" y="{ly + 2}" font-size="12" fill="#2b3442">{label}</text>')
    x += 29 + 8 * len(label) + 28
notes = [
    "Why two models: the semantic model reads any skeleton, but its output cannot be turned back into motion. The role model can be decoded and put on a character (IK + retargeter; 79% of limb movement kept), but only for bipeds and quadrupeds (it needs labelled hips, head, legs, arms).",
    "Transfer to a body never seen: without insects in training the model finds held-out insect motions at about chance (median rank 11 of 26 [13]; with insects: 3). Without the spider it finds common actions among the spider's own 9 clips (walk, jump, attack, idle) but rarer ones not, and among all insect-type clips only weakly (median rank 55 of 165 [82]). Each body type needs example clips.",
]
import textwrap
k = 0
for t in notes:
    for ln in textwrap.wrap(t, 175):
        parts.append(f'<text x="30" y="{ly + 30 + k * 18}" font-size="12" fill="#4b5563">{escape(ln)}</text>')
        k += 1
    k += 0
H_NEEDED = ly + 30 + k * 18 + 14
parts[0] = parts[0].replace(f'height="{H}" viewBox="0 0 {W} {H}"', f'height="{H_NEEDED}" viewBox="0 0 {W} {H_NEEDED}"')
parts[2] = parts[2].replace(f'height="{H}"', f'height="{H_NEEDED}"')
parts[3] = parts[3].replace(f'height="{H - 8}"', f'height="{H_NEEDED - 8}"')
parts.append("</svg>")
OUT.write_text("\n".join(parts) + "\n")
print("wrote", OUT)
