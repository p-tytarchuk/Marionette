"""Synthetic role-labelled 2D skeletons (skeleton-roles/1) of the four classes plus "other": random proportions, rest poses, joint counts and yaws, deterministic from a seed.

  python scripts/rigging/skeleton_synth.py write --n 2000 --seed 0     # data/synthetic/skeletons/skeletons.jsonl (one spec per line) + summary.json
  python scripts/rigging/skeleton_synth.py sheet [--seed 0]            # a contact sheet of one body per variant at a few yaws -> data/synthetic/skeletons/sheet.png
  python scripts/rigging/skeleton_synth.py realism                     # does the geometry look like the real skeletons? train on synthetic / test on real, and the reverse
  python scripts/rigging/skeleton_synth.py --self-test                 # every variant validates, derives its class, is deterministic, and keeps its class and carriers at every yaw

Why: the format (docs/SKELETON_SPEC.md) needs inputs that are plentiful, labelled by construction and include the awkward cases (a centaur, an angel, an upright cobra, a fish, a bear whose far legs
have no artwork). Each body is built in 3D (x lateral, y up, z forward), then drawn at a yaw: x' = z cos f + x sin f, so f = 0 is the profile of a body facing right. Which limb is near or far, which has
no artwork in the drawing (visible=false, only near profile, only for far-side limbs) and the drawing order all follow from that geometry; the class and the carriers never depend on it.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
import skeleton_spec as ss                                                  # noqa: E402
OUT = ROOT / "data" / "synthetic" / "skeletons"

# class -> {variant: weight}
VARIANTS = {
    "biped": {"humanoid": 0.30, "detailed_humanoid": 0.25, "chibi": 0.10, "tailed_humanoid": 0.10, "long_armed": 0.05, "theropod": 0.20},
    "quadruped": {"dog": 0.35, "stocky": 0.25, "long_neck": 0.15, "short_legged": 0.15, "centaur": 0.10},
    "limbless": {"snake_lying": 0.30, "snake_coiled": 0.10, "cobra_raised": 0.10, "fish": 0.25, "eel": 0.10, "worm": 0.15},
    "winged": {"bird": 0.40, "bat": 0.15, "dragon": 0.20, "angel": 0.10, "insect_winged": 0.10, "penguin": 0.05},
    "other": {"spider": 0.30, "insect": 0.25, "tripod": 0.10, "pogo": 0.05, "armed_blob": 0.15, "octopus": 0.15},
}
CLASS_OF = {v: c for c, vs in VARIANTS.items() for v in vs}
Y_AXIS = np.array([0.0, 1.0, 0.0])


# ----------------------------------------------------------------------------- 3D body
class Body:
    def __init__(self):
        self.parts, self.joints = [], []

    def part(self, kind, pid, attach, pts, **meta):
        """a chain of joints (3D points) hanging from the joint `attach` (None for the root part); returns the joint ids"""
        self.parts.append(dict(id=pid, kind=kind, **meta))
        ids, prev = [], attach
        for k, p in enumerate(pts):
            jid = f"{pid}{k + 1}"
            self.joints.append(dict(id=jid, parent=prev, P=np.asarray(p, float), part=pid))
            ids.append(jid)
            prev = jid
        return ids

    def pos(self, jid):
        return next(j["P"] for j in self.joints if j["id"] == jid)


def walk(start, steps):
    """points starting at `start`, each next point = previous + (a step vector)"""
    pts = [np.asarray(start, float)]
    for s in steps:
        pts.append(pts[-1] + np.asarray(s, float))
    return pts


def curve(rng, start, direction, n, length, wobble=0.15, up_bias=0.0):
    """n points along a gently bending path from `start` heading `direction` (3D unit-ish), total length `length`"""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    pts, step = [np.asarray(start, float)], length / max(n - 1, 1)
    for _ in range(n - 1):
        d = d + rng.normal(0, wobble, 3) + np.array([0, up_bias, 0])
        d /= np.linalg.norm(d)
        pts.append(pts[-1] + d * step)
    return pts


def leg_points(rng, start, n, style, forward=1.0):
    """a leg from `start` down to the ground (y = 0): n joints zigzagging in the sagittal (y, z) plane, then scaled so that the last joint stands on the floor"""
    zig = {"plantigrade": 0.07, "digitigrade": 0.20, "straight": 0.03}[style]
    steps, sign = [], 1
    for k in range(n - 1):
        steps.append(np.array([rng.normal(0, 0.01), -1.0, sign * zig * forward * (0.7 + 0.6 * rng.random())]))
        sign = -sign
    pts = walk(start, steps)
    drop = pts[0][1] - pts[-1][1]
    scale = pts[0][1] / drop if drop > 1e-9 else 1.0
    return [pts[0] + (p - pts[0]) * np.array([1, scale, 1]) for p in pts]


def limb_points(rng, start, n, length, direction):
    return curve(rng, start, direction, n, length, wobble=0.08)


def with_midpoints(pts):
    """a helper (twist) joint halfway along every bone: real 3D rigs have many, and a limb stays a chain"""
    out = [pts[0]]
    for a, c in zip(pts, pts[1:]):
        out += [(a + c) / 2, c]
    return out


def swing(rng, pts, deg):
    """rest-pose variation: rotate a limb about its first joint in the sagittal plane (the pose of frame 0 is not always the neutral stance)"""
    a = math.radians(rng.uniform(-deg, deg))
    c, s = math.cos(a), math.sin(a)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return [pts[0] + R @ (p - pts[0]) for p in pts]


# ----------------------------------------------------------------------------- variants (each returns a Body; the intended class is CLASS_OF[variant])
def humanoid(rng, variant):
    b = Body()
    H = 1.0
    chibi = variant == "chibi"
    leg_len, torso = (rng.uniform(0.28, 0.38) if chibi else rng.uniform(0.42, 0.55)), (rng.uniform(0.25, 0.32) if chibi else rng.uniform(0.28, 0.40))
    head_len = rng.uniform(0.30, 0.38) if chibi else rng.uniform(0.12, 0.20)
    hw, sw = rng.uniform(0.05, 0.10), rng.uniform(0.09, 0.14)
    pelvis_y = leg_len
    n_sp = int(rng.integers(2, 5))
    trunk = curve(rng, (0, pelvis_y, 0), (0, 1, rng.uniform(-0.1, 0.1)), n_sp, torso, wobble=0.04)
    t = b.part("trunk", "trunk", None, trunk)
    b.part("head", "head", t[-1], curve(rng, trunk[-1] + np.array([0, 0.02, 0]), (0, 1, 0.1), int(rng.integers(2, 4)), head_len, wobble=0.05))
    arm_len = rng.uniform(0.60, 0.80) if variant == "long_armed" else rng.uniform(0.32, 0.45)
    pose = rng.choice(["T", "A", "down"], p=[0.45, 0.40, 0.15])                 # real rigs are often in a T or A pose (the first realism check had 0.25 / 0.45 / 0.30)
    ang = {"T": 85, "A": 35, "down": 8}[pose] + rng.uniform(-6, 6)
    detailed = variant == "detailed_humanoid"
    helpers = detailed or rng.random() < 0.3
    for side, sx in (("L", 1), ("R", -1)):
        a = math.radians(ang)
        d = (sx * math.sin(a), -math.cos(a), rng.uniform(-0.1, 0.2))
        apts = limb_points(rng, trunk[-1] + np.array([sx * sw, -0.02, 0]), int(rng.integers(3, 5)), arm_len, d)
        aids = b.part("arm", f"arm_{side}", t[-1], with_midpoints(apts) if helpers else apts, side=side, girdle="pectoral")
        pts = swing(rng, leg_points(rng, (sx * hw, pelvis_y, 0), int(rng.integers(3, 5)), "straight", 1.0), 18)
        lids = b.part("leg", f"leg_{side}", t[0], with_midpoints(pts) if helpers else pts, side=side, girdle="pelvic")
        if detailed:                                                         # five fingers (3 joints each) and a toe: short twigs, as in the real rigs
            for k in range(5):
                b.part("extra", f"finger_{side}{k + 1}", aids[-1], curve(rng, apts[-1] + np.array([0, -0.005, (k - 2) * 0.01]), (sx * 0.2, -1, (k - 2) * 0.25), 3, 0.05), secondary=False)
            b.part("extra", f"toe_{side}", lids[-1], [pts[-1] + np.array([0, 0, 0.04]), pts[-1] + np.array([0, 0, 0.07])], secondary=False)
    if variant == "tailed_humanoid":
        b.part("tail", "tail", t[0], curve(rng, trunk[0] + np.array([0, -0.02, -0.03]), (0, -0.2, -1), int(rng.integers(2, 5)), rng.uniform(0.25, 0.5), up_bias=0.05), secondary=True)
    return b


def theropod(rng):
    """two legs, a horizontal trunk, a long tail, small arms: a dinosaur, an ostrich or a kangaroo (the 'bipeds' of AnyTop are of this kind, not humanoids)"""
    b = Body()
    tilt = math.radians(rng.uniform(10, 50))
    leg_h = rng.uniform(0.4, 0.7)
    trunk = curve(rng, (0, leg_h + 0.05, 0), (0, math.sin(tilt), math.cos(tilt)), int(rng.integers(3, 6)), rng.uniform(0.6, 1.0), wobble=0.03)
    t = b.part("trunk", "trunk", None, trunk)
    neck = curve(rng, trunk[-1], (0, rng.uniform(0.3, 1.0), 0.8), int(rng.integers(2, 4)), rng.uniform(0.15, 0.5), wobble=0.1)
    b.part("head", "head", t[-1], neck[1:] + [neck[-1] + np.array([0, -0.02, rng.uniform(0.06, 0.18)])])
    for side, sx in (("L", 1), ("R", -1)):
        pts = leg_points(rng, (sx * rng.uniform(0.04, 0.09), trunk[0][1] - 0.02, trunk[0][2]), int(rng.integers(3, 5)), "digitigrade", 1.0)
        b.part("leg", f"leg_{side}", t[0], swing(rng, pts, 15), side=side, girdle="pelvic")
        b.part("arm", f"arm_{side}", t[-1], limb_points(rng, trunk[-1] + np.array([sx * 0.04, -0.02, 0.02]), int(rng.integers(2, 4)), rng.uniform(0.08, 0.25), (sx * 0.3, -0.8, 0.5)), side=side, girdle="pectoral")
    b.part("tail", "tail", t[0], curve(rng, trunk[0] + np.array([0, 0, -0.02]), (0, rng.uniform(-0.2, 0.3), -1), int(rng.integers(4, 8)), rng.uniform(0.5, 1.0), wobble=0.05), secondary=True)
    return b


def quadruped(rng, variant):
    b = Body()
    long_neck, short, centaur = variant == "long_neck", variant == "short_legged", variant == "centaur"
    L = rng.uniform(0.7, 1.1) if variant != "stocky" else rng.uniform(0.6, 0.9)
    leg_h = rng.uniform(0.45, 0.8) if long_neck else (rng.uniform(0.12, 0.22) if short else rng.uniform(0.28, 0.5))
    body_y = leg_h + rng.uniform(0.04, 0.10)
    hw = rng.uniform(0.05, 0.12) * (1.5 if variant == "stocky" else 1.0)
    n_sp = int(rng.integers(3, 6))
    trunk = [np.array([0, body_y + rng.normal(0, 0.01), z]) for z in np.linspace(0, L, n_sp)]
    t = b.part("trunk", "trunk", None, trunk)
    neck_n = int(rng.integers(4, 6)) if long_neck else int(rng.integers(1, 3))
    neck_len = rng.uniform(0.5, 0.9) if long_neck else rng.uniform(0.15, 0.3)
    head_dir = (0, 1.0 if long_neck else 0.5, 1.0 if not long_neck else 0.3)
    neck = curve(rng, trunk[-1], head_dir, neck_n + 1, neck_len, wobble=0.1)
    b.part("head", "head", t[-1], neck[1:] + [neck[-1] + np.array([0, -0.03, rng.uniform(0.08, 0.2)])]) if neck_n >= 1 else None
    for side, sx in (("L", 1), ("R", -1)):
        for tag, anchor, g, z, style in (("hind", t[0], "pelvic", 0.0, "digitigrade"), ("fore", t[-1], "pectoral", L, "plantigrade")):
            pts = leg_points(rng, (sx * hw, body_y - 0.02, z), int(rng.integers(3, 5)), style, 1.0 if tag == "hind" else -1.0)
            b.part("leg", f"leg_{tag}_{side}", anchor, swing(rng, pts, 22), side=side, girdle=g)
    b.part("tail", "tail", t[0], curve(rng, trunk[0] + np.array([0, 0.01, -0.02]), (0, rng.uniform(-0.3, 0.6), -1), int(rng.integers(2, 6)), rng.uniform(0.15, 0.5)), secondary=True)
    if rng.random() < 0.5:
        hid = [j for j in b.joints if j["part"] == "head"][-1]
        b.part("extra", "ear", hid["id"], curve(rng, hid["P"] + np.array([0, 0.02, -0.02]), (0, 1, -0.2), 2, rng.uniform(0.05, 0.12)), secondary=True)
    if centaur:                                                             # a human torso on top: 4 legs + 2 arms is still a quadruped
        tt = b.part("extra", "torso_up", t[-1], curve(rng, trunk[-1] + np.array([0, 0.02, 0]), (0, 1, 0.1), 3, 0.35, wobble=0.03), secondary=False)
        top = next(j["P"] for j in b.joints if j["id"] == tt[-1])
        for side, sx in (("L", 1), ("R", -1)):
            b.part("arm", f"arm_{side}", tt[-1], limb_points(rng, top + np.array([sx * 0.1, -0.02, 0]), 3, 0.4, (sx * 0.6, -0.8, 0.1)), side=side, girdle="pectoral")
    return b


def limbless(rng, variant):
    b = Body()
    if variant == "fish":
        n, L = int(rng.integers(5, 9)), rng.uniform(0.6, 1.0)
        trunk = [np.array([0, 0.3, z]) for z in np.linspace(0, L, n)]
        t = b.part("trunk", "trunk", None, trunk)
        b.part("head", "head", t[-1], [trunk[-1] + np.array([0, 0, 0.1])])
        b.part("tail", "tail", t[0], curve(rng, trunk[0], (0, rng.uniform(-0.2, 0.4), -1), int(rng.integers(2, 4)), rng.uniform(0.15, 0.3)), secondary=True)
        b.part("fin", "fin_dorsal", t[n // 2], curve(rng, trunk[n // 2], (0, 1, -0.3), 2, 0.15), side="C")
        for side, sx in (("L", 1), ("R", -1)):
            b.part("fin", f"fin_pect_{side}", t[-2], curve(rng, trunk[-2], (sx * 0.8, -0.5, -0.4), 2, 0.15), side=side)
        return b
    n = {"snake_lying": int(rng.integers(10, 26)), "snake_coiled": int(rng.integers(14, 26)), "cobra_raised": int(rng.integers(12, 22)), "eel": int(rng.integers(16, 30)), "worm": int(rng.integers(5, 9))}[variant]
    L = {"worm": rng.uniform(0.3, 0.5)}.get(variant, rng.uniform(1.0, 2.0))
    s = np.linspace(0, 1, n)
    # 2D drawings show the wave and the coil in the picture plane (the plane of the body's length and height) far more often than seen from above
    plane = rng.choice([math.pi / 2, rng.uniform(0.3, 1.2), 0.0], p=[0.6, 0.25, 0.15])
    if variant == "snake_coiled":
        th = s * rng.uniform(2.5, 4.5) * math.pi
        r = 0.05 + 0.18 * (1 - s)
        if rng.random() < 0.7:
            pts = [np.array([0.0, 0.25 + r[i] * math.sin(th[i]), r[i] * math.cos(th[i])]) for i in range(n)]      # a spiral standing in the picture plane
        else:
            pts = [np.array([r[i] * math.cos(th[i]), 0.05 + 0.02 * i / n, r[i] * math.sin(th[i])]) for i in range(n)]    # a flat coil seen from above
    elif variant == "cobra_raised":
        low = int(n * 0.5)
        pts = [np.array([0.05 * math.sin(i * 0.9), 0.04, i * L / n]) for i in range(low)]
        base = pts[-1]
        pts += [base + np.array([0.02 * math.sin(i), 0.04 + (i + 1) * L * 0.5 / (n - low), 0.0]) for i in range(n - low)]
    else:
        waves = rng.uniform(1.0, 3.0)                                        # waves along the body, so a long eel is smooth, not jagged
        amp = L * rng.uniform(0.04, 0.12) * (0.5 if variant == "worm" else 1.0)
        z = s * L
        w = amp * np.sin(2 * math.pi * waves * s + rng.uniform(0, 6.28))
        pts = [np.array([w[i] * math.cos(plane), 0.06 + abs(w[i] * math.sin(plane)) + amp, z[i]]) for i in range(n)]
    if variant == "snake_lying" and rng.random() < 0.15:                      # stored upright, as one of the manifest's serpentine rigs is
        pts = [np.array([p[0], p[2], p[1]]) for p in pts]
    t = b.part("trunk", "trunk", None, pts[:-1])
    b.part("head", "head", t[-1], [pts[-1]])
    return b


def winged(rng, variant):
    b = Body()
    if variant in ("dragon", "angel"):
        base = humanoid(rng, "humanoid") if variant == "angel" else quadruped(rng, "dog")
        b.parts, b.joints = base.parts, base.joints
        t = [j["id"] for j in b.joints if j["part"] == "trunk"]
        top = b.pos(t[-1])
        s = rng.uniform(0.2, 1.0)
        for side, sx in (("L", 1), ("R", -1)):
            d = (sx * (0.3 + 0.9 * s), 0.3 + 0.7 * s, -0.5 * (1 - s))
            b.part("wing", f"wing_{side}", t[-2] if len(t) > 1 else t[-1], limb_points(rng, top + np.array([sx * 0.04, 0.0, -0.02]), int(rng.integers(3, 6)), rng.uniform(0.5, 1.4), d), side=side, girdle="pectoral")
        return b
    if variant == "insect_winged":
        n = int(rng.integers(2, 4))
        trunk = [np.array([0, 0.3, z]) for z in np.linspace(0, rng.uniform(0.3, 0.5), n)]
        t = b.part("trunk", "trunk", None, trunk)
        b.part("head", "head", t[-1], [trunk[-1] + np.array([0, 0.02, 0.06])])
        for k in range(3):
            for side, sx in (("L", 1), ("R", -1)):
                pts = leg_points(rng, (sx * 0.05, 0.28, trunk[min(k, n - 1)][2]), 3, "digitigrade", 1.0)
                b.part("leg", f"leg_{side}{k + 1}", t[min(k, n - 1)], pts, side=side, girdle=None)
        for side, sx in (("L", 1), ("R", -1)):
            b.part("wing", f"wing_{side}", t[1 if n > 1 else 0], limb_points(rng, trunk[min(1, n - 1)] + np.array([sx * 0.03, 0.03, 0]), 3, rng.uniform(0.3, 0.6), (sx * 0.8, 0.5, -0.4)), side=side, girdle="pectoral")
        return b
    # bird, bat, penguin
    pen, bat = variant == "penguin", variant == "bat"
    L = rng.uniform(0.3, 0.6)
    tilt = math.radians(rng.uniform(55, 85) if pen else rng.uniform(15, 60))
    d = np.array([0, math.sin(tilt), math.cos(tilt)])
    leg_h = rng.uniform(0.08, 0.16) if (pen or bat) else rng.uniform(0.2, 0.4)
    trunk = curve(rng, (0, leg_h + 0.1, 0), d, int(rng.integers(2, 4)), L, wobble=0.03)
    t = b.part("trunk", "trunk", None, trunk)
    neck = curve(rng, trunk[-1], (0, 1, 0.5), int(rng.integers(2, 5)), rng.uniform(0.1, 0.3), wobble=0.15)
    b.part("head", "head", t[-1], neck[1:] + [neck[-1] + np.array([0, 0, 0.06])])
    for side, sx in (("L", 1), ("R", -1)):
        pts = leg_points(rng, (sx * 0.03, trunk[0][1] - 0.02, trunk[0][2]), int(rng.integers(3, 5)), "digitigrade", 1.0)
        b.part("leg", f"leg_{side}", t[0], swing(rng, pts, 15), side=side, girdle="pelvic")
        s = rng.uniform(0.0, 0.3) if pen else rng.uniform(0.2, 1.0)
        span = rng.uniform(0.15, 0.3) if pen else (rng.uniform(0.8, 1.6) if bat else rng.uniform(0.5, 1.2))
        wd = (sx * (0.2 + 0.9 * s), -0.4 + 1.0 * s, -0.6 * (1 - s))
        b.part("wing", f"wing_{side}", t[-1], limb_points(rng, trunk[-1] + np.array([sx * 0.03, 0.0, -0.02]), int(rng.integers(3, 6)), span, wd), side=side, girdle="pectoral")
    if not pen:
        b.part("tail", "tail", t[0], curve(rng, trunk[0], (0, -0.1, -1), int(rng.integers(2, 4)), rng.uniform(0.1, 0.35)), secondary=True)
    return b


def other(rng, variant):
    b = Body()
    if variant in ("spider", "insect", "octopus"):
        n = 2 if variant != "octopus" else 1
        trunk = [np.array([0, 0.25, z]) for z in np.linspace(0, 0.2, n + 1)]
        t = b.part("trunk", "trunk", None, trunk)
        b.part("head", "head", t[-1], [trunk[-1] + np.array([0, 0.01, 0.05])])
        k_pairs = 4 if variant in ("spider", "octopus") else 3
        for k in range(k_pairs):
            for side, sx in (("L", 1), ("R", -1)):
                z = trunk[min(k * (n + 1) // k_pairs, n)][2]
                if variant == "octopus":
                    pts = curve(rng, (sx * 0.04, 0.25, z), (sx * 1.0, -0.6, 0.5 - k * 0.3), 4, rng.uniform(0.3, 0.5))
                    b.part("arm", f"arm_{side}{k + 1}", t[0], pts, side=side, girdle=None)
                else:
                    pts = leg_points(rng, (sx * 0.04, 0.24, z), 3, "digitigrade", 1.0)
                    pts = [pts[0]] + [p + np.array([sx * 0.05 * i, 0, 0]) for i, p in enumerate(pts[1:], 1)]
                    b.part("leg", f"leg_{side}{k + 1}", t[min(k * (n + 1) // k_pairs, n)], pts, side=side, girdle=None)
        return b
    if variant == "tripod":
        b = quadruped(rng, "dog")
        b.parts = [p for p in b.parts if p["id"] != "leg_fore_R"]
        b.joints = [j for j in b.joints if j["part"] != "leg_fore_R"]
        return b
    if variant == "pogo":
        b = humanoid(rng, "humanoid")
        b.parts = [p for p in b.parts if p["id"] != "leg_R"]
        b.joints = [j for j in b.joints if j["part"] != "leg_R"]
        return b
    b = humanoid(rng, "humanoid")                                            # armed_blob: arms, no legs
    b.parts = [p for p in b.parts if p["kind"] != "leg"]
    b.joints = [j for j in b.joints if not j["part"].startswith("leg_")]
    return b


BUILDERS = {"biped": lambda rng, v: theropod(rng) if v == "theropod" else humanoid(rng, v), "quadruped": quadruped, "limbless": limbless, "winged": winged, "other": other}


# ----------------------------------------------------------------------------- drawing: 3D body -> 2D spec at a yaw
def draw(body, name, yaw, rng, hide_far=0.5, meta=None):
    f = math.radians(yaw)
    c, s = math.cos(f), math.sin(f)
    pos = {}
    for j in body.joints:
        x, y, z = j["P"]
        pos[j["id"]] = (z * c + x * s, y, x * c - z * s)                      # screen x', height, depth toward the camera
    parts = []
    for p in body.parts:
        q = dict(p)
        if p["kind"] in ("leg", "arm", "wing") or (p["kind"] == "fin" and p.get("side") in ("L", "R")):
            u = float(np.mean([pos[j["id"]][2] for j in body.joints if j["part"] == p["id"]]))
            q["layer"] = "front" if u > 0.0 else "behind"
            if abs(c) > 0.8 and u < -1e-3 and rng.random() < hide_far:     # near the profile the far-side limbs may have no artwork; they still exist on the body
                q["visible"] = False
        parts.append(q)
    joints = [dict(id=j["id"], parent=j["parent"], pos=[round(float(pos[j["id"]][0]), 4), round(float(pos[j["id"]][1]), 4)], part=j["part"]) for j in body.joints]
    facing = "right" if c >= 0.5 else "left" if c <= -0.5 else ("front" if s < 0 else "back")
    spec = dict(spec=ss.VERSION, id=name, view=dict(yaw_deg=round(yaw, 1), facing=facing), parts=parts, joints=joints)
    if meta:
        spec["meta"] = meta
    return spec


def body3d(variant, body_seed):
    """the 3D body of a generated spec (variant and body_seed are in its meta): the viewer re-draws it at any yaw"""
    return BUILDERS[CLASS_OF[variant]](np.random.default_rng(int(body_seed)), variant)


def generate(seed, variant=None, cls=None, yaw=None, hide_far=0.5, body_seed=None):
    """One spec. The body depends on (body_seed or seed, variant); the yaw and the hidden limbs on the seed, so the same body can be drawn at several yaws."""
    rng = np.random.default_rng(seed)
    if variant is None:
        cls = cls or rng.choice(list(VARIANTS))
        vs = VARIANTS[cls]
        variant = rng.choice(list(vs), p=np.array(list(vs.values())) / sum(vs.values()))
    cls = CLASS_OF[variant]
    brng = np.random.default_rng(seed if body_seed is None else body_seed)
    body = BUILDERS[cls](brng, variant)
    yaw = float(rng.uniform(0, 360)) if yaw is None else float(yaw)
    name = f"synth_{variant}_{seed}"
    sk = draw(body, name, yaw, np.random.default_rng([seed, 1]), hide_far, meta=dict(variant=variant, seed=int(seed), body_seed=int(seed if body_seed is None else body_seed)))
    sk["declared_class"] = cls
    return sk


# ----------------------------------------------------------------------------- commands
def write(n, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes = list(VARIANTS)
    with open(OUT / "skeletons.jsonl", "w") as fh:
        stat = collections.Counter()
        for i in range(n):
            sk = generate(int(rng.integers(0, 2 ** 31)), cls=classes[i % len(classes)])
            errs, _ = ss.validate(sk)
            assert not errs and ss.derive_class(sk)[0] == sk["declared_class"], (sk["id"], errs)
            fh.write(json.dumps(sk) + "\n")
            stat[(sk["declared_class"], sk["meta"]["variant"])] += 1
    json.dump({f"{k[0]}/{k[1]}": v for k, v in sorted(stat.items())}, open(OUT / "summary.json", "w"), indent=1)
    print(f"{n} specs -> {OUT / 'skeletons.jsonl'}")
    for k, v in sorted(stat.items()):
        print(f"  {k[0]:10s} {k[1]:16s} {v}")


def to_tree(sk):
    """(parents as indices, 2D positions) of a spec, for the features of skeleton_classes.py"""
    idx = {j["id"]: i for i, j in enumerate(sk["joints"])}
    return [idx[j["parent"]] if j["parent"] is not None else -1 for j in sk["joints"]], np.array([j["pos"] for j in sk["joints"]], float)


def sheet(seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("biped", "quadruped"), ("limbless",), ("winged",), ("other",)]
    colors = {"trunk": "#222", "head": "#6a2", "tail": "#a63", "leg": "#25c", "arm": "#c52", "wing": "#a2c", "fin": "#2aa", "extra": "#888"}
    yaws = [0, 60, 120]
    OUT.mkdir(parents=True, exist_ok=True)
    for gi, classes in enumerate(groups):
        variants = [(c, v) for c in classes for v in VARIANTS[c]]
        fig, axs = plt.subplots(len(variants), len(yaws), figsize=(len(yaws) * 3.2, len(variants) * 2.0), squeeze=False)
        for r, (c, v) in enumerate(variants):
            for k, yaw in enumerate(yaws):
                ax = axs[r][k]
                sk = generate(seed, variant=v, yaw=yaw, hide_far=0.0)
                kind = {p["id"]: p["kind"] for p in sk["parts"]}
                pos = {j["id"]: j["pos"] for j in sk["joints"]}
                for j in sk["joints"]:
                    if j["parent"] is not None:
                        a, b = pos[j["parent"]], pos[j["id"]]
                        ax.plot([a[0], b[0]], [a[1], b[1]], "-o", color=colors[kind[j["part"]]], lw=1.4, ms=2)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(f"{v}  yaw {yaw}" if k == 0 else f"yaw {yaw}", fontsize=8, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{gi + 1}_{'_'.join(classes)}.png", dpi=80)
        plt.close(fig)
    print(f"sheets in {OUT} (trunk black, head green, tail brown, leg blue, arm orange, wing purple, fin teal, extra grey; yaws {yaws}, far limbs all drawn)")


def realism(n=3000):
    """Do synthetic skeletons look like the real ones to the structure features of skeleton_classes.py? Train on synthetic / test on real, and the reverse (balanced accuracy over the four classes)."""
    import json as _json
    import skeleton_classes as sc
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(1)
    classes = list(VARIANTS)
    syn = [generate(int(rng.integers(0, 2 ** 31)), cls=classes[i % 5]) for i in range(n)]
    Xs, ys = [], []
    for sk in syn:
        parents, Q = to_tree(sk)
        f = sc.features(sc.tree_adjacency(parents, Q), Q)
        Xs.append([f[k] for k in sc.FEATURES])
        ys.append(sc.CLASSES.index(sk["declared_class"]))
    Xs, ys = np.array(Xs, float), np.array(ys)
    skel = _json.load(open(sc.OUT / "skeletons.json"))
    man = [s for s in skel if s["source"] == "manifest"]
    tb = [s for s in skel if s["source"] == "truebones"]
    Xm, ym, _, _ = sc.table(man, sc.YAWS)
    Xt, yt, _, _ = sc.table(tb, sc.YAWS)
    mk = lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=0)

    def score(y, pred):
        cm = confusion_matrix(y, pred, labels=range(5))
        rec = [cm[k, k] / cm[k].sum() if cm[k].sum() else float("nan") for k in range(5)]
        return sc.bal4(cm), rec

    out = ["# Realism of the synthetic skeletons (structure features of skeleton_classes.py)", "",
           f"{n} synthetic specs (balanced over the 5 classes, random yaw) against {len(man)} 3D-manifest skeletons and {len(tb)} Truebones creatures (each at 6 yaws). bal4 = mean recall over biped / quadruped / limbless / winged.", "",
           "| trained on | tested on | bal4 | recall biped / quadruped / limbless / winged / other |", "|---|---|---|---|"]
    for name, (Xa, ya), tests in (("synthetic", (Xs, ys), (("3D manifest", Xm, ym), ("Truebones", Xt, yt))),
                                  ("3D manifest", (Xm, ym), (("synthetic", Xs, ys), ("Truebones", Xt, yt)))):
        tree = mk().fit(Xa, ya)
        for tname, Xb, yb in tests:
            b, rec = score(yb, tree.predict(Xb))
            out.append(f"| {name} | {tname} | {b:.3f} | " + " / ".join("-" if math.isnan(r) else f"{r:.2f}" for r in rec) + " |")
    out += ["", "Reference from docs/SKELETON_CLASSES.md: trained on the 3D manifest, bal4 0.66 in cross-validation inside it and 0.50 on Truebones.", ""]
    (OUT / "realism.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def self_test():
    seen = collections.Counter()
    for variant, cls in CLASS_OF.items():
        for seed in range(25):
            sk = generate(seed, variant=variant)
            errs, _ = ss.validate(sk)
            assert not errs, (variant, seed, errs)
            assert ss.derive_class(sk)[0] == cls, (variant, seed, ss.derive_class(sk))
            assert sk["declared_class"] == cls
            assert all(math.isfinite(v) for j in sk["joints"] for v in j["pos"]), (variant, seed)
            assert generate(seed, variant=variant) == sk, f"{variant} {seed} is not deterministic"
            seen[variant] += 1
    # the same body drawn at different yaws keeps its class and its carriers; only positions, layers and hidden limbs change
    for variant in CLASS_OF:
        ref = None
        for yaw in (0, 45, 90, 135, 180, 270):
            sk = generate(7, variant=variant, yaw=yaw, hide_far=1.0)
            c = ss.carriers(sk)
            core = {k: v for k, v in c.items() if k not in ("hidden", "trunk_axis")}
            ref = ref or core
            assert core == ref, (variant, yaw, core, ref)
            assert ss.derive_class(sk)[0] == CLASS_OF[variant]
    # far limbs are hidden only near the profile, never at the front or back, and only when asked
    assert not ss.carriers(generate(3, variant="dog", yaw=0, hide_far=0.0))["hidden"]
    assert len(ss.carriers(generate(3, variant="dog", yaw=0, hide_far=1.0))["hidden"]) == 2
    assert not ss.carriers(generate(3, variant="dog", yaw=90, hide_far=1.0))["hidden"]
    # the awkward cases are what they claim to be
    assert ss.carriers(generate(1, variant="centaur"))["arms"] == 2 and ss.derive_class(generate(1, variant="centaur"))[0] == "quadruped"
    assert ss.carriers(generate(1, variant="angel"))["legs"] == 2 and ss.derive_class(generate(1, variant="angel"))[0] == "winged"
    assert ss.carriers(generate(1, variant="fish"))["fins"] == 3 and ss.derive_class(generate(1, variant="fish"))[0] == "limbless"
    assert ss.carriers(generate(1, variant="spider"))["legs"] == 8 and ss.carriers(generate(1, variant="tripod"))["legs"] == 3
    assert ss.carriers(generate(1, variant="octopus"))["arms"] == 8 and ss.derive_class(generate(1, variant="octopus"))[0] == "other"
    assert ss.derive_class(generate(1, variant="cobra_raised"))[0] == "limbless"
    # no warnings except the unpaired limbs of the deliberately lopsided bodies; every body has a head; every paired limb says near or far
    for variant in CLASS_OF:
        sk = generate(5, variant=variant)
        errs, warns = ss.validate(sk)
        assert not errs and all(w.startswith(("leg:", "arm:")) for w in warns), (variant, warns)
        assert bool(warns) == (variant in ("tripod", "pogo")), (variant, warns)      # only these two are lopsided on purpose (3 legs, 1 leg)
        assert ss.carriers(sk)["head"], f"{variant} has no head"
        assert all(p.get("layer") in ("front", "behind") for p in sk["parts"] if p["kind"] in ("leg", "arm", "wing")), variant
    # a bird in profile with the far side hidden: one wing and one leg have no artwork, and the class is still winged
    bird = generate(4, variant="bird", yaw=0, hide_far=1.0)
    hidden = ss.carriers(bird)["hidden"]
    assert len(hidden) == 2 and any(h.startswith("wing") for h in hidden) and any(h.startswith("leg") for h in hidden), hidden
    assert ss.derive_class(bird)[0] == "winged"
    # a different seed gives a different body
    assert generate(1, variant="dog") != generate(2, variant="dog")
    # body3d reproduces the body of a generated spec: the same joints once drawn at the spec's own yaw
    sk = generate(9, variant="bird", yaw=40)
    body = body3d(sk["meta"]["variant"], sk["meta"]["body_seed"])
    f = math.radians(40)
    assert [j["id"] for j in body.joints] == [j["id"] for j in sk["joints"]]
    assert all(abs((j["P"][2] * math.cos(f) + j["P"][0] * math.sin(f)) - sj["pos"][0]) < 1e-3 for j, sj in zip(body.joints, sk["joints"]))
    print(f"skeleton_synth self-test: {len(CLASS_OF)} variants x 25 seeds valid, class correct and deterministic; yaw-invariant carriers; hidden-limb rule; awkward cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["write", "sheet", "realism"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.cmd == "write":
        write(a.n, a.seed)
    elif a.cmd == "sheet":
        sheet(a.seed)
    elif a.cmd == "realism":
        realism()
    else:
        ap.error("give a command or --self-test")
