#!/usr/bin/env python3
"""Rule-based motion retargeting: play a clip made for one body (e.g. a human backflip) on another
(e.g. a quadruped), no training involved.

  python3 scripts/rigging/retarget.py --list-targets                       # which characters can be targets
  python3 scripts/rigging/retarget.py --query backflip                     # find source clips by keyword
  python3 scripts/rigging/retarget.py --source-clip Backflip --target objaverse_<id>
  python3 scripts/rigging/retarget.py --self-test                          # identity + sanity tests

Idea (all in the normalized frame: metres, Y up, +Z forward, feet on y = 0, see normalize.py):
 1 Body.   The hips' rotation change from the rest pose, taken in world space, is applied to the target
           hips. A backflip is a rotation about the left-right axis, and that axis is the same on a
           human and on a quadruped, so the flip carries over. Translation follows the source hips
           path scaled by the hips-height ratio; the body rotates about its trunk centre (the middle of
           the four limb roots), not about the hips, so a long body does not sweep through the floor.
 2 Spine.  Source spine/neck/head (the tree path hips -> head) is resampled onto the target chain by
           normalized arc length; each target joint takes the interpolated world-space rotation change.
 3 Limbs.  Each limb segment's direction is compared with its neutral direction in the body frame
           (source: legs = rest pose, arms = hanging down; target: its own rest pose); the same swing is
           applied to the target segment. Legs drive the hind limbs, arms drive the forelimbs
           (--forelimbs arms), or each foreleg copies the opposite hind-limb leg (--forelimbs legs, a
           diagonal gait; the auto setting chooses this for travelling clips such as walks).
 4 Contacts. Source foot contacts (low and still) are copied to the target feet (hind: same side; fore:
           the opposite side, a diagonal gait). During a contact the target foot is pinned to the ground
           with analytic two-bone IK (no skating), and the body is moved up or down just enough so that no
           joint is under the ground and the pinned feet can reach.
 Every joint outside these roles (tail, ears, jaw, fingers...) is carried rigidly with its parent.

Output: data/characters/<target>/anims/rt__<source>.glb plus an entry in anims.json (kept when
convert_clips.py is rerun). The entry lists metrics: foot skate, ground penetration, IK reach
failures, and the extra lift the ground fix had to add.
"""
import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
from convert_clips import Skip, qcontinuous, qmul, qnorm, qrot, read_glb, safe_name, write_clip  # noqa: E402

SRC = ROOT / "sources" / "unimate"
CHARS = ROOT / "data" / "characters"
DOWN = np.array([0.0, -1.0, 0.0])


# ----------------------------------------------------------------------------- quaternion / fk helpers (w first)
def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qslerp(a, b, w):
    a, b = qnorm(a), qnorm(b)
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    s = np.sin(th)
    small = s < 1e-6
    wa = np.where(small, 1 - w, np.sin((1 - w) * th) / np.where(small, 1, s))
    wb = np.where(small, w, np.sin(w * th) / np.where(small, 1, s))
    return qnorm(wa * a + wb * b)


def q_swing(a, b):
    """Shortest-arc rotation taking unit vector(s) a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (a * b).sum(-1, keepdims=True)
    c = np.cross(a, b)
    q = np.concatenate([1 + d, c], -1)
    anti = (1 + d[..., 0]) < 1e-9
    if np.any(anti):  # opposite vectors: half-turn about any axis perpendicular to a
        ref = np.where(np.abs(a[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        ax = np.cross(a, ref)
        ax /= np.maximum(np.linalg.norm(ax, axis=-1, keepdims=True), 1e-12)
        q = np.where(anti[..., None], np.concatenate([np.zeros_like(d), ax], -1), q)
    return qnorm(q)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fk_pr(parents, pos, rot):
    """World positions and rotations: pos (T,J,3), rot (T,J,4) local -> Wp (T,J,3), Wr (T,J,4)."""
    T, J = pos.shape[:2]
    Wp, Wr, done = np.zeros((T, J, 3)), np.zeros((T, J, 4)), [False] * J

    def go(j):
        if done[j]:
            return
        p = parents[j]
        if p < 0:
            Wp[:, j], Wr[:, j] = pos[:, j], rot[:, j]
        else:
            go(p)
            Wp[:, j] = Wp[:, p] + qrot(Wr[:, p], pos[:, j])
            Wr[:, j] = qmul(Wr[:, p], rot[:, j])
        done[j] = True

    for j in range(J):
        go(j)
    return Wp, Wr


# ----------------------------------------------------------------------------- skeleton / clip loading
def _acc(js, b, i):
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
    return np.frombuffer(b, "<f4", a["count"] * n, bv.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, n).astype(float)


def load_skeleton(path):
    """Joints (names, parents, rest local pos/rot (w first)/scale) from a model.glb (skin joints) or a clip GLB (all nodes)."""
    js, _ = read_glb(path)
    nodes = js["nodes"]
    par = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    if js.get("skins"):
        joints = []
        for s in js["skins"]:
            for j in s["joints"]:
                if j not in joints:
                    joints.append(j)
    else:
        joints = list(range(len(nodes)))
    idx = {j: k for k, j in enumerate(joints)}
    names = [nodes[j].get("name", "") for j in joints]
    if len(set(names)) != len(names):
        raise Skip("duplicate_joint_names")
    sk = dict(names=names, parents=[idx.get(par.get(j), -1) for j in joints],
              loc=np.array([nodes[j].get("translation", [0, 0, 0]) for j in joints], float),
              rot=np.array([nodes[j].get("rotation", [0, 0, 0, 1]) for j in joints], float)[:, [3, 0, 1, 2]],
              scale=np.array([nodes[j].get("scale", [1, 1, 1]) for j in joints], float))
    sk["rot_xyzw"] = sk["rot"][:, [1, 2, 3, 0]]
    sk["Wp0"], sk["Wr0"] = (a[0] for a in fk_pr(sk["parents"], sk["loc"][None], sk["rot"][None]))
    return sk


def load_clip(path):
    """Skeleton + per-frame local rotation/translation of a clip GLB written by convert_clips.py."""
    sk = load_skeleton(path)
    js, b = read_glb(path)
    an = js["animations"][0]
    T = js["accessors"][an["samplers"][0]["input"]]["count"]
    J = len(sk["names"])
    pos = np.tile(sk["loc"][None], (T, 1, 1))
    rot = np.tile(sk["rot"][None], (T, 1, 1))
    for ch in an["channels"]:
        out = _acc(js, b, an["samplers"][ch["sampler"]]["output"])
        if ch["target"]["path"] == "translation":
            pos[:, ch["target"]["node"]] = out
        else:
            rot[:, ch["target"]["node"]] = out[:, [3, 0, 1, 2]]
    sk.update(apos=pos, arot=rot, fps=int((an.get("extras") or {}).get("fps", 30)), extras=an.get("extras") or {})
    sk["Wp"], sk["Wr"] = fk_pr(sk["parents"], pos, rot)
    return sk


_cache = {}


def clean_names(rig_ds, rig_id):
    """raw joint name -> clean label, from export/<ds>/{joint_names,clean_joint_names}.json (aligned by index)."""
    key = (rig_ds, rig_id)
    if key not in _cache:
        d = SRC / "export" / rig_ds
        jn, cn = json.loads((d / "joint_names.json").read_text()), json.loads((d / "clean_joint_names.json").read_text())
        if rig_id not in jn or rig_id not in cn or len(jn[rig_id]) != len(cn[rig_id]):
            _cache[key] = None
        else:
            m = {}
            for r, c in zip(jn[rig_id], cn[rig_id]):
                m.setdefault(r, c)
            _cache[key] = m
    return _cache[key]


# ----------------------------------------------------------------------------- roles
LEG = ["Thigh", "Shin", "Foot", "Toe"]
ARM = ["Upper Arm", "Forearm", "Hand"]


def label_roles(sk, cmap):
    """hips, spine path (hips -> head), legs and arms chains per side. Raises Skip when not usable."""
    if cmap is None:
        raise Skip("no_clean_names")
    names, par = sk["names"], sk["parents"]
    clean = [cmap.get(n, "") for n in names]
    kids = collections.defaultdict(list)
    for j, p in enumerate(par):
        kids[p].append(j)

    def depth(j):
        d = 0
        while par[j] >= 0:
            j, d = par[j], d + 1
        return d

    def find(label):
        c = [j for j, n in enumerate(clean) if n == label]
        return min(c, key=depth) if c else None

    def follow(start, labels):
        chain, cur = [start], start
        for lab in labels[1:]:
            nxt = None
            frontier = list(kids[cur])
            for _ in range(3):   # allow up to two helper joints between labelled ones
                hit = [j for j in frontier if clean[j] == lab]
                if hit:
                    nxt = hit[0]
                    break
                frontier = [k for j in frontier for k in kids[j]]
            if nxt is None:
                break
            chain.append(nxt)
            cur = nxt
        return chain

    hips = find("Hips")
    head = find("Head")
    if hips is None or head is None:
        raise Skip("no_hips_or_head")
    spine, j = [], head
    while j >= 0 and j != hips:
        spine.append(j)
        j = par[j]
    if j != hips:
        raise Skip("head_not_below_hips")
    spine = [hips] + spine[::-1]
    roles = dict(hips=hips, head=head, spine=spine, clean=clean)
    for side, pre in (("L", "Left "), ("R", "Right ")):
        for key, labels in (("leg", LEG), ("arm", ARM)):
            s = find(pre + labels[0])
            roles[f"{key}_{side}"] = follow(s, [pre + x for x in labels]) if s is not None else None
    if not (roles["leg_L"] and roles["leg_R"] and len(roles["leg_L"]) >= 3 and len(roles["leg_R"]) >= 3):
        raise Skip("no_legs")
    # sides: with +Z forward, the character's left is +X
    if sk["Wp0"][roles["leg_L"][0]][0] < sk["Wp0"][roles["leg_R"][0]][0]:
        for key in ("leg", "arm"):
            roles[f"{key}_L"], roles[f"{key}_R"] = roles[f"{key}_R"], roles[f"{key}_L"]
        roles["sides_swapped"] = True
    return roles


def arms_ok(roles):
    return all(roles.get(f"arm_{s}") and len(roles[f"arm_{s}"]) >= 3 for s in "LR")


def audit(sk, roles):
    """Is the rest pose a standing stance? Returns (ok, reasons, stats)."""
    W = sk["Wp0"]
    H = W[roles["hips"]][1]
    why = []
    if H <= 1e-6:
        return False, ["hips_on_ground"], {}
    quad = body_class(sk, roles) == "quadruped"
    # legs must stand; a biped's arms are free (T-pose), a quadruped's forelimbs must stand too
    limbs = [roles["leg_L"], roles["leg_R"]] + ([roles["arm_L"], roles["arm_R"]] if arms_ok(roles) and quad else [])
    for ch in limbs:
        end, root = ch[min(2, len(ch) - 1)], ch[0]
        if W[end][1] > 0.3 * max(H, W[root][1]):
            why.append("limb_end_off_ground")
            break
        if W[end][1] >= W[root][1]:
            why.append("limb_end_above_root")
            break
    four = arms_ok(roles)
    stats = dict(hips_h=float(H), trunk_len=float(np.linalg.norm(W[roles["head"]] - W[roles["hips"]])))
    if four and (roles["leg_L"][0] != roles["arm_L"][0]):
        hind = (W[roles["leg_L"][0]][2] + W[roles["leg_R"][0]][2]) / 2
        fore = (W[roles["arm_L"][0]][2] + W[roles["arm_R"][0]][2]) / 2
        stats["hind_minus_fore_z"] = float(hind - fore)
    return (not why), why, stats


def body_class(sk, roles):
    """quadruped when the trunk runs horizontally (hips -> head mostly along z), biped when it is upright."""
    v = sk["Wp0"][roles["head"]] - sk["Wp0"][roles["hips"]]
    return "biped" if abs(v[1]) > 1.2 * math.hypot(v[0], v[2]) else "quadruped"


# ----------------------------------------------------------------------------- retargeting core
def neutral_dir(sk, chain, i, arm):
    """Neutral direction of segment i (joint chain[i] -> chain[i+1]): the rest direction, except that
    arms held out sideways (T-pose) hang down in neutral. A standing quadruped's forelegs keep their rest direction."""
    d = unit(sk["Wp0"][chain[i + 1]] - sk["Wp0"][chain[i]])
    if arm and abs(d[1]) < 0.5:  # held more than 60 degrees away from vertical (a T-pose): neutral is hanging down
        return DOWN.copy()
    return d


def smooth(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, k, mode="valid")


def contacts_from_source(src, roles, fps, thr_h=0.05, thr_v=0.6, min_len=3):
    """Boolean (T,) per source foot: low and (nearly) still."""
    H = src["Wp0"][roles["hips"]][1]
    out = {}
    for s in "LR":
        end = roles[f"leg_{s}"][min(2, len(roles[f"leg_{s}"]) - 1)]
        p = src["Wp"][:, end]
        h = (p[:, 1] - src["Wp0"][end][1]) / H
        sp = np.r_[0.0, np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)] * fps / H
        c = (h < thr_h) & (sp < thr_v)
        out[s] = clean_runs(c, min_len)
    return out


def clean_runs(c, min_len):
    c = c.copy()
    for val in (True, False):       # drop short contact runs, then fill short gaps
        i = 0
        while i < len(c):
            j = i
            while j < len(c) and c[j] == c[i]:
                j += 1
            if c[i] == val and (j - i) < min_len and i > 0 and j < len(c):
                c[i:j] = not val
            i = j
    return c


def twist_x(q):
    """Unwrapped rotation angle (radians) of q (w first) about the lateral X axis: the twist part of the
    swing-twist split q = swing * twist, kept continuous over time."""
    return np.unwrap(2 * np.arctan2(q[:, 1], q[:, 0]), period=2 * np.pi)


def rx(theta):
    return np.stack([np.cos(theta / 2), np.sin(theta / 2), np.zeros_like(theta), np.zeros_like(theta)], -1)


def heading_psi(Dp):
    """Body heading (radians, about Y) from where the body's left-right axis points, so it stays defined through
    flips; unwrapped over time. Rotating x/z by psi: (x, z) -> (x cos + z sin, -x sin + z cos)."""
    T = len(Dp)
    lat = qrot(Dp, np.broadcast_to(np.array([1.0, 0, 0]), (T, 3)))
    psi = np.arctan2(-lat[:, 2], lat[:, 0])
    weak = np.hypot(lat[:, 0], lat[:, 2]) < 0.2
    for t in range(T):                         # left-right axis near vertical (rolled over): keep the last heading
        if weak[t]:
            psi[t] = psi[t - 1] if t else 0.0
    return np.unwrap(psi)


def shape_body_rotation(Dp, cs, tilt_limit_deg, stance_limit_deg=12.0):
    """Make a human body rotation workable for a quadruped (whose trunk lies horizontal at rest).
    Only the rotation about the lateral axis (pitch: flips, bows, crouch lean) is changed (after the heading is taken out):
      airborne  the rotation is stretched so each flight ends on a whole number of turns (a flip that the
                human finishes after landing is finished in the air instead);
      grounded  the remaining tilt is soft-limited (L * tanh(angle / L)), so the trunk does not stand up on its
                head while the feet are down.
    Everything else (yaw, roll) is left as it is. Returns the new world rotation change (T, 4)."""
    T = len(Dp)
    # heading first: the pitch about the body's own lateral axis is only meaningful once a turn (yaw) is taken out
    psi = heading_psi(Dp)
    Yq = np.stack([np.cos(psi / 2), np.zeros(T), np.sin(psi / 2), np.zeros(T)], -1)
    Tilt = qmul(qconj(Yq), Dp)
    th = twist_x(Tilt)
    Sw = qmul(Tilt, qconj(rx(th)))
    air = clean_runs(~cs["L"] & ~cs["R"], 3)
    both = cs["L"] & cs["R"]                    # all feet down: the pitch is largely fixed by the leg lengths
    lim = np.where(both, math.radians(stance_limit_deg), math.radians(tilt_limit_deg))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    eff = np.zeros(T)
    big = math.radians(30)
    Theta, i = 0.0, 0                      # whole turns completed so far, in radians
    while i < T:
        j = i
        while j < T and air[j] == air[i]:
            j += 1
        if not air[i]:
            eff[i:j] = Theta + lim[i:j] * np.tanh(wrap(th[i:j] - Theta) / lim[i:j])
        else:
            raw0 = th[i - 1] if i > 0 else th[0]
            eff0 = eff[i - 1] if i > 0 else 0.0
            land = th[j] if j < T else th[-1]
            net = land - raw0
            if abs(net) > big:             # a flip: stretch it to end on a whole number of turns
                Theta_next = 2 * np.pi * round(land / (2 * np.pi))
                prog = np.clip((th[i:j] - raw0) / net, 0.0, 1.0)
                eff[i:j] = eff0 + prog * (Theta_next - eff0)
                Theta = Theta_next
            else:                          # a hop: follow the source
                eff[i:j] = eff0 + (th[i:j] - raw0)
        i = j
    eff = smooth(eff, 1.5)
    return qmul(Yq, qmul(Sw, rx(eff)))


def two_bone(R0, M0, E, l1, l2, rest_pole=None):
    """Positions of the middle joint for a two-bone chain from R0 to E. The bend direction is the current
    middle joint's (M0), but it is pulled towards the rest-pose bend (carried with the body) when the limb is
    nearly straight or when M0 bends the other way, so a knee or elbow never flips. Returns (M, E_reached, ok)."""
    v = E - R0
    d = np.linalg.norm(v)
    u = v / max(d, 1e-12)
    dmax, dmin = (l1 + l2) * 0.999, abs(l1 - l2) + 1e-6
    ok = dmin * 0.97 <= d <= (l1 + l2) * 1.03          # small clamps are not failures; the foot is clamped either way
    dc = min(max(d, dmin), dmax)
    pole = (M0 - R0) - u * np.dot(M0 - R0, u)
    pn = np.linalg.norm(pole)
    pu = pole / max(pn, 1e-12)
    if rest_pole is not None:
        rp = rest_pole - u * np.dot(rest_pole, u)
        if np.linalg.norm(rp) > 1e-9:
            ru = rp / np.linalg.norm(rp)
            w = float(np.clip((pn / l1 - 0.05) / 0.15, 0.0, 1.0)) * float(np.clip(np.dot(pu, ru) / 0.5, 0.0, 1.0))
            pole = w * pu + (1 - w) * ru
            pn = np.linalg.norm(pole)
    if pn < 1e-9:
        pole = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(pole) < 1e-9:
            pole = np.cross(u, [0, 0, 1.0])
    v_ = pole / np.linalg.norm(pole)
    ca = (l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc)
    sa = math.sqrt(max(0.0, 1 - ca * ca))
    return R0 + l1 * (ca * u + sa * v_), R0 + u * dc, ok


def retarget(src, src_roles, tgt, tgt_roles, params, fps):
    T = len(src["apos"])
    Jt = len(tgt["names"])
    par = tgt["parents"]
    info = {}
    # ---------------- source quantities
    Wp_s, Wr_s, Wp_s0, Wr_s0 = src["Wp"], src["Wr"], src["Wp0"], src["Wr0"]
    hs = src_roles["hips"]
    D = lambda j: qmul(Wr_s[:, j], qconj(Wr_s0[j]))[...]           # world rotation change from rest, joint j
    Dp_s = D(hs)                                                    # source body rotation change
    Dp = Dp_s                                                       # target body rotation change (shaped below)
    Hs = Wp_s0[hs][1]
    Ht = tgt["Wp0"][tgt_roles["hips"]][1]
    rho = Ht / Hs
    # ---------------- target rest, body centre
    th = tgt_roles["hips"]
    limb_roots = [tgt_roles[k][0] for k in ("leg_L", "leg_R", "arm_L", "arm_R") if tgt_roles.get(k)]
    four = arms_ok(tgt_roles) and tgt_roles["leg_L"][0] != tgt_roles["arm_L"][0] and body_class(tgt, tgt_roles) == "quadruped"
    cs = contacts_from_source(src, src_roles, fps)
    if four and params["shape"]:
        Dp = shape_body_rotation(Dp_s, cs, params["tilt_limit"], params["stance_tilt"])
    Corr = qmul(Dp, qconj(Dp_s))                                    # what shaping changed; applied to the spine too
    C_rest = tgt["Wp0"][limb_roots].mean(0) if four else tgt["Wp0"][th].copy()
    off = C_rest - tgt["Wp0"][th]
    dp = Wp_s[:, hs] - Wp_s0[hs]
    C = np.zeros((T, 3))
    C[:, 0] = C_rest[0] + rho * params["stride"] * dp[:, 0]
    C[:, 2] = C_rest[2] + rho * params["stride"] * dp[:, 2]
    C[:, 1] = C_rest[1] * (1 + params["vgain"] * (Wp_s[:, hs, 1] / Hs - 1))
    H_des = C - qrot(Dp, off)                                       # hips position that puts the trunk centre on C
    info.update(rho=float(rho), center_offset=float(np.linalg.norm(off)))

    # ---------------- world rotations of the mapped target joints
    Wr_t = {}                                                       # joint -> (T,4) world rotation
    # spine chain: resample by normalized arc length
    def arclen(sk, chain):
        p = sk["Wp0"][chain]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        return s / max(s[-1], 1e-9)

    ss, ts = src_roles["spine"], tgt_roles["spine"]
    us, ut = arclen(src, ss), arclen(tgt, ts)
    # spine joints: the target body rotation, plus each joint's bend relative to the hips (scaled for a quadruped,
    # whose trunk does not fold the way a human's does)
    ident = np.tile(np.array([1.0, 0, 0, 0]), (T, 1))
    gain = params["spine_gain"] if four else 1.0
    Dsp = [qmul(Dp, qslerp(ident, qmul(qconj(Dp_s), D(j)), gain)) for j in ss]
    for k, u in zip(ts, ut):
        i = int(np.clip(np.searchsorted(us, u, side="right") - 1, 0, len(ss) - 2))
        w = (u - us[i]) / max(us[i + 1] - us[i], 1e-9)
        Wr_t[k] = qmul(qslerp(Dsp[i], Dsp[i + 1], w), np.broadcast_to(tgt["Wr0"][k], (T, 4)))

    # limbs: swing relative to the neutral direction, in the body frame
    mode = params["forelimbs"]
    if mode == "auto":
        path = np.linalg.norm(np.diff(Wp_s[:, hs][:, [0, 2]], axis=0), axis=1).sum() / Hs
        mode = "legs" if (path > 1.5 and four) else "arms"
    info["forelimbs"] = mode
    pairs = [("leg_L", "leg_L"), ("leg_R", "leg_R")]
    if four:
        pairs += [("arm_L", "leg_R"), ("arm_R", "leg_L")] if mode == "legs" else [("arm_L", "arm_L"), ("arm_R", "arm_R")]
    elif arms_ok(tgt_roles) and arms_ok(src_roles):
        pairs += [("arm_L", "arm_L"), ("arm_R", "arm_R")]
    Dpi = qconj(Dp_s)
    for tk, sk_key in pairs:
        tc, sc = tgt_roles[tk], src_roles.get(sk_key)
        if not sc or len(sc) < 3:
            continue
        for i in range(min(2, len(tc) - 1, len(sc) - 1)):
            d_s = unit(Wp_s[:, sc[i + 1]] - Wp_s[:, sc[i]])
            d_rel = qrot(Dpi, d_s)
            n_s = neutral_dir(src, sc, i, sk_key.startswith("arm"))
            q_rel = q_swing(np.broadcast_to(n_s, d_rel.shape), d_rel)
            d_rest_t = unit(tgt["Wp0"][tc[i + 1]] - tgt["Wp0"][tc[i]])
            n_t = neutral_dir(tgt, tc, i, tk.startswith("arm"))
            d_t = qrot(Dp, qrot(q_rel, np.broadcast_to(n_t, d_rel.shape)))
            Qref = qmul(Dp, np.broadcast_to(tgt["Wr0"][tc[i]], (T, 4)))
            carried = qrot(Dp, np.broadcast_to(d_rest_t, d_t.shape))
            Wr_t[tc[i]] = qmul(q_swing(carried, d_t), Qref)

    # ---------------- local rotations (unmapped joints keep their rest local rotation = rigid with parent)
    L = np.tile(tgt["rot"][None], (T, 1, 1))
    Wcur = {}
    order, seen = [], set()

    def visit(j):
        if j in seen:
            return
        seen.add(j)
        if par[j] >= 0:
            visit(par[j])
        order.append(j)

    for j in range(Jt):
        visit(j)
    for j in order:
        p = par[j]
        if j in Wr_t:
            Wj = Wr_t[j]
            if p >= 0:
                L[:, j] = qmul(qconj(Wcur[p]), Wj)
            else:
                L[:, j] = Wj
            Wcur[j] = Wj
        else:
            Wcur[j] = qmul(Wcur[p], L[:, j]) if p >= 0 else L[:, j]
    pos = np.tile(tgt["loc"][None], (T, 1, 1))
    root = [j for j in range(Jt) if par[j] < 0][0]
    # root translation so that the hips land on H_des
    pos[:, root] = 0.0
    Wp0_, _ = fk_pr(par, pos, L)
    pos[:, root] = H_des - Wp0_[:, th]
    Wp, Wr = fk_pr(par, pos, L)

    # ---------------- contacts, ground fix, IK
    metrics = dict(lift_max=0.0, lift_mean=0.0, penetration_max=0.0, reach_fail_frames=0, skate_mean=0.0, skate_max=0.0, contact_frames=0)
    limbs = {k: tgt_roles[k] for k in ("leg_L", "leg_R", "arm_L", "arm_R") if tgt_roles.get(k) and len(tgt_roles[k]) >= 3}
    cmap_t = {"leg_L": cs["L"], "leg_R": cs["R"]}
    if four:
        cmap_t.update({"arm_L": cs["R"], "arm_R": cs["L"]})
    cmap_t = {k: v for k, v in cmap_t.items() if k in limbs}
    ends = {k: ch[2] for k, ch in limbs.items()}
    end_rest_y = {k: tgt["Wp0"][e][1] for k, e in ends.items()}
    end_set = {}   # joints that must stay off the floor: legs, plus the forelimbs of a quadruped
    for k, ch in limbs.items():
        if k.startswith("leg") or four:
            for q in ch[2:]:
                end_set[q] = 0.5 * tgt["Wp0"][q][1]
    floor_j = np.array([end_set.get(j, 0.0) for j in range(Jt)])
    # digits (fingers, toes): with the foot kept flat (see ik_limb) they hang at their rest offsets below the end
    # joint, so the end joint's clearance floor is raised by that drop (used by the clearance IK below)
    digits_of = {}
    for k, ch in limbs.items():
        if k.startswith("leg") or four:
            digits_of[k] = [j for j in range(Jt) if _under(par, j, ch[2])]
    tol = 1e-4

    pins = {}

    psi = heading_psi(Dp)
    rot2 = lambda v, ps: np.array([v[0] * math.cos(ps) + v[1] * math.sin(ps), -v[0] * math.sin(ps) + v[1] * math.cos(ps)])
    src_leg = {"leg_L": "leg_L", "leg_R": "leg_R", "arm_L": "leg_R", "arm_R": "leg_L"}     # forelegs copy the diagonal leg

    def make_pin(k, a):
        """Ground position the foot is held at for a contact run starting at frame a: the target's own stance under
        the body (rest offset from the trunk centre, turned by the body heading) plus how far the source foot is
        from ITS stance, scaled by the size ratio. A foot planted under a human lands under the animal; a step
        forward becomes a proportionally smaller step."""
        e = ends[k]
        se_chain = src_roles[src_leg[k]]
        se = se_chain[min(2, len(se_chain) - 1)]
        ps = psi[a]
        dev = (Wp_s[a, se] - Wp_s[a, hs])[[0, 2]] - rot2((Wp_s0[se] - Wp_s0[hs])[[0, 2]], ps)
        nat = C[a][[0, 2]] + rot2((tgt["Wp0"][e] - C_rest)[[0, 2]], ps)
        xz = nat + rho * params["stride"] * dev
        return np.array([xz[0], end_rest_y[k], xz[1]])

    if params["ik"]:
        # One pin per contact run of every limb. A run is cut (the foot takes a step: it is re-pinned) when the
        # body has turned more than ~35 degrees or travelled about half a limb's reach since the pin was set.
        yaw_t = psi
        wrap_ = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        for k, c in cmap_t.items():
            reach = np.linalg.norm(tgt["Wp0"][limbs[k][1]] - tgt["Wp0"][limbs[k][0]]) + np.linalg.norm(tgt["Wp0"][limbs[k][2]] - tgt["Wp0"][limbs[k][1]])
            i = 0
            while i < T:
                if c[i]:
                    j = i
                    while j < T and c[j]:
                        j += 1
                    st = i
                    for t in range(i + 1, j + 1):
                        cut = t == j or abs(wrap_(yaw_t[t] - yaw_t[st])) > math.radians(35) or \
                            np.linalg.norm((H_des[t] - H_des[st])[[0, 2]]) > 0.5 * reach
                        if cut:
                            if t - st >= 2:
                                pins[(k, st, t)] = make_pin(k, st)
                            st = t
                    i = j
                else:
                    i += 1
    # Root height correction, then limb IK.
    #  * Feet, hands and toes are handled by IK: pinned to the ground during contact, lifted to the floor when a
    #    swinging foot would go under it. They never push the body up.
    #  * The root is lifted only for everything else (head, spine, tail...) and lowered only so that pinned feet
    #    can be reached.
    ramp = 5
    Wp_pre = Wp.copy()
    limb_ends = [q for ch in limbs.values() for q in ch[2:] if q in end_set]
    grounded = cs["L"] | cs["R"] | clean_runs(~(~cs["L"] & ~cs["R"]), 3)   # the body is on the ground (not in flight)
    exempt = np.zeros((T, Jt), bool)
    exempt[np.ix_(grounded, limb_ends)] = True                      # grounded frames: foot floor is IK's job; in flight the body rises
    # grounded frames: the digits below a limb end are posed by the foot IK, so their pre-IK height says nothing about
    # where they end up (the post-IK residual below still checks them)
    exempt_pre = exempt.copy()
    exempt_pre[np.ix_(grounded, [j for dg in digits_of.values() for j in dg])] = True
    need = np.where(exempt_pre, -np.inf, floor_j[None] - Wp_pre[:, :, 1])
    lo_base = need.max(1)
    seglen_of = lambda ch: (np.linalg.norm(tgt["Wp0"][ch[1]] - tgt["Wp0"][ch[0]]), np.linalg.norm(tgt["Wp0"][ch[2]] - tgt["Wp0"][ch[1]]))
    # horizontal reach correction: while feet are pinned the body must stay over them. Where the scaled source path
    # would pull a limb root out of reach, move the root just enough (smoothed); the drift is reported.
    shift = np.zeros((T, 2))
    if pins:
        for _ in range(4):
            acc, cnt = np.zeros((T, 2)), np.zeros(T)
            for (k, a, b), pin in pins.items():
                ch = limbs[k]
                reach = sum(seglen_of(ch))
                for t in range(a, b):
                    R0 = Wp_pre[t, ch[0]] + np.array([shift[t, 0], 0.0, shift[t, 1]])
                    dxz = (pin - R0)[[0, 2]]
                    dh = np.linalg.norm(dxz)
                    dv = abs(R0[1] - pin[1])
                    ok_h = math.sqrt(max((0.97 * reach) ** 2 - dv * dv, 0.0))
                    if dh > ok_h and dh > 1e-9:
                        acc[t] += dxz / dh * (dh - ok_h)
                        cnt[t] += 1
            shift += np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), 0.0)
            shift = smooth(shift, 2.0)
    metrics["drift_max_m"] = float(np.linalg.norm(shift, axis=1).max()) if pins else 0.0
    hi = np.full(T, np.inf)
    for (k, a, b) in pins:
        ch = limbs[k]
        l1 = np.linalg.norm(tgt["Wp0"][ch[1]] - tgt["Wp0"][ch[0]])
        l2 = np.linalg.norm(tgt["Wp0"][ch[2]] - tgt["Wp0"][ch[1]])
        pin = pins[(k, a, b)]
        for t in range(a, b):
            R0 = Wp_pre[t, ch[0]] + np.array([shift[t, 0], 0.0, shift[t, 1]])
            dh = np.linalg.norm((R0 - pin)[[0, 2]])
            vmax = math.sqrt(max(((l1 + l2) * 0.98) ** 2 - dh * dh, 0.0))
            hi[t] = min(hi[t], vmax - (R0[1] - pin[1]))
    metrics["lift_binding_joints"] = dict(collections.Counter(tgt["names"][j] for j in need.argmax(1)[lo_base > 1e-3]).most_common(4))
    rest_pole, bone_local, seglen = {}, {}, {}

    def between(child, anc):
        """Joints strictly between anc and child on the tree path (a rig may have an unlabelled joint, e.g. a
        twist joint, between two limb joints: the IK must go through it, not assume child's parent is anc)."""
        out, j = [], par[child]
        while j >= 0 and j != anc:
            out.append(j)
            j = par[j]
        return out[::-1]

    mid = {k: (between(ch[1], ch[0]), between(ch[2], ch[1])) for k, ch in limbs.items()}
    for k, ch in limbs.items():
        u0 = unit(tgt["Wp0"][ch[2]] - tgt["Wp0"][ch[0]])
        m0 = tgt["Wp0"][ch[1]] - tgt["Wp0"][ch[0]]
        rest_pole[k] = m0 - u0 * np.dot(m0, u0)
        seglen[k] = (np.linalg.norm(tgt["Wp0"][ch[1]] - tgt["Wp0"][ch[0]]), np.linalg.norm(tgt["Wp0"][ch[2]] - tgt["Wp0"][ch[1]]))
        for i in (0, 1):
            bone_local[(k, i)] = qrot(qconj(tgt["Wr0"][ch[i]]), unit(tgt["Wp0"][ch[i + 1]] - tgt["Wp0"][ch[i]]))
    sm = lambda x: x * x * (3 - 2 * x)

    def ik_limb(Lw, posw, Wp, Wr, t, k, E_target, w, count_reach=False):
        """Move the end of limb k at frame t towards E_target with weight w (two-bone IK), rebuild its subtree."""
        ch = limbs[k]
        a_, b_, c_ = ch[0], ch[1], ch[2]
        l1, l2 = seglen[k]
        R0, M0, E0 = Wp[t, a_], Wp[t, b_], Wp[t, c_]
        E = E0 + w * (E_target - E0)
        E[1] = max(E[1], floor_j[c_])
        M, Er, ok = two_bone(R0, M0, E, l1, l2, qrot(Dp[t], rest_pole[k]))
        if not ok and count_reach:
            metrics["reach_fail_frames"] += 1
            metrics.setdefault("reach_fail_by_limb", {}).setdefault(k, []).append((t, round(float(np.linalg.norm(E - R0) / (l1 + l2)), 2)))
        d1n, d2n = unit(M - R0), unit(Er - M)
        # IK rotations are built from the rest pose carried by the actual parent (independent of the swung pose,
        # so no axis flips) and blended with the unconstrained rotation by the weight
        Pa = Wr[t, par[a_]] if par[a_] >= 0 else np.array([1.0, 0, 0, 0])
        Qa = qmul(Pa, tgt["rot"][a_])
        Wa = qslerp(Wr[t, a_], qmul(q_swing(qrot(Qa, bone_local[(k, 0)]), d1n), Qa), w)
        Pb = Wa                                 # world rotation of b_'s actual parent (through any joints in between)
        for j_ in mid[k][0]:
            Pb = qmul(Pb, Lw[t, j_])
        Qb = qmul(Pb, tgt["rot"][b_])
        Wb = qslerp(Wr[t, b_], qmul(q_swing(qrot(Qb, bone_local[(k, 1)]), d2n), Qb), w)
        Lw[t, a_] = qmul(qconj(Pa), Wa)
        Lw[t, b_] = qmul(qconj(Pb), Wb)
        # keep the foot flat: rest orientation turned by the body heading only (blended with the weight)
        Pc = Wb
        for j_ in mid[k][1]:
            Pc = qmul(Pc, Lw[t, j_])
        f = qrot(Dp[t], np.array([0.0, 0, 1]))
        hq = np.array([math.cos(math.atan2(f[0], f[2]) / 2), 0.0, math.sin(math.atan2(f[0], f[2]) / 2), 0.0])
        Wf = qslerp(qmul(Pc, Lw[t, c_]), qmul(hq, tgt["Wr0"][c_]), w)
        Lw[t, c_] = qmul(qconj(Pc), Wf)
        for jj in order:                       # the limb and everything below it moves with the new rotations
            pj = par[jj]
            if jj in (a_, b_) or _under(par, jj, a_):
                Wr[t, jj] = qmul(Wr[t, pj], Lw[t, jj]) if pj >= 0 else Lw[t, jj]
                Wp[t, jj] = Wp[t, pj] + qrot(Wr[t, pj], posw[t, jj]) if pj >= 0 else posw[t, jj]

    lift_floor = np.zeros(T)
    Lw, posw, delta = L, pos, np.zeros(T)
    for it in range(4 if params["ik"] else 1):
        Lw, posw = L.copy(), pos.copy()
        metrics["reach_fail_frames"] = 0
        metrics.pop("reach_fail_by_limb", None)
        if params["ik"]:
            lo_t = np.maximum(lo_base, lift_floor)
            delta = np.maximum(np.minimum(0.0, hi), lo_t)
            delta = np.where(np.isfinite(delta), delta, lo_t)
            delta = np.maximum(smooth(delta, 1.5), lo_t)
        posw[:, root, 1] += delta
        posw[:, root, 0] += shift[:, 0]
        posw[:, root, 2] += shift[:, 1]
        Wp, Wr = fk_pr(par, posw, Lw)
        if params["ik"]:
            for k in limbs:                    # clearance: a swinging foot that would be under the floor is lifted to it
                if k not in cmap_t:
                    continue
                e = limbs[k][2]
                drop = max((tgt["Wp0"][e][1] - tgt["Wp0"][j][1] for j in digits_of.get(k, [])), default=0.0)
                fl = max(floor_j[e], drop)
                # IK weight: 1 where the end joint is under its floor, falling off over 4 frames either side so
                # the limb eases in and out of the correction (a hard switch shows up as a rotation spike)
                depth = np.where(grounded, fl - Wp[:, e, 1], -1.0)
                w_raw = np.where(depth > 1e-4, sm(np.clip(depth / 0.005, 0.0, 1.0)), 0.0)
                w_t = w_raw.copy()
                for off_ in range(1, 4):
                    f_ = 1 - off_ / 4
                    w_t[off_:] = np.maximum(w_t[off_:], w_raw[:-off_] * f_)
                    w_t[:-off_] = np.maximum(w_t[:-off_], w_raw[off_:] * f_)
                for t in np.nonzero(w_t > 1e-3)[0]:
                    ik_limb(Lw, posw, Wp, Wr, t, k, np.array([Wp[t, e, 0], max(fl, Wp[t, e, 1]), Wp[t, e, 2]]), float(w_t[t]))
            for (k, a, b) in pins:
                pin = pins[(k, a, b)]
                for t in range(max(0, a - ramp), min(T, b + ramp)):
                    w = sm(min(1.0, (t - (a - ramp)) / ramp)) if t < a else (1.0 if t < b else sm(max(0.0, 1 - (t - b + 1) / (ramp + 1))))
                    if w > 1e-6:
                        ik_limb(Lw, posw, Wp, Wr, t, k, pin, w, count_reach=(a <= t < b))
        Wp, Wr = fk_pr(par, posw, Lw)
        resid = np.maximum(np.where(exempt, -np.inf, floor_j[None] - Wp[:, :, 1]).max(1), 0)
        if not params["ik"] or resid.max() < 2e-3:
            break
        lift_floor = delta + smooth(resid, 1.0)
    L, pos = Lw, posw
    metrics["lift_max"] = float(delta.max())
    metrics["lift_mean"] = float(np.maximum(delta, 0).mean())

    # ---------------- metrics
    pen = np.maximum(floor_j[None] - Wp[:, :, 1], 0)
    metrics["penetration_max"] = float(pen.max())
    if pen.max() > 0:
        t_, j_ = np.unravel_index(pen.argmax(), pen.shape)
        metrics["penetration_at"] = f"frame {int(t_)} {tgt['names'][j_]} (floor {floor_j[j_]:.3f}, y {Wp[t_, j_, 1]:.3f})"
    sk_list = []
    for (k, a, b) in pins:
        e = ends[k]
        stepping = any(kk == k and aa == b for (kk, aa, _) in pins)      # the next pin starts here: the last frames are a step
        b_eff = max(a + 1, b - ramp) if stepping else b
        v = np.linalg.norm(np.diff(Wp[a:b_eff, e][:, [0, 2]], axis=0), axis=1) * fps if b_eff - a > 1 else np.zeros(1)
        sk_list.append(v)
        metrics["contact_frames"] += b - a
    if sk_list:
        allv = np.concatenate(sk_list)
        metrics["skate_mean"], metrics["skate_max"] = float(allv.mean()), float(allv.max())
    L = qcontinuous(qnorm(L))
    step = np.degrees(2 * np.arccos(np.clip(np.abs((L[1:] * L[:-1]).sum(-1)), 0, 1)))
    metrics["rot_step_max_deg"] = float(step.max()) if len(step) else 0.0
    if len(step):
        t_, j_ = np.unravel_index(step.argmax(), step.shape)
        metrics["rot_step_at"] = f"frame {int(t_)} {tgt['names'][j_]}"
        metrics["rot_steps_over_60deg"] = int((step > 60).sum())
    metrics["root_travel_m"] = float(np.linalg.norm(np.diff(pos[:, root][:, [0, 2]], axis=0), axis=1).sum())
    info.update(metrics)
    return L, pos[:, root], info, Wp


def _under(par, j, anc):
    while j >= 0:
        j = par[j]
        if j == anc:
            return True
    return False


# ----------------------------------------------------------------------------- targets and sources on disk
def target_dirs():
    return sorted(p for p in CHARS.glob("*_*") if (p / "model.glb").exists() and not p.name.startswith("_"))


def rig_of(key):
    src, _, rid = key.partition("_")
    return src, ("mixamo" if src == "mixamo" else rid)


def list_targets(args):
    rows = []
    names_hint = re.compile(r"croc|alligator|gator|lizard|reptile|dragon|salamander|komodo|iguana|gecko|dino|rex", re.I)
    inv = collections.defaultdict(list)
    ip = ROOT / "data" / "inventory" / "clips.csv"
    if ip.exists():
        for r in csv.DictReader(open(ip, newline="")):
            inv[r["object_type"]].append(r["clip"] + " " + r["caption"])
    for d in target_dirs():
        ds, rid = rig_of(d.name)
        row = dict(key=d.name, source=ds, status="ok", reason="", cls="", hips_h="", trunk_over_h="", limbs="", hint="")
        try:
            sk = load_skeleton(d / "model.glb")
            roles = label_roles(sk, clean_names(ds, rid))
            ok, why, st = audit(sk, roles)
            row.update(cls=body_class(sk, roles), hips_h=round(st.get("hips_h", 0), 3),
                       trunk_over_h=round(st.get("trunk_len", 0) / max(st.get("hips_h", 1e-9), 1e-9), 2),
                       limbs="4" if arms_ok(roles) and roles["leg_L"][0] != roles["arm_L"][0] else "2")
            if not ok:
                row.update(status="rest_pose", reason=",".join(why))
            elif row["cls"] == "quadruped" and row["limbs"] != "4":
                row.update(status="rest_pose", reason="quadruped_without_fore_limbs")
            elif "hind_minus_fore_z" in st and st["hind_minus_fore_z"] > -0.05 * st["trunk_len"] and row["cls"] == "quadruped":
                row.update(status="rest_pose", reason="hind_not_behind_fore")
        except Skip as e:
            row.update(status="no_roles", reason=str(e))
        row["hint"] = "yes" if any(names_hint.search(t) for t in inv.get(rid, [])) else ""
        row["clips"] = len(inv.get(rid, [])) if ds != "mixamo" else ""
        rows.append(row)
    out = ROOT / "data" / "retarget_targets.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = collections.Counter((r["cls"] or "?", r["status"]) for r in rows)
    print(f"targets audited: {len(rows)}  -> {out}")
    for (c, s), n in sorted(by.items()):
        print(f"  {c:10s} {s:10s} {n}")
    print("reasons:", dict(collections.Counter(r["reason"] for r in rows if r["reason"])))
    cand = [r for r in rows if r["cls"] == "quadruped" and r["status"] == "ok"]
    cand.sort(key=lambda r: (r["hint"] != "yes", -float(r["trunk_over_h"] or 0)))
    print("quadruped targets (lizard/dragon-like names first, then longest body for its height):")
    for r in cand[:12]:
        print(f"  {r['key']}  trunk/height {r['trunk_over_h']}  hint={r['hint'] or '-'}  clips={r['clips']}")


def clip_index():
    rep = {}
    p = ROOT / "data" / "clips_report.csv"
    for r in csv.DictReader(open(p, newline="")):
        if r["status"] == "ok":
            rep[r["clip"]] = r["out"]
    inv = {r["clip"]: r for r in csv.DictReader(open(ROOT / "data" / "inventory" / "clips.csv", newline=""))}
    return rep, inv


def query_clips(q, k, cls="biped", semantic=True):
    """Best source clips for a text: cosine over CLIP caption embeddings (scripts/text/embed.py --text) when they exist,
    else the keyword match below. Names are not used (a clip's name can be wrong, see README)."""
    rep, inv = clip_index()
    if semantic and (ROOT / "data" / "embeddings" / "text.npy").exists():
        try:
            from embed import TextIndex
            hits = TextIndex().rank(q, cls=cls, usable=False, only=set(rep), top=k * 3)
            seen, out = set(), []
            for s, ds, c, _ in hits:
                if c in rep and c in inv and c not in seen:
                    seen.add(c)
                    out.append((c, s, inv[c]))
            return out[:k]
        except Exception as e:                                       # no model / no cache: keyword fallback
            print(f"(semantic search unavailable: {type(e).__name__}: {e}; using keywords)")
    toks = [t for t in re.findall(r"[a-z]+", q.lower()) if t]
    scored = []
    df = collections.Counter()
    docs = {}
    for c, r in inv.items():
        if c not in rep or r["character_class"] != cls:
            continue
        text = " ".join([c.replace("_", " "), r["caption"], r["detail"], r["generic"]]).lower()
        docs[c] = text
        for t in set(toks):
            if t in text:
                df[t] += 1
    n = max(len(docs), 1)
    for c, text in docs.items():
        s = sum((1 + text.count(t)) * math.log(1 + n / (1 + df[t])) for t in toks if t in text)
        if s > 0:
            scored.append((s, c))
    scored.sort(reverse=True)
    return [(c, s, inv[c]) for s, c in scored[:k]]


def source_path(clip):
    rep, _ = clip_index()
    if clip not in rep:
        raise SystemExit(f"clip {clip!r} not found among converted clips (data/clips_report.csv)")
    p = Path(rep[clip])
    return p if p.is_absolute() else ROOT / p


def source_roles(path, clip, inv):
    r = inv[clip]
    sk = load_clip(path)
    return sk, label_roles(sk, clean_names(r["dataset"], r["object_type"]))


# ----------------------------------------------------------------------------- run
def run_one(src_clip, target_key, args, write=True):
    rep, inv = clip_index()
    src, sroles = source_roles(source_path(src_clip), src_clip, inv)
    ds, rid = rig_of(target_key)
    tdir = CHARS / target_key
    tgt = load_skeleton(tdir / "model.glb")
    troles = label_roles(tgt, clean_names(ds, rid))
    ok, why, _ = audit(tgt, troles)
    if not ok and not args.force:
        raise SystemExit(f"{target_key}: rest pose is not a standing stance ({','.join(why)}); use --force to try anyway")
    params = dict(stride=args.stride_scale, vgain=args.vertical_gain, forelimbs=args.forelimbs, ik=not args.no_ik,
                  shape=not args.no_shaping, tilt_limit=args.tilt_limit, stance_tilt=args.stance_tilt, spine_gain=args.spine_gain)
    L, root_pos, info, Wp = retarget(src, sroles, tgt, troles, params, src["fps"])
    info["target_class"] = body_class(tgt, troles)
    info["source_clip"] = src_clip
    if not write:
        return L, root_pos, info, Wp, tgt, troles
    root = [j for j in range(len(tgt["names"])) if tgt["parents"][j] < 0][0]
    name = "rt__" + safe_name(src_clip, set())
    adir = tdir / "anims"
    adir.mkdir(exist_ok=True)
    write_clip(adir / f"{name}.glb", tgt["names"], tgt["parents"], tgt["loc"], tgt["rot_xyzw"], tgt["scale"], src["fps"],
               L[..., [1, 2, 3, 0]], {root: root_pos}, name, dict(source_clip=src_clip, fps=src["fps"], units="m", retargeted=True))
    f = adir / "anims.json"
    meta = json.loads(f.read_text()) if f.exists() else {}
    r = inv[src_clip]
    meta[name] = dict(file=f"{name}.glb", source_clip=src_clip, fps=src["fps"], frames=int(len(L)), duration_s=round(len(L) / src["fps"], 3),
                      loop=bool(int(r["loop_guess"] or 0)), loop_gap_deg=float(r["loop_gap_deg"]), root_travel_m=round(info["root_travel_m"], 3),
                      root_net_m=0.0, trimmed_frames=0, start_offset_m=0.0,
                      motion_captions=[t for t in (r["caption"], r["detail"], r["generic"]) if t],
                      retargeted={k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    f.write_text(json.dumps(meta, indent=1))
    return L, root_pos, info, Wp, tgt, troles


def self_test(args):
    """Stage tests that need no judgement: (1) a clip retargeted onto its own skeleton reproduces itself."""
    rep, inv = clip_index()
    ok_all = True
    for clip in ("Backflip", "Walking", "Jump_3", "Standing_Jump_3"):
        if clip not in rep:
            continue
        src, roles = source_roles(source_path(clip), clip, inv)
        params = dict(stride=1.0, vgain=1.0, forelimbs="arms", ik=False, shape=True, tilt_limit=35.0, stance_tilt=12.0, spine_gain=0.6)
        L, rp, info, Wp = retarget(src, roles, src, roles, params, src["fps"])
        # what the method promises on an identical skeleton: hips + spine positions, and the direction of the
        # two upper segments of every limb (shoulder and toe joints are carried rigidly, so they may differ)
        err = max(np.abs(Wp[:, j] - src["Wp"][:, j]).max() for j in roles["spine"])
        ang = 0.0
        for k in ("leg_L", "leg_R", "arm_L", "arm_R"):
            ch = roles[k]
            for i in range(min(2, len(ch) - 1)):
                d0 = unit(src["Wp"][:, ch[i + 1]] - src["Wp"][:, ch[i]])
                d1 = unit(Wp[:, ch[i + 1]] - Wp[:, ch[i]])
                ang = max(ang, float(np.degrees(np.arccos(np.clip((d0 * d1).sum(-1), -1, 1))).max()))
        H = src["Wp0"][roles["hips"]][1] * 2
        print(f"identity {clip:18s} spine error {err:.1e} m ({err / H:.1e} of body height), limb direction error {ang:.4f} deg, rho={info['rho']:.3f}")
        ok_all &= err / H < 1e-4 and ang < 0.05
    print("identity test:", "PASS" if ok_all else "FAIL")
    return ok_all


def batch(args):
    """Retarget several source clips onto several targets and write one metrics row per pair
    (data/retarget_report.csv). GLBs are only written with --write."""
    clips = args.batch.split(",")
    keys = [r["key"] for r in csv.DictReader(open(ROOT / "data" / "retarget_targets.csv", newline=""))
            if r["status"] == "ok" and r["cls"] == args.batch_class][:args.batch_targets]
    cols = ["clip", "target", "target_class", "forelimbs", "lift_max", "lift_mean", "drift_max_m", "skate_mean", "skate_max",
            "penetration_max", "reach_fail_frames", "rot_steps_over_60deg", "root_travel_m", "error"]
    out = ROOT / "data" / "retarget_report.csv"
    rows = []
    for clip in clips:
        for key in keys:
            row = dict(clip=clip, target=key, error="")
            try:
                _, _, info, *_ = run_one(clip, key, args, write=args.write)
                row.update({k: (round(info[k], 4) if isinstance(info.get(k), float) else info.get(k, "")) for k in cols if k in info})
            except (SystemExit, Skip, Exception) as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            print(f"  {clip:28s} {key[:38]:38s} lift {row.get('lift_max', '')!s:>7}  skate {row.get('skate_mean', '')!s:>7}  reach_fail {row.get('reach_fail_frames', '')!s:>4}  {row['error']}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    print(f"\n{len(ok)}/{len(rows)} runs ok -> {out}")
    for k in ("lift_max", "skate_mean", "penetration_max", "reach_fail_frames", "rot_steps_over_60deg"):
        v = [float(r[k]) for r in ok if r.get(k, "") != ""]
        if v:
            print(f"  {k:22s} median {np.median(v):8.3f}  worst {max(v):8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-targets", action="store_true")
    ap.add_argument("--query", help="text; lists the best matching source clips (CLIP caption embeddings, keyword fallback)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--source-clip", help="clip name as in data/inventory/clips.csv (e.g. Backflip)")
    ap.add_argument("--target", help="character folder in data/characters (e.g. objaverse_<id>)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--batch", help="comma list of source clips; retarget each onto --batch-targets targets and write data/retarget_report.csv")
    ap.add_argument("--batch-targets", type=int, default=6)
    ap.add_argument("--batch-class", choices=["quadruped", "biped"], default="quadruped")
    ap.add_argument("--write", action="store_true", help="with --batch: also write the clip GLBs")
    ap.add_argument("--forelimbs", choices=["auto", "arms", "legs"], default="auto")
    ap.add_argument("--stride-scale", type=float, default=1.0, help="multiplies the scaled horizontal root travel")
    ap.add_argument("--vertical-gain", type=float, default=1.0, help="multiplies the source's vertical excursion (jump height)")
    ap.add_argument("--tilt-limit", type=float, default=35.0, help="quadruped targets: soft limit (degrees) of the body pitch while feet are down")
    ap.add_argument("--stance-tilt", type=float, default=12.0, help="quadruped targets: soft limit (degrees) of the body pitch while both feet are down")
    ap.add_argument("--spine-gain", type=float, default=0.6, help="quadruped targets: scale of the spine bend relative to the hips")
    ap.add_argument("--no-shaping", action="store_true", help="transfer the body rotation as it is (no flip completion / tilt limit)")
    ap.add_argument("--no-ik", help="skip foot pinning and the ground/reach correction")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.list_targets:
        return list_targets(args)
    if args.query:
        for c, s, r in query_clips(args.query, args.top):
            print(f"{s:6.2f}  {c:40s}  {r['caption']}")
        return
    if args.self_test:
        return sys.exit(0 if self_test(args) else 1)
    if args.batch:
        return batch(args)
    if args.source_clip and args.target:
        _, _, info, _, _, _ = run_one(args.source_clip, args.target, args)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}, indent=1))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
