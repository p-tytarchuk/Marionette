#!/usr/bin/env python3
"""Marionette motion -> joint angles of a 2D (Lottie) rig: the hand-over from the motion model to scripts/lottie2d/bake_lottie.py.

  conda env Mario:   python3 scripts/lottie2d/motion_to_2d.py "a reindeer jumps" --id 7805483 [--seed 0] [--mode auto] [--forelimbs auto|legs|arms] [--travel]
  conda env loomie:  python3 scripts/lottie2d/bake_lottie.py 7805483 --motion data/side_view_animals/motion/<name>.json

The motion model decodes a prompt to a clip on the Mixamo reference skeleton (generate_motion.generate(..., target=None, keep_proxy=True): same front end,
sampler, latent cap and candidate gate as the 3D path, no GLB, no retarget). This file does what retarget.py does for a quadruped target, in the side plane
instead of 3D, onto the 14 joints of rig_mesh.py's rig (data/side_view_animals/rig/<id>.json):

  side plane   forward = (left hip - right hip) x up, so it stays horizontal when the body pitches; a vector (x, y, z) becomes the rig's (v . forward, -y), y down,
               the animal facing +x. Angles are atan2 in that frame (positive = clockwise on screen, like build_mesh.deform).
  body pitch   the hips' world rotation change applied to the forward axis -> pitch, unwrapped (a flip is 360 degrees). It becomes the rotation of the root joint spine_mid.
  limbs        each segment's angle in the body frame (world angle minus pitch) minus its neutral angle (the rest pose; arms held out sideways hang down, as in
               retarget.neutral_dir) is the swing; the rig joint's local rotation is that swing minus its parent segment's swing, so the rig's own rest pose is kept.
               Hind limbs <- the legs (thigh, shin, foot); forelimbs <- the opposite leg (a diagonal gait) when the hips travel more than 1.5 hips-heights, else the arms
               (retarget.py "auto"); the rig's third limb (a leg kicked up in the artwork) takes the other leg's swing.
  spine        the hips -> head chord's body-frame bend x 0.6 (retarget's --spine-gain for a quadruped) goes to spine_front 0.15, neck 0.25, head 0.20 of it.
  root         vertical: hips height relative to the rest pose, scaled by the rig's hips height / the human's (retarget's rho). Horizontal: none (the animal stays in place)
               unless --travel.
Not done (retarget.py does it in 3D): foot contacts, foot pinning with two-bone IK, floor / reach fixes, tilt soft limits. Feet can slide and, in a flip, the
body can cross the ground line.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
DEFAULT_DIR = ROOT / "data" / "side_view_animals"
DOWN = math.pi / 2                  # straight down in the rig's y-down frame: atan2(1, 0)
SPINE_SHARE = dict(spine_front=0.15, neck=0.25, head=0.20)     # of the hips -> head bend; sums to retarget's quadruped spine gain 0.6


def seg_angle(v, fwd):
    """(T, 3) world vectors -> (T,) angle in the rig's side frame (x forward, y down)."""
    return np.arctan2(-v[:, 1], (v * fwd).sum(1))


def ffill(a):
    a = a.copy()
    ok = ~np.isnan(a)
    if not ok.any():
        return np.zeros_like(a)
    first = int(np.argmax(ok))
    a[:first] = a[first]
    for i in range(first + 1, len(a)):
        if np.isnan(a[i]):
            a[i] = a[i - 1]
    return a


def swing(a_world, beta, a_rest):
    """body-frame angle minus its neutral angle, continuous in time, starting in (-pi, pi]."""
    d = np.unwrap(a_world - beta) - a_rest
    return d - 2 * np.pi * np.round(d[0] / (2 * np.pi))


def motion_to_angles(px, ref, rig, forelimbs="auto", travel=False):
    """px: proxy clip (generate_motion keep_proxy), ref: (skeleton, roles) -> dict(frames [{joint id: rad}], offsets [[dx, dy] px], info)."""
    import retarget as R
    from convert_clips import qmul, qrot
    sk, roles = ref
    Wp, Wr, W0, Wr0 = px["Wp"], px["Wr"], sk["Wp0"], sk["Wr0"]
    T = len(Wp)
    hips, head = roles["hips"], roles["head"]
    up = np.array([0.0, 1.0, 0.0])
    lat = R.unit(Wp[:, roles["leg_L"][0]] - Wp[:, roles["leg_R"][0]])          # the character's left is +X, so this is +X at rest
    fwd = R.unit(np.cross(lat, up))                                            # +Z at rest, horizontal always
    Dw = qmul(Wr[:, hips], R.qconj(np.broadcast_to(Wr0[hips], (T, 4))))
    beta = np.unwrap(seg_angle(qrot(Dw, np.broadcast_to(np.array([0.0, 0.0, 1.0]), (T, 3))), fwd))

    def chain_swings(chain, arm):
        out = []
        for i in range(len(chain) - 1):
            v = Wp[:, chain[i + 1]] - Wp[:, chain[i]]
            a = seg_angle(v, fwd)
            seen = np.hypot((v * fwd).sum(1), v[:, 1]) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            a[seen < 0.15] = np.nan                                           # seen end-on (an arm held out sideways): keep the last angle
            d0 = W0[chain[i + 1]] - W0[chain[i]]
            rest = DOWN if (arm and abs(d0[1]) / max(np.linalg.norm(d0), 1e-9) < 0.5) else math.atan2(-d0[1], d0[2])
            out.append(swing(ffill(a), beta, rest))
        return out

    def locals_(sw):
        return [sw[0]] + [sw[k] - sw[k - 1] for k in range(1, len(sw))]

    joints = {j["id"]: j for j in rig["joints"]}
    ang = {jid: np.zeros(T) for jid in joints}
    H = float(W0[hips][1])
    travelled = float(np.linalg.norm(px["root"][-1, [0, 2]] - px["root"][0, [0, 2]]))
    mode = forelimbs if forelimbs != "auto" else ("legs" if travelled > 1.5 * H else "arms")
    src = {"leg_back_L": ("leg_L", False), "leg_back_R": ("leg_R", False)}
    src.update({"leg_front_L": ("leg_R", False), "leg_front_R": ("leg_L", False)} if mode == "legs" else {"leg_front_L": ("arm_L", True), "leg_front_R": ("arm_R", True)})
    for limb, (key, arm) in src.items():
        if f"{limb}_upper" not in joints or not roles.get(key):
            continue
        for part, loc in zip(("upper", "lower", "foot"), locals_(chain_swings(roles[key], arm))):
            if f"{limb}_{part}" in joints:
                ang[f"{limb}_{part}"] = loc
    chord = Wp[:, head] - Wp[:, hips]
    c0 = W0[head] - W0[hips]
    bend = swing(seg_angle(chord, fwd), beta, math.atan2(-c0[1], c0[2]))
    if "spine_mid" in ang:
        ang["spine_mid"] = beta
    for jid, share in SPINE_SHARE.items():
        if jid in ang:
            ang[jid] = share * bend
    # root: hips height relative to the rest pose, in rig pixels (rho = rig hips height / human hips height); rig floor = lowest foot tip
    floor = max(p[1] for j in joints.values() for p in (j.get("tip_pos"), j["rest_pos"]) if p)
    rho = (floor - joints["spine_mid"]["rest_pos"][1]) / H
    dy = -rho * (Wp[:, hips, 1] - H)
    dx = rho * ((px["root"][:, [0, 2]] - px["root"][0, [0, 2]]) * fwd[0, [0, 2]]).sum(1) if travel else np.zeros(T)
    frames = [{jid: round(float(ang[jid][t]), 4) for jid in ang} for t in range(T)]
    info = dict(forelimbs=mode, hips_travel_in_hips_heights=round(travelled / H, 2), rho_px_per_m=round(rho, 1), pitch_range_deg=round(float(np.degrees(np.ptp(beta))), 1),
                max_swing_deg={jid: round(float(np.degrees(np.abs(a).max())), 1) for jid, a in ang.items()}, dy_range_px=[round(float(dy.min()), 1), round(float(dy.max()), 1)])
    return dict(frames=frames, offsets=[[round(float(x), 2), round(float(y), 2)] for x, y in zip(dx, dy)], info=info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--id", required=True, help="rig id (data/side_view_animals/rig/<id>.json)")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--mode", default="auto", choices=["auto", "plain", "walk", "run"])
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--forelimbs", default="auto", choices=["auto", "legs", "arms"])
    ap.add_argument("--travel", action="store_true", help="also move the animal forward (default: in place)")
    ap.add_argument("--name", help="output name (default: a slug of the prompt)")
    a = ap.parse_args()
    import generate_motion as G
    rig = json.loads((a.dir / "rig" / f"{a.id}.json").read_text())
    t0 = time.time()
    r = G.generate(a.prompt, None, guidance=a.guidance, seed=a.seed, mode=a.mode, tries=a.tries, keep_proxy=True)
    px, ref = r.pop("proxy"), r.pop("ref")
    m = motion_to_angles(px, ref, rig, a.forelimbs, a.travel)
    name = a.name or re.sub(r"[^a-z0-9]+", "_", r["prompt"].lower()).strip("_")[:40] + f"_s{a.seed}"
    out = dict(prompt=r["typed"], used_prompt=r["prompt"], mode=r["mode"], seed=a.seed, guidance=a.guidance, gate_passed=r["gate_passed"], quality=r["quality"], fps=px["fps"],
               duration_s=r["duration_s"], rig=a.id, loop=False, forelimbs=m["info"]["forelimbs"], info=m["info"], timing_ms=r["timing_ms"], frames=m["frames"], offsets=m["offsets"])
    (a.dir / "motion").mkdir(exist_ok=True)
    path = a.dir / "motion" / f"{name}.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"prompt {out['prompt']!r} -> {out['used_prompt']!r}, mode {out['mode']}, gate passed {out['gate_passed']}, {len(m['frames'])} frames at {px['fps']} fps ({time.time() - t0:.1f} s)")
    print(json.dumps(m["info"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
