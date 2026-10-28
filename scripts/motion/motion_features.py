#!/usr/bin/env python3
"""Class-agnostic motion features for every usable clip, for the motion autoencoder (scripts/motion/motion_ae.py).

  python3 scripts/motion/motion_features.py          # -> data/embeddings/motion_feats.npz (+ motion_feats_index.csv)

A clip on any rig (biped or quadruped, 3,000+ different skeletons) is reduced to the same per-frame vector using the
labelled roles of retarget.py (hips, head, legs, arms) and forward kinematics on the normalized clip GLBs. All
lengths are divided by the rig's rest hips height H, so size drops out; directions are taken in the body's heading
frame (yaw taken out), so where the character faces drops out.

Per frame (F = 41):
  0-1    root velocity in the heading frame (forward, sideways), in H per frame of the stored timeline
  2      yaw rate (radians per stored frame)
  3      hips height above the clip's lowest foot joint / H (v2; v1 used the absolute height, which floated for rigs with a vertical root offset)
  4-9    hips tilt (pitch and roll, the rotation left after taking the yaw out) as the 6D representation
  10-12  head position relative to the hips, heading frame, / H
  13-36  knee/elbow and foot/hand positions of the 4 limbs relative to the hips (L leg, R leg, L arm, R arm; 2 joints x 3 each)
  37-40  limb present flags (a rig without labelled arms has zeros there and flag 0)

Timeline: a clip of up to 128 frames keeps its own timing (padded by holding the last frame, `n_valid` says how many
are real); a longer clip is resampled to 128 frames (`duration_s` keeps the true length), so one fixed-size array
describes a whole action whatever its length. Absolute signals are resampled, and velocities are derived afterwards.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "embeddings"
T_FIX = 128
F = 41
FLOOR_FROM_FEET = True        # hips height above the lowest foot joint of the clip (see absolute_signals)
NAMES = (["vel_fwd", "vel_side", "yaw_rate", "hips_h"] + [f"tilt6d_{i}" for i in range(6)] + ["head_x", "head_y", "head_z"] +
         [f"{limb}_{j}_{a}" for limb in ("legL", "legR", "armL", "armR") for j in ("mid", "end") for a in "xyz"] +
         [f"has_{limb}" for limb in ("legL", "legR", "armL", "armR")])


def qmat6d(q):
    """(T,4) w-first unit quaternions -> (T,6): first two columns of the rotation matrix."""
    w, x, y, z = q.T
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    return np.concatenate([c0, c1], -1)


def resample(x, n):
    """Linear resample (T, ...) -> (n, ...) over the same span."""
    T = len(x)
    if T == n:
        return x
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def absolute_signals(sk, roles):
    """Per native frame: hips position, heading psi (unwrapped), and the body pose in the heading frame (T, 37-?)."""
    import retarget as R
    from convert_clips import qmul, qrot
    Wp, Wr, Wp0, Wr0 = sk["Wp"], sk["Wr"], sk["Wp0"], sk["Wr0"]
    T = len(Wp)
    hips, head = roles["hips"], roles["head"]
    H = max(float(Wp0[hips][1]), 1e-3)
    D = qmul(Wr[:, hips], R.qconj(Wr0[hips]))
    psi = R.heading_psi(D)
    Yq = np.stack([np.cos(psi / 2), np.zeros(T), np.sin(psi / 2), np.zeros(T)], -1)
    Yi = R.qconj(Yq)
    tilt = qmul(Yi, D)
    tilt = tilt * np.where(tilt[:, :1] < 0, -1.0, 1.0)             # keep w >= 0 so the 6D value is continuous
    local = lambda j: qrot(Yi, Wp[:, j] - Wp[:, hips]) / H
    # hips height is measured above the clip's lowest foot joint, not above y = 0: a few hundred Objaverse clips carry a constant
    # vertical offset of about one hips height (the rest height counted twice), which made walkers "float" in the features
    feet = [roles[k][2] for k in ("leg_L", "leg_R") if roles.get(k) and len(roles[k]) >= 3]
    floor = float(Wp[:, feet, 1].min()) if FLOOR_FROM_FEET else 0.0
    cols, flags = [], []
    cols.append(local(head))
    for key in ("leg_L", "leg_R", "arm_L", "arm_R"):
        ch = roles.get(key)
        if ch and len(ch) >= 3:
            cols += [local(ch[1]), local(ch[2])]
            flags.append(1.0)
        else:
            cols += [np.zeros((T, 3)), np.zeros((T, 3))]
            flags.append(0.0)
    pose = np.concatenate([qmat6d(tilt)] + cols, 1)                # (T, 6 + 3 + 24)
    return dict(hips_xz=Wp[:, hips][:, [0, 2]] / H, hips_h=(Wp[:, hips, 1] - floor) / H, psi=psi, pose=pose, flags=np.array(flags), H=H)


def clip_features(sk, roles):
    """-> feats (128, F) float32, n_valid, duration_s"""
    a = absolute_signals(sk, roles)
    T = len(a["psi"])
    dur = T / sk["fps"]
    if T > T_FIX:
        a = {k: (resample(v, T_FIX) if k in ("hips_xz", "hips_h", "psi", "pose") else v) for k, v in a.items()}
        n_valid = T_FIX
    else:
        n_valid = T
        pad = lambda v: np.concatenate([v, np.repeat(v[-1:], T_FIX - T, 0)]) if T < T_FIX else v
        a = {k: (pad(v) if k in ("hips_xz", "hips_h", "psi", "pose") else v) for k, v in a.items()}
    psi = a["psi"]
    d = np.diff(a["hips_xz"], axis=0, prepend=a["hips_xz"][:1])    # world displacement per stored frame, in H
    c, s = np.cos(psi), np.sin(psi)
    fwd = d[:, 0] * s + d[:, 1] * c                                # components along the heading (x sin, z cos) and across it
    side = d[:, 0] * c - d[:, 1] * s
    yaw = np.diff(psi, prepend=psi[:1])
    out = np.zeros((T_FIX, F), np.float32)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = fwd, side, yaw, a["hips_h"]
    out[:, 4:37] = a["pose"]
    out[:, 37:41] = a["flags"]
    out[n_valid:, :3] = 0.0                                        # held pose: no velocity
    return out, n_valid, dur


def main():
    import retarget as R
    OUT.mkdir(parents=True, exist_ok=True)
    rep, inv = R.clip_index()
    rows = [r for r in inv.values() if r["usable"] == "1"]
    feats, meta, skipped = [], [], {}
    for i, r in enumerate(rows):
        c = r["clip"]
        try:
            p = Path(rep[c])
            sk = R.load_clip(p if p.is_absolute() else ROOT / p)
            roles = R.label_roles(sk, R.clean_names(r["dataset"], r["object_type"]))
            f, nv, dur = clip_features(sk, roles)
            if not np.isfinite(f).all():
                raise ValueError("non_finite")
            feats.append(f)
            meta.append(dict(dataset=r["dataset"], clip=c, cls=r["character_class"], body=R.body_class(sk, roles), n_valid=nv,
                             frames=len(sk["Wp"]), duration_s=round(dur, 3), rig=r["object_type"] if r["dataset"] != "mixamo" else "mixamo",
                             armsL=int(bool(roles.get("arm_L"))), armsR=int(bool(roles.get("arm_R")))))
        except Exception as e:
            k = type(e).__name__ + ": " + str(e)[:40]
            skipped[k] = skipped.get(k, 0) + 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    X = np.stack(feats)
    np.savez_compressed(OUT / "motion_feats.npz", feats=X, names=np.array(NAMES), n_valid=np.array([m["n_valid"] for m in meta]),
                        duration_s=np.array([m["duration_s"] for m in meta], np.float32))
    with open(OUT / "motion_feats_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    print(f"{len(meta)} clips of {len(rows)} usable -> {OUT / 'motion_feats.npz'} {X.shape}")
    print("skipped:", skipped)
    print("by body:", {b: sum(m["body"] == b for m in meta) for b in ("biped", "quadruped")})
    print("feature std:", dict(zip(NAMES[:10], np.round(X.reshape(-1, F).std(0)[:10], 3))))


if __name__ == "__main__":
    main()
