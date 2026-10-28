#!/usr/bin/env python3
"""Morphology-agnostic motion features for the semantic motion embedding (scripts/motion/motion_sem.py).

  python3 scripts/motion/motion_sem_features.py        # every clip with a caption -> data/embeddings/sem_feats.npz + sem_feats_index.csv

No role labels are needed (no hips, legs, arms): a biped, a quadruped, a spider, a bird or a snake all become a SET of
joint tokens, read straight from the raw clip NPZ (forward kinematics in the source's own units, Y up). Everything is
divided by the body size S (the largest extent of the rest pose), so size drops out, and nothing depends on which way the
character faces, so the same numbers describe "a walk" on any body.

Per joint and frame (4 channels, on a 128-frame timeline):
  height above the lowest point the clip reaches / S,  horizontal distance from the body centroid / S,
  vertical speed (S per second),  horizontal speed (S per second).
Per joint, static (8): rest height, rest radius, tree depth, children, leaf flag, subtree size, bone length / S, root flag.
Per frame, whole body (5): centroid height, centroid horizontal speed, centroid vertical speed, yaw rate of the body's
long axis, mean joint radius.  Per clip, scalar (8): log duration, log time-warp, joint count, leaf fraction, three
rest extents / S, fps / 30.
Timeline: clips of up to 128 frames keep their timing (padded by holding the last frame); longer ones are resampled to 128
(the warp factor is a scalar); speeds are always per real second.
"""
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "embeddings"
T_FIX, MAX_FK = 128, 1024
C_JOINT, C_STATIC, C_GLOB, C_SCAL = 4, 8, 5, 8


def resample(x, n):
    T = len(x)
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def to_timeline(x, T):
    """x (T, ...) -> (128, ...): padded by holding the last frame, or resampled when longer. Returns (array, warp)."""
    if T <= T_FIX:
        return np.concatenate([x, np.repeat(x[-1:], T_FIX - T, 0)]) if T < T_FIX else x, 1.0
    return resample(x, T_FIX), T / T_FIX


def clip_features(path):
    d = np.load(path, allow_pickle=False)
    return features_from_arrays(d["parents"].astype(int), d["rest_local_pos"].astype(float), d["rest_local_rot"].astype(float),
                                d["anim_local_pos"].astype(float), d["anim_local_rot"].astype(float), max(int(d["fps"]), 1))


def features_from_arrays(parents, rest_p, rest_r, pos, rot, fps):
    """parents (J,), rest local pos/rot (J,3|4, w first), animation local pos/rot (T,J,3|4), fps -> the feature tuple (see module doc)."""
    import retarget as R
    from convert_clips import qnorm
    J = len(parents)
    T0 = len(pos)
    idx = np.unique(np.linspace(0, T0 - 1, min(T0, MAX_FK)).round().astype(int))
    dt = np.maximum(np.diff(idx), 1) / fps                                    # seconds between the frames we use
    Wp, _ = R.fk_pr(parents.tolist(), pos[idx], qnorm(rot[idx]))
    P0, _ = R.fk_pr(parents.tolist(), rest_p[None], qnorm(rest_r[None]))
    P0 = P0[0]
    ext = np.ptp(P0, axis=0)
    S = max(float(ext.max()), 1e-6)
    y0 = float(Wp[..., 1].min())
    c = Wp.mean(1)                                                             # (T', 3) body centroid
    h = (Wp[..., 1] - y0) / S
    rad = np.linalg.norm((Wp - c[:, None])[..., [0, 2]], axis=-1) / S
    n = len(idx)

    def speed(x, axis_fn):
        if n < 2:
            return np.zeros((n,) + x.shape[1:2])
        o = axis_fn(np.diff(x, axis=0))
        v = o / dt.reshape((-1,) + (1,) * (o.ndim - 1))
        return np.concatenate([v[:1], v], 0)

    vy = speed(Wp, lambda dx: dx[..., 1]) / S
    vh = speed(Wp, lambda dx: np.linalg.norm(dx[..., [0, 2]], axis=-1)) / S
    joint = np.stack([h, rad, vy, vh], -1)                                    # (T', J, 4)
    cy = (c[:, 1] - y0) / S
    cvh = speed(c[:, None], lambda dx: np.linalg.norm(dx[..., [0, 2]], axis=-1))[:, 0] / S
    cvy = speed(c[:, None], lambda dx: dx[..., 1])[:, 0] / S
    # yaw rate of the body's long horizontal axis (principal axis of the joint x/z cloud; sign-free, so wrapped to +-90 deg)
    xz = (Wp - c[:, None])[..., [0, 2]]
    cov = np.einsum("tji,tjk->tik", xz, xz)
    ang = 0.5 * np.arctan2(2 * cov[:, 0, 1], cov[:, 0, 0] - cov[:, 1, 1])
    da = np.diff(ang, prepend=ang[:1])
    da = (da + np.pi / 2) % np.pi - np.pi / 2
    yaw = np.concatenate([da[1:] / dt, [0.0]]) if n > 1 else np.zeros(1)
    glob = np.stack([cy, cvh, cvy, yaw, rad.mean(1)], -1)                     # (T', 5)
    T_eff = len(idx)
    joint = np.clip(joint, -20, 20)
    glob = np.clip(glob, -20, 20)
    # timeline: warp is measured on the real clip length, not on the frames we kept
    jt, warp = to_timeline(joint, T_eff) if T0 <= T_FIX else (resample(joint, T_FIX), T0 / T_FIX)
    gt, _ = (to_timeline(glob, T_eff) if T0 <= T_FIX else (resample(glob, T_FIX), 1.0))
    n_valid = min(T0, T_FIX)
    # static joint descriptors
    depth = np.zeros(J)
    for j in range(J):
        a, k = j, 0
        while parents[a] >= 0 and k < J:
            a, k = parents[a], k + 1
        depth[j] = k
    kids = np.bincount(parents[parents >= 0], minlength=J)
    sub = np.ones(J)
    for j in np.argsort(-depth):
        if parents[j] >= 0:
            sub[parents[j]] += sub[j]
    bone = np.where(parents >= 0, np.linalg.norm(rest_p, axis=1), 0.0) / S
    rc = P0.mean(0)
    stat = np.stack([(P0[:, 1] - P0[:, 1].min()) / S, np.linalg.norm((P0 - rc)[:, [0, 2]], axis=1) / S, depth / max(depth.max(), 1),
                     np.minimum(kids, 4) / 4, (kids == 0).astype(float), sub / J, np.minimum(bone, 2.0), (parents < 0).astype(float)], 1)
    ex_h = sorted([ext[0] / S, ext[2] / S], reverse=True)
    scal = np.array([math.log(T0 / fps + 1e-3), math.log(warp), J / 100.0, (kids == 0).mean(), ext[1] / S, ex_h[0], ex_h[1], fps / 30.0])
    return (jt.transpose(1, 0, 2).astype(np.float16), stat.astype(np.float16), gt.astype(np.float16), scal.astype(np.float32), n_valid)


def _job(args):
    i, path = args
    try:
        r = clip_features(path)
        if not all(np.isfinite(np.asarray(a, dtype=float)).all() for a in r[:4]):
            return i, None, "non_finite"
        return i, r, ""
    except Exception as e:
        return i, None, f"{type(e).__name__}: {str(e)[:50]}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "inventory" / "clips.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    keep = [r for r in rows if r["finite"] == "1" and r["caption"] and r["frames"] and int(r["frames"]) >= 8 and r["quality_list"] != "filtered"]
    print(f"{len(rows)} clips, {len(keep)} with a caption, finite, >= 8 frames and not on the quality-filter list", flush=True)
    res = {}
    skipped = {}
    with ProcessPoolExecutor(6) as ex:
        for k, (i, r, err) in enumerate(ex.map(_job, [(i, r["path"]) for i, r in enumerate(keep)], chunksize=16)):
            if r is None:
                skipped[err] = skipped.get(err, 0) + 1
            else:
                res[i] = r
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(keep)}", flush=True)
    order = sorted(res)
    meta = [keep[i] for i in order]
    # extra non-human clips from the template rigs (scripts/dataset/template_clips.py), listed as dataset "template"
    import template_clips as TC
    extra = []
    for c in TC.read_clips():
        f = features_from_arrays(c["parents"], c["rest_p"], c["rest_r"], c["pos"], c["rot"], c["fps"])
        extra.append((c, f))
        meta.append(dict(dataset="template", clip=c["clip"], object_type=c["rig"], character_class=c["cls"], usable="0", frames=len(c["pos"]), fps=c["fps"],
                         duration_s=round(len(c["pos"]) / c["fps"], 3), joints=len(c["names"]), category="", caption=c["caption"], generic="", detail=""))
    print(f"+ {len(extra)} template clips", flush=True)
    jt = [res[i][0] for i in order] + [f[0] for _, f in extra]
    offs = np.cumsum([0] + [len(a) for a in jt])
    np.savez(OUT / "sem_feats.npz", joint=np.concatenate(jt), static=np.concatenate([res[i][1] for i in order] + [f[1] for _, f in extra]), offsets=offs,
             glob=np.stack([res[i][2] for i in order] + [f[2] for _, f in extra]), scal=np.stack([res[i][3] for i in order] + [f[3] for _, f in extra]),
             n_valid=np.array([res[i][4] for i in order] + [f[4] for _, f in extra]))
    cols = ["dataset", "clip", "object_type", "character_class", "usable", "frames", "fps", "duration_s", "joints", "category", "caption", "generic", "detail"]
    with open(OUT / "sem_feats_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(meta)
    import collections
    print(f"{len(meta)} clips ({len(order)} + {len(extra)} template) -> {OUT / 'sem_feats.npz'}; joints in total {offs[-1]}; skipped {skipped}")
    print("by class:", dict(collections.Counter(m["character_class"] for m in meta)))


if __name__ == "__main__":
    main()
