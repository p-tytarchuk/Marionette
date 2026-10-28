#!/usr/bin/env python3
"""Convert the usable UniML3D clips (data/inventory/clips.csv) into GLB animation clips on the
normalized character skeletons (data/characters/*/model.glb).

Run scripts/dataset/inventory.py and scripts/dataset/normalize.py first. Each output clip is a GLB with only the
skeleton (same joint names, hierarchy and rest pose as model.glb) and one animation: 30 fps,
rotation channels for every joint, translation channels for the root and any joint that moves.
Units are metres, Y-up, +Z forward, root motion kept in the clip (no baking out).

Objaverse (one character per rig, one clip file per clip):
  data/characters/objaverse_<id>/anims/<action>.glb   + anims/anims.json
  The clip rest pose equals the character rest pose up to a uniform scale, a yaw and a shift
  (fitted per rig, residual ~1e-7). Rotations are unchanged; the root gets  s * R_yaw * p + t0.
Mixamo (one shared rig, 50 characters with different proportions):
  data/clips/mixamo/<clip>.glb  + data/clips/mixamo/clips.json   (22-joint core skeleton in metres,
  the Mixamo reference proportions, hips at 0.998 m)
  data/characters/mixamo_<name>/anims/anims.json   = {"_shared": {..., "root_scale": k}}
  where k = the character's hips height / the reference hips height; multiply the root translation
  by k to play a shared clip on that character. The 43 finger/helper joints outside the core are
  dropped (listed in clips.json).
anims.json entries: file, source_clip, fps, frames, duration_s, loop (guess), loop_gap_deg,
root_travel_m, root_net_m, start_offset_m (the removed x/z start offset), trimmed_frames, motion_captions (1-3).
Each clip starts on the rest-pose root in x/z (--keep-start-xz disables); clips whose root starts more than
--max-root-offset body heights above/below the rest root are skipped as root_frame_mismatch.

Every written clip is read back and its joint world positions are compared with the source NPZ
(forward kinematics, all frames); the relative error must stay below --tol or the clip is reported
as failed and removed. Report: data/clips_report.csv.

Usage:
  python3 scripts/dataset/convert_clips.py --limit 50 --dry-run
  python3 scripts/dataset/convert_clips.py
"""
import argparse
import collections
import csv
import json
import math
import re
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
from inventory import fk_rest  # noqa: E402  (rest-pose FK, w-first quaternions)

SRC = ROOT / "sources" / "unimate"
CHARS = ROOT / "data" / "characters"
MIXAMO_CM_TO_M = 0.01


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- quaternions (w first)
def qmul(a, b):
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qrot(q, v):
    qv = q[..., 1:]
    t = 2 * np.cross(qv, v)
    return v + q[..., :1] * t + np.cross(qv, t)


def qnorm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def qyaw(theta):
    return np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])


def qcontinuous(q):
    """Flip signs so consecutive frames stay in the same hemisphere (shortest-path slerp)."""
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    return q


def fk_frames(parents, pos, rot):
    """World positions (T, J, 3) from local pos (T, J, 3) and local quats (T, J, 4, w first)."""
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
    return Wp


def umeyama(A, B):
    """B ~ s R A + t"""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, D, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return (D * np.diag(S)).sum() / A0.var(0).sum(), R, mb - (D * np.diag(S)).sum() / A0.var(0).sum() * R @ ma


def yaw_of(R):
    return math.atan2(R[0, 2], R[2, 2])


# ----------------------------------------------------------------------------- glb io
def read_glb(path):
    d = Path(path).read_bytes()
    clen, _ = struct.unpack_from("<I4s", d, 12)
    js = json.loads(d[20:20 + clen])
    off = 20 + clen
    blen = struct.unpack_from("<I", d, off)[0] if off + 8 <= len(d) else 0
    return js, d[off + 8:off + 8 + blen]


def model_skeleton(path):
    js, _ = read_glb(path)
    nodes = js["nodes"]
    par = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    joints = []
    for s in js["skins"]:
        for j in s["joints"]:
            if j not in joints:
                joints.append(j)
    names = [nodes[j].get("name", "") for j in joints]
    if len(set(names)) != len(names):
        raise Skip("duplicate_joint_names")
    idx = {j: k for k, j in enumerate(joints)}
    parents = [idx.get(par.get(j), -1) for j in joints]
    t = np.array([nodes[j].get("translation", [0, 0, 0]) for j in joints], float)
    r = np.array([nodes[j].get("rotation", [0, 0, 0, 1]) for j in joints], float)  # xyzw
    sc = np.array([nodes[j].get("scale", [1, 1, 1]) for j in joints], float)
    rw = r[:, [3, 0, 1, 2]]
    pos, rot = t, rw
    Wp = fk_frames(parents, pos[None], rot[None])[0]
    return dict(names=names, parents=parents, loc=t, rot_xyzw=r, scale=sc, world=Wp)


def write_clip(path, names, parents, rest_loc, rest_rot_xyzw, rest_scale, fps, rot_xyzw, pos, name, extras):
    """rot_xyzw (T, J, 4); pos (T, J, 3) or None per joint -> dict j: (T,3). Writes skeleton + animation."""
    T, J = rot_xyzw.shape[:2]
    blobs, views, accs = [], [], []

    def add(arr, typ, minmax=False):
        a = np.ascontiguousarray(arr, dtype="<f4")
        pad = -sum(len(b) for b in blobs) % 4
        if pad:
            blobs.append(b"\0" * pad)
        off = sum(len(b) for b in blobs)
        blobs.append(a.tobytes())
        views.append(dict(buffer=0, byteOffset=off, byteLength=a.nbytes))
        acc = dict(bufferView=len(views) - 1, componentType=5126, count=int(a.shape[0]), type=typ)
        if minmax:
            acc["min"], acc["max"] = [float(a.min())], [float(a.max())]
        accs.append(acc)
        return len(accs) - 1

    nodes = [dict(name=n, translation=rest_loc[i].tolist(), rotation=rest_rot_xyzw[i].tolist(), scale=rest_scale[i].tolist())
             for i, n in enumerate(names)]
    for i, p in enumerate(parents):
        if p >= 0:
            nodes[p].setdefault("children", []).append(i)
    times = add(np.arange(T) / fps, "SCALAR", True)
    samplers, channels = [], []
    for j in range(J):
        samplers.append(dict(input=times, output=add(rot_xyzw[:, j], "VEC4"), interpolation="LINEAR"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=j, path="rotation")))
        if j in pos:
            samplers.append(dict(input=times, output=add(pos[j], "VEC3"), interpolation="LINEAR"))
            channels.append(dict(sampler=len(samplers) - 1, target=dict(node=j, path="translation")))
    blob = b"".join(blobs)
    blob += b"\0" * (-len(blob) % 4)
    js = dict(asset=dict(version="2.0", generator="Marionette convert_clips.py"),
              scene=0, scenes=[dict(nodes=[i for i, p in enumerate(parents) if p < 0])], nodes=nodes,
              animations=[dict(name=name, samplers=samplers, channels=channels, extras=extras)],
              accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(blob)
    Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(j), b"JSON") + j
                           + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


def read_clip_world(path):
    """Independent read-back: names -> world positions per frame via FK from the file's own accessors."""
    js, b = read_glb(path)
    nodes = js["nodes"]

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(b, "<f4", a["count"] * n, bv.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, n).astype(float)

    an = js["animations"][0]
    T = js["accessors"][an["samplers"][0]["input"]]["count"]
    J = len(nodes)
    pos = np.tile(np.array([n["translation"] for n in nodes], float)[None], (T, 1, 1))
    rot = np.tile(np.array([n["rotation"] for n in nodes], float)[None][..., [3, 0, 1, 2]], (T, 1, 1))
    for ch in an["channels"]:
        out = acc(an["samplers"][ch["sampler"]]["output"])
        j = ch["target"]["node"]
        if ch["target"]["path"] == "translation":
            pos[:, j] = out
        else:
            rot[:, j] = out[:, [3, 0, 1, 2]]
    par = [-1] * J
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            par[c] = i
    return [n["name"] for n in nodes], fk_frames(par, pos, rot), T


# ----------------------------------------------------------------------------- skeleton reduction
def reduce_skeleton(parents, pos, rot, keep):
    """Keep only joints whose index is in `keep`; dropped joints between kept ones are composed in."""
    keepset = set(keep)
    order = [i for i in range(len(parents)) if i in keepset]
    new_idx = {j: k for k, j in enumerate(order)}
    op, oq, opar = [], [], []
    for j in order:
        chain = [j]
        p = parents[j]
        while p >= 0 and p not in keepset:
            chain.append(p)
            p = parents[p]
        opar.append(new_idx.get(p, -1))
        cp, cq = pos[:, chain[-1]], rot[:, chain[-1]]
        for c in reversed(chain[:-1]):
            cp, cq = cp + qrot(cq, pos[:, c]), qmul(cq, rot[:, c])
        op.append(cp)
        oq.append(cq)
    return order, opar, np.stack(op, 1), np.stack(oq, 1)


# ----------------------------------------------------------------------------- conversion core
def convert(npz, trim, fit, keep_names, recenter=True):
    """-> dict(names, parents, rest (loc, rot_xyzw, scale), rot_xyzw, pos{j}, world_expected, root stats)
    fit = dict(s, yaw, t0, rest_loc_override=None)"""
    d = np.load(npz, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    parents = d["parents"].astype(int)
    pos = d["anim_local_pos"].astype(float)[trim:]
    rot = qnorm(d["anim_local_rot"].astype(float)[trim:])
    rest_pos = d["rest_local_pos"].astype(float)[None]
    rest_rot = qnorm(d["rest_local_rot"].astype(float))[None]
    if len(pos) < 2:
        raise Skip("too_short_after_trim")
    full_world = fk_frames(parents, pos, rot)  # NPZ frame, source units
    keep = [names.index(n) for n in keep_names]
    if len(keep) != len(keep_names):
        raise Skip("joint_missing")
    order, opar, kp, kq = reduce_skeleton(parents, pos, rot, keep)
    _, _, rp, rq = reduce_skeleton(parents, rest_pos, rest_rot, keep)
    s, Rq, t0 = fit["s"], qyaw(fit["yaw"]), np.asarray(fit["t0"])
    roots = [k for k, p in enumerate(opar) if p < 0]
    kp = kp * s
    rp = rp * s
    for k in roots:
        kp[:, k] = qrot(Rq, kp[:, k]) + t0
        rp[:, k] = qrot(Rq, rp[:, k]) + t0
        kq[:, k] = qmul(Rq, kq[:, k])
        rq[:, k] = qmul(Rq, rq[:, k])
    kq = qcontinuous(kq)
    expected = {names[j]: qrot(Rq, full_world[:, j] * s) + t0 for j in order}
    dxz = float(np.linalg.norm(kp[0, roots[0]][[0, 2]] - rp[0, roots[0]][[0, 2]]))
    dy = float(abs(kp[0, roots[0], 1] - rp[0, roots[0], 1]))
    if recenter:  # start the clip on the rest-pose root in x/z; the path itself (root motion) is untouched
        shift = rp[0, roots[0]] - kp[0, roots[0]]
        shift[1] = 0.0
        for k in roots:
            kp[:, k] += shift
        expected = {n: v + shift for n, v in expected.items()}
    return dict(names=[names[j] for j in order], parents=opar, rest_pos=rp[0], rest_rot=rq[0], pos=kp, rot=kq,
                expected=expected, roots=roots, root_dy=dy, root_dxz=dxz)


def write_verify(f, args, c, height, names, parents, loc, rot_xyzw0, scale, fps, rot, moving, act, extras):
    """Write the clip (to a temp file in a dry run), read it back and compare its joint world positions with
    the source forward kinematics. The file is removed on a dry run or when the error exceeds --tol."""
    target = Path(tempfile.mkdtemp()) / "clip.glb" if args.dry_run else f
    write_clip(target, names, parents, loc, rot_xyzw0, scale, fps, rot, moving, act, extras)
    names2, W, _ = read_clip_world(target)
    e = max(np.abs(W[:, k] - c["expected"][n]).max() for k, n in enumerate(names2)) / height
    if args.dry_run or e > args.tol:
        target.unlink(missing_ok=True)
        if args.dry_run:
            target.parent.rmdir()
    if e > args.tol:
        raise Skip(f"verify_failed({e:.2g})")
    return f"{e:.2e}"


def root_stats(c):
    r = c["pos"][:, c["roots"][0]]
    step = np.linalg.norm(np.diff(r[:, [0, 2]], axis=0), axis=1)
    return float(step.sum()), float(np.linalg.norm(r[-1, [0, 2]] - r[0, [0, 2]]))


def safe_name(s, used):
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._") or "clip"
    name, k = base, 2
    while name.lower() in used:
        name, k = f"{base}_{k}", k + 1
    used.add(name.lower())
    return name


def captions(row):
    out = []
    for k in ("caption", "detail", "generic"):
        v = (row.get(k) or "").strip()
        if v and v not in out:
            out.append(v)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default=str(ROOT / "data" / "inventory" / "clips.csv"))
    ap.add_argument("--report", default=str(ROOT / "data" / "clips_report.csv"))
    ap.add_argument("--clips-dir", default=str(ROOT / "data" / "clips"))
    ap.add_argument("--datasets", default="objaverse,mixamo")
    ap.add_argument("--tol", type=float, default=1e-4, help="max joint position error / body height")
    ap.add_argument("--keep-start-xz", action="store_true",
                    help="do not move each clip's starting root x/z onto the rest-pose root")
    ap.add_argument("--max-root-offset", type=float, default=2.0,
                    help="skip a clip whose root starts more than this many body heights above/below the rest-pose root "
                         "(the clip's root path is then in another frame than the rest pose)")
    ap.add_argument("--limit", type=int, default=0, help="clips per dataset (trial)")
    ap.add_argument("--ids", help="comma list of rig ids (objaverse) to process")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="convert and verify in memory only for the check; write nothing")
    args = ap.parse_args()
    if not Path(args.inventory).exists():
        sys.exit("run scripts/dataset/inventory.py first")

    rows = [r for r in csv.DictReader(open(args.inventory, newline="")) if r["usable"] == "1"]
    want = set(args.datasets.split(","))
    report = []

    def rep(r, status, reason="", out="", frames="", err="", rel="", dxz=""):
        report.append(dict(dataset=r["dataset"], rig=r["object_type"], clip=r["clip"], status=status, reason=reason,
                           out=out, frames=frames, verify_err=err, root_dy_rel=rel, root_dxz_rel=dxz))

    # ---------------------------------------------------------------- objaverse
    if "objaverse" in want:
        by_rig = collections.defaultdict(list)
        for r in rows:
            if r["dataset"] == "objaverse" and (not args.ids or r["object_type"] in args.ids.split(",")):
                by_rig[r["object_type"]].append(r)
        rigs = sorted(by_rig)[:args.limit] if args.limit else sorted(by_rig)
        print(f"objaverse: {sum(len(by_rig[k]) for k in rigs)} clips on {len(rigs)} rigs", file=sys.stderr)

        def do_rig(rid):
            clips = by_rig[rid]
            cdir = CHARS / f"objaverse_{rid}"
            res, meta = [], {}
            try:
                model = model_skeleton(cdir / "model.glb")
                first = np.load(clips[0]["path"], allow_pickle=False)
                names = [str(n) for n in first["names"]]
                if set(names) != set(model["names"]):
                    raise Skip("joint_set_mismatch")
                P = fk_rest(first["parents"].astype(int), first["rest_local_pos"].astype(float), first["rest_local_rot"].astype(float))
                gi = [model["names"].index(n) for n in names]
                s, R, _ = umeyama(P, model["world"][gi])
                if np.linalg.det(R) < 0 or abs(R[1, 1] - 1) > 1e-4:
                    raise Skip("rig_fit_not_yaw_only")
                yaw = yaw_of(R)
                Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
                t0 = model["world"][gi].mean(0) - s * Ry @ P.mean(0)
                resid = np.abs(model["world"][gi] - (s * P @ Ry.T + t0)).max()
                height = float(np.ptp(model["world"][:, 1])) or 1.0
                if resid > args.tol * height:
                    raise Skip(f"rig_fit_residual({resid:.2g})")
                fit = dict(s=s, yaw=yaw, t0=t0)
            except Skip as e:
                return [(r, "skipped", str(e), "", "", "") for r in clips], {}
            except Exception as e:
                return [(r, "failed", f"{type(e).__name__}: {e}", "", "", "") for r in clips], {}
            used = set()
            adir = cdir / "anims"
            for r in clips:
                try:
                    trim = int(r["trim"] or 0)
                    c = convert(r["path"], trim, fit, model["names"], not args.keep_start_xz)
                    rel = c["root_dy"] / height
                    if rel > args.max_root_offset:
                        res.append((r, "skipped", f"root_frame_mismatch({rel:.3g})", "", "", "", f"{rel:.3f}"))
                        continue
                    # nodes of the clip file = model.glb skeleton (same order); map converted arrays onto it
                    m = {n: k for k, n in enumerate(c["names"])}
                    J = len(model["names"])
                    order = [m[n] for n in model["names"]]
                    rot = c["rot"][:, order][..., [1, 2, 3, 0]]
                    pos = c["pos"][:, order]
                    moving = {j: pos[:, j] for j in range(J) if model["parents"][j] < 0 or
                              np.abs(pos[:, j] - model["loc"][j]).max() > 1e-6 * height}
                    act = safe_name(r["action"] or r["clip"], used)
                    fps = int(float(r["fps"]))
                    rt, rn = root_stats(c)
                    f = adir / f"{act}.glb"
                    if not args.dry_run:
                        adir.mkdir(parents=True, exist_ok=True)
                    err = write_verify(f, args, c, height, model["names"], model["parents"], model["loc"], model["rot_xyzw"],
                                       model["scale"], fps, rot, moving, act, dict(source_clip=r["clip"], fps=fps, units="m"))
                    meta[act] = dict(file=f"{act}.glb", source_clip=r["clip"], fps=fps, frames=int(len(rot)),
                                     duration_s=round(len(rot) / fps, 3), loop=bool(int(r["loop_guess"] or 0)),
                                     loop_gap_deg=float(r["loop_gap_deg"]), root_travel_m=round(rt, 3), root_net_m=round(rn, 3),
                                     start_offset_m=0.0 if args.keep_start_xz else round(c["root_dxz"], 4),
                                     trimmed_frames=trim, motion_captions=captions(r))
                    res.append((r, "ok", "", "(dry run)" if args.dry_run else str(f.relative_to(ROOT)), len(rot), err, f"{rel:.3f}", f"{c['root_dxz'] / height:.3f}"))
                except Skip as e:
                    res.append((r, "skipped", str(e), "", "", ""))
                except Exception as e:
                    res.append((r, "failed", f"{type(e).__name__}: {e}", "", "", ""))
            if meta and not args.dry_run:
                prev = adir / "anims.json"
                if prev.exists():        # keep clips made by scripts/rigging/retarget.py (rt__ prefix)
                    meta.update({k: v for k, v in json.loads(prev.read_text()).items() if k.startswith("rt__")})
                prev.write_text(json.dumps(meta, indent=1))
            return res, meta

        with ThreadPoolExecutor(args.workers) as ex:
            for i, (res, _) in enumerate(ex.map(do_rig, rigs), 1):
                for r, *rest in res:
                    rep(r, *rest)
                if i % 500 == 0:
                    print(f"  {i}/{len(rigs)} rigs", file=sys.stderr)

    # ---------------------------------------------------------------- mixamo
    if "mixamo" in want:
        clips = [r for r in rows if r["dataset"] == "mixamo"]
        clips = clips[:args.limit] if args.limit else clips
        chars = sorted(p for p in CHARS.glob("mixamo_*/model.glb"))
        print(f"mixamo: {len(clips)} clips, {len(chars)} characters", file=sys.stderr)
        if clips and chars:
            ref_model = model_skeleton(chars[0])
            keep = ref_model["names"]
            first = np.load(clips[0]["path"], allow_pickle=False)
            names = [str(n) for n in first["names"]]
            face = json.loads((SRC / "export" / "mixamo" / "face_joint_names.json").read_text())["mixamo"]
            Pr = fk_rest(first["parents"].astype(int), first["rest_local_pos"].astype(float), first["rest_local_rot"].astype(float))
            v = Pr[names.index(face["l_hip"]["raw"])] - Pr[names.index(face["r_hip"]["raw"])]
            fwd = np.cross(v, [0, 1, 0])
            deg = math.degrees(-math.atan2(fwd[0], fwd[2]))
            yaw = math.radians(90 * round(deg / 90) if abs(deg - 90 * round(deg / 90)) <= 5 else deg)
            fit = dict(s=MIXAMO_CM_TO_M, yaw=yaw, t0=np.zeros(3))
            height = float(np.ptp(Pr[:, 1])) * MIXAMO_CM_TO_M
            hips_ref = float(Pr[names.index(keep[[i for i, p in enumerate(ref_model["parents"]) if p < 0][0]])][1]) * MIXAMO_CM_TO_M
            cdir = Path(args.clips_dir) / "mixamo"
            # reference skeleton (rest) in metres
            used, meta = set(), {}

            def do_clip(r):
                try:
                    c = convert(r["path"], int(r["trim"] or 0), fit, keep, not args.keep_start_xz)
                    if c["root_dy"] / height > args.max_root_offset:
                        return r, None, None, None, f"root_frame_mismatch({c['root_dy'] / height:.3g})"
                    J = len(c["names"])
                    rot = c["rot"][..., [1, 2, 3, 0]]
                    moving = {j: c["pos"][:, j] for j in range(J) if c["parents"][j] < 0
                              or np.abs(c["pos"][:, j] - c["rest_pos"][j]).max() > 1e-6 * height}
                    return r, c, rot, moving, None
                except Skip as e:
                    return r, None, None, None, str(e)
                except Exception as e:
                    return r, None, None, None, f"failed:{type(e).__name__}: {e}"

            if not args.dry_run:
                cdir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(args.workers) as ex:
                for i, (r, c, rot, moving, why) in enumerate(ex.map(do_clip, clips), 1):
                    if why:
                        rep(r, "failed" if why.startswith("failed") else "skipped", why)
                        continue
                    act = safe_name(r["clip"], used)
                    fps = int(float(r["fps"]))
                    rt, rn = root_stats(c)
                    f = cdir / f"{act}.glb"
                    err = ""
                    try:
                        err = write_verify(f, args, c, height, c["names"], c["parents"], c["rest_pos"], c["rest_rot"][..., [1, 2, 3, 0]],
                                           np.ones((len(c["names"]), 3)), fps, rot, moving, act,
                                           dict(source_clip=r["clip"], fps=fps, units="m", skeleton="mixamo_core_ref"))
                    except Skip as e:
                        rep(r, "skipped", str(e))
                        continue
                    meta[act] = dict(file=f"{act}.glb", source_clip=r["clip"], fps=fps, frames=int(len(rot)),
                                     duration_s=round(len(rot) / fps, 3), loop=bool(int(r["loop_guess"] or 0)),
                                     loop_gap_deg=float(r["loop_gap_deg"]), root_travel_m=round(rt, 3), root_net_m=round(rn, 3),
                                     start_offset_m=0.0 if args.keep_start_xz else round(c["root_dxz"], 4),
                                     trimmed_frames=int(r["trim"] or 0), motion_captions=captions(r))
                    rep(r, "ok", "", "(dry run)" if args.dry_run else (str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)), len(rot), err,
                        f"{c['root_dy'] / height:.3f}", f"{c['root_dxz'] / height:.3f}")
                    if i % 500 == 0:
                        print(f"  {i}/{len(clips)} clips", file=sys.stderr)
            if not args.dry_run and meta:
                dropped = [n for n in names if n not in set(keep)]
                (cdir / "clips.json").write_text(json.dumps(dict(
                    _skeleton=dict(joints=keep, units="m", reference_hips_height_m=round(hips_ref, 5),
                                   note="core skeleton of the Mixamo rig at reference proportions; multiply root translation by "
                                        "root_scale from the character's anims.json",
                                   dropped_joints=dropped), **meta), indent=1))
                for p in chars:
                    sk = model_skeleton(p)
                    root = [i for i, q in enumerate(sk["parents"]) if q < 0][0]
                    h = float(sk["world"][root][1])
                    adir = p.parent / "anims"
                    adir.mkdir(exist_ok=True)
                    prev = adir / "anims.json"
                    keep_rt = {k: v for k, v in json.loads(prev.read_text()).items() if k.startswith("rt__")} if prev.exists() else {}
                    prev.write_text(json.dumps({"_shared": dict(
                        clips_index=str(Path("../../..") / "clips" / "mixamo" / "clips.json"), clips_dir="../../../clips/mixamo",
                        root_scale=round(h / hips_ref, 6), character_hips_height_m=round(h, 4),
                        reference_hips_height_m=round(hips_ref, 5)), **keep_rt}, indent=1))

    # ---------------------------------------------------------------- report
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, ["dataset", "rig", "clip", "status", "reason", "out", "frames", "verify_err", "root_dy_rel", "root_dxz_rel"])
        w.writeheader()
        w.writerows(report)
    st = collections.Counter((r["dataset"], r["status"]) for r in report)
    print(f"\nreport: {args.report}{'  (dry run: nothing written)' if args.dry_run else ''}")
    print("status:", {f"{a}:{b}": n for (a, b), n in sorted(st.items())})
    print("reasons:", dict(collections.Counter(r["reason"].split("(")[0] for r in report if r["reason"])))
    errs = [float(r["verify_err"]) for r in report if r["verify_err"]]
    if errs:
        print(f"max verify error (relative to body height): {max(errs):.2e}  over {len(errs)} clips")


if __name__ == "__main__":
    main()
