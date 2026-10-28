#!/usr/bin/env python3
"""Joint tokens for Lottie animations, in the same format as the 3D clips of scripts/motion/motion_sem_features.py, so one semantic motion model can read both.

A Lottie layer is a joint. Its pivot (the anchor point, which is also where the layer rotates) is followed through the composition by 2D forward
kinematics: layer matrix M = T(position) R(rotation) S(scale) T(-anchor), world = parent's world matrix times M, nested precomps compose through the
precomp layer, layer start/stretch (st, sr) are honoured; time remapping, 3D layers and expressions are not (expressions: the animation is flagged). Y is flipped
(Lottie's y points down) and every length is divided by the body size S (the largest extent of the joints' first-frame positions), so the tokens do not depend on
the comp size. The channels are the planar versions of the 3D ones:

  per joint and frame (4): height above the lowest point reached / S, |x - centroid x| / S, vertical speed (S per second), |horizontal speed| (S per second)
  per joint, static (8):   rest height, rest radius, tree depth, children, leaf, subtree size, bone length / S, root flag     (same formulas as 3D)
  per frame, whole body (5): centroid height, centroid horizontal speed, centroid vertical speed, yaw rate (always 0 in 2D), mean joint radius
  per animation (8):       log duration, log time-warp, joint count / 100, leaf fraction, height extent / S, width extent / S, 0 (no second horizontal axis), fps / 30

The only systematic 2D markers are the second horizontal extent (0) and the yaw rate (0).

Which joints are tokens (defaults chosen by measurement on 600 relevant animations): animated shape groups are joints too (a shape layer's groups often carry
the limbs); the whole parent tree of anything that moves is kept, static parts of a rig included, backgrounds not parented to it are not; layers without a parent
hang from the nearest pivot so there is one tree; at most --max-joints (96). Layers only: 69% of the animations give tokens; with groups: 85%. The skeleton shape
stays separable from the 3D clips (AUC 0.996-1.000 for every variant tried: layers only, + groups, + whole tree, + nearest-pivot or virtual root, every layer),
so the difference to 3D skeletons is structural (median 10 joints vs 49, leaf share 0.5-0.6 vs 0.21, bones 0.2-0.3 of body size vs 0.07), not an artefact.
Output keys match data/embeddings/sem_feats.npz (joint, static, offsets, glob, scal, n_valid).

    python scripts/lottie2d/lottie_tokens.py selftest                                   # forward-kinematics checks on synthetic animations
    python scripts/lottie2d/lottie_tokens.py build --shards 3,5 --limit 300 --relevant  # tokens for the Qwen-relevant rows of some shards -> data/embeddings/lottie_feats.npz
    python scripts/lottie2d/lottie_tokens.py build --json-dir <folder of .json>         # tokens for loose Lottie files
    python scripts/lottie2d/lottie_tokens.py build-all --shards all --workers 6         # every Qwen-relevant animation, one worker per corpus shard -> lottie_feats.npz
    python scripts/lottie2d/lottie_tokens.py demo --shards 3,5,7 --limit 200 --relevant # sample export for viewer/lottie_tokens.html -> data/lottie_tokens_demo/
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "lottie", Path(__file__).resolve().parent / "lottie"):
    if _p.exists():
        sys.path.insert(0, str(_p))
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
from lottie_curves import sample_property  # noqa: E402
from motion_sem_features import T_FIX, resample, to_timeline  # noqa: E402

MAX_FRAMES = 1024


# ----------------------------------------------------------------------------- property sampling
def _vec(k) -> np.ndarray | None:
    if isinstance(k, (int, float)):
        return np.array([float(k)])
    if isinstance(k, (list, tuple)) and k and not isinstance(k[0], dict):
        try:
            return np.array([float(x) for x in k])
        except (TypeError, ValueError):
            return None
    return None


def values(prop, local: np.ndarray, default, dim: int) -> np.ndarray:
    """(F, dim): an animated or static transform property at layer-local frames; missing or malformed -> default."""
    F = len(local)
    out = np.tile(np.array(default, float)[:dim], (F, 1))
    if not isinstance(prop, dict):
        return out
    if prop.get("a") == 1:
        v = sample_property(prop, local)
        if v is not None and np.isfinite(v).any():
            v = v.copy()
            for d in range(v.shape[1]):   # fill NaNs by holding the nearest valid value
                col = v[:, d]
                ok = np.isfinite(col)
                if ok.any():
                    col[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), col[ok])
            n = min(dim, v.shape[1])
            out[:, :n] = v[:, :n]
        return out
    s = _vec(prop.get("k"))
    if s is not None:
        n = min(dim, len(s))
        out[:, :n] = s[:n]
    return out


def position(ks: dict, local: np.ndarray) -> np.ndarray:
    p = ks.get("p")
    if isinstance(p, dict) and p.get("s") is True:   # split dimensions
        return np.stack([values(p.get("x"), local, [0], 1)[:, 0], values(p.get("y"), local, [0], 1)[:, 0]], 1)
    return values(p, local, [0, 0, 0], 2)


def layer_matrix(ks: dict, local: np.ndarray) -> np.ndarray:
    """(F, 3, 3): T(p) R(r) S(s) T(-a)."""
    p = position(ks, local)
    a = values(ks.get("a"), local, [0, 0, 0], 2)
    s = values(ks.get("s"), local, [100, 100, 100], 2) / 100.0
    r = np.deg2rad(values(ks.get("r") if ks.get("r") is not None else ks.get("rz"), local, [0], 1)[:, 0])
    c, sn = np.cos(r), np.sin(r)
    M = np.zeros((len(local), 3, 3))
    M[:, 0, 0], M[:, 0, 1] = c * s[:, 0], -sn * s[:, 1]
    M[:, 1, 0], M[:, 1, 1] = sn * s[:, 0], c * s[:, 1]
    M[:, 0, 2] = p[:, 0] - (M[:, 0, 0] * a[:, 0] + M[:, 0, 1] * a[:, 1])
    M[:, 1, 2] = p[:, 1] - (M[:, 1, 0] * a[:, 0] + M[:, 1, 1] * a[:, 1])
    M[:, 2, 2] = 1.0
    return M


# ----------------------------------------------------------------------------- forward kinematics
def lottie_fk(lj: dict, groups: bool = True) -> dict | None:
    """Joints (one per layer, precomps expanded): names, parents, pivot world positions (F, J, 2) with y up, fps, frame count. None if unusable."""
    fr = float(lj.get("fr") or 30)
    ip, op = float(lj.get("ip", 0) or 0), float(lj.get("op", 0) or 0)
    if op - ip < 2:
        return None
    frames = np.arange(ip, op, 1.0)
    if len(frames) > MAX_FRAMES:
        frames = np.linspace(ip, op - 1, MAX_FRAMES)
    assets = {a.get("id"): a for a in lj.get("assets") or [] if isinstance(a, dict) and a.get("layers")}
    names, parents, pos, kinds = [], [], [], []

    def walk_shapes(items, world, parent_joint, local, depth):
        """Shape groups inside a shape layer: a group whose own transform is animated becomes a joint (parent = the nearest enclosing joint); static groups
        only pass their transform on."""
        if depth > 12 or not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict) or it.get("ty") != "gr":
                continue
            sub = it.get("it") or []
            tr = next((x for x in sub if isinstance(x, dict) and x.get("ty") == "tr"), None)
            if tr is None:
                walk_shapes(sub, world, parent_joint, local, depth + 1)
                continue
            Wg = world @ layer_matrix(tr, local)
            if any(isinstance(tr.get(k), dict) and tr[k].get("a") == 1 for k in ("p", "a", "s", "r")):
                p = position(tr, local)
                pivot = np.einsum("fij,fj->fi", world, np.concatenate([p, np.ones((len(p), 1))], 1))[:, :2]
                names.append(str(it.get("nm", ""))); parents.append(parent_joint); pos.append(pivot); kinds.append(1)
                walk_shapes(sub, Wg, len(names) - 1, local, depth + 1)
            else:
                walk_shapes(sub, Wg, parent_joint, local, depth + 1)

    def walk(layers, comp_frames, root_world, root_joint, depth):
        if depth > 6 or not isinstance(layers, list):
            return
        by_ind = {l.get("ind"): l for l in layers if isinstance(l, dict) and l.get("ind") is not None}
        done: dict = {}   # layer ind -> (world matrix, joint index)

        def resolve(l, stack):
            ind = l.get("ind")
            if ind in done:
                return done[ind]
            ks = l.get("ks") or {}
            st, sr = float(l.get("st", 0) or 0), float(l.get("sr", 1) or 1) or 1.0
            local = (comp_frames - st) / sr
            par = by_ind.get(l.get("parent"))
            if par is not None and l.get("parent") not in stack and par is not l:
                pw, pj = resolve(par, stack | {ind})
            else:
                pw, pj = root_world, root_joint
            M = layer_matrix(ks, local)
            W = pw @ M
            p = position(ks, local)
            pivot = np.einsum("fij,fj->fi", pw, np.concatenate([p, np.ones((len(p), 1))], 1))[:, :2]   # the layer's origin in the composition
            j = len(names)
            names.append(str(l.get("nm", "")))
            parents.append(pj)
            pos.append(pivot)
            kinds.append(0)
            done[ind] = (W, j)
            if groups and l.get("ty") == 4:
                walk_shapes(l.get("shapes"), W, j, local, 0)
            return done[ind]

        for l in layers:
            if not isinstance(l, dict):
                continue
            if l.get("ind") is None:
                l = {**l, "ind": id(l)}
                by_ind[l["ind"]] = l
            W, j = resolve(l, frozenset())
            if l.get("ty") == 0 and l.get("refId") in assets:   # precomp: its layers live in the precomp layer's coordinates and time
                st, sr = float(l.get("st", 0) or 0), float(l.get("sr", 1) or 1) or 1.0
                walk(assets[l["refId"]]["layers"], (comp_frames - st) / sr, W, j, depth + 1)

    walk(lj.get("layers"), frames, np.tile(np.eye(3), (len(frames), 1, 1)), -1, 0)
    if len(names) < 2:
        return None
    P = np.stack(pos, 1)                       # (F, J, 2), y down
    P[..., 1] *= -1.0                          # y up
    if not np.isfinite(P).all() or np.abs(P).max() > 1e7:
        return None
    return dict(names=names, parents=np.array(parents, int), pos=P, fps=fr, frames=len(frames), kinds=np.array(kinds, int))


# ----------------------------------------------------------------------------- tokens
def select_joints(parents: np.ndarray, P: np.ndarray, max_joints: int, keep: str = "component"):
    """Which joints become tokens (sorted; parents before children).
    moving:    joints that move and every ancestor of one (at most max_joints, the most moving first);
    component: the whole parent tree of anything that moves (static parts of a rig stay), filled up to max_joints by depth;
    all:       every joint up to max_joints (moving ones first)."""
    n = len(parents)
    move = np.ptp(P, axis=0).max(-1)                                     # (J,) travel of each pivot
    thr = 1e-3 * max(float(np.ptp(P[0], axis=0).max()), 1e-6)
    keepset: set = set()
    for j in np.argsort(-move):
        if move[j] <= thr and keepset:
            break
        chain, a = [], int(j)
        while a >= 0 and a not in keepset:
            chain.append(a)
            a = int(parents[a])
        if len(keepset) + len(chain) > max_joints:
            continue
        keepset.update(chain)
    if keep == "moving" or not keepset:
        return sorted(keepset)
    root = np.arange(n)
    for j in range(n):                                                    # topmost ancestor of every joint
        a, k = j, 0
        while parents[a] >= 0 and k < n:
            a, k = int(parents[a]), k + 1
        root[j] = a
    depth = np.zeros(n, int)
    for j in range(n):
        a, k = j, 0
        while parents[a] >= 0 and k < n:
            a, k = int(parents[a]), k + 1
        depth[j] = k
    live = {int(root[j]) for j in keepset}
    cand = [j for j in range(n) if j not in keepset and (keep == "all" or int(root[j]) in live)]
    for j in sorted(cand, key=lambda j: (depth[j], j)):
        if len(keepset) >= max_joints:
            break
        if parents[j] < 0 or int(parents[j]) in keepset:
            keepset.add(j)
    return sorted(keepset)


def tokens_from_fk(fk: dict, max_joints: int = 96, keep: str = "component", roots: str = "nearest", info: dict | None = None):
    """-> (joint (J,128,4) f16, static (J,8) f16, glob (128,5) f16, scal (8,) f32, n_valid) or None."""
    par0, P0, fps = fk["parents"], fk["pos"], fk["fps"]
    idx = select_joints(par0, P0, max_joints, keep)
    if len(idx) < 2:
        return None
    remap = {j: k for k, j in enumerate(idx)}
    parents = np.array([remap.get(int(par0[j]), -1) if par0[j] >= 0 else -1 for j in idx])
    # a kept joint's parent is always kept (ancestors are included), so no joint is orphaned except real roots
    Wp = P0[:, idx]                                                      # (T, J, 2) x, y(up)
    if roots != "none" and (parents < 0).sum() > 1:
        rest0 = Wp[0]
        rts = [j for j in range(len(parents)) if parents[j] < 0]
        if roots == "virtual":                                           # one static joint at the first-frame centroid becomes the parent of every root
            Wp = np.concatenate([Wp, np.repeat(rest0.mean(0)[None, None], Wp.shape[0], 0)], 1)
            parents = np.append(parents, -1)
            parents[rts] = len(parents) - 1
        else:                                                            # nearest: the biggest tree stays the root, the others hang from the nearest pivot outside themselves
            def sub(j):
                return {k for k in range(len(parents)) if any(a == j for a in _anc(parents, k))}
            sizes = {r: len(sub(r)) for r in rts}
            main = max(rts, key=lambda r: sizes[r])
            for r in sorted(rts, key=lambda r: -sizes[r]):
                if r == main:
                    continue
                own = sub(r)
                others = [k for k in range(len(parents)) if k not in own]
                parents[r] = min(others, key=lambda k: np.linalg.norm(rest0[k] - rest0[r]))
    if info is not None:
        info["parents"] = parents.tolist()                              # in kept-joint index space, after the roots are joined
    T0, J = Wp.shape[:2]
    rest = Wp[0]
    ext = np.ptp(rest, axis=0)
    S = max(float(ext.max()), float(np.ptp(Wp, axis=(0, 1)).max()) * 0.25, 1e-6)
    y0 = float(Wp[..., 1].min())
    dt = np.full(max(T0 - 1, 1), 1.0 / fps)
    c = Wp.mean(1)
    h = (Wp[..., 1] - y0) / S
    rad = np.abs(Wp[..., 0] - c[:, None, 0]) / S

    def speed(x):                                                         # (T, ...) -> same shape, per second
        if T0 < 2:
            return np.zeros_like(x)
        v = np.diff(x, axis=0) / dt.reshape((-1,) + (1,) * (x.ndim - 1))
        return np.concatenate([v[:1], v], 0)

    vy = speed(Wp[..., 1]) / S
    vh = np.abs(speed(Wp[..., 0])) / S
    joint = np.clip(np.stack([h, rad, vy, vh], -1), -20, 20)
    cy = (c[:, 1] - y0) / S
    cvh = np.abs(speed(c[:, 0])) / S
    cvy = speed(c[:, 1]) / S
    glob = np.clip(np.stack([cy, cvh, cvy, np.zeros(T0), rad.mean(1)], -1), -20, 20)
    jt, warp = to_timeline(joint, T0) if T0 <= T_FIX else (resample(joint, T_FIX), T0 / T_FIX)
    gt, _ = to_timeline(glob, T0) if T0 <= T_FIX else (resample(glob, T_FIX), 1.0)
    n_valid = min(T0, T_FIX)
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
    bone = np.array([np.linalg.norm(rest[j] - rest[parents[j]]) if parents[j] >= 0 else 0.0 for j in range(J)]) / S
    rc = rest.mean(0)
    stat = np.stack([(rest[:, 1] - rest[:, 1].min()) / S, np.abs(rest[:, 0] - rc[0]) / S, depth / max(depth.max(), 1), np.minimum(kids, 4) / 4,
                     (kids == 0).astype(float), sub / J, np.minimum(bone, 2.0), (parents < 0).astype(float)], 1)
    scal = np.array([math.log(T0 / fps + 1e-3), math.log(warp), J / 100.0, (kids == 0).mean(), ext[1] / S, ext[0] / S, 0.0, fps / 30.0])
    out = (jt.transpose(1, 0, 2).astype(np.float16), stat.astype(np.float16), gt.astype(np.float16), scal.astype(np.float32), n_valid)
    if not all(np.isfinite(np.asarray(a, float)).all() for a in out[:4]):
        return None
    return out


def _anc(parents, k):
    out, n = [k], 0
    while parents[k] >= 0 and n < len(parents):
        k = int(parents[k]); out.append(k); n += 1
    return out


def lottie_tokens(lj: dict, max_joints: int = 96, groups: bool = True, keep: str = "component", roots: str = "nearest"):
    fk = lottie_fk(lj, groups)
    return tokens_from_fk(fk, max_joints, keep, roots) if fk else None


# ----------------------------------------------------------------------------- self test
def _rot(deg0, deg1, f1):
    return {"a": 1, "k": [{"t": 0, "s": [deg0], "e": [deg1], "i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}}, {"t": f1, "s": [deg1]}]}


def selftest() -> None:
    def layer(ind, p, parent=None, r=None, s=None, a=None, **kw):
        ks = {"p": {"a": 0, "k": list(p) + [0]}, "a": {"a": 0, "k": (a or [0, 0]) + [0]}, "s": {"a": 0, "k": (s or [100, 100]) + [100]}, "r": r or {"a": 0, "k": 0}}
        d = {"ind": ind, "ty": 3, "nm": f"L{ind}", "ks": ks}
        if parent is not None:
            d["parent"] = parent
        d.update(kw)
        return d

    base = {"fr": 30, "ip": 0, "op": 31, "w": 400, "h": 400}
    # 1. parent rotates 0 -> 90 deg about its pivot (100,100); child at (50,0) in the parent's frame: (150,100) at frame 0, (100,150) at frame 30 (y down)
    fk = lottie_fk({**base, "layers": [layer(1, [100, 100], r=_rot(0, 90, 30)), layer(2, [50, 0], parent=1)]})
    y = lambda v: -v   # lottie_fk returns y up
    assert np.allclose(fk["pos"][0, 1], [150, y(100)], atol=1e-6) and np.allclose(fk["pos"][30, 1], [100, y(150)], atol=1e-6), fk["pos"][[0, 30], 1]
    # 2. parent scale 200%: the child offset doubles; the parent's anchor does not move its pivot
    fk = lottie_fk({**base, "layers": [layer(1, [100, 100], s=[200, 200], a=[7, 9]), layer(2, [10, 0], parent=1)]})
    # the child's pivot = parent_world @ (10, 0): M = T(p) S(2) T(-a): world(10,0) = p + 2*((10,0) - a) = (100 + 2*(10-7), 100 + 2*(0-9))
    assert np.allclose(fk["pos"][0, 1], [100 + 2 * (10 - 7), y(100 + 2 * (0 - 9))], atol=1e-6), fk["pos"][0, 1]
    # 3. precomp: layer at (10,20) containing a layer at (5,5) -> world (15,25); the precomp layer's rotation (90 deg) turns the inner offset
    comp = {"id": "c", "layers": [layer(1, [5, 5])]}
    fk = lottie_fk({**base, "assets": [comp], "layers": [layer(1, [10, 20], ty=0, refId="c")]})
    assert np.allclose(fk["pos"][0, 1], [15, y(25)], atol=1e-6), fk["pos"][0]
    fk = lottie_fk({**base, "assets": [comp], "layers": [layer(1, [10, 20], ty=0, refId="c", r={"a": 0, "k": 90})]})
    assert np.allclose(fk["pos"][0, 1], [10 - 5, y(20 + 5)], atol=1e-6), fk["pos"][0]   # R(90): (5,5) -> (-5,5)
    # 4. tokens: shapes, finite, a static joint with no moving descendant is dropped, a moving child keeps its static root
    lj = {**base, "layers": [layer(1, [100, 100]), layer(2, [50, 0], parent=1, r=_rot(0, 90, 30)), layer(3, [10, 10], parent=1), layer(4, [0, 40], parent=2, r=_rot(0, 45, 30))]}
    out = lottie_tokens(lj)
    assert out is not None and out[0].shape[1:] == (128, 4) and out[1].shape[1] == 8 and out[2].shape == (128, 5) and out[3].shape == (8,)
    # 5. shape-group joints: a group rotating 0 -> 90 deg about its own anchor inside a shape layer at (100,100); the group's pivot is at its position (30,0) in the layer
    grp = {"ty": "gr", "nm": "arm", "it": [{"ty": "tr", "p": {"a": 0, "k": [30, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": _rot(0, 90, 30)},],}
    inner = {"ty": "gr", "nm": "hand", "it": [{"ty": "tr", "p": {"a": 0, "k": [20, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}}]}
    grp["it"].insert(0, inner)   # a static child group inside the animated one: it passes the transform on and is not a joint
    fk = lottie_fk({**base, "layers": [layer(1, [100, 100], ty=4, shapes=[grp])]}, groups=True)
    assert fk["names"] == ["L1", "arm"] and fk["parents"].tolist() == [-1, 0], (fk["names"], fk["parents"])
    assert np.allclose(fk["pos"][0, 1], [130, y(100)]) and np.allclose(fk["pos"][30, 1], [130, y(100)]), fk["pos"][[0, 30], 1]   # a group rotates about its own pivot: the pivot does not move
    comp_g = {**base, "layers": [layer(1, [100, 100], ty=4, shapes=[{"ty": "gr", "nm": "outer", "it": [{"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": _rot(0, 90, 30)},
                       {"ty": "gr", "nm": "child", "it": [{"ty": "tr", "p": {"a": 0, "k": [40, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": _rot(0, 10, 30)}]}]}])]}
    fk = lottie_fk(comp_g, groups=True)
    assert fk["names"] == ["L1", "outer", "child"] and fk["parents"].tolist() == [-1, 0, 1]
    assert np.allclose(fk["pos"][30, 2], [100, y(140)], atol=1e-6), fk["pos"][30]   # the child group's pivot rides the outer group's rotation: (40,0) -> (0,40)
    print("selftest passed: rotation about a pivot, scale + anchor, precomp offset and rotation, token shapes;", out[0].shape[0], "joints kept of 4")


# ----------------------------------------------------------------------------- build
def iter_corpus(args):
    import orjson
    import zstandard
    corpus = ROOT.parent / "Datasets" / "LottieAnimation-660K"
    relevant = {}
    if args.relevant:
        for f in sorted(glob.glob(str(ROOT / "data" / "vlm" / "animation" / "shard-*.jsonl.zst"))):
            if Path(f).name.startswith("._"):
                continue
            with open(f, "rb") as fh:
                for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                    r = orjson.loads(line)
                    if r.get("relevant"):
                        relevant[str(r["id"])] = (r["shard"], r.get("body_plan"), r.get("caption_lottie") or "")
    shards = [int(s) for s in args.shards.split(",")]
    rx = re.compile(rb'^\{"id":"(\d+)"')
    for s in shards:
        path = glob.glob(str(corpus / "snapshots" / "*" / "data" / f"metadata-{s:05d}-of-00068.jsonl.zst"))[0]
        n = 0
        with open(path, "rb") as fh:
            for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                m = rx.match(line[:40])
                if not m:
                    continue
                i = m.group(1).decode()
                if args.relevant and i not in relevant:
                    continue
                r = orjson.loads(line)
                lj = r["lottie_json"]
                lj = orjson.loads(lj) if isinstance(lj, str) else lj
                yield i, lj, dict(dataset="lottie", clip=i, object_type=(relevant.get(i) or (0, "", ""))[1] or "", caption=(relevant.get(i) or (0, "", ""))[2] or r.get("caption", ""))
                n += 1
                if args.limit and n >= args.limit:
                    break


def iter_dir(args):
    for f in sorted(Path(args.json_dir).glob("*.json")):
        if f.name.startswith("._"):
            continue
        yield f.stem, json.loads(f.read_text()), dict(dataset="lottie", clip=f.stem, object_type="", caption="")


def cmd_build(args) -> None:
    src = iter_dir(args) if args.json_dir else iter_corpus(args)
    jt, st, gl, sc, nv, meta, skipped = [], [], [], [], [], [], {}
    for k, (i, lj, m) in enumerate(src):
        try:
            r = lottie_tokens(lj, args.max_joints, args.groups, args.keep, args.roots)
        except Exception as e:   # one bad file must not stop a build
            skipped[type(e).__name__] = skipped.get(type(e).__name__, 0) + 1
            continue
        if r is None:
            skipped["unusable"] = skipped.get("unusable", 0) + 1
            continue
        jt.append(r[0]); st.append(r[1]); gl.append(r[2]); sc.append(r[3]); nv.append(r[4])
        meta.append(dict(m, joints=len(r[0]), fps=lj.get("fr"), frames=r[4], duration_s=round(float(np.exp(r[3][0])), 3)))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} animations, {len(jt)} tokenised", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    offs = np.cumsum([0] + [len(a) for a in jt])
    np.savez(out, joint=np.concatenate(jt), static=np.concatenate(st), offsets=offs, glob=np.stack(gl), scal=np.stack(sc), n_valid=np.array(nv))
    cols = ["dataset", "clip", "object_type", "joints", "fps", "frames", "duration_s", "caption"]
    with open(out.with_name(out.stem + "_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(meta)
    print(f"{len(jt)} animations tokenised, {int(offs[-1])} joint tokens, skipped {skipped} -> {out}")


def make_record(i, lj: dict, max_joints: int = 96, groups: bool = True, keep: str = "component", roots: str = "nearest"):
    """The record the viewer shows for one animation: the Lottie JSON, joint pivots per frame (y down, pixels), the kept joints, their tokens.
    Returns (record, status) with status 'ok' or 'no_tokens', or (None, 'no_joints') when there is nothing to show."""
    fk = lottie_fk(lj, groups)
    if fk is None:
        return None, "no_joints"
    info: dict = {}
    tok = tokens_from_fk(fk, max_joints, keep, roots, info)
    sel = select_joints(fk["parents"], fk["pos"], max_joints, keep) if tok is not None else []
    P = fk["pos"].copy()
    P[..., 1] *= -1.0                                         # back to Lottie's y-down pixels
    rec = dict(id=i, lottie=lj, fk=dict(names=fk["names"], parents=fk["parents"].tolist(), pos=np.round(P, 1).tolist(), fps=fk["fps"], frames=fk["frames"],
               ip=lj.get("ip", 0), selected=sel, kinds=fk["kinds"].tolist()),
               tokens=dict(joint=np.round(tok[0].astype(float), 3).tolist(), static=np.round(tok[1].astype(float), 3).tolist(), scal=np.round(tok[3].astype(float), 3).tolist(),
                           n_valid=int(tok[4]), parents=info.get("parents")) if tok else None)
    return rec, ("ok" if tok else "no_tokens")


def cmd_demo(args) -> None:
    """Export sample animations for viewer/lottie_tokens.html (the 'samples' list)."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index, n_ok, n_none = [], 0, 0
    for i, lj, m in iter_corpus(args):
        if len(json.dumps(lj)) > 3_000_000:
            continue
        rec, status = make_record(i, lj, args.max_joints, args.groups, args.keep, args.roots)
        if rec is None or (status == "ok" and n_ok >= args.n) or (status != "ok" and n_none >= args.n_none):
            continue
        (out / f"{i}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index.append(dict(id=i, body_plan=m["object_type"], caption=m["caption"], w=lj.get("w"), h=lj.get("h"), kept=len(rec["fk"]["selected"]), total=len(rec["fk"]["names"]),
                          status=status, frames=rec["fk"]["frames"]))
        n_ok += status == "ok"
        n_none += status != "ok"
        if n_ok >= args.n and n_none >= args.n_none:
            break
    (out / "index.json").write_text(json.dumps(index))
    print(f"{n_ok} tokenised + {n_none} without tokens -> {out}")


def cmd_browser_index(args) -> None:
    """data/analysis/lottie_browser_index.json: every Qwen-relevant animation (tokenised or not) as columns, for the full-dataset mode of the viewer."""
    rel = load_relevant()
    tok = {}
    f = ROOT / "data" / "embeddings" / "lottie_feats_index.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            tok[r["clip"]] = (int(r["joints"]), int(r["frames"]))
    cols = dict(id=[], shard=[], bp=[], subj=[], joints=[], frames=[], cap=[])
    for s in sorted(rel):
        for i, (bp, subj, cap) in rel[s].items():
            j, fr = tok.get(i, (0, 0))
            cols["id"].append(i); cols["shard"].append(s); cols["bp"].append(bp); cols["subj"].append(subj)
            cols["joints"].append(j); cols["frames"].append(fr); cols["cap"].append(cap[:160])
    out = ROOT / "data" / "analysis" / "lottie_browser_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cols, separators=(",", ":")))
    n = len(cols["id"])
    print(f"{n:,} animations ({sum(1 for j in cols['joints'] if j) :,} with tokens) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- parallel build over the whole corpus
def load_relevant() -> dict:
    """metadata shard -> {id: (body_plan, subject, caption)} for every row Qwen called relevant (data/vlm/animation)."""
    import orjson
    import zstandard
    by: dict = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "vlm" / "animation" / "shard-*.jsonl.zst"))):
        if Path(f).name.startswith("._"):
            continue
        with open(f, "rb") as fh:
            for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                r = orjson.loads(line)
                if r.get("relevant"):
                    by.setdefault(int(r["shard"]), {})[str(r["id"])] = (r.get("body_plan") or "", r.get("subject") or "", r.get("caption_lottie") or "")
    return by


def _shard_job(job):
    """One corpus shard -> tokens of its relevant rows. Runs in a worker process."""
    import orjson
    import zstandard
    shard, ids, opts = job
    corpus = ROOT.parent / "Datasets" / "LottieAnimation-660K"
    path = glob.glob(str(corpus / "snapshots" / "*" / "data" / f"metadata-{shard:05d}-of-00068.jsonl.zst"))[0]
    rx = re.compile(rb'^\{"id":"(\d+)"')
    out = dict(shard=shard, jt=[], st=[], gl=[], sc=[], nv=[], meta=[], skipped={}, rows=0)
    with open(path, "rb") as fh:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
            m = rx.match(line[:40])
            if not m or m.group(1).decode() not in ids:
                continue
            i = m.group(1).decode()
            out["rows"] += 1
            try:
                row = orjson.loads(line)
                lj = row["lottie_json"]
                lj = orjson.loads(lj) if isinstance(lj, str) else lj
                r = lottie_tokens(lj, opts["max_joints"], opts["groups"], opts["keep"], opts["roots"])
            except Exception as e:
                out["skipped"][type(e).__name__] = out["skipped"].get(type(e).__name__, 0) + 1
                continue
            if r is None:
                out["skipped"]["unusable"] = out["skipped"].get("unusable", 0) + 1
                continue
            bp, subj, cap = ids[i]
            out["jt"].append(r[0]); out["st"].append(r[1]); out["gl"].append(r[2]); out["sc"].append(r[3]); out["nv"].append(r[4])
            out["meta"].append(dict(dataset="lottie", clip=i, object_type=bp, subject=subj, shard=shard, joints=len(r[0]), fps=lj.get("fr"), frames=r[4],
                                    duration_s=round(float(np.exp(r[3][0])), 3), caption=cap))
    return out


def cmd_build_all(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import time
    rel = load_relevant()
    if args.shards != "all":
        rel = {s: v for s, v in rel.items() if s in {int(x) for x in args.shards.split(",")}}
    total = sum(len(v) for v in rel.values())
    print(f"{total:,} relevant animations in {len(rel)} corpus shards; {args.workers} workers", flush=True)
    opts = dict(max_joints=args.max_joints, groups=args.groups, keep=args.keep, roots=args.roots)
    res, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_shard_job, [(s, rel[s], opts) for s in sorted(rel)]):
            res[r["shard"]] = r
            done += r["rows"]
            print(f"  [{time.time() - t0:5.0f}s] shard {r['shard']:2d}: {len(r['jt']):5d} tokenised of {r['rows']:5d}  skipped {r['skipped']}   total {done:,}/{total:,}", flush=True)
    order = sorted(res)
    jt = [a for s in order for a in res[s]["jt"]]
    st = [a for s in order for a in res[s]["st"]]
    gl = [a for s in order for a in res[s]["gl"]]
    sc = [a for s in order for a in res[s]["sc"]]
    nv = [a for s in order for a in res[s]["nv"]]
    meta = [m for s in order for m in res[s]["meta"]]
    offs = np.cumsum([0] + [len(a) for a in jt])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, joint=np.concatenate(jt), static=np.concatenate(st), offsets=offs, glob=np.stack(gl), scal=np.stack(sc), n_valid=np.array(nv))
    cols = ["dataset", "clip", "object_type", "subject", "shard", "joints", "fps", "frames", "duration_s", "caption"]
    with open(out.with_name(out.stem + "_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(meta)
    sk = collections.Counter()
    for r in res.values():
        sk.update(r["skipped"])
    print(f"{len(jt):,} animations tokenised of {total:,}; {int(offs[-1]):,} joint tokens; skipped {dict(sk)}; {time.time() - t0:.0f} s -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "build", "build-all", "demo", "browser-index"])
    ap.add_argument("--shards", default="3", help="comma list of corpus shards, or 'all' with build-all")
    ap.add_argument("--limit", type=int, default=0, help="rows per shard (0 = all)")
    ap.add_argument("--relevant", action="store_true", help="only rows Qwen called relevant (data/vlm)")
    ap.add_argument("--json-dir")
    ap.add_argument("--max-joints", type=int, default=96)
    ap.add_argument("--groups", action=argparse.BooleanOptionalAction, default=True, help="animated shape groups become joints (--no-groups: layers only)")
    ap.add_argument("--keep", default="component", choices=["moving", "component", "all"], help="which joints are tokens (see select_joints)")
    ap.add_argument("--roots", default="nearest", choices=["none", "nearest", "virtual"], help="how several root layers are joined into one tree")
    ap.add_argument("--out", default=str(ROOT / "data" / "embeddings" / "lottie_feats.npz"))
    ap.add_argument("--workers", type=int, default=6, help="build-all: worker processes (one corpus shard each)")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "lottie_tokens_demo"))
    ap.add_argument("--n", type=int, default=40, help="demo: animations with tokens")
    ap.add_argument("--n-none", type=int, default=6, help="demo: animations without tokens (to show why)")
    args = ap.parse_args()
    {"selftest": lambda a: selftest(), "build": cmd_build, "build-all": cmd_build_all, "demo": cmd_demo, "browser-index": cmd_browser_index}[args.cmd](args)


if __name__ == "__main__":
    main()
