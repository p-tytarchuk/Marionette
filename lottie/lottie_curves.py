"""Sample animated transform properties of a Lottie file into (t, y) arrays.

Handles both keyframe dialects (bodymovin <5.5 with "e" end values, >=5.5
where the end value is the next keyframe's "s"), hold keyframes ("h": 1),
cubic-bezier temporal easing (per-dimension or shared), split position
("p": {"s": true, "x": ..., "y": ...}), and layer time offset/stretch
("st"/"sr").

Known approximations (fine for primitive coverage, not for rendering):
- spatial bezier paths on position ("ti"/"to") are ignored -> linear in space;
- precomp layers are sampled in the precomp's own time over the parent
  composition's window (time remapping "tm" is ignored).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# Lottie transform key -> track name (rotation is converted to radians in _layer_tracks)
TRANSFORM_PROPS = {"r": "rotation", "p": "position", "s": "scale"}


@dataclass
class Track:
    layer_path: str      # e.g. "comp_0/leg_front" -- human-readable, not unique
    layer_index: int     # layer "ind" (parenting key), -1 if absent
    parent_index: int    # parent "ind", -1 for root layers
    prop: str            # "rotation" | "position.x" | "scale.y" ...
    n_keyframes: int
    t: np.ndarray        # seconds, composition time
    y: np.ndarray        # rotation in radians; position in px; scale in %


def _as_vec(v) -> np.ndarray:
    return np.atleast_1d(np.asarray(v, dtype=float))


def _ease_component(e, i: int) -> float:
    v = e[i] if isinstance(e, list) and len(e) > i else (e[0] if isinstance(e, list) else e)
    return float(v)


def _bezier_progress(x: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """y(u) where x(u) = x for the cubic bezier (0,0),(x1,y1),(x2,y2),(1,1)."""
    x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
    lo = np.zeros_like(x)
    hi = np.ones_like(x)
    for _ in range(40):  # bisection: x(u) is monotone for x1,x2 in [0,1]
        u = 0.5 * (lo + hi)
        xu = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
        lo = np.where(xu < x, u, lo)
        hi = np.where(xu >= x, u, hi)
    u = 0.5 * (lo + hi)
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3


def sample_property(prop: dict, frames: np.ndarray, linear: bool = False) -> np.ndarray | None:
    """Evaluate an animated property at local frame times. Returns (F, D).
    linear=True ignores bezier easing (straight lines between keyframes);
    only used to measure how much the easing matters for coverage."""
    kfs = prop.get("k")
    if not isinstance(kfs, list) or not kfs or not isinstance(kfs[0], dict):
        return None
    kfs = [k for k in kfs if isinstance(k, dict) and "t" in k]
    if len(kfs) < 2:
        return None
    starts = []
    for idx, k in enumerate(kfs):
        if "s" in k:
            starts.append(_as_vec(k["s"]))
        elif idx > 0 and "e" in kfs[idx - 1]:
            starts.append(_as_vec(kfs[idx - 1]["e"]))
        else:
            starts.append(starts[-1] if starts else None)
    if starts[0] is None:
        return None
    dim = len(starts[0])
    starts = [s if s is not None and len(s) == dim else starts[0] for s in starts]

    times = np.array([float(k["t"]) for k in kfs])
    out = np.full((len(frames), dim), np.nan)
    out[frames <= times[0]] = starts[0]
    out[frames >= times[-1]] = starts[-1]
    for i in range(len(kfs) - 1):
        k, t0, t1 = kfs[i], times[i], times[i + 1]
        mask = (frames >= t0) & (frames < t1)  # >= : frames landing exactly on a keyframe
        if not mask.any() or t1 <= t0:
            continue
        v0 = starts[i]
        v1 = _as_vec(k["e"]) if "e" in k and len(_as_vec(k["e"])) == dim else starts[i + 1]
        if k.get("h") == 1:
            out[mask] = v0
            continue
        x = (frames[mask] - t0) / (t1 - t0)
        if linear:
            out[mask] = v0 + x[:, None] * (v1 - v0)
            continue
        o, inn = k.get("o") or {}, k.get("i") or {}
        for d in range(dim):
            try:
                p = _bezier_progress(x, _ease_component(o.get("x", 0), d), _ease_component(o.get("y", 0), d),
                                     _ease_component(inn.get("x", 1), d), _ease_component(inn.get("y", 1), d))
            except (TypeError, ValueError, IndexError):
                p = x  # malformed easing -> linear
            out[mask, d] = v0[d] + p * (v1[d] - v0[d])
    return out


def _is_animated(prop) -> bool:
    return isinstance(prop, dict) and prop.get("a") == 1


def _layer_tracks(layer: dict, path: str, comp_frames: np.ndarray, fr: float, linear: bool) -> Iterator[Track]:
    ks = layer.get("ks") or {}
    st, sr = float(layer.get("st", 0) or 0), float(layer.get("sr", 1) or 1)
    local = (comp_frames - st) / sr
    t = comp_frames / fr
    common = dict(layer_path=path, layer_index=int(layer.get("ind", -1)),
                  parent_index=int(layer.get("parent", -1)))
    for key, name in TRANSFORM_PROPS.items():
        prop = ks.get(key)
        if key == "r" and prop is None:
            prop = ks.get("rz")  # 3D layers store rotation as rz
        if not isinstance(prop, dict):
            continue
        if key == "p" and prop.get("s") is True:  # split dimensions
            for axis in ("x", "y"):
                sub = prop.get(axis)
                if _is_animated(sub):
                    vals = sample_property(sub, local, linear)
                    if vals is not None:
                        yield Track(prop=f"position.{axis}", n_keyframes=len(sub["k"]), t=t, y=vals[:, 0], **common)
            continue
        if not _is_animated(prop):
            continue
        vals = sample_property(prop, local, linear)
        if vals is None:
            continue
        if key == "r":
            yield Track(prop=name, n_keyframes=len(prop["k"]), t=t, y=np.deg2rad(vals[:, 0]), **common)
        else:
            for d, axis in enumerate(("x", "y")[: min(2, vals.shape[1])]):
                yield Track(prop=f"{name}.{axis}", n_keyframes=len(prop["k"]), t=t, y=vals[:, d], **common)


def iter_tracks(lottie: dict, linear: bool = False) -> Iterator[Track]:
    """All animated rotation/position/scale tracks, including inside precomps."""
    fr = float(lottie.get("fr") or 30)
    ip, op = float(lottie.get("ip", 0) or 0), float(lottie.get("op", 0) or 0)
    if op <= ip:
        return
    frames = np.arange(ip, op + 1e-9, 1.0)
    assets = {a.get("id"): a for a in lottie.get("assets") or [] if isinstance(a, dict) and "layers" in a}

    def walk(layers, prefix, depth):
        if depth > 8:
            return
        for li, layer in enumerate(layers or []):
            if not isinstance(layer, dict):
                continue
            path = f"{prefix}/{layer.get('nm', li)}"
            yield from _layer_tracks(layer, path, frames, fr, linear)
            if layer.get("ty") == 0 and layer.get("refId") in assets:
                yield from walk(assets[layer["refId"]].get("layers"), path, depth + 1)

    yield from walk(lottie.get("layers"), "root", 0)
