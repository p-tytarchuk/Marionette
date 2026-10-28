"""Curve statistics copied from Loomie (tools/fit_coverage.py, tools/coverage_report.py) so that
Marionette does not import from ../Loomie: loop test, periodicity test (normalized squared-difference
autocorrelation, independent of the sine fit) and the curve-shape fingerprint used for dedupe."""

from __future__ import annotations

import hashlib

import numpy as np

LOOP_TOL = 0.05       # closed curve: |y_end - y_start| < LOOP_TOL * range
PERIODIC_ACF = 0.8    # self-similarity a repeat must reach to count as periodic
MAX_SAMPLES = 600     # long clips are downsampled; the primitive has 4 dof
MAX_ABS = 1e5         # rotation rad / px / %: beyond this the file is corrupt, not animated
NEGLIGIBLE = {"rotation": np.deg2rad(1.0), "position": 1.0, "scale": 1.0}   # total range below this = no visible motion


def is_loop(y: np.ndarray) -> bool:
    rng = float(y.max() - y.min())
    return rng > 0 and abs(float(y[-1] - y[0])) < LOOP_TOL * rng


def autocorr_peak(y: np.ndarray) -> tuple[float, int]:
    """sim(k) = 1 - mean((y[t]-y[t+k])^2) / (2 var y), searched from the first k with sim < 0 up to n/2
    (at least two repetitions must fit). Returns (best sim, first lag within 0.005 of it); (0, 0) if none."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    var = float(y.var())
    if var < 1e-12:
        return 0.0, 0
    sim = np.array([1.0] + [1.0 - float(np.mean((y[:-k] - y[k:]) ** 2)) / (2 * var) for k in range(1, n // 2 + 1)])
    below = np.nonzero(sim < 0)[0]
    if not len(below):
        return 0.0, 0
    k0 = int(below[0])
    best = float(sim[k0:].max())
    k = k0 + int(np.nonzero(sim[k0:] >= best - 0.005)[0][0])
    return best, k


def curve_shape(t: np.ndarray, y: np.ndarray, n: int = 32) -> tuple:
    """The track resampled at n evenly spaced points of its clip, in whole degrees (rotation) / units."""
    u = (t - t[0]) / max(float(t[-1] - t[0]), 1e-9)
    return tuple(np.round(np.interp(np.linspace(0, 1, n), u, np.rad2deg(y))).astype(int).tolist())


def fingerprint(shapes: list[tuple]) -> str:
    """Animation-level dedupe key: hash of the sorted multiset of (n_keyframes, duration, curve_shape)."""
    return hashlib.sha1(repr(sorted(shapes)).encode()).hexdigest()[:16]
