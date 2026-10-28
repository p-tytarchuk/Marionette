"""Motion primitive m(t) = r + a * sin(2*pi*(s*t + phi)) and its least-squares fit.

phi is the phase in cycles (fractions of a period), always in [0, 1) --
bounded independently of s, so it quantizes into a fixed set of value
tokens. Same convention as eq. (2) in docs/presentation.html.

Single source of truth for the primitive's semantics. Everything that
evaluates or fits a curve (coverage, data synthesis, baking) should import
from here so the definition of `clamp` can't drift between tools.

clamp semantics (settled; the data generator, fits and eval rely on it): clamp=1 means one-sided
motion, m(t) = r + a * max(0, sin(...)) -- the bone rests at r and only
swings to one side (e.g. a paw lift that never goes below the ground line).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

TWO_PI = 2.0 * np.pi


@dataclass
class Fit:
    r: float
    a: float
    s: float
    phi: float
    clamp: int
    r2: float

    def as_dict(self) -> dict:
        return asdict(self)


def wrap_phase(phi: float) -> float:
    """phi mod 1 in [0, 1); float rounding can make -1e-17 % 1.0 == 1.0."""
    phi = float(phi) % 1.0
    return 0.0 if phi >= 1.0 else phi


def evaluate(t: np.ndarray, r: float, a: float, s: float, phi: float, clamp: int = 0) -> np.ndarray:
    wave = np.sin(TWO_PI * (s * np.asarray(t) + phi))
    if clamp:
        wave = np.maximum(wave, 0.0)
    return r + a * wave


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / ss_tot


def _linear_fit_at(t: np.ndarray, y: np.ndarray, s: float) -> tuple[float, float, float, float]:
    """For fixed frequency s the model is linear in (r, A, B):
    r + A sin(2pi s t) + B cos(2pi s t).  Returns (r, a, phi, sse)."""
    w = TWO_PI * s * t
    X = np.column_stack([np.ones_like(t), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r, A, B = coef
    sse = float(np.sum((X @ coef - y) ** 2))
    a = float(np.hypot(A, B))
    # a sin(2pi (s t + phi)) = a cos(2pi phi) sin(2pi s t) + a sin(2pi phi) cos(2pi s t)
    phi = wrap_phase(np.arctan2(B, A) / TWO_PI)
    return float(r), a, phi, sse


def fit_unclamped(t: np.ndarray, y: np.ndarray, s_max: float, n_grid: int = 400) -> Fit:
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    s_min = 0.25 / max(duration, 1e-6)  # slowest: a quarter period across the clip
    if s_max <= s_min:
        s_max = s_min * 4
    grid = np.geomspace(s_min, s_max, n_grid)
    sses = np.array([_linear_fit_at(t, y, s)[3] for s in grid])
    i = int(np.argmin(sses))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda s: _linear_fit_at(t, y, s)[3], bounds=(lo, hi), method="bounded")
    s_best = float(res.x) if res.fun <= sses[i] else float(grid[i])
    r, a, phi, _ = _linear_fit_at(t, y, s_best)
    return Fit(r, a, s_best, phi, 0, r_squared(y, evaluate(t, r, a, s_best, phi)))


def _clamped_grid(t: np.ndarray, y: np.ndarray, s_grid: np.ndarray, n_phase: int = 32,
                  chunk: int = 20) -> np.ndarray:
    """Global search for the one-sided variant. For fixed (s, phi) the model
    r + a*max(0, sin(2pi(s t + phi))) is linear in (r, a), so every grid point
    is scored in closed form (simple regression of y on the clamped wave).
    Returns an array of (sse, r, a, s, phi) rows, one per grid point."""
    phis = np.arange(n_phase) / n_phase
    yc = y - y.mean()
    syy = float(yc @ yc)
    rows = []
    for i in range(0, len(s_grid), chunk):
        sc = s_grid[i:i + chunk]
        w = np.maximum(0.0, np.sin(TWO_PI * (sc[:, None, None] * t[None, None, :] + phis[None, :, None])))
        wm = w.mean(axis=-1)
        wc = w - wm[..., None]
        varw = np.einsum("spn,spn->sp", wc, wc)
        cov = wc @ yc
        ok = varw > 1e-12
        a = np.where(ok, cov / np.where(ok, varw, 1.0), 0.0)
        sse = np.where(ok, syy - cov * a, syy)
        r = y.mean() - a * wm
        S, P = np.meshgrid(sc, phis, indexing="ij")
        rows.append(np.stack([sse, r, a, S, P], axis=-1).reshape(-1, 5))
    return np.concatenate(rows)


def fit_clamped(t: np.ndarray, y: np.ndarray, s_max: float, init: Fit, n_s: int = 200,
                n_phase: int = 32, n_refine: int = 3) -> Fit:
    """Fit of the one-sided variant, robust to local optima.

    Starts for the nonlinear refinement (least_squares over r, a, s, phi):
    - the n_refine best points of a global (s, phi) grid (_clamped_grid), at
      most one per frequency, so they are not all the same basin;
    - the 8 starts seeded from the unclamped fit (2 signs x 4 phases), so the
      result is never worse than the previous, seed-only search.
    The s grid reaches below the unclamped one (0.1 / duration): at very low
    frequency a clamped half-wave acts as a single step or bump."""
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    s_lo = min(0.1 / max(duration, 1e-6), s_max / 4)
    grid = _clamped_grid(t, y, np.geomspace(s_lo, s_max, n_s), n_phase)
    starts, used_s = [], set()
    for row in grid[np.argsort(grid[:, 0])]:
        if row[3] in used_s:
            continue
        used_s.add(row[3])
        starts.append([row[1], row[2], row[3], row[4]])
        if len(starts) == n_refine:
            break
    s0 = min(max(init.s, 1e-3), s_max)
    for sign in (1.0, -1.0):  # rest at the bottom (a>0) or the top (a<0)
        r0 = float(y.min() if sign > 0 else y.max())
        a0 = sign * float(y.max() - y.min())
        starts += [[r0, a0, s0, k / 4] for k in range(4)]

    def resid(p):
        return evaluate(t, p[0], p[1], p[2], p[3], clamp=1) - y

    best: Fit | None = None
    for x0 in starts:
        x0[2] = min(max(x0[2], 1e-4), s_max)
        try:
            sol = least_squares(resid, x0, bounds=([-np.inf, -np.inf, 1e-4, -np.inf], [np.inf, np.inf, s_max, np.inf]),
                                max_nfev=200)
        except ValueError:
            continue
        r, a, s, phi = (float(v) for v in sol.x)
        f = Fit(r, a, s, wrap_phase(phi), 1, r_squared(y, evaluate(t, r, a, s, phi, clamp=1)))
        if np.isfinite(f.r2) and (best is None or f.r2 > best.r2):
            best = f
    return best if best is not None else Fit(init.r, init.a, init.s, init.phi, 1, -np.inf)


def fit(t: np.ndarray, y: np.ndarray, fps: float, s_cap: float = 8.0) -> tuple[Fit, Fit]:
    """Fit both variants. s is capped below Nyquist and at a physical
    s_cap (cycles/sec) -- faster "motion" in a Lottie file is jitter, not gait."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    s_max = min(s_cap, 0.45 * fps)
    unclamped = fit_unclamped(t, y, s_max)
    clamped = fit_clamped(t, y, s_max, unclamped)
    return unclamped, clamped


# ---------------------------------------------------------------- Fourier (analysis only)
# NOT part of the model's output language. The grammar, quantizer and trained
# checkpoints all assume the single-sine primitive above (6 fields per bone).
# This is a K-harmonic fit used to measure how much of a real curve the sine
# leaves on the table: m(t) = r + sum_{k=1..K} a_k sin(2 pi (k s t + phi_k)).

@dataclass
class FourierFit:
    r: float
    s: float
    a: list[float]
    phi: list[float]
    r2: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_fourier(t: np.ndarray, r: float, s: float, a: list[float], phi: list[float]) -> np.ndarray:
    t = np.asarray(t)
    out = np.full_like(t, r, dtype=float)
    for k, (ak, pk) in enumerate(zip(a, phi), start=1):
        out += ak * np.sin(TWO_PI * (k * s * t + pk))
    return out


def _fourier_design(t: np.ndarray, s: float, order: int) -> np.ndarray:
    cols = [np.ones_like(t)]
    for k in range(1, order + 1):
        w = TWO_PI * k * s * t
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def _fourier_linear(t: np.ndarray, y: np.ndarray, s: float, order: int):
    X = _fourier_design(t, s, order)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, float(np.sum((X @ coef - y) ** 2))


def fit_fourier(t: np.ndarray, y: np.ndarray, fps: float, order: int = 4, s_cap: float = 8.0,
                n_grid: int = 400) -> FourierFit:
    """Fundamental s by 1-D search (for fixed s the model is linear in r and
    the 2*order sin/cos coefficients). Highest harmonic order*s stays below
    the same Nyquist-based cap the single-sine fit uses."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    s_min = 0.25 / max(duration, 1e-6)
    s_max = min(s_cap, 0.45 * fps / order)
    if s_max <= s_min:
        s_max = s_min * 4
    grid = np.geomspace(s_min, s_max, n_grid)
    sses = np.array([_fourier_linear(t, y, s, order)[1] for s in grid])
    i = int(np.argmin(sses))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    res = minimize_scalar(lambda s: _fourier_linear(t, y, s, order)[1], bounds=(lo, hi), method="bounded")
    s_best = float(res.x) if res.fun <= sses[i] else float(grid[i])
    coef, _ = _fourier_linear(t, y, s_best, order)
    A, B = coef[1::2], coef[2::2]
    a = np.hypot(A, B)
    phi = [wrap_phase(np.arctan2(b, aa) / TWO_PI) for aa, b in zip(A, B)]
    yh = evaluate_fourier(t, float(coef[0]), s_best, list(a), phi)
    return FourierFit(float(coef[0]), s_best, [float(v) for v in a], phi, r_squared(y, yh))
