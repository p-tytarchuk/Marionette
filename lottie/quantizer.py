"""Value quantization for the curve fields (r, a, s, phi, clamp): the single source of truth for
the value-token vocabulary, as `primitive.py` is for the primitive itself.

One token per field (`c1 = 6`, see CLAUDE.md) needs the numbers to come from a fixed token set.
Each field has its own set of levels; a token is just the index into it. Bin counts cost
vocabulary rows, not output length. Values are in the template's units (radians, Hz, cycles).

    field   levels                                   index
    r       nr uniform levels in [-R, R]             0 .. nr-1        (r wrapped to [-pi, pi) first)
    a       na signed uniform levels in [-A, A]      0 .. na-1        (na odd, zero is the middle index)
    s       ns log-spaced levels in [s_min, s_max]   0 .. ns-1        (outside the range is clipped)
    phi     nphi uniform levels k / nphi             0 .. nphi-1      (wraps: 0.999 -> 0)
    clamp   0 / 1

Default scheme: the one recommended by tools/quantize_study.py (129 / 65 / 128 / 64, 388 value
tokens: r step 1.4 deg, a step 2.8 deg, s step 3.5 %, phi step 0.016 cycle).

Token names are `<r17>`, `<a3>`, `<s90>`, `<p12>`, `<c0>`/`<c1>`; joints, class and template come
from the template's `grammar_joint_values` and are not handled here.

    from quantizer import DEFAULT, encode_curve, decode_curve, curve_to_tokens, value_tokens
    python tools/quantizer.py            # vocabulary size, round-trip and error-bound self-check
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELDS = ("r", "a", "s", "phi", "clamp")
PREFIX = {"r": "r", "a": "a", "s": "s", "phi": "p", "clamp": "c"}


@dataclass(frozen=True)
class Scheme:
    r_max: float = np.pi / 2   # rad, r in [-r_max, r_max]
    a_max: float = np.pi / 2   # rad, a in [-a_max, a_max]
    s_min: float = 0.05        # Hz
    s_max: float = 4.0         # Hz
    nr: int = 129
    na: int = 65               # odd: zero is a level
    ns: int = 128
    nphi: int = 64

    def __post_init__(self):
        if self.na % 2 == 0:
            raise ValueError("na must be odd so that a = 0 is a level")
        if not 0 < self.s_min < self.s_max:
            raise ValueError("need 0 < s_min < s_max")

    def size(self, field: str) -> int:
        return {"r": self.nr, "a": self.na, "s": self.ns, "phi": self.nphi, "clamp": 2}[field]

    @property
    def n_tokens(self) -> int:
        return sum(self.size(f) for f in FIELDS)

    # ---- levels (index -> value)
    def levels(self, field: str) -> np.ndarray:
        if field == "r":
            return np.linspace(-self.r_max, self.r_max, self.nr)
        if field == "a":
            return np.linspace(-self.a_max, self.a_max, self.na)
        if field == "s":
            return np.exp(np.linspace(np.log(self.s_min), np.log(self.s_max), self.ns))
        if field == "phi":
            return np.arange(self.nphi) / self.nphi
        if field == "clamp":
            return np.array([0, 1])
        raise KeyError(field)

    # ---- value -> index
    def index(self, field: str, x: float) -> int:
        if field == "clamp":
            return 1 if x else 0
        if field == "phi":
            return int(round((float(x) % 1.0) * self.nphi)) % self.nphi
        if field == "r":
            x = (float(x) + np.pi) % (2 * np.pi) - np.pi
            lo, hi, n = -self.r_max, self.r_max, self.nr
        elif field == "a":
            lo, hi, n = -self.a_max, self.a_max, self.na
        elif field == "s":
            x, lo, hi, n = np.log(np.clip(float(x), self.s_min, self.s_max)), np.log(self.s_min), np.log(self.s_max), self.ns
        else:
            raise KeyError(field)
        return int(np.clip(round((x - lo) / (hi - lo) * (n - 1)), 0, n - 1))

    def value(self, field: str, i: int) -> float:
        if not 0 <= i < self.size(field):
            raise ValueError(f"{field} index {i} outside 0..{self.size(field) - 1}")
        v = self.levels(field)[i]
        return int(v) if field == "clamp" else float(v)


DEFAULT = Scheme()


# ------------------------------------------------------------------ curves

def encode_curve(curve: dict, scheme: Scheme = DEFAULT) -> dict[str, int]:
    """{'r','a','s','phi','clamp'} values -> indices."""
    return {f: scheme.index(f, curve[f]) for f in FIELDS}


def decode_curve(idx: dict[str, int], scheme: Scheme = DEFAULT) -> dict:
    return {f: scheme.value(f, idx[f]) for f in FIELDS}


def quantize_curve(curve: dict, scheme: Scheme = DEFAULT) -> dict:
    """What the model can actually say: the nearest representable curve (joint is passed through)."""
    out = decode_curve(encode_curve(curve, scheme), scheme)
    return {**({"joint": curve["joint"]} if "joint" in curve else {}), **out}


def token(field: str, i: int) -> str:
    return f"<{PREFIX[field]}{i}>"


def curve_to_tokens(curve: dict, scheme: Scheme = DEFAULT) -> list[str]:
    """The 5 value tokens of a curve, in grammar order (r a s phi clamp). The joint token is separate."""
    idx = encode_curve(curve, scheme)
    return [token(f, idx[f]) for f in FIELDS]


def tokens_to_curve(tokens: list[str], scheme: Scheme = DEFAULT) -> dict:
    if len(tokens) != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} value tokens, got {len(tokens)}")
    idx = {}
    for f, tok in zip(FIELDS, tokens):
        p = PREFIX[f]
        if not (tok.startswith(f"<{p}") and tok.endswith(">") and tok[len(p) + 1:-1].isdigit()):
            raise ValueError(f"{tok!r} is not a {f} token")
        idx[f] = int(tok[len(p) + 1:-1])
    return decode_curve(idx, scheme)


def value_tokens(scheme: Scheme = DEFAULT) -> dict[str, list[str]]:
    """Whole vocabulary per field, in index order: what to add to the tokenizer and to grammar/."""
    return {f: [token(f, i) for i in range(scheme.size(f))] for f in FIELDS}


def max_error(scheme: Scheme = DEFAULT) -> dict[str, float]:
    """Worst-case quantization error per field inside the range: half a step (s: relative)."""
    return {"r_deg": float(np.degrees(scheme.r_max / (scheme.nr - 1))),
            "a_deg": float(np.degrees(scheme.a_max / (scheme.na // 2)) / 2),
            "s_rel": float(np.exp(np.log(scheme.s_max / scheme.s_min) / (scheme.ns - 1) / 2) - 1),
            "phi_cycles": 0.5 / scheme.nphi}


if __name__ == "__main__":
    s = DEFAULT
    rng = np.random.default_rng(0)
    for _ in range(2000):
        c = {"r": rng.uniform(-3.5, 3.5), "a": rng.uniform(-2, 2), "s": rng.uniform(0.01, 6), "phi": rng.uniform(-1, 2),
             "clamp": int(rng.integers(2))}
        q = quantize_curve(c, s)
        assert quantize_curve(q, s) == q, "quantization is not idempotent"
        assert tokens_to_curve(curve_to_tokens(c, s), s) == q, "token round trip"
    for f in FIELDS:
        lv = s.levels(f)
        assert all(s.index(f, v) == i for i, v in enumerate(lv)), f"levels of {f} do not index to themselves"
    assert s.value("a", s.na // 2) == 0.0 and s.index("phi", 0.9999) == 0
    print(f"value tokens: {s.n_tokens}  " + "  ".join(f"{f}={s.size(f)}" for f in FIELDS))
    print("worst-case error inside the range:", {k: round(v, 4) for k, v in max_error(s).items()})
    print("self-check ok")
