#!/usr/bin/env python3
"""How often is a generated walk bad? Compare the v1 pipeline (absolute hips height) with v2 (hips height above the lowest foot).

  python3 scripts/motion/walk_check.py v1      # uses motion_feats_v1.npz, motion_v1.npz, motion_ae_v1.pt, z_diffusion_v1.pt
  python3 scripts/motion/walk_check.py v2      # the current files

A walking clip is "bad" if any of: the hips float (mean hips height outside the normal band for that version), it travels implausibly fast for a walk
(more than 2.5 hips heights per second), or it does not step (step score below 40% of the real walking median, see gait_check.py).
The same rule is applied to REAL forward-walking clips, so the real rate is the floor a generator can be expected to reach.
The hips band is [0.8, 1.1] for v1 and, because v2 measures above the ankle (about 0.1 lower), [0.7, 1.0] for v2.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]


GUIDANCE = (1.0, 2.0, 4.0)
RESCALE = 0.0


class Ctx:
    """Everything needed to score generated walks: data, models, the real-walk reference and the bad-walk rule."""


def prepare(version="v2"):
    import torch
    import diffusion_test as DT
    import gait_check as GC
    import motion_ae as MA
    c = Ctx()
    sfx = "_v1" if version == "v1" else ""
    DT.EMB = MA.EMB = EMB
    DT.CKPT = EMB / f"z_diffusion{sfx}.pt"
    MA.CKPT = EMB / f"motion_ae{sfx}.pt"
    c.dev = "mps" if torch.backends.mps.is_available() else "cpu"
    z = np.load(EMB / f"motion{sfx}.npz")
    if version == "v2":
        P = DT.load_pairs()
    else:
        temb = np.load(EMB / "text.npy") - np.load(EMB / "text_mean.npy")
        temb = (temb / np.maximum(np.linalg.norm(temb, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        P = dict(z=z["z"].astype(np.float32), split=z["split"], cls=z["cls"], keys=[str(k) for k in z["key"]], text_clip=z["text_clip"], temb=temb, dur=z["duration_s"])
    D = np.load(EMB / f"motion_feats{sfx}.npz")["feats"]
    idx = {(r["dataset"], r["clip"]): i for i, r in enumerate(csv.DictReader(open(EMB / "motion_feats_index.csv", newline="")))}
    c.F = np.stack([D[idx[tuple(k.split("|", 1))]] for k in P["keys"]])
    c.caps = [json.loads(t).get("caption", "") for t in z["captions"]]
    c.nvs = z["n_valid"]
    c.P, c.DT, c.MA, c.GC, c.torch, c.version = P, DT, MA, GC, torch, version
    c.ae, c.mean, c.std, _ = MA.load_model(c.dev)
    c.model, c.zm, c.zs, _ = DT.load_model(c.dev)
    c.lo, c.hi = (0.8, 1.1) if version == "v1" else (0.7, 1.0)
    c.sel = [i for i, t in enumerate(c.caps) if P["cls"][i] == "biped" and re.search(r"\bwalks?\b", t)
             and not re.search(r"in place|backward|turn|jump|stop|sit|kneel|pick|carry|crouch|circle", t)
             and 30 <= int(c.nvs[i]) <= 128 and float(P["dur"][i]) < 6 and np.abs(c.F[i, :int(c.nvs[i]), 0]).sum() / float(P["dur"][i]) > 0.5]
    c.real = [c.stats(c.F[i], int(c.nvs[i]), float(P["dur"][i])) for i in c.sel]
    c.ref_step = float(np.median([r["step"] for r in c.real]))
    return c


def _stats(self, f, nv, dur):
    f = f[:nv]
    g = self.GC.gait(f, nv)
    return dict(hips=float(f[:, 3].mean()), speed=float(np.abs(f[:, 0]).sum() / max(dur, 1e-3)), step=g["step"], dur=float(dur))


def _decode(self, zz):
    torch = self.torch
    with torch.no_grad():
        zt = torch.from_numpy(np.asarray(zz, np.float32)).to(self.dev)
        X = self.ae.decode(zt).cpu().numpy() * self.std + self.mean
        aux = self.ae.aux(zt).cpu().numpy()
    return X, np.clip(np.round(aux[:, 1] * 128), 8, 128).astype(int), np.exp(aux[:, 0])


def _judge(self, rows):
    floaty = np.array([not (self.lo <= r["hips"] <= self.hi) for r in rows])
    fast = np.array([r["speed"] > 2.5 for r in rows])
    still = np.array([r["step"] < 0.4 * self.ref_step for r in rows])
    return floaty, fast, still


def _report(self, name, zz, extra=""):
    """Score latents zz (n, 256, un-standardised) with the bad-walk rule; prints one line and returns the per-sample rows."""
    X, nv, du = self.decode(zz)
    rows = [self.stats(X[k], int(nv[k]), float(du[k])) for k in range(len(zz))]
    fl, fa, st = self.judge(rows)
    b = fl | fa | st
    print(f"  {name:52s} n={len(rows):3d} bad {100 * b.mean():4.1f}%  (floating hips {100 * fl.mean():4.1f}%, too fast {100 * fa.mean():4.1f}%, not stepping {100 * st.mean():4.1f}%)  median speed {np.median([r['speed'] for r in rows]):.2f} H/s{extra(rows) if callable(extra) else extra}")
    return rows


Ctx.stats, Ctx.decode, Ctx.judge, Ctx.report = _stats, _decode, _judge, _report


def main(version):
    c = prepare(version)
    print(f"[{version}] rescale {RESCALE}, hips band {c.lo}-{c.hi}, real forward-walking clips: {len(c.sel)}, median step score {c.ref_step:.3f}")
    fl, fa, st = c.judge(c.real)
    b = fl | fa | st
    print(f"  REAL forward-walking clips                       n={len(c.sel):3d} bad {100 * b.mean():4.1f}%  (floating hips {100 * fl.mean():4.1f}%, too fast {100 * fa.mean():4.1f}%, not stepping {100 * st.mean():4.1f}%)")
    T = c.P["text_clip"][c.sel]
    for g in GUIDANCE:
        zn = c.DT.sample(c.model, c.torch, c.dev, T, guidance=g, seed=5, rescale=RESCALE)
        c.report(f"GENERATED from the {len(c.sel)} real walking captions, g={g}", zn * c.zs + c.zm)
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import encode, load_text_encoder
    mu = np.load(EMB / "text_mean.npy")
    E = encode(["An object walks forward."], load_text_encoder())[0] - mu
    E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    for g in GUIDANCE:
        zn = c.DT.sample(c.model, c.torch, c.dev, np.repeat(E, 200, 0), guidance=g, seed=11, rescale=RESCALE)
        c.report(f'GENERATED "An object walks forward." g={g}', zn * c.zs + c.zm)


if __name__ == "__main__":
    args = [a for a in sys.argv[2:] if not a.startswith("rescale=")]
    for a in sys.argv[2:]:
        if a.startswith("rescale="):
            RESCALE = float(a.split("=")[1])
    if args:
        GUIDANCE = tuple(float(x) for x in args)
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
