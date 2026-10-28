#!/usr/bin/env python3
"""Does a clip (real, reconstructed or generated) have a stepping gait? Numbers for the "walking is weak" diagnosis.

  python3 scripts/motion/gait_check.py                 # real walking clips vs their reconstructions vs diffusion samples for walking prompts

A clip is 128 x 41 role features (scripts/motion/motion_features.py). For the two feet (index 16-18 left, 22-24 right: x lateral, y up, z forward,
relative to the hips, in hips heights):
  swing    mean over the feet of the standard deviation over time of the forward position (how far the feet move back and forth);
  alt      minus the correlation of the two feet's forward positions (1 = perfectly alternating legs, 0 = unrelated, negative = in step);
  lift     mean standard deviation of the feet's height (feet leaving the ground);
  step     swing * max(alt, 0): large only if the feet swing AND alternate; a standing figure scores near 0.
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
EMB = ROOT / "data" / "embeddings"


def gait(f, n_valid):
    f = np.asarray(f, float)[:n_valid]
    if len(f) < 8:
        return dict(swing=0.0, alt=0.0, lift=0.0, step=0.0)
    zl, zr = f[:, 18], f[:, 24]
    swing = 0.5 * (zl.std() + zr.std())
    c = np.corrcoef(zl, zr)[0, 1] if zl.std() > 1e-6 and zr.std() > 1e-6 else 0.0
    alt = -float(c)
    lift = 0.5 * (f[:, 17].std() + f[:, 23].std())
    return dict(swing=float(swing), alt=alt, lift=float(lift), step=float(swing * max(alt, 0.0)))


def summarise(name, rows, ref=None):
    a = {k: np.array([r[k] for r in rows]) for k in ("swing", "alt", "lift", "step")}
    line = f"{name:48s} n={len(rows):3d}  swing {np.median(a['swing']):.3f}  alt {np.median(a['alt']):5.2f}  lift {np.median(a['lift']):.3f}  step {np.median(a['step']):.3f}"
    if ref is not None:
        line += f"  | stepping (step > 40% of real median): {100 * np.mean(a['step'] > 0.4 * ref):3.0f}%"
    print(line)
    return np.median(a["step"])


def walking_clips(P, meta_index, feats, split, body="biped"):
    """indices into P (pairs) of walking clips with a usable length, from the captions"""
    out = []
    for i, k in enumerate(P["keys"]):
        if P["split"][i] != split or P["cls"][i] != body:
            continue
        cap = [t for t in P["captions"][i]]
        if not re.search(r"\bwalks?\b", cap[0]) or re.search(r"backward|turn|jump|stop|sit|kneel|pick|carry|crouch", cap[0]):
            continue
        n = int(P["n_valid"][i])
        if 40 <= n <= 128 and float(P["dur"][i]) < 5.0:
            out.append(i)
    return out


def main():
    import torch
    import diffusion_test as DT
    import motion_ae as MA
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    P = DT.load_pairs()
    z = np.load(EMB / "motion.npz")
    import json
    P["captions"] = [json.loads(c).get("caption", "") for c in z["captions"]]
    P["captions"] = [[c] for c in P["captions"]]
    P["n_valid"] = z["n_valid"]
    D = np.load(EMB / "motion_feats.npz")["feats"]
    import csv
    idx = {(r["dataset"], r["clip"]): i for i, r in enumerate(csv.DictReader(open(EMB / "motion_feats_index.csv", newline="")))}
    F = np.stack([D[idx[tuple(k.split("|", 1))]] for k in P["keys"]])
    ae, mean, std, _ = MA.load_model(dev)
    for split in ("train", "val"):
        w = walking_clips(P, idx, F, split)
        real = [gait(F[i], P["n_valid"][i]) for i in w]
        ref = summarise(f"REAL walking clips ({split})", real)
        with torch.no_grad():
            Z = torch.from_numpy(P["z"][w].astype(np.float32)).to(dev)
            X = ae.decode(Z).cpu().numpy() * std + mean
        rec = [gait(X[k], P["n_valid"][i]) for k, i in enumerate(w)]
        summarise(f"  their reconstructions (encode -> decode)", rec, ref)
        if split == "val":
            ref_val = ref
    # diffusion samples for walking prompts (captions of real TRAIN walking clips, so the phrasing is in distribution)
    model, zm, zs, _ = DT.load_model(dev)
    wt = walking_clips(P, idx, F, "train")
    rng = np.random.default_rng(0)
    pick = rng.choice(wt, 40, replace=False)
    T = P["text_clip"][pick]
    for g in (1.0, 2.0, 4.0):
        zn = DT.sample(model, torch, dev, T, guidance=g)
        zz = (zn * zs + zm).astype(np.float32)
        with torch.no_grad():
            X = ae.decode(torch.from_numpy(zz).to(dev)).cpu().numpy() * std + mean
            aux = ae.aux(torch.from_numpy(zz).to(dev)).cpu().numpy()
        nv = np.clip(np.round(aux[:, 1] * 128), 8, 128).astype(int)
        summarise(f"GENERATED for walking prompts, guidance {g}", [gait(X[k], nv[k]) for k in range(len(pick))], ref_val)
    # the same prompts' real clips, for the generated-vs-real comparison on identical text
    summarise("real clips behind those prompts (train)", [gait(F[i], P["n_valid"][i]) for i in pick], ref_val)


if __name__ == "__main__":
    main()
