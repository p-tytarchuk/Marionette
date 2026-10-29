#!/usr/bin/env python3
"""Decode latents of the role model (scripts/motion/motion_ae.py) and draw them, to see what the decoder actually produces.

  python3 scripts/motion/decode_latents.py            # -> docs/figures/decoded_latents.png + data/embeddings/decoded_examples.npz

Three kinds of example (all in the role-space the decoder outputs, drawn as a stick figure: hips, head, a chest point
for the arms, and the knee/foot and elbow/hand of each limb; left blue, right orange; the hips path is the integrated
root velocity and yaw rate):
  1. held-out clips (rigs never seen in training), original features above, encode -> decode below;
  2. a straight line between the latents of two clips, decoded at 25%, 50%, 75%;
  3. random latents drawn from a Gaussian fitted to the training latents (what a diffusion model would hand to the decoder),
     each labelled with the captions its semantic head `sem` is nearest to.
A figure with the stick skeletons is NOT an animation on a character: the decoded output is role features, and turning it
into joint rotations needs an IK step that is not built.
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
import motion_ae as MA
import render_motion as RM
from PIL import Image, ImageDraw, ImageFont

OUT_PNG = ROOT / "docs" / "figures" / "decoded_latents.png"
LEFT, RIGHT, MID = RM.LEFT, RM.RIGHT, RM.MID
# joint order: 0 hips, 1 head, 2 chest (point on the spine, arms start here), 3 legL knee, 4 legL foot, 5 legR knee, 6 legR foot,
#              7 armL elbow, 8 armL hand, 9 armR elbow, 10 armR hand
PARENTS = [-1, 0, 0, 0, 3, 0, 5, 2, 7, 2, 9]
COLOURS = [MID, MID, MID, LEFT, LEFT, RIGHT, RIGHT, LEFT, LEFT, RIGHT, RIGHT]


def features_to_skeleton(f, n_valid, duration_s):
    """(128, 41) role features -> the dict render_motion.render() draws. Lengths are in hips heights (H = 1)."""
    f = np.asarray(f, dtype=float)[:n_valid]
    fwd, side, yaw, hh = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    psi = np.cumsum(yaw)
    c, s = np.cos(psi), np.sin(psi)
    hips = np.stack([np.cumsum(fwd * s + side * c), hh, np.cumsum(fwd * c - side * s)], -1)

    def world(local):                                           # heading frame -> world (rotate by psi about Y)
        return hips + np.stack([local[:, 0] * c + local[:, 2] * s, local[:, 1], -local[:, 0] * s + local[:, 2] * c], -1)

    head = f[:, 10:13]
    limb = lambda k: f[:, 13 + 3 * k:16 + 3 * k]                # 0 legL knee, 1 legL foot, 2 legR knee, 3 legR foot, 4-7 arms
    has = f[:, 37:41].mean(0) > 0.5
    P = np.zeros((len(f), 11, 3))
    P[:, 0] = hips
    P[:, 1] = world(head)
    P[:, 2] = world(0.75 * head)
    for j, k, ok in ((3, 0, has[0]), (4, 1, has[0]), (5, 2, has[1]), (6, 3, has[1]), (7, 4, has[2]), (8, 5, has[2]), (9, 6, has[3]), (10, 7, has[3])):
        P[:, j] = world(limb(k)) if ok else P[:, 0 if j < 7 else 2]
    return dict(names=[str(i) for i in range(11)], parents=PARENTS, Wp=P, fps=max(n_valid / max(duration_s, 1e-3), 1.0), hips=0, colours=COLOURS)


def strip(f, n_valid, duration_s, title, sub=""):
    """One row of 8 frames with a caption bar on the left."""
    RM.COLS, RM.ROWS = 8, 1
    sk = features_to_skeleton(f, n_valid, duration_s)
    img = RM.render(sk, n_frames=8, panel=(150, 190))
    bar = Image.new("RGB", (330, img.height), RM.BG)
    d = ImageDraw.Draw(bar)
    try:
        big, small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15), ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        big = small = ImageFont.load_default()
    d.text((10, 8), title, fill=(230, 233, 240), font=big)
    y = 32
    for ln in sub.split("\n"):
        d.text((10, y), ln, fill=(150, 158, 175), font=small)
        y += 16
    out = Image.new("RGB", (bar.width + img.width, img.height), RM.BG)
    out.paste(bar, (0, 0))
    out.paste(img, (bar.width, 0))
    return out


def wrap(text, n=44):
    import textwrap
    return "\n".join(textwrap.wrap(text, n)[:4])


def motion_energy(model, D, mean, std, idx, dev, torch):
    """How much movement the decoder keeps: per held-out clip, the standard deviation over time of the limb joints (decoded / original),
    and the root travel (sum of forward speed, decoded / original). 1 = kept, 0 = frozen."""
    MAe = MA
    mu = MAe.encode_all(model, D, np.asarray(idx), mean, std, dev, torch)[0]
    with torch.no_grad():
        X = model.decode(torch.from_numpy(mu).to(dev)).cpu().numpy() * std + mean
    amp, trav = [], []
    for k, i in enumerate(idx):
        n = int(D["n_valid"][i])
        o, d = D["feats"][i, :n], X[k, :n]
        a0, a1 = o[:, 13:37].std(0).mean(), d[:, 13:37].std(0).mean()
        amp.append(a1 / max(a0, 1e-6))
        t0, t1 = np.abs(o[:, 0]).sum(), np.abs(d[:, 0]).sum()
        trav.append(t1 / t0 if t0 > 0.3 else np.nan)
    return np.array(amp), np.array(trav)


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, mean, std, torch = MA.load_model(dev)
    D = MA.load_data()
    meta = D["meta"]
    va = [i for i, m in enumerate(meta) if m["split"] == "val"]
    tr = np.array([i for i, m in enumerate(meta) if m["split"] == "train"])

    def encode(idx):
        return MA.encode_all(model, D, np.asarray(idx), mean, std, dev, torch)[0]       # latent means (n, 256)

    def decode(z):
        with torch.no_grad():
            zt = torch.from_numpy(np.asarray(z, np.float32)).to(dev)
            x = model.decode(zt).cpu().numpy() * std + mean
            aux = model.aux(zt).cpu().numpy()
            sem = torch.nn.functional.normalize(model.sem(zt), dim=-1).cpu().numpy()
        x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)
        nv = np.clip(np.round(aux[:, 1] * MA.T_FIX), 8, MA.T_FIX).astype(int)
        return x, nv, np.exp(aux[:, 0]), sem

    caption = lambda i: [t for s, r, t in D["caps"][(meta[i]["dataset"], meta[i]["clip"])] if s == "caption"][0]

    # ---- 0. how much movement survives, over every held-out clip
    amp, trav = motion_energy(model, D, mean, std, va, dev, torch)
    cyc = np.array([bool(re.search(r"walk|run|jog|march|trot|step", caption(i))) for i in va])
    sl = lambda a, m: f"median {np.nanmedian(a[m]):.2f}, below 0.5 for {100 * np.nanmean(a[m] < 0.5):.0f}% (n={int(np.isfinite(a[m]).sum())})"
    print(f"movement kept (limb-joint spread over time, decoded / original; 1 = same, 0 = frozen), {len(va)} held-out clips:")
    print(f"  all clips        {sl(amp, np.ones(len(va), bool))}")
    print(f"  walk/run captions {sl(amp, cyc)}")
    print(f"  other captions    {sl(amp, ~cyc)}")
    print(f"root travel kept (decoded / original, clips that travel more than 0.3 hips heights): all {sl(trav, np.ones(len(va), bool))}; walk/run {sl(trav, cyc)}")

    # ---- 1. held-out clips: pick one per action word (deterministic: the first held-out clip whose caption has it and lasts 1-4 s)
    picks, used = [], set()
    for kw in ("walks", "runs", "jumps", "flip", "kicks", "sits", "waves", "punch"):
        for i in va:
            if i in used or not re.search(kw, caption(i)) or not (30 <= D["n_valid"][i] <= 120) or meta[i]["body"] != "biped":
                continue
            picks.append(i)
            used.add(i)
            break
    quad = [i for i in va if meta[i]["body"] == "quadruped" and D["n_valid"][i] >= 25]
    if quad:
        picks.append(quad[0])
    Z = encode(picks)
    X, NV, DUR, _ = decode(Z)
    rows, saved = [], {}
    for k, i in enumerate(picks):
        cap = caption(i)
        err = np.abs(X[k, :D["n_valid"][i]] - D["feats"][i, :D["n_valid"][i]])
        rows.append(strip(D["feats"][i], D["n_valid"][i], D["dur"][i], f"original: {meta[i]['clip'][:28]}", wrap(f'"{cap}"\n{meta[i]["body"]}, {D["dur"][i]:.1f} s')))
        rows.append(strip(X[k], D["n_valid"][i], D["dur"][i], "decoded from its latent", f"mean error: pose {np.abs(err[:, 13:37]).mean():.3f}, travel {np.abs(err[:, :2]).mean():.4f}\n(hips heights; clip travels {np.abs(D['feats'][i, :D['n_valid'][i], 0]).sum():.2f})"))
        saved[f"clip{k}_key"] = meta[i]["clip"]
        saved[f"clip{k}_orig"], saved[f"clip{k}_dec"] = D["feats"][i], X[k]
    # ---- 2. interpolation between the first two picks
    if len(picks) >= 2:
        za, zb = Z[0], Z[min(2, len(Z) - 1)]
        for a in (0.25, 0.5, 0.75):
            z = (1 - a) * za + a * zb
            x, nv, du, sem = decode(z[None])
            rows.append(strip(x[0], int(nv[0]), float(du[0]), f"latent {int((1 - a) * 100)}% A + {int(a * 100)}% B", f"A = {meta[picks[0]]['clip'][:22]}\nB = {meta[picks[min(2, len(picks) - 1)]]['clip'][:22]}"))
    # ---- 3. random latents from a Gaussian fitted to the training latents
    Ztr = encode(tr)
    rng = np.random.default_rng(1)
    cov = np.cov(Ztr.T) + 1e-6 * np.eye(Ztr.shape[1])
    Zr = rng.multivariate_normal(Ztr.mean(0), cov, 4)
    X2, NV2, DU2, S2 = decode(Zr)
    temb = np.load(MA.EMB / "text.npy")
    temb = temb - np.load(MA.EMB / "text_mean.npy")
    temb /= np.maximum(np.linalg.norm(temb, axis=1, keepdims=True), 1e-9)
    import csv
    texts = {}
    with open(MA.EMB / "text_index.csv", newline="") as f:
        for r in csv.DictReader(f):
            texts.setdefault(int(r["row"]), r["text"])
    for k in range(len(Zr)):
        top = np.argsort(-(temb @ S2[k]))[:3]
        sub = "reads as:\n" + "\n".join(f'"{texts.get(int(t), "?")[:40]}"' for t in top)
        rows.append(strip(X2[k], int(NV2[k]), float(DU2[k]), f"random latent {k + 1}", sub))
        saved[f"random{k}_dec"] = X2[k]
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height + 4 for r in rows)), RM.BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT_PNG)
    if len(rows) == 25:                               # a compact sheet for the docs: good cases, one weak case, a blend, random latents
        pick = [12, 13, 6, 7, 14, 15, 0, 1, 19, 21, 23]
        sm = Image.new("RGB", (W, sum(rows[i].height + 4 for i in pick)), RM.BG)
        y = 0
        for i in pick:
            sm.paste(rows[i], (0, y))
            y += rows[i].height + 4
        sm = sm.resize((int(sm.width * 0.72), int(sm.height * 0.72)), Image.LANCZOS)
        sm.save(OUT_PNG.with_name("decoded_latents_summary.png"))
    np.savez_compressed(MA.EMB / "decoded_examples.npz", **saved)
    print(f"{len(rows)} strips -> {OUT_PNG}  {sheet.size}")


if __name__ == "__main__":
    main()
