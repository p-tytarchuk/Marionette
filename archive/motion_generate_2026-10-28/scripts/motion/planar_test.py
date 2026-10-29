#!/usr/bin/env python3
"""2D (sagittal-plane) test: does a model that only sees forward / up / pitch motion generate flips better?

  python3 scripts/motion/planar_test.py features   <variant>     # data/embeddings/<variant>/ with the (possibly planarised) clip features
  python3 scripts/motion/planar_test.py ae         <variant>     # motion autoencoder, same settings as data/embeddings/motion_ae.pt (150 epochs)
  python3 scripts/motion/planar_test.py diffusion  <variant>     # export latents + text-conditioned diffusion, same settings as z_diffusion.pt (8,000 steps)
  python3 scripts/motion/planar_test.py eval       prod planar control [--seeds 24]   # flips per candidate, same seeds for every variant

Variants:
  prod     the models in data/embeddings (read only)
  planar   features with everything outside the sagittal plane removed (see planarise)
  control  the unchanged features, retrained here from scratch with the same code and seeds (it came out bit-identical to prod)
  full_s1, planar_s1   (any name starting with "full" / "planar" followed by a suffix) the same two trainings with another random seed: `--seed 1` shifts every seed of the
           autoencoder and diffusion training (initialisation, batches, noise) by 1, which gives the training-to-training spread

Everything is written to data/embeddings/<variant>/; the production files are not touched.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
LINKS = ("text.npy", "text_mean.npy", "text_index.csv", "text_meta.json", "motion_feats_index.csv")
LATERAL = [10] + [13 + 3 * j for j in range(8)]            # x of the head and of the 8 limb joints (heading frame: x across, y up, z forward)


def vdir(variant):
    return EMB if variant == "prod" else EMB / variant


def planarise(F, nv, train):
    """Remove what is not forward / up / pitch: sideways velocity and yaw rate -> 0; the hips tilt keeps only its pitch (the rotation about the across axis, from the
    body-up axis; roll and twist go); the across (x) coordinate of the head and limb joints -> its training mean (stance width stays, side-to-side movement goes).
    Returns the new features and the constants."""
    G = F.copy()
    G[..., 1] = 0.0
    G[..., 2] = 0.0
    up = F[..., 7:10]
    a = np.arctan2(up[..., 2], up[..., 1])
    G[..., 4:10] = np.stack([np.ones_like(a), np.zeros_like(a), np.zeros_like(a), np.zeros_like(a), np.cos(a), np.sin(a)], -1)
    consts = {}
    for col in LATERAL:
        limb = None if col == 10 else (col - 13) // 6
        vals, present = [], []
        for i in train:
            if limb is None or F[i, 0, 37 + limb] > 0.5:
                vals.append(F[i, :nv[i], col])
        c = float(np.concatenate(vals).mean())
        consts[col] = c
        if limb is None:
            G[..., col] = c
        else:
            G[..., col] = np.where(F[..., 37 + limb] > 0.5, c, 0.0)
    return G, consts


def cmd_features(variant):
    import motion_ae as MA
    d = vdir(variant)
    d.mkdir(exist_ok=True)
    for f in LINKS:
        if not (d / f).exists():
            (d / f).symlink_to(EMB / f)
    z = np.load(EMB / "motion_feats.npz")
    F, nv = z["feats"], z["n_valid"]
    meta = list(csv.DictReader(open(EMB / "motion_feats_index.csv", newline="")))
    train = [i for i, m in enumerate(meta) if MA.split_of(m) == "train"]
    if variant.startswith("planar"):
        G, consts = planarise(F, nv, train)
        print("constants (x of head / limb joints):", {k: round(v, 3) for k, v in consts.items()})
        print("fraction of feature variance kept (planar / full), per feature group:")
        rows = np.concatenate([F[i, :nv[i]] for i in train]), np.concatenate([G[i, :nv[i]] for i in train])
        for name, cols in dict(vel=[0, 1], yaw=[2], tilt=range(4, 10), head=range(10, 13), limbs=range(13, 37)).items():
            print(f"  {name:5s} {rows[1][:, list(cols)].var(0).sum() / rows[0][:, list(cols)].var(0).sum():.3f}")
    else:
        G = F
    np.savez_compressed(d / "motion_feats.npz", **{**{k: z[k] for k in z.files}, "feats": G})
    print("wrote", d / "motion_feats.npz", G.shape)


def shift_seeds(seed):
    """motion_ae.cmd_train and diffusion_test.cmd_train seed with 0 inside the functions; add `seed` to every seed they use (torch.manual_seed, np.random.default_rng)"""
    if not seed:
        return
    import torch
    m, g = torch.manual_seed, np.random.default_rng
    torch.manual_seed = lambda s: m(s + seed)
    np.random.default_rng = lambda s=None: g(None if s is None else s + seed)


def cmd_ae(variant, epochs=150, seed=0):
    import motion_ae as MA
    shift_seeds(seed)
    d = vdir(variant)
    MA.EMB, MA.CKPT = d, d / "motion_ae.pt"
    MA.cmd_train(argparse.Namespace(epochs=epochs, batch=128, lr=3e-4, kl=1e-4, nce=0.1, tau=0.1, neg_thr=0.6))


def cmd_diffusion(variant, steps=8000, seed=0):
    import motion_ae as MA
    import diffusion_test as DT
    shift_seeds(seed)
    d = vdir(variant)
    MA.EMB, MA.CKPT = d, d / "motion_ae.pt"
    MA.cmd_export(argparse.Namespace())
    DT.EMB = d
    DT.cmd_train(argparse.Namespace(ckpt="z_diffusion.pt", para=False, para_frac=0.5, steps=steps, batch=256, lr=3e-4))
    # held-out retrieval of the autoencoder (the number that was reported for the production one)
    MA.cmd_eval(argparse.Namespace())


class Variant:
    def __init__(self, name):
        import torch
        import motion_ae as MA
        import diffusion_test as DT
        import decoded_to_character as DC
        d = vdir(name)
        self.name, self.d, self.MA, self.DT, self.DC, self.torch = name, d, MA, DT, DC, torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        MA.EMB, MA.CKPT = d, d / "motion_ae.pt"
        DT.EMB, DT.CKPT = d, d / "z_diffusion.pt"
        self.P = DT.load_pairs()
        self.ae, self.mean, self.std, _ = MA.load_model(self.dev)
        self.model, self.zm, self.zs, _ = DT.load_model(self.dev)
        tr = np.where(self.P["split"] == "train")[0]
        self.T = self.P["text_clip"][tr]
        self.nrm = np.linalg.norm((self.P["z"][tr] - self.zm) / self.zs, axis=1)

    def decode(self, zz):
        torch = self.torch
        with torch.no_grad():
            zt = torch.from_numpy(np.asarray(zz, np.float32)).to(self.dev)
            X = self.ae.decode(zt).cpu().numpy() * self.std + self.mean
            aux = self.ae.aux(zt).cpu().numpy()
        return X, np.clip(np.round(aux[:, 1] * 128), 8, 128).astype(int), np.exp(aux[:, 0])

    def cap(self, zn, E, factor=0.9, k=40):
        nn = np.argsort(-(self.T @ E[0]))[:k]
        return zn * min(1.0, factor * float(self.nrm[nn].mean()) / float(np.linalg.norm(zn)))


def cmd_eval(variants, seeds, prompts, straight=0):
    import generate_motion as G
    from embed import encode, load_text_encoder
    enc = load_text_encoder()
    mu0 = np.load(EMB / "text_mean.npy")
    ref = None
    out = {}
    for name in variants:
        V = Variant(name)
        ref = ref or V.DC.ref_skeleton()
        for prompt in prompts:
            E = encode([prompt], enc)[0] - mu0
            E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
            rows = []
            for seed in range(100, 100 + seeds):
                for k in range(4):
                    zn = V.DT.sample(V.model, V.torch, V.dev, E, guidance=2.0, seed=seed + 7919 * k)
                    zn = V.cap(zn, E)
                    X, nv_, du_ = V.decode(zn * V.zs + V.zm)
                    x, n, dur = X[0].copy(), int(nv_[0]), float(du_[0])
                    x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)
                    if straight:
                        x = G.straighten(x, straight)
                    px = V.DC.proxy_from_features(x, n, dur, ref, body_smooth=3.0)
                    sp, mx = G.proxy_steps(px)
                    rows.append((sp <= G.MAX_SPIKES and mx <= G.MAX_STEP, G.tilt_pitch_range(x, n, V.DC) >= G.ROT_RANGE, G.landing_up(x, n, V.DC) >= G.LAND_UP,
                                 sp, n, dur, float(np.abs(x[:n, 0]).sum() / dur), float(np.abs(x[:n, 1]).sum() / dur), float(np.abs(x[:n, 2]).sum())))
            A = np.array(rows, float)
            p = A[:, :3]
            per_seed = p.prod(1).reshape(seeds, 4).sum(1) > 0
            m = lambda v: f"{100 * v.mean():3.0f}%"
            print(f"{name + (f'+st{straight}' if straight else ''):10s} {prompt:34s} per candidate: smooth {m(p[:, 0])} complete {m(p[:, 1])} upright {m(p[:, 2])} all {m(p.prod(1))} | complete&upright {m(p[:, 1] * p[:, 2])} "
                  f"| seeds passing in 4 tries {per_seed.sum()}/{seeds} | spikes/clip {A[:, 3].mean():.2f} dur {A[:, 5].mean():.2f}s fwd {A[:, 6].mean():.2f} side {A[:, 7].mean():.2f} yaw {A[:, 8].mean():.2f}", flush=True)
            out[(name, prompt)] = A
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "ae", "diffusion", "eval"])
    ap.add_argument("variant", nargs="+")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--straight", type=int, default=0, help="eval: straighten the decoded clips (generate_motion.straighten): 1 = no sideways velocity / yaw, 2 = also pitch-only tilt")
    ap.add_argument("--seed", type=int, default=0, help="ae / diffusion: added to every training seed")
    ap.add_argument("--prompts", nargs="+", default=["An object performs a backflip.", "An object performs a front flip.", "An object does a cartwheel.",
                                                     "An object jumps.", "An object walks forward.", "An object runs forward."])
    a = ap.parse_args()
    if a.stage == "features":
        cmd_features(a.variant[0])
    elif a.stage == "ae":
        cmd_ae(a.variant[0], a.epochs, a.seed)
    elif a.stage == "diffusion":
        cmd_diffusion(a.variant[0], a.steps, a.seed)
    else:
        cmd_eval(a.variant, a.seeds, a.prompts, a.straight)


if __name__ == "__main__":
    main()
