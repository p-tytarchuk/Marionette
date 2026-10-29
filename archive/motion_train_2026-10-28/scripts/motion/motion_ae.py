#!/usr/bin/env python3
"""Semantic motion embedding + decodable latent (TMR-style), trained on the clip features of motion_features.py.

  python3 scripts/motion/motion_ae.py --train [--epochs 200]     # -> data/embeddings/motion_ae.pt
  python3 scripts/motion/motion_ae.py --eval                     # reconstruction + text<->motion retrieval on held-out clips
  python3 scripts/motion/motion_ae.py --export                   # -> data/embeddings/motion.npz (every clip: sem, z, captions)
  python3 scripts/motion/motion_ae.py --query "a backflip" --cls quadruped

One transformer encoder reads a clip (128 x 41 features) and gives
  z    (256)  the latent mean of a variational autoencoder; a decoder rebuilds the features from it, so a diffusion
              model can generate z and the decoder turns it into motion (role-space features, see docs/EMBED.md);
  sem  (512)  Linear(z), L2-normalized and trained with InfoNCE towards the FROZEN CLIP text embedding of the clip's
              captions (centred by the mean caption embedding), so any prompt embedded with CLIP lands near the motions
              that match it (retrieval, and a conditioning signal that tolerates new phrasings).
Held out: ~10% of Objaverse rigs and ~10% of Mixamo clip families, never seen in training. Negatives whose caption is
close to the positive's are masked (many clips share the same few captions), as in TMR.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
CKPT = EMB / "motion_ae.pt"
D_MODEL, Z_DIM, SEM_DIM, T_FIX = 256, 256, 512, 128
STYLES = ("caption", "generic", "detail")


# ----------------------------------------------------------------------------- data
def split_of(row, frac=10):
    """'val' for ~10% of rigs (Objaverse) or clip families (Mixamo: 'Jump', 'Jump_2'... together), else 'train'."""
    key = ("m:" + re.sub(r"_\d+$", "", row["clip"])) if row["dataset"] == "mixamo" else "o:" + row["rig"]
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < frac else "train"


def load_data(text_source="dataset"):
    z = np.load(EMB / "motion_feats.npz")
    feats, n_valid, dur = z["feats"], z["n_valid"], z["duration_s"]
    with open(EMB / "motion_feats_index.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    temb = np.load(EMB / "text.npy")
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import text_mean
    mu = text_mean()
    temb = temb - mu
    temb /= np.maximum(np.linalg.norm(temb, axis=1, keepdims=True), 1e-9)
    caps = {}
    with open(EMB / "text_index.csv", newline="") as f:
        for r in csv.DictReader(f):
            caps.setdefault((r["dataset"], r["clip"]), []).append((r["style"], int(r["row"]), r["text"]))
    keep = [i for i, m in enumerate(meta) if (m["dataset"], m["clip"]) in caps]
    meta = [meta[i] for i in keep]
    feats, n_valid, dur = feats[keep], n_valid[keep], dur[keep]
    for m in meta:
        m["split"] = split_of(m)
    return dict(feats=feats, n_valid=n_valid, dur=dur, meta=meta, caps=caps, temb=temb.astype(np.float32), mu=mu)


def standardise(feats, n_valid, train_idx):
    """Per-feature mean/std over the real (unpadded) frames of the training clips; the presence flags stay as they are."""
    rows = np.concatenate([feats[i, :n_valid[i]] for i in train_idx])
    mean, std = rows.mean(0), rows.std(0) + 1e-4
    mean[-4:], std[-4:] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------------- model
def build_model():
    import torch
    import torch.nn as nn

    class MotionAE(nn.Module):
        def __init__(self, f_in):
            super().__init__()
            self.inp = nn.Linear(f_in, D_MODEL)
            self.pos = nn.Parameter(torch.randn(T_FIX + 2, D_MODEL) * 0.02)
            self.tok = nn.Parameter(torch.randn(2, D_MODEL) * 0.02)             # mu / logvar tokens
            enc = nn.TransformerEncoderLayer(D_MODEL, 4, 512, 0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, 4)
            self.to_mu, self.to_lv = nn.Linear(D_MODEL, Z_DIM), nn.Linear(D_MODEL, Z_DIM)
            self.sem = nn.Linear(Z_DIM, SEM_DIM)
            self.aux = nn.Linear(Z_DIM, 2)                                       # log duration, valid fraction
            self.mem = nn.Linear(Z_DIM, 4 * D_MODEL)                            # the latent as 4 memory tokens
            self.q = nn.Parameter(torch.randn(T_FIX, D_MODEL) * 0.02)
            dec = nn.TransformerDecoderLayer(D_MODEL, 4, 512, 0.1, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(dec, 4)
            self.out = nn.Linear(D_MODEL, f_in)

        def encode(self, x, valid):
            B = x.shape[0]
            h = self.inp(x) + self.pos[2:][None]
            h = torch.cat([(self.tok + self.pos[:2])[None].expand(B, -1, -1), h], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), ~valid], 1)
            h = self.enc(h, src_key_padding_mask=pad)
            return self.to_mu(h[:, 0]), self.to_lv(h[:, 1])

        def decode(self, z):
            B = z.shape[0]
            mem = self.mem(z).view(B, 4, D_MODEL)
            return self.out(self.dec(self.q[None].expand(B, -1, -1), mem))

    return MotionAE


def make_batch(D, idx, mean, std, rng, dev, torch):
    x = torch.from_numpy((D["feats"][idx] - mean) / std).to(dev)
    nv = D["n_valid"][idx]
    valid = torch.from_numpy(np.arange(T_FIX)[None] < nv[:, None]).to(dev)
    txt = []
    for i in idx:
        caps = D["caps"][(D["meta"][i]["dataset"], D["meta"][i]["clip"])]
        txt.append(caps[rng.integers(len(caps))][1])
    return x, valid, torch.from_numpy(D["temb"][txt]).to(dev), torch.from_numpy(np.log(np.maximum(D["dur"][idx], 1e-2))).to(dev), nv


def info_nce(a, b, mask, tau):
    """Symmetric InfoNCE between motion embeddings a and text embeddings b; mask[i, j] = j is a near-duplicate of i (not a negative)."""
    import torch
    import torch.nn.functional as Fn
    s = a @ b.T / tau
    eye = torch.eye(len(a), dtype=torch.bool, device=a.device)
    s = s.masked_fill(mask & ~eye, -1e4)
    t = torch.arange(len(a), device=a.device)
    return 0.5 * (Fn.cross_entropy(s, t) + Fn.cross_entropy(s.T, t))


def cmd_train(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    D = load_data()
    tr = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "train"])
    va = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "val"])
    print(f"{len(D['meta'])} clips: {len(tr)} train, {len(va)} held out; device {dev}")
    mean, std = standardise(D["feats"], D["n_valid"], tr)
    model = build_model()(D["feats"].shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = dict(rec=0.0, kl=0.0, nce=0.0, aux=0.0)
        n = 0
        for b in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[b:b + args.batch]
            x, valid, T, ldur, nv = make_batch(D, idx, mean, std, rng, dev, torch)
            mu, lv = model.encode(x, valid)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            xr = model.decode(z)
            w = valid[..., None].float()
            rec = (torch.nn.functional.smooth_l1_loss(xr, x, reduction="none") * w).sum() / (w.sum() * x.shape[-1])
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            sem = torch.nn.functional.normalize(model.sem(mu), dim=-1)
            mask = (T @ T.T) > args.neg_thr
            nce = info_nce(sem, T, mask, args.tau)
            aux = model.aux(z)
            auxl = torch.nn.functional.mse_loss(aux[:, 0], ldur) + torch.nn.functional.mse_loss(aux[:, 1], torch.from_numpy(nv / T_FIX).float().to(dev))
            loss = rec + args.kl * kl + args.nce * nce + 0.1 * auxl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in zip(tot, (rec, kl, nce, auxl)):
                tot[k] += float(v)
            n += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            msg = " ".join(f"{k} {v / n:.4f}" for k, v in tot.items())
            r = evaluate(model, D, va, mean, std, dev, torch, quick=True)
            print(f"epoch {ep:3d}  {msg}  | held-out recon {r['recon_smoothl1']:.4f}  T->M R@5 {r['t2m']['R@5']:.0f}%  [{time.time() - t0:.0f}s]", flush=True)
    torch.save(dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std), args=vars(args), f_in=int(D["feats"].shape[2])), CKPT)
    print("saved", CKPT)


# ----------------------------------------------------------------------------- eval
def encode_all(model, D, idx, mean, std, dev, torch, batch=256):
    model.eval()
    mus, sems = [], []
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            ii = idx[b:b + batch]
            x = torch.from_numpy((D["feats"][ii] - mean) / std).to(dev)
            valid = torch.from_numpy(np.arange(T_FIX)[None] < D["n_valid"][ii][:, None]).to(dev)
            mu, _ = model.encode(x, valid)
            mus.append(mu.cpu())
            sems.append(torch.nn.functional.normalize(model.sem(mu), dim=-1).cpu())
    return torch.cat(mus).numpy(), torch.cat(sems).numpy()


def ranks(sim):
    """rank of the true pair (diagonal) among its row: how many scores are strictly higher."""
    return (sim > np.diag(sim)[:, None]).sum(1)


def recall(r, name=None):
    return {"R@1": 100 * (r < 1).mean(), "R@5": 100 * (r < 5).mean(), "R@10": 100 * (r < 10).mean(), "medR": float(np.median(r)), "n": len(r)}


def evaluate(model, D, idx, mean, std, dev, torch, quick=False, per_class=False):
    import torch.nn.functional as Fn
    mu, sem = encode_all(model, D, idx, mean, std, dev, torch)
    T = np.stack([D["temb"][[r for s, r, _ in D["caps"][(D["meta"][i]["dataset"], D["meta"][i]["clip"])] if s == "caption"][0]] for i in idx])
    sim = T @ sem.T                                                      # text i vs motion j
    out = {"t2m": recall(ranks(sim)), "m2t": recall(ranks(sim.T))}
    # reconstruction on held-out clips (decode the mean latent)
    with torch.no_grad():
        x = torch.from_numpy((D["feats"][idx] - mean) / std).to(dev)
        xr = model.decode(torch.from_numpy(mu).to(dev))
        valid = torch.from_numpy(np.arange(T_FIX)[None] < D["n_valid"][idx][:, None]).to(dev)
        w = valid[..., None].float()
        out["recon_smoothl1"] = float((Fn.smooth_l1_loss(xr, x, reduction="none") * w).sum() / (w.sum() * x.shape[-1]))
        if not quick:
            raw_err = ((xr.cpu().numpy() * std + mean) - D["feats"][idx])
            wv = valid.cpu().numpy()[..., None]
            names = list(np.load(EMB / "motion_feats.npz")["names"])
            groups = dict(root_velocity=[0, 1], yaw_rate=[2], hips_height=[3], hips_tilt=range(4, 10), head=range(10, 13), limb_joints=range(13, 37))
            base = ((D["feats"][idx] - mean) * 0 + 0)                    # error of predicting the training mean
            out["recon_mae"] = {g: float((np.abs(raw_err[..., list(c)]) * wv).sum() / (wv.sum() * len(list(c)))) for g, c in groups.items()}
            out["mean_baseline_mae"] = {g: float((np.abs(D["feats"][idx][..., list(c)] - mean[list(c)]) * wv).sum() / (wv.sum() * len(list(c)))) for g, c in groups.items()}
    if per_class:
        out["by_body"] = {}
        for body in ("biped", "quadruped"):
            sel = np.array([k for k, i in enumerate(idx) if D["meta"][i]["body"] == body])
            if len(sel) > 5:
                s = sim[np.ix_(sel, sel)]
                out["by_body"][body] = {"t2m": recall(ranks(s)), "m2t": recall(ranks(s.T))}
    return out


def load_model(dev):
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)          # tensors and plain values only
    model = build_model()(ck["f_in"]).to(dev)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["mean"].numpy(), ck["std"].numpy(), torch


def cmd_eval(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, mean, std, torch = load_model(dev)
    D = load_data()
    va = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "val"])
    r = evaluate(model, D, va, mean, std, dev, torch, per_class=True)
    print(f"held-out clips: {len(va)} (rigs / clip families never seen in training)")
    f = lambda d: f"R@1 {d['R@1']:.1f}%  R@5 {d['R@5']:.1f}%  R@10 {d['R@10']:.1f}%  median rank {d['medR']:.0f} of {d['n']}"
    print("text -> motion:", f(r["t2m"]))
    print("motion -> text:", f(r["m2t"]))
    for b, d in r.get("by_body", {}).items():
        print(f"  {b:9s} (gallery = that class only)  t2m {f(d['t2m'])}   m2t {f(d['m2t'])}")
    print(f"reconstruction (smooth-L1, standardised): {r['recon_smoothl1']:.4f}")
    print("reconstruction MAE (H units / radians) vs predicting the training mean:")
    for g in r["recon_mae"]:
        print(f"  {g:14s} {r['recon_mae'][g]:.4f}   (mean baseline {r['mean_baseline_mae'][g]:.4f})")
    # chance for reference
    n = r["t2m"]["n"]
    print(f"chance: R@1 {100 / n:.2f}%  R@5 {500 / n:.2f}%  R@10 {1000 / n:.2f}%  median rank {n / 2:.0f}")


def cmd_export(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, mean, std, torch = load_model(dev)
    D = load_data()
    idx = np.arange(len(D["meta"]))
    mu, sem = encode_all(model, D, idx, mean, std, dev, torch)
    keys = [f"{m['dataset']}|{m['clip']}" for m in D["meta"]]
    caps = [json.dumps({s: t for s, r, t in D["caps"][(m["dataset"], m["clip"])]}) for m in D["meta"]]
    text_row = np.array([[r for s, r, t in D["caps"][(m["dataset"], m["clip"])] if s == "caption"][0] for m in D["meta"]])
    np.savez_compressed(EMB / "motion.npz", key=np.array(keys), z=mu.astype(np.float32), sem=sem.astype(np.float32),
                        text_clip=D["temb"][text_row], captions=np.array(caps), split=np.array([m["split"] for m in D["meta"]]),
                        cls=np.array([m["body"] for m in D["meta"]]), duration_s=D["dur"], n_valid=D["n_valid"],
                        text_mean=D["mu"].astype(np.float32))
    print(f"{len(keys)} clips -> {EMB / 'motion.npz'}  z {mu.shape}  sem {sem.shape}")


def cmd_query(args):
    import torch
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import encode, load_text_encoder
    z = np.load(EMB / "motion.npz")
    from embed import text_mean
    mu = text_mean()
    enc = load_text_encoder()
    for q in args.query:
        e = encode([q], enc)[0][0] - mu
        e /= np.linalg.norm(e)
        s = z["sem"] @ e
        order = np.argsort(-s)
        print(f"\n{q!r}")
        shown = 0
        for i in order:
            if args.cls and z["cls"][i] != args.cls:
                continue
            caps = json.loads(z["captions"][i])
            print(f"  {s[i]:6.3f}  {z['split'][i]:5s} {z['key'][i].split('|')[1][:40]:40s} {caps.get('caption', '')[:70]}")
            shown += 1
            if shown >= args.top:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--cls", choices=["biped", "quadruped"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=1e-4)
    ap.add_argument("--nce", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--neg-thr", type=float, default=0.6, help="mask negatives whose caption is this similar (centred CLIP cosine) to the positive's")
    args = ap.parse_args()
    if args.train:
        cmd_train(args)
    elif args.eval:
        cmd_eval(args)
    elif args.export:
        cmd_export(args)
    elif args.query:
        cmd_query(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
