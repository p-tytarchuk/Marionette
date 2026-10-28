#!/usr/bin/env python3
"""Semantic motion embedding for ANY body (biped, quadruped, insect, bird, fish, snake...) aligned with CLIP text.

  python3 scripts/motion/motion_sem.py --train [--epochs 60] [--holdout-class insectoid]   # -> data/embeddings/motion_sem.pt
  python3 scripts/motion/motion_sem.py --eval                                                # retrieval per body class
  python3 scripts/motion/motion_sem.py --export                                              # -> data/embeddings/motion_sem.npz
  python3 scripts/motion/motion_sem.py --query "walks" "runs" "jumps"                        # best clips per body class

Input: the joint-set features of motion_sem_features.py (no role labels). Each joint is a token: a small temporal conv
reads its 4 channels over 128 frames, a static descriptor (rest height, depth in the tree, leaf, subtree size...) says
what kind of joint it is, and a transformer over the SET of joints (no joint order, no joint names) plus a whole-body
token gives one clip vector. A linear head maps it to 512 numbers, L2-normalized and trained with InfoNCE towards the
frozen CLIP text embedding of the clip's captions (centred by the mean caption embedding). So "walks" lands near walking
motions whether the legs are 2, 4 or 8, because only the words tie the clips together, never the skeleton.
Training samples bodies evenly (biped / quadruped / other), not by clip count, so rare bodies are not drowned out.
--holdout-class X trains without that class entirely, to test whether the meaning transfers to a body never seen.
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
SEM_DIM, D, J_MAX, T_FIX = 512, 192, 128, 128
GROUP = {"biped": "biped", "quadruped": "quadruped"}          # everything else is "other"


def split_of(r, frac=10):
    if r["dataset"] == "template":                      # 65 clips on 8 rigs: all for training (too few to hold any out)
        return "train"
    key = ("m:" + re.sub(r"_\d+$", "", r["clip"])) if r["dataset"] == "mixamo" else "o:" + r["object_type"]
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < frac else "train"


def load_data():
    z = np.load(EMB / "sem_feats.npz")
    with open(EMB / "sem_feats_index.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    caps = {}
    with open(EMB / "text_index.csv", newline="") as f:
        for r in csv.DictReader(f):
            caps.setdefault((r["dataset"], r["clip"]), []).append((r["style"], int(r["row"]), r["text"]))
    temb = np.load(EMB / "text.npy")
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import text_mean
    mu = text_mean()
    temb = temb - mu
    temb = (temb / np.maximum(np.linalg.norm(temb, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    for m in meta:
        m["split"] = split_of(m)
        m["body"] = GROUP.get(m["character_class"], "other")
    return dict(joint=z["joint"], static=z["static"], offsets=z["offsets"], glob=z["glob"], scal=z["scal"], meta=meta, caps=caps, temb=temb, mu=mu)


def batch_tensors(D, idx, torch, dev):
    """Pad a batch of clips to (B, J, 128, 4) joint dynamics + (B, J, 8) static + masks."""
    off = D["offsets"]
    Js = [int(off[i + 1] - off[i]) for i in idx]
    Jm = min(max(Js), J_MAX)
    B = len(idx)
    dyn = np.zeros((B, Jm, T_FIX, 4), np.float16)
    st = np.zeros((B, Jm, 8), np.float16)
    mask = np.zeros((B, Jm), bool)
    for b, i in enumerate(idx):
        a, e = int(off[i]), int(off[i + 1])
        n = min(e - a, Jm)
        sel = np.arange(a, e) if e - a <= Jm else a + np.sort(np.random.choice(e - a, Jm, replace=False))
        dyn[b, :n], st[b, :n], mask[b, :n] = D["joint"][sel], D["static"][sel], True
    t = lambda x: torch.from_numpy(x).to(dev)
    return (t(dyn).float(), t(st).float(), t(mask), t(D["glob"][idx]).float(), t(D["scal"][idx]).float())


def build_model():
    import torch
    import torch.nn as nn

    class JointEnc(nn.Module):
        """4 channels x 128 frames of one joint -> one D-vector (shared by all joints of all bodies)."""
        def __init__(self, c_in):
            super().__init__()
            ch = [c_in, 32, 64, 96]
            L = []
            for a, b in zip(ch[:-1], ch[1:]):
                L += [nn.Conv1d(a, b, 5, stride=2, padding=2), nn.GELU()]
            self.net = nn.Sequential(*L)                                          # 128 -> 16 steps
            self.fc = nn.Linear(96 * 16, D)

        def forward(self, x):                                                    # (N, T, C)
            return self.fc(self.net(x.transpose(1, 2)).flatten(1))

    class SemMotion(nn.Module):
        def __init__(self):
            super().__init__()
            self.jenc, self.genc = JointEnc(4), JointEnc(5)
            self.stat = nn.Sequential(nn.Linear(8, D), nn.GELU(), nn.Linear(D, D))
            self.scal = nn.Sequential(nn.Linear(8, D), nn.GELU(), nn.Linear(D, D))
            self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            layer = nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 4)
            self.norm = nn.LayerNorm(D)
            self.sem = nn.Linear(D, SEM_DIM)

        def forward(self, dyn, st, mask, glob, scal):
            B, J = dyn.shape[:2]
            h = self.jenc(dyn.flatten(0, 1)).view(B, J, D) + self.stat(st)
            g = (self.genc(glob) + self.scal(scal))[:, None]                      # whole-body token
            x = torch.cat([self.cls.expand(B, -1, -1), g, h], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=dyn.device), ~mask], 1)
            x = self.norm(self.tf(x, src_key_padding_mask=pad))
            return torch.nn.functional.normalize(self.sem(x[:, 0]), dim=-1)

    return SemMotion


def info_nce(a, b, mask, tau):
    import torch
    import torch.nn.functional as Fn
    s = a @ b.T / tau
    eye = torch.eye(len(a), dtype=torch.bool, device=a.device)
    s = s.masked_fill(mask & ~eye, -1e4)
    t = torch.arange(len(a), device=a.device)
    return 0.5 * (Fn.cross_entropy(s, t) + Fn.cross_entropy(s.T, t))


def pick_text(D, i, rng):
    caps = D["caps"][(D["meta"][i]["dataset"], D["meta"][i]["clip"])]
    return caps[rng.integers(len(caps))][1]


def encode_all(model, D, idx, dev, torch, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            out.append(model(*batch_tensors(D, idx[b:b + batch], torch, dev)).cpu())
    return torch.cat(out).numpy()


def ranks(sim):
    return (sim > np.diag(sim)[:, None]).sum(1)


def recall(r):
    return {"R@1": 100 * (r < 1).mean(), "R@5": 100 * (r < 5).mean(), "R@10": 100 * (r < 10).mean(), "medR": float(np.median(r)), "n": len(r)}


def text_of(D, i, style="caption"):
    return [r for s, r, _ in D["caps"][(D["meta"][i]["dataset"], D["meta"][i]["clip"])] if s == style][0]


def evaluate(model, D, idx, dev, torch):
    """Retrieval inside the held-out set: pooled, and inside each body group (gallery = that group only)."""
    emb = encode_all(model, D, idx, dev, torch)
    T = np.stack([D["temb"][text_of(D, i)] for i in idx])
    out = {"all": {"t2m": recall(ranks(T @ emb.T)), "m2t": recall(ranks(emb @ T.T))}}
    for g in ("biped", "quadruped", "other"):
        sel = np.array([k for k, i in enumerate(idx) if D["meta"][i]["body"] == g])
        if len(sel) >= 10:
            s = T[sel] @ emb[sel].T
            out[g] = {"t2m": recall(ranks(s)), "m2t": recall(ranks(s.T))}
    return out


def cmd_train(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    D = load_data()
    held = args.holdout_class
    held_obj = args.holdout_object
    tr = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "train" and m["character_class"] != held and m["object_type"] != held_obj])
    va = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "val"])
    va_held = np.array([i for i, m in enumerate(D["meta"]) if held and m["character_class"] == held])      # the never-seen body: all of its clips
    groups = {g: np.array([i for i in tr if D["meta"][i]["body"] == g]) for g in ("biped", "quadruped", "other")}
    print(f"{len(D['meta'])} clips; train {len(tr)} " + str({g: len(v) for g, v in groups.items()}) + f"; held-out {len(va)}; never-seen class {held}: {len(va_held)}; device {dev}")
    model = build_model()().to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    per_epoch = math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * per_epoch, pct_start=0.05)
    w = np.array([args.balance[0], args.balance[1], args.balance[2]], float)
    w /= w.sum()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for _ in range(per_epoch):
            parts = rng.choice(3, args.batch, p=w)                                # sample bodies evenly, then a clip of that body
            idx = np.array([rng.choice(groups[("biped", "quadruped", "other")[p]]) for p in parts])
            x = batch_tensors(D, idx, torch, dev)
            T = torch.from_numpy(D["temb"][[pick_text(D, i, rng) for i in idx]]).to(dev)
            e = model(*x)
            loss = info_nce(e, T, (T @ T.T) > args.neg_thr, args.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss.detach())
        if ep % 5 == 0 or ep == args.epochs - 1:
            r = evaluate(model, D, va, dev, torch)
            line = " ".join(f"{g} {r[g]['t2m']['R@5']:.0f}" for g in r)
            print(f"epoch {ep:3d} loss {loss_sum / per_epoch:.3f} | held-out text->motion R@5 by body: {line} [{time.time() - t0:.0f}s]", flush=True)
    path = EMB / (f"motion_sem_no_{held}.pt" if held else f"motion_sem_no_{held_obj}.pt" if held_obj else "motion_sem.pt")
    torch.save(dict(model=model.state_dict(), args={k: (v if not isinstance(v, tuple) else list(v)) for k, v in vars(args).items()}), path)
    print("saved", path)
    if held:
        zero_shot_report(model, D, va_held, dev, torch)


def zero_shot_report(model, D, idx, dev, torch):
    """How well does a body the model never saw get found by its own captions? (gallery = those clips only, and all val clips)"""
    emb = encode_all(model, D, idx, dev, torch)
    T = np.stack([D["temb"][text_of(D, i)] for i in idx])
    r = recall(ranks(T @ emb.T))
    print(f"never-seen class, text->motion within its {len(idx)} clips: R@1 {r['R@1']:.1f}%  R@5 {r['R@5']:.1f}%  R@10 {r['R@10']:.1f}%  median rank {r['medR']:.0f} (chance median {len(idx) / 2:.0f})")


def load_model(path, dev):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()().to(dev)
    model.load_state_dict(ck["model"])
    return model.eval(), torch


def cmd_eval(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    path = EMB / (f"motion_sem_no_{args.holdout_class}.pt" if args.holdout_class else "motion_sem.pt")
    model, torch = load_model(path, dev)
    D = load_data()
    va = np.array([i for i, m in enumerate(D["meta"]) if m["split"] == "val"])
    r = evaluate(model, D, va, dev, torch)
    f = lambda d: f"R@1 {d['R@1']:5.1f}%  R@5 {d['R@5']:5.1f}%  R@10 {d['R@10']:5.1f}%  median rank {d['medR']:4.0f} of {d['n']}"
    for g, d in r.items():
        print(f"{g:9s} text->motion {f(d['t2m'])}\n{'':9s} motion->text {f(d['m2t'])}   (chance median {d['t2m']['n'] / 2:.0f})")
    if args.holdout_class:
        idx = np.array([i for i, m in enumerate(D["meta"]) if m["character_class"] == args.holdout_class])
        zero_shot_report(model, D, idx, dev, torch)


def cmd_export(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, torch = load_model(EMB / "motion_sem.pt", dev)
    D = load_data()
    idx = np.arange(len(D["meta"]))
    emb = encode_all(model, D, idx, dev, torch)
    M = D["meta"]
    np.savez_compressed(EMB / "motion_sem.npz", key=np.array([f"{m['dataset']}|{m['clip']}" for m in M]), sem=emb.astype(np.float32),
                        cls=np.array([m["character_class"] for m in M]), split=np.array([m["split"] for m in M]),
                        captions=np.array([json.dumps({s: t for s, r, t in D["caps"][(m["dataset"], m["clip"])]}) for m in M]),
                        text_clip=np.stack([D["temb"][text_of(D, i)] for i in idx]).astype(np.float32), text_mean=D["mu"].astype(np.float32),
                        duration_s=np.array([float(m["duration_s"] or 0) for m in M], np.float32))
    print(f"{len(M)} clips -> {EMB / 'motion_sem.npz'}  sem {emb.shape}")


def cmd_query(args):
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import encode, load_text_encoder
    z = np.load(EMB / "motion_sem.npz")
    from embed import text_mean
    mu = text_mean()
    enc = load_text_encoder()
    classes = args.cls or ["biped", "quadruped", "insectoid", "avian", "marine", "serpentine"]
    for q in args.query:
        e = encode([q], enc)[0][0] - mu
        e /= np.linalg.norm(e)
        s = z["sem"] @ e
        print(f"\n{q!r}")
        for c in classes:
            ii = np.where(z["cls"] == c)[0]
            if not len(ii):
                continue
            for i in ii[np.argsort(-s[ii])][:args.top]:
                cap = json.loads(z["captions"][i]).get("caption", "")
                print(f"  {c:10s} {s[i]:6.3f}  {z['key'][i].split('|')[1][:34]:34s} {cap[:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--cls", nargs="+")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--holdout-class", help="train without this character_class (e.g. insectoid) to test transfer to an unseen body")
    ap.add_argument("--holdout-object", help="train without this object_type (e.g. spider, a template rig) to test one body")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--neg-thr", type=float, default=0.6)
    ap.add_argument("--balance", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="sampling weights biped quadruped other")
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
