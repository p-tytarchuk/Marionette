#!/usr/bin/env python3
"""Quick test: can a text-conditioned diffusion model generate the motion latents z of motion_ae.py from a prompt?

  python3 scripts/motion/diffusion_test.py --train [--steps 8000]    # -> data/embeddings/z_diffusion.pt
  python3 scripts/motion/diffusion_test.py --eval                    # held-out prompts: prompt match, retrieval, realism, diversity
  python3 scripts/motion/diffusion_test.py --samples "a person waves" "walks forward" ...   # generate, decode, draw (docs/figures/diffusion_samples.png)

Data: the saved pairs in data/embeddings/motion.npz: the latent z (256) of each clip and the CLIP embedding of its captions (centred, 512).
Only the 5,500 training clips are used for training; the 555 held-out clips (rigs and clip families never seen) are for evaluation.
Model: a small residual MLP denoiser (4 blocks, width 1024, text and time enter through adaptive scale/shift), DDPM with a cosine schedule
(1000 steps, noise prediction), text dropped 10% of the time so classifier-free guidance works; sampling is DDIM with 50 steps.
This is a quick feasibility test (a few minutes of training), not a tuned generator.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
CKPT = EMB / "z_diffusion.pt"
T_STEPS = 1000


def load_pairs(para=False):
    """para: also attach, per clip, the paraphrase embeddings of its 'caption' (data/embeddings/text_para.npz, split 'train', made by scripts/text/paraphrase.py): P["para"][i] is an (n, 512) array"""
    z = np.load(EMB / "motion.npz")
    keys = [str(k) for k in z["key"]]
    temb = np.load(EMB / "text.npy")
    mu = np.load(EMB / "text_mean.npy")
    temb = temb - mu
    temb = (temb / np.maximum(np.linalg.norm(temb, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    rows = {}
    with open(EMB / "text_index.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["dataset"], r["clip"]), []).append((r["style"], int(r["row"])))
    caps = [rows[tuple(k.split("|", 1))] for k in keys]
    out = dict(z=z["z"].astype(np.float32), sem=z["sem"], split=z["split"], cls=z["cls"], keys=keys, caps=caps, temb=temb,
               text_clip=z["text_clip"], mu=mu, dur=z["duration_s"])
    if para:
        import json
        D = np.load(EMB / "text_para.npz")
        m = D["split"] == "train"
        by = {}
        for e, c, k in zip(D["emb"][m], D["caption"][m], D["cls"][m]):
            by.setdefault((str(c), str(k)), []).append(e)
        by = {k: np.stack(v) for k, v in by.items()}
        text = [json.loads(c).get("caption", "").strip() for c in z["captions"]]
        out["para"] = [by.get((text[i], str(z["cls"][i]))) for i in range(len(keys))]
    return out


def build():
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.ln = nn.LayerNorm(w, elementwise_affine=False)
            self.ada = nn.Linear(w, 2 * w)
            self.ff = nn.Sequential(nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))

        def forward(self, h, cond):
            sc, sh = self.ada(cond).chunk(2, -1)
            return h + self.ff(self.ln(h) * (1 + sc) + sh)

    class Denoiser(nn.Module):
        def __init__(self, zd=256, td=512, w=1024, nb=4):
            super().__init__()
            self.inp = nn.Linear(zd, w)
            self.t = nn.Sequential(nn.Linear(128, w), nn.SiLU(), nn.Linear(w, w))
            self.c = nn.Sequential(nn.Linear(td, w), nn.SiLU(), nn.Linear(w, w))
            self.null = nn.Parameter(torch.zeros(td))
            self.blocks = nn.ModuleList([Block(w) for _ in range(nb)])
            self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, zd))

        def forward(self, x, t, text, drop=None):
            half = 64
            fr = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
            a = t.float()[:, None] * fr[None]
            te = self.t(torch.cat([a.sin(), a.cos()], -1))
            if drop is not None:
                text = torch.where(drop[:, None], self.null[None].expand_as(text), text)
            cond = te + self.c(text)
            h = self.inp(x)
            for b in self.blocks:
                h = b(h, cond)
            return self.out(h)

    return Denoiser


def schedule(torch, dev):
    s = 0.008
    t = torch.arange(T_STEPS + 1, dtype=torch.float64) / T_STEPS
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1)[1:]
    return ab.float().to(dev)


def cmd_train(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    global CKPT
    if args.ckpt:
        CKPT = EMB / args.ckpt
    P = load_pairs(para=args.para)
    tr = np.where(P["split"] == "train")[0]
    if args.para:
        print(f"paraphrases for {sum(P['para'][i] is not None for i in tr)} of {len(tr)} training clips, {sum(len(P['para'][i]) for i in tr if P['para'][i] is not None)} in all; "
              f"text: {args.para_frac:.0%} paraphrase when the clip has one, else an original caption", flush=True)
    zm, zs = P["z"][tr].mean(0), P["z"][tr].std(0) + 1e-6
    Zn = torch.from_numpy((P["z"] - zm) / zs).to(dev)
    model = build()().to(dev)
    ema = build()().to(dev)
    ema.load_state_dict(model.state_dict())
    ab = schedule(torch, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"{len(tr)} training clips, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, device {dev}")
    import time
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.choice(tr, args.batch)
        x0 = Zn[torch.from_numpy(idx).to(dev)]
        def pick(i):
            if args.para and P["para"][i] is not None and rng.random() < args.para_frac:
                return P["para"][i][rng.integers(len(P["para"][i]))]
            return P["temb"][P["caps"][i][rng.integers(len(P["caps"][i]))][1]]
        text = torch.from_numpy(np.stack([pick(i) for i in idx])).to(dev)
        t = torch.randint(0, T_STEPS, (args.batch,), device=dev)
        eps = torch.randn_like(x0)
        a = ab[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        drop = torch.rand(args.batch, device=dev) < 0.1
        loss = torch.nn.functional.mse_loss(model(xt, t, text, drop), eps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step:5d} loss {float(loss):.4f} [{time.time() - t0:.0f}s]", flush=True)
    torch.save(dict(model=ema.state_dict(), zm=torch.from_numpy(zm), zs=torch.from_numpy(zs), args={k: v for k, v in vars(args).items()}), CKPT)
    print("saved", CKPT)


def load_model(dev):
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = build()().to(dev)
    m.load_state_dict(ck["model"])
    return m.eval(), ck["zm"].numpy(), ck["zs"].numpy(), torch


def sample(model, torch, dev, text, guidance=2.0, steps=50, seed=0, rescale=0.0, x0_fn=None):
    """DDIM (eta 0) with classifier-free guidance. text: (n, 512) float32 numpy. Returns standardised latents (n, 256).
    rescale (0-1): rescaled guidance (Lin et al. 2023): the guided noise estimate is scaled back to the spread of the conditional one and
    mixed in with this weight (it did not help here, see docs/EMBED.md).
    x0_fn(x0, i, n_steps) -> x0: optional constraint applied to the clean-latent estimate at each step (scripts/motion/speed_sampler.py uses it to keep a walk's
    speed in the plausible range for its prompt); the noise estimate is recomputed from the corrected x0 so the DDIM update stays consistent."""
    ab = schedule(torch, dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(text)
    x = torch.randn(n, 256, generator=g).to(dev)
    text = torch.from_numpy(np.asarray(text, np.float32)).to(dev)
    ts = torch.linspace(T_STEPS - 1, 0, steps).round().long().to(dev)
    allnull = torch.ones(n, dtype=torch.bool, device=dev)
    nodrop = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for i, t in enumerate(ts):
            tt = t.expand(n)
            e_c = model(x, tt, text, nodrop)
            if guidance == 1.0:
                e = e_c
            else:
                e_u = model(x, tt, text, allnull)
                e = e_u + guidance * (e_c - e_u)
                if rescale > 0:
                    e_r = e * (e_c.std(dim=-1, keepdim=True) / e.std(dim=-1, keepdim=True).clamp_min(1e-6))
                    e = rescale * e_r + (1 - rescale) * e
            a = ab[t]
            a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-6, 6)
            if x0_fn is not None:
                x0_new = x0_fn(x0, i, len(ts))
                e = e - (a.sqrt() / (1 - a).sqrt().clamp_min(1e-6)) * (x0_new - x0)       # x = sqrt(a) x0 + sqrt(1-a) e: move e by what the correction moved x0
                x0 = x0_new                                                               # (no correction -> exactly the plain sampler)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
    return x.cpu().numpy()


def frechet(a, b):
    from scipy.linalg import sqrtm
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a.T), np.cov(b.T)
    s = sqrtm(ca @ cb)
    s = s.real if np.iscomplexobj(s) else s
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * s))


def ranks(sim):
    return (sim > np.diag(sim)[:, None]).sum(1)


def recall(r):
    return f"R@1 {100 * (r < 1).mean():4.1f}%  R@5 {100 * (r < 5).mean():4.1f}%  R@10 {100 * (r < 10).mean():4.1f}%  median rank {np.median(r):4.0f}"


def cmd_eval(args):
    import torch
    import motion_ae as MA
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    P = load_pairs()
    model, zm, zs, torch = load_model(dev)
    ae, mean, std, _ = MA.load_model(dev)
    va = np.where(P["split"] == "val")[0]
    tr = np.where(P["split"] == "train")[0]
    T = P["text_clip"][va]                                              # the 'caption' prompt of each held-out clip
    sem_real = P["sem"][va]
    n = len(va)

    def to_sem(zn):
        z = zn * zs + zm
        with torch.no_grad():
            s = torch.nn.functional.normalize(ae.sem(torch.from_numpy(z.astype(np.float32)).to(dev)), dim=-1).cpu().numpy()
        return z, s

    print(f"{n} held-out prompts (the caption of each held-out clip); real latents of these clips are never seen by the diffusion model\n")
    chance = (T @ np.roll(sem_real, 1, 0).T).diagonal().mean()
    print(f"reference: prompt vs the REAL motion of the same clip: cosine {np.mean((T * sem_real).sum(1)):.3f}; vs a different clip's motion {chance:.3f}")
    print(f"  direct retrieval with the semantic head (text -> real held-out motions): {recall(ranks(T @ sem_real.T))}\n")
    res = {}
    for g in args.guidance:
        zn = sample(model, torch, dev, T, guidance=g)
        z, sem = to_sem(zn)
        cos = (T * sem).sum(1).mean()
        r = ranks(sem @ sem_real.T)                                      # generated motion -> nearest REAL held-out motion
        fd = frechet(zn, (P["z"][va] - zm) / zs)
        res[g] = z
        print(f"guidance {g}: prompt match cosine {cos:.3f} | generated -> real held-out clip of that prompt: {recall(r)} | Frechet distance to real held-out latents {fd:.1f}")
    zn0 = sample(model, torch, dev, np.zeros_like(T), guidance=1.0)      # no text at all (null condition would be better, but zeros is also "no prompt")
    print(f"  no prompt (zeros): prompt match cosine {(T * to_sem(zn0)[1]).sum(1).mean():.3f}, Frechet distance {frechet(zn0, (P['z'][va] - zm) / zs):.1f}")
    fd_floor = frechet((P["z"][tr][:555] - zm) / zs, (P["z"][va] - zm) / zs)
    print(f"  noise floor: Frechet distance between 555 real TRAIN latents and the 555 held-out latents: {fd_floor:.1f}")
    # diversity and novelty, per guidance: 8 samples of each of 40 prompts. The diversity baseline is real TRAIN clips with near-identical
    # captions (not random clips, which are unrelated to each other); novelty is the distance to the nearest training latent, against
    # the same for real held-out latents (a copy of a training clip would be near 0)
    rng = np.random.default_rng(0)
    pick = rng.choice(n, 40, replace=False)
    Ztr = P["z"][tr]
    ref = np.mean([np.min(np.linalg.norm(Ztr - P["z"][va[i]], axis=1)) for i in pick])
    pw = lambda s: (1 - s @ s.T)[np.triu_indices(len(s), 1)].mean()
    for g in args.guidance:
        dg, dr, nn_d = [], [], []
        for i in pick:
            zn = sample(model, torch, dev, np.repeat(T[i:i + 1], 8, 0), guidance=g, seed=int(i))
            zz, s = to_sem(zn)
            dg.append(pw(s))
            near = tr[np.argsort(-(P["text_clip"][tr] @ T[i]))[:8]]
            dr.append(pw(P["sem"][near]))
            nn_d.append(np.mean([np.min(np.linalg.norm(Ztr - q, axis=1)) for q in zz]))
        print(f"guidance {g}: same-prompt diversity (pairwise cosine distance of sem) generated {np.mean(dg):.3f} vs real clips with near-identical captions {np.mean(dr):.3f}"
              f" | distance to the nearest training latent: generated {np.mean(nn_d):.2f} vs real held-out {ref:.2f}")
    # decoded motion: how much do generated clips move compared with real ones
    zg = res[args.guidance[0]]
    with torch.no_grad():
        Xg = ae.decode(torch.from_numpy(zg.astype(np.float32)).to(dev)).cpu().numpy() * std + mean
    D = np.load(EMB / "motion_feats.npz")["feats"]
    import csv as _csv
    idx = {(r["dataset"], r["clip"]): i for i, r in enumerate(_csv.DictReader(open(EMB / "motion_feats_index.csv", newline="")))}
    real = np.stack([D[idx[tuple(P["keys"][i].split("|", 1))]] for i in va])
    spread = lambda X: X[:, :, 13:37].std(1).mean(1)
    sr, sg = spread(real), spread(Xg)
    print(f"decoded limb movement (spread over time): generated median {np.median(sg):.3f} vs real held-out {np.median(sr):.3f}; generated clips below 40% of the real median: {100 * (sg < 0.4 * np.median(sr)).mean():.0f}% (real: {100 * (sr < 0.4 * np.median(sr)).mean():.0f}%)")


def cmd_samples(args):
    import torch
    import motion_ae as MA
    import decode_latents as DL
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    from embed import encode, load_text_encoder
    model, zm, zs, torch = load_model(dev)
    ae, mean, std, _ = MA.load_model(dev)
    enc = load_text_encoder()
    mu = np.load(EMB / "text_mean.npy")
    prompts = args.samples
    E = encode(prompts, enc)[0] - mu
    E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    rows, out = [], {}
    sm = None
    if args.speed_aware or args.dur_band is not None:                                           # keep a clip's travel speed in the range that is plausible for its prompt (scripts/motion/speed_sampler.py)
        import walk_check as WC
        from speed_sampler import SpeedModel
        sm = SpeedModel(WC.prepare("v2"))
    for k, p in enumerate(prompts):
        txt = np.repeat(E[k:k + 1], args.n, 0)
        if sm is None:
            zn = sample(model, torch, dev, txt, guidance=args.g, seed=k)
        elif args.dur_band is not None:                            # duration band, with the speed cap only if --cap-q is given (the mean + k sd cap is unreliable for runs)
            zn = sm.sample(txt, guidance=args.g, k=99.0 if args.cap_q is None else 0.5, seed=k, exact=True, keep_gait=True, dur_k=args.dur_band, cap_q=args.cap_q)
        else:
            zn = sm.sample(txt, guidance=args.g, k=0.5, seed=k, exact=True, keep_gait=True)
        z = (zn * zs + zm).astype(np.float32)
        with torch.no_grad():
            zt = torch.from_numpy(z).to(dev)
            X = ae.decode(zt).cpu().numpy() * std + mean
            aux = ae.aux(zt).cpu().numpy()
            sem = torch.nn.functional.normalize(ae.sem(zt), dim=-1).cpu().numpy()
        X[..., 37:41] = (X[..., 37:41] > 0.5).astype(np.float32)
        for j in range(args.n):
            nv, dur = int(np.clip(round(aux[j, 1] * 128), 8, 128)), float(np.exp(aux[j, 0]))
            cos = float(sem[j] @ E[k])
            rows.append(DL.strip(X[j], nv, dur, f'"{p}" #{j + 1}', f"prompt match {cos:.2f}\n{dur:.1f} s"))
            out[f"{k}_{j}"] = (X[j], nv, dur)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height + 4 for r in rows)), DL.RM.BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    path = Path(args.out) if args.out else ROOT / "docs" / "figures" / "diffusion_samples.png"
    sheet.save(path)
    print("saved", path, sheet.size)
    if args.characters:
        import decoded_to_character as DC
        ref = DC.ref_skeleton()
        for key, (x, nv, dur) in out.items():
            k, j = map(int, key.split("_"))
            if j != 0:
                continue
            label = "gen_" + "_".join(prompts[k].lower().split())[:24]
            px = DC.proxy_from_features(x, nv, dur, ref)
            path = DC.OUT_DIR / f"{label}.glb"
            DC.write_proxy(path, px, ref, label)
            for tk in ("mixamo_Aciana", "objaverse_a734df8176ca47c68b0a90d1a17b5132"):
                info = DC.to_character(path, tk, label, f"generated from the prompt: {prompts[k]}", ref, extra=dict(kind="diffusion sample", prompt=prompts[k]))
                print(f"  {label:30s} {tk[:14]:14s} skate {info['skate_mean']:.3f} fails {info['reach_fail_frames']} pen {info['penetration_max']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--samples", nargs="+")
    ap.add_argument("--speed-aware", action="store_true", help="with --samples: use the speed-aware sampler (scripts/motion/speed_sampler.py)")
    ap.add_argument("--dur-band", type=float, default=None, help="with --samples --speed-aware: use only the duration band (prompt's prior +- this many sd) and no speed cap")
    ap.add_argument("--cap-q", type=float, default=None, help="with --dur-band: also cap the speed at this quantile of the nearest captions' real speeds")
    ap.add_argument("--out", help="with --samples: where to save the figure")
    ap.add_argument("--characters", action="store_true", help="with --samples: also put the first sample of each prompt on characters")
    ap.add_argument("--para", action="store_true", help="with --train: also train on the paraphrases of the captions (scripts/text/paraphrase.py)")
    ap.add_argument("--para-frac", type=float, default=0.5, help="with --para: share of the training texts that are paraphrases (for clips that have some)")
    ap.add_argument("--ckpt", help="checkpoint file name in data/embeddings (default z_diffusion.pt); for --train, --eval and --samples")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--guidance", type=float, nargs="+", default=[2.0, 1.0, 4.0])
    ap.add_argument("--g", type=float, default=2.0)
    ap.add_argument("-n", type=int, default=2)
    a = ap.parse_args()
    global CKPT
    if a.ckpt:
        CKPT = EMB / a.ckpt
    if a.train:
        cmd_train(a)
    elif a.eval:
        cmd_eval(a)
    elif a.samples:
        cmd_samples(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
