#!/usr/bin/env python3
"""Speed-aware sampling for the latent diffusion model: keep a generated clip's travel speed in the range that is plausible for its prompt.

  python3 scripts/motion/speed_sampler.py --walks        # bad-walk rate (scripts/motion/walk_check.py rule) with and without the constraint
  python3 scripts/motion/speed_sampler.py --check        # side effects: running and standing prompts, held-out prompt adherence and diversity

Why: classifier-free guidance makes generated walks travel 35-58% farther than real ones (a "too fast" failure in up to 17% of walks; docs/EMBED.md).
How (no retraining):
  1. speed probe: a ridge regression from a latent z to the log of the travel speed of the clip the decoder makes from it (hips heights per second,
     log(speed + 0.05)); R^2 0.85 on held-out clips (only 0.13 inside walks: it reads gross speed, not fine speed);
  2. speed prior: for a prompt, the real speeds (log) of the clips of the 40 nearest training captions (CLIP cosine): mean mu and spread sd
     (the real clip of a walking prompt lies within 1 sd 82% of the time, within 2 sd 100%);
  3. at every DDIM step the clean-latent estimate is read with the probe; if its speed lies above mu + k*sd (and, with --two-sided, below mu - k*sd) it is
     moved along the probe direction, by the smallest step that puts it on the edge of the band; the noise estimate is recomputed from the corrected estimate.
The prompt decides the band, so "runs forward" keeps its higher speeds and "walks in place" its low ones.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]


class SpeedModel:
    def __init__(self, c, k_nn=40, floor_sd=0.25):
        P = c.P
        self.c = c
        self.k_nn, self.floor_sd = k_nn, floor_sd
        Z = P["z"]
        self.tr = np.where(P["split"] == "train")[0]
        X, nv, du = c.decode(Z)
        path = np.array([np.abs(X[k, :nv[k], 0]).sum() for k in range(len(Z))])
        self.logspeed_dec = np.log(path / du + 0.05)                     # what the walk rule measures, from the decoded clip
        self.real_speed = np.array([np.abs(c.F[i, :int(c.nvs[i]), 0]).sum() / max(float(P["dur"][i]), 1e-3) for i in range(len(Z))])
        A = np.c_[Z[self.tr], np.ones(len(self.tr))]
        w = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ self.logspeed_dec[self.tr])
        self.w, self.b = w[:-1].astype(np.float32), float(w[-1])
        self.T = P["text_clip"]
        self.real_dur = np.asarray(P["dur"], np.float64)

    def dur_prior(self, text, floor_sd=0.15):
        """(n, 512) prompt embeddings -> (mu, sd) of the log duration (seconds) of the real clips with the nearest training captions."""
        mu, sd = [], []
        for q in np.atleast_2d(text):
            nn = self.tr[np.argsort(-(self.T[self.tr] @ q))[:self.k_nn]]
            y = np.log(np.maximum(self.real_dur[nn], 1e-2))
            mu.append(y.mean())
            sd.append(max(y.std(), floor_sd))
        return np.array(mu, np.float32), np.array(sd, np.float32)

    def prior(self, text, with_samples=False):
        """(n, 512) prompt embeddings -> (mu, sd) of the log speed of the real clips with the nearest training captions
        (and, with_samples, the (n, k) array of those log speeds)."""
        mu, sd, ys = [], [], []
        for q in np.atleast_2d(text):
            nn = self.tr[np.argsort(-(self.T[self.tr] @ q))[:self.k_nn]]
            y = np.log(self.real_speed[nn] + 0.05)
            mu.append(y.mean())
            sd.append(max(y.std(), self.floor_sd))
            ys.append(y)
        out = (np.array(mu, np.float32), np.array(sd, np.float32))
        return out + (np.stack(ys).astype(np.float32),) if with_samples else out

    def constraint(self, text, k=1.0, two_sided=False, start=0.0):
        torch, dev, c = self.c.torch, self.c.dev, self.c
        mu, sd = self.prior(text)
        mu, sd = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
        w = torch.from_numpy(self.w).to(dev)
        zm, zs = torch.from_numpy(c.zm.astype(np.float32)).to(dev), torch.from_numpy(c.zs.astype(np.float32)).to(dev)
        ww = (w * w).sum()

        def fn(x0, i, n):
            if i / n < start:
                return x0
            z0 = x0 * zs + zm
            s = z0 @ w + self.b
            excess = torch.relu(s - (mu + k * sd))
            if two_sided:
                excess = excess - torch.relu((mu - k * sd) - s)
            return (z0 - (excess / ww)[:, None] * w[None]) / zs - zm / zs
        return fn

    def log_speed(self, z):
        """Differentiable log(speed + 0.05) of the clip the decoder makes from latents z (n, 256), in hips heights per second:
        path length over the frames the model says are real (a soft mask on the predicted length) divided by the predicted duration."""
        c = self.c
        torch = c.torch
        X = c.ae.decode(z)
        aux = c.ae.aux(z)
        f0 = X[..., 0] * float(c.std[0]) + float(c.mean[0])
        nvp = (aux[:, 1] * 128).clamp(8, 128)
        t = torch.arange(128, device=z.device, dtype=z.dtype)[None]
        mask = torch.sigmoid((nvp[:, None] - t - 0.5) / 1.0)
        path = (mask * f0.abs()).sum(1)
        return torch.log(path / aux[:, 0].exp() + 0.05)

    def stride(self, z):
        """Differentiable stride amplitude of the decoded clip: the standard deviation over its real frames of the two feet's forward position
        (features 18 and 24), averaged: the 'swing' of scripts/motion/gait_check.py."""
        c = self.c
        torch = c.torch
        X = c.ae.decode(z)
        aux = c.ae.aux(z)
        nvp = (aux[:, 1] * 128).clamp(8, 128)
        t = torch.arange(128, device=z.device, dtype=z.dtype)[None]
        w = torch.sigmoid((nvp[:, None] - t - 0.5) / 1.0)
        w = w / w.sum(1, keepdim=True)
        out = 0.0
        for j in (18, 24):
            f = X[..., j] * float(c.std[j]) + float(c.mean[j])
            m = (w * f).sum(1, keepdim=True)
            out = out + ((w * (f - m) ** 2).sum(1) + 1e-8).sqrt()
        return 0.5 * out

    def constraint_exact(self, text, k=1.0, start=0.0, iters=1, target="cap", tol=0.15, seed=0, keep_gait=False, dur_k=None, max_step=None, cap_q=None):
        """Like constraint(), but reads the speed from the real decoder (with gradients) instead of the linear probe.
        target "cap": speeds above mu + k sd are pulled down to it. target "sample": every sample gets its own target speed, drawn from the real
        clips of the nearest captions, and is steered to within +-tol (log units) of it, so the speeds keep the real spread. target "sample-cap":
        the same per-sample draw, used only as an upper limit (target + tol): nothing is sped up or slowed below its natural speed."""
        torch, dev, c = self.c.torch, self.c.dev, self.c
        mu, sd, ys = self.prior(text, with_samples=True)
        if dur_k is not None:
            dmu, dsd = self.dur_prior(text)
            dlo, dhi = torch.from_numpy(dmu - dur_k * dsd).to(dev), torch.from_numpy(dmu + dur_k * dsd).to(dev)
        rng = np.random.default_rng(seed)
        tgt = torch.from_numpy(ys[np.arange(len(ys)), rng.integers(ys.shape[1], size=len(ys))]).to(dev) if target in ("sample", "sample-cap") else None
        # cap: the prior mean + k sd, or (cap_q) that quantile of the nearest captions' real log speeds: a mixed neighbourhood ("runs and then falls")
        # drags a mean down, a high quantile still allows the fast clips among the neighbours
        cap = torch.from_numpy((np.quantile(ys, cap_q, axis=1) if cap_q is not None else mu + k * sd).astype(np.float32)).to(dev)
        zm, zs = torch.from_numpy(c.zm.astype(np.float32)).to(dev), torch.from_numpy(c.zs.astype(np.float32)).to(dev)

        def fn(x0, i, n):
            if i / n < start:
                return x0
            z0 = (x0 * zs + zm).detach()
            for _ in range(iters):
                with torch.enable_grad():
                    z = z0.clone().requires_grad_(True)
                    sp = self.log_speed(z)
                    g = torch.autograd.grad(sp.sum(), z)[0]
                    if dur_k is not None:
                        ld = c.ae.aux(z)[:, 0]                                # the decoder's log duration (a linear head on z)
                        gd = torch.autograd.grad(ld.sum(), z)[0]
                    if keep_gait:                                             # steer without changing the stride: drop the stride direction from the step
                        h = torch.autograd.grad(self.stride(z).sum(), z)[0]
                        proj = lambda v: v - ((v * h).sum(1, keepdim=True) / (h * h).sum(1, keepdim=True).clamp_min(1e-8)) * h
                        g_step = proj(g)
                        if dur_k is not None:
                            gd = proj(gd)
                    else:
                        g_step = g
                    gg = (g * g_step).sum(1).clamp_min(1e-6)
                if tgt is None:
                    excess = torch.relu(sp.detach() - cap)
                elif target == "sample-cap":                                  # each sample's own limit, only ever pulled down
                    excess = torch.relu(sp.detach() - (tgt + tol))
                else:
                    d = sp.detach() - tgt
                    excess = torch.sign(d) * torch.relu(d.abs() - tol)
                lim = (lambda dz: dz if max_step is None else dz * (max_step / dz.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0))   # trust region
                if dur_k is None:
                    z0 = z0 + lim(-(excess / gg)[:, None] * g_step)
                else:
                    # speed cap and duration band together: the smallest step that satisfies the violated ones (linearised), a 2x2 solve per sample
                    ld = ld.detach()
                    e_d = torch.relu(dlo - ld) - torch.relu(ld - dhi)          # >0: duration must grow, <0: shrink
                    G = torch.stack([g_step, gd], 1)                           # (n, 2, 256)
                    act = torch.stack([(excess > 0).float(), (e_d != 0).float()], 1)
                    G = G * act[:, :, None]
                    tgt2 = torch.stack([-excess, e_d], 1) * act                # g_s . delta = -excess, g_d . delta = e_d
                    A = G @ G.transpose(1, 2)                                  # (n, 2, 2); solved in closed form (batched solve is missing on MPS)
                    a, b, d = A[:, 0, 0] + 1e-4, A[:, 0, 1], A[:, 1, 1] + 1e-4
                    det = (a * d - b * b).clamp_min(1e-8)
                    t1, t2 = tgt2[:, 0], tgt2[:, 1]
                    lam = torch.stack([(d * t1 - b * t2) / det, (a * t2 - b * t1) / det], 1)
                    z0 = z0 + lim((G.transpose(1, 2) @ lam[:, :, None])[:, :, 0])
            return (z0 - zm) / zs
        return fn

    def sample(self, text, guidance=2.0, k=1.0, two_sided=False, start=0.0, seed=5, steps=50, exact=False, target="cap", tol=0.15, keep_gait=False, dur_k=None, max_step=None, cap_q=None):
        c = self.c
        if k is None:
            fn = None
        elif exact:
            fn = self.constraint_exact(text, k, start, target=target, tol=tol, seed=seed, keep_gait=keep_gait, dur_k=dur_k, max_step=max_step, cap_q=cap_q)
        else:
            fn = self.constraint(text, k, two_sided, start)
        return c.DT.sample(c.model, c.torch, c.dev, text, guidance=guidance, seed=seed, steps=steps, x0_fn=fn)


def cmd_walks(args):
    import walk_check as WC
    c = WC.prepare("v2")
    sm = SpeedModel(c)
    T = c.P["text_clip"][c.sel]
    print(f"real forward-walking clips: {len(c.sel)}; walks are bad if the hips float, travel is above 2.5 hips heights/s, or the legs do not step\n")
    fl, fa, st = c.judge(c.real)
    print(f"  REAL clips                                       n={len(c.sel)} bad {100 * (fl | fa | st).mean():4.1f}%  (floating hips {100 * fl.mean():4.1f}%, too fast {100 * fa.mean():4.1f}%, not stepping {100 * st.mean():4.1f}%)  median speed {np.median([r['speed'] for r in c.real]):.2f} H/s")
    for g in args.guidance:
        zn = sm.sample(T, guidance=g, k=None)
        c.report(f"plain sampler, g={g}", zn * c.zs + c.zm)
        for k in args.k:
            for st in args.start:
                zn = sm.sample(T, guidance=g, k=k, two_sided=args.two_sided, start=st, exact=args.exact, target=args.target, tol=args.tol, keep_gait=args.keep_gait, max_step=args.max_step, dur_k=args.dur_k, cap_q=args.cap_q)
                rows = c.report(f"speed-aware{' (decoder' + (', keep gait)' if args.keep_gait else ')') if args.exact else ' (probe)'}, g={g}{', dur band ' + str(args.dur_k) if args.dur_k is not None else ''}, {args.target}{f' mu+{k}sd' if args.target == 'cap' else f' tol {args.tol}'}", zn * c.zs + c.zm,
                                extra=lambda r: f"  speed 10/50/90th pct {np.percentile([x['speed'] for x in r], 10):.2f}/{np.percentile([x['speed'] for x in r], 50):.2f}/{np.percentile([x['speed'] for x in r], 90):.2f}  duration {np.percentile([x['dur'] for x in r], 10):.2f}/{np.percentile([x['dur'] for x in r], 50):.2f}/{np.percentile([x['dur'] for x in r], 90):.2f}")
        print()


def cmd_check(args):
    import walk_check as WC
    from embed import encode, load_text_encoder
    c = WC.prepare("v2")
    sm = SpeedModel(c)
    enc = load_text_encoder()
    mu0 = np.load(c.MA.EMB / "text_mean.npy")
    emb = lambda ts: (lambda E: (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32))(encode(ts, enc)[0] - mu0)
    g, k = args.guidance[0], args.k[0]
    print(f"side effects with guidance {g}, band +-{k} sd (cap only): speed of generated clips (hips heights per second, median of 100 samples), plain vs speed-aware\n")
    for p in ("An object walks forward.", "An object runs forward.", "An object jogs forward.", "An object walks in place.", "An object stands still.", "An object walks slowly.", "An object sprints."):
        T = np.repeat(emb([p]), 100, 0)
        out = []
        for kk in (None, k):
            zn = sm.sample(T, guidance=g, k=kk, seed=3, exact=args.exact, target=args.target, tol=args.tol, keep_gait=args.keep_gait, dur_k=args.dur_k, cap_q=args.cap_q, max_step=args.max_step)
            X, nv, du = c.decode(zn * c.zs + c.zm)
            sp = np.array([np.abs(X[j, :nv[j], 0]).sum() / du[j] for j in range(100)])
            st = np.array([c.GC.gait(X[j], nv[j])["step"] for j in range(100)])
            out.append((np.median(sp), np.percentile(sp, 90), np.median(st)))
        mu, sd = sm.prior(emb([p]))
        print(f"  {p:28s} prompt prior {np.exp(mu[0]) - 0.05:5.2f} H/s | plain: median {out[0][0]:.2f}, 90th pct {out[0][1]:.2f}, step {out[0][2]:.3f} | speed-aware: median {out[1][0]:.2f}, 90th pct {out[1][1]:.2f}, step {out[1][2]:.3f}")
    # held-out prompt adherence and diversity (the evaluation of diffusion_test.py --eval, for this sampler)
    P = c.P
    va = np.where(P["split"] == "val")[0]
    T = P["text_clip"][va]
    sem_real = P["sem"][va]
    ranks = lambda sim: (sim > np.diag(sim)[:, None]).sum(1)
    print(f"\nheld-out prompts ({len(va)}): generated latent -> its real clip (R@1 / R@5 / R@10, median rank), prompt-match cosine")
    for name, kk in (("plain", None), (f"speed-aware band +-{k}", k)):
        zn = sm.sample(T, guidance=g, k=kk, seed=0, exact=args.exact, target=args.target, tol=args.tol, keep_gait=args.keep_gait, dur_k=args.dur_k, cap_q=args.cap_q, max_step=args.max_step)
        z = zn * c.zs + c.zm
        with c.torch.no_grad():
            s = c.torch.nn.functional.normalize(c.ae.sem(c.torch.from_numpy(z.astype(np.float32)).to(c.dev)), dim=-1).cpu().numpy()
        r = ranks(s @ sem_real.T)
        print(f"  {name:28s} R@1 {100 * (r < 1).mean():4.1f}%  R@5 {100 * (r < 5).mean():4.1f}%  R@10 {100 * (r < 10).mean():4.1f}%  median {np.median(r):3.0f}  cosine {(T * s).sum(1).mean():.3f}")
    cmd_diversity(args, c, sm, g, k)


def run_set(c, sm):
    """real travelling running clips (biped, caption says runs, no in-place/turn/other action words, travelling): indices into the pairs"""
    import re
    ex = "in place|backward|turn|jump|stop|sit|kneel|pick|carry|crouch|circle|walk"
    return [i for i, t in enumerate(c.caps) if c.P["cls"][i] == "biped" and re.search(r"\bruns?\b", t) and not re.search(ex, t) and sm.real_speed[i] > 0.5]


def cmd_runs(args):
    """Duration and speed of generated runs against real running clips, plain vs the constraints, by guidance."""
    import walk_check as WC
    c = WC.prepare("v2")
    sm = SpeedModel(c)
    run = run_set(c, sm)
    P = c.P
    d_real = np.array([float(P["dur"][i]) for i in run])
    sp_real = sm.real_speed[run]
    st_real = np.array([c.GC.gait(c.F[i], int(c.nvs[i]))["step"] for i in run])
    lo, hi = np.percentile(d_real, [10, 90])
    thr = 0.4 * np.median(st_real)
    pct = lambda a: "/".join(f"{x:.2f}" for x in np.percentile(a, [10, 50, 90]))
    print(f"{len(run)} real travelling running clips: duration 10/50/90th pct {pct(d_real)} s, speed {pct(sp_real)} H/s, not stepping (step < {thr:.3f}) {100 * np.mean(st_real < thr):.1f}%\n"
          f"(share of generated clips shorter than the real 10th percentile {lo:.2f} s should be about 10%; longer than the real 90th percentile {hi:.2f} s about 10%)")
    T = P["text_clip"][run]
    st0 = args.start[0]
    dk = args.dur_k if args.dur_k is not None else 1.0
    capname = f"quantile {args.cap_q}" if args.cap_q is not None else "+0.5 sd"
    configs = [("plain", dict(k=None)),
               (f"duration band +-{dk} sd", dict(k=99.0, exact=True, keep_gait=True, dur_k=dk, start=st0, max_step=args.max_step)),
               (f"speed cap {capname}", dict(k=0.5, exact=True, keep_gait=True, start=st0, max_step=args.max_step, cap_q=args.cap_q)),
               (f"speed cap {capname} and duration band +-{dk} sd", dict(k=0.5, exact=True, keep_gait=True, dur_k=dk, start=st0, max_step=args.max_step, cap_q=args.cap_q))]
    for g in args.guidance:
        print(f"\nguidance {g}")
        for name, kw in configs:
            zn = sm.sample(T, guidance=g, seed=5, **kw)
            X, nv, du = c.decode(zn * c.zs + c.zm)
            sp = np.array([np.abs(X[j, :nv[j], 0]).sum() / du[j] for j in range(len(T))])
            st = np.array([c.GC.gait(X[j], int(nv[j]))["step"] for j in range(len(T))])
            print(f"  {name:48s} duration {pct(du):15s} s  short {100 * np.mean(du < 0.98 * lo):4.1f}% (strict {100 * np.mean(du < lo):4.1f}%)  long {100 * np.mean(du > hi):4.1f}% | speed {pct(sp):16s} | not stepping {100 * np.mean(st < thr):4.1f}%")


def cmd_figure(args):
    """docs/figures/diffusion_speed_aware.png: the same prompts and seeds, plain sampler above, speed-aware below (stick figures of the decoded clips)."""
    import walk_check as WC
    import decode_latents as DL
    from PIL import Image
    from embed import encode, load_text_encoder
    c = WC.prepare("v2")
    sm = SpeedModel(c)
    enc = load_text_encoder()
    mu0 = np.load(c.MA.EMB / "text_mean.npy")
    g, k = args.guidance[0], args.k[0]
    rows = []
    for p in ("An object walks forward.", "An object runs forward."):
        E = encode([p], enc)[0] - mu0
        E = np.repeat((E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32), 3, 0)
        for name, kk in (("plain", None), ("speed-aware", k)):
            zn = sm.sample(E, guidance=g, k=kk, seed=7, exact=args.exact, target=args.target, tol=args.tol, keep_gait=args.keep_gait, dur_k=args.dur_k, cap_q=args.cap_q, max_step=args.max_step)
            X, nv, du = c.decode(zn * c.zs + c.zm)
            for j in range(3):
                spd = np.abs(X[j, :nv[j], 0]).sum() / du[j]
                st = c.GC.gait(X[j], nv[j])["step"]
                rows.append(DL.strip(X[j], int(nv[j]), float(du[j]), f'"{p[10:-1]}" {name} #{j + 1}', f"travel {spd:.2f} hips heights/s\nstep score {st:.2f}, {du[j]:.1f} s"))
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height + 4 for r in rows)), DL.RM.BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    path = ROOT / "docs" / "figures" / "diffusion_speed_aware.png"
    sheet.save(path)
    print("saved", path, sheet.size)


def cmd_diversity(args, c, sm, g, k):
    """same-prompt diversity (8 samples of each of 40 held-out prompts, mean pairwise cosine distance of sem) of plain vs speed-aware sampling"""
    P = c.P
    va = np.where(P["split"] == "val")[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(va, 40, replace=False)
    pw = lambda s: (1 - s @ s.T)[np.triu_indices(len(s), 1)].mean()
    print("\nsame-prompt diversity (8 samples of each of 40 held-out prompts; real clips with near-identical captions: ~0.43)")
    for name, kk in (("plain", None), ("speed-aware", k)):
        d = []
        for i in pick:
            T = np.repeat(P["text_clip"][i:i + 1], 8, 0)
            zn = sm.sample(T, guidance=g, k=kk, seed=int(i), exact=args.exact, target=args.target, tol=args.tol, keep_gait=args.keep_gait, dur_k=args.dur_k, cap_q=args.cap_q, max_step=args.max_step)
            z = zn * c.zs + c.zm
            with c.torch.no_grad():
                s = c.torch.nn.functional.normalize(c.ae.sem(c.torch.from_numpy(z.astype(np.float32)).to(c.dev)), dim=-1).cpu().numpy()
            d.append(pw(s))
        print(f"  {name:14s} {np.mean(d):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walks", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-step", type=float, default=None, help="trust region: largest correction (norm in latent units; a latent has norm about 16) per sampling step")
    ap.add_argument("--runs", action="store_true", help="duration and speed of generated runs against real running clips")
    ap.add_argument("--cap-q", type=float, default=None, help="speed cap = this quantile of the nearest captions' real speeds (instead of mean + k sd)")
    ap.add_argument("--dur-k", type=float, default=None, help="duration band: the prompt's prior mean +- this many standard deviations (log seconds)")
    ap.add_argument("--figure", action="store_true", help="draw plain vs speed-aware samples for a walking and a running prompt")
    ap.add_argument("--guidance", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--k", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    ap.add_argument("--start", type=float, nargs="+", default=[0.0], help="fraction of the sampling steps after which the correction starts")
    ap.add_argument("--two-sided", action="store_true")
    ap.add_argument("--target", choices=["cap", "sample", "sample-cap"], default="cap")
    ap.add_argument("--tol", type=float, default=0.15, help="with --target sample: allowed distance (log units) from the sampled speed")
    ap.add_argument("--keep-gait", action="store_true", help="steer speed along the direction that leaves the stride amplitude unchanged")
    ap.add_argument("--exact", action="store_true", help="read the speed from the decoder with gradients instead of the linear probe")
    args = ap.parse_args()
    if args.walks:
        cmd_walks(args)
    elif args.check:
        cmd_check(args)
    elif args.figure:
        cmd_figure(args)
    elif args.runs:
        cmd_runs(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
