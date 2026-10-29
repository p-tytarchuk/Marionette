"""Side effects of capping the latent INSIDE every denoising step (generate_motion.generate(step_cap=...)), on the held-out prompts. Read-only; writes data/analysis/step_cap_check.md.

  python scripts/generate/step_cap_check.py [--factor 0.9] [--n-ik 250]

The flip sweep (flip_sweep.py) found that capping the clean-latent estimate at every step gives more completed, upright flips than rescaling the finished latent (production), but more spikes. A setting that
helps flips must not hurt anything else, so this repeats the held-out check of docs/EMBED.md ("--znorm 0.9 on the held-out set") for: no cap, the production cap after sampling, and the cap inside the sampler.
Same recipe as speed_sampler.py --check: all validation prompts as one batch, guidance 2, seed 0; adherence = the generated latent's semantic head against the real clips (R@1 / R@5 / R@10, median rank) and its
cosine with the prompt; plus, for a random subset, the IK proxy's rotation steps over 60 degrees per clip, the share of clips with any, and the median duration (real clips 2.17 s).
Needs the conda env Mario. About two minutes.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "analysis"


def run(factor, n_ik):
    import generate_motion as G
    import torch
    S = G._load()
    c, DC, ref = S["c"], S["DC"], S["ref"]
    P = c.P
    va = np.where(P["split"] == "val")[0]
    T = P["text_clip"][va].astype(np.float32)
    sem_real = P["sem"][va]
    tr = np.where(P["split"] == "train")[0]
    Ttr = P["text_clip"][tr]
    nrm_tr = np.linalg.norm((P["z"][tr] - c.zm) / c.zs, axis=1)
    nn = np.argsort(-(T @ Ttr.T), axis=1)[:, :40]                              # per prompt: its 40 nearest training captions, as cap_latent_norm does
    target = (factor * nrm_tr[nn].mean(1)).astype(np.float32)[:, None]
    tgt_t = torch.from_numpy(target).to(c.dev)

    def sample(step_cap):
        fn = (lambda x0, i, n: x0 * torch.clamp(tgt_t / x0.norm(dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)) if step_cap else None
        return c.DT.sample(c.model, c.torch, c.dev, T, guidance=2.0, seed=0, steps=50, x0_fn=fn)

    def post_cap(zn):
        return zn * np.minimum(1.0, target / np.linalg.norm(zn, axis=1, keepdims=True))

    rng = np.random.RandomState(0)
    ik_rows = rng.choice(len(va), n_ik, replace=False)
    ranks = lambda sim: (sim > np.diag(sim)[:, None]).sum(1)
    rows = []
    for name, make in (("no cap", lambda: sample(False)), (f"production: cap after sampling {factor:g}", lambda: post_cap(sample(False))), (f"cap inside every step {factor:g}", lambda: sample(True))):
        t0 = time.time()
        zn = make()
        z = zn * c.zs + c.zm
        with c.torch.no_grad():
            s = c.torch.nn.functional.normalize(c.ae.sem(c.torch.from_numpy(z.astype(np.float32)).to(c.dev)), dim=-1).cpu().numpy()
        r = ranks(s @ sem_real.T)
        X, nv, du = c.decode(z)
        spikes = []
        for j in ik_rows:
            x = X[j].copy()
            x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)
            spikes.append(G.proxy_steps(DC.proxy_from_features(x, int(nv[j]), float(du[j]), ref, body_smooth=3.0))[0])
        spikes = np.array(spikes)
        rows.append((name, 100 * (r < 1).mean(), 100 * (r < 5).mean(), 100 * (r < 10).mean(), float(np.median(r)), float((T * s).sum(1).mean()), float(spikes.mean()), 100 * float((spikes > 0).mean()),
                     float(np.median(du)), float(np.median(np.linalg.norm(zn, axis=1)))))
        print(f"{name:42s} R@5 {rows[-1][2]:.1f}  spikes {rows[-1][6]:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    lines = ["# Cap inside the sampler: side effects on the held-out prompts", "",
             f"{len(va)} validation prompts, guidance 2, seed 0, one sample each; spikes (IK proxy, rotation steps over 60 degrees) over a random {n_ik} of them. Real clips: median duration 2.17 s, median |z| 15.8.", "",
             "| setting | R@1 | R@5 | R@10 | median rank | prompt cosine | spikes per clip | clips with any | median duration (s) | median \\|z\\| |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.0f} | {r[5]:.3f} | {r[6]:.2f} | {r[7]:.0f}% | {r[8]:.2f} | {r[9]:.1f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step_cap_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factor", type=float, default=0.9)
    ap.add_argument("--n-ik", type=int, default=250)
    a = ap.parse_args()
    run(a.factor, a.n_ik)
