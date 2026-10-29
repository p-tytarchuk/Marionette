"""How often does a prompt give a good clip, for each sampler setting? Read-only on the dataset; writes data/analysis/flip_sweep.md (and .json).

  python scripts/generate/flip_sweep.py --seeds 100 [--prompts "An object does a front flip." "An object performs a front flip."]

For every prompt and setting it draws `--seeds` single samples (no gate, one candidate each) and judges each with the production quality criteria of generate_motion.py
(MAX_SPIKES, MAX_STEP, ROT_RANGE, LAND_UP): a completed turn (pitch range >= 300 degrees for a flip prompt), an upright landing, and a smooth IK proxy. "good" = all of them.
Settings: the production one (guidance 2, the latent rescaled after sampling to at most 0.9 x the nearest captions' mean norm), no cap, lower guidance, rescaled guidance (Lin et al. 2023, never
tried for flips), and a cap applied to the clean-latent estimate INSIDE every denoising step (x0_fn hook) instead of after sampling. For the production setting it also draws 8 candidates per seed and
reports the chance that the gate finds a good one within 1, 2, 4 and 8 tries (candidate k of seed s uses seed s + 7919 k, as generate() does).
Needs the conda env Mario. About a minute for 100 seeds.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "analysis"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run(prompts, n_seeds, smooth, tries_list):
    import generate_motion as G
    t0 = time.time()
    S = G._load()
    c, DC = S["c"], S["DC"]
    print(f"models loaded in {time.time() - t0:.0f} s", flush=True)
    import torch
    if "fe" not in S:
        import text_frontend as TF
        S["fe"] = TF.Frontend()

    def embed(prompt):
        E = S["encode"]([prompt], S["enc"])[0] - S["mu0"]
        return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)

    def target_norm(E, factor):
        G.cap_latent_norm(np.ones((1, 256), np.float32), E, S, 1.0)                       # fills S["z_norm"]
        T, nrm = S["z_norm"]
        return factor * float(nrm[np.argsort(-(T @ E[0]))[:40]].mean())

    def draw(E, guidance, seed, rescale=0.0, post_cap=0.9, step_cap=0.0):
        fn = None
        if step_cap:
            tgt = target_norm(E, step_cap)
            fn = lambda x0, i, n: x0 * torch.clamp(tgt / x0.norm(dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
        z_raw = c.DT.sample(c.model, c.torch, c.dev, E, guidance=guidance, seed=seed, steps=50, rescale=rescale, x0_fn=fn)
        return z_raw, G.cap_latent_norm(z_raw, E, S, post_cap)

    def judge(z, need_rot):
        X, nv_, du_ = c.decode(z * c.zs + c.zm)
        x, nv, dur = X[0].copy(), int(nv_[0]), float(du_[0])
        x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)
        px = DC.proxy_from_features(x, nv, dur, S["ref"], body_smooth=smooth)
        sp, mx = G.proxy_steps(px)
        rng, lu = G.tilt_pitch_range(x, nv, DC), G.landing_up(x, nv, DC)
        smooth_ok = sp <= G.MAX_SPIKES and mx <= G.MAX_STEP
        turn_ok = (rng >= G.ROT_RANGE) if need_rot else True
        land_ok = (lu >= G.LAND_UP) if need_rot else True
        return dict(smooth=bool(smooth_ok), turn=bool(turn_ok), land=bool(land_ok), good=bool(smooth_ok and turn_ok and land_ok), rng=float(rng), z=float(np.linalg.norm(z)))

    settings = {
        "production: g 2, cap after sampling 0.9": dict(guidance=2.0, post_cap=0.9),
        "no cap (g 2)": dict(guidance=2.0, post_cap=0.0),
        "g 1.5, cap 0.9": dict(guidance=1.5, post_cap=0.9),
        "g 1.0, cap 0.9": dict(guidance=1.0, post_cap=0.9),
        "g 2, rescaled guidance 0.7, cap 0.9": dict(guidance=2.0, rescale=0.7, post_cap=0.9),
        "g 2, cap inside every step 0.9": dict(guidance=2.0, post_cap=0.0, step_cap=0.9),
        "g 2, cap inside every step 0.9 + after 0.9": dict(guidance=2.0, post_cap=0.9, step_cap=0.9),
    }
    results = {}
    for prompt in prompts:
        E = embed(prompt)
        need_rot = G.needs_rotation(prompt)
        results[prompt] = dict(need_rot=bool(need_rot), settings={})
        for name, kw in settings.items():
            rows = [judge(draw(E, seed=s, **kw)[1], need_rot) for s in range(n_seeds)]
            results[prompt]["settings"][name] = dict(n=len(rows), **{k: sum(r[k] for r in rows) for k in ("smooth", "turn", "land", "good")}, median_rng=float(np.median([r["rng"] for r in rows])),
                                                     median_z=float(np.median([r["z"] for r in rows])))
            print(f"{prompt[:34]:34s} {name:46s} good {results[prompt]['settings'][name]['good']:3d}/{n_seeds}", flush=True)
        # the gate: the chance of at least one good candidate within t tries, production setting
        kw = settings["production: g 2, cap after sampling 0.9"]
        first_good = []
        for s in range(n_seeds):
            fg = None
            for k in range(max(tries_list)):
                if judge(draw(E, seed=s + 7919 * k, **kw)[1], need_rot)["good"]:
                    fg = k + 1
                    break
            first_good.append(fg)
        results[prompt]["tries"] = {t: sum(1 for f in first_good if f is not None and f <= t) for t in tries_list}
        results[prompt]["tries_n"] = n_seeds
        print(f"{prompt[:34]:34s} gate: " + ", ".join(f"{t} tries {results[prompt]['tries'][t]}/{n_seeds}" for t in tries_list), flush=True)

    lines = ["# Flip sweep: how often does a setting give a good clip?", "",
             f"{n_seeds} seeds per cell, one sample each (no gate), judged by the production criteria (smooth: at most {G.MAX_SPIKES} rotation step over 60 degrees and none over {G.MAX_STEP:g}; turn: pitch range at least {G.ROT_RANGE:g}; "
             f"landing: upright at least {G.LAND_UP}). 'good' = all three. Wilson 95% interval in brackets. median |z| is the latent norm after the cap (real flips 12.3-17.5).", ""]
    for prompt, r in results.items():
        lines += [f"**{prompt}** (rotation required: {r['need_rot']})", "", "| setting | good | turn completed | lands upright | smooth | median pitch range | median \\|z\\| |", "|---|---|---|---|---|---|---|"]
        for name, v in r["settings"].items():
            lo, hi = wilson(v["good"], v["n"])
            lines.append(f"| {name} | {v['good']}/{v['n']} ({100 * v['good'] / v['n']:.0f}%, {100 * lo:.0f}-{100 * hi:.0f}) | {v['turn']} | {v['land']} | {v['smooth']} | {v['median_rng']:.0f} | {v['median_z']:.1f} |")
        lines += ["", "Chance that the candidate gate (production setting) finds a good clip within t tries: " + ", ".join(f"{t}: {100 * r['tries'][t] / r['tries_n']:.0f}%" for t in tries_list), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flip_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "flip_sweep.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--prompts", nargs="+", default=["An object does a front flip.", "An object performs a front flip.", "An object performs a backflip."])
    ap.add_argument("--tries", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    run(a.prompts, a.seeds, a.smooth, a.tries)
