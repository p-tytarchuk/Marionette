"""Run one generated clip through every stage and show where it breaks. Read-only on the dataset; writes only to data/analysis/trace_generation/<label>/.

  python scripts/generate/trace_generation.py "An object does a front flip." --seed 11 --target mixamo_Aciana [--reference Front_Flip] [--check-seed 110]

The chain (generate_motion.generate), one stage at a time, with the numbers each stage is judged by:
  1 text      prompt -> the training caption wording (rules) -> CLIP embedding
  2 sampler   latent diffusion with guidance -> a latent z; |z| is compared with real clips (real flips 12-17.5, guided samples about 22) and then capped (znorm)
  3 decoder   z -> role features (root velocity, yaw rate, hips height and tilt, head, the 4 limbs' knee/elbow and foot/hand, contacts) and a duration
  4 IK        role features -> a clip on the 22-joint Mixamo skeleton (two-bone IK), smoothed in the body frame; the quality gate counts rotation steps over 60 degrees
  5 retarget  the proxy clip -> the target character (retarget.py): reach failures, sliding, lift, penetration, rotation steps
A REAL clip (default Front_Flip) goes through stages 4 and 5 from its own measured features, so every number has a reference: what the IK and the retargeter do to a clip that is fine.
--check-seed N reproduces an existing generated clip with seed N (the same settings) and compares it with the file on disk, to show that this trace is the production pipeline.
Needs the conda env Mario (torch, transformers). The first call loads the models (about a minute).
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "analysis" / "trace_generation"


def step_series(L):
    """per frame: the largest rotation step (degrees) over all joints between that frame and the next; also the joint that makes it"""
    import retarget as R
    from convert_clips import qmul
    T, J = L.shape[:2]
    dq = qmul(R.qconj(L[:-1].reshape(-1, 4)), L[1:].reshape(-1, 4))
    a = np.degrees(2 * np.arccos(np.clip(np.abs(dq[:, 0]), 0, 1))).reshape(T - 1, J)
    return a.max(1), a.argmax(1)


def retarget_proxy(px, ref, target, tmp):
    """stage 5 without writing into data/: the proxy goes through a scratch GLB exactly as to_character does, then retarget.py"""
    import decoded_to_character as DC
    import retarget as R
    p = Path(tmp) / "proxy.glb"
    DC.write_proxy(p, px, ref, "proxy")
    src = R.load_clip(str(p))
    sroles = R.label_roles(src, R.clean_names("mixamo", "mixamo"))
    ds, rid = R.rig_of(target)
    tgt = R.load_skeleton(R.CHARS / target / "model.glb")
    troles = R.label_roles(tgt, R.clean_names(ds, rid))
    params = dict(stride=1.0, vgain=1.0, forelimbs="auto", ik=True, shape=True, tilt_limit=35.0, stance_tilt=12.0, spine_gain=0.6)   # the values to_character uses
    L, root, info, Wp = R.retarget(src, sroles, tgt, troles, params, src["fps"])
    return dict(L=L, Wp=Wp, info=info, parents=tgt["parents"], names=tgt["names"])


def series_from_features(x, nv):
    """body pitch (degrees, unwrapped; 0 = upright, +-180 = upside down) and hips height (hips heights) per frame, from role features"""
    up = x[:nv, 8:10]                                                        # body-up axis (y, z)
    pitch = np.degrees(np.unwrap(np.arctan2(up[:, 1], up[:, 0])))
    return pitch, x[:nv, 3]


def generated(G, S, prompt, seed, guidance, znorm, smooth, tries, rewrite=True):
    """stages 1 to 4 as generate() runs them (plain mode), keeping every intermediate; the candidate loop and the quality gate are the production ones"""
    DC, c, sm = S["DC"], S["c"], S["sm"]
    if "fe" not in S:
        import text_frontend as TF
        S["fe"] = TF.Frontend()
    typed = prompt
    if rewrite:
        prompt = S["fe"].motion_text(S["fe"].split(prompt)[1] or prompt) or prompt
    E = S["encode"]([prompt], S["enc"])[0] - S["mu0"]
    E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    need_rot = tries > 1 and G.needs_rotation(prompt)
    cands = []
    for k in range(tries):
        z_raw = sm.sample(E, guidance=guidance, seed=seed + 7919 * k, k=None)
        z = G.cap_latent_norm(z_raw, E, S, znorm)
        X, nv_, du_ = c.decode(z * c.zs + c.zm)
        x, nv, dur = X[0].copy(), int(nv_[0]), float(du_[0])
        x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)
        px = DC.proxy_from_features(x, nv, dur, S["ref"], body_smooth=smooth)
        sp, mx = G.proxy_steps(px)
        rng, lu = G.tilt_pitch_range(x, nv, DC), G.landing_up(x, nv, DC)
        short = need_rot and rng < G.ROT_RANGE
        down = need_rot and lu < G.LAND_UP
        ok = sp <= G.MAX_SPIKES and mx <= G.MAX_STEP and not short and not down
        cands.append(dict(k=k, ok=bool(ok), score=sp + mx / 100 + (5 if short else 0) + (2 if down else 0), z_raw=float(np.linalg.norm(z_raw)), z_capped=float(np.linalg.norm(z)),
                          x=x, nv=nv, dur=dur, px=px, spikes=sp, max_step=mx, pitch_range=rng, landing_up=lu, short=bool(short), down=bool(down)))
        if ok:
            break
    best = next((cd for cd in cands if cd["ok"]), None) or min(cands, key=lambda cd: cd["score"])
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:30]
    best["label"] = f"gen_{slug}_s{seed}_g{guidance:g}_plain" + (f"_sm{smooth:g}" if smooth else "") + (f"_zn{znorm:g}" if znorm else "") + (f"_t{best['k']}" if best["k"] else "")
    return dict(typed=typed, prompt=prompt, candidates=cands, best=best)


def real_reference(DC, ref, name, smooth):
    """a real clip, from its own measured features through the same IK: the proxy a perfect generator would give"""
    import motion_features as MF
    import retarget as R
    rep, _ = R.clip_index()
    sk = R.load_clip(str(R.ROOT / rep[name]))
    rl = R.label_roles(sk, R.clean_names("mixamo", "mixamo"))
    f, nv, dur = MF.clip_features(sk, rl)
    return dict(x=f, nv=nv, dur=dur, px=DC.proxy_from_features(f, nv, dur, ref, body_smooth=smooth))


def strip(ax_row, Wp, parents, frames, title, ylim, flip_y=False):
    import matplotlib.pyplot as plt
    n = len(frames)
    for ax, t in zip(ax_row, frames):
        P = Wp[t]
        z0 = P[:, 2].mean()
        for j, p in enumerate(parents):
            if p >= 0:
                ax.plot([P[p, 2] - z0, P[j, 2] - z0], [P[p, 1], P[j, 1]], "-", color="#245", lw=1.1)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"f{t}", fontsize=7)
    ax_row[0].set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center")


def run(prompt, seed, guidance, znorm, smooth, tries, target, reference, check_seed, rewrite=True):
    import generate_motion as G
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    S = G._load()
    DC, ref = S["DC"], S["ref"]
    print(f"models loaded in {time.time() - t0:.0f} s", flush=True)
    g = generated(G, S, prompt, seed, guidance, znorm, smooth, tries, rewrite)
    b = g["best"]
    out = OUT / b["label"]
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    rt = retarget_proxy(b["px"], ref, target, tmp)
    real = real_reference(DC, ref, reference, smooth)
    rt_real = retarget_proxy(real["px"], ref, target, tmp)
    rows = {}
    for tag, x, nv, px, r in (("generated", b["x"], b["nv"], b["px"], rt), (reference + " (real)", real["x"], real["nv"], real["px"], rt_real)):
        sp, mx = G.proxy_steps(px)
        pitch, hh = series_from_features(x, nv)
        info = r["info"]
        rows[tag] = dict(frames=int(nv), duration_s=round(float(b["dur"] if tag == "generated" else real["dur"]), 2), pitch_range_deg=round(float(np.ptp(pitch))), hips_height_min=round(float(hh.min()), 2),
                         hips_height_max=round(float(hh.max()), 2), landing_up=round(float(G.landing_up(x, nv, DC)), 2), proxy_spikes=sp, proxy_max_step_deg=round(mx),
                         target_rot_steps_over_60=int(info["rot_steps_over_60deg"]), target_reach_fail_frames=int(info["reach_fail_frames"]), target_skate_mean=round(float(info["skate_mean"]), 3),
                         target_lift_max_m=round(float(info["lift_max"]), 3), target_penetration_max=round(float(info["penetration_max"]), 3))
    # the series
    names = ref[0]["names"]
    s_gen, j_gen = step_series(b["px"]["L"])
    s_real, j_real = step_series(real["px"]["L"])
    t_gen, tj_gen = step_series(rt["L"])
    t_real, _ = step_series(rt_real["L"])
    pit_g, hh_g = series_from_features(b["x"], b["nv"])
    pit_r, hh_r = series_from_features(real["x"], real["nv"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    ax[0][0].plot(np.arange(len(pit_g)) / 30, pit_g, label=f"generated (seed {seed})", color="#c33")
    ax[0][0].plot(np.arange(len(pit_r)) / 30, pit_r, label=reference + " (real)", color="#37c")
    ax[0][0].set_title("body pitch (decoded role features)", fontsize=9)
    ax[0][0].set_ylabel("degrees")
    ax[0][1].plot(np.arange(len(hh_g)) / 30, hh_g, color="#c33")
    ax[0][1].plot(np.arange(len(hh_r)) / 30, hh_r, color="#37c")
    ax[0][1].set_title("hips height above the lowest foot (hips heights)", fontsize=9)
    ax[1][0].plot(np.arange(len(s_gen)) / 30, s_gen, color="#c33")
    ax[1][0].plot(np.arange(len(s_real)) / 30, s_real, color="#37c")
    ax[1][0].axhline(60, color="#999", ls="--", lw=0.8)
    ax[1][0].set_title("stage 4, IK proxy: largest joint rotation step per frame (deg; dashed = 60)", fontsize=9)
    ax[1][1].plot(np.arange(len(t_gen)) / 30, t_gen, color="#c33")
    ax[1][1].plot(np.arange(len(t_real)) / 30, t_real, color="#37c")
    ax[1][1].axhline(60, color="#999", ls="--", lw=0.8)
    ax[1][1].set_title(f"stage 5, on {target.replace('mixamo_', '')}: largest joint rotation step per frame", fontsize=9)
    for a in ax.ravel():
        a.set_xlabel("seconds", fontsize=8)
    ax[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "series.png", dpi=85)
    plt.close(fig)
    # film strips: the generated proxy and its result, the real reference and its result
    nfr = 9
    fr = lambda n: np.linspace(0, n - 1, nfr).round().astype(int)
    fig, axs = plt.subplots(4, nfr, figsize=(nfr * 1.5, 4 * 1.7))
    rp = ref[0]["parents"]
    strip(axs[0], b["px"]["Wp"], rp, fr(len(b["px"]["Wp"])), "generated\nproxy", (-0.1, 2.4))
    strip(axs[1], rt["Wp"], rt["parents"], fr(len(rt["Wp"])), f"generated on\n{target.replace('mixamo_', '')}", (-0.1, 2.4))
    strip(axs[2], real["px"]["Wp"], rp, fr(len(real["px"]["Wp"])), f"{reference}\nproxy (real)", (-0.1, 2.4))
    strip(axs[3], rt_real["Wp"], rt_real["parents"], fr(len(rt_real["Wp"])), f"{reference} on\n{target.replace('mixamo_', '')}", (-0.1, 2.4))
    fig.tight_layout()
    fig.savefig(out / "filmstrips.png", dpi=85)
    plt.close(fig)
    worst = int(np.argmax(s_gen))
    summary = dict(label=b["label"], typed=g["typed"], prompt_used=g["prompt"], seed=seed, guidance=guidance, znorm=znorm, smooth=smooth, target=target, reference=reference,
                   candidates=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in cd.items() if k in ("k", "ok", "z_raw", "z_capped", "spikes", "max_step", "pitch_range", "landing_up", "short", "down")} for cd in g["candidates"]],
                   chosen_candidate=b["k"], gate=dict(max_spikes=G.MAX_SPIKES, max_step=G.MAX_STEP, rot_range=G.ROT_RANGE, land_up=G.LAND_UP), stages=rows,
                   worst_proxy_step=dict(frame=worst, degrees=round(float(s_gen[worst])), joint=names[int(j_gen[worst])]),
                   spike_frames_proxy=[int(i) for i in np.flatnonzero(s_gen > 60)], spike_frames_target=[int(i) for i in np.flatnonzero(t_gen > 60)])
    # reproduce an existing clip, to show this is the production pipeline
    if check_seed is not None:
        g2 = generated(G, S, prompt, check_seed, guidance, znorm, smooth, tries, rewrite)
        lab = g2["best"]["label"]
        f = ROOT / "data" / "decoded" / f"{lab}.glb"
        if f.exists():
            import retarget as R
            old = R.load_clip(str(f))
            n = min(len(old["Wp"]), len(g2["best"]["px"]["Wp"]))
            summary["reproduction"] = dict(label=lab, frames_here=len(g2["best"]["px"]["Wp"]), frames_on_disk=len(old["Wp"]),
                                           max_joint_position_difference_m=round(float(np.abs(old["Wp"][:n] - g2["best"]["px"]["Wp"][:n]).max()), 5))
        else:
            summary["reproduction"] = dict(label=lab, error="no such file on disk")
    (out / "trace.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nfigures: {out / 'filmstrips.png'}  {out / 'series.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--znorm", type=float, default=0.9)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--target", default="mixamo_Aciana")
    ap.add_argument("--reference", default="Front_Flip", help="a real shared clip to run through the same IK and retargeter")
    ap.add_argument("--no-rewrite", action="store_true", help="use the prompt exactly as typed (the generator page's rewrite option off)")
    ap.add_argument("--check-seed", type=int, default=None, help="also reproduce the generated clip with this seed and compare it with data/decoded")
    a = ap.parse_args()
    run(a.prompt, a.seed, a.guidance, a.znorm, a.smooth, a.tries, a.target, a.reference, a.check_seed, not a.no_rewrite)
