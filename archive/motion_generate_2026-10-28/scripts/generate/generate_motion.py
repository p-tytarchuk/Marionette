#!/usr/bin/env python3
"""Text prompt -> generated motion -> a clip on a character (what viewer/generate.html calls).

  python3 scripts/generate/generate_motion.py "An object runs forward." --target mixamo_Aciana [--guidance 2] [--seed 0] [--mode auto|walk|run|plain] [--no-rewrite] [--smooth 3] [--znorm 0.9] [--tries 4] [--straight 0|1|2]

By default the prompt is first rewritten into the wording of the training captions ("A cat do it backflip" -> "An object performs a backflip.", scripts/text/rewrite_prompt.py, Qwen2.5-1.5B, loaded on the
first call); --no-rewrite uses the text as typed. Reworded prompts: generated R@5 +4 to +9 points; canonical prompts unchanged (docs/EMBED.md, "Prompt rewriting").

Chain: CLIP text embedding -> latent diffusion (data/embeddings/z_diffusion.pt) -> role decoder -> Mixamo-skeleton proxy (two-bone IK)
-> rule-based retargeter -> data/characters/<target>/anims/rt__dec_gen_<...>.glb (listed by viewer/compare.html).
Sampler modes (docs/EMBED.md, "Speed-aware sampler" and "Runs: a duration band"):
  walk   speed cap at the prompt's prior mean + 0.5 sd (validated for walking prompts)
  run    duration band +-1 sd and the 0.9-quantile speed cap (validated for running prompts)
  plain  the unconstrained sampler
  auto   walk/run if the prompt says so, plain otherwise (the constraints were not validated for other actions)
Needs the conda env "Mario" (torch, transformers). The first call loads the models and the training pairs (about a minute).
"""
import argparse
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]

_lock = threading.Lock()
_state = {}


def pick_mode(prompt, mode="auto"):
    if mode != "auto":
        return mode
    if re.search(r"\b(runs?|running|sprints?|sprinting|jogs?|jogging)\b", prompt, re.I):
        return "run"
    if re.search(r"\b(walks?|walking|strolls?|strolling)\b", prompt, re.I):
        return "walk"
    return "plain"


def cap_latent_norm(zn, E, S, factor=1.0, k=40):
    """Classifier-free guidance pushes the sampled latent far from the data: |z| (standardised) is about 22 at guidance 2, real clips 12.6-19.6, and the decoder's output
    gets rough there (backflips: 6.9 rotation spikes per clip against 2.5 once the norm is capped, docs/EMBED.md "Latent norm"). The latent is scaled down so that its norm is at most
    `factor` x the mean norm of the k training clips whose captions are nearest to the prompt (about 15.6 for flips). Never scales up. factor 0 = off."""
    if not factor:
        return zn
    c = S["c"]
    if "z_norm" not in S:
        tr = np.where(c.P["split"] == "train")[0]
        S["z_norm"] = (c.P["text_clip"][tr], np.linalg.norm((c.P["z"][tr] - c.zm) / c.zs, axis=1))
    T, nrm = S["z_norm"]
    nn = np.argsort(-(T @ E[0]))[:k]
    target = factor * float(nrm[nn].mean())
    return zn * min(1.0, target / float(np.linalg.norm(zn)))


# Candidate gate (plain mode, docs/EMBED.md "Seed 1: lying, then a pop"): a sample is accepted if the IK proxy has at most MAX_SPIKES joint rotation steps over 60 degrees, none over MAX_STEP degrees,
# and, for flip prompts (needs_rotation), the hips tilt completes a turn (pitch range >= ROT_RANGE degrees; real flips: 335-569) and the clip ends upright (body-up axis, worst of the
# last LAND_FRAMES frames, at least LAND_UP, i.e. within 37 degrees of vertical; real flips that land on their feet: 0.84-1.0, many generated ones end lying or kneeling, 0.2-0.5).
MAX_SPIKES, MAX_STEP, ROT_RANGE = 1, 100.0, 300.0
LAND_UP, LAND_FRAMES = 0.8, 8


def gate_report(prompt, quality, land=True):
    """The candidate gate applied to the quality that generate() stores with a clip (decoded.quality: spikes, max_step_deg, pitch_range_deg, landing_up), whether or not the gate was on when the clip was made
    (tries 1 means no choice among candidates, not that the clip would fail). Returns None for a clip without stored quality (older clips, dataset clips), else dict(passed, reasons, needs_rotation). The stored step, range and
    landing are rounded, so a value within the rounding of a limit can differ from the float comparison made while generating."""
    if not isinstance(quality, dict) or quality.get("spikes") is None:
        return None
    sp, mx, rng, lu = quality.get("spikes"), quality.get("max_step_deg"), quality.get("pitch_range_deg"), quality.get("landing_up")
    need = bool(needs_rotation(prompt or ""))
    reasons = []
    if sp > MAX_SPIKES:
        reasons.append(f"{sp} rotation steps over 60 degrees (the gate allows {MAX_SPIKES})")
    if mx is not None and mx > MAX_STEP:
        reasons.append(f"largest rotation step {mx} degrees (the gate allows {MAX_STEP:g})")
    if need and rng is not None and rng < ROT_RANGE:
        reasons.append(f"the body turns only {rng} degrees (a flip must turn {ROT_RANGE:g})")
    if need and land and lu is not None and lu < LAND_UP:
        reasons.append(f"ends {lu} upright (a flip must end at {LAND_UP} or more)")
    return dict(passed=not reasons, reasons=reasons, needs_rotation=need)


def _gate_selftest():
    flip = "An object does a front flip."
    # the clip traced on 2026-10-28 (seed 11, one candidate): a smooth proxy, but a 135-degree turn that ends lying down
    r = gate_report(flip, dict(spikes=1, max_step_deg=95, pitch_range_deg=135, landing_up=-0.14))
    assert r and not r["passed"] and len(r["reasons"]) == 2 and r["needs_rotation"], r
    # candidate 2 of the same seed with the rewritten prompt: a clean flip
    r = gate_report("An object performs a front flip.", dict(spikes=0, max_step_deg=47, pitch_range_deg=359, landing_up=0.96))
    assert r and r["passed"] and not r["reasons"], r
    # a prompt that needs no turn is not held to one
    r = gate_report("An object waves.", dict(spikes=0, max_step_deg=20, pitch_range_deg=5, landing_up=0.1))
    assert r and r["passed"] and not r["needs_rotation"], r
    # too many spikes or a step over the limit fails any prompt; --no-landing style: land=False ignores the landing
    assert not gate_report("An object waves.", dict(spikes=MAX_SPIKES + 1, max_step_deg=20))["passed"]
    assert not gate_report("An object waves.", dict(spikes=0, max_step_deg=MAX_STEP + 1))["passed"]
    assert gate_report(flip, dict(spikes=0, max_step_deg=40, pitch_range_deg=340, landing_up=0.1), land=False)["passed"]
    # no stored quality: no verdict
    assert gate_report(flip, None) is None and gate_report(flip, {}) is None
    print("generate_motion gate_report self-test: the traced bad clip fails twice, the clean candidate passes, non-flip prompts are not held to a turn, limits and missing data behave")


def body_up(x, n, DC):
    """The hips' own up axis in the heading frame, (n, 3): y is 1 when the body is upright, 0 on its side, -1 upside down."""
    from scipy.spatial.transform import Rotation as Rot
    q = DC.quat_from_6d(np.asarray(x, float)[:n, 4:10])
    return Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :, 1]


def tilt_pitch_range(x, n, DC):
    """Range (degrees) of the unwrapped pitch of the decoded hips tilt over the clip: about 360 for a flip, 100-150 for a fall onto the back."""
    up = body_up(x, n, DC)
    p = np.unwrap(np.arctan2(-up[:, 2], up[:, 1]))
    return float(np.degrees(p.max() - p.min()))


def landing_up(x, n, DC):
    """Worst (lowest) body-up y over the last LAND_FRAMES frames of the clip: how upright it ends."""
    return float(body_up(x, n, DC)[-LAND_FRAMES:, 1].min())


def needs_rotation(prompt):
    """True for flip prompts: the 16 real training clips captioned back flip / front flip / flips backward all turn the hips by 335-415+ degrees (tilt_pitch_range). Cartwheels
    (about the other axis, range 150-190) and somersaults (one clip) are not covered. A word rule like pick_mode, not a classifier; the nearest-caption version did not work (docs/EMBED.md)."""
    return bool(re.search(r"\b(back|front)?\s?flip(s|ped|ping)?\b", prompt, re.I))


def straighten(x, level=1):
    """Make a decoded clip a straight-line motion (docs/EMBED.md "2D test"): level 1 zeroes the sideways velocity and the yaw rate (features 1, 2); level 2 also keeps only the pitch of the
    hips tilt (the rotation about the across axis, from the body-up axis; roll and twist go). Applied to the decoded features before the IK proxy, so the gate sees the straightened clip."""
    x = np.array(x, np.float32)
    if level >= 1:
        x[..., 1] = 0.0
        x[..., 2] = 0.0
    if level >= 2:
        a = np.arctan2(x[..., 9], x[..., 8])                                   # body-up axis: (7, 8, 9) = (x, y, z)
        x[..., 4:10] = np.stack([np.ones_like(a), np.zeros_like(a), np.zeros_like(a), np.zeros_like(a), np.cos(a), np.sin(a)], -1)
    return x


def proxy_steps(px):
    """(number of joint rotation steps over 60 degrees between frames, largest step in degrees) of a proxy clip."""
    import retarget as R
    from convert_clips import qmul
    L = px["L"]
    dq = qmul(R.qconj(L[:-1].reshape(-1, 4)), L[1:].reshape(-1, 4))
    a = np.degrees(2 * np.arccos(np.clip(np.abs(dq[:, 0]), 0, 1)))
    return int((a > 60).sum()), float(a.max())


def _load():
    """Models and data, once."""
    if _state:
        return _state
    import walk_check as WC
    import decoded_to_character as DC
    from speed_sampler import SpeedModel
    from embed import encode, load_text_encoder
    t0 = time.time()
    c = WC.prepare("v2")
    _state.update(c=c, sm=SpeedModel(c), enc=load_text_encoder(), encode=encode, DC=DC, ref=DC.ref_skeleton(),
                  mu0=np.load(c.MA.EMB / "text_mean.npy"), load_s=round(time.time() - t0, 1))
    return _state


def sample_step_cap(S, E, guidance, seed, factor, k=40):
    """Plain sampling (DDIM, classifier-free guidance) with the clean-latent estimate capped at EVERY step to `factor` x the mean |z| of the k training clips whose captions are nearest to the prompt (the target of
    cap_latent_norm), instead of rescaling the finished latent. scripts/generate/flip_sweep.py: more completed, upright flips than the cap after sampling (22% / 21% against 12% / 15% of single samples for a front flip and
    a backflip), but more spikes; scripts/generate/step_cap_check.py: the side effects on the held-out prompts. Opt-in (step_cap > 0); desktop runtime only."""
    import torch
    c = S["c"]
    cap_latent_norm(np.ones((1, 256), np.float32), E, S, 1.0)           # fills S["z_norm"] (the training latents' norms) on first use
    T, nrm = S["z_norm"]
    tgt = factor * float(nrm[np.argsort(-(T @ E[0]))[:k]].mean())
    fn = lambda x0, i, n: x0 * torch.clamp(tgt / x0.norm(dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
    return c.DT.sample(c.model, c.torch, c.dev, E, guidance=guidance, seed=seed, steps=50, x0_fn=fn)


def generate(prompt, target, guidance=2.0, seed=0, mode="auto", rewrite=True, smooth=3.0, znorm=0.9, tries=4, straight=0, land=True, frontend="rules", runtime="desktop", threads=4, keep_proxy=False, step_cap=0.0):
    """One sample for a prompt, put on `target` (a character folder name). Returns what the viewers need plus the retargeter's metrics.
    rewrite: first turn the prompt into the dataset's caption wording (the original is returned as `typed`, the text used as `prompt`).
    smooth: sigma (stored frames) of the body-frame low-pass on the decoded limb joints before IK (decoded_to_character.body_frame_filter); 0 = off. Generated flips have about 16 rotation
    spikes per clip without it, 7 with 3 (docs/EMBED.md, "Backflip spikes").
    frontend: "rules" (default; word lists and rules, no language model, nothing to load) or "llm" (Qwen2.5-1.5B rewrites the prompt; better on garbled wording, see docs/EMBED.md); only used when rewrite is on.
    land: with the gate on, flip prompts must also end upright (LAND_UP); False = only smoothness and a completed turn are required.
    runtime: "desktop" (torch) or "mobile" (scripts/generate/mobile_runtime.py: ONNX Runtime on the CPU with `threads` threads, int8 denoiser and decoder; plain mode only, walk / run use the desktop sampler).
    The result carries timing_ms (front end, text encoder, sampler, decoder, IK + gate, retarget + file writing, total).
    tries: plain mode only. Up to this many candidates (seed, seed + 7919, ...) are drawn and the first one that passes the gate above is used; if none passes, the best-scoring one. 1 = no gate.
    target=None: no GLB and no retargeting (scripts/lottie2d/motion_to_2d.py, the 2D / Lottie path). keep_proxy: the result also carries `proxy` (the clip on the Mixamo skeleton: Wp, Wr, root, fps) and `ref`."""
    typed = prompt = " ".join(prompt.split())
    if not prompt:
        raise ValueError("empty prompt")
    T0 = time.perf_counter()
    tm = dict(frontend=0.0, text_encoder=0.0, sampler=0.0, decoder=0.0, ik_gate=0.0, retarget=0.0)
    with _lock:
        tl = time.perf_counter()
        S = _load()
        load_ms = 1000 * (time.perf_counter() - tl)                  # one-time model loading is reported separately from the generation time
        T0 = time.perf_counter()
        c, sm, DC = S["c"], S["sm"], S["DC"]
        if rewrite and frontend == "rules":                       # no language model: split off any shape words, then write the motion in the dataset's wording by rules (scripts/text/text_frontend.py)
            if "fe" not in S:
                import text_frontend as TF
                S["fe"] = TF.Frontend()
            prompt = S["fe"].motion_text(S["fe"].split(prompt)[1] or prompt) or prompt
        elif rewrite:
            if "rw" not in S:
                from rewrite_prompt import Rewriter
                S["rw"] = Rewriter(encoder=S["enc"])
                import types
                import text_to_character as TC
                S["rw"].PP = types.SimpleNamespace(load_llm=lambda name: TC.shared_llm())      # one Qwen2.5-3B for splitting and rewriting, loaded only when a prompt needs rewriting
            prompt = S["rw"].rewrite([prompt])[0]
        tm["frontend"] = time.perf_counter() - T0
        mode_used = pick_mode(prompt, mode)
        mob, mobile_note = None, None
        if runtime == "mobile":
            import mobile_runtime as MR
            if not MR.available():
                raise ValueError("mobile mode needs data/mobile/*.onnx (python3 scripts/generate/export_onnx.py) and onnxruntime")
            if mode_used != "plain":
                mobile_note = "walk / run use the desktop sampler: the speed constraint is not ported to the mobile runtime"
            else:
                tl = time.perf_counter()
                mob = MR.get(S, threads)
                dt = time.perf_counter() - tl
                load_ms += 1000 * dt
                T0 += dt
        t1 = time.perf_counter()
        if mob:
            E = mob.encode(prompt)
        else:
            E = S["encode"]([prompt], S["enc"])[0] - S["mu0"]
            E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
        tm["text_encoder"] = time.perf_counter() - t1
        t0 = time.time()
        kw = dict(plain=dict(k=None),
                  walk=dict(k=0.5, exact=True, keep_gait=True),
                  run=dict(k=0.5, exact=True, keep_gait=True, dur_k=1.0, cap_q=0.9))[mode_used]
        scap = step_cap if mode_used == "plain" else 0
        if scap and runtime == "mobile":
            raise ValueError("the cap inside the sampler is not ported to the mobile runtime")
        zcap = znorm if mode_used == "plain" else 0          # walk/run keep their validated constraints (the cap changes their speed and duration, docs/EMBED.md "Latent norm")
        ntry = max(1, int(tries)) if mode_used == "plain" else 1
        need_rot = ntry > 1 and needs_rotation(prompt)
        cands = []
        for k in range(ntry):
            t1 = time.perf_counter()
            zn = mob.sample(E, guidance, seed + 7919 * k) if mob else (sample_step_cap(S, E, guidance, seed + 7919 * k, scap) if scap else sm.sample(E, guidance=guidance, seed=seed + 7919 * k, **kw))
            zn = cap_latent_norm(zn, E, S, zcap)
            t2 = time.perf_counter()
            X, nv_, du_ = (mob or c).decode(zn * c.zs + c.zm)
            t3 = time.perf_counter()
            tm["sampler"] += t2 - t1
            tm["decoder"] += t3 - t2
            x, nv, dur = X[0].copy(), int(nv_[0]), float(du_[0])
            x[..., 37:41] = (x[..., 37:41] > 0.5).astype(np.float32)               # contact flags
            if straight:
                x = straighten(x, straight)
            px = DC.proxy_from_features(x, nv, dur, S["ref"], body_smooth=smooth)
            sp, mx = proxy_steps(px)
            rng = tilt_pitch_range(x, nv, DC)
            lu = landing_up(x, nv, DC)
            tm["ik_gate"] += time.perf_counter() - t3
            short = need_rot and rng < ROT_RANGE                                     # a flip prompt whose sample does not complete a turn
            down = need_rot and land and lu < LAND_UP                                         # ... or ends lying / crouched over instead of upright
            ok = sp <= MAX_SPIKES and mx <= MAX_STEP and not short and not down
            cands.append((ok, sp + mx / 100 + (5 if short else 0) + (2 if down else 0), k, x, nv, dur, px,
                          dict(spikes=sp, max_step_deg=round(mx), pitch_range_deg=round(rng), landing_up=round(lu, 2), rotation_required=need_rot)))
            if ok or ntry == 1:
                break
        best = next((cd for cd in cands if cd[0]), None) or min(cands, key=lambda cd: cd[1])
        _, _, kbest, x, nv, dur, px, quality = best
        gen_s = time.time() - t0
        spd = float(np.abs(x[:nv, 0]).sum() / dur)
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:30]
        label = f"gen_{slug}_s{seed}_g{guidance:g}_{mode_used}" + (f"_sm{smooth:g}" if smooth else "") + (f"_zn{zcap:g}" if zcap else "") + (f"_sc{scap:g}" if scap else "") + (f"_t{kbest}" if kbest else "") + (f"_st{straight}" if straight else "")      # the slug is of the text the model saw (rewritten or as typed)
        path = DC.OUT_DIR / f"{label}.glb"
        t1 = time.perf_counter()
        if target is None:
            info = {}
        else:
            DC.write_proxy(path, px, S["ref"], label)
            info = DC.to_character(path, target, label, prompt, S["ref"],
                                   extra=dict(kind="generated", prompt=prompt, typed=typed, guidance=guidance, seed=seed, mode=mode_used, smooth=smooth, znorm=zcap, step_cap=scap, tries_used=kbest + 1, quality=quality, straight=straight))
        tm["retarget"] = time.perf_counter() - t1
        timing = {k: round(1000 * v, 1) for k, v in tm.items()}
        timing.update(one_time_load=round(load_ms, 1), total=round(1000 * (time.perf_counter() - T0), 1), candidates=len(cands), runtime=("mobile" if mob else "desktop"), threads=int(threads) if mob else None)
        extra_out = dict(proxy=px, ref=S["ref"]) if keep_proxy else {}
        return dict(**extra_out, timing_ms=timing, mobile_note=mobile_note, target=target, name=f"rt__dec_{label}", prompt=prompt, typed=typed, mode=mode_used, guidance=guidance, seed=seed, smooth=smooth, znorm=zcap, step_cap=scap, tries_used=kbest + 1, gate_passed=bool(best[0]), quality=quality, straight=straight, land=land, frontend=frontend if rewrite else "off",
                    duration_s=round(dur, 2), speed_hips_h_per_s=round(spd, 2), generate_s=round(gen_s, 1), load_s=S["load_s"],
                    metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items() if isinstance(v, (int, float, str))})


def main():
    if "--gate-selftest" in sys.argv:
        return _gate_selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--target", default="mixamo_Aciana")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "walk", "run", "plain"], default="auto")
    ap.add_argument("--no-rewrite", action="store_true", help="use the prompt as typed")
    ap.add_argument("--frontend", choices=["rules", "llm"], default="rules", help="how the prompt is rewritten into the dataset's wording: rules (no language model) or the Qwen2.5-1.5B LLM"),
    ap.add_argument("--smooth", type=float, default=3.0, help="body-frame smoothing of the decoded limbs, frames (0 = off)")
    ap.add_argument("--step-cap", type=float, default=0.0, help="plain mode, opt-in: cap the latent inside every denoising step at this x the neighbours' mean norm (see sample_step_cap; 0 = off)")
    ap.add_argument("--znorm", type=float, default=0.9, help="plain mode: cap |z| at this x the mean norm of the prompt's nearest training latents (0 = off), see cap_latent_norm")
    ap.add_argument("--straight", type=int, choices=[0, 1, 2], default=0, help="straight-line motion: 1 = zero sideways velocity and yaw rate of the decoded clip, 2 = also keep only the pitch of the hips tilt (see straighten)")
    ap.add_argument("--mobile", action="store_true", help="run the models as a phone would: ONNX Runtime on the CPU, int8 denoiser + decoder (needs data/mobile, see mobile_runtime.py)")
    ap.add_argument("--threads", type=int, default=4, help="--mobile: CPU threads")
    ap.add_argument("--no-landing", action="store_true", help="flip prompts: do not require an upright ending in the gate (see LAND_UP)")
    ap.add_argument("--tries", type=int, default=4, help="plain mode: draw up to this many candidates and keep the first that passes the quality gate (1 = off), see MAX_SPIKES in generate_motion.py")
    a = ap.parse_args()
    r = generate(a.prompt, a.target, a.guidance, a.seed, a.mode, rewrite=not a.no_rewrite, smooth=a.smooth, znorm=a.znorm, step_cap=a.step_cap, tries=a.tries, straight=a.straight, land=not a.no_landing, frontend=a.frontend, runtime="mobile" if a.mobile else "desktop", threads=a.threads)
    print(r)
    print(f"view: http://127.0.0.1:8765/viewer/compare.html#{r['target']}|{r['name']}   (python viewer/serve.py)")


if __name__ == "__main__":
    main()
