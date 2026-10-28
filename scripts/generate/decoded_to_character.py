#!/usr/bin/env python3
"""Put decoded latents on a character: role features -> human skeleton clip (IK) -> any biped or quadruped target (retarget.py).

  python3 scripts/generate/decoded_to_character.py --selftest      # features of real clips -> IK -> features again: how faithful is the IK?
  python3 scripts/generate/decoded_to_character.py --examples      # decode the example latents, write them on characters (viewer/compare.html)

Stage 1 (this file): the role model decodes a latent to role features (root velocity, yaw rate, hips height and tilt, head, and the
knee/elbow and foot/hand of the 4 limbs, relative to the hips in the heading frame, in hips heights). They are turned into a clip on the
Mixamo reference skeleton (22 joints, data/clips/mixamo): the hips follow the integrated root motion and tilt, the spine bends so the
head lands where the features say, and each limb is solved with two-bone IK to the decoded foot/hand, its bend taken from the decoded
knee/elbow. Hips height is measured above the lowest foot (v2 features), so the decoded feet already touch the floor and legs are not stretched (leg_normalise=True restores the v1 behaviour).
Stage 2: the human clip goes through retarget.py's rule-based retargeter (foot contacts, pinning, ground fix) onto the chosen character,
exactly as a dataset clip does; the result is saved as rt__dec_<label>.glb next to the character, so viewer/compare.html lists it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
import retarget as R
from convert_clips import qmul, qrot, write_clip

REF_CLIP = ROOT / "data" / "clips" / "mixamo" / "Backflip.glb"
OUT_DIR = ROOT / "data" / "decoded"
FPS = 30


def ref_skeleton():
    sk = R.load_clip(str(REF_CLIP))
    roles = R.label_roles(sk, R.clean_names("mixamo", "mixamo"))
    return sk, roles


def resample(x, n):
    T = len(x)
    if T == n:
        return x
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def quat_from_6d(c):
    """(T, 6) first two matrix columns -> (T, 4) w-first unit quaternions, continuous in time."""
    from scipy.spatial.transform import Rotation
    a1 = R.unit(c[:, :3])
    a2 = c[:, 3:] - (a1 * c[:, 3:]).sum(1, keepdims=True) * a1
    a2 = R.unit(a2)
    a3 = np.cross(a1, a2)
    M = np.stack([a1, a2, a3], -1)                               # columns
    q = Rotation.from_matrix(M).as_quat()[:, [3, 0, 1, 2]]
    from convert_clips import qcontinuous
    return qcontinuous(q)


def body_frame_filter(f, sigma=0.0, median=0):
    """Smooth the limb joints (features 13-36) of a (n, 41) clip in the BODY frame: the heading-frame offsets are rotated back by the hips tilt, filtered over time (median of `median`
    frames to remove single-frame outliers, then a Gaussian of `sigma` frames), and rotated forward again. The tilt itself is untouched. A limb that moves with the body is nearly constant
    in the body frame, so this does not blur a flip the way filtering in the heading frame does."""
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.spatial.transform import Rotation as Rot
    f = np.array(f, float)
    q = quat_from_6d(f[:, 4:10])
    Rm = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    for s_ in range(13, 37, 3):
        b = np.einsum("tji,tj->ti", Rm, f[:, s_:s_ + 3])
        if median:
            b = median_filter(b, size=(median, 1), mode="nearest")
        if sigma:
            b = gaussian_filter1d(b, sigma, axis=0, mode="nearest")
        f[:, s_:s_ + 3] = np.einsum("tij,tj->ti", Rm, b)
    return f


def proxy_from_features(f, n_valid, dur, ref=None, leg_normalise=False, min_reach=None, chord_smooth=0.0, pole_smooth=0.0, body_smooth=0.0, body_median=0):
    """(128, 41) role features -> clip on the Mixamo skeleton. Returns dict(L (T,J,4 w-first), root (T,3), Wp, Wr, fps, info).
    IK stabilisers (off by default; generated clips fold limbs onto the joint, where the two-bone solve is unstable, see docs/EMBED.md "Backflip spikes"):
    min_reach: the foot/hand is kept at least this fraction of the limb length (upper + lower bone) from the hip/shoulder joint; chord_smooth / pole_smooth: Gaussian sigma (frames at 30 fps)
    applied to the hip-to-foot direction and to the bend (knee/elbow) direction before the per-frame solve; body_smooth / body_median: body_frame_filter() on the decoded limb joints (the effective one)."""
    sk, roles = ref or ref_skeleton()
    par, J = sk["parents"], len(sk["names"])
    hips, head = roles["hips"], roles["head"]
    H = float(sk["Wp0"][hips][1])
    f = np.asarray(f, float)[:n_valid]
    if body_smooth or body_median:
        f = body_frame_filter(f, body_smooth, body_median)
    T = max(int(round(dur * FPS)) + 1, 4)
    # absolute signals on the stored timeline, then resampled to 30 fps
    fwd, side, yaw = f[:, 0], f[:, 1], f[:, 2]
    psi = np.cumsum(yaw)
    c, s = np.cos(psi), np.sin(psi)
    hxz = np.stack([np.cumsum(fwd * s + side * c), np.cumsum(fwd * c - side * s)], -1)
    S = lambda x: resample(x, T)
    psi, hxz, hh, tilt6, headl, limbs = S(psi), S(hxz), S(f[:, 3]), S(f[:, 4:10]), S(f[:, 10:13]), S(f[:, 13:37])
    flags = (f[:, 37:41].mean(0) > 0.5)
    ident = np.tile(np.array([1.0, 0, 0, 0]), (T, 1))
    Yq = np.stack([np.cos(psi / 2), np.zeros(T), np.sin(psi / 2), np.zeros(T)], -1)
    D = qmul(Yq, quat_from_6d(tilt6))                            # hips world rotation change from rest
    # feature 3 is the hips height above the clip's lowest foot joint (v2 features): the proxy's lowest ankle stands at its rest height
    ankle_y = float(sk["Wp0"][roles["leg_L"][2]][1])
    hips_w = np.stack([hxz[:, 0] * H, hh * H + ankle_y, hxz[:, 1] * H], -1)
    # ---- hips + spine (carried with the hips, bent so the head lands on the decoded head)
    u0 = R.unit(sk["Wp0"][head] - sk["Wp0"][hips])
    u_h = qrot(D, np.broadcast_to(u0, (T, 3)))
    u_d = R.unit(qrot(Yq, headl))
    bend = R.q_swing(u_h, u_d)
    chain = roles["spine"]
    seg = [np.linalg.norm(sk["Wp0"][chain[i + 1]] - sk["Wp0"][chain[i]]) for i in range(len(chain) - 1)]
    cum = np.r_[0.0, np.cumsum(seg)] / max(sum(seg), 1e-9)
    Wr_t = {hips: qmul(D, np.broadcast_to(sk["Wr0"][hips], (T, 4)))}
    for j, cf in zip(chain[1:], cum[1:]):
        Wr_t[j] = qmul(R.qslerp(ident, bend, cf), qmul(D, np.broadcast_to(sk["Wr0"][j], (T, 4))))
    L = np.tile(sk["rot"][None], (T, 1, 1))
    pos = np.tile(sk["loc"][None], (T, 1, 1))
    root = [j for j in range(J) if par[j] < 0][0]
    order, seen = [], set()

    def visit(j):
        if j in seen:
            return
        seen.add(j)
        if par[j] >= 0:
            visit(par[j])
        order.append(j)

    for j in range(J):
        visit(j)

    def local_from_world(Wr_t, L):
        Wc = {}
        for j in order:
            p = par[j]
            if j in Wr_t:
                L[:, j] = qmul(R.qconj(Wc[p]), Wr_t[j]) if p >= 0 else Wr_t[j]
                Wc[j] = Wr_t[j]
            else:
                Wc[j] = qmul(Wc[p], L[:, j]) if p >= 0 else L[:, j]
        return Wc

    local_from_world(Wr_t, L)
    pos[:, root] = hips_w if root == hips else 0.0
    Wp, Wr = R.fk_pr(par, pos, L)
    # ---- limbs: two-bone IK to the decoded foot/hand, bend towards the decoded knee/elbow
    info = {}
    names = ("leg_L", "leg_R", "arm_L", "arm_R")
    for li, key in enumerate(names):
        ch = roles[key]
        a, b, cc = ch[0], ch[1], ch[2]
        if not flags[li]:
            continue
        l1 = float(np.linalg.norm(sk["Wp0"][b] - sk["Wp0"][a]))
        l2 = float(np.linalg.norm(sk["Wp0"][cc] - sk["Wp0"][b]))
        Mloc, Eloc = limbs[:, 6 * li:6 * li + 3], limbs[:, 6 * li + 3:6 * li + 6]
        Mw = hips_w + qrot(Yq, Mloc * H)
        Ew = hips_w + qrot(Yq, Eloc * H)
        R0 = Wp[:, a]
        if min_reach or chord_smooth or pole_smooth:
            from scipy.ndimage import gaussian_filter1d
            Ltot = l1 + l2
            chord = Ew - R0
            dist = np.linalg.norm(chord, axis=1)
            cs = gaussian_filter1d(chord, chord_smooth, axis=0, mode="nearest") if chord_smooth else chord
            us = cs / np.maximum(np.linalg.norm(cs, axis=1, keepdims=True), 1e-9)
            Ew = R0 + us * (np.maximum(dist, min_reach * Ltot) if min_reach else dist)[:, None]
            if pole_smooth:
                rel = Mw - R0
                par_ = (rel * us).sum(1, keepdims=True) * us
                pv = gaussian_filter1d(rel - par_, pole_smooth, axis=0, mode="nearest")
                Mw = R0 + par_ + pv
        k = 1.0
        if leg_normalise and key.startswith("leg"):
            reach = np.percentile(np.linalg.norm(Ew - R0, axis=1), 98)
            k = float(np.clip((l1 + l2) * 0.995 / max(reach, 0.6 * (l1 + l2)), 0.8, 1.25))
            Ew = R0 + (Ew - R0) * k
            Mw = R0 + (Mw - R0) * k
        info[key + "_scale"] = round(k, 3)
        bl0 = qrot(R.qconj(sk["Wr0"][a]), R.unit(sk["Wp0"][b] - sk["Wp0"][a]))
        bl1 = qrot(R.qconj(sk["Wr0"][b]), R.unit(sk["Wp0"][cc] - sk["Wp0"][b]))
        Wa_all, Wb_all, fails = np.zeros((T, 4)), np.zeros((T, 4)), 0
        for t in range(T):
            Mt, Er, ok = R.two_bone(R0[t], Mw[t], Ew[t], l1, l2, None)
            fails += (not ok)
            d1, d2 = R.unit(Mt - R0[t]), R.unit(Er - Mt)
            Pa = Wr[t, par[a]]
            Qa = qmul(Pa, sk["rot"][a])
            Wa = qmul(R.q_swing(qrot(Qa, bl0), d1), Qa)
            Qb = qmul(Wa, sk["rot"][b])
            Wb = qmul(R.q_swing(qrot(Qb, bl1), d2), Qb)
            Wa_all[t], Wb_all[t] = Wa, Wb
        info[key + "_unreachable_frames"] = int(fails)
        Wr_t[a], Wr_t[b] = Wa_all, Wb_all
    L = np.tile(sk["rot"][None], (T, 1, 1))
    local_from_world(Wr_t, L)
    Wp, Wr = R.fk_pr(par, pos, L)
    return dict(L=L, pos=pos, root=pos[:, root], Wp=Wp, Wr=Wr, fps=FPS, info=info, frames=T)


def write_proxy(path, proxy, ref, name):
    sk, roles = ref
    root = [j for j in range(len(sk["names"])) if sk["parents"][j] < 0][0]
    write_clip(Path(path), sk["names"], sk["parents"], sk["loc"], sk["rot_xyzw"], sk["scale"], FPS, proxy["L"][..., [1, 2, 3, 0]],
               {root: proxy["root"]}, name, dict(fps=FPS, units="m", decoded_proxy=True))


def to_character(proxy_path, target_key, label, caption, ref=None, extra=None):
    """Stage 2: retarget the proxy clip onto a character with retarget.py and register it for the viewers. Returns the metrics."""
    import argparse as ap
    sk_src, roles_src = ref or ref_skeleton()
    src = R.load_clip(str(proxy_path))
    sroles = R.label_roles(src, R.clean_names("mixamo", "mixamo"))
    ds, rid = R.rig_of(target_key)
    tdir = R.CHARS / target_key
    tgt = R.load_skeleton(tdir / "model.glb")
    troles = R.label_roles(tgt, R.clean_names(ds, rid))
    params = dict(stride=1.0, vgain=1.0, forelimbs="auto", ik=True, shape=True, tilt_limit=35.0, stance_tilt=12.0, spine_gain=0.6)
    L, root_pos, info, Wp = R.retarget(src, sroles, tgt, troles, params, src["fps"])
    info["target_class"] = R.body_class(tgt, troles)
    name = f"rt__dec_{label}"
    adir = tdir / "anims"
    adir.mkdir(exist_ok=True)
    root = [j for j in range(len(tgt["names"])) if tgt["parents"][j] < 0][0]
    write_clip(adir / f"{name}.glb", tgt["names"], tgt["parents"], tgt["loc"], tgt["rot_xyzw"], tgt["scale"], src["fps"], L[..., [1, 2, 3, 0]],
               {root: root_pos}, name, dict(source_clip=f"decoded:{label}", fps=src["fps"], units="m", retargeted=True))
    f = adir / "anims.json"
    meta = json.loads(f.read_text()) if f.exists() else {}
    meta[name] = dict(file=f"{name}.glb", source_clip=f"decoded:{label}", source_file=str(Path(proxy_path).relative_to(ROOT)), fps=src["fps"],
                      frames=int(len(L)), duration_s=round(len(L) / src["fps"], 3), loop=False, loop_gap_deg=0.0,
                      root_travel_m=round(info["root_travel_m"], 3), root_net_m=0.0, trimmed_frames=0, start_offset_m=0.0,
                      motion_captions=[caption] if caption else [], decoded=extra or {},
                      retargeted={k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    f.write_text(json.dumps(meta, indent=1))
    return info


# ----------------------------------------------------------------------------- self-test and examples
def selftest():
    """Real clips -> role features -> IK proxy -> role features again. The error is what stage 1 itself costs, before any decoder."""
    import motion_features as MF
    ref = ref_skeleton()
    sk0, roles = ref
    rep, inv = R.clip_index()
    groups = dict(root_velocity=[0, 1], yaw_rate=[2], hips_height=[3], hips_tilt=range(4, 10), head=range(10, 13), limb_joints=range(13, 37))
    err = {g: [] for g in groups}
    base = {g: [] for g in groups}
    allf = np.load(ROOT / "data" / "embeddings" / "motion_feats.npz")["feats"]
    mean = allf.reshape(-1, 41).mean(0)
    names = ["Backflip", "Waving_Gesture", "Medium_Run", "Rifle_Punch", "Standard_Walk", "Jumping_13", "Sitting_Thumbs_Up", "Goalkeeper_Drop_Kick"]
    used = []
    for nm in names:
        if nm not in rep or nm not in inv:
            continue
        sk = R.load_clip(str(R.ROOT / rep[nm]))
        rl = R.label_roles(sk, R.clean_names("mixamo", "mixamo"))
        f, nv, dur = MF.clip_features(sk, rl)
        px = proxy_from_features(f, nv, dur, ref)
        s2 = dict(sk0)
        s2.update(Wp=px["Wp"], Wr=px["Wr"], fps=FPS)
        f2, nv2, _ = MF.clip_features(s2, roles)
        # compare on the common timeline: resample the original to the proxy's length
        T2 = min(nv2, nv)
        a, b = f[:nv][np.linspace(0, nv - 1, T2).round().astype(int)], f2[:nv2][np.linspace(0, nv2 - 1, T2).round().astype(int)]
        for g, ix in groups.items():
            ix = list(ix)
            err[g].append(np.abs(a[:, ix] - b[:, ix]).mean())
            base[g].append(np.abs(a[:, ix] - mean[ix]).mean())
        used.append(nm)
    print(f"clips: {used}")
    print("stage-1 fidelity (mean absolute error in hips heights / radians; the same clip's features vs the IK result; mean-baseline = predicting the training mean)")
    for g in groups:
        print(f"  {g:14s} {np.mean(err[g]):.4f}   (baseline {np.mean(base[g]):.4f})")


def examples():
    """Decode the example latents of decode_latents.py and put them on characters."""
    import motion_ae as MA
    import torch
    import decode_latents as DL
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, mean, std, torch = MA.load_model(dev)
    D = MA.load_data()
    meta = D["meta"]
    va = [i for i, m in enumerate(meta) if m["split"] == "val"]
    caption = lambda i: [t for s, r, t in D["caps"][(meta[i]["dataset"], meta[i]["clip"])] if s == "caption"][0]
    import re
    picks = []
    for kw in ("waves", "punch", "flip", "kicks", "sits", "walks"):
        for i in va:
            if re.search(kw, caption(i)) and 30 <= D["n_valid"][i] <= 120 and meta[i]["body"] == "biped":
                picks.append(i)
                break
    Z = MA.encode_all(model, D, np.array(picks), mean, std, dev, torch)[0]
    tr = np.array([i for i, m in enumerate(meta) if m["split"] == "train"])
    Ztr = MA.encode_all(model, D, tr, mean, std, dev, torch)[0]
    rng = np.random.default_rng(1)
    Zr = rng.multivariate_normal(Ztr.mean(0), np.cov(Ztr.T) + 1e-6 * np.eye(Ztr.shape[1]), 3)
    items = [(f"{meta[i]['clip'].lower()[:24]}", caption(i), z) for i, z in zip(picks, Z)] + [(f"random{k + 1}", "a random latent", z) for k, z in enumerate(Zr)]
    ref = ref_skeleton()
    targets = ["mixamo_Aciana", "objaverse_a734df8176ca47c68b0a90d1a17b5132"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, cap, z in items:
        with torch.no_grad():
            zt = torch.from_numpy(np.asarray(z, np.float32))[None].to(dev)
            x = model.decode(zt).cpu().numpy()[0] * std + mean
            aux = model.aux(zt).cpu().numpy()[0]
        x[:, 37:41] = (x[:, 37:41] > 0.5).astype(np.float32)
        nv, dur = int(np.clip(round(aux[1] * MA.T_FIX), 8, MA.T_FIX)), float(np.exp(aux[0]))
        px = proxy_from_features(x, nv, dur, ref)
        path = OUT_DIR / f"{label}.glb"
        write_proxy(path, px, ref, label)
        line = f"{label:26s} {dur:4.1f}s"
        for tk in targets:
            try:
                info = to_character(path, tk, label, f"decoded latent: {cap}", ref, extra=dict(kind="decoded latent", frames=nv))
                line += f" | {tk[:14]:14s} skate {info['skate_mean']:.3f} fails {info['reach_fail_frames']:3d} pen {info['penetration_max']:.3f} lift {info['lift_max']:.2f}"
            except SystemExit as e:
                line += f" | {tk[:14]} skipped: {e}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--examples", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.examples:
        examples()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
