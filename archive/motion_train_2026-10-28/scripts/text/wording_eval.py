#!/usr/bin/env python3
"""How well does a diffusion checkpoint follow differently worded prompts?  (the test for scripts/text/paraphrase.py)

  python3 scripts/text/wording_eval.py [--ckpt z_diffusion.pt] [--guidance 2.0]

Three measures, all on held-out material (the 555 validation clips never seen by the diffusion model; paraphrases made by scripts/text/paraphrase.py --split val):
  1. text-only: the semantic head's direct retrieval (text -> real validation motions), canonical caption vs paraphrase of the SAME clips. If paraphrases cost nothing here, CLIP is already
     absorbing the wording and the generator's problem is elsewhere.
  2. generated: prompt -> diffusion sample -> semantic head -> rank of the clip the prompt describes among the 555 real validation motions (R@1/5/10, median rank), canonical vs paraphrase sets
     (val_casual: different LLM instruction than the training paraphrases; val_std: same instruction as the training paraphrases, so optimistic).
  3. hand-written informal prompts (HAND below, written by hand, not by an LLM): precision@5 of the nearest real clips (by semantic head, over all 6,055 clips) whose caption matches the
     action word, mean of 4 seeds; and for the flip prompts the share of samples whose body is upside down (min up-axis of the hips tilt < 0).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]

# (prompt, regex the caption of a matching real clip must contain)
HAND = [("A cat do it backflip", "flip|somersault"), ("person does a back flip", "flip|somersault"), ("do a backflip", "flip|somersault"), ("someone flips backwards", "flip|somersault"),
        ("backflip", "flip|somersault"), ("a man doing a somersault in the air", "flip|somersault"),
        ("dog running fast", r"\bruns?\b|running|sprint|jog"), ("somebody is running", r"\bruns?\b|running|sprint|jog"), ("sprint forward", r"\bruns?\b|running|sprint|jog"),
        ("a person walking slowly", r"walk"), ("strolling forward", r"walk|stroll"), ("walk", r"walk"),
        ("jump high", r"jump|leap|hop"), ("hop up in the air", r"jump|leap|hop"), ("he jumps over something", r"jump|leap|hop"),
        ("kicking", r"kick"), ("he kicks the ball", r"kick"), ("do a kick", r"kick"),
        ("wave hello", r"wave|waving"), ("waving hand", r"wave|waving"),
        ("sit down on the floor", r"\bsits?\b|sitting|seated"), ("take a seat", r"\bsits?\b|sitting|seated"),
        ("dancing", r"danc"), ("do a dance", r"danc"), ("a guy dancing", r"danc"),
        ("throw a punch", r"punch|box"), ("boxing", r"punch|box"), ("punch", r"punch|box"),
        ("crawl on the ground", r"crawl"), ("climb a ladder", r"climb"), ("squat down", r"squat|crouch"),
        ("turn around", r"turn"), ("spin around", r"spin|turn|twirl"), ("fall down", r"fall|collaps"), ("lie down", r"lie|lay|lying"),
        ("stretch arms", r"stretch|arms"), ("stand still", r"stand|still|idle"), ("clap hands", r"clap"), ("bow", r"\bbow"), ("throw a ball", r"throw|toss|pitch"),
        ("swimming", r"swim")]
FLIP = [i for i, (p, rx) in enumerate(HAND) if rx == "flip|somersault"]


def ranks(sim):
    return (sim > np.diag(sim)[:, None]).sum(1)


def recall(r):
    return f"R@1 {100 * (r < 1).mean():4.1f}%  R@5 {100 * (r < 5).mean():4.1f}%  R@10 {100 * (r < 10).mean():4.1f}%  median {np.median(r):4.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="z_diffusion.pt")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for parts 1+2 (the noise of the diffusion sampler)")
    ap.add_argument("--min-cos", type=float, default=0.5)
    ap.add_argument("--frontend", choices=["rules", "hybrid", "snap"], help="rewrite with the language-model-free front end of scripts/text/text_frontend.py (split off the shape words, then rules / nearest caption) instead of the LLM; implies --rewrite")
    ap.add_argument("--rewrite", action="store_true", help="rewrite every prompt into dataset form first (scripts/text/rewrite_prompt.py), canonical captions too")
    a = ap.parse_args()
    a.rewrite = a.rewrite or bool(a.frontend)
    import torch
    import diffusion_test as DT
    import motion_ae as MA
    from embed import encode, load_text_encoder
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    DT.EMB = MA.EMB = EMB
    DT.CKPT = EMB / a.ckpt
    P = DT.load_pairs()
    model, zm, zs, _ = DT.load_model(dev)
    ae, mean, std, _ = MA.load_model(dev)
    va = np.where(P["split"] == "val")[0]
    sem_va = P["sem"][va]
    caps = [__import__("json").loads(c).get("caption", "") for c in np.load(EMB / "motion.npz")["captions"]]
    cap_clips = {}
    for j, i in enumerate(va):
        cap_clips.setdefault(caps[i], []).append(j)
    enc = load_text_encoder()
    rw = None
    if a.rewrite:
        from rewrite_prompt import Rewriter
        rw = Rewriter(encoder=enc)
        if a.frontend:
            import text_frontend as TF
            rw = TF.Adapter(rw, a.frontend)
    print(f"checkpoint {a.ckpt}, guidance {a.guidance}{', prompts rewritten into dataset form' if rw else ''}; {len(va)} held-out clips\n")

    def to_sem(zn):
        with torch.no_grad():
            return torch.nn.functional.normalize(ae.sem(torch.from_numpy((zn * zs + zm).astype(np.float32)).to(dev)), dim=-1).cpu().numpy()

    # paraphrase sets: rows (embedding, clip index within va)
    D = np.load(EMB / "text_para.npz")
    sets = {}
    for name in ("val_casual", "val_std"):
        m = D["split"] == name
        emb, target, txt = [], [], []
        for e, c, t in zip(D["emb"][m], D["caption"][m], D["text"][m]):
            for j in cap_clips.get(str(c), []):
                emb.append(e); target.append(j); txt.append(str(t))
        if emb:
            sets[name] = (np.stack(emb), np.array(target), txt)
    print("1+2. held-out clips: prompt -> real validation motion (text-only retrieval, and generated sample -> real motion)")
    for name, (emb, target, txt) in sets.items():
        canon = P["text_clip"][va][target]
        if rw:
            canon = rw.embed(rw.rewrite([caps[va[j]] for j in target]))
            emb = rw.embed(rw.rewrite(txt))
        for label, T in (("canonical caption", canon), (f"paraphrase ({name})", emb)):
            sim = T @ sem_va.T
            r_text = (sim > sim[np.arange(len(T)), target][:, None]).sum(1)
            zn = DT.sample(model, torch, dev, T, guidance=a.guidance, seed=a.seed)
            sg = to_sem(zn) @ sem_va.T
            r_gen = (sg > sg[np.arange(len(T)), target][:, None]).sum(1)
            print(f"  {label:34s} n={len(T):4d} | text-only retrieval: {recall(r_text)} | generated: {recall(r_gen)}")
    # 3. hand-written prompts
    mu = np.load(EMB / "text_mean.npy")
    if rw:
        E = rw.embed(rw.rewrite([p for p, _ in HAND]))
        print("\nrewritten hand prompts:", "; ".join(f"{p} -> {r}" for (p, _), r in zip(HAND[:8], rw.rewrite([p for p, _ in HAND[:8]]))))
    else:
        E = encode([p for p, _ in HAND], enc)[0] - mu
        E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    Sall = P["sem"]
    allcaps = caps
    prec = np.zeros((len(HAND), a.seeds))
    inv = np.zeros((len(FLIP), a.seeds))
    for s in range(a.seeds):
        zn = DT.sample(model, torch, dev, E, guidance=a.guidance, seed=s)
        sg = to_sem(zn)
        top = np.argsort(-(sg @ Sall.T), axis=1)[:, :5]
        for k, (p, rx) in enumerate(HAND):
            prec[k, s] = np.mean([bool(re.search(rx, allcaps[i], re.I)) for i in top[k]])
        with torch.no_grad():
            X = ae.decode(torch.from_numpy((zn[FLIP] * zs + zm).astype(np.float32)).to(dev)).cpu().numpy() * std + mean
            nv = np.clip(np.round(ae.aux(torch.from_numpy((zn[FLIP] * zs + zm).astype(np.float32)).to(dev)).cpu().numpy()[:, 1] * 128), 8, 128).astype(int)
        inv[:, s] = [X[j, :nv[j], 8].min() < 0 for j in range(len(FLIP))]
    print(f"\n3. {len(HAND)} hand-written informal prompts, {a.seeds} seeds: precision@5 of the nearest real clips whose caption contains the action word")
    for k, (p, rx) in enumerate(HAND):
        print(f"  {prec[k].mean():4.2f}  {p}")
    print(f"  MEAN precision@5 {prec.mean():.3f}   (prompts with precision >= 0.6: {int((prec.mean(1) >= 0.6).sum())} of {len(HAND)})")
    print(f"  flip prompts: share of samples with the body upside down: {inv.mean() * 100:.0f}%  ({', '.join(f'{HAND[i][0]}: {inv[j].mean() * 100:.0f}%' for j, i in enumerate(FLIP))})")


if __name__ == "__main__":
    main()
