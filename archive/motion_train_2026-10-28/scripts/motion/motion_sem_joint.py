#!/usr/bin/env python3
"""Pilot: one semantic motion model for 3D clips AND Lottie animations, and a test of whether meaning transfers across the two.

  python3 scripts/motion/motion_sem_joint.py train  [--lottie 11000] [--epochs 40]     # -> data/embeddings/motion_sem_joint_pilot.pt, log in data/logs/
  python3 scripts/motion/motion_sem_joint.py eval                                      # cross-domain test of the trained model and of the 3D-only model

Same network, loss and 3D data as scripts/motion/motion_sem.py (imported, not changed). Added: a random subset of data/embeddings/lottie_feats.npz (scripts/lottie2d/lottie_tokens.py)
with the dataset's own captions encoded by CLIP; each domain's caption embeddings are centred by their own mean (the 3D ones by the fixed text_mean, as before).
Batches are single-domain (a 3D batch is sampled by body group as before, a Lottie batch uniformly; --p-lottie picks the domain per step), so a caption of one domain is
never a false negative for a clip of the other, and nothing in the loss rewards separating the domains.

Cross-domain test (there are no paired 3D/2D clips): for an action word (walk, wave, point, nod, ...) found in the captions of both domains, take the held-out clips of
one domain whose caption has the word, retrieve the 5 nearest clips of the OTHER domain by motion embedding, and count how many of those have the word in their caption.
Reported as precision@5 against the base rate of the word in the gallery (lift 1 = chance). The 3D-only model (motion_sem.pt) run on the Lottie tokens is the reference.
"""
from __future__ import annotations

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
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
import motion_sem as MS  # noqa: E402

ACTIONS = {
    "walk": r"\bwalk\w*\b", "run": r"\b(run|runs|running|jog\w*|sprint\w*)\b", "jump": r"\b(jump\w*|leap\w*|hop\w*)\b", "wave": r"\bwav(e|es|ing)\b",
    "point": r"\bpoint(s|ing)?\b", "nod": r"\bnod(s|ding)?\b", "clap": r"\bclap\w*\b", "dance": r"\bdanc\w*\b", "type": r"\btyp(e|es|ing)\b", "kick": r"\bkick\w*\b",
    "bend": r"\bbend\w*\b", "lift": r"\b(lift\w*|rais\w*)\b", "turn": r"\b(turn\w*|spin\w*)\b", "shake": r"\bshak\w*\b", "sit": r"\bsit(s|ting)?\b",
    "throw": r"\bthrow\w*\b", "swing": r"\bswing\w*\b", "bounce": r"\bbounc\w*\b", "stretch": r"\bstretch\w*\b", "look": r"\blook\w*\b",
}
ACT_RE = {k: re.compile(v, re.I) for k, v in ACTIONS.items()}


def lottie_split(clip: str) -> str:
    return "val" if int(hashlib.md5(("l:" + clip).encode()).hexdigest(), 16) % 100 < 9 else "train"


def load_joint(n_lottie: int, seed: int = 0, captions: str = "full"):
    """3D data (motion_sem.load_data) + a random subset of the Lottie tokens in the same layout; returns (D, n3, lottie_idx)."""
    D = MS.load_data()
    n3 = len(D["meta"])
    z = np.load(EMB / "lottie_feats.npz")
    rows = list(csv.DictReader(open(EMB / "lottie_feats_index.csv", newline="")))
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(rows), min(n_lottie, len(rows)), replace=False))
    off = z["offsets"]
    sl = [slice(int(off[i]), int(off[i + 1])) for i in sel]
    joint_all, static_all = z["joint"], z["static"]                        # an NpzFile reads the array again on every access: take each once
    Lj = np.concatenate([joint_all[s] for s in sl])
    Ls = np.concatenate([static_all[s] for s in sl])
    del joint_all, static_all
    new_off = np.cumsum([0] + [s.stop - s.start for s in sl])
    from embed import encode, load_text_encoder
    caps = [rows[i]["caption"] or "" for i in sel]
    acts = [""] * len(sel)                                                 # the action caption of each clip ("" = none found)
    if captions in ("actions", "mixed"):                                   # the motion phrases (scripts/lottie2d/lottie_actions.py)
        act = {r["clip"]: r["action_caption"] for r in csv.DictReader(open(EMB / "lottie_actions.csv", newline=""))}
        acts = [act.get(rows[i]["clip"]) or "" for i in sel]
        print(f"action captions found for {sum(bool(a) for a in acts)} of {len(sel)} clips, mean {np.mean([len(a.split()) for a in acts if a]):.1f} words (full captions {np.mean([len(c.split()) for c in caps]):.1f})", flush=True)
    full = list(caps)
    if captions == "actions":                                              # action captions only; a clip without one keeps its full caption
        caps = [a or c for a, c in zip(acts, caps)]
    t0 = time.time()
    texts = caps + ([a for a in acts if a] if captions == "mixed" else [])
    emb, cut = encode(texts, load_text_encoder())
    print(f"encoded {len(texts)} Lottie texts with CLIP in {time.time() - t0:.0f}s ({cut} longer than 77 tokens, truncated)", flush=True)
    tr = np.array([lottie_split(rows[i]["clip"]) == "train" for i in sel])
    tr_all = np.concatenate([tr] + ([tr[[bool(a) for a in acts]]] if captions == "mixed" else []))
    mu_l = emb[tr_all].mean(0)                                             # one Lottie text mean (full and action texts together), from the training clips only
    emb = emb - mu_l
    emb = (emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    act_row = {}
    if captions == "mixed":                                                # row of each clip's action text in the embedding table, after the N full captions
        have = [k for k, a in enumerate(acts) if a]
        act_row = {k: len(sel) + n for n, k in enumerate(have)}
    meta, capmap = list(D["meta"]), dict(D["caps"])
    for k, i in enumerate(sel):
        r = rows[i]
        meta.append(dict(dataset="lottie", clip=r["clip"], object_type=r["object_type"], character_class="lottie", body="lottie", split=lottie_split(r["clip"]),
                         caption=r["caption"], duration_s=r["duration_s"]))                      # meta caption = the full original, it is the evaluation truth
        entries = [("caption", len(D["temb"]) + k, caps[k])]                                     # "caption" first: text_of() and the in-domain score use it
        if k in act_row:
            entries.append(("action", len(D["temb"]) + act_row[k], acts[k]))                    # training samples one of the entries per step
        capmap[("lottie", r["clip"])] = entries
    D2 = dict(joint=np.concatenate([D["joint"], Lj]), static=np.concatenate([D["static"], Ls]), offsets=np.concatenate([D["offsets"], D["offsets"][-1] + new_off[1:]]),
              glob=np.concatenate([D["glob"], z["glob"][sel]]), scal=np.concatenate([D["scal"], z["scal"][sel]]), meta=meta, caps=capmap,
              temb=np.concatenate([D["temb"], emb]), mu=D["mu"], mu_l=mu_l)
    return D2, n3, np.arange(n3, n3 + len(sel))


def caption_of(D, i) -> str:
    m = D["meta"][i]
    if m["dataset"] == "lottie":
        return m["caption"]                                                # always the full original caption, whatever text the model was trained on
    return " ".join(t for s, r, t in D["caps"][(m["dataset"], m["clip"])] if s in ("caption", "detail", "generic"))


def cmd_train(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    D, n3, L = load_joint(args.lottie, seed=args.seed, captions=args.captions)
    tr3 = [i for i in range(n3) if D["meta"][i]["split"] == "train"]
    trL = np.array([i for i in L if D["meta"][i]["split"] == "train"])
    va3 = np.array([i for i in range(n3) if D["meta"][i]["split"] == "val"])
    vaL = np.array([i for i in L if D["meta"][i]["split"] == "val"])
    groups = {g: np.array([i for i in tr3 if D["meta"][i]["body"] == g]) for g in ("biped", "quadruped", "other")}
    print(f"3D: train {len(tr3)} {{{', '.join(f'{g} {len(v)}' for g, v in groups.items())}}}, val {len(va3)} | Lottie: train {len(trL)}, val {len(vaL)} | device {dev}", flush=True)
    model = MS.build_model()().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    per_epoch = args.steps or math.ceil((len(tr3) + len(trL)) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * per_epoch, pct_start=0.05)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ls = {"3d": [], "lottie": []}
        for _ in range(per_epoch):
            lot = rng.random() < args.p_lottie
            if lot:
                idx = rng.choice(trL, args.batch)
            else:
                parts = rng.choice(3, args.batch)                            # bodies sampled evenly, then a clip of that body
                idx = np.array([rng.choice(groups[("biped", "quadruped", "other")[p]]) for p in parts])
            x = MS.batch_tensors(D, idx, torch, dev)
            T = torch.from_numpy(D["temb"][[MS.pick_text(D, i, rng) for i in idx]]).to(dev)
            loss = MS.info_nce(model(*x), T, (T @ T.T) > args.neg_thr, args.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ls["lottie" if lot else "3d"].append(float(loss.detach()))
        if ep % 5 == 0 or ep == args.epochs - 1:
            e3 = MS.evaluate(model, D, va3, dev, torch)["all"]["t2m"]
            eL = in_domain(model, D, vaL, dev, torch)
            print(f"epoch {ep:3d} loss 3D {np.mean(ls['3d']):.3f} Lottie {np.mean(ls['lottie']):.3f} | held-out text->motion R@5: 3D {e3['R@5']:.1f}%  Lottie {eL['R@5']:.1f}% (median rank {eL['medR']:.0f} of {eL['n']}) [{time.time() - t0:.0f}s]", flush=True)
    path = EMB / f"motion_sem_joint_{args.tag}.pt"
    torch.save(dict(model=model.state_dict(), args=vars(args)), path)
    print("saved", path, flush=True)
    cross_domain(model, MS.load_model(EMB / "motion_sem.pt", dev)[0], D, n3, L, dev, torch, args)


def in_domain(model, D, idx, dev, torch):
    emb = MS.encode_all(model, D, idx, dev, torch)
    T = np.stack([D["temb"][MS.text_of(D, i)] for i in idx])
    return MS.recall(MS.ranks(T @ emb.T))


def stems_of(text: str) -> set:
    return {k for k, rx in ACT_RE.items() if rx.search(text)}


def cross_pairs(emb_q, stems_q, emb_g, stems_g, min_q, k=5):
    """Per action word: precision@k of the other domain's nearest clips, and its base rate in the gallery."""
    out = {}
    sims = emb_q @ emb_g.T
    top = np.argsort(-sims, axis=1)[:, :k]
    for a in ACTIONS:
        q = np.array([a in s for s in stems_q])
        g = np.array([a in s for s in stems_g])
        if q.sum() < min_q or g.sum() < min_q:
            continue
        prec = float(g[top[q]].mean())
        base = float(g.mean())
        out[a] = dict(n_query=int(q.sum()), n_gallery=int(g.sum()), precision=prec, base_rate=base, lift=prec / max(base, 1e-9))
    return out


def cross_domain(model, base_model, D, n3, L, dev, torch, args):
    va3 = np.array([i for i in range(n3) if D["meta"][i]["split"] == "val"])
    vaL = np.array([i for i in L if D["meta"][i]["split"] == "val"])
    all3 = np.arange(n3)
    res = {}
    texts = {i: caption_of(D, i) for i in list(all3) + list(L)}
    st = {i: stems_of(texts[i]) for i in texts}
    print("\naction words found in captions (3D / Lottie):", {a: (sum(a in st[i] for i in all3), sum(a in st[i] for i in L)) for a in ACTIONS if sum(a in st[i] for i in L) >= args.min_query}, flush=True)
    for name, m in (("joint model (3D + Lottie)", model), ("3D-only model on Lottie tokens", base_model)):
        e3 = MS.encode_all(m, D, all3, dev, torch)
        eL = MS.encode_all(m, D, L, dev, torch)
        pos3 = {i: k for k, i in enumerate(all3)}
        posL = {i: k for k, i in enumerate(L)}
        a = cross_pairs(eL[[posL[i] for i in vaL]], [st[i] for i in vaL], e3, [st[i] for i in all3], args.min_query)       # Lottie query -> 3D gallery
        b = cross_pairs(e3[[pos3[i] for i in va3]], [st[i] for i in va3], eL, [st[i] for i in L], args.min_query)           # 3D query -> Lottie gallery
        # how separable are the two domains in the embedding?
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        n = min(len(vaL), len(va3))
        X = np.r_[eL[[posL[i] for i in vaL[:n]]], e3[[pos3[i] for i in va3[:n]]]]
        auc = float(cross_val_score(LogisticRegression(max_iter=2000), X, np.r_[np.ones(n), np.zeros(n)], cv=5, scoring="roc_auc").mean())
        res[name] = {"lottie_to_3d": a, "3d_to_lottie": b, "domain_auc": auc}
        print(f"\n== {name}   (domain classifier on the clip embeddings: AUC {auc:.3f}; 0.5 = the embedding does not show the domain)")
        for tag, r in (("Lottie query -> 3D clips", a), ("3D query -> Lottie clips", b)):
            if not r:
                print(f"  {tag}: no action word with >= {args.min_query} held-out queries")
                continue
            print(f"  {tag}: macro lift {np.mean([v['lift'] for v in r.values()]):.2f} over {len(r)} words (1.0 = chance)")
            for w_, v in sorted(r.items(), key=lambda t: -t[1]["lift"]):
                print(f"     {w_:8s} n_query {v['n_query']:4d}  precision@5 {100 * v['precision']:5.1f}%  base {100 * v['base_rate']:5.1f}%  lift {v['lift']:5.2f}")
    out = ROOT / "data" / "analysis" / f"cross_domain_{args.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("\n->", out)


def cmd_eval(args):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    D, n3, L = load_joint(args.lottie, seed=args.seed, captions=args.captions)
    model = MS.load_model(EMB / f"motion_sem_joint_{args.tag}.pt", dev)[0]
    cross_domain(model, MS.load_model(EMB / "motion_sem.pt", dev)[0], D, n3, L, dev, torch, args)


def cmd_summary(args):
    """One table of finished runs: cross-domain precision@5, lifts, domain AUC (data/analysis/cross_domain_<tag>.json) and the final in-domain scores (data/logs/motion_sem_joint_<tag>.log)."""
    tags = args.tags or ["pilot", "actions", "mixed"]
    print(f"{'tag':12s} | {'Lottie->3D prec@5':>17s} {'med lift':>8s} {'>=2':>5s} | {'3D->Lottie prec@5':>17s} {'med lift':>8s} {'>=2':>6s} | {'AUC':>5s} | {'3D R@5':>6s} {'Lottie R@5':>10s}")
    rows = {}
    for t in tags:
        f = ROOT / "data" / "analysis" / f"cross_domain_{t}.json"
        if not f.exists():
            print(f"{t:12s} | (not finished)")
            continue
        j = json.loads(f.read_text())["joint model (3D + Lottie)"]
        out = []
        for d in ("lottie_to_3d", "3d_to_lottie"):
            v = list(j[d].values())
            n = np.array([x["n_query"] for x in v], float)
            p = float((np.array([x["precision"] for x in v]) * n).sum() / n.sum())
            lift = np.array([x["lift"] for x in v])
            out += [100 * p, float(np.median(lift)), f"{int((lift >= 2).sum())}/{len(v)}"]
        r3 = rl = float("nan")
        lg = ROOT / "data" / "logs" / f"motion_sem_joint_{t}.log"
        if lg.exists():
            m = [ln for ln in lg.read_text().splitlines() if ln.startswith("epoch")]
            if m:
                mm = re.search(r"3D ([\d.]+)%\s+Lottie ([\d.]+)%", m[-1])
                r3, rl = float(mm.group(1)), float(mm.group(2))
        rows[t] = out[:2] + out[3:5] + [j["domain_auc"], r3, rl]
        print(f"{t:12s} | {out[0]:16.1f}% {out[1]:8.2f} {out[2]:>5s} | {out[3]:16.1f}% {out[4]:8.2f} {out[5]:>6s} | {j['domain_auc']:.3f} | {r3:5.1f}% {rl:9.1f}%")
    if args.group:
        print("\nmean +- std over the seeds of each group (Lottie->3D prec@5, its median lift; 3D->Lottie prec@5, its median lift; domain AUC; 3D R@5; Lottie R@5)")
        for g in args.group:
            name, ts = g.split("=")
            a = np.array([rows[t] for t in ts.split(",") if t in rows])
            if len(a) > 1:
                print(f"{name:8s} n={len(a)} | " + "  ".join(f"{m:6.2f}+-{sd:.2f}" for m, sd in zip(a.mean(0), a.std(0, ddof=1))))
            elif len(a) == 1:
                print(f"{name:8s} n=1  | (needs at least 2 finished runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval", "summary"])
    ap.add_argument("--lottie", type=int, default=11000, help="Lottie clips sampled from lottie_feats.npz (about 9% of them are held out)")
    ap.add_argument("--captions", choices=["full", "actions", "mixed"], default="full", help="text the Lottie clips are trained on: the full caption, its motion phrases, or one of the two at random per step")
    ap.add_argument("--tag", default="pilot", help="names the model and result files (motion_sem_joint_<tag>.pt, cross_domain_<tag>.json)")
    ap.add_argument("--seed", type=int, default=0, help="training randomness AND which Lottie clips are sampled (so the held-out Lottie clips change too)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0, help="steps per epoch (0 = (3D + Lottie train clips) / batch)")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--neg-thr", type=float, default=0.6)
    ap.add_argument("--p-lottie", type=float, default=0.5, help="share of steps whose batch is Lottie")
    ap.add_argument("--tags", nargs="*", help="summary: run tags (default pilot actions mixed)")
    ap.add_argument("--group", nargs="*", help="summary: groups to average, written name=tag,tag,tag (e.g. full=pilot,full_s1,full_s2)")
    ap.add_argument("--min-query", type=int, default=8, help="smallest number of held-out queries (and gallery clips) for an action word to be reported")
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
