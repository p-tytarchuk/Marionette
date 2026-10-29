#!/usr/bin/env python3
"""Text -> nearest Lottie animations (and nearest 3D clips) with a trained joint model (scripts/motion/motion_sem_joint.py). Retrieval, not generation.

  python3 scripts/lottie2d/lottie_search.py build --tag mixed --captions mixed      # embed every tokenised Lottie clip and every 3D clip -> data/embeddings/lottie_search_mixed.npz
  python3 scripts/lottie2d/lottie_search.py query "a person waves hello" --tag mixed

The joint model maps a clip's joint tokens into the CLIP text space of its own domain (each domain's caption embeddings are centred by that domain's mean, 3D by
the fixed text_mean, Lottie by the mean of the training texts: both are stored in the index). So a query is encoded with CLIP once and centred twice, once per domain,
and ranked by cosine against that domain's clip embeddings. All 45,850 tokenised Lottie clips and all 12,703 3D clips are searched, training clips included: this is a
browsing tool, not an evaluation. The viewer page is viewer/lottie_search.html (endpoint /api/lottie_search in viewer/serve.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]


def index_path(tag: str) -> Path:
    return EMB / f"lottie_search_{tag}.npz"


def cmd_build(args) -> None:
    import torch
    import motion_sem as MS
    import motion_sem_joint as MJ
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    # the text means the model was trained with: rebuild the same sample (same seed, size and caption type)
    D, n3, L = MJ.load_joint(args.lottie, seed=args.seed, captions=args.captions)
    mu_l, mu_3d = D["mu_l"], D["mu"]
    model = MS.load_model(EMB / f"motion_sem_joint_{args.tag}.pt", dev)[0]
    e3 = MS.encode_all(model, D, np.arange(n3), dev, torch)
    keys3 = [f"{m['dataset']}|{m['clip']}" for m in D["meta"][:n3]]
    del D
    z = np.load(EMB / "lottie_feats.npz")
    DL = dict(joint=z["joint"], static=z["static"], offsets=z["offsets"], glob=z["glob"], scal=z["scal"])
    n = len(DL["offsets"]) - 1
    print(f"embedding {n:,} Lottie clips and {n3:,} 3D clips ...", flush=True)
    eL = MS.encode_all(model, DL, np.arange(n), dev, torch)
    rows = list(csv.DictReader(open(EMB / "lottie_feats_index.csv", newline="")))
    assert len(rows) == n
    np.savez_compressed(index_path(args.tag), lottie_emb=eL.astype(np.float16), lottie_ids=np.array([r["clip"] for r in rows]), lottie_shard=np.array([int(r["shard"]) for r in rows]),
                        d3_emb=e3.astype(np.float16), d3_keys=np.array(keys3), mu_l=mu_l.astype(np.float32), mu_3d=mu_3d.astype(np.float32),
                        trained_on=np.array(f"{args.tag}: captions={args.captions}, seed={args.seed}, lottie={args.lottie}"))
    print(f"-> {index_path(args.tag)}  ({time.time() - t0:.0f} s)")


class Searcher:
    """Loaded once (the server keeps one per index tag)."""

    def __init__(self, tag: str):
        z = np.load(index_path(tag))
        self.tag = tag
        self.lot, self.lot_ids, self.lot_shard = z["lottie_emb"].astype(np.float32), z["lottie_ids"], z["lottie_shard"]
        self.d3, self.d3_keys = z["d3_emb"].astype(np.float32), z["d3_keys"]
        self.mu_l, self.mu_3d, self.trained_on = z["mu_l"], z["mu_3d"], str(z["trained_on"])
        self.lot_meta = {r["clip"]: r for r in csv.DictReader(open(EMB / "lottie_feats_index.csv", newline=""))}
        self.d3_meta = {f"{r['dataset']}|{r['clip']}": r for r in csv.DictReader(open(EMB / "sem_feats_index.csv", newline=""))}
        self._enc = None
        self._fe = None

    def text_for(self, prompt: str, rewrite: bool) -> str:
        """as typed, or the motion part written in the 3D captions' wording by the rules front end (scripts/text/text_frontend.py)"""
        if not rewrite:
            return prompt
        if self._fe is None:
            import text_frontend as TF
            self._fe = TF.Frontend()
        shape, motion = self._fe.split(prompt)
        return self._fe.motion_text(motion) or prompt

    def search(self, prompt: str, k: int = 12, rewrite: bool = True) -> dict:
        from embed import encode, load_text_encoder
        if self._enc is None:
            self._enc = load_text_encoder()
        text = self.text_for(prompt, rewrite)
        e = encode([text], self._enc)[0][0]
        out = {"text_used": text, "index": self.trained_on}
        for name, emb, mu in (("lottie", self.lot, self.mu_l), ("d3", self.d3, self.mu_3d)):
            q = e - mu
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            s = emb @ q
            top = np.argsort(-s)[:k]
            hits = []
            for i in top:
                if name == "lottie":
                    cid = str(self.lot_ids[i])
                    m = self.lot_meta[cid]
                    hits.append(dict(id=cid, shard=int(self.lot_shard[i]), score=float(s[i]), body_plan=m["object_type"], subject=m["subject"], joints=int(m["joints"]), frames=int(m["frames"]), caption=m["caption"]))
                else:
                    key = str(self.d3_keys[i])
                    m = self.d3_meta[key]
                    hits.append(dict(key=key, score=float(s[i]), dataset=m["dataset"], cls=m["character_class"], caption=m["caption"], frames=m["frames"], fps=m["fps"]))
            out[name] = hits
        return out


def cmd_query(args) -> None:
    s = Searcher(args.tag)
    r = s.search(args.query, args.k, not args.as_typed)
    print(f"text used: {r['text_used']!r}   (index: {r['index']})\n\nLottie:")
    for h in r["lottie"]:
        print(f"  {h['score']:.3f} {h['id']:>9s} {h['body_plan']} {h['caption'][:110]}")
    print("\n3D:")
    for h in r["d3"]:
        print(f"  {h['score']:.3f} {h['cls']:10s} {h['key'].split('|')[1][:30]:30s} {h['caption'][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tag", default="mixed", help="the trained model: data/embeddings/motion_sem_joint_<tag>.pt")
    b.add_argument("--captions", default="mixed", choices=["full", "actions", "mixed"], help="what that model was trained with (to rebuild its text mean)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--lottie", type=int, default=11000)
    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("--tag", default="mixed")
    q.add_argument("--k", type=int, default=8)
    q.add_argument("--as-typed", action="store_true")
    args = ap.parse_args()
    {"build": cmd_build, "query": cmd_query}[args.cmd](args)


if __name__ == "__main__":
    main()
