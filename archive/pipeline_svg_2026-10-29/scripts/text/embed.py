#!/usr/bin/env python3
"""Caption embeddings and text retrieval for the clips (docs/EMBED.md).

  python3 scripts/text/embed.py --text                      # embed every caption -> data/embeddings/
  python3 scripts/text/embed.py --query "walks slowly"      # rank clips by caption similarity
  python3 scripts/text/embed.py --query backflip --cls quadruped --top 10 --usable

The text encoder is OpenAI CLIP ViT-B/32 (text tower, frozen, 512 numbers per text), as in the pipeline plan.
Weights come from the Hugging Face cache (openai/clip-vit-base-patch32, downloaded on first use).
Captions: three styles per clip in data/inventory/clips.csv (caption, generic, detail). Each distinct string is
embedded once; text_index.csv maps (dataset, clip, style) to a row of text.npy. Embeddings are L2-normalized
(CLIPTextModelWithProjection returns them un-normalized).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
CLIP_NAME = "openai/clip-vit-base-patch32"
STYLES = ("caption", "generic", "detail")
PREFIX = "An object "           # every caption starts with it; it carries no information about the motion


def text_mean():
    """Mean of the dataset-caption embeddings, fixed in data/embeddings/text_mean.npy (so adding captions later does not move it)."""
    f = EMB / "text_mean.npy"
    return np.load(f) if f.exists() else np.load(EMB / "text.npy").mean(0)


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_text_encoder():
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_NAME).eval().to(device())
    return tok, model, torch


def encode(texts, enc=None, batch=256):
    """(N, 512) float32, L2-normalized. Also returns how many texts exceed CLIP's 77 tokens."""
    tok, model, torch = enc or load_text_encoder()
    out, cut = [], 0
    for i in range(0, len(texts), batch):
        chunk = list(texts[i:i + batch])
        ids = tok(chunk, padding=True, truncation=True, max_length=77, return_tensors="pt")
        full = tok(chunk, truncation=False)["input_ids"]
        cut += sum(len(x) > 77 for x in full)
        with torch.no_grad():
            e = model(**{k: v.to(device()) for k, v in ids.items()}).text_embeds
        out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    return np.concatenate(out), cut


def clean(text, strip):
    return text[len(PREFIX):] if strip and text.startswith(PREFIX) else text


# ----------------------------------------------------------------------------- stage 1: captions
def read_clips():
    with open(ROOT / "data" / "inventory" / "clips.csv", newline="") as f:
        return list(csv.DictReader(f))


def cmd_text(args):
    EMB.mkdir(parents=True, exist_ok=True)
    clips = read_clips()
    pairs = [(r["dataset"], r["clip"], s, r[s]) for r in clips for s in STYLES if r[s]]
    uniq = sorted({t for *_, t in pairs})
    row_of = {t: i for i, t in enumerate(uniq)}
    texts = [clean(t, args.strip_prefix) for t in uniq]
    print(f"{len(clips)} clips, {len(pairs)} captions, {len(uniq)} distinct; strip_prefix={args.strip_prefix}")
    emb, cut = encode(texts)
    print(f"embedded {emb.shape}; over 77 tokens (cut): {cut} ({100 * cut / len(uniq):.2f}% of distinct captions)")
    np.save(EMB / "text.npy", emb)
    with open(EMB / "text_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "clip", "style", "row", "text"])
        for ds, clip, s, t in pairs:
            w.writerow([ds, clip, s, row_of[t], t])
    (EMB / "text_meta.json").write_text(json.dumps(dict(model=CLIP_NAME, dim=int(emb.shape[1]), strip_prefix=bool(args.strip_prefix),
                                                        distinct=len(uniq), captions=len(pairs), truncated=int(cut),
                                                        mean_cosine=float((emb @ emb.mean(0)).mean())), indent=1))


class TextIndex:
    """Caption embeddings of every clip, for retrieval. Centred by the mean caption embedding (all captions share
    the same sentence shape, so raw cosines are high everywhere; centring leaves the part that differs)."""

    def __init__(self, centre=True):
        self.meta = json.loads((EMB / "text_meta.json").read_text())
        self.emb = np.load(EMB / "text.npy")
        self.mu = text_mean() if centre else np.zeros(self.emb.shape[1], np.float32)
        self.clips = {(r["dataset"], r["clip"]): r for r in read_clips()}
        self.rows = {}
        with open(EMB / "text_index.csv", newline="") as f:
            for r in csv.DictReader(f):
                self.rows.setdefault((r["dataset"], r["clip"]), []).append((r["style"], int(r["row"]), r["text"]))
        self._enc = None

    def embed_query(self, text):
        self._enc = self._enc or load_text_encoder()
        return encode([clean(text, self.meta["strip_prefix"])], self._enc)[0][0]

    def _unit(self, v):
        v = v - self.mu
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    def rank(self, text, cls=None, usable=True, dataset=None, only=None, top=10):
        """[(score, dataset, clip, best caption)] by the best cosine over a clip's captions."""
        q = self._unit(self.embed_query(text))
        sims = self._unit(self.emb) @ q
        out = []
        for key, caps in self.rows.items():
            r = self.clips[key]
            if (cls and r["character_class"] != cls) or (usable and r["usable"] != "1") or (dataset and key[0] != dataset):
                continue
            if only is not None and key[1] not in only:
                continue
            s, t = max((sims[row], txt) for _, row, txt in caps)
            out.append((float(s), key[0], key[1], t))
        out.sort(reverse=True)
        return out[:top]


def cmd_query(args):
    idx = TextIndex(centre=not args.no_centre)
    for q in args.query:
        print(f"\n{q!r}")
        for s, ds, clip, t in idx.rank(q, cls=args.cls, usable=args.usable, dataset=args.dataset, top=args.top):
            print(f"  {s:6.3f}  {ds:9s} {clip[:44]:44s} {t[:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true", help="embed all captions -> data/embeddings/text.npy + text_index.csv")
    ap.add_argument("--strip-prefix", action="store_true", help="with --text: drop the leading 'An object ' before embedding")
    ap.add_argument("--query", nargs="+", help="one or more texts; lists the best matching clips")
    ap.add_argument("--cls", choices=["biped", "quadruped", "avian", "insectoid", "marine", "serpentine"], help="filter by source class")
    ap.add_argument("--dataset", choices=["mixamo", "objaverse"])
    ap.add_argument("--usable", action="store_true", default=True)
    ap.add_argument("--all-clips", dest="usable", action="store_false", help="include clips marked not usable")
    ap.add_argument("--no-centre", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    if args.text:
        cmd_text(args)
    elif args.query:
        cmd_query(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
