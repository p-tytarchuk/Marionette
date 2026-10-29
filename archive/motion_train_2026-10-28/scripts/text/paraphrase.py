#!/usr/bin/env python3
"""Paraphrases of motion captions with a local instruction-tuned LLM (sources/models/Qwen2.5-*-Instruct), for text-conditioning augmentation.

  python3 scripts/text/paraphrase.py --pilot 40 [--model 3B|1.5B] [--style std|casual]     # time and show a few, writes nothing
  python3 scripts/text/paraphrase.py --split train --model 3B --style std                  # distinct captions of the TRAIN clips -> data/embeddings/para_train.jsonl (resumable)
  python3 scripts/text/paraphrase.py --split val --model 1.5B --style casual               # captions of the held-out clips -> para_val.jsonl (an evaluation set, never trained on)
  python3 scripts/text/paraphrase.py --embed                                               # filter + CLIP-embed the jsonl files -> text_para.npz

Rules: a train caption is paraphrased once per (caption, class); a quadruped caption may get an animal subject, a biped caption a person / "the character" / no subject, never the other way round.
Output lines: {"caption", "cls", "style", "variants": [...]}. The filter (--embed) drops a variant whose centred CLIP cosine to its caption is below --min-cos, or that loses the caption's main action word
(flip, kick, punch, wave, jump, sit, dance, crawl, climb, throw, walk, run, crouch ...: the same 4-letter stem must remain).
Nothing the other scripts read (text.npy, text_index.csv, text_mean.npy, motion*.npz, z_diffusion.pt) is changed; text_mean.npy is only read, to centre the new embeddings the same way.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
MODELS = dict(**{"3B": ROOT / "sources" / "models" / "Qwen2.5-3B-Instruct", "1.5B": ROOT / "sources" / "models" / "Qwen2.5-1.5B-Instruct"})
ACTIONS = ["flip", "kick", "punch", "wave", "jump", "sit", "danc", "crawl", "climb", "throw", "walk", "run", "crouch", "spin", "roll", "fall", "bow", "squat", "stretch", "swim", "fly", "hop", "turn", "kneel", "lie", "stand"]

SUBJ = {"biped": "a person, someone, the character, he/she, or no subject at all (imperative or noun phrase)",
        "quadruped": "an animal, the creature, it, or no subject at all (imperative or noun phrase); do not name a species"}
STYLE = {
    "std": "Write {n} different ways a user might type this motion request into a text-to-animation tool. Vary the wording and sentence structure: some short, some longer, some imperative (\"do a ...\"), "
           "some noun phrases. Keep exactly the same action(s), order and details. Subject: {subj}.",
    "casual": "Write {n} versions as a hurried casual user might type them: lowercase, missing articles, slightly ungrammatical, small typos allowed, sometimes only a few words. Keep exactly the same action(s). Subject: {subj}.",
}


def build_prompt(caption, cls, style, n):
    subj = SUBJ["quadruped" if cls == "quadruped" else "biped"]
    return (f"Motion description: \"{caption}\"\n{STYLE[style].format(n=n, subj=subj)}\n"
            f"Output exactly {n} lines, one version per line, no numbering, no quotes, no extra text.")


def load_llm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELS[name], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    return tok, model, dev, torch


def generate(llm, items, style, n=5, temp=0.8, max_new=110):
    """items: [(caption, cls)] -> list of variant lists (raw, unfiltered)"""
    tok, model, dev, torch = llm
    msgs = [tok.apply_chat_template([{"role": "system", "content": "You rewrite short motion descriptions. Follow the output format exactly."},
                                     {"role": "user", "content": build_prompt(c, k, style, n)}], tokenize=False, add_generation_prompt=True) for c, k in items]
    enc = tok(msgs, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95, pad_token_id=tok.pad_token_id)
    res = []
    for o in out[:, enc["input_ids"].shape[1]:]:
        lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip().strip('"') for l in tok.decode(o, skip_special_tokens=True).splitlines()]
        res.append([l for l in lines if l])
    return res


def distinct_captions(split):
    """[(caption, cls)] of the clips in motion.npz of this split, from the 'caption' style of the dataset"""
    z = np.load(EMB / "motion.npz")
    seen = {}
    for cap, cls, sp in zip(z["captions"], z["cls"], z["split"]):
        if sp != split:
            continue
        c = json.loads(cap).get("caption", "").strip()
        if c:
            seen.setdefault((c, str(cls)), 0)
    return sorted(seen)


def run_split(args):
    items = distinct_captions(args.split)
    out = EMB / f"para_{args.split}.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(l)["caption"], json.loads(l)["cls"]) for l in open(out)}
    todo = [it for it in items if it not in done]
    print(f"{len(items)} distinct (caption, class) in {args.split}, {len(done)} done, {len(todo)} to do", flush=True)
    if args.limit:
        todo = todo[:args.limit]
    llm = load_llm(args.model)
    t0 = time.time()
    with open(out, "a") as f:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            for (c, k), v in zip(chunk, generate(llm, chunk, args.style, args.n)):
                f.write(json.dumps(dict(caption=c, cls=k, style=args.style, model=args.model, variants=v)) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  {b + len(chunk)}/{len(todo)}  {el:.0f}s  ({el / (b + len(chunk)):.2f} s/caption)", flush=True)


def pilot(args):
    items = distinct_captions("train")[::max(1, len(distinct_captions('train')) // args.pilot)][:args.pilot]
    llm = load_llm(args.model)
    t0 = time.time()
    res = []
    for b in range(0, len(items), args.batch):
        res += generate(llm, items[b:b + args.batch], args.style, args.n)
    el = time.time() - t0
    print(f"{len(items)} captions in {el:.0f}s = {el / len(items):.2f} s/caption (batch {args.batch}, {args.model}, {args.style})")
    for (c, k), v in list(zip(items, res))[:14]:
        print(f"[{k}] {c}")
        for x in v:
            print("      ", x)


def stems(text):
    t = text.lower()
    return {a for a in ACTIONS if a in t}


def keep(variant, caption, cos, min_cos):
    if not (2 <= len(variant.split()) <= 30) or variant.lower().strip(" .") == caption.lower().strip(" ."):
        return False
    if cos < min_cos:
        return False
    need = stems(caption)
    return all(s in variant.lower() or (s == "flip" and "somersault" in variant.lower()) for s in need) if need else True


def embed(args):
    from embed import encode, load_text_encoder
    mu = np.load(EMB / "text_mean.npy")
    enc = load_text_encoder()
    rows = []
    for name in args.files:
        for l in open(EMB / f"para_{name}.jsonl"):
            r = json.loads(l)
            for v in r["variants"]:
                rows.append((name, r["caption"], r["cls"], r["style"], v))
    texts = sorted({t for _, c, _, _, v in rows for t in (c, v)})
    E = encode(texts, enc)[0] - mu
    E = (E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    ix = {t: i for i, t in enumerate(texts)}
    kept = []
    for name, c, k, st, v in rows:
        cos = float(E[ix[c]] @ E[ix[v]])
        if keep(v, c, cos, args.min_cos):
            kept.append((name, c, k, st, v, cos))
    kept = list(dict.fromkeys(kept))
    print(f"{len(rows)} variants, {len(kept)} kept (min cos {args.min_cos}); cosine of kept: median {np.median([x[5] for x in kept]):.2f}")
    for name in args.files:
        sub = [x for x in kept if x[0] == name]
        print(f"  {name}: {len(sub)} variants for {len({(x[1], x[2]) for x in sub})} captions")
    out = EMB / "text_para.npz"
    np.savez(out, split=np.array([x[0] for x in kept]), caption=np.array([x[1] for x in kept]), cls=np.array([x[2] for x in kept]),
             style=np.array([x[3] for x in kept]), text=np.array([x[4] for x in kept]), cos=np.array([x[5] for x in kept], np.float32),
             emb=np.stack([E[ix[x[4]]] for x in kept]))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=int)
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--files", nargs="+", default=["train", "val"])
    ap.add_argument("--model", choices=list(MODELS), default="3B")
    ap.add_argument("--style", choices=list(STYLE), default="std")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-cos", type=float, default=0.6)
    a = ap.parse_args()
    if a.pilot:
        pilot(a)
    elif a.split:
        run_split(a)
    elif a.embed:
        embed(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
