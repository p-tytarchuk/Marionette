#!/usr/bin/env python3
"""One prompt -> a character (by shape retrieval) performing a generated motion.

  python3 scripts/generate/text_to_character.py "a small brown dog doing a backflip" [--top 3] [--seed 0] [--shape-only]
  python3 scripts/generate/text_to_character.py --split "a knight in armor swings a sword" "run"       # only the prompt split

Steps
  1. split   a local text-only LLM (Qwen2.5-1.5B-Instruct, greedy, few-shot) splits the prompt into a SHAPE part ("a small brown dog") and a MOTION part ("does a backflip"); either may be empty
  2. shape   the shape text is embedded with CLIP's text tower and matched against the CLIP image embeddings of the rest-pose portraits of the characters
             (scripts/generate/shape_retrieval.py, data/shape/all/clip_image.npz); only characters that are retarget targets (data/retarget_targets.csv, status ok) are candidates
  3. motion  scripts/generate_motion.generate(motion text, character): rewriting, diffusion, gate, decoder, retargeting
The result is a clip on a real character from the dataset (mesh, skeleton, skin weights); no new body is generated.
"""
import argparse
import csv
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
SHAPE = ROOT / "data" / "shape" / "all"
DEFAULT_CHARACTER, DEFAULT_MOTION = "mixamo_Aciana", "walks forward"
LLM = "1.5B"            # Qwen2.5-1.5B-Instruct (text-only, 2.9 GB) does the splitting and the rewriting; the 3B gave the same results on the test in docs/EMBED.md "Small LLM" and loads twice as slowly

SYSTEM = ("You split an animation request into two parts: SHAPE, the look of the character (what it is, size, colour, clothes), and MOTION, what it does (a verb phrase). "
          "Keep the user's words; do not invent. If a part is missing write none. Answer with exactly two lines: SHAPE: ... and MOTION: ...")
EXAMPLES = [("A cat doing a backflip", "a cat", "does a backflip"),
            ("a big red dragon walking forward", "a big red dragon", "walks forward"),
            ("knight in armor swings a sword and then bows", "a knight in armor", "swings a sword and then bows"),
            ("run", "none", "runs"),
            ("a robot dog", "a robot dog", "none"),
            ("cute cartoon fox jumping and waving", "a cute cartoon fox", "jumps and waves"),
            ("person walks in a circle", "a person", "walks in a circle")]


def messages(prompt):
    m = [{"role": "system", "content": SYSTEM}]
    for p, s, mo in EXAMPLES:
        m += [{"role": "user", "content": p}, {"role": "assistant", "content": f"SHAPE: {s}\nMOTION: {mo}"}]
    return m + [{"role": "user", "content": prompt}]


_shared = {}
_lock = threading.Lock()
_llm_lock = threading.Lock()


def shared_llm():
    """The Qwen2.5-1.5B (LLM above) used for splitting; generate_motion hands the same copy to the prompt rewriter, so only one is in memory."""
    with _llm_lock:
        if "llm" not in _shared:
            import paraphrase as PP
            _shared["llm"] = PP.load_llm(LLM)
        return _shared["llm"]


def get_index():
    if "idx" not in _shared:
        _shared["idx"] = ShapeIndex()
    return _shared["idx"]


def split_prompts(prompts, llm=None):
    """-> [(shape, motion)] ('' when missing)"""
    with _lock:
        tok, model, dev, torch = llm or shared_llm()
        return _split(prompts, tok, model, dev, torch)


def _split(prompts, tok, model, dev, torch):
    out = []
    for p in prompts:
        text = tok.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        g = lambda tag: (re.search(rf"{tag}:\s*(.*)", raw) or [None, ""])[1].strip().strip('."')
        s, mo = g("SHAPE"), g("MOTION")
        out.append(("" if s.lower() in ("none", "", "someone", "something", "it", "an object", "a character", "a creature") else s, "" if mo.lower() in ("none", "") else mo))
    return out


class ShapeIndex:
    def __init__(self):
        z = np.load(SHAPE / "clip_image.npz")
        v = z["views"].mean(1)
        self.X = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.keys = [str(k) for k in z["keys"]]
        ok = {r["key"]: r for r in csv.DictReader(open(ROOT / "data" / "retarget_targets.csv", newline=""))}
        self.target = np.array([ok.get(k, {}).get("status") == "ok" for k in self.keys])
        self.cls = [ok.get(k, {}).get("cls", "") for k in self.keys]
        self.clip = None

    def search(self, shape, top=3, targets_only=True):
        """-> [(character key, class, cosine)]"""
        import shape_retrieval as SR
        e = SR.text_embed([shape])[0] if self.clip is None else self.clip(shape)
        s = self.X @ e
        if targets_only:
            s = np.where(self.target, s, -9)
        return [(self.keys[j], self.cls[j], float(s[j])) for j in np.argsort(-s)[:top]]


def choose(prompt, top=6, frontend="rules"):
    """What the generator page needs: split the prompt, and if it names a shape, rank the characters.
    -> dict(shape, motion, hits=[dict(key, cls, cos)]) ; hits is empty when the prompt has no shape part (the caller keeps its own character and the prompt as typed)."""
    if frontend == "rules":
        if "fe" not in _shared:
            import text_frontend
            _shared["fe"] = text_frontend.Frontend()
        shape, motion = _shared["fe"].split(prompt)
    else:
        shape, motion = split_prompts([prompt])[0]
    hits = [dict(key=k, cls=c, cos=round(s, 3)) for k, c, s in get_index().search(shape, top)] if shape else []
    return dict(shape=shape, motion=motion, hits=hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--split", action="store_true", help="only print the shape / motion split")
    ap.add_argument("--shape-only", action="store_true", help="retrieve the character, do not generate motion")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--frontend", choices=["rules", "llm"], default="rules", help="split and rewrite the prompt by rules (no language model) or with the Qwen2.5-1.5B LLM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-characters", action="store_true", help="also characters that are not retarget targets (no motion can be put on them)")
    a = ap.parse_args()
    if a.frontend == "rules":
        import text_frontend
        fe = text_frontend.Frontend()
        parts = [fe.split(p) for p in a.prompt]
    else:
        parts = split_prompts(a.prompt)
    if a.split:
        for p, (s, m) in zip(a.prompt, parts):
            print(f"{p!r:60s} SHAPE {s!r:36s} MOTION {m!r}")
        return
    idx = ShapeIndex()
    import generate_motion as G
    for p, (shape, motion) in zip(a.prompt, parts):
        hits = idx.search(shape, a.top, not a.all_characters) if shape else [(DEFAULT_CHARACTER, "biped", 0.0)]
        print(f"{p!r}: shape {shape or '(none: default character)'!r}, motion {motion or '(none: ' + DEFAULT_MOTION + ')'!r}")
        for k, c, s in hits:
            print(f"   {k} ({c}) cos {s:.3f}")
        if a.shape_only:
            continue
        r = G.generate(motion or DEFAULT_MOTION, hits[0][0], seed=a.seed, frontend=a.frontend)
        print(f"   motion {r['prompt']!r} on {r['target']}: {r['name']} (gate passed {r['gate_passed']}, quality {r['quality']})")
        print(f"   view: http://127.0.0.1:8765/viewer/compare.html#{r['target']}|{r['name']}")


if __name__ == "__main__":
    main()
