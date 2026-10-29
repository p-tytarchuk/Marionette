#!/usr/bin/env python3
"""Rewrite a user's prompt into the wording of the training captions, at inference ("A cat do it backflip" -> "An object performs a backflip.").

  python3 scripts/text/rewrite_prompt.py "person does a back flip" "dog running fast"      # prints the rewrites

How: the CLIP text embedding of the prompt picks the nearest demonstrations among the TRAIN-split (reworded request -> dataset caption) pairs made by scripts/text/paraphrase.py
(data/embeddings/text_para.npz, split "train"), plus the nearest plain dataset captions for the style; a local instruction-tuned LLM (Qwen2.5-3B-Instruct, greedy) then writes the caption for the
request. The result is kept only if it is a sane sentence and its centred CLIP cosine to the request is at least MIN_COS (else the request is used unchanged). Only training material is used,
so validation clips never leak into the demonstrations. Rewrites are cached in data/embeddings/rewrite_cache.json (keyed by prompt).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
CACHE = EMB / "rewrite_cache.json"
MIN_COS = 0.3
K_PAIRS, K_CAPS = 6, 4

SYSTEM = ("You convert a user's animation request into the caption style of a motion-capture dataset. "
          "Rules: one sentence that begins with \"An object\", present tense, plain words, only body movement; keep every action and their order from the request; "
          "do not add details that are not in the request; at most 18 words. Whatever the subject of the request (a person, a cat, a dog, he, she), write \"An object\". "
          "Output only the caption.")


class Rewriter:
    def __init__(self, llm="3B", encoder=None):
        import paraphrase as PP
        from embed import encode, load_text_encoder
        self.PP, self.encode = PP, encode
        self.enc = encoder or load_text_encoder()
        self.mu = np.load(EMB / "text_mean.npy")
        D = np.load(EMB / "text_para.npz")
        m = D["split"] == "train"
        self.req, self.cap, self.req_emb = list(D["text"][m]), list(D["caption"][m]), D["emb"][m]
        z = np.load(EMB / "motion.npz")
        seen = {}
        for c, e, sp in zip(z["captions"], z["text_clip"], z["split"]):
            t = json.loads(c).get("caption", "").strip()
            if sp == "train" and t and t not in seen:
                seen[t] = e
        self.caps, self.caps_emb = list(seen), np.stack(list(seen.values()))
        self.llm_name, self.llm = llm, None
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def embed(self, texts):
        E = self.encode(list(texts), self.enc)[0] - self.mu
        return (E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def build(self, prompt, e):
        near = np.argsort(-(self.req_emb @ e))
        pairs, used = [], set()
        for i in near:
            if self.cap[i] in used:
                continue
            used.add(self.cap[i])
            pairs.append((self.req[i], self.cap[i]))
            if len(pairs) == K_PAIRS:
                break
        caps = [self.caps[i] for i in np.argsort(-(self.caps_emb @ e))[:K_CAPS]]
        ex = "\n".join(f"Request: {r}\nCaption: {c}" for r, c in pairs)
        sty = "\n".join(f"- {c}" for c in caps)
        return f"Examples of the conversion:\n{ex}\n\nOther captions of similar actions in the dataset:\n{sty}\n\nRequest: {prompt}\nCaption:"

    def rewrite(self, prompts, batch=16):
        prompts = [" ".join(p.split()) for p in prompts]
        todo = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if todo:
            if self.llm is None:
                self.llm = self.PP.load_llm(self.llm_name)
            tok, model, dev, torch = self.llm
            E = self.embed(todo)
            for b in range(0, len(todo), batch):
                chunk = todo[b:b + batch]
                msgs = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": self.build(p, E[b + j])}], tokenize=False, add_generation_prompt=True)
                        for j, p in enumerate(chunk)]
                enc = tok(msgs, return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
                raw = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().splitlines() for o in out]
                outs = [re.sub(r"^(Caption:|-)\s*", "", (r[0] if r else "")).strip().strip('"') for r in raw]
                R = self.embed([o if o else "x" for o in outs])
                for j, (p, o) in enumerate(zip(chunk, outs)):
                    ok = 3 <= len(o.split()) <= 25 and o.lower().startswith("an object") and float(R[j] @ E[b + j]) >= MIN_COS
                    self.cache[p] = o if ok else p
                print(f"  rewrote {b + len(chunk)}/{len(todo)}", flush=True)
            CACHE.write_text(json.dumps(self.cache, indent=0))
        return [self.cache[p] for p in prompts]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    rw = Rewriter()
    for p, r in zip(sys.argv[1:], rw.rewrite(sys.argv[1:])):
        print(f"{p!r:50s} -> {r!r}")


if __name__ == "__main__":
    main()
