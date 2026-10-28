#!/usr/bin/env python3
"""Pilot: can Qwen3-VL act as a CAPTION CHECKER?  Given the filmstrip of a clip and a caption, does it say Yes/No to "does the caption describe the motion"?

  python3 scripts/captioning/qwen_check_pilot.py --dir DIR --vlm 8B [--noun animal|character] [--tag _8B]    # DIR holds clips.json and NNN.png from qwen_mesh_pilot.py --render

For each clip: the TRUE caption (should be Yes), a RANDOM caption of another clip, and a HARD negative: the CLIP-nearest caption of another clip that names a different main action (no shared action word).
Score = logit("Yes") - logit("No") of the first answer token. Reported: AUC true vs random and true vs hard, accuracy at threshold 0, how many TRUE captions are flagged (score < 0) and how many wrong ones are caught,
and the true captions the model distrusts most (a list to check by eye against the renders; real dataset errors would be here, so would model errors).
Saves DIR/check<tag>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
ACTIONS = ["flip", "kick", "punch", "wave", "jump", "hop", "sit", "danc", "crawl", "climb", "throw", "walk", "run", "crouch", "spin", "roll", "fall", "bow", "squat", "stretch", "turn", "kneel", "lie",
           "head", "tail", "rear", "attack", "lower", "raise", "bite", "stand", "sway", "swim", "fly", "shake", "sniff", "gallop", "jog", "sneak", "lung"]


def stems(t):
    t = t.lower()
    return {a for a in ACTIONS if a in t}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--vlm", choices=["2B", "4B", "8B"], default="8B")
    ap.add_argument("--noun", default="animal", help="what the character is called in the prompt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from embed import encode, load_text_encoder
    d = Path(a.dir)
    cl = json.loads((d / "clips.json").read_text())
    n = len(cl)
    caps = [c["caption"] for c in cl]
    E, _ = encode(caps, load_text_encoder())
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E.T
    rng = np.random.default_rng(0)
    rand_neg, hard_neg = [], []
    for i in range(n):
        others = [j for j in range(n) if j != i and caps[j] != caps[i]]
        rand_neg.append(int(rng.choice(others)))
        si = stems(caps[i])
        cand = [j for j in others if not (si & stems(caps[j]))] or others
        hard_neg.append(int(max(cand, key=lambda j: sim[i, j])))
    path = ROOT / "sources" / "models" / f"Qwen3-VL-{a.vlm}-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float16 if a.vlm == "2B" else torch.bfloat16, attn_implementation="eager").to(dev).eval()
    tok = proc.tokenizer
    yes, no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]

    def score(img, dur, caption):
        text = (f"The image shows 8 frames of a 3D {a.noun} character performing an action, seen from the side, in time order (left to right, then the top row before the bottom row). "
                f"The clip lasts {dur:.1f} seconds.\nCaption: \"{caption}\"\n"
                "Does the caption correctly describe the main motion shown in the frames? Answer No if the caption names an action that is not visible or the main action is different. Answer with one word: Yes or No.")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        return float(lg[yes] - lg[no])

    t0 = time.time()
    S = {k: [] for k in ("true", "random", "hard")}
    for i, c in enumerate(cl):
        img = Image.open(d / f"{i:03d}.png").convert("RGB")
        S["true"].append(score(img, c["dur"], caps[i]))
        S["random"].append(score(img, c["dur"], caps[rand_neg[i]]))
        S["hard"].append(score(img, c["dur"], caps[hard_neg[i]]))
        if i % 10 == 0:
            print(f"  {i}/{n} {time.time() - t0:.0f}s", flush=True)
    sec = (time.time() - t0) / (3 * n)
    t, r, h = (np.array(S[k]) for k in ("true", "random", "hard"))
    print(f"\nchecker {a.vlm} ({sec:.1f} s per judgement), {n} clips, {a.noun}")
    print(f"  AUC true vs random negative {auc(t, r):.2f} | true vs hard negative {auc(t, h):.2f}   (0.5 = no signal)")
    print(f"  at threshold 0: true captions accepted {100 * np.mean(t > 0):.0f}% | random negatives rejected {100 * np.mean(r <= 0):.0f}% | hard negatives rejected {100 * np.mean(h <= 0):.0f}%")
    print(f"  mean score: true {t.mean():.2f}  random {r.mean():.2f}  hard {h.mean():.2f}")
    flagged = np.argsort(t)[:12]
    print("\n  true captions the checker distrusts most (score, clip index, caption | caption of the hard negative scored for comparison):")
    for i in flagged:
        print(f"   {t[i]:6.2f}  #{i:02d} {cl[i]['caption'][:70]}")
    (d / f"check{a.tag}.json").write_text(json.dumps(dict(scores=S, rand_neg=rand_neg, hard_neg=hard_neg, vlm=a.vlm, sec=sec), indent=1))


if __name__ == "__main__":
    main()
