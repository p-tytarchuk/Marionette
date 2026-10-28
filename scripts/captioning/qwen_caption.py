#!/usr/bin/env python3
"""Caption clips with Qwen3-VL-2B from rendered stick-figure filmstrips, and keep the model's own motion embedding.

  python3 scripts/captioning/qwen_caption.py --eval 60 --prompt v1,v2,v3    # compare prompts on a sample (CLIP retrieval vs the dataset captions)
  python3 scripts/captioning/qwen_caption.py --run --prompt v3              # all usable clips, resumable -> data/captions/qwen_<prompt>.jsonl
  python3 scripts/captioning/qwen_caption.py --run --limit 100              # first 100 only

Per clip: render_motion.py draws 8 frames (4 x 2 grid); Qwen3-VL-2B (sources/models, fp16, eager attention on MPS,
the SDPA kernel fails there) writes a SHORT and a DETAIL motion caption. The embedding is the mean of the last-layer
hidden states over the image tokens of the prompt pass (2048 numbers): the model's own view of the motion.
Each JSONL line: dataset, clip, short, detail, raw, emb (float16, base64), seconds.
"""
import argparse
import base64
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
MODEL = ROOT / "sources" / "models" / "Qwen3-VL-2B-Instruct"
OUT = ROOT / "data" / "captions"

HEAD = ("The image shows 8 frames of a stick-figure character moving, in time order (left to right, then the top row "
        "before the bottom row). Blue lines are the left limbs and orange lines the right limbs. The floor ticks slide "
        "when the character travels. The clip lasts {dur:.1f} seconds. ")
FORMAT = ("Describe only the motion, not the body, and call the character 'object'. Answer with exactly two lines:\n"
          "SHORT: one sentence of at most 10 words starting with 'An object'.\n"
          "DETAIL: one sentence of 15 to 30 words starting with 'An object'.")
EXAMPLES = ("Examples of the style:\nSHORT: An object performs a backflip and lands in a crouch.\n"
            "DETAIL: An object swings its arms back and crouches, backflips, lands in a crouch, then straightens up.\n"
            "SHORT: An object walks forward, then turns around.\nSHORT: An object stands and nods its head.\n")
PROMPTS = {
    "v1": HEAD + FORMAT,
    # v2: commit to what is visible, frame by frame, before answering (chain of thought, costs more tokens)
    "v2": HEAD + "First, in one line starting with FRAMES:, say in a few words what the character does in each of the 8 frames "
           "(for example whether it is upright, bent, off the ground, upside down, or moving forward). Then answer:\n"
           "SHORT: one sentence of at most 10 words starting with 'An object'.\n"
           "DETAIL: one sentence of 15 to 30 words starting with 'An object'. Name the action (walk, run, jump, flip, turn, "
           "crouch, kick, punch, wave, sit, fall, dance, idle...) when you can see it.",
    # v3: examples of the target style
    "v3": HEAD + FORMAT + "\n" + EXAMPLES,
}


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="eager").to(dev).eval()
    return proc, model, torch, dev


def parse(raw):
    g = lambda tag: (re.search(rf"{tag}:\s*(.+)", raw) or [None, ""])[1].strip().strip("*").strip()
    return g("SHORT"), g("DETAIL")


def caption(enc, sk, prompt, max_new=110):
    """-> (short, detail, raw, emb (2048,) float32)"""
    from render_motion import render
    proc, model, torch, dev = enc
    img = render(sk)
    dur = len(sk["Wp"]) / sk["fps"]
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPTS[prompt].format(dur=dur)}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
    prefill = out.hidden_states[0][-1][0]                                   # (tokens, 2048) last layer, prompt pass
    img_id = model.config.image_token_id
    mask = (inp["input_ids"][0] == img_id)
    emb = prefill[mask].float().mean(0).cpu().numpy()
    raw = proc.batch_decode(out.sequences[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    short, detail = parse(raw)
    return short, detail, raw, emb


def usable_clips(order="mixamo-first"):
    with open(ROOT / "data" / "inventory" / "clips.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["usable"] == "1"]
    key = {"mixamo-first": lambda r: (r["dataset"] != "mixamo", r["character_class"] != "quadruped")}[order]
    return sorted(rows, key=key)


def sample(n, seed=0):
    rows = usable_clips()
    rnd = random.Random(seed)
    groups = {"mixamo": [r for r in rows if r["dataset"] == "mixamo"],
              "biped": [r for r in rows if r["dataset"] == "objaverse" and r["character_class"] == "biped"],
              "quad": [r for r in rows if r["character_class"] == "quadruped"]}
    out = []
    for g in groups.values():
        out += rnd.sample([r for r in g if r["caption"]], n // 3)
    return out


def cmd_eval(args):
    from embed import encode, load_text_encoder
    import render_motion as RM
    enc = load_model()
    tenc = None
    rows = sample(args.eval)
    sks = []
    for r in rows:
        sks.append(RM.load_for_render(r["clip"]))
    truth = [r["caption"] for r in rows]
    tenc = load_text_encoder()
    T, _ = encode(truth, tenc)
    for pname in args.prompt.split(","):
        t0 = time.time()
        shorts, details = [], []
        for r, sk in zip(rows, sks):
            s, d, raw, _ = caption(enc, sk, pname)
            shorts.append(s or "An object moves.")
            details.append(d or s or "An object moves.")
        dt = time.time() - t0
        print(f"\nprompt {pname}: {dt / len(rows):.1f} s per clip")
        for name, txt in (("short", shorts), ("detail", details)):
            E, _ = encode(txt, tenc)
            sim = E @ T.T                                                    # query = Qwen caption, gallery = the dataset captions
            rank = (sim > np.diag(sim)[:, None]).sum(1)
            # ties among identical truth captions would cost nothing here: they count as rank 0 only when strictly greater
            print(f"  {name:6s} diag cos {np.diag(sim).mean():.3f} (random pair {sim.mean():.3f}) | R@1 {100 * (rank < 1).mean():.0f}%  R@5 {100 * (rank < 5).mean():.0f}%  R@10 {100 * (rank < 10).mean():.0f}%  median rank {np.median(rank):.0f} of {len(rows)}")
        for i in range(0, len(rows), max(1, len(rows) // 5)):
            print(f"   [{rows[i]['clip'][:28]}] Q: {shorts[i]}  | T: {truth[i]}")


def done_keys(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            out.add((d["dataset"], d["clip"]))
        except Exception:
            pass
    return out


def cmd_run(args):
    import render_motion as RM
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"qwen_{args.prompt}.jsonl"
    done = done_keys(path)
    rows = [r for r in usable_clips() if (r["dataset"], r["clip"]) not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(done)} done, {len(rows)} to do -> {path}")
    enc = load_model()
    t_start = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(rows):
            t0 = time.time()
            try:
                sk = RM.load_for_render(r["clip"])
                s, d, raw, emb = caption(enc, sk, args.prompt)
                rec = dict(dataset=r["dataset"], clip=r["clip"], cls=r["character_class"], short=s, detail=d, raw=raw,
                           emb=base64.b64encode(emb.astype(np.float16).tobytes()).decode(), seconds=round(time.time() - t0, 2))
            except Exception as e:
                rec = dict(dataset=r["dataset"], clip=r["clip"], cls=r["character_class"], error=f"{type(e).__name__}: {e}")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t_start
                print(f"  {i + 1}/{len(rows)}  {el / (i + 1):.1f} s per clip, about {(len(rows) - i - 1) * el / (i + 1) / 3600:.1f} h left", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=int, help="compare prompts on a sample of N usable clips")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--prompt", default="v3", help="prompt id (v1, v2, v3); a comma list with --eval")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.eval:
        cmd_eval(args)
    elif args.run:
        cmd_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
