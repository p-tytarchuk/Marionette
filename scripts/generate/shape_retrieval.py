#!/usr/bin/env python3
"""Prompt -> character by shape: retrieval over the normalized characters (pilot on a stratified sample).

  python3 scripts/generate/shape_retrieval.py sample [--n 100]                      # data/shape/pilot/chars.json: characters stratified by rig category (seed 0)
  /usr/local/bin/python3 scripts/generate/shape_retrieval.py render                 # rest-pose portraits data/shape/pilot/img/<key>.png: front, side, 3/4 (needs viewer/serve.py + playwright)
  python3 scripts/generate/shape_retrieval.py clip                                  # CLIP image embeddings of the portraits (zero-shot image-text retrieval)
  python3 scripts/generate/shape_retrieval.py caption [--vlm 2B]                    # Qwen3-VL shape captions of the portraits -> CLIP text embeddings
  python3 scripts/generate/shape_retrieval.py eval                                  # prompt -> characters: category precision, and the top hits as contact sheets
  python3 scripts/generate/shape_retrieval.py query "a small four-legged animal"    # top characters of the pilot set for a prompt

Two ways to index a character by its looks, both in the CLIP text space the motion generator already uses:
  image    CLIP image tower on the 3-view portrait (no captioning model needed)
  caption  Qwen3-VL describes the portrait in a sentence, CLIP text tower embeds the sentence
The pilot has no shape labels, so the check is the rig category of data/inventory/rigs.csv (humanoid = Mixamo, bipedal, quadrupedal; a topology-based label, not a
perfect one) plus looking at the retrieved portraits.
"""
import argparse
import csv
import json
import random
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "shape" / "pilot"                # set by --set
CLIP_NAME = "openai/clip-vit-base-patch32"
QUOTA = dict(humanoid=20, bipedal=60, quadrupedal=70)        # the kept characters are only biped / quadruped rigs (data/manifest.csv)
QUERIES = {                                   # category -> prompts that should retrieve it
    "humanoid": ["a human character", "a person standing"],
    "bipedal": ["a humanoid monster", "a two-legged creature standing upright", "a robot"],
    "quadrupedal": ["a four-legged animal", "a dog", "a horse", "a cat"],
}
CAPTION_PROMPT = ("The image shows three views of one 3D character in a T-pose or rest pose: front, side and three-quarter. Describe what the character is and its body shape in ONE sentence "
                  "of at most 25 words: the kind of creature or person, build, number of legs, and notable parts (tail, wings, horns, long neck). Start with 'A'. Do not mention the views or the pose.")


def cmd_sample(a):
    rig = {r["object_type"]: r["category"] for r in csv.DictReader(open(ROOT / "data" / "inventory" / "rigs.csv", newline="")) if r["dataset"] == "objaverse"}
    rows = []
    for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", newline="")):
        key = f"{r['source']}_{r['id']}"
        if (ROOT / "data" / "characters" / key / "model.glb").exists():
            cat = "humanoid" if r["source"] == "mixamo" else rig.get(r["id"], "")
            rows.append(dict(key=key, category=cat, manifest_class=r["category"]))
    rnd = random.Random(0)
    rnd.shuffle(rows)
    if a.set == "all":
        OUT.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: r["key"])
        (OUT / "chars.json").write_text(json.dumps(rows, indent=1))
        print(len(rows), "characters")
        return
    scale = a.n / sum(QUOTA.values())
    got, per = [], {}
    for r in rows:
        q = max(1, round(QUOTA.get(r["category"], 0) * scale))
        if r["category"] in QUOTA and per.get(r["category"], 0) < q:
            per[r["category"]] = per.get(r["category"], 0) + 1
            got.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chars.json").write_text(json.dumps(got, indent=1))
    print(len(got), "characters", per)


def chars():
    """the characters of the set that have a portrait (a render in progress gives a partial set)"""
    return [c for c in json.loads((OUT / "chars.json").read_text()) if (OUT / "img" / f"{c['key']}.png").exists() or not (OUT / "img").exists()]


def cmd_render(a):
    from PIL import Image
    from playwright.sync_api import sync_playwright
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    cl = [c for c in json.loads((OUT / "chars.json").read_text()) if not (OUT / "img" / f"{c['key']}.png").exists()]
    W = H = a.size
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": W * 3, "height": H * a.batch})
        for i in range(0, len(cl), a.batch):
            part = cl[i:i + a.batch]
            pg.goto(f"http://127.0.0.1:8765/viewer/portrait.html?chars={urllib.parse.quote(';'.join(c['key'] for c in part))}&w={W}&h={H}")
            pg.wait_for_function("window.__ready === true", timeout=300000)
            pg.locator("canvas").screenshot(path=str(OUT / "_batch.png"))
            failed = set(pg.evaluate("window.__failed"))
            im = Image.open(OUT / "_batch.png").convert("RGB")
            for k, c in enumerate(part):
                if c["key"] not in failed:
                    im.crop((0, k * H, 3 * W, (k + 1) * H)).save(OUT / "img" / f"{c['key']}.png")
            print("rendered", i + len(part), "of", len(cl), "failed", sorted(failed), flush=True)
        b.close()


def images(cl=None):
    from PIL import Image
    return [Image.open(OUT / "img" / f"{c['key']}.png").convert("RGB") for c in (cl or chars())]


_clip = {}


def clip_model():
    """loaded once per process (the generator page calls text_embed for every prompt)"""
    if not _clip:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _clip["m"] = (CLIPModel.from_pretrained(CLIP_NAME).eval().to(dev), CLIPProcessor.from_pretrained(CLIP_NAME), torch, dev)
    return _clip["m"]


def cmd_clip(a):
    m, proc, torch, dev = clip_model()
    cl = chars()                                               # fixed once: a render in progress keeps adding portraits
    ims = images(cl)
    W = ims[0].width // 3
    emb = []
    for part in range(3):                                      # one embedding per view; the mean is the character's
        views = [im.crop((part * W, 0, (part + 1) * W, im.height)) for im in ims]
        with torch.no_grad():
            e = []
            for i in range(0, len(views), 32):
                x = proc(images=views[i:i + 32], return_tensors="pt").to(dev)
                e.append(torch.nn.functional.normalize(m.visual_projection(m.vision_model(pixel_values=x['pixel_values']).pooler_output), dim=-1).cpu())
        emb.append(torch.cat(e).numpy())
    np.savez(OUT / "clip_image.npz", views=np.stack(emb, 1), keys=np.array([c["key"] for c in cl]))
    print("saved", OUT / "clip_image.npz", np.stack(emb, 1).shape)


def cmd_caption(a):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = ROOT / "sources" / "models" / f"Qwen3-VL-{a.vlm}-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float16 if a.vlm == "2B" else torch.bfloat16, attn_implementation="eager").to(dev).eval()
    caps, t0 = [], time.time()
    for i, im in enumerate(images()):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": CAPTION_PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=60, do_sample=False)
        caps.append(proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().split("\n")[0])
        if i % 10 == 0:
            print(i, f"{time.time() - t0:.0f}s", caps[-1], flush=True)
    (OUT / f"captions_{a.vlm}.json").write_text(json.dumps([dict(key=c["key"], caption=t) for c, t in zip(chars(), caps)], indent=1, ensure_ascii=False))
    print(f"{len(caps)} captions, {(time.time() - t0) / len(caps):.1f} s each")


def text_embed(texts):
    m, proc, torch, dev = clip_model()
    x = proc(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
    with torch.no_grad():
        return torch.nn.functional.normalize(m.text_projection(m.text_model(**x).pooler_output), dim=-1).cpu().numpy()


def indexes(vlm):
    """name -> (n_chars, 512) unit vectors, comparable with a CLIP text embedding of a prompt"""
    out = {}
    z = np.load(OUT / "clip_image.npz")
    v = z["views"]
    out["image(mean of 3 views)"] = (v.mean(1) / np.linalg.norm(v.mean(1), axis=1, keepdims=True))
    out["image(front)"], out["image(side)"], out["image(3/4)"] = v[:, 0], v[:, 1], v[:, 2]
    f = OUT / f"captions_{vlm}.json"
    if f.exists():
        out[f"caption(Qwen3-VL-{vlm})"] = text_embed([c["caption"] for c in json.loads(f.read_text())])
        both = out["image(mean of 3 views)"] + out[f"caption(Qwen3-VL-{vlm})"]
        out["image + caption"] = both / np.linalg.norm(both, axis=1, keepdims=True)
    return out


def cmd_eval(a):
    cl = chars()
    cat = np.array([c["category"] for c in cl])
    idx = indexes(a.vlm)
    prompts = [(c, q) for c, qs in QUERIES.items() for q in qs]
    Q = text_embed([q for _, q in prompts])
    print(f"{len(cl)} characters: " + ", ".join(f"{k} {int((cat == k).sum())}" for k in QUERIES))
    for name, X in idx.items():
        S = Q @ X.T
        p5, hit1 = [], []
        for (c, q), s in zip(prompts, S):
            top = np.argsort(-s)[:5]
            p5.append(float((cat[top] == c).mean()))
            hit1.append(float(cat[top[0]] == c))
        by = {}
        for (c, q), v in zip(prompts, p5):
            by.setdefault(c, []).append(v)
        print(f"{name:28s} precision@5 {100 * np.mean(p5):3.0f}%  top-1 {100 * np.mean(hit1):3.0f}%   " + "  ".join(f"{c} {100 * np.mean(v):.0f}" for c, v in by.items()), flush=True)
    # random baseline: share of each category weighted like the queries
    print("chance precision@5 about", f"{100 * np.mean([np.mean(cat == c) for c, _ in prompts]):.0f}%")
    if a.sheets:
        from PIL import Image, ImageDraw
        name = a.sheets
        X = idx[name]
        pick = [("", q) for q in a.sheet_prompts] if a.sheet_prompts else ([(c, q) for c, q in prompts][::2] if a.sheets_n == 0 else prompts[:a.sheets_n])
        Wt, Ht = 384, 128
        sheet = Image.new("RGB", (Wt * 5, Ht * len(pick)), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        Qp = text_embed([q for _, q in pick])
        for r, ((c, q), s) in enumerate(zip(pick, Qp @ X.T)):
            for k, j in enumerate(np.argsort(-s)[:5]):
                im = Image.open(OUT / "img" / f"{cl[j]['key']}.png").convert("RGB").resize((Wt, Ht))
                sheet.paste(im, (k * Wt, r * Ht))
                d.text((k * Wt + 4, r * Ht + 2), f"{q} | {cl[j]['category']}", fill=(255, 220, 120))
        out = ROOT / "docs" / "figures" / a.sheet_out
        sheet.save(out)
        print("saved", out)


def cmd_query(a):
    cl = chars()
    idx = indexes(a.vlm)
    q = text_embed([a.prompt])[0]
    for name, X in idx.items():
        top = np.argsort(-(X @ q))[:a.top]
        print(f"{name}: " + ", ".join(f"{cl[j]['key'][:24]} ({cl[j]['category']})" for j in top))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "render", "clip", "caption", "eval", "query"])
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vlm", default="2B")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--sheets", help="eval: save a contact sheet of the top hits for this index name")
    ap.add_argument("--sheets-n", type=int, default=0)
    ap.add_argument("--sheet-prompts", nargs="+", help="eval --sheets: use these prompts for the contact sheet")
    ap.add_argument("--sheet-out", default="shape_retrieval_pilot.png")
    ap.add_argument("--set", choices=["pilot", "all"], default="pilot", help="pilot: the stratified sample of 150; all: every normalized character (data/shape/all)")
    a = ap.parse_args()
    global OUT
    OUT = ROOT / "data" / "shape" / a.set
    dict(sample=cmd_sample, render=cmd_render, clip=cmd_clip, caption=cmd_caption, eval=cmd_eval, query=cmd_query)[a.cmd](a)


if __name__ == "__main__":
    main()
