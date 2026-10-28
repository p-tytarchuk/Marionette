#!/usr/bin/env python3
"""Pilot: does Qwen3-VL-2B caption motion better from MESH renders than from stick figures?  (decision experiment before captioning all clips)

  /usr/local/bin/python3 scripts/captioning/qwen_mesh_pilot.py --render --out DIR [--n 60] [--set mixamo|quad]   # 4x2 grids of mesh frames of N clips (needs playwright + viewer/serve.py running)
  python3 scripts/captioning/qwen_mesh_pilot.py --caption --out DIR [--prompts mesh,mesh_plain,stick]            # Qwen3-VL-2B captions + retrieval score against the dataset captions (conda env Mario)
  (--set quad: usable quadruped clips, each on its own normalized character; prompts mesh_quad,mesh_quad_plain)

The clips: N random usable Mixamo clips that have a dataset caption, one per distinct caption (seed 0), played on Aciana by viewer/strip.html (side camera, 8 evenly spaced frames, same scale).
Score (as in qwen_caption.py --eval): CLIP embedding of the Qwen caption against the N true captions; R@k = the true caption of the clip is among the k nearest (chance k/N), median rank;
and the action hit rate: the caption names the same main action (flip, kick, punch, wave, jump, sit, dance, crawl, climb, throw, walk, run, crouch, spin, roll, fall, bow, squat, stretch, turn, kneel, lie)
as the dataset caption, over the clips whose dataset caption names one.
"""
import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
ACTIONS = ["flip", "kick", "punch", "wave", "jump", "sit", "danc", "crawl", "climb", "throw", "walk", "run", "crouch", "spin", "roll", "fall", "bow", "squat", "stretch", "turn", "kneel", "lie"]
ACTIONS_Q = ["walk", "run", "jump", "hop", "head", "tail", "rear", "attack", "crouch", "lower", "raise", "turn", "sit", "lie", "bite", "stand", "sway", "fall", "roll", "spin", "kick", "swim", "fly", "shake", "sniff"]
TARGET, ROOT_SCALE = "mixamo_Aciana", 1.004162
FORMAT = ("Describe only the motion, not the body, and call the character 'object'. Answer with exactly two lines:\n"
          "SHORT: one sentence of at most 10 words starting with 'An object'.\n"
          "DETAIL: one sentence of 15 to 30 words starting with 'An object'.")
EXAMPLES = ("Examples of the style:\nSHORT: An object performs a backflip and lands in a crouch.\n"
            "DETAIL: An object swings its arms back and crouches, backflips, lands in a crouch, then straightens up.\n"
            "SHORT: An object walks forward, then turns around.\nSHORT: An object stands and nods its head.\n")
MESH_Q = ("The image shows 8 frames of a 3D four-legged animal character performing an action, seen from the side, in time order (left to right, then the top row before the bottom row). "
          "The clip lasts {dur:.1f} seconds. Look at the legs, the head, the tail and the height of the body from frame to frame. ")
EXAMPLES_Q = ("Examples of the style:\nSHORT: An object walks in place.\nSHORT: An object lowers its head and then raises it.\n"
              "SHORT: An object rears onto its hind legs and attacks forward.\nSHORT: An object crouches down and springs upward.\nSHORT: An object runs forward, then stops and turns its head.\n")
MESH = ("The image shows 8 frames of a 3D character performing an action, seen from the side, in time order (left to right, then the top row before the bottom row). "
        "The clip lasts {dur:.1f} seconds. Look at how the pose and the height of the body change from frame to frame. ")


def clips(n, which="mixamo"):
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "inventory" / "clips.csv", newline="")) if r["usable"] == "1" and r["caption"]]
    if which == "mixamo":
        rows = [r for r in rows if r["dataset"] == "mixamo" and (ROOT / "data" / "clips" / "mixamo" / f"{r['clip']}.glb").exists()]
    else:
        out = {r["clip"]: r["out"] for r in csv.DictReader(open(ROOT / "data" / "clips_report.csv", newline="")) if r["status"] == "ok"}
        rows = [dict(r, out=out[r["clip"]]) for r in rows if r["character_class"] == "quadruped" and r["clip"] in out and (ROOT / out[r["clip"]]).exists()]
    rnd = random.Random(0)
    rnd.shuffle(rows)
    res, seen = [], set()
    for r in rows:
        if r["caption"] in seen:
            continue
        seen.add(r["caption"])
        c = dict(clip=r["clip"], caption=r["caption"], dur=float(r["duration_s"]))
        if which == "quad":
            c["target"] = Path(r["out"]).parts[2]
            c["anim"] = Path(r["out"]).stem
        res.append(c)
        if len(res) == n:
            break
    return res


def cmd_render(a):
    from PIL import Image
    from playwright.sync_api import sync_playwright
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cl = clips(a.n, a.set)
    (out / "clips.json").write_text(json.dumps(cl, indent=1))
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 2000, "height": 400})
        for i, c in enumerate(cl):
            if a.set == "quad":
                rows, W, H, span = f"{c['target']}|{c['anim']}", 330, 260, 2.0
            else:
                rows, W, H, span = f"{TARGET}|/data/clips/mixamo/{c['clip']}.glb|{ROOT_SCALE}", 250, 330, 1.9
            pg.goto(f"http://127.0.0.1:8765/viewer/strip.html?rows={urllib.parse.quote(rows)}&n=8&w={W}&h={H}&span={span}")
            pg.wait_for_function("window.__ready === true", timeout=120000)
            pg.locator("canvas").screenshot(path=str(out / "_strip.png"))
            im = Image.open(out / "_strip.png").convert("RGB")
            grid = Image.new("RGB", (W * 4, H * 2))
            for k in range(8):
                grid.paste(im.crop((k * W, 0, (k + 1) * W, H)), ((k % 4) * W, (k // 4) * H))
            grid.save(out / f"{i:03d}.png")
            if i % 10 == 0:
                print("rendered", i, c["clip"], flush=True)
        b.close()


def parse(raw):
    g = lambda tag: (re.search(rf"{tag}:\s*(.+)", raw) or [None, ""])[1].strip().strip("*").strip()
    short, detail = g("SHORT"), g("DETAIL")
    if not short:                                   # the 8B model follows the format but drops the "SHORT:" / "DETAIL:" labels: first line = short, second = detail
        lines = [l.strip().strip("*").strip() for l in raw.splitlines() if l.strip()]
        short, detail = (lines[0] if lines else ""), (lines[1] if len(lines) > 1 else "")
    return short, detail


def cmd_caption(a):
    import numpy as np
    from PIL import Image
    import qwen_caption as QC
    from embed import encode, load_text_encoder
    out = Path(a.out)
    cl = json.loads((out / "clips.json").read_text())
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = ROOT / "sources" / "models" / f"Qwen3-VL-{a.vlm}-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float16 if a.vlm == "2B" else torch.bfloat16, attn_implementation="eager").to(dev).eval()
    tenc = load_text_encoder()
    truth, _ = encode([c["caption"] for c in cl], tenc)
    prompts = {"mesh": MESH + QC.FORMAT + "\n" + EXAMPLES, "mesh_plain": MESH + FORMAT, "mesh_quad": MESH_Q + FORMAT + "\n" + EXAMPLES_Q, "mesh_quad_plain": MESH_Q + FORMAT}
    results = {}
    for pname in a.prompts.split(","):
        shorts, raws, t0 = [], [], time.time()
        for i, c in enumerate(cl):
            if pname == "stick":
                import render_motion as RM
                img = RM.render(RM.load_for_render(c["clip"]))
                text = QC.PROMPTS["v3"].format(dur=c["dur"])
            else:
                img = Image.open(out / f"{i:03d}.png").convert("RGB")
                text = prompts[pname].format(dur=c["dur"])
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=100, do_sample=False)
            raw = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            s, d = parse(raw)
            raws.append(raw)
            shorts.append(s or d or "An object moves.")
        E, _ = encode(shorts, tenc)
        sim = E @ truth.T
        rank = (sim > np.diag(sim)[:, None]).sum(1)
        hits = []
        for c, s in zip(cl, shorts):
            need = {x for x in (ACTIONS_Q if "target" in c else ACTIONS) if x in c["caption"].lower()}
            if need:
                hits.append(any(x in s.lower() for x in need))
        n = len(cl)
        results[pname] = dict(sec=(time.time() - t0) / n, r1=float((rank < 1).mean()), r5=float((rank < 5).mean()), r10=float((rank < 10).mean()), med=float(np.median(rank)),
                              cos=float(np.diag(sim).mean()), rand=float(sim.mean()), hit=float(np.mean(hits)), nhit=len(hits), shorts=shorts, raws=raws)
        r = results[pname]
        print(f"\n{pname}: {r['sec']:.1f} s/clip | diag cos {r['cos']:.3f} (random pair {r['rand']:.3f}) | R@1 {100 * r['r1']:.0f}% R@5 {100 * r['r5']:.0f}% (chance {500 / n:.0f}%) R@10 {100 * r['r10']:.0f}% median rank {r['med']:.0f} of {n} (chance {n / 2:.0f}) | "
              f"names the same action: {100 * r['hit']:.0f}% of {r['nhit']}", flush=True)
        for i in range(0, n, max(1, n // 8)):
            print(f"   [{cl[i]['clip'][:26]}] Q: {shorts[i]} | T: {cl[i]['caption']}")
    (out / f"results{a.tag}.json").write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--caption", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--set", choices=["mixamo", "quad"], default="mixamo")
    ap.add_argument("--vlm", choices=["2B", "4B", "8B"], default="2B", help="Qwen3-VL size (sources/models/Qwen3-VL-<size>-Instruct; 2B in fp16, larger in bfloat16)")
    ap.add_argument("--tag", default="", help="suffix of the results file, e.g. _8B")
    ap.add_argument("--prompts", default="mesh,mesh_plain,stick")
    a = ap.parse_args()
    if a.render:
        cmd_render(a)
    elif a.caption:
        cmd_caption(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
