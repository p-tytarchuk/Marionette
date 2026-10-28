"""Filter LottieSVG-small down to single, full-body, side-view animals: the candidate
artwork for a flat-style 2D mesh (outline -> triangles -> bone weights, build_mesh.py).

Source: ../Datasets/LottieSVG-small, the VLM-filtered "living" subset of LottieSVG-10M
(CC BY-NC-SA 4.0, research use only; the output is derived data, gitignored). Uses the
corrected manifest_v2.jsonl (see ../I2V/README.md) and starts from its 9,440
subtype == "animal" rows.

Three stages, each resumable:

  extract   (conda env loomie: needs zstandard)   animal rows -> data/side_view_animals/
            index.jsonl + images/<id>.png + svg/<id>.svg, with cheap SVG statistics (filled
            vs stroke-only shapes, path count, aspect ratio) and a species group from the
            caption / tags.
  classify  (conda env Mario: needs torch + transformers)   Qwen3-VL-2B-Instruct (../Models)
            looks at each PNG that passes the SVG-statistics gate (cheap_ok) and answers a JSON form: how many animals, whole body in
            frame, view (side / front / back / top / three_quarter), legs count, whether it
            is a flat colour illustration or a line icon. -> vlm.jsonl (greedy decoding).
  select    (any env)   index + vlm -> selected.jsonl with the rules in SELECT below.
  verify    (conda env Mario)   stricter second pass over selected.jsonl only (a few hundred images, so it
            can use a bigger image size): focused questions (body plan, strict profile, which way the head
            points, attached to another object) plus the silhouette's bounding-box aspect measured on the PNG.
            -> verify.jsonl, strict.jsonl (the rig candidates). The first pass is lenient: it let through
            three-quarter dogs, a koala clinging to a tree, snails. It also requires a standing / walking pose and
            a natural (not toy / robot / letter-shaped) animal.

    python tools/filter_side_view_animals.py extract
    python tools/filter_side_view_animals.py classify [--limit 200] [--device mps]
    python tools/filter_side_view_animals.py select
    python tools/filter_side_view_animals.py verify [--body-plan quadruped|any]
    python tools/filter_side_view_animals.py sheet [--source strict|selected]     # contact sheets in the data dir
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT.parent / "Datasets" / "LottieSVG-small"
DEFAULT_MODEL = ROOT.parent / "Models" / "Qwen3-VL-2B-Instruct"
DEFAULT_OUT = ROOT / "data" / "side_view_animals"

# species group from caption + tags: first group with a hit wins, so the order matters
GROUPS = {
    "bird": ["bird", "birds", "parrot", "owl", "eagle", "duck", "chicken", "hen", "rooster", "penguin", "flamingo", "swan", "goose", "pigeon", "dove", "peacock", "toucan", "sparrow", "crow"],
    "fish": ["fish", "shark", "whale", "dolphin", "octopus", "jellyfish", "seahorse", "crab", "turtle", "tortoise"],
    "insect": ["butterfly", "bee", "ant", "spider", "ladybug", "beetle", "snail", "dragonfly", "insect", "bug", "worm", "caterpillar"],
    "reptile": ["snake", "lizard", "crocodile", "alligator", "frog", "gecko", "chameleon", "dragon", "dinosaur", "dino"],
    "quadruped": ["cat", "kitten", "kitty", "dog", "puppy", "pup", "horse", "pony", "foal", "unicorn", "donkey", "zebra", "fox", "wolf", "lion", "lioness", "tiger", "leopard", "cheetah",
                  "panther", "jaguar", "cow", "bull", "calf", "ox", "deer", "reindeer", "elk", "moose", "goat", "sheep", "lamb", "pig", "piglet", "boar", "bear", "panda", "elephant", "giraffe",
                  "camel", "rhino", "hippo", "hamster", "mouse", "rat", "rabbit", "bunny", "hare", "raccoon", "squirrel", "hedgehog", "llama", "alpaca", "koala", "sloth", "monkey", "kangaroo",
                  "dachshund", "terrier", "retriever", "bulldog", "poodle", "husky", "corgi", "labrador", "beagle", "pug"],
}
_GROUP_RE = {g: re.compile(r"\b(" + "|".join(w) + r")s?\b", re.I) for g, w in GROUPS.items()}

# what the VLM is asked; one image, a JSON answer
PROMPT = (
    "This is a vector illustration. Answer with one JSON object and nothing else, with exactly these keys:\n"
    '"animal_count": number of animals drawn (0 if none, 1 for a single animal),\n'
    '"full_body": true if the whole animal from head to feet is visible and not cropped, else false,\n'
    '"view": the viewing direction of the animal\'s body, one of "side", "front", "back", "top", "three_quarter", "none",\n'
    '"legs_visible": number of legs visible (0-8),\n'
    '"style": "flat" if it is filled colour shapes, "line" if it is mostly outline strokes, "icon" if it is a small symbol or logo,\n'
    '"background": true if there is a scene, ground, text or other objects around the animal, else false.'
)

# select: a candidate is kept when the VLM says
SELECT = dict(animal_count=1, full_body=True, view="side", style="flat", background=False)
# verify: second, focused pass
VERIFY_PROMPT = (
    "This is a vector illustration of one animal. Answer with one JSON object and nothing else, with exactly these keys:\n"
    '"species": the animal\'s name in one or two words,\n'
    '"body_plan": one of "quadruped" (walks on four legs), "biped", "bird", "fish", "insect", "snake", "other",\n'
    '"profile": true only if the animal is in strict side profile: its body is stretched horizontally across the image, the head is at one end of the body and seen from the side, '
    "and the legs hang below the body. false if the animal faces the viewer, is seen from above or from behind, is a head or portrait, is curled up, or is turned three-quarters toward the viewer,\n"
    '"head_side": "left" or "right" (the side of the image the head is on), or "viewer" if the head faces the viewer,\n'
    '"attached": true if the animal touches, holds, climbs, rides or sits on another object (tree, branch, person, syringe, badge, ground scene), else false,\n'
    '"pose": one of "standing" (all four feet on the ground: standing, walking, trotting or running), "sitting", "lying", "rearing" (up on the hind legs), "jumping", "other",\n'
    '"natural": true if it is a real animal (cartoon animals are fine); false if it is a toy, a robot, a machine, a letter or number shape, or an abstract design.'
)
STRICT = dict(body_plan="quadruped", profile=True, attached=False, pose="standing", natural=True)
HEAD_SIDES = ("left", "right")
NOT_QUADRUPED_GROUPS = ("bird", "fish", "insect")   # caption / tag group; the 2B model calls an ostrich a quadruped, the caption does not
MIN_ASPECT, MAX_ASPECT = 0.8, 3.0      # width / height of the silhouette's bounding box: side-view quadrupeds are about as wide as tall or wider
MIN_FILLED_SHAPES = 5      # a flat mesh needs several filled parts, not one blob
MAX_SVG_BYTES = 200_000    # megabytes of paths are not a character template


def cheap_ok(r: dict) -> bool:
    """SVG-statistics gate, applied before the VLM (it costs about 2 s per image) and again in select."""
    return r["svg_bytes"] <= MAX_SVG_BYTES and r["filled"] >= MIN_FILLED_SHAPES


def iter_shard(path: Path):
    with open(path, "rb") as fh:
        import zstandard
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        try:
            for line in reader:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except zstandard.ZstdError:
            return


def species_group(text: str) -> str:
    for g, rx in _GROUP_RE.items():
        if rx.search(text):
            return g
    return "other"


def svg_stats(svg: str) -> dict:
    """Cheap, regex-level facts about the SVG: enough to drop line icons and blobs before the VLM sees them."""
    shapes = re.findall(r"<(path|circle|ellipse|rect|polygon|polyline|line)\b([^>]*)>", svg, re.I)
    filled = stroke_only = 0
    for _, attrs in shapes:
        style = re.search(r'style="([^"]*)"', attrs)
        s = (style.group(1) if style else "") + " " + attrs
        fill = re.search(r"fill\s*[:=]\s*\"?([^;\"\s]+)", s)
        has_fill = not fill or fill.group(1).lower() not in ("none", "transparent")
        if has_fill and not fill and "stroke" in s:
            has_fill = False                      # no fill attribute but a stroke: an outline (the SVG default fill is black, but icon sets set stroke on purpose)
        filled += has_fill
        stroke_only += not has_fill
    vb = re.search(r'viewBox="([-\d.\s,eE]+)"', svg)
    aspect = None
    if vb:
        p = [float(x) for x in re.split(r"[\s,]+", vb.group(1).strip()) if x]
        if len(p) == 4 and p[3] > 0:
            aspect = round(p[2] / p[3], 3)
    return dict(shapes=len(shapes), filled=filled, stroke_only=stroke_only, aspect=aspect, svg_bytes=len(svg.encode()))


def extract(args) -> None:
    ds, out = args.dataset, args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(exist_ok=True)
    manifest = {}
    for line in open(ds / "manifest_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["subtype"] == "animal":
            manifest[r["id"]] = r
    print(f"{len(manifest)} animal rows in manifest_v2", flush=True)
    rows, seen = [], set()
    for shard in sorted((ds / "data" / "living").glob("metadata-*.jsonl.zst")):
        if shard.name.startswith("._"):
            continue
        tar_name = shard.name.replace("metadata", "images").replace(".jsonl.zst", ".tar")
        want = {}
        for rec in iter_shard(shard):
            rid = str(rec["id"])
            if rid in manifest and rid not in seen:
                want[manifest[rid]["media_key"]] = rec
        if not want:
            continue
        with tarfile.open(ds / "data" / "living" / tar_name) as tf:
            for m in tf:
                rec = want.get(m.name)
                if rec is None:
                    continue
                rid = str(rec["id"])
                seen.add(rid)
                svg = rec.get("svg_code") or ""
                (out / "images" / f"{rid}.png").write_bytes(tf.extractfile(m).read())
                (out / "svg" / f"{rid}.svg").write_text(svg, encoding="utf-8")
                tags = rec.get("tag") or []
                text = f"{rec.get('caption', '')} {' '.join(tags) if isinstance(tags, list) else tags}"
                rows.append(dict(id=rid, caption=rec.get("caption"), tags=tags, style_tag=rec.get("style"), name=rec.get("svg_filename"), group=species_group(text),
                                 width=rec.get("width"), height=rec.get("height"), **svg_stats(svg)))
        print(f"{shard.name}: {len(rows)} so far", flush=True)
    with open(out / "index.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out / 'index.jsonl'} (manifest had {len(manifest)}; {len(manifest) - len(rows)} not found in the shards)")


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def flat_rgb(path: Path):
    """The PNG composited on white (the icons are transparent), as RGB."""
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return img.convert("RGB")


def load_vlm(args, max_pixels):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    proc = AutoProcessor.from_pretrained(str(args.model), min_pixels=128 * 28 * 28, max_pixels=max_pixels)
    # eager attention: fp16 SDPA fails on MPS for this model ("Failed to create function state object ... prefill_attention")
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model), torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    return proc, model, device


def ask(vlm, img, prompt: str) -> str:
    import torch
    proc, model, device = vlm
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def run_vlm(args, ids, prompt, path: Path, max_pixels: int) -> None:
    """Ask `prompt` of every image in `ids` not yet in `path` (jsonl: id, parsed, raw); resumable, flushed per image."""
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = [i for i in ids if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    vlm = load_vlm(args, max_pixels)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fh:
        for n, rid in enumerate(todo, 1):
            raw = ask(vlm, flat_rgb(args.out / "images" / f"{rid}.png"), prompt)
            fh.write(json.dumps(dict(id=rid, parsed=parse_json(raw), raw=raw), ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"{n}/{len(todo)}  {(time.time() - t0) / n:.2f} s/image", flush=True)


def classify(args) -> None:
    ids = [r["id"] for r in map(json.loads, open(args.out / "index.jsonl", encoding="utf-8")) if cheap_ok(r)]
    run_vlm(args, ids, PROMPT, args.out / "vlm.jsonl", args.max_pixels)


def silhouette(path: Path) -> dict:
    """Bounding box of the drawn pixels (alpha > 16, or not near-white when the PNG is opaque): width / height and the share of the box that is drawn."""
    import numpy as np
    from PIL import Image
    img = Image.open(path)
    a = np.asarray(img.convert("RGBA"))
    mask = a[..., 3] > 16
    if mask.all():                                  # opaque PNG: the white page is background
        mask = (a[..., :3] < 240).any(-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return dict(sil_aspect=None, sil_fill=None)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return dict(sil_aspect=round(float(w / h), 3), sil_fill=round(float(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean()), 3))


def verify(args) -> None:
    out = args.out
    cand = [json.loads(l) for l in open(out / "selected.jsonl", encoding="utf-8")]
    run_vlm(args, [r["id"] for r in cand], VERIFY_PROMPT, out / "verify.jsonl", args.verify_max_pixels)
    ver = {}
    for l in open(out / "verify.jsonl", encoding="utf-8"):
        r = json.loads(l)
        ver[r["id"]] = r
    body_plan = None if args.body_plan == "any" else args.body_plan
    kept, why = [], {}

    def drop(reason):
        why[reason] = why.get(reason, 0) + 1

    for r in cand:
        v = ver.get(r["id"])
        if v is None:
            drop("not verified yet")
            continue
        p = v["parsed"]
        if not p:
            drop("unparsable VLM answer")
            continue
        sil = silhouette(out / "images" / f"{r['id']}.png")
        a = sil["sil_aspect"]
        if body_plan and (p.get("body_plan") != body_plan or (body_plan == "quadruped" and r["group"] in NOT_QUADRUPED_GROUPS)):
            drop(f"body plan is not {body_plan}")
        elif p.get("profile") is not STRICT["profile"]:
            drop("not strict profile")
        elif p.get("head_side") not in HEAD_SIDES:
            drop("head faces the viewer")
        elif p.get("attached") is not STRICT["attached"]:
            drop("attached to another object")
        elif p.get("pose") != STRICT["pose"]:
            drop("not a standing / walking pose")
        elif p.get("natural") is not STRICT["natural"]:
            drop("toy, robot, letter or abstract shape")
        elif a is None or not MIN_ASPECT <= a <= MAX_ASPECT:
            drop(f"silhouette aspect outside {MIN_ASPECT}-{MAX_ASPECT}")
        else:
            kept.append(dict(r, verify=p, **sil))
    with open(out / "strict.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(kept)} of {len(cand)} candidates kept -> {out / 'strict.jsonl'}")
    print("species:", sorted({r["verify"].get("species", "?") for r in kept}))
    print("dropped:", dict(sorted(why.items(), key=lambda kv: -kv[1])))


def select(args) -> None:
    out = args.out
    idx = {}
    for l in open(out / "index.jsonl", encoding="utf-8"):
        r = json.loads(l)
        idx[r["id"]] = r
    vlm = {}
    for l in open(out / "vlm.jsonl", encoding="utf-8"):
        r = json.loads(l)
        vlm[r["id"]] = r
    kept, why = [], {}

    def drop(reason):
        why[reason] = why.get(reason, 0) + 1

    for rid, r in idx.items():
        if not cheap_ok(r):
            drop("svg too large" if r["svg_bytes"] > MAX_SVG_BYTES else "fewer than %d filled shapes" % MIN_FILLED_SHAPES)
            continue
        v = vlm.get(rid)
        if v is None:
            drop("not classified yet")
            continue
        p = v["parsed"]
        if not p:
            drop("unparsable VLM answer")
            continue
        if p.get("animal_count") != SELECT["animal_count"]:
            drop("not exactly one animal")
        elif p.get("full_body") is not SELECT["full_body"]:
            drop("cropped / partial body")
        elif p.get("view") != SELECT["view"]:
            drop("view is not side")
        elif p.get("style") != SELECT["style"]:
            drop("not flat colour")
        elif p.get("background") is not SELECT["background"]:
            drop("scene / text around the animal")
        else:
            kept.append(dict(r, vlm=p))
    with open(out / "selected.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    groups = {}
    for r in kept:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    print(f"{len(kept)} of {len(idx)} kept -> {out / 'selected.jsonl'}")
    print("species groups:", dict(sorted(groups.items(), key=lambda kv: -kv[1])))
    print("dropped:", dict(sorted(why.items(), key=lambda kv: -kv[1])))


def sheet(args) -> None:
    """Contact sheets of strict.jsonl (or selected.jsonl with --from selected): sheet_<n>.png, 6 x 5 animals per page, labelled with id, species and head side."""
    from PIL import Image, ImageDraw
    out = args.out
    src = out / ("strict.jsonl" if args.source == "strict" else "selected.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    W, cols, per = 220, 6, 30
    for page in range(0, max(1, len(rows)), per):
        chunk = rows[page:page + per]
        n_rows = (len(chunk) + cols - 1) // cols
        im = Image.new("RGB", (cols * W, max(1, n_rows) * (W + 24)), "white")
        d = ImageDraw.Draw(im)
        for i, r in enumerate(chunk):
            x, y = (i % cols) * W, (i // cols) * (W + 24)
            t = flat_rgb(out / "images" / f"{r['id']}.png")
            t.thumbnail((W - 6, W - 6))
            im.paste(t, (x + 3, y + 3))
            v = r.get("verify") or {}
            d.text((x + 4, y + W), f"{r['id']} {v.get('species', r['group'])}", fill="black")
            d.text((x + 4, y + W + 11), f"{r['group']} head {v.get('head_side', '?')} filled {r['filled']}", fill="black")
        path = out / f"sheet_{args.source}_{page // per + 1}.png"
        im.save(path)
        print(path)
    print(f"{len(rows)} animals on {(len(rows) + per - 1) // per} page(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["extract", "classify", "select", "verify", "sheet"])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="classify / verify: stop after N images (0 = all)")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28, help="classify: image resolution cap for the VLM")
    ap.add_argument("--source", default="strict", choices=["strict", "selected"], help="sheet: which list to draw")
    ap.add_argument("--verify-max-pixels", type=int, default=1024 * 28 * 28, help="verify: image resolution cap for the VLM")
    ap.add_argument("--body-plan", default="quadruped", choices=["quadruped", "any"], help="verify: required body plan (the 4leg template needs quadruped)")
    args = ap.parse_args()
    {"extract": extract, "classify": classify, "select": select, "verify": verify, "sheet": sheet}[args.stage](args)


if __name__ == "__main__":
    main()
