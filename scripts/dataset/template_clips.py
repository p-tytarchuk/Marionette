#!/usr/bin/env python3
"""Non-human clips from the 8 template rigs (sources/templates_skeleton_only/*.glb) as extra training data for the semantic
motion embedding, with captions written from the clip names.

  python3 scripts/dataset/template_clips.py --text     # embed the captions and append them to data/embeddings/text.npy / text_index.csv
  python3 scripts/dataset/template_clips.py --list     # what would be added

65 clips: bird 4, dragon 4, fish 6, fox 13, horse 13, kaiju 9, snake 7, spider 9 (the human rig's 178 clips are not used: bipeds are
well covered, and the Rest_Pose clip of every rig is dropped). The point is body variety: a spider with eight legs, a snake, a fish, a
winged dragon. They are tiny sets (the spider has 9 clips), so they add examples of other bodies, they do not make a dataset.
Captions are derived from the action names ("Walk" -> "An object walks forward.") in the style of the dataset's own captions. They
were NOT checked against the motion, so a few may be loose (e.g. "Fetch", "Eating"); that is a limitation, noted in docs/EMBED.md.
motion_sem_features.py reads the clips through read_clips() below; they are listed as dataset "template".
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
SRC = ROOT / "sources" / "templates_skeleton_only"
FPS = 30

CLASS = {"bird": "avian", "dragon": "uncertain", "fish": "marine", "fox": "quadruped", "horse": "quadruped",
         "kaiju": "uncertain", "snake": "serpentine", "spider": "insectoid"}
CAPTIONS = {
    "bird": {"Flap": "An object flaps its wings.", "Glide": "An object glides with its wings spread.", "Idle": "An object stands still and idles.", "Walk": "An object walks forward."},
    "dragon": {"Fly_Flap": "An object flaps its wings to fly.", "Fly_Glide": "An object glides with its wings spread.", "Idle": "An object stands and idles.", "Walk": "An object walks forward."},
    "fish": {"Bite": "An object lunges and bites.", "Dead_Floating": "An object floats motionless.", "Hit": "An object is hit and recoils.",
             "Idle": "An object hovers in place, gently moving its fins.", "Swim_Horizontal": "An object swims forward.", "Swim_Vertical": "An object swims up and down."},
    "fox": {"Bark": "An object barks.", "Bite": "An object lunges and bites.", "Death": "An object collapses and dies.", "Fall": "An object falls.",
            "Fetch": "An object fetches something.", "Howl": "An object raises its head and howls.", "Idle": "An object stands idle.",
            "Idle_Alert": "An object stands alert and looks around.", "Jump": "An object jumps.", "Run": "An object runs forward.", "Sit": "An object sits down.",
            "Sneak": "An object sneaks forward, low to the ground.", "Walk": "An object walks forward."},
    "horse": {"Death": "An object collapses and dies.", "Eating": "An object lowers its head and eats.", "Head_But": "An object butts its head forward.",
              "Idle": "An object stands idle.", "Kick": "An object kicks back with its hind legs.", "Rear": "An object rears up on its hind legs.",
              "Run": "An object runs forward.", "Sleep": "An object sleeps.", "Trot": "An object trots forward.", "Turn_Left": "An object turns to the left.",
              "Turn_Right": "An object turns to the right.", "Walk": "An object walks forward.", "lay_to_idle": "An object gets up from lying down and stands."},
    "kaiju": {"Attack": "An object attacks forward.", "Death": "An object collapses and dies.", "Hit": "An object is hit and recoils.", "Idle": "An object stands idle.",
              "Roar": "An object roars.", "Swim": "An object swims.", "Tail_Attack": "An object swings its tail to attack.", "Walk": "An object walks.", "Walk_RM": "An object walks forward."},
    "snake": {"Bite": "An object lunges and bites.", "Coiled": "An object stays coiled.", "Dance": "An object sways and dances.", "Death": "An object collapses and dies.",
              "Hit": "An object is hit and recoils.", "Idle": "An object idles.", "Side_winding": "An object slithers sideways."},
    "spider": {"Attack": "An object attacks forward.", "Bite": "An object lunges and bites.", "Death": "An object dies.", "Death_2": "An object dies.",
               "Eating": "An object eats.", "Hit": "An object is hit and recoils.", "Idle": "An object stands idle.", "Jump": "An object jumps.", "Walk": "An object walks forward."},
}


def _interp(t, v, grid, rotation, step):
    """Sample a keyframe track (times t, values v) on a time grid: linear (normalised lerp for quaternions) or hold."""
    i1 = np.clip(np.searchsorted(t, grid, side="right"), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(grid), int)
    i0 = np.maximum(i1 - 1, 0)
    if len(t) == 1:
        return np.repeat(v[:1], len(grid), 0)
    if step:
        return v[i0]
    w = np.clip((grid - t[i0]) / np.maximum(t[i1] - t[i0], 1e-9), 0, 1)[:, None]
    a, b = v[i0], v[i1]
    if rotation:
        b = b * np.where((a * b).sum(1, keepdims=True) < 0, -1.0, 1.0)
    out = a * (1 - w) + b * w
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9) if rotation else out


def read_clips():
    """-> list of dict(rig, action, clip, parents, names, rest_p, rest_r (w first), pos (T,J,3), rot (T,J,4), fps, caption, cls)"""
    import retarget as R
    from convert_clips import read_glb
    out = []
    for f in sorted(glob.glob(str(SRC / "*.glb"))):
        rig = Path(f).stem
        if rig == "human":
            continue
        sk = R.load_skeleton(f)
        js, b = read_glb(f)
        nodes = js["nodes"]
        # skin joints in the order load_skeleton used (node index -> joint index)
        if js.get("skins"):
            joints = []
            for s in js["skins"]:
                for j in s["joints"]:
                    if j not in joints:
                        joints.append(j)
        else:
            joints = list(range(len(nodes)))                              # same rule as retarget.load_skeleton
        idx = {j: k for k, j in enumerate(joints)}
        for an in js["animations"]:
            name = an.get("name", "")
            if name == "Rest_Pose" or name not in CAPTIONS[rig]:
                continue
            tracks, tmax = {}, 0.0
            for ch in an["channels"]:
                node, path = ch["target"]["node"], ch["target"]["path"]
                if node not in idx or path == "scale":
                    continue
                smp = an["samplers"][ch["sampler"]]
                t = R._acc(js, b, smp["input"])[:, 0]
                v = R._acc(js, b, smp["output"])
                tracks[(idx[node], path)] = (t, v, smp.get("interpolation", "LINEAR") == "STEP")
                tmax = max(tmax, float(t.max()))
            T = max(int(round(tmax * FPS)) + 1, 2)
            grid = np.arange(T) / FPS
            J = len(sk["names"])
            pos = np.tile(sk["loc"][None], (T, 1, 1))
            rot = np.tile(sk["rot"][None], (T, 1, 1))
            for (j, path), (t, v, step) in tracks.items():
                if path == "translation":
                    pos[:, j] = _interp(t, v, grid, False, step)
                elif path == "rotation":
                    rot[:, j] = _interp(t, v[:, [3, 0, 1, 2]], grid, True, step)
            out.append(dict(rig=rig, action=name, clip=f"{rig}-{name}", parents=np.array(sk["parents"]), names=sk["names"], rest_p=sk["loc"], rest_r=sk["rot"],
                            pos=pos, rot=rot, fps=FPS, caption=CAPTIONS[rig][name], cls=CLASS[rig]))
    return out


def cmd_text():
    """Append the template captions to the CLIP text index (existing rows keep their numbers, so the trained models stay valid)."""
    from embed import EMB, encode, load_text_encoder
    clips = read_clips()
    idx_path = EMB / "text_index.csv"
    rows = list(csv.reader(open(idx_path, newline="")))
    have = {(r[0], r[1]) for r in rows[1:]}
    new = [c for c in clips if ("template", c["clip"]) not in have]
    if not new:
        print("template captions already indexed")
        return
    emb = np.load(EMB / "text.npy")
    texts = {}
    with open(idx_path, newline="") as f:
        for r in csv.DictReader(f):
            texts.setdefault(r["text"], int(r["row"]))
    fresh = sorted({c["caption"] for c in new if c["caption"] not in texts})
    if fresh:
        E, cut = encode(fresh, load_text_encoder())
        for i, t in enumerate(fresh):
            texts[t] = len(emb) + i
        emb = np.concatenate([emb, E.astype(emb.dtype)])
        np.save(EMB / "text.npy", emb)
    with open(idx_path, "a", newline="") as f:
        w = csv.writer(f)
        for c in new:
            w.writerow(["template", c["clip"], "caption", texts[c["caption"]], c["caption"]])
    print(f"{len(new)} template captions indexed ({len(fresh)} new strings); text.npy now {len(emb)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.text:
        cmd_text()
    elif a.list:
        import collections
        cl = read_clips()
        print(len(cl), "clips", dict(collections.Counter(c["rig"] for c in cl)))
        for c in cl[:6] + [x for x in cl if x["rig"] == "spider"]:
            print(f"  {c['clip']:22s} {len(c['pos']):3d} frames, {len(c['names'])} joints  {c['cls']:10s} {c['caption']}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
