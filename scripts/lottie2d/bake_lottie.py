"""Bake a rigged flat mesh (rig_mesh.py) into a Lottie animation, and a viewer that plays it as SVG.

A flat-colour part only needs its outline to move, so for every part the boundary loops of the triangle mesh (edges used by one triangle) become Lottie
path shapes, one per loop (holes work through the even-odd fill rule), and the skinned position of every boundary vertex is written as one path keyframe per
frame (build_mesh.deform, the same 2D linear blend skinning as the rig viewer). Interior vertices only exist to compute smooth weights and are not
written. Loops that never move are written as plain, non-animated paths. Part order is the SVG's draw order (Lottie lists the top layer first, so it is reversed).

The input is a list of per-frame joint angles, {joint id: radians}, so any motion source can drive it: today the hand-set trot of the rig spec (primitive m(t) of
tools/primitive.py); next the motion Marionette generates, mapped to the joints.

Writes data/side_view_animals/lottie/<id>_<name>.json and lottie_viewer.html (self-contained: lottie-web 5.13.0 from tools/viewer_assets is inlined,
SVG renderer; play / pause, a frame scrubber, and a button that saves the current frame as a standalone .svg).

    python tools/bake_lottie.py 7805483 [--fps 30] [--cycles 2]                 # the rig spec's hand-set gait
    python tools/bake_lottie.py 7805483 --motion data/side_view_animals/motion/<name>.json [--step 2]     # Marionette's motion (scripts/lottie2d/motion_to_2d.py)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # build_mesh.py (copied from Loomie)
sys.path.insert(0, str(ROOT / "lottie"))                    # primitive.py (copied from Loomie)
import build_mesh as BM  # noqa: E402
import primitive as PR  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "side_view_animals"
LOTTIE_JS = ROOT / "viewer" / "lottie.min.js"
STATIC_EPS = 0.01          # px: a loop that moves less than this in every frame is written as a static path

NAMED = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "green": "#008000", "blue": "#0000ff", "none": "#000000"}


def hex_rgba(c: str) -> list[float]:
    c = NAMED.get(c.strip().lower(), c.strip())
    m = re.fullmatch(r"#([0-9a-fA-F]{3})", c)
    if m:
        c = "#" + "".join(ch * 2 for ch in m.group(1))
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", c)
    if m:
        v = int(m.group(1), 16)
        return [round(((v >> 16) & 255) / 255, 4), round(((v >> 8) & 255) / 255, 4), round((v & 255) / 255, 4), 1]
    m = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", c)
    if m:
        return [round(int(g) / 255, 4) for g in m.groups()] + [1]
    return [0.53, 0.53, 0.53, 1]


def boundary_loops(tris) -> list[list[int]]:
    """Closed vertex loops along the edges that belong to exactly one triangle."""
    count = defaultdict(int)
    for a in tris:
        for k in range(3):
            i, j = a[k], a[(k + 1) % 3]
            count[(min(i, j), max(i, j))] += 1
    adj = defaultdict(list)
    for (i, j), n in count.items():
        if n == 1:
            adj[i].append(j)
            adj[j].append(i)
    used, loops = set(), []
    for (i, j), n in count.items():
        if n != 1 or (i, j) in used:
            continue
        loop, prev, cur = [i], i, j
        used.add((i, j))
        while cur != i:
            loop.append(cur)
            nxt = next((v for v in adj[cur] if (min(cur, v), max(cur, v)) not in used), None)
            if nxt is None:
                break
            used.add((min(cur, nxt), max(cur, nxt)))
            prev, cur = cur, nxt
        if len(loop) >= 3 and cur == i:
            loops.append(loop)
    return loops


def shape(pts) -> dict:
    n = len(pts)
    z = [[0, 0]] * n
    return {"i": z, "o": z, "v": [[round(float(x), 2), round(float(y), 2)] for x, y in pts], "c": True}


def bake(rig: dict, frames: list[dict], fps: int, name: str, loop: bool = True, offsets=None, step: int = 1) -> dict:
    """rig: rig_mesh.py's json; frames: per-frame {joint id: angle (rad)}; offsets: per-frame [dx, dy] added to every vertex (root motion, px).
    loop: the animation is periodic, so an extra last keyframe repeats frame 0; otherwise the last pose is held. step: one keyframe every `step` frames (linear in between)."""
    joints = rig["joints"]
    n_frames = len(frames)
    off = np.array(offsets, float) if offsets is not None else np.zeros((n_frames, 2))
    last = n_frames if loop else n_frames - 1
    keys = sorted(set(range(0, last + 1, step)) | {last})
    # pass 1: skinned boundary of every loop in every frame (frame n_frames repeats frame 0: the loop closes)
    baked = []
    for part in rig["parts"]:
        v = np.array(part["v"], float)
        W = np.zeros((len(v), len(joints)))
        for i, ws in enumerate(part["w"]):
            for k, w in ws:
                W[i, k] = w
        loops = boundary_loops(part["t"])
        deformed = [BM.deform(v, joints, W, frames[f % n_frames]) + off[f % n_frames] for f in keys]
        baked.append((part, [np.stack([d[loop] for d in deformed]) for loop in loops]))     # (keyframes, n, 2) per loop
    # the artboard has to hold every frame (a swinging leg leaves the rest-pose box), and Lottie has no negative origin: shift the coordinates
    allpts = np.concatenate([t.reshape(-1, 2) for _, tr in baked for t in tr])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.03 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    layers = []
    for pi, (part, tracks_per_loop) in enumerate(baked):
        items = []
        for li, tracks in enumerate(tracks_per_loop):
            tracks = tracks - lo
            if float(np.abs(tracks - tracks[0]).max()) < STATIC_EPS:
                ks = {"a": 0, "k": shape(tracks[0])}
            else:
                ks = {"a": 1, "k": [dict(t=int(f), s=[shape(tracks[n])], i={"x": 1, "y": 1}, o={"x": 0, "y": 0}) if f < last else dict(t=int(f), s=[shape(tracks[n])])
                                    for n, f in enumerate(keys)]}
            items.append({"ty": "sh", "ind": li, "nm": f"loop {li}", "ks": ks})
        items.append({"ty": "fl", "nm": "fill", "c": {"a": 0, "k": hex_rgba(part["fill"])}, "o": {"a": 0, "k": 100}, "r": 2})     # r 2 = even-odd
        items.append({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0},
                      "o": {"a": 0, "k": 100}, "sk": {"a": 0, "k": 0}, "sa": {"a": 0, "k": 0}})
        layers.append({"ddd": 0, "ind": 0, "ty": 4, "nm": f"part {pi} {part['fill']}", "sr": 1,
                       "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0}, "p": {"a": 0, "k": [0, 0, 0]}, "a": {"a": 0, "k": [0, 0, 0]}, "s": {"a": 0, "k": [100, 100, 100]}},
                       "ao": 0, "shapes": [{"ty": "gr", "nm": f"part {pi}", "it": items}], "ip": 0, "op": n_frames, "st": 0, "bm": 0})
    layers.reverse()                                                   # Lottie draws the first layer on top; the SVG's first part is the bottom one
    for i, l in enumerate(layers, 1):
        l["ind"] = i
    return {"v": "5.7.0", "fr": fps, "ip": 0, "op": n_frames, "w": int(math.ceil(hi[0] - lo[0])), "h": int(math.ceil(hi[1] - lo[1])), "nm": name, "ddd": 0, "assets": [], "layers": layers}


def demo_frames(rig: dict, fps: int, cycles: int) -> list[dict]:
    """The rig spec's hand-set curves evaluated at every frame; the loop is `cycles` periods of the base frequency."""
    curves = rig["demo"]["curves"]
    base = min(c["s"] for c in curves.values())
    bad = [k for k, c in curves.items() if abs(c["s"] / base - round(c["s"] / base)) > 1e-6]
    if bad:
        print("WARNING: frequencies that are not multiples of the base frequency do not loop seamlessly:", bad)
    n = int(round(cycles * fps / base))
    return [{k: float(PR.evaluate(f / fps, c["r"], c["a"], c["s"], c["phi"], c["clamp"])) for k, c in curves.items()} for f in range(n)]


VIEWER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lottie viewer</title>
<style>:root{--bg:#f6f5f2;--fg:#1f1f1d;--mut:#6b6a66;--line:#d9d7d0;--card:#fff}@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a988f;--line:#34332f;--card:#1e1e1c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}header{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
button,input,label{font:inherit;color:var(--fg)}button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}input[type=range]{width:240px;vertical-align:middle}
#info{color:var(--mut);margin-left:auto}main{display:flex;justify-content:center;padding:12px 16px}#stage{background:#fff;border:1px solid var(--line);border-radius:8px;width:min(100%,640px);aspect-ratio:__W__/__H__}#stage svg{width:100%;height:100%;display:block}</style></head>
<body><header><button id="play">Pause</button><label>frame <input type="range" id="fr" min="0" max="__N__" step="1" value="0"> <span id="frv">0</span></label>
<button id="svg">Save this frame as .svg</button><button id="json">Save the .json</button><span id="info">__INFO__</span></header>
<main><div id="stage"></div></main>
<script>__LOTTIE__</script>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);
const anim=lottie.loadAnimation({container:$('stage'),renderer:'svg',loop:true,autoplay:true,animationData:DATA,rendererSettings:{preserveAspectRatio:'xMidYMid meet'}});
let playing=true;
$('play').onclick=()=>{playing=!playing;playing?anim.play():anim.pause();$('play').textContent=playing?'Pause':'Play'};
$('fr').oninput=e=>{playing=false;$('play').textContent='Play';anim.goToAndStop(+e.target.value,true);$('frv').textContent=e.target.value};
anim.addEventListener('enterFrame',e=>{if(playing){const f=Math.floor(e.currentTime);$('fr').value=f;$('frv').textContent=f}});
function save(name,type,text){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('svg').onclick=()=>{const s=$('stage').querySelector('svg').cloneNode(true);s.setAttribute('xmlns','http://www.w3.org/2000/svg');s.setAttribute('viewBox','0 0 '+DATA.w+' '+DATA.h);s.removeAttribute('style');save('__NAME___frame'+$('fr').value+'.svg','image/svg+xml',new XMLSerializer().serializeToString(s))};
$('json').onclick=()=>save('__NAME__.json','application/json',JSON.stringify(DATA));</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=2, help="how many periods of the demo gait the loop contains")
    ap.add_argument("--name", default="gait")
    ap.add_argument("--motion", type=Path, help="a motion json from Marionette's scripts/lottie2d/motion_to_2d.py (frames, offsets, fps) instead of the rig spec's hand-set gait")
    ap.add_argument("--step", type=int, default=1, help="one keyframe every N frames (linear between): smaller files, softer motion")
    args = ap.parse_args()
    d = args.dir
    rig = json.loads((d / "rig" / f"{args.id}.json").read_text())
    if args.motion:
        mo = json.loads(args.motion.read_text())
        frames, fps, offsets, loop, name = mo["frames"], mo["fps"], mo.get("offsets"), mo.get("loop", False), f"{args.id}_{args.motion.stem}"
    else:
        frames, fps, offsets, loop, name = demo_frames(rig, args.fps, args.cycles), args.fps, None, True, f"{args.id}_{args.name}"
    args.fps = fps
    lot = bake(rig, frames, fps, name, loop=loop, offsets=offsets, step=args.step)
    (d / "lottie").mkdir(exist_ok=True)
    text = json.dumps(lot, separators=(",", ":"))
    (d / "lottie" / f"{name}.json").write_text(text, encoding="utf-8")
    n_sh = sum(1 for l in lot["layers"] for g in l["shapes"] for s in g["it"] if s["ty"] == "sh")
    n_anim = sum(1 for l in lot["layers"] for g in l["shapes"] for s in g["it"] if s["ty"] == "sh" and s["ks"]["a"])
    n_vert = sum(len(s["ks"]["k"][0]["s"][0]["v"]) if s["ks"]["a"] else len(s["ks"]["k"]["v"]) for l in lot["layers"] for g in l["shapes"] for s in g["it"] if s["ty"] == "sh")
    info = f"{rig.get('species')} {args.id}: {len(lot['layers'])} layers, {n_sh} paths ({n_anim} animated), {n_vert} outline vertices, {len(frames)} frames at {args.fps} fps, keyframe every {args.step}, {len(text) / 1e3:.0f} KB"
    html = (VIEWER.replace("__LOTTIE__", LOTTIE_JS.read_text(encoding="utf-8").replace("</script", "<\\/script")).replace("__DATA__", text)
            .replace("__W__", str(lot["w"])).replace("__H__", str(lot["h"])).replace("__N__", str(len(frames))).replace("__NAME__", name).replace("__INFO__", info))
    (d / "lottie_viewer.html").write_text(html, encoding="utf-8")
    print(info)
    print(f"wrote {d / 'lottie' / (name + '.json')} and {d / 'lottie_viewer.html'}")


if __name__ == "__main__":
    main()
