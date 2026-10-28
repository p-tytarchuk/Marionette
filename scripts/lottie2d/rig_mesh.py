"""Bone weights for a mesh made by svg_to_mesh.py, and a rig viewer.

Input: data/side_view_animals/mesh/<id>.json (parts: fill, vertices, triangles) and a hand-placed skeleton, data/side_view_animals/rig_spec/<id>.json
(joint ids and parents of the 4leg template, positions in the SVG's own coordinates). The weights are build_mesh.py's v1: inverse distance to the
bone segments, the k_max nearest bones, normalised per vertex; the same joints deform every part, so the antlers, scarf and hooves follow the bones
they sit on. If the spec says "flip", x is mirrored (mesh and joints) so the animal faces +x like the Loomie templates.

Writes data/side_view_animals/rig/<id>.json (joints, per-part vertices / triangles / weights, demo curves) and rig_viewer.html: bones, weight map
(dominant bone, or one bone's weight), manual joint angle, and a hand-set gait in the primitive m(t) of tools/primitive.py. The deformation in the
viewer is the same 2D linear blend skinning as build_mesh.deform (checked against it in --check).

    python scripts/lottie2d/rig_mesh.py 7805483 [--k-max 4] [--power 4]
    python scripts/lottie2d/rig_mesh.py 7805483 --check          # JS deformation vs build_mesh.deform, needs node
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # build_mesh.py (copied from Loomie)
import build_mesh as BM  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "side_view_animals"

# the viewer's deformation, kept between markers so --check can run exactly this code in node
DEFORM_JS = """/*DEFORM*/
function worldMatrices(joints, angles) {
  const M = new Array(joints.length), idx = {};
  joints.forEach((j, i) => idx[j.id] = i);
  function world(i) {
    if (M[i]) return M[i];
    const j = joints[i], a = angles[j.id] || 0, c = Math.cos(a), s = Math.sin(a), px = j.rest_pos[0], py = j.rest_pos[1];
    const L = [c, -s, px - c * px + s * py, s, c, py - s * px - c * py];
    if (j.parent === null) return M[i] = L;
    const P = world(idx[j.parent]);
    return M[i] = [P[0] * L[0] + P[1] * L[3], P[0] * L[1] + P[1] * L[4], P[0] * L[2] + P[1] * L[5] + P[2],
                   P[3] * L[0] + P[4] * L[3], P[3] * L[1] + P[4] * L[4], P[3] * L[2] + P[4] * L[5] + P[5]];
  }
  joints.forEach((_, i) => world(i));
  return M;
}
function deform(verts, weights, joints, angles) {
  const M = worldMatrices(joints, angles), out = new Array(verts.length);
  for (let n = 0; n < verts.length; n++) {
    const x = verts[n][0], y = verts[n][1]; let ox = 0, oy = 0;
    for (const [k, w] of weights[n]) { const m = M[k]; ox += w * (m[0] * x + m[1] * y + m[2]); oy += w * (m[3] * x + m[4] * y + m[5]); }
    out[n] = [ox, oy];
  }
  return {pts: out, M};
}
/*END*/"""

VIEWER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rig viewer</title>
<style>:root{--bg:#f6f5f2;--fg:#1f1f1d;--mut:#6b6a66;--line:#d9d7d0;--card:#fff}@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a988f;--line:#34332f;--card:#1e1e1c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}header{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
select,button,input,label{font:inherit;color:var(--fg)}select,button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px}button{cursor:pointer}
input[type=range]{width:160px;vertical-align:middle}#info{color:var(--mut);margin-left:auto}main{padding:12px 16px}
canvas{width:100%;height:calc(100vh - 120px);background:#fff;border:1px solid var(--line);border-radius:8px;cursor:grab;touch-action:none}.mut{color:var(--mut)}</style></head>
<body><header><button id="play">Play gait</button>
<label>joint <select id="joint"></select></label><label>angle <input type="range" id="ang" min="-1.6" max="1.6" step="0.01" value="0"> <span id="angv" class="mut">0.00</span></label><button id="reset">Reset pose</button>
<label>color <select id="mode"><option value="art">artwork</option><option value="dom">dominant bone</option><option value="w">weight of selected joint</option></select></label>
<label><input type="checkbox" id="wire"> triangles</label><label><input type="checkbox" id="bones" checked> bones</label><span id="info"></span></header>
<main><canvas id="c"></canvas></main>
<script>const RIG=__RIG__;
__DEFORM__
const $=id=>document.getElementById(id),cv=$('c'),ctx=cv.getContext('2d'),J=RIG.joints;let view={s:1,x:0,y:0},angles={},playing=false,t0=performance.now();
const pal=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#0aa6a6','#bcbd22','#f032e6','#8c564b','#7f7f7f','#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728','#17becf'];
J.forEach((j,i)=>{const o=document.createElement('option');o.value=j.id;o.textContent=j.id;$('joint').appendChild(o)});
const flat=[];RIG.parts.forEach(p=>p.t.forEach(a=>flat.push(a)));
function fit(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;const[vx,vy,vw,vh]=RIG.vb,s=Math.min(cv.width/vw,cv.height/vh)*.9;view={s,x:(cv.width-vw*s)/2-vx*s,y:(cv.height-vh*s)/2-vy*s}}
function curve(c,t){const w=Math.sin(2*Math.PI*(c.s*t+c.phi));return c.r+c.a*(c.clamp?Math.max(0,w):w)}
function pose(t){if(playing){for(const k in RIG.demo.curves)angles[k]=curve(RIG.demo.curves[k],t)}}
function draw(now){const t=(now-t0)/1000;pose(t);ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);ctx.setTransform(view.s,0,0,view.s,view.x,view.y);
const lw=(devicePixelRatio||1)/view.s,mode=$('mode').value,sel=J.findIndex(j=>j.id===$('joint').value);let M=null;
for(const p of RIG.parts){const d=deform(p.v,p.w,J,angles);M=d.M;const v=d.pts;
 if(mode==='art'){ctx.fillStyle=p.fill;ctx.beginPath();for(const a of p.t){ctx.moveTo(v[a[0]][0],v[a[0]][1]);ctx.lineTo(v[a[1]][0],v[a[1]][1]);ctx.lineTo(v[a[2]][0],v[a[2]][1]);ctx.closePath()}ctx.fill()}
 else{for(const a of p.t){let c;if(mode==='dom'){const dom=[0,1,2].map(q=>p.w[a[q]].reduce((b,e)=>e[1]>b[1]?e:b,[0,-1])[0]);c=pal[dom[0]%pal.length]}
  else{const w=[0,1,2].reduce((s,q)=>s+((p.w[a[q]].find(e=>e[0]===sel)||[0,0])[1]),0)/3;c='hsl(215,85%,'+(96-w*62)+'%)'}
  ctx.fillStyle=c;ctx.beginPath();ctx.moveTo(v[a[0]][0],v[a[0]][1]);ctx.lineTo(v[a[1]][0],v[a[1]][1]);ctx.lineTo(v[a[2]][0],v[a[2]][1]);ctx.closePath();ctx.fill()}}
 if($('wire').checked){ctx.strokeStyle='rgba(20,20,20,.5)';ctx.lineWidth=lw*.7;ctx.beginPath();for(const a of p.t){ctx.moveTo(v[a[0]][0],v[a[0]][1]);ctx.lineTo(v[a[1]][0],v[a[1]][1]);ctx.lineTo(v[a[2]][0],v[a[2]][1]);ctx.closePath()}ctx.stroke()}}
if($('bones').checked&&M){const pos=J.map((j,i)=>{const m=M[i],x=j.rest_pos[0],y=j.rest_pos[1];return[m[0]*x+m[1]*y+m[2],m[3]*x+m[4]*y+m[5]]});
 ctx.lineWidth=lw*2.2;J.forEach((j,i)=>{ctx.strokeStyle=i===sel?'#d1242f':'#0b57d0';const a=pos[i];if(j.parent!==null){const b=pos[J.findIndex(q=>q.id===j.parent)];ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(a[0],a[1]);ctx.stroke()}
  if(j.tip_pos){const m=M[i],x=j.tip_pos[0],y=j.tip_pos[1];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(m[0]*x+m[1]*y+m[2],m[3]*x+m[4]*y+m[5]);ctx.stroke()}});
 J.forEach((j,i)=>{ctx.fillStyle=i===sel?'#d1242f':'#fff';ctx.strokeStyle='#0b57d0';ctx.lineWidth=lw*1.6;ctx.beginPath();ctx.arc(pos[i][0],pos[i][1],4.5*lw,0,7);ctx.fill();ctx.stroke()})}
requestAnimationFrame(draw)}
$('play').onclick=()=>{playing=!playing;t0=performance.now();$('play').textContent=playing?'Pause gait':'Play gait'};
$('reset').onclick=()=>{angles={};playing=false;$('play').textContent='Play gait';sync()};
function sync(){$('ang').value=angles[$('joint').value]||0;$('angv').textContent=(+$('ang').value).toFixed(2)}
$('joint').onchange=sync;$('ang').oninput=e=>{playing=false;$('play').textContent='Play gait';angles[$('joint').value]=+e.target.value;$('angv').textContent((+e.target.value).toFixed(2))};
addEventListener('resize',fit);
let drag=null;cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,view.x,view.y];cv.setPointerCapture(e.pointerId)};cv.onpointerup=()=>drag=null;
cv.onpointermove=e=>{if(!drag)return;const d=devicePixelRatio||1;view.x=drag[2]+(e.clientX-drag[0])*d;view.y=drag[3]+(e.clientY-drag[1])*d};
cv.onwheel=e=>{e.preventDefault();const d=devicePixelRatio||1,r=cv.getBoundingClientRect(),mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d,k=Math.exp(-e.deltaY*.0015);view.x=mx-(mx-view.x)*k;view.y=my-(my-view.y)*k;view.s*=k};
const nv=RIG.parts.reduce((s,p)=>s+p.v.length,0),nt=flat.length;$('info').textContent=RIG.species+' '+RIG.id+': '+J.length+' joints, '+RIG.parts.length+' parts, '+nv+' vertices, '+nt+' triangles, k_max '+RIG.build.k_max;
fit();sync();requestAnimationFrame(draw);</script></body></html>"""


def build(args) -> dict:
    d = args.dir
    mesh = json.loads((d / "mesh" / f"{args.id}.json").read_text())
    spec = json.loads((d / "rig_spec" / f"{args.id}.json").read_text())
    joints = spec["joints"]
    flip, W0 = spec.get("flip", False), mesh["vb"][0] + mesh["vb"][2]
    fx = (lambda p: [round(W0 - p[0], 2), p[1]]) if flip else (lambda p: [p[0], p[1]])
    for j in joints:
        j["rest_pos"] = fx(j["rest_pos"])
        if "tip_pos" in j:
            j["tip_pos"] = fx(j["tip_pos"])
    pos = {j["id"]: np.array(j["rest_pos"], float) for j in joints}
    for j in joints:                                   # rest_angle: chord to the first child, else to the tip (as in the templates)
        kids = [k for k in joints if k["parent"] == j["id"]]
        tgt = pos[kids[0]["id"]] if kids else (np.array(j["tip_pos"], float) if "tip_pos" in j else None)
        j["rest_angle"] = round(float(math.atan2(*(tgt - pos[j["id"]])[::-1])), 3) if tgt is not None else 0.0
        j.setdefault("movable", True)
    parts = [p for p in mesh["parts"] if "v" in p]
    verts = np.vstack([np.array(p["v"], float) for p in parts])
    if flip:
        verts[:, 0] = W0 - verts[:, 0]
    W = BM.inverse_distance_weights(verts, joints, args.k_max, args.power)
    ids = [j["id"] for j in joints]
    out, at = [], 0
    for p in parts:
        n = len(p["v"])
        tri = np.array(p["t"])
        if flip:
            tri = tri[:, [0, 2, 1]]                    # keep the winding after the mirror
        out.append(dict(fill=p["fill"], v=np.round(verts[at:at + n], 2).tolist(), t=tri.tolist(),
                        w=[[[int(k), round(float(W[at + i, k]), 4)] for k in np.nonzero(W[at + i] > 1e-3)[0]] for i in range(n)]))
        at += n
    dom = W.argmax(1)
    usage = {ids[k]: int((dom == k).sum()) for k in range(len(ids))}
    rig = dict(id=args.id, species=mesh.get("species"), vb=mesh["vb"], flip=flip, note=spec.get("note"), skeleton_source="4leg template ids, hand-placed",
               joints=joints, parts=out, demo=spec.get("demo", {}), build=dict(weights="inverse-distance-to-bone-segment", k_max=args.k_max, power=args.power),
               dominant_vertex_count=usage, limits_deg=spec.get("limits_deg", {}))
    (d / "rig").mkdir(exist_ok=True)
    (d / "rig" / f"{args.id}.json").write_text(json.dumps(rig), encoding="utf-8")
    html = VIEWER.replace("__DEFORM__", DEFORM_JS).replace("__RIG__", json.dumps(rig, separators=(",", ":")))
    (d / "rig_viewer.html").write_text(html, encoding="utf-8")
    print(f"{len(verts)} vertices, {len(joints)} joints, {sum(len(p['t']) for p in out)} triangles; weights <= {args.k_max} bones per vertex")
    print("vertices dominated by each bone:", usage)
    unused = [k for k, v in usage.items() if v == 0]
    if unused:
        print("WARNING: no vertex is dominated by", unused)
    print(f"wrote {d / 'rig' / (args.id + '.json')} and {d / 'rig_viewer.html'} ({len(html) / 1e6:.2f} MB)")
    return rig


def check(args, rig) -> None:
    """Run the viewer's deform() in node on random poses and compare with build_mesh.deform."""
    joints = rig["joints"]
    verts = np.vstack([np.array(p["v"], float) for p in rig["parts"]])
    W = np.zeros((len(verts), len(joints)))
    n = 0
    for p in rig["parts"]:
        for ws in p["w"]:
            for k, w in ws:
                W[n, k] = w
            n += 1
    rng = np.random.default_rng(0)
    worst = 0.0
    weights = [ws for p in rig["parts"] for ws in p["w"]]
    js = DEFORM_JS + "\nconst R = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" \
                     "process.stdout.write(JSON.stringify(deform(R.verts, R.weights, R.joints, R.angles).pts));"
    n_trials = 8
    for trial in range(n_trials):
        angles = {j["id"]: float(rng.uniform(-0.9, 0.9)) for j in joints}
        ref = BM.deform(verts, joints, W, angles)
        payload = json.dumps(dict(verts=verts.tolist(), weights=weights, joints=joints, angles=angles))
        res = subprocess.run(["node", "-e", js], input=payload, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"node failed: {res.stderr.strip()[:300]}")
        worst = max(worst, float(np.abs(np.array(json.loads(res.stdout)) - ref).max()))
    print(f"JS deform vs build_mesh.deform on {n_trials} random poses: max |difference| = {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("MISMATCH between the viewer's deformation and build_mesh.deform")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", help="animation id: reads <dir>/mesh/<id>.json and <dir>/rig_spec/<id>.json")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--k-max", type=int, default=4, help="bones per vertex")
    ap.add_argument("--power", type=float, default=4.0, help="inverse-distance exponent")
    ap.add_argument("--check", action="store_true", help="compare the viewer's JS deformation with build_mesh.deform (needs node)")
    args = ap.parse_args()
    rig = build(args)
    if args.check:
        check(args, rig)


if __name__ == "__main__":
    main()
