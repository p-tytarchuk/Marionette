"""First mesh from real flat artwork: every filled part of an SVG animal -> constrained Delaunay triangles, and a viewer.

Input: the SVGs that tools/filter_side_view_animals.py kept (data/side_view_animals/strict.jsonl, CC BY-NC-SA, research only).
Each filled shape (path, circle, ellipse, rect, polygon; groups and transforms are applied, strokes and gradients are not) becomes one part with
its fill colour: the outline is resampled to a roughly even spacing, holes are found by nesting (even-odd), and `triangle` (as in build_mesh.py)
fills it with triangles. Parts keep the SVG draw order. There is no skeleton and no bone weights yet: this is only the mesh (the step before
build_mesh.py's weights), so the artwork can be inspected.

Writes data/side_view_animals/mesh/<id>.json and one self-contained viewer, data/side_view_animals/mesh_viewer.html (dropdown, fill / wireframe /
vertex / outline toggles, drag to pan, wheel to zoom).

    python tools/svg_to_mesh.py                      # everything in strict.jsonl
    python tools/svg_to_mesh.py --ids 7805483 5886776
    python tools/svg_to_mesh.py --spacing 0.025      # outline spacing as a fraction of the drawing's diagonal
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import triangle as tr
from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "side_view_animals"
CURVE_STEPS = 10
MAX_TRIANGLES = 1500   # per part


class FlatPen(BasePen):
    """Collects subpaths as lists of points, curves flattened to CURVE_STEPS segments."""

    def __init__(self):
        super().__init__(None)
        self.subs, self.cur = [], None

    def _moveTo(self, p):
        self._end()
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            u = 1 - t
            self.cur.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                             u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            u = 1 - t
            self.cur.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))

    def _end(self):
        if self.cur and len(self.cur) >= 3:
            self.subs.append(self.cur)
        self.cur = None

    def _closePath(self):
        self._end()

    def _endPath(self):
        self._end()


def parse_transform(s: str | None) -> np.ndarray:
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        a = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        t = np.eye(3)
        if name == "matrix" and len(a) == 6:
            t = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate":
            t[0, 2], t[1, 2] = a[0], (a[1] if len(a) > 1 else 0)
        elif name == "scale":
            t[0, 0], t[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        elif name == "rotate":
            c, s_ = math.cos(math.radians(a[0])), math.sin(math.radians(a[0]))
            t = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            if len(a) == 3:
                t = np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]]) @ t @ np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]])
        m = m @ t
    return m


def circle_pts(cx, cy, rx, ry, n=48):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def paint(el, inherited: str) -> str:
    """fill of an element: its attribute / style, else the parent's. 'none' means not filled; gradients get a mid grey."""
    f = el.get("fill")
    m = re.search(r"(?:^|;)\s*fill\s*:\s*([^;]+)", el.get("style") or "")
    if m:
        f = m.group(1).strip()
    f = f if f is not None else inherited
    return "#888888" if f.startswith("url(") else f


def svg_parts(svg: str):
    """-> (viewBox (x, y, w, h), [{fill, polys: [np (n, 2)]}] in draw order)"""
    if re.search(r"<!(DOCTYPE|ENTITY)", svg, re.I):         # untrusted dataset files: no DTDs / entities (XXE, billion laughs); icon SVGs do not need them
        raise ValueError("SVG with a DOCTYPE / ENTITY declaration refused")
    svg = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", svg, count=0)
    root = ET.fromstring(svg)
    vb = [float(x) for x in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if x]
    if len(vb) != 4:
        vb = [0, 0, float(re.sub(r"[^\d.]", "", root.get("width", "100")) or 100), float(re.sub(r"[^\d.]", "", root.get("height", "100")) or 100)]
    parts = []

    def walk(el, ctm, fill):
        ctm = ctm @ parse_transform(el.get("transform"))
        tag = el.tag.split("}")[-1]
        fill = paint(el, fill)
        if tag in ("defs", "clipPath", "mask", "style", "title", "linearGradient", "radialGradient", "symbol"):
            return
        subs = []
        if tag == "path" and el.get("d"):
            pen = FlatPen()
            try:
                parse_path(el.get("d"), pen)
                pen._end()
            except Exception:
                pen.subs = []
            subs = pen.subs
        elif tag == "circle":
            subs = [circle_pts(float(el.get("cx", 0)), float(el.get("cy", 0)), float(el.get("r", 0)), float(el.get("r", 0)))]
        elif tag == "ellipse":
            subs = [circle_pts(float(el.get("cx", 0)), float(el.get("cy", 0)), float(el.get("rx", 0)), float(el.get("ry", 0)))]
        elif tag == "rect":
            x, y, w, h = (float(el.get(k, 0)) for k in ("x", "y", "width", "height"))
            subs = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
        elif tag == "polygon" and el.get("points"):
            v = [float(q) for q in re.split(r"[\s,]+", el.get("points").strip()) if q]
            subs = [list(zip(v[0::2], v[1::2]))]
        if subs and fill.lower() not in ("none", "transparent"):
            polys = []
            for s in subs:
                p = np.array(s, float)
                p = (np.c_[p, np.ones(len(p))] @ ctm.T)[:, :2]
                polys.append(p)
            parts.append(dict(fill=fill if fill != "inherit" else "#000000", polys=polys))
        for ch in el:
            walk(ch, ctm, fill)

    walk(root, np.eye(3), "#000000")
    return vb, parts


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def resample(p, h):
    """Subdivide every edge longer than h (the corners stay), drop repeated points."""
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / h)))
        out.extend(a + (b - a) * (i / n) for i in range(n))
    out = np.array(out)
    keep = np.r_[True, np.linalg.norm(np.diff(out, axis=0), axis=1) > h * 1e-3]
    out = out[keep]
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= h * 1e-3:
        out = out[:-1]
    return out


def triangulate_part(polys, h):
    """PSLG of the part's subpaths, holes by nesting depth (odd = hole) -> (vertices, triangles) or None."""
    polys = [p for p in polys if len(p) >= 3 and abs(area(p)) > h * h * 0.05]
    if not polys:
        return None
    polys.sort(key=lambda p: -abs(area(p)))
    depth = []
    for i, p in enumerate(polys):
        probe = p[0]
        depth.append(sum(MplPath(q).contains_point(probe) for q in polys[:i] if abs(area(q)) > abs(area(p))))
    verts, segs, holes = [], [], []
    for p, d in zip(polys, depth):
        r = resample(p, h)
        if len(r) < 3:
            continue
        base = len(verts)
        verts.extend(r.tolist())
        segs.extend([(base + i, base + (i + 1) % len(r)) for i in range(len(r))])
        if d % 2 == 1:
            one = tr.triangulate(dict(vertices=r, segments=[(i, (i + 1) % len(r)) for i in range(len(r))]), "p")
            holes.append(one["vertices"][one["triangles"][0]].mean(0).tolist())
    A = dict(vertices=np.array(verts), segments=np.array(segs))
    if holes:
        A["holes"] = np.array(holes)
    # the area limit gives h-sized triangles; S caps the Steiner points added by the quality refinement: very small input angles (near-coincident
    # points, slivers, overlapping outlines) make `q` refine without end, which gave 3.4 million vertices for one rat
    part_area = sum(abs(area(p)) * (1 if d % 2 == 0 else -1) for p, d in zip(polys, depth))
    t = tr.triangulate(A, f"pq25a{max(h * h * 0.8, abs(part_area) / MAX_TRIANGLES):.6f}S{MAX_TRIANGLES}")
    return t["vertices"], t["triangles"]


def _child(conn, polys, h):
    try:
        conn.send(triangulate_part(polys, h))
    except Exception as e:
        conn.send(e)


def safe_triangulate(polys, h, timeout=10.0):
    """triangulate_part in a forked child: `triangle` can segfault or loop on degenerate outlines (two dinosaurs did), and neither can be caught in-process.
    Returns (result, None) or (None, reason)."""
    ctx = multiprocessing.get_context("fork")
    a, b = ctx.Pipe(False)
    pr = ctx.Process(target=_child, args=(b, polys, h))
    pr.start()
    b.close()
    got = None
    if a.poll(timeout):
        try:
            got = a.recv()
        except EOFError:
            got = None
    pr.join(0.5)
    if pr.is_alive():
        pr.kill()
        pr.join()
        return None, f"triangle timed out after {timeout:g}s"
    if isinstance(got, Exception):
        return None, str(got)[:80]
    if got is None and pr.exitcode not in (0, None):
        return None, f"triangle crashed (exit {pr.exitcode})"
    return got, None


def build(svg_path: Path, spacing: float):
    vb, parts = svg_parts(svg_path.read_text(encoding="utf-8"))
    h = spacing * math.hypot(vb[2], vb[3])
    out = []
    for part in parts:
        r, err = safe_triangulate(part["polys"], h)
        if err:                                     # self-intersecting or degenerate outline: skip the part, say so
            out.append(dict(fill=part["fill"], error=err))
            continue
        if r is None:
            continue
        v, t = r
        out.append(dict(fill=part["fill"], v=np.round(v, 2).tolist(), t=t.tolist()))
    return dict(vb=vb, spacing=round(h, 3), parts=out)


VIEWER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mesh viewer</title>
<style>:root{--bg:#f6f5f2;--fg:#1f1f1d;--mut:#6b6a66;--line:#d9d7d0;--card:#fff}@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a988f;--line:#34332f;--card:#1e1e1c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
select,label{font:inherit;color:var(--fg)}select{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px;max-width:60vw}
#info{color:var(--mut);margin-left:auto}main{padding:12px 16px}canvas{width:100%;height:calc(100vh - 110px);background:#fff;border:1px solid var(--line);border-radius:8px;cursor:grab;touch-action:none}</style></head>
<body><header><select id="sel"></select><label><input type="checkbox" id="fill" checked> fill</label><label><input type="checkbox" id="wire" checked> triangles</label>
<label><input type="checkbox" id="pts"> vertices</label><label><input type="checkbox" id="line"> outlines</label><span id="info"></span></header>
<main><canvas id="c"></canvas></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),cv=$('c'),ctx=cv.getContext('2d');let cur=null,view={s:1,x:0,y:0};
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.species+' ('+d.id+')';$('sel').appendChild(o)});
function fit(){const r=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;cv.width=r.width*dpr;cv.height=r.height*dpr;const[vx,vy,vw,vh]=cur.vb,s=Math.min(cv.width/vw,cv.height/vh)*.92;view={s,x:(cv.width-vw*s)/2-vx*s,y:(cv.height-vh*s)/2-vy*s}}
function pick(i){cur=DATA[i];let nv=0,nt=0;cur.parts.forEach(p=>{if(p.v){nv+=p.v.length;nt+=p.t.length}});$('info').textContent=cur.parts.length+' parts, '+nv+' vertices, '+nt+' triangles, spacing '+cur.spacing+(cur.parts.some(p=>p.error)?' (some parts skipped)':'');fit();draw()}
function draw(){ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);ctx.setTransform(view.s,0,0,view.s,view.x,view.y);const lw=1/view.s*(devicePixelRatio||1);
for(const p of cur.parts){if(!p.v)continue;const v=p.v,t=p.t;
if($('fill').checked){ctx.fillStyle=p.fill;ctx.beginPath();for(const a of t){ctx.moveTo(v[a[0]][0],v[a[0]][1]);ctx.lineTo(v[a[1]][0],v[a[1]][1]);ctx.lineTo(v[a[2]][0],v[a[2]][1]);ctx.closePath()}ctx.fill()}
if($('wire').checked){ctx.strokeStyle='rgba(20,20,20,.55)';ctx.lineWidth=lw*.8;ctx.beginPath();for(const a of t){ctx.moveTo(v[a[0]][0],v[a[0]][1]);ctx.lineTo(v[a[1]][0],v[a[1]][1]);ctx.lineTo(v[a[2]][0],v[a[2]][1]);ctx.closePath()}ctx.stroke()}
if($('pts').checked){ctx.fillStyle='#d1242f';const r=2.2*lw;for(const q of v){ctx.beginPath();ctx.arc(q[0],q[1],r,0,7);ctx.fill()}}}
if($('line').checked){ctx.strokeStyle='#0b57d0';ctx.lineWidth=lw*1.6;for(const p of cur.parts){if(!p.v)continue;const e={};for(const a of p.t)for(let k=0;k<3;k++){const i=a[k],j=a[(k+1)%3],key=i<j?i+'_'+j:j+'_'+i;e[key]=(e[key]||0)+1}
ctx.beginPath();for(const k in e)if(e[k]===1){const[i,j]=k.split('_');ctx.moveTo(p.v[i][0],p.v[i][1]);ctx.lineTo(p.v[j][0],p.v[j][1])}ctx.stroke()}}}
$('sel').onchange=e=>pick(+e.target.value);['fill','wire','pts','line'].forEach(i=>$(i).onchange=draw);addEventListener('resize',()=>{fit();draw()});
let drag=null;cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,view.x,view.y];cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'};cv.onpointerup=()=>{drag=null;cv.style.cursor='grab'};
cv.onpointermove=e=>{if(!drag)return;const d=devicePixelRatio||1;view.x=drag[2]+(e.clientX-drag[0])*d;view.y=drag[3]+(e.clientY-drag[1])*d;draw()};
cv.onwheel=e=>{e.preventDefault();const d=devicePixelRatio||1,r=cv.getBoundingClientRect(),mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d,k=Math.exp(-e.deltaY*.0015);view.x=mx-(mx-view.x)*k;view.y=my-(my-view.y)*k;view.s*=k;draw()};
pick(0);</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--ids", nargs="*", help="SVG ids (default: every row of strict.jsonl)")
    ap.add_argument("--spacing", type=float, default=0.03, help="outline spacing as a fraction of the drawing's diagonal (default 0.03)")
    args = ap.parse_args()
    d = args.dir
    rows = {r["id"]: r for r in map(json.loads, open(d / "strict.jsonl", encoding="utf-8"))}
    ids = args.ids or list(rows)
    (d / "mesh").mkdir(exist_ok=True)
    pack = []
    for rid in ids:
        r = rows.get(rid, dict(id=rid, group="?", verify={}))
        t0 = time.time()
        m = build(d / "svg" / f"{rid}.svg", args.spacing)
        m.update(id=rid, species=(r.get("verify") or {}).get("species", r.get("group")), caption=r.get("caption"))
        (d / "mesh" / f"{rid}.json").write_text(json.dumps(m), encoding="utf-8")
        nv = sum(len(p["v"]) for p in m["parts"] if "v" in p)
        nt = sum(len(p["t"]) for p in m["parts"] if "t" in p)
        bad = sum("error" in p for p in m["parts"])
        for p in m["parts"]:
            if "error" in p:
                print(f"{'':>9}   skipped a {p['fill']} part: {p['error']}")
        print(f"{rid:>9} {str(m['species'])[:16]:16s} {len(m['parts']):3d} parts {nv:6d} vertices {nt:6d} triangles{f'  ({bad} parts skipped)' if bad else ''}  {time.time() - t0:.2f}s", flush=True)
        pack.append(m)
    html = VIEWER.replace("__DATA__", json.dumps(pack, separators=(",", ":")))
    (d / "mesh_viewer.html").write_text(html, encoding="utf-8")
    print(f"wrote {d / 'mesh_viewer.html'} ({len(html) / 1e6:.1f} MB, {len(pack)} animals)")


if __name__ == "__main__":
    main()
