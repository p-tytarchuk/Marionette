#!/usr/bin/env python3
"""Filter canonical rigged meshes by triangle count and remove duplicates.

Reads the GLBs in sources/unimate/canonical_assets/{mixamo,objaverse}, computes per-asset stats
(triangles, joints, skeleton signature, geometry hash) and decides keep/drop. Writes
data/manifest.csv (every asset, with a `keep` flag and a `reason`) and prints a summary.
Only the GLB JSON chunk (and, for the geometry hash, the POSITION buffer views) is read.

Filters, applied in this order (the first one that fails is the recorded reason):
  parse_error      GLB unreadable or without a skinned mesh
  flagged          listed in export/<src>/filtered_objects.txt (bad rest pose, held out)
  class            category not in --classes
  too_many_tris    triangles > --max-tris
  too_few_tris     triangles < --min-tris
  no_clips         objaverse object type with no clips in clips.csv (--min-clips)
  duplicate_mesh   identical geometry (+ skeleton) already kept
  skeleton_cap     more than --max-per-skeleton kept assets share one skeleton
                   (deterministic pick; sources in --cap-exempt are not capped)

Usage:
  python3 scripts/dataset/filter_dedupe.py
  python3 scripts/dataset/filter_dedupe.py --max-tris 20000 --classes bipedal,quadrupedal --max-per-skeleton 10
"""
import argparse
import collections
import csv
import hashlib
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sources" / "unimate"
SOURCES = ("mixamo", "objaverse")
TRIANGLES = 4  # glTF primitive mode; a missing `mode` also means triangles


def read_glb(path, want_bin=False):
    """Return (json, bin_chunk_offset). Reads only the JSON chunk unless want_bin."""
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError("not a GLB")
        clen, ctype = struct.unpack("<I4s", f.read(8))
        js = json.loads(f.read(clen))
        bin_off = None
        hdr = f.read(8)
        if len(hdr) == 8:
            bin_off = f.tell()
    return js, bin_off


def skeleton_signature(js):
    """Hash of (joint name, parent index) over the skin joints, in skin order."""
    nodes = js["nodes"]
    parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parent[c] = i
    order = []
    for s in js.get("skins", []):
        for j in s["joints"]:
            if j not in order:
                order.append(j)
    index = {j: k for k, j in enumerate(order)}
    sig = [(nodes[j].get("name", ""), index.get(parent.get(j), -1)) for j in order]
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:12], len(order)


def analyze(path, geom_hash):
    js, bin_off = read_glb(path)
    tris = verts = 0
    skinned = False
    tex = bool(js.get("textures"))
    h = hashlib.sha1()
    hashed_all = bin_off is not None
    with open(path, "rb") as f:
        for m in js.get("meshes", []):
            for p in m["primitives"]:
                attrs = p["attributes"]
                skinned |= "WEIGHTS_0" in attrs and "JOINTS_0" in attrs
                acc = js["accessors"][attrs["POSITION"]]
                verts += acc["count"]
                if p.get("mode", TRIANGLES) == TRIANGLES:
                    tris += (js["accessors"][p["indices"]]["count"] if "indices" in p else acc["count"]) // 3
                if geom_hash:
                    bv = js["bufferViews"][acc["bufferView"]] if "bufferView" in acc else None
                    if bv is None or bin_off is None or "uri" in js["buffers"][bv["buffer"]]:
                        hashed_all = False
                        continue
                    stride = bv.get("byteStride", 12)
                    start = bin_off + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                    f.seek(start)
                    h.update(f.read(stride * (acc["count"] - 1) + 12))
                    if "indices" in p:
                        ia = js["accessors"][p["indices"]]
                        h.update(struct.pack("<II", ia["count"], ia["componentType"]))
    sig, njoints = skeleton_signature(js)
    if not geom_hash or not hashed_all:
        # fall back to counts: exact only in the sense of "very likely identical"
        h.update(f"{verts}:{tris}".encode())
    return dict(tris=tris, verts=verts, joints=njoints, skeleton=sig, textured=int(tex),
                skinned=int(skinned), geom=h.hexdigest()[:16])


def load_categories(src):
    cg = json.loads((SRC / "export" / src / "category_groups.json").read_text())
    return {obj: cat for cat, objs in cg.items() for obj in objs}


def load_flagged(src):
    p = SRC / "export" / src / "filtered_objects.txt"
    if not p.exists():
        return set()
    return {ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def load_clip_counts():
    counts = collections.Counter()
    with open(SRC / "clips.csv", newline="") as f:
        for r in csv.DictReader(f):
            counts[(r["dataset"], r["object_type"])] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tris", type=int, default=20000)
    ap.add_argument("--min-tris", type=int, default=200)
    ap.add_argument("--classes", default="bipedal,quadrupedal", help="comma list, or 'all'")
    ap.add_argument("--min-clips", type=int, default=1, help="objaverse only; mixamo clips are shared")
    ap.add_argument("--max-per-skeleton", type=int, default=10, help="0 = no cap")
    ap.add_argument("--cap-exempt", default="mixamo", help="sources not capped (one shared rig by design)")
    ap.add_argument("--no-geom-hash", action="store_true", help="skip reading vertex buffers (duplicates by counts only)")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "manifest.csv"))
    args = ap.parse_args()

    classes = None if args.classes == "all" else set(args.classes.split(","))
    exempt = set(args.cap_exempt.split(","))
    clip_counts = load_clip_counts()

    assets = []
    for src in args.sources.split(","):
        d = SRC / "canonical_assets" / src
        cats, flagged = load_categories(src), load_flagged(src)
        for p in sorted(d.glob("*.glb")):
            if p.name.startswith("._"):
                continue
            obj = p.stem
            assets.append(dict(source=src, id=obj, path=str(p), category=cats.get("mixamo" if src == "mixamo" else obj, "unknown"),  # mixamo: one shared rig type
                               flagged=obj in flagged,
                               n_clips=clip_counts[(src, "mixamo" if src == "mixamo" else obj)]))

    def work(a):
        try:
            a.update(analyze(a["path"], not args.no_geom_hash))
        except Exception as e:  # unreadable asset: record, do not abort the run
            a.update(error=f"{type(e).__name__}: {e}")
        return a

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(work, assets), 1):
            if i % 500 == 0:
                print(f"  analyzed {i}/{len(assets)}", file=sys.stderr)

    # first failing per-asset filter
    for a in assets:
        a["reason"] = ""
        if "error" in a or not a.get("skinned"):
            a["reason"] = "parse_error"
        elif a["flagged"]:
            a["reason"] = "flagged"
        elif classes and a["category"] not in classes:
            a["reason"] = "class"
        elif a["tris"] > args.max_tris:
            a["reason"] = "too_many_tris"
        elif a["tris"] < args.min_tris:
            a["reason"] = "too_few_tris"
        elif a["source"] == "objaverse" and a["n_clips"] < args.min_clips:
            a["reason"] = "no_clips"

    # dedupe among survivors, in a deterministic order
    key = lambda a: hashlib.sha1(a["id"].encode()).hexdigest()
    seen_geom, per_skel = set(), collections.Counter()
    for a in sorted((a for a in assets if not a["reason"]), key=key):
        g = (a["source"], a["geom"], a["skeleton"])
        if g in seen_geom:
            a["reason"] = "duplicate_mesh"
            continue
        seen_geom.add(g)
        sk = (a["source"], a["skeleton"])
        if args.max_per_skeleton and a["source"] not in exempt and per_skel[sk] >= args.max_per_skeleton:
            a["reason"] = "skeleton_cap"
            continue
        per_skel[sk] += 1

    cols = ["source", "id", "category", "keep", "reason", "tris", "verts", "joints", "skeleton",
            "geom", "n_clips", "textured", "flagged", "path"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for a in assets:
            a["keep"] = int(not a["reason"])
            w.writerow(a)

    kept = [a for a in assets if a["keep"]]
    print(f"\nwrote {out}  ({len(assets)} assets, {len(kept)} kept)")
    print("dropped by reason:", dict(collections.Counter(a["reason"] for a in assets if a["reason"])))
    print("kept per source/category/skeletons:")
    for (s, c), n in sorted(collections.Counter((a["source"], a["category"]) for a in kept).items()):
        sk = len({a["skeleton"] for a in kept if a["source"] == s and a["category"] == c})
        print(f"  {s:9s} {c:18s} {n:5d} meshes  {sk:5d} unique skeletons")


if __name__ == "__main__":
    main()
