#!/usr/bin/env python3
"""Normalize kept characters (data/manifest.csv, keep=1) to the dataset conventions.

For every asset it writes data/characters/<source>_<id>/model.glb plus normalization.json, with:
  * Y-up, facing +Z   (yaw fixed from the l_hip/r_hip pair in export/<src>/face_joint_names.json)
  * feet on the ground plane y = 0 (lowest vertices, 0.1% trimmed), body centred on x/z
  * real-world scale in metres (see "Scale policy")
The transform is baked into the vertices, normals, joint offsets (scale) and inverse bind matrices;
the odd 90-degree Armature rotation the source files carry is folded into the skeleton root, so the
mesh and the skeleton end up in the same frame under identity parent nodes. Skin weights are untouched.
normalization.json holds the 4x4 world transform T (new = T @ old) so motion clips can be moved
into the same frame (scale the root translation by T).

Scale policy (true size cannot be recovered: canonical assets are normalized to skeleton
diameter 2 and Objaverse units are arbitrary):
  mixamo      one scale for all characters so that the median height is --mixamo-height
              (relative sizes between characters are preserved)
  objaverse   each character is scaled to its class height: --biped-height / --quadruped-height
              (--other-height for other classes); --objaverse-scale canonical instead uses one
              scale per class (median height -> target), keeping the relative proportions
  overrides   --overrides CSV with columns id,height_m sets the final height of single assets
"Height" is the vertical extent of the mesh after grounding.

Skipped (reported in data/normalize_report.csv): morph targets/sparse accessors, compressed
meshes, meshes under joints, bind pose inconsistent with the node hierarchy, shear/reflection.
Flags (kept, but look at them): facing_unverified, hip_tilt, up_suspect, yaw_changed.

Usage:
  python3 scripts/dataset/normalize.py --limit 20 --dry-run
  python3 scripts/dataset/normalize.py
"""
import argparse
import collections
import csv
import json
import math
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FACE_DIR = ROOT / "sources" / "unimate" / "export"
CLASS_OF = {"bipedal": "biped", "quadrupedal": "quadruped"}
GROUP_SOURCES = {"mixamo"}  # one scale per source+class (relative sizes are real there)
CT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
BAD_EXT = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"}


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- GLB io
class Glb:
    def __init__(self, path):
        data = Path(path).read_bytes()
        magic, _, _ = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF":
            raise Skip("not_glb")
        clen, _ = struct.unpack_from("<I4s", data, 12)
        self.js = json.loads(data[20:20 + clen])
        off = 20 + clen
        if off + 8 > len(data):
            raise Skip("no_bin_chunk")
        blen, _ = struct.unpack_from("<I4s", data, off)
        self.bin = bytearray(data[off + 8:off + 8 + blen])

    def save(self, path):
        j = json.dumps(self.js, separators=(",", ":")).encode()
        j += b" " * (-len(j) % 4)
        b = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(j) + 8 + len(b)
        Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(j), b"JSON")
                               + j + struct.pack("<I4s", len(b), b"BIN\0") + b)

    def acc(self, i):
        """Writable (count, ncomp) numpy view of accessor i, honouring byteStride."""
        a = self.js["accessors"][i]
        if "sparse" in a:
            raise Skip("sparse_accessor")
        bv = self.js["bufferViews"][a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        n = NC[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", dt.itemsize * n)
        return np.ndarray((a["count"], n), dt, self.bin, off, (stride, dt.itemsize))


# ----------------------------------------------------------------------------- math
def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def local_mat(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def decompose(M):
    A = M[:3, :3]
    sc = np.linalg.norm(A, axis=0)
    R = A / sc
    if np.linalg.det(R) < 0:
        raise Skip("reflection")
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-4:
        raise Skip("shear")
    return M[:3, 3], mat_to_quat(R), sc


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ----------------------------------------------------------------------------- scene structure
def parents(js):
    par = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []):
            par[c] = i
    return par


def world_mats(js):
    nodes, par, memo = js["nodes"], parents(js), {}

    def G(i):
        if i not in memo:
            M = local_mat(nodes[i])
            memo[i] = G(par[i]) @ M if i in par else M
        return memo[i]

    return G


def structure(g):
    js = g.js
    nodes = js["nodes"]
    if BAD_EXT & set(js.get("extensionsRequired", [])):
        raise Skip("compressed_mesh")
    if not js.get("skins"):
        raise Skip("no_skin")
    if any("targets" in p for m in js["meshes"] for p in m["primitives"]):
        raise Skip("morph_targets")
    par = parents(js)
    joints = []
    for s in js["skins"]:
        for j in s["joints"]:
            if j not in joints:
                joints.append(j)
    J = set(joints)
    roots = [j for j in joints if par.get(j) not in J]
    chain = set()
    for r in roots:
        p = par.get(r)
        while p is not None:
            if p in J:
                raise Skip("interleaved_joints")
            chain.add(p)
            p = par.get(p)
    meshes = [i for i, n in enumerate(nodes) if "mesh" in n]
    for m in meshes:
        if "skin" not in nodes[m]:
            raise Skip("unskinned_mesh")
        p = par.get(m)
        while p is not None:
            if p in J:
                raise Skip("mesh_under_joint")
            p = par.get(p)
    attr = collections.defaultdict(set)  # attribute name -> accessor indices
    for mi in {nodes[m]["mesh"] for m in meshes}:
        for p in js["meshes"][mi]["primitives"]:
            for k in ("POSITION", "NORMAL", "TANGENT"):
                if k in p["attributes"]:
                    attr[k].add(p["attributes"][k])
    ibm = {s["inverseBindMatrices"] for s in js["skins"] if "inverseBindMatrices" in s}
    if len(ibm) != len(js["skins"]):
        raise Skip("no_inverse_bind_matrices")
    return dict(joints=joints, roots=roots, chain=chain, meshes=meshes, attr=attr, ibm=ibm)


def bind_error(g, st, G):
    """max |G_j @ IBM_j - I| over the skin joints (0 when the bind pose is the node rest pose)."""
    err = 0.0
    for s in g.js["skins"]:
        M = g.acc(s["inverseBindMatrices"]).astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)
        for k, j in enumerate(s["joints"]):
            err = max(err, np.abs(G(j) @ M[k] - np.eye(4)).max())
    return err


# ----------------------------------------------------------------------------- measuring
def vertices(g, st):
    return np.concatenate([g.acc(i).astype(float) for i in sorted(st["attr"]["POSITION"])])


def facing_yaw(g, st, G, face, snap_deg):
    """Yaw (radians, about +Y) that turns the character's forward onto +Z, or None."""
    names = {}
    for j in st["joints"]:
        names.setdefault(g.js["nodes"][j].get("name"), j)
    l, r = (face or {}).get("l_hip", {}).get("raw"), (face or {}).get("r_hip", {}).get("raw")
    if l not in names or r not in names:
        return None, None
    v = G(names[l])[:3, 3] - G(names[r])[:3, 3]
    fwd = np.cross(v, [0, 1, 0])
    tilt = abs(v[1]) / max(np.linalg.norm(v), 1e-9)
    if math.hypot(fwd[0], fwd[2]) < 0.2 * max(np.linalg.norm(v), 1e-9):
        return None, tilt
    deg = math.degrees(-math.atan2(fwd[0], fwd[2]))
    k = round(deg / 90)
    if abs(deg - 90 * k) <= snap_deg:
        deg = 90.0 * k
    return math.radians(((deg + 180) % 360) - 180), tilt


def measure(g, st, face, args):
    G = world_mats(g.js)
    berr = bind_error(g, st, G)
    if berr > 1e-3:
        raise Skip(f"bind_inconsistent({berr:.3g})")
    V = vertices(g, st)
    yaw, tilt = facing_yaw(g, st, G, face, args.snap_deg)
    Ry = rot_y(yaw or 0.0)
    V1 = V @ Ry.T
    lo, hi = np.quantile(V1[:, 1], [args.trim, 1.0])
    ext = V1.max(0) - V1.min(0)
    return dict(G=G, V=V, yaw=yaw, tilt=tilt, Ry=Ry, V1=V1, ground=float(lo), height=float(hi - lo),
                cx=float((V1[:, 0].min() + V1[:, 0].max()) / 2), cz=float((V1[:, 2].min() + V1[:, 2].max()) / 2),
                ext=ext, bind_err=berr)


# ----------------------------------------------------------------------------- applying
def apply_transform(g, st, m, s):
    """Bake new = T @ old into the file; returns the 4x4 matrix T."""
    js, nodes = g.js, g.js["nodes"]
    A = s * m["Ry"]
    t = -s * np.array([m["cx"], m["ground"], m["cz"]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = A, t
    G = m["G"]

    # Skeleton roots take the whole old world transform (the chain above them becomes identity);
    # the uniform scale goes into every joint offset, not into a node scale, so the file stores
    # metric bone lengths and node scales stay at their authored values.
    new_root = {r: decompose(T @ G(r)) for r in st["roots"]}
    par_children = [(r, nodes[r]) for r in st["roots"]]
    seen = set()
    while par_children:
        k, n = par_children.pop()
        if k in seen:
            continue
        seen.add(k)
        if k not in new_root:
            if "matrix" in n:
                tr, q, sc = decompose(local_mat(n))
                n.pop("matrix")
                n["translation"], n["rotation"], n["scale"] = tr.tolist(), q.tolist(), sc.tolist()
            if "translation" in n:
                n["translation"] = (s * np.array(n["translation"], float)).tolist()
        for c in n.get("children", []):
            par_children.append((c, nodes[c]))
    for r, (tr, q, sc) in new_root.items():
        n = nodes[r]
        n.pop("matrix", None)
        n["translation"], n["rotation"], n["scale"] = tr.tolist(), q.tolist(), (sc / s).tolist()
    for c in list(st["chain"]) + st["meshes"]:
        for k in ("matrix", "translation", "rotation", "scale"):
            nodes[c].pop(k, None)

    # stale pipeline metadata on the old armature node (a rest root position in the old frame)
    for c in st["chain"]:
        ex = nodes[c].get("extras", {})
        if isinstance(ex.get("canonical_motion_rest_root"), str):
            p = np.array(json.loads(ex["canonical_motion_rest_root"]), float)
            ex["canonical_motion_rest_root"] = json.dumps((A @ p + t).tolist())

    for i in st["attr"]["POSITION"]:
        d = g.acc(i)
        d[:] = (d.astype(float) @ A.T + t).astype("<f4")
        a = js["accessors"][i]
        a["min"], a["max"] = d.astype(float).min(0).tolist(), d.astype(float).max(0).tolist()
    Rn = m["Ry"]
    for i in st["attr"]["NORMAL"]:
        d = g.acc(i)
        d[:] = (d.astype(float) @ Rn.T).astype("<f4")
    for i in st["attr"]["TANGENT"]:
        d = g.acc(i)
        d[:, :3] = (d[:, :3].astype(float) @ Rn.T).astype("<f4")
    Gn = world_mats(js)  # fresh: the node tree above has changed
    for sk in js["skins"]:
        d = g.acc(sk["inverseBindMatrices"])
        M = np.array([np.linalg.inv(Gn(j)) for j in sk["joints"]])
        d[:] = M.transpose(0, 2, 1).reshape(-1, 16).astype("<f4")
    return T


def rest_world(g, st, idx=None):
    """Skin the rest pose: sum_k w_k * G_j IBM_j v, in the frame the file defines."""
    js, nodes = g.js, g.js["nodes"]
    G = world_mats(js)
    skin_of = {}
    out = []
    for m in st["meshes"]:
        sk = js["skins"][nodes[m]["skin"]]
        M = g.acc(sk["inverseBindMatrices"]).astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)
        jm = np.array([G(j) @ M[k] for k, j in enumerate(sk["joints"])])
        for p in js["meshes"][nodes[m]["mesh"]]["primitives"]:
            v = g.acc(p["attributes"]["POSITION"]).astype(float)
            vh = np.c_[v, np.ones(len(v))]
            res = np.zeros((len(v), 3))
            for s in ("0", "1"):
                if f"JOINTS_{s}" not in p["attributes"]:
                    continue
                J = g.acc(p["attributes"][f"JOINTS_{s}"]).astype(int)
                wa = js["accessors"][p["attributes"][f"WEIGHTS_{s}"]]
                W = g.acc(p["attributes"][f"WEIGHTS_{s}"]).astype(float)
                W /= {5126: 1.0, 5121: 255.0, 5123: 65535.0}[wa["componentType"]]
                for k in range(J.shape[1]):
                    res += W[:, k:k + 1] * np.einsum("nij,nj->ni", jm[J[:, k]], vh)[:, :3]
            out.append(res)
    return np.concatenate(out)


# ----------------------------------------------------------------------------- driver
def load_faces():
    faces = {}
    for src in ("mixamo", "objaverse"):
        p = FACE_DIR / src / "face_joint_names.json"
        faces[src] = json.loads(p.read_text()) if p.exists() else {}
    return faces


def face_for(faces, a):
    return faces[a["source"]].get("mixamo" if a["source"] == "mixamo" else a["id"])


def target_height(a, args, overrides):
    if a["id"] in overrides:
        return overrides[a["id"]]
    return {"biped": args.biped_height, "quadruped": args.quadruped_height}.get(a["cls"], args.other_height)


def process(a, args, faces, group_scale, overrides):
    row = dict(source=a["source"], id=a["id"], cls=a["cls"], status="ok", reason="", scale="", yaw_deg="",
               facing="", height_in="", height_out="", scale_mode="", flags="", verify_err="", out="")
    try:
        g = Glb(a["path"])
        st = structure(g)
        m = measure(g, st, face_for(faces, a), args)
        flags = []
        if m["yaw"] is None:
            flags.append("facing_unverified")
        elif abs(m["yaw"]) > 1e-6:
            flags.append("yaw_changed")
        if m["tilt"] is not None and m["tilt"] > 0.25:
            flags.append("hip_tilt")
        up_ext = m["ext"][1]
        if a["cls"] == "biped" and up_ext < 0.5 * max(m["ext"][0], m["ext"][2]):
            flags.append("up_suspect")
        if a["id"] in overrides:
            mode, s = "override", overrides[a["id"]] / m["height"]
        elif (a["source"], a["cls"]) in group_scale:
            mode, s = "group", group_scale[(a["source"], a["cls"])]
        else:
            mode, s = "height", target_height(a, args, overrides) / m["height"]
        if not (1e-4 < s < 1e4):
            raise Skip(f"absurd_scale({s:.3g})")
        row.update(scale=f"{s:.6g}", yaw_deg=f"{math.degrees(m['yaw'] or 0):.2f}",
                   facing="hips" if m["yaw"] is not None else "none", height_in=f"{m['height']:.4g}",
                   height_out=f"{m['height'] * s:.4g}", scale_mode=mode, flags=" ".join(flags))
        T = apply_transform(g, st, m, s)

        # verify: skinning the rest pose of the new file must equal T applied to the old one
        g0 = Glb(a["path"])
        old = rest_world(g0, structure(g0))
        new = rest_world(g, st)
        err = np.abs(new - (old @ T[:3, :3].T + T[:3, 3])).max()
        ground_err = abs(np.quantile(new[:, 1], args.trim))
        if err > 1e-3 * max(m["height"] * s, 1e-9) + 1e-5 or ground_err > 1e-3:
            raise Skip(f"verify_failed(err={err:.3g},ground={ground_err:.3g})")
        row["verify_err"] = f"{err:.2e}"

        if not args.dry_run:
            d = Path(args.out_dir) / f"{a['source']}_{a['id']}"
            d.mkdir(parents=True, exist_ok=True)
            g.save(d / "model.glb")
            ext = new.max(0) - new.min(0)
            (d / "normalization.json").write_text(json.dumps(dict(
                id=a["id"], source=a["source"], **{"class": a["cls"]}, scale=s, scale_mode=mode,
                yaw_deg=math.degrees(m["yaw"] or 0), translation=T[:3, 3].tolist(), T=T.tolist(),
                convention="new = T @ old; Y-up, +Z forward, metres, ground at y=0, centred on x/z",
                height_m=float(new[:, 1].max() - np.quantile(new[:, 1], args.trim)), extent_m=ext.tolist(),
                flags=flags, source_path=a["path"]), indent=2))
            out = (d / "model.glb").resolve()
            row["out"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
        else:
            row["out"] = "(dry run)"
    except Skip as e:
        row.update(status="skipped", reason=str(e))
    except Exception as e:  # keep going; the report lists it
        row.update(status="failed", reason=f"{type(e).__name__}: {e}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "characters"))
    ap.add_argument("--report", default=str(ROOT / "data" / "normalize_report.csv"))
    ap.add_argument("--biped-height", type=float, default=1.7)
    ap.add_argument("--quadruped-height", type=float, default=0.9, help="placeholder: pick per your data")
    ap.add_argument("--other-height", type=float, default=1.0)
    ap.add_argument("--mixamo-height", type=float, default=1.75, help="median Mixamo character height")
    ap.add_argument("--objaverse-scale", choices=["height", "canonical"], default="height")
    ap.add_argument("--overrides", help="CSV with columns id,height_m")
    ap.add_argument("--trim", type=float, default=0.001, help="fraction of lowest vertices ignored for the ground")
    ap.add_argument("--snap-deg", type=float, default=5.0, help="snap yaw to a multiple of 90 within this angle")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", help="comma list of ids to process")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="compute and verify, write nothing but the report")
    args = ap.parse_args()

    assets = []
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["keep"] == "1":
                assets.append(dict(source=r["source"], id=r["id"], path=r["path"],
                                   cls=CLASS_OF.get(r["category"], r["category"])))
    if args.ids:
        want = set(args.ids.split(","))
        assets = [a for a in assets if a["id"] in want]
    if args.limit:
        assets = assets[:args.limit]
    overrides = {}
    if args.overrides:
        with open(args.overrides, newline="") as f:
            overrides = {r["id"]: float(r["height_m"]) for r in csv.DictReader(f)}
    faces = load_faces()

    # one scale per (source, class) where relative sizes are meaningful
    group_scale = {}
    group_keys = {(a["source"], a["cls"]) for a in assets if a["source"] in GROUP_SOURCES
                  or args.objaverse_scale == "canonical"}
    for key in sorted(group_keys):
        members = [a for a in assets if (a["source"], a["cls"]) == key and a["id"] not in overrides]

        def height_of(a):
            try:
                g = Glb(a["path"])
                st = structure(g)
                return measure(g, st, face_for(faces, a), args)["height"]
            except Skip:
                return None

        with ThreadPoolExecutor(args.workers) as ex:
            hs = [h for h in ex.map(height_of, members) if h]
        if hs:
            tgt = args.mixamo_height if key[0] == "mixamo" else target_height(dict(id="", cls=key[1]), args, {})
            group_scale[key] = tgt / float(np.median(hs))
            print(f"group {key}: {len(hs)} assets, median height {np.median(hs):.3f} -> scale {group_scale[key]:.4f}",
                  file=sys.stderr)

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, row in enumerate(ex.map(lambda a: process(a, args, faces, group_scale, overrides), assets), 1):
            rows.append(row)
            if i % 250 == 0:
                print(f"  {i}/{len(assets)}", file=sys.stderr)

    cols = list(rows[0]) if rows else []
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\nreport: {args.report}  ({len(rows)} assets{', dry run' if args.dry_run else ''})")
    print("status:", dict(collections.Counter(r["status"] for r in rows)))
    print("skipped/failed reasons:", dict(collections.Counter(r["reason"].split("(")[0] for r in rows if r["reason"])))
    print("flags:", dict(collections.Counter(f for r in rows for f in r["flags"].split())))
    for cls in sorted({r["cls"] for r in rows}):
        h = [float(r["height_out"]) for r in rows if r["cls"] == cls and r["status"] == "ok"]
        if h:
            print(f"  {cls:10s} n={len(h):5d} height_out m: min {min(h):.2f} median {np.median(h):.2f} max {max(h):.2f}")
    ve = [float(r["verify_err"]) for r in rows if r["verify_err"]]
    if ve:
        print(f"max verify error (rest-pose skinning vs T applied to the original): {max(ve):.2e}")


if __name__ == "__main__":
    main()
