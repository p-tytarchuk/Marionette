"""Build a named, skinned rig (GLB) from an Anymate prediction and its role assignment, retarget a clip onto it, and skin the mesh with the result.

Inputs per character (data/anymate/<key>/, written by anymate_predict.py and anymate_roles.py): joints.npy, parents.npy (predicted, model frame),
frame.npz (model frame -> character frame: v_model = (v - center) * scale), roles.json (role -> joint indices), skins_top6.npz (per mesh vertex, per bone).
The mesh is read from data/characters/<key>/model.glb (the vertex order is the one the skins were predicted for).

Rig: the predicted joints rooted at the hips (minimum spanning tree of anymate_roles.build_tree). Role joints get the clean names the retargeter reads
(Hips, Head, Left/Right Thigh/Shin/Foot/Toe, Left/Right Upper Arm/Forearm/Hand; a quadruped's "arm" is the fore limb); the other chain joints between
two role joints are dropped (retarget.label_roles allows at most two helper joints between labelled ones; the dropped joint's skin weight goes to its
nearest kept ancestor), everything else (spine, fingers, tail, ...) keeps its predicted position. Rest rotations are identity, local offsets are
position differences. A predicted bone (parent -> child) skins to the end of the segment nearer the root; the 4 largest weights per vertex are kept.

    python scripts/rigging/anymate_rig.py run --n-biped 4 --n-quad 4 --clips Walking,Jump_3        # build rigs, retarget, write metrics + figures
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "data" / "anymate"
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
import anymate_roles as AR  # noqa: E402
import retarget as RT  # noqa: E402
from convert_clips import qrot, safe_name, write_clip  # noqa: E402

LEG = ["Thigh", "Shin", "Foot", "Toe"]
ARM = ["Upper Arm", "Forearm", "Hand"]


def is_anc(par, a, b) -> bool:
    while b >= 0:
        if b == a:
            return True
        b = par[b]
    return False


def build_rig(key: str) -> dict:
    d = OUT / key
    J, par_pred = np.load(d / "joints.npy"), np.load(d / "parents.npy")
    roles = json.loads((d / "roles.json").read_text())["roles"]
    fr = np.load(d / "frame.npz")
    center, scale = fr["center"], float(fr["scale"])
    W = J / scale + center                                  # joint positions in the character's own frame (metres)
    adj, D = AR.build_tree(J, par_pred)
    hips = roles["hips"]
    tpar, _ = AR.rooted(adj, hips, D)
    n = len(J)
    chains = {k: roles[k] for k in ("leg_L", "leg_R", "arm_L", "arm_R")}

    names: dict[int, str] = {hips: "Hips", roles["head"]: "Head"}
    for i, j in enumerate(roles["spine"][1:-1]):
        names.setdefault(j, f"Spine{i}")
    for s, side in (("L", "Left"), ("R", "Right")):
        for j, lab in zip(chains[f"leg_{s}"], LEG):
            names[j] = f"{side} {lab}"
        for j, lab in zip(chains[f"arm_{s}"], ARM):
            names[j] = f"{side} {lab}"
    roles_used = [j for k in chains for j in chains[k]] + [hips, roles["head"]]
    if len(set(roles_used)) != len(roles_used):
        raise AR.Fail("role_joint_used_twice")

    drop = set()
    for ch in chains.values():
        for a, b in zip(ch[:-1], ch[1:]):
            if not is_anc(tpar, a, b):
                raise AR.Fail("chain_not_descending_from_hips")
            x = tpar[b]
            while x != a:
                drop.add(x)
                x = tpar[x]
    depth = np.array([len(AR.up(tpar, j)) - 1 for j in range(n)])
    kept = sorted((j for j in range(n) if j not in drop), key=lambda j: (depth[j], j))
    new = {j: k for k, j in enumerate(kept)}

    def kept_anc(j):   # nearest kept joint, j itself included
        while j not in new:
            j = tpar[j]
        return j

    parents = [-1 if j == hips else new[kept_anc(tpar[j])] for j in kept]
    Wk = W[kept]
    loc = np.array([Wk[k] if p < 0 else Wk[k] - Wk[p] for k, p in enumerate(parents)])
    nm = [names.get(j, f"joint_{j}") for j in kept]

    # skin: bone b <-> predicted joint i_b with parent par_pred[i_b]; the segment drives the end nearer the root
    sk = np.load(d / "skins_top6.npz")
    idx, wt = sk["index"].astype(int), sk["weight"].astype(float)
    bones = [i for i in range(n) if par_pred[i] != i]
    drv = np.array([new[kept_anc(par_pred[i] if depth[par_pred[i]] <= depth[i] else i)] for i in bones])
    V = len(idx)
    jj, ww = np.zeros((V, 4), np.uint16), np.zeros((V, 4), np.float32)
    for v in range(V):
        acc: dict[int, float] = {}
        for b, w in zip(idx[v], wt[v]):
            if w > 0 and 0 <= b < len(drv):
                acc[int(drv[b])] = acc.get(int(drv[b]), 0.0) + float(w)
        if not acc:
            acc = {0: 1.0}
        top = sorted(acc.items(), key=lambda t: -t[1])[:4]
        tot = sum(w for _, w in top)
        for q, (jn, w) in enumerate(top):
            jj[v, q], ww[v, q] = jn, w / tot
    return dict(names=nm, parents=parents, loc=loc, Wp0=Wk, joints=jj, weights=ww, kept=kept, roles=roles, n_dropped=len(drop), n_pred=n)


def character_mesh(key: str) -> trimesh.Trimesh:
    sc = trimesh.load(str(ROOT / "data" / "characters" / key / "model.glb"), force="scene")
    return trimesh.util.concatenate([g.copy() for g in sc.dump() if isinstance(g, trimesh.Trimesh)])


def write_skinned_glb(path: Path, rig: dict, mesh: trimesh.Trimesh) -> None:
    names, parents, loc, Wp0 = rig["names"], rig["parents"], rig["loc"], rig["Wp0"]
    J = len(names)
    blobs: list[bytes] = []
    views: list[dict] = []
    accs: list[dict] = []

    def add(arr, comp, typ, target=None, minmax=False):
        a = np.ascontiguousarray(arr)
        pad = -sum(len(b) for b in blobs) % 4
        if pad:
            blobs.append(b"\0" * pad)
        off = sum(len(b) for b in blobs)
        blobs.append(a.tobytes())
        bv = dict(buffer=0, byteOffset=off, byteLength=a.nbytes)
        if target:
            bv["target"] = target
        views.append(bv)
        acc = dict(bufferView=len(views) - 1, componentType=comp, count=int(a.shape[0]), type=typ)
        if minmax:
            acc["min"], acc["max"] = a.min(0).tolist(), a.max(0).tolist()
        accs.append(acc)
        return len(accs) - 1

    v = np.asarray(mesh.vertices, "<f4")
    ibm = np.zeros((J, 4, 4), "<f4")
    ibm[:] = np.eye(4, dtype="<f4")
    ibm[:, 3, :3] = -Wp0                                    # column-major translate(-rest position); the rest rotation is identity
    prim = dict(attributes=dict(
        POSITION=add(v, 5126, "VEC3", 34962, True), NORMAL=add(np.asarray(mesh.vertex_normals, "<f4"), 5126, "VEC3", 34962),
        JOINTS_0=add(rig["joints"].astype("<u2"), 5123, "VEC4", 34962), WEIGHTS_0=add(rig["weights"].astype("<f4"), 5126, "VEC4", 34962)),
        indices=add(np.asarray(mesh.faces, "<u4").reshape(-1), 5125, "SCALAR", 34963), material=0)
    ibm_acc = add(ibm.reshape(J, 16), 5126, "MAT4")
    accs[ibm_acc]["count"] = J
    nodes = [dict(name=n, translation=loc[i].tolist()) for i, n in enumerate(names)]
    for i, p in enumerate(parents):
        if p >= 0:
            nodes[p].setdefault("children", []).append(i)
    nodes.append(dict(name="mesh", mesh=0, skin=0))
    blob = b"".join(blobs)
    blob += b"\0" * (-len(blob) % 4)
    js = dict(asset=dict(version="2.0", generator="Marionette anymate_rig.py"), scene=0,
              scenes=[dict(nodes=[i for i, p in enumerate(parents) if p < 0] + [J])], nodes=nodes,
              meshes=[dict(primitives=[prim])], skins=[dict(joints=list(range(J)), inverseBindMatrices=ibm_acc, skeleton=0)],
              materials=[dict(pbrMetallicRoughness=dict(baseColorFactor=[0.75, 0.75, 0.78, 1], metallicFactor=0, roughnessFactor=0.9))],
              accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(blob)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


# ----------------------------------------------------------------------------- retargeting
PARAMS = dict(stride=1.0, vgain=1.0, forelimbs="auto", ik=True, shape=True, tilt_limit=35.0, stance_tilt=12.0, spine_gain=0.6)
ARGS = SimpleNamespace(stride_scale=1.0, vertical_gain=1.0, forelimbs="auto", no_ik=None, no_shaping=False, tilt_limit=35.0, stance_tilt=12.0, spine_gain=0.6,
                       force=True)
CLEAN = {f"{s} {l}": f"{s} {l}" for s in ("Left", "Right") for l in LEG + ARM} | {"Hips": "Hips", "Head": "Head"}   # raw name == clean label


def retarget_on_rig(key: str, clip: str, rig_dir: Path):
    rep, inv = RT.clip_index()
    src, sroles = RT.source_roles(RT.source_path(clip), clip, inv)
    tgt = RT.load_skeleton(rig_dir / "rig.glb")
    troles = RT.label_roles(tgt, CLEAN)
    ok, why, _ = RT.audit(tgt, troles)
    L, root_pos, info, Wp = RT.retarget(src, sroles, tgt, troles, PARAMS, src["fps"])
    info["target_class"] = RT.body_class(tgt, troles)
    info["rest_pose_ok"] = bool(ok)
    info["rest_pose_why"] = ",".join(why)
    root = [j for j in range(len(tgt["names"])) if tgt["parents"][j] < 0][0]
    (rig_dir / "anims").mkdir(exist_ok=True)
    write_clip(rig_dir / "anims" / f"rt__{safe_name(clip, set())}.glb", tgt["names"], tgt["parents"], tgt["loc"], tgt["rot_xyzw"], tgt["scale"], src["fps"],
               L[..., [1, 2, 3, 0]], {root: root_pos}, clip, dict(source_clip=clip, fps=src["fps"], units="m", retargeted=True))
    f = rig_dir / "anims" / "anims.json"
    meta = json.loads(f.read_text()) if f.exists() else {}
    r = inv[clip]
    name = "rt__" + safe_name(clip, set())
    meta[name] = dict(file=f"{name}.glb", source_clip=clip, fps=src["fps"], frames=int(len(L)), duration_s=round(len(L) / src["fps"], 3),
                      loop=bool(int(r["loop_guess"] or 0)), loop_gap_deg=float(r["loop_gap_deg"]), root_travel_m=round(info["root_travel_m"], 3),
                      motion_captions=[t for t in (r["caption"], r["detail"], r["generic"]) if t],
                      retargeted={k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()})
    f.write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return L, root_pos, info, tgt, troles, src["fps"]


def skin_frames(rig: dict, tgt: dict, mesh: trimesh.Trimesh, L, root_pos, frames) -> list[np.ndarray]:
    """Linear blend skinning of the mesh at the given frames (rest rotations are identity)."""
    T = len(L)
    pos = np.tile(tgt["loc"][None], (T, 1, 1))
    root = [j for j in range(len(tgt["names"])) if tgt["parents"][j] < 0][0]
    pos[:, root] = root_pos
    Wp, Wr = RT.fk_pr(tgt["parents"], pos, L)
    v0 = np.asarray(mesh.vertices, float)
    out = []
    for t in frames:
        acc = np.zeros_like(v0)
        for q in range(4):
            j = rig["joints"][:, q].astype(int)
            acc += rig["weights"][:, q, None] * (Wp[t, j] + qrot(Wr[t, j], v0 - tgt["Wp0"][j]))
        out.append(acc)
    return out, Wp


def pick(args) -> list[tuple[str, str]]:
    sel = {"biped": [], "quadruped": []}
    for d in sorted(OUT.iterdir()):
        f = d / "roles.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        m = json.loads((d / "metrics.json").read_text())
        if m.get("status") == "ok" and zlib.crc32(d.name.encode()) % 2 == 1:   # the held-out half of anymate_roles.py
            sel[r["cls"]].append((d.name, r["cls"]))
    out = []
    for cls, n in (("biped", args.n_biped), ("quadruped", args.n_quad)):
        step = max(1, len(sel[cls]) // max(n, 1))
        out += sel[cls][::step][:n]
    return out


def cmd_run(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chars = pick(args)
    clips = args.clips.split(",")
    rows, figs = [], []
    for key, cls in chars:
        rd = OUT / key / "rig"
        rd.mkdir(exist_ok=True)
        try:
            rig = build_rig(key)
            mesh = character_mesh(key)
            assert len(mesh.vertices) == len(rig["joints"]), "mesh and skin rows differ"
            write_skinned_glb(rd / "rig.glb", rig, mesh)
        except Exception as e:
            print(f"{key}: rig FAILED {type(e).__name__}: {e}", flush=True)
            rows.append(dict(key=key, cls=cls, clip="", error=f"rig:{type(e).__name__}:{e}"))
            continue
        for clip in clips:
            row = dict(key=key, cls=cls, clip=clip, n_pred_joints=rig["n_pred"], n_rig_joints=len(rig["names"]))
            try:
                L, root_pos, info, tgt, troles, fps = retarget_on_rig(key, clip, rd)
                row.update({f"anymate_{k}": v for k, v in info.items()})
            except BaseException as e:   # SystemExit included: the retargeter exits on some inputs
                row["error"] = f"retarget:{type(e).__name__}:{e}"
                rows.append(row)
                print(f"{key} {clip}: {row['error']}", flush=True)
                continue
            try:   # the same clip on the character's own rig, same settings
                _, _, info0, _, _, _ = RT.run_one(clip, key, ARGS, write=False)
                row.update({f"rig_{k}": v for k, v in info0.items()})
            except BaseException as e:
                row["rig_error"] = f"{type(e).__name__}:{e}"
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in
                              ("key", "clip", "anymate_reach_fail_frames", "rig_reach_fail_frames", "anymate_skate_mean", "rig_skate_mean",
                               "anymate_rot_steps_over_60deg", "rig_rot_steps_over_60deg", "anymate_rest_pose_ok", "error", "rig_error")}), flush=True)
            T = len(L)
            frames = np.linspace(0, T - 1, 6).round().astype(int)
            sk, Wp = skin_frames(rig, tgt, mesh, L, root_pos, frames)
            figs.append((key, cls, clip, frames, sk, Wp, tgt, mesh))
    if rows:
        cols = sorted({k for r in rows for k in r})
        with open(OUT / "retarget_on_anymate_rigs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)
    for clip in clips:   # one filmstrip sheet per clip
        sel = [x for x in figs if x[2] == clip]
        if not sel:
            continue
        fig, axs = plt.subplots(len(sel), 1, figsize=(15, 2.4 * len(sel)))
        axs = np.atleast_1d(axs)
        for ax, (key, cls, c, frames, sk, Wp, tgt, mesh) in zip(axs, sel):
            span = max(np.ptp(np.asarray(mesh.vertices)[:, 2]), np.ptp(np.asarray(mesh.vertices)[:, 0])) * 1.15
            for k, (t, v) in enumerate(zip(frames, sk)):
                hz = Wp[t, 0, 2]
                s = max(1, len(v) // 1500)
                ax.scatter(v[::s, 2] - hz + k * span, v[::s, 1], s=1, c="0.55")
                for j, p in enumerate(tgt["parents"]):
                    if p >= 0:
                        ax.plot([Wp[t, p, 2] - hz + k * span, Wp[t, j, 2] - hz + k * span], [Wp[t, p, 1], Wp[t, j, 1]], c="tab:red", lw=0.8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_ylabel(f"{cls[:4]} …{key[-8:]}", fontsize=8)
        fig.suptitle(f"'{clip}' retargeted onto Anymate-predicted rigs, mesh skinned with the predicted weights (6 frames, side view)", fontsize=11)
        plt.tight_layout()
        plt.savefig(ROOT / "docs" / "figures" / f"anymate_rigs_{safe_name(clip, set())}.png", dpi=70)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n-biped", type=int, default=4)
    r.add_argument("--n-quad", type=int, default=4)
    r.add_argument("--clips", default="Walking,Jump_3")
    args = ap.parse_args()
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
