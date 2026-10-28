"""Run the pretrained Anymate predictors (joints -> connectivity -> skinning) on the CPU of this Mac.

Anymate (../Anymate, https://github.com/yfde/Anymate) assumes CUDA; this driver patches only what is needed from outside, the repo is untouched:
torch.load gets map_location="cpu" and weights_only=False (the checkpoints store a pickled model_config; they are the Hugging Face files
yfdeng/Anymate, downloaded by the user). The Michelangelo and PointLLM weights are not needed: the checkpoints use the 'bert' point encoder
with load_bert=False. It changes into the repo root itself.

Model frame (measured on Anymate_test.pt): bbox centred on the origin, longest side = 2, Y up. `pc` = 8192 surface points + normals,
`vox` = filled 64^3 occupancy (indices of occupied cells, cell = floor((p + 1) / 2 * 64)) dilated by one cell towards lower indices, as in the
repo's reduce(). binvox (a Linux binary) is replaced by trimesh voxelization: IoU 0.84-0.96 against the stored `vox` on six test assets.
The stored `mesh_pc` of the test set is in another frame (feet at y = 0, longest side 1); to_model_frame() converts it.

    python scripts/rigging/anymate_predict.py test --n 5                                 # test split: stored inputs, and inputs rebuilt from the mesh
    python scripts/rigging/anymate_predict.py characters --n-mixamo 4 --n-biped 4 --n-quad 4  # my characters -> data/anymate/<id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPO = Path(os.environ.get("ANYMATE_REPO", ROOT.parent / "Anymate"))
TEST_PT = ROOT.parent / "Datasets" / "Anymate" / "Anymate_test.pt"
os.chdir(REPO)                                      # the repo uses paths relative to its root
sys.path[:0] = [str(REPO), str(REPO / "ThirdParty" / "PointLLM")]

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{"map_location": "cpu", "weights_only": False, **k})   # noqa: E731

from Anymate.args import anymate_args  # noqa: E402
from Anymate.utils.utils import get_connectivity, get_joint, get_skinning, load_checkpoint  # noqa: E402

DEV = "cpu"
N_POINTS = 8192
RES = 64


def load_models():
    a = anymate_args
    return [load_checkpoint(p, DEV, a.num_joints) for p in (a.checkpoint_joint, a.checkpoint_conn, a.checkpoint_skin)]


def to_model_frame(v: np.ndarray) -> np.ndarray:
    """Test-set mesh_pc frame (feet on y = 0, longest side 1) -> the model frame (centred, longest side 2)."""
    v = v.copy()
    v[:, 0] *= 2
    v[:, 2] *= 2
    v[:, 1] = 2 * v[:, 1] - v[:, 1].max()
    return v


def normalize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre the bbox on the origin and scale the longest side to 2. Returns (v', center, scale): v' = (v - center) * scale."""
    c = 0.5 * (v.min(0) + v.max(0))
    s = 2.0 / float((v.max(0) - v.min(0)).max())
    return (v - c) * s, c, s


def voxelize(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Occupied cells (N, 3) int of the 64^3 grid over [-1, 1]^3: solid fill, then the dilation of Anymate's reduce()."""
    vg = trimesh.Trimesh(v, f, process=False).voxelized(2.0 / RES).fill()
    g = np.unique(np.clip(np.floor((vg.points + 1) / 2 * RES).astype(int), 0, RES - 1), axis=0)
    a = np.zeros((RES,) * 3, bool)
    a[tuple(g.T)] = True
    a[:-1] |= a[1:].copy()
    a[:, :-1] |= a[:, 1:].copy()
    a[:, :, :-1] |= a[:, :, 1:].copy()
    return np.argwhere(a)


def make_inputs(v: np.ndarray, f: np.ndarray, seed: int = 0) -> dict:
    """pc (8192, 6), vox (N, 3) int8, vertices (V, 6) for a mesh already in the model frame."""
    m = trimesh.Trimesh(v, f, process=False)
    np.random.seed(seed)   # trimesh samples from numpy's global generator
    pts, fi = m.sample(N_POINTS, return_index=True)
    pc = np.concatenate([pts, m.face_normals[fi]], axis=1).astype(np.float32)
    vert = np.concatenate([m.vertices, m.vertex_normals], axis=1).astype(np.float32)
    return {"pc": torch.from_numpy(pc), "vox": torch.from_numpy(voxelize(v, f)).to(torch.int8), "vertices": torch.from_numpy(vert)}


def predict(models, inp: dict):
    mj, mc, ms = models
    joints = get_joint(inp["pc"], mj, device=DEV, vox=inp["vox"])
    conns = get_connectivity(inp["pc"], joints, mc, device=DEV)
    skins = get_skinning(inp["pc"], joints, conns, ms, vertices=inp["vertices"], device=DEV)
    return joints, conns, skins


def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    d = torch.cdist(a, b)
    return 0.5 * (d.min(0).values.mean() + d.min(1).values.mean()).item()


def pr(pred: torch.Tensor, gt: torch.Tensor, thr: float) -> tuple[float, float]:
    d = torch.cdist(gt, pred)
    return (d.min(0).values < thr).float().mean().item(), (d.min(1).values < thr).float().mean().item()


def cmd_test(args) -> None:
    data = torch.load(TEST_PT, mmap=True, weights_only=True)
    models = load_models()
    rng = np.random.default_rng(args.seed)
    print("columns: joints gt/pred | chamfer, P/R@0.1 with stored pc+vox | same with inputs rebuilt from the mesh | skin net on the GT skeleton: top-bone agreement")
    for i in rng.choice(len(data), args.n, replace=False):
        x = data[int(i)]
        n = int(x["joints_num"])
        gt_j = x["joints"][:n]
        v = to_model_frame(x["mesh_pc"][:, :3].numpy())
        vert = torch.cat([torch.from_numpy(v), x["mesh_pc"][:, 3:]], 1).float()
        t = time.time()
        j, c, _ = predict(models, {"pc": x["pc"], "vox": x["vox"], "vertices": vert})
        dt = time.time() - t
        p, r = pr(j, gt_j, 0.1)
        inp = make_inputs(v, x["mesh_face"].numpy())
        j2, c2, _ = predict(models, inp)
        p2, r2 = pr(j2, gt_j, 0.1)
        s_gt = get_skinning(x["pc"], gt_j, x["conns"][:n].long(), models[2], vertices=vert, bones=x["bones"][: int(x["bones_num"])], device=DEV)
        top_gt = x["mesh_skins_index"].long().gather(1, x["mesh_skins_weight"].float().argmax(1, keepdim=True)).squeeze(1)
        agree = (s_gt.argmax(1) == top_gt).float().mean().item()
        print(f"{x['name'][:26]:26s} {n:3d}/{len(j):3d} | {chamfer(j, gt_j):.3f} {p:.2f}/{r:.2f} | {len(j2):3d} {chamfer(j2, gt_j):.3f} {p2:.2f}/{r2:.2f} | "
              f"skin {agree:.2f}  ({dt:.0f}s)", flush=True)


def scene_mesh(path: Path) -> trimesh.Trimesh:
    sc = trimesh.load(str(path), force="scene")
    parts = [g.copy() for g in sc.dump()]
    return trimesh.util.concatenate([p for p in parts if isinstance(p, trimesh.Trimesh)])


def pick_characters(args) -> list[tuple[str, str, str]]:
    allrows = list(csv.DictReader(open(ROOT / "data" / "retarget_targets.csv")))
    rows = [r for r in allrows if r["status"] == "ok"]
    rng = np.random.default_rng(args.seed)
    status_of = {r["key"]: r["status"] for r in allrows}
    out = []
    for name, sel, n in (("mixamo", [r for r in rows if r["source"] == "mixamo" and r["cls"] == "biped"], args.n_mixamo),
                         ("biped", [r for r in rows if r["source"] == "objaverse" and r["cls"] == "biped"], args.n_biped),
                         ("quad", [r for r in allrows if r["cls"] == "quadruped" and r["status"] in args.quad_status.split(",")], args.n_quad)):
        for k in rng.choice(len(sel), min(n, len(sel)), replace=False):
            out.append((sel[int(k)]["key"], name, sel[int(k)]["status"]))
    for key in args.keys or []:
        out.append((key, "named", status_of.get(key, "")))
    return out


def add_skeleton(scene: trimesh.Scene, j: np.ndarray, par: np.ndarray, color, r: float) -> None:
    for p in j:
        s = trimesh.creation.icosphere(radius=r)
        s.apply_translation(p)
        s.visual.face_colors = color
        scene.add_geometry(s)
    for i, q in enumerate(par):
        if q < 0 or q == i:
            continue
        a, b = j[q], j[i]
        h = float(np.linalg.norm(b - a))
        if h < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=r * 0.5, height=h)
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (b - a) / h))
        cyl.apply_translation((a + b) / 2)
        cyl.visual.face_colors = color
        scene.add_geometry(cyl)


def cmd_characters(args) -> None:
    sys.path.insert(0, str(HERE))
    from retarget import load_skeleton  # noqa: E402
    models = load_models()
    out_root = ROOT / "data" / "anymate"
    results = []
    for key, group, status in pick_characters(args):
        if args.skip_existing and (out_root / key / "metrics.json").exists():
            results.append(json.loads((out_root / key / "metrics.json").read_text()))
            continue
        cdir = ROOT / "data" / "characters" / key
        t0 = time.time()
        try:
            mesh = scene_mesh(cdir / "model.glb")
            sk = load_skeleton(cdir / "model.glb")
            v, c, s = normalize(np.asarray(mesh.vertices, float))
            gt_j = torch.tensor((sk["Wp0"] - c) * s, dtype=torch.float32)
            inp = make_inputs(v, np.asarray(mesh.faces))
            j, conns, skins = predict(models, inp)
        except Exception as e:   # one bad character must not stop the batch
            print(f"{key}: FAILED {type(e).__name__}: {e}", flush=True)
            results.append({"key": key, "group": group, "error": f"{type(e).__name__}: {e}"})
            continue
        p05, r05 = pr(j, gt_j, 0.05)
        p10, r10 = pr(j, gt_j, 0.10)
        m = {"key": key, "group": group, "status": status, "n_gt": len(gt_j), "n_pred": len(j), "roots_pred": int((conns == torch.arange(len(conns))).sum()),
             "chamfer_norm": chamfer(j, gt_j), "chamfer_m": chamfer(j, gt_j) / s, "scale_to_unit": s, "extent_m": 2.0 / s,
             "P@0.05": p05, "R@0.05": r05, "P@0.1": p10, "R@0.1": r10, "vox_cells": int(len(inp["vox"])), "verts": int(len(v)), "seconds": round(time.time() - t0, 1)}
        results.append(m)
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}), flush=True)
        od = out_root / key
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / "joints.npy", j.numpy())
        np.save(od / "parents.npy", conns.numpy())
        np.save(od / "gt_joints.npy", gt_j.numpy())
        np.save(od / "gt_parents.npy", np.asarray(sk["parents"]))
        np.savez_compressed(od / "frame.npz", center=c, scale=s)
        sw, si = skins.topk(min(6, skins.shape[1]), dim=1)
        np.savez_compressed(od / "skins_top6.npz", index=si.numpy().astype(np.int8), weight=sw.numpy().astype(np.float16))
        (od / "metrics.json").write_text(json.dumps(m, indent=1))
        if args.no_vis:
            continue
        # visual check: mesh coloured by the predicted dominant bone (translucent), predicted skeleton red/blue, existing rig green
        pal = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0], [166, 86, 40], [247, 129, 191], [153, 153, 153], [30, 200, 200], [200, 200, 30]])
        top = skins.argmax(1).numpy()
        vm = trimesh.Trimesh(v, np.asarray(mesh.faces), process=False)
        vm.visual.vertex_colors = np.c_[pal[top % 10], np.full(len(v), 140)].astype(np.uint8)
        sc = trimesh.Scene(vm)
        add_skeleton(sc, j.numpy(), conns.numpy(), [255, 0, 0, 255], 0.012)
        add_skeleton(sc, gt_j.numpy(), np.asarray(sk["parents"]), [0, 160, 0, 255], 0.012)
        sc.export(od / "vis.glb")
    (out_root / "summary.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test")
    t.add_argument("--n", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("characters")
    c.add_argument("--n-mixamo", type=int, default=4)
    c.add_argument("--n-biped", type=int, default=4)
    c.add_argument("--n-quad", type=int, default=4)
    c.add_argument("--keys", nargs="*", help="extra character folder names")
    c.add_argument("--quad-status", default="ok", help="comma list of retarget_targets.csv statuses to draw quadrupeds from (ok,rest_pose)")
    c.add_argument("--skip-existing", action="store_true")
    c.add_argument("--no-vis", action="store_true", help="do not write vis.glb")
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"test": cmd_test, "characters": cmd_characters}[args.cmd](args)


if __name__ == "__main__":
    main()
