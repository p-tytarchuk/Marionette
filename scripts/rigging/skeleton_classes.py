"""Exploratory: can the skeleton alone say biped / quadruped / limbless / winged?  Read-only on the dataset; writes only data/analysis/skeleton_classes/.

  python scripts/rigging/skeleton_classes.py build      # skeletons from the 3D manifest, AnyTop's 70 creatures and the Anymate predictions -> skeletons.json
  python scripts/rigging/skeleton_classes.py eval       # features (names are never used), 2D projections at several yaws, shallow decision trees, report.md
  python scripts/rigging/skeleton_classes.py lottie-build [--shards 8]   # real Lottie rigs (frame 0, the token joint selection) of a few corpus shards, labelled by Qwen's body plan -> lottie_skeletons.json
  python scripts/rigging/skeleton_classes.py lottie-eval  # the same features on them: does structure separate 2leg from 4leg, and what does the 3D-trained tree say

Why: the shape stage will hand the motion stage a 2D skeleton drawn at an unknown yaw, and the movement mode that looks right (walk, flap, slither) depends
on which carriers the skeleton has. A skeleton is a rooted tree with joint positions, so the class has to come from tree shape and geometry, not from names.
Every skeleton, whatever its source, is read the same way (a raw GLB's node hierarchy including ancestor transforms, which gives a Y-up world pose), so a
classifier cannot learn the pipeline instead of the shape. The 2D view of a skeleton at yaw f has x' = z cos f + x sin f and y; f = 0 is the profile of a +Z-facing body.
Labels: the manifest's `category` (bipedal, quadrupedal, avian, serpentine, ...), AnyTop's Truebones groups (FLYING, BIPEDS, SNAKES, QUADROPEDS, MILLIPEDS).
"""
import argparse
import collections
import csv
import io
import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
OUT = ROOT / "data" / "analysis" / "skeleton_classes"
TRUEBONES = ROOT.parent / "Datasets" / "AnyTop" / "dataset" / "truebones" / "zoo" / "truebones_processed"
CLASSES = ["biped", "quadruped", "limbless", "winged", "other"]
CATEGORY = {"bipedal": "biped", "quadrupedal": "quadruped", "serpentine": "limbless", "avian": "winged"}      # manifest category -> class; the rest is "other"
GROUP = {"FLYING": "winged", "BIPEDS": "biped", "SNAKES": "limbless", "QUADROPEDS": "quadruped", "MILLIPEDS": "other"}   # Truebones_skeletons.txt headings
sys.setrecursionlimit(5000)


# ----------------------------------------------------------------------------- reading skeletons
def glb_json(path):
    """The JSON chunk of a GLB (the binary chunk, which holds the meshes, is not read)."""
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError("not a GLB")
        clen, _ = struct.unpack("<I4s", f.read(8))
        return json.loads(f.read(clen))


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def glb_skeleton(path):
    """(parents, positions) of the skin joints in world space: every ancestor node's translation, rotation and scale is applied, so a Z-up armature or a 0.01 scale
    is handled. A joint's parent is its nearest ancestor that is also a joint; several roots get a virtual root at their mean (so the skeleton is one tree)."""
    js = glb_json(path)
    nodes = js["nodes"]
    par = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    joints = []
    for s in js.get("skins", []):
        for j in s["joints"]:
            if j not in joints:
                joints.append(j)
    if len(joints) < 3:
        raise ValueError("fewer than 3 joints")
    memo = {}

    def world(i):
        if i in memo:
            return memo[i]
        n = nodes[i]
        if "matrix" in n:
            M = np.array(n["matrix"], float).reshape(4, 4).T
        else:
            M = np.eye(4)
            M[:3, :3] = quat_matrix(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]), float)[None, :]
            M[:3, 3] = n.get("translation", [0, 0, 0])
        memo[i] = (world(par[i]) @ M) if i in par else M
        return memo[i]

    jset, index = set(joints), {j: k for k, j in enumerate(joints)}

    def joint_parent(j):
        p = par.get(j)
        while p is not None and p not in jset:
            p = par.get(p)
        return index[p] if p is not None else -1

    parents = [joint_parent(j) for j in joints]
    P = np.array([world(j)[:3, 3] for j in joints])
    return add_virtual_root(parents, P)


def add_virtual_root(parents, P):
    roots = [i for i, p in enumerate(parents) if p < 0]
    if len(roots) <= 1:
        return list(parents), np.asarray(P, float)
    parents = list(parents) + [-1]
    parents = [len(parents) - 1 if p < 0 and i < len(parents) - 1 else p for i, p in enumerate(parents)]
    return parents, np.vstack([P, P[roots].mean(0)])


class _SafeUnpickler(pickle.Unpickler):
    """cond.npy is a pickled dict inside a .npy (a third-party download). np.load(allow_pickle=True) would run whatever the file says, so only numpy array types and
    plain containers are allowed here; anything else raises instead of executing."""
    ALLOWED = {("builtins", n) for n in ("dict", "list", "tuple", "set", "str", "int", "float", "complex", "bytes", "bool", "slice", "range")} | {
        ("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or (module.startswith("numpy") and name in ("_reconstruct", "scalar", "_frombuffer")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name} in a pickled .npy")


def load_npy_dict(path):
    """the dict stored with np.save(path, dict): read the .npy header, then unpickle the rest with the restricted loader"""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)      # step over the header
        obj = _SafeUnpickler(io.BytesIO(f.read())).load()                     # np.save of an object stores the whole 0-d object array as a pickle
    return obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == () else obj


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    skel, skipped = [], collections.Counter()
    # ---- 3D manifest: one skeleton per distinct skeleton hash ----
    seen = set()
    for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", newline="")):
        if r["skeleton"] in seen:
            continue
        seen.add(r["skeleton"])
        try:
            parents, P = glb_skeleton(r["path"])
        except Exception as e:
            skipped[f"manifest: {type(e).__name__}: {e}"] += 1
            continue
        if not np.isfinite(P).all() or np.ptp(P, axis=0).max() <= 1e-9:
            skipped["manifest: degenerate positions"] += 1
            continue
        skel.append(dict(id=f"{r['source']}_{r['id']}", source="manifest", label=CATEGORY.get(r["category"], "other"), detail=r["category"], parents=parents, P=P))
    # ---- AnyTop / Truebones: 70 creatures, rest pose = cumulative offsets (a BVH rest pose has zero rotations) ----
    cond = load_npy_dict(TRUEBONES / "cond.npy")
    group, label = None, {}
    for line in open(TRUEBONES / "Truebones_skeletons.txt", encoding="utf-8"):
        line = line.strip()
        if line.startswith("####"):
            group = next((v for k, v in GROUP.items() if k in line.upper()), "other")
        elif line:
            label[line] = (group, line)
    for name, c in cond.items():
        parents = [int(p) for p in c["parents"]]
        parents = [-1 if (k == 0 or p == k) else p for k, p in enumerate(parents)]                      # AnyTop marks the root by parent 0 or by itself
        P = np.zeros((len(parents), 3))
        for k, p in enumerate(parents):
            P[k] = (P[p] if p >= 0 else 0) + c["offsets"][k]
        cls, _ = label.get(name, ("other", name))
        skel.append(dict(id=f"truebones_{name}", source="truebones", label=cls, detail=name, parents=parents, P=P))
    # ---- Anymate: predicted skeleton and the dataset's own skeleton of the same character ----
    for d in sorted((ROOT / "data" / "anymate").glob("*/")):
        if d.name.startswith("._") or not (d / "joints.npy").exists():
            continue
        src, _, cid = d.name.partition("_")
        row = next((r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", newline="")) if r["source"] == src and r["id"] == cid), None)
        cat = row["category"] if row else "?"
        for tag, jf, pf in (("pred", "joints.npy", "parents.npy"), ("gt", "gt_joints.npy", "gt_parents.npy")):
            parents, P = [int(p) for p in np.load(d / pf)], np.load(d / jf).astype(float)
            parents = [-1 if (p < 0 or p == k) else p for k, p in enumerate(parents)]
            parents, P = add_virtual_root(parents, P)
            skel.append(dict(id=f"anymate_{tag}_{d.name}", source=f"anymate_{tag}", label=CATEGORY.get(cat, "other"), detail=cat, parents=parents, P=P))
    json.dump([dict(sk, P=np.round(sk["P"], 6).tolist()) for sk in skel], open(OUT / "skeletons.json", "w"))
    c = collections.Counter((s["source"], s["label"]) for s in skel)
    print(f"{len(skel)} skeletons -> {OUT / 'skeletons.json'}")
    for k in sorted(c):
        print("  ", k, c[k])
    print("skipped:", dict(skipped))


# ----------------------------------------------------------------------------- features (2D, names never used)
YAWS = [0, 30, 60, 90, 120, 150]
FEATURES = ["n_pruned", "chain_frac", "leaf_frac", "branch", "limbs", "low_limbs", "mid_limbs", "high_limbs", "log_aspect", "y_cent", "max_chain", "diam_frac", "path_ratio", "arm_leg_ratio"]
NO_ASPECT = [f for f in FEATURES if f != "log_aspect"]           # the one feature that depends on the yaw (the horizontal extent): a variant without it is tested too


def project(P, yaw):
    f = np.radians(yaw)
    return np.stack([P[:, 2] * np.cos(f) + P[:, 0] * np.sin(f), P[:, 1]], 1)


def tree_adjacency(parents, P):
    """One undirected tree from the parent links and the 3D positions: a minimum spanning tree in which a predicted edge costs 1% of its length, so the given edges are kept,
    stray pieces are joined and cycles are broken (the same idea as build_tree in anymate_roles.py; 97 of the 221 Anymate predictions have a cycle or no root).
    It is built once per skeleton, in 3D, and reused at every yaw, so overlapping joints in a 2D view cannot change the structure."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    n = len(P)
    W = np.linalg.norm(P[:, None] - P[None], axis=-1) + 1e-6
    for i, p in enumerate(parents):
        if 0 <= p < n and p != i:
            W[i, p] = W[p, i] = W[i, p] * 0.01
    np.fill_diagonal(W, 0)
    T = minimum_spanning_tree(csr_matrix(W)).toarray()
    adj = [[] for _ in range(n)]
    for a, b in zip(*np.nonzero(T)):
        adj[a].append(int(b))
        adj[b].append(int(a))
    return adj


def farthest(adj, w, src):
    """(joint, distance) farthest from src along the tree, w = edge length function"""
    dist, stack = {src: 0.0}, [src]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w(u, v)
                stack.append(v)
    far = max(dist, key=dist.get)
    return far, dist[far]


def prune(adj, Q, S, tau=0.06):
    """Remove short twigs (fingers, toes, ears, eyes): a chain that ends in a leaf and joins the rest at a joint with 3 or more neighbours, shorter than tau x the body size.
    It makes a 50-joint hand-and-finger rig and a 10-joint Lottie rig comparable. Returns (adjacency, positions) of the remaining joints, renumbered."""
    nb = [set(a) for a in adj]
    alive = set(range(len(adj)))
    w = lambda u, v: float(np.hypot(*(Q[u] - Q[v])))
    changed = True
    while changed:
        changed = False
        for leaf in [i for i in sorted(alive) if len(nb[i]) == 1]:
            if leaf not in alive or len(nb[leaf]) != 1:
                continue
            chain, prev, cur, length = [leaf], -1, leaf, 0.0
            while True:
                nxt = [v for v in nb[cur] if v != prev]
                if not nxt:
                    break
                length += w(cur, nxt[0])
                prev, cur = cur, nxt[0]
                if len(nb[cur]) != 2:
                    break
                chain.append(cur)
            if len(nb[cur]) >= 3 and length < tau * S:
                for node in chain:
                    alive.discard(node)
                    for v in nb[node]:
                        nb[v].discard(node)
                    nb[node] = set()
                changed = True
    keep = sorted(alive)
    index = {k: i for i, k in enumerate(keep)}
    return [[index[v] for v in sorted(nb[k])] for k in keep], Q[keep]


def features(adj, Q):
    """Scale-free features of an undirected tree with 2D joint positions Q (x' horizontal, y up). The root plays no part. The tree is pruned of short twigs first.
    A "limb" is a leaf chain, up to the first joint with 3 or more neighbours, that is at least 12% of the body size and 2 bones long; low / mid / high = where its
    end is: bottom 15% / between / top 40% of the height. diam_frac = longest path / total bone length (1 for a snake, small for a branching body);
    arm_leg_ratio = mean high limb / mean low limb length (wings are long, arms are shorter than legs). Distinct joints that overlap in the picture still count."""
    ext = np.ptp(Q, axis=0)
    H, W = max(ext[1], 1e-9), max(ext[0], 1e-9)
    S = max(H, W)
    adj, Q = prune(adj, Q, S)
    n = len(adj)
    deg = np.array([len(a) for a in adj])
    w = lambda u, v: float(np.hypot(*(Q[u] - Q[v])))
    leaves = np.flatnonzero(deg == 1)
    clen, cbones = np.zeros(len(leaves)), np.zeros(len(leaves), int)
    for k, leaf in enumerate(leaves):
        prev, cur = -1, leaf
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            if not nxt:
                break
            clen[k] += w(cur, nxt[0])
            cbones[k] += 1
            prev, cur = cur, nxt[0]
            if deg[cur] != 2:
                break
    limb = (clen >= 0.12 * S) & (cbones >= 2)
    rel = (Q[leaves, 1] - Q[:, 1].min()) / H if len(leaves) else np.zeros(0)
    low, high = limb & (rel <= 0.15), limb & (rel >= 0.6)
    tot = sum(w(u, v) for u in range(n) for v in adj[u]) / 2
    a, _ = farthest(adj, w, 0)
    _, diameter = farthest(adj, w, a)
    return dict(n_pruned=n, chain_frac=float((deg <= 2).mean()), leaf_frac=len(leaves) / n, branch=int((deg >= 3).sum()), limbs=int(limb.sum()), low_limbs=int(low.sum()),
                mid_limbs=int((limb & ~low & ~high).sum()), high_limbs=int(high.sum()), log_aspect=float(np.log10(W / H)), y_cent=float((Q[:, 1].mean() - Q[:, 1].min()) / H),
                max_chain=float(clen.max() / S) if len(clen) else 0.0, diam_frac=float(diameter / max(tot, 1e-9)), path_ratio=float(diameter / S),
                arm_leg_ratio=float(clen[high].mean() / clen[low].mean()) if low.any() and high.any() else 0.0)


def table(skel, yaws):
    X, y, g, meta = [], [], [], []
    for s in skel:
        P = np.asarray(s["P"])
        adj = tree_adjacency(s["parents"], P)
        for yaw in yaws:
            f = features(adj, project(P, yaw))
            X.append([f[k] for k in FEATURES])
            y.append(CLASSES.index(s["label"]))
            g.append(s["id"])
            meta.append((s["source"], s["detail"], yaw))
    return np.array(X, float), np.array(y), np.array(g), meta


def report_rows(y, pred, title):
    """per-class recall / precision and the confusion matrix, as markdown"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    lines = [f"**{title}**", "", "| class | n | recall | precision |", "|---|---|---|---|"]
    for k, c in enumerate(CLASSES):
        n = cm[k].sum()
        if n:
            lines.append(f"| {c} | {n} | {cm[k, k] / n:.2f} | {cm[k, k] / max(cm[:, k].sum(), 1):.2f} |")
    rec = [cm[k, k] / cm[k].sum() for k in range(len(CLASSES)) if cm[k].sum()]
    lines += ["", f"balanced accuracy {np.mean(rec):.3f}, plain accuracy {np.trace(cm) / cm.sum():.3f}", "", "| true \\ predicted | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    lines += [f"| {c} | " + " | ".join(str(v) for v in cm[k]) + " |" for k, c in enumerate(CLASSES)]
    return "\n".join(lines) + "\n", cm, float(np.mean(rec))


def bal4(cm):
    """mean recall over the four target classes (biped, quadruped, limbless, winged); "other" is not a target, only a place a skeleton can end up"""
    return float(np.mean([cm[k, k] / cm[k].sum() for k in range(4) if cm[k].sum()]))


def evaluate():
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier, export_text
    skel = json.load(open(OUT / "skeletons.json"))
    anymate_ids = {s["id"].split("anymate_gt_")[1] for s in skel if s["id"].startswith("anymate_gt_")}
    man = [s for s in skel if s["source"] == "manifest" and s["id"] not in anymate_ids]
    tb = [s for s in skel if s["source"] == "truebones"]
    am = [s for s in skel if s["source"].startswith("anymate")]
    mk = lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=0)
    Xm, ym, gm, _ = table(man, YAWS)
    Xt, yt, gt, _ = table(tb, YAWS)
    Xa, ya, ga, ma = table(am, YAWS)
    is_pred = np.array([m[0] == "anymate_pred" for m in ma])
    yawcol = np.tile(YAWS, len(gm) // len(YAWS))
    rng = np.random.RandomState(0)
    ids = sorted(set(gm))
    test_ids = set(rng.choice(ids, len(ids) // 3, replace=False))
    is_te = np.array([g in test_ids for g in gm])
    out = ["# Skeleton classes: biped / quadruped / limbless / winged (exploratory)", "",
           f"Skeletons: {len(man)} from the 3D manifest (one per distinct skeleton), {len(tb)} AnyTop creatures, {len(am) // 2} Anymate characters x (predicted, original). "
           f"Each is projected at yaws {YAWS} (0 = profile); features never use joint names; short twigs (fingers, toes) are pruned first. Shallow decision trees (depth 4, balanced class weights). "
           "'bal4' = mean recall over the four target classes; 'other' (insects, marine, rigid objects, uncertain) is a fifth class that a skeleton can be sent to.", ""]
    summary = {}
    for name, cols in (("with the aspect feature", FEATURES), ("without it (yaw-free)", NO_ASPECT)):
        ix = [FEATURES.index(f) for f in cols]
        out += [f"## Features {name}", ""]
        pred = np.zeros_like(ym)
        for tr, te in GroupKFold(5).split(Xm, ym, gm):
            pred[te] = mk().fit(Xm[tr][:, ix], ym[tr]).predict(Xm[te][:, ix])
        txt, cm, b5 = report_rows(ym, pred, "1. Cross-validation inside the 3D manifest (all yaws, folds grouped by skeleton)")
        out += [txt, f"bal4 {bal4(cm):.3f}", ""]
        r1 = (b5, bal4(cm))
        final = mk().fit(Xm[:, ix], ym)
        txt, cm, b5 = report_rows(yt, final.predict(Xt[:, ix]), "2. Train on the 3D manifest, test on AnyTop's 70 creatures (all yaws)")
        out += [txt, f"bal4 {bal4(cm):.3f}", ""]
        r2 = (b5, bal4(cm))
        profile = mk().fit(Xm[~is_te & (yawcol == 0)][:, ix], ym[~is_te & (yawcol == 0)])
        allyaw = mk().fit(Xm[~is_te][:, ix], ym[~is_te])
        out += ["**3. Yaw: bal4 on held-out skeletons by view**", "", "| yaw | trained on profile only | trained on all yaws |", "|---|---|---|"]
        curve = []
        for yaw in YAWS:
            sel = is_te & (yawcol == yaw)
            a = bal4(confusion_matrix(ym[sel], profile.predict(Xm[sel][:, ix]), labels=range(5)))
            b = bal4(confusion_matrix(ym[sel], allyaw.predict(Xm[sel][:, ix]), labels=range(5)))
            curve.append((yaw, a, b))
            out.append(f"| {yaw} | {a:.3f} | {b:.3f} |")
        out += [""]
        pa = final.predict(Xa[:, ix])
        r4 = []
        for tag, sel in (("original skeleton", ~is_pred), ("Anymate-predicted skeleton", is_pred)):
            txt, cm, b5 = report_rows(ya[sel], pa[sel], f"4. Anymate characters, {tag} (biped / quadruped only)")
            out += [txt]
            r4.append(float(np.trace(cm[:2, :2]) / cm[:2].sum()))
        out += ["**Rules of the tree trained on the whole manifest**", "", "```", export_text(final, feature_names=cols), "```", ""]
        summary[name] = dict(cv=r1, truebones=r2, yaw=curve, anymate_acc=r4)
    (OUT / "report.md").write_text("\n".join(out), encoding="utf-8")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(out))


def _lottie_shard(job):
    """worker: the wanted clips of one corpus shard -> skeleton records (frame 0, the joints the tokens use, 2D, y up)"""
    import glob
    import orjson
    import re
    import zstandard
    import lottie_tokens as lt
    shard, want = job
    path = glob.glob(str(ROOT.parent / "Datasets" / "LottieAnimation-660K" / "snapshots" / "*" / "data" / f"metadata-{shard:05d}-of-00068.jsonl.zst"))[0]
    rx, out = re.compile(rb'^\{"id":"(\d+)"'), []
    with open(path, "rb") as fh:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
            m = rx.match(line[:40])
            if not m or m.group(1).decode() not in want:
                continue
            cid = m.group(1).decode()
            lj = orjson.loads(line)["lottie_json"]
            lj = orjson.loads(lj) if isinstance(lj, str) else lj
            fk = lt.lottie_fk(lj)
            if fk is None:
                continue
            par0, P0 = np.asarray(fk["parents"]), np.asarray(fk["pos"])
            idx = lt.select_joints(par0, P0, 96, "component")
            if len(idx) < 3:
                continue
            remap = {j: k for k, j in enumerate(idx)}
            parents = [remap.get(int(par0[j]), -1) if par0[j] >= 0 else -1 for j in idx]
            out.append(dict(id=f"lottie_{cid}", source="lottie", label=want[cid][0], detail=want[cid][1], parents=parents, P=np.round(P0[0][idx], 5).tolist()))
    return out


def lottie_build(n_shards):
    import multiprocessing
    import random
    import lottie_tokens as lt
    rel = lt.load_relevant()                                               # shard -> {id: (body_plan, subject, caption)}
    rng = random.Random(0)
    jobs = []
    for sh in rng.sample(sorted(rel), n_shards):
        four = [i for i, v in rel[sh].items() if v[0] == "4leg"]
        two = [i for i, v in rel[sh].items() if v[0] == "2leg"]
        pick = four + rng.sample(two, min(len(two), max(60, len(four) * 2)))
        jobs.append((sh, {i: (rel[sh][i][0], rel[sh][i][1]) for i in pick}))
    with multiprocessing.Pool(min(6, n_shards)) as pool:
        recs = [r for part in pool.map(_lottie_shard, jobs) for r in part]
    json.dump(recs, open(OUT / "lottie_skeletons.json", "w"))
    c = collections.Counter(r["label"] for r in recs)
    print(f"{len(recs)} Lottie rigs from shards {[j[0] for j in jobs]}: {dict(c)}; median joints {int(np.median([len(r['parents']) for r in recs]))}")


def lottie_eval():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    lot = json.load(open(OUT / "lottie_skeletons.json"))
    rows, y = [], []
    for r in lot:
        P = np.asarray(r["P"])
        f = features(tree_adjacency(r["parents"], P), P)
        rows.append([f[k] for k in FEATURES])
        y.append(1 if r["label"] == "4leg" else 0)
    X, y = np.array(rows, float), np.array(y)
    out = ["# Lottie rigs: does structure separate 2leg from 4leg? (labels: Qwen body plan)", "",
           f"{len(y)} rigs ({int(y.sum())} four-leg, {int((1 - y).sum())} two-leg), native 2D (no yaw to vary), frame 0, the joints the tokens use. Median joints {int(np.median([len(r['parents']) for r in lot]))}.", "",
           "| feature | AUC four-leg vs two-leg |", "|---|---|"]
    out += [f"| {f} | {roc_auc_score(y, X[:, k]):.3f} |" for k, f in enumerate(FEATURES)]
    pred = np.zeros_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pred[te] = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=0).fit(X[tr], y[tr]).predict(X[te])
    rec = [float((pred[y == c] == c).mean()) for c in (0, 1)]
    out += ["", f"A depth-3 tree trained on the Lottie rigs themselves (5-fold CV): recall two-leg {rec[0]:.2f}, four-leg {rec[1]:.2f}, balanced accuracy {np.mean(rec):.3f}.", ""]
    # the tree trained on the 3D manifest (all yaws, all features), applied unchanged
    skel = json.load(open(OUT / "skeletons.json"))
    man = [s for s in skel if s["source"] == "manifest"]
    Xm, ym, _, _ = table(man, YAWS)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=0).fit(Xm, ym)
    pl = tree.predict(X)
    out += ["**The tree trained on the 3D manifest, applied unchanged to the Lottie rigs**", "", "| Qwen body plan \\ predicted | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for c, name in ((0, "2leg"), (1, "4leg")):
        out.append(f"| {name} (n={int((y == c).sum())}) | " + " | ".join(str(int((pl[y == c] == k).sum())) for k in range(len(CLASSES))) + " |")
    out.append("")
    (OUT / "lottie_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "eval", "lottie-build", "lottie-eval"])
    ap.add_argument("--shards", type=int, default=8, help="lottie-build: how many corpus shards to sample")
    a = ap.parse_args()
    {"build": build, "eval": evaluate, "lottie-eval": lottie_eval}.get(a.cmd, lambda: lottie_build(a.shards))()
