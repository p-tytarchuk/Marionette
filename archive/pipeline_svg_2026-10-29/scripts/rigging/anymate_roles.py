"""Map the joints Anymate predicts (positions + parent index, no names) onto the retargeter's roles: hips, head, spine, leg_L/R, arm_L/R.

The roles are the ones retarget.label_roles() derives from joint names (hips, head, spine path, leg = Thigh/Shin/Foot[/Toe], arm = Upper Arm/Forearm/Hand;
for a quadruped "leg" = hind limb, "arm" = fore limb; the character's left is +X, +Z is forward, Y up, as in data/characters).

Method (geometry and topology only, the class biped / quadruped is an input, as in the project plan):
  1. the predicted parent array is turned into one tree (minimum spanning tree where predicted edges cost 1% of their length: it keeps them, joins
     disconnected pieces and breaks cycles; some predictions have no root or two);
  2. limb end points are leaves: biped feet = lowest leaf on each side of the body, hands = leaf farthest from the midline outside the leg subtree,
     head = highest leaf; quadruped feet = four low leaf clusters (two pairs along Z), head = most forward remaining leaf;
  3. hips = the joint where the two leg paths meet (biped: lowest common ancestor of the feet, tree rooted at the head; quadruped: highest joint
     on the path between the two hind feet), arm start likewise;
  4. the role joints are picked on each limb path at the arc-length fractions where the character's own rig has them, measured as the median over a
     calibration half of the characters (the dense predicted skeleton has many more joints than the rig: fingers, twist bones, extra spine joints).

    python scripts/rigging/anymate_roles.py eval            # calibrate on half of data/anymate/*, score both halves, write roles.json per character
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "data" / "anymate"
HEIGHT_RULE = True   # quadrupeds: knee, ankle, elbow and wrist from calibrated heights above the ground instead of arc-length fractions
LEG, ARM = ["thigh", "knee", "ankle", "toe"], ["upper_arm", "elbow", "wrist"]


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- tree helpers
def build_tree(J: np.ndarray, parents: np.ndarray):
    n = len(J)
    D = np.linalg.norm(J[:, None] - J[None], axis=-1)
    W = D + 1e-6
    for i, p in enumerate(parents):
        if 0 <= p < n and p != i:
            W[i, p] = W[p, i] = W[i, p] * 0.01
    np.fill_diagonal(W, 0)
    T = minimum_spanning_tree(csr_matrix(W)).toarray()
    adj = [[] for _ in range(n)]
    for a, b in zip(*np.nonzero(T)):
        adj[a].append(int(b))
        adj[b].append(int(a))
    return adj, D


def rooted(adj, r, D):
    n = len(adj)
    par, cum, order, seen = [-1] * n, np.zeros(n), [r], [False] * n
    seen[r] = True
    for u in order:
        for v in adj[u]:
            if not seen[v]:
                seen[v], par[v], cum[v] = True, u, cum[u] + D[u, v]
                order.append(v)
    return par, cum


def up(par, j):
    out = [j]
    while par[j] >= 0:
        j = par[j]
        out.append(j)
    return out


def lca(par, a, b):
    sa = set(up(par, a))
    return next(x for x in up(par, b) if x in sa)


def path(par, a, b):
    pa, pb = up(par, a), up(par, b)
    m = lca(par, a, b)
    return pa[: pa.index(m) + 1] + pb[: pb.index(m)][::-1]


def trunk_node(p, adj, D):
    """The joint of path p where the rest of the body attaches: the one with the longest branch hanging off the path."""
    onp = set(p)

    def reach(u, frm):
        best, stack = 0.0, [(u, frm, 0.0)]
        while stack:
            v, f, d = stack.pop()
            best = max(best, d)
            stack += [(w, v, d + D[v, w]) for w in adj[v] if w != f and w not in onp]
        return best

    r = [max([reach(w, v) + D[v, w] for w in adj[v] if w not in onp], default=0.0) for v in p]
    return p[int(np.argmax(r))]


def cum_along(p, D):
    return np.r_[0.0, np.cumsum([D[p[i], p[i + 1]] for i in range(len(p) - 1)])]


# ----------------------------------------------------------------------------- limb paths
def character_points(key: str, n: int = 60000) -> np.ndarray:
    """The character's mesh vertices in the model frame (centred, longest side 2), cached as data/anymate/<key>/pts.npy."""
    f = OUT / key / "pts.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    import trimesh
    sc = trimesh.load(str(ROOT / "data" / "characters" / key / "model.glb"), force="scene")
    v = np.concatenate([np.asarray(g.vertices, float) for g in sc.dump() if hasattr(g, "vertices")])
    fr = np.load(OUT / key / "frame.npz")
    p = (v - fr["center"]) * float(fr["scale"])
    if len(p) > n:
        p = p[np.random.default_rng(0).choice(len(p), n, replace=False)]
    np.save(f, p.astype(np.float16))
    return p.astype(np.float32)


def feet_from_mesh(J: np.ndarray, leaves: list[int], pts: np.ndarray):
    """The four feet of a standing quadruped from the mesh: the lowest points fall in four blobs on the ground plane; each blob is matched to the nearest
    predicted leaf. Returns (hind, fore), each [left, right] joint indices (+Z is forward, the character's left is +X)."""
    import itertools
    from scipy.cluster.hierarchy import fcluster, linkage
    y0, Hm = float(pts[:, 1].min()), float(np.ptp(pts[:, 1]))
    ext = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 2])))
    low = pts[pts[:, 1] < y0 + 0.12 * Hm][:, [0, 2]]
    if len(low) < 40:
        raise Fail("no_foot_points")
    lab = fcluster(linkage(low, "ward"), min(6, len(low) // 10), "maxclust")
    cen = np.array([low[lab == i].mean(0) for i in np.unique(lab) if (lab == i).sum() >= 0.03 * len(low)])
    if len(cen) < 4:
        raise Fail("fewer_than_4_foot_blobs")
    xm, best = float(np.median(pts[:, 0])), None
    for combo in itertools.combinations(range(len(cen)), 4):
        P = cen[list(combo)]
        o = np.argsort(P[:, 1])
        hind, fore = P[o[:2]], P[o[2:]]
        sc = 0.0
        for pr in (hind, fore):
            sc += 10 * (abs(pr[0, 0] - pr[1, 0]) < 0.04 * ext) + abs(pr[0, 1] - pr[1, 1]) / ext + abs(pr[:, 0].sum() - 2 * xm) / ext
        sc += 10 * (fore[:, 1].mean() - hind[:, 1].mean() < 0.15 * ext)
        if best is None or sc < best[0]:
            best = (sc, P[o])
    from scipy.optimize import linear_sum_assignment
    cand = [i for i in range(len(J)) if J[i, 1] < y0 + 0.3 * Hm]   # leaves or not: a predicted limb often ends a little above its foot
    if len(cand) < 4:
        raise Fail("no_joints_near_the_ground")
    isleaf = set(leaves)
    cost = np.array([[np.hypot(*(J[i, [0, 2]] - c)) + (0 if i in isleaf else 0.03 * ext) + 0.3 * (J[i, 1] - y0) for i in cand] for c in best[1]])
    rows_, cols_ = linear_sum_assignment(cost)
    got = [cand[j] for j in cols_]
    hind, fore = sorted(got[:2], key=lambda i: -J[i, 0]), sorted(got[2:], key=lambda i: -J[i, 0])
    return hind, fore


def find_paths(J: np.ndarray, parents: np.ndarray, cls: str, pts: np.ndarray | None = None) -> dict:
    """hips, spine (hips -> head leaf), leg_L/R and arm_L/R paths (junction excluded, end leaf included), as joint index lists."""
    adj, D = build_tree(J, parents)
    n = len(J)
    leaves = [i for i in range(n) if len(adj[i]) == 1]
    y, xm = J[:, 1], float(np.median(J[:, 0]))
    ymin, H = float(y.min()), float(np.ptp(y))
    if len(leaves) < 5 or H < 1e-6:
        raise Fail("too_few_leaves")
    if cls == "biped":
        head_leaf = max(leaves, key=lambda i: y[i])
        par, cum = rooted(adj, head_leaf, D)
        feet = {}
        for s, sg in (("L", 1), ("R", -1)):
            c = [i for i in leaves if sg * (J[i, 0] - xm) > 0 and y[i] < ymin + 0.3 * H]
            if not c:
                raise Fail("no_foot_" + s)
            feet[s] = min(c, key=lambda i: (round(float(y[i]), 3), -cum[i]))
        hips = lca(par, feet["L"], feet["R"])
        below = {i for i in range(n) if hips in up(par, i)}
        hands = {}
        for s, sg in (("L", 1), ("R", -1)):
            c = [i for i in leaves if i not in below and i != head_leaf and sg * (J[i, 0] - xm) > 0]
            if not c:
                raise Fail("no_hand_" + s)
            hands[s] = max(c, key=lambda i: abs(J[i, 0] - xm))
        top = lca(par, hands["L"], hands["R"])
        if hips == head_leaf or hips == top:
            raise Fail("degenerate_junction")
        out = dict(hips=hips, spine=path(par, hips, head_leaf))
        for s in "LR":
            out[f"leg_{s}"] = path(par, hips, feet[s])[1:]
            out[f"arm_{s}"] = path(par, top, hands[s])[1:]
    else:
        ext = float(max(np.ptp(J[:, 0]), np.ptp(J[:, 2])))
        low = [i for i in leaves if y[i] < ymin + 0.25 * H]
        groups: list[list[int]] = []
        for i in low:   # single linkage in the ground plane
            hit = [g for g in groups if any(np.hypot(*(J[i, [0, 2]] - J[k, [0, 2]])) < 0.08 * ext for k in g)]
            merged = [i] + [k for g in hit for k in g]
            groups = [g for g in groups if g not in hit] + [merged]
        if pts is not None:
            hind, fore = feet_from_mesh(J, leaves, pts)
        else:
            reps = [min(g, key=lambda i: y[i]) for g in groups]
            if len(reps) > 4:
                reps = sorted(reps, key=lambda i: -abs(J[i, 0] - xm))[:4]
            if len(reps) < 4:
                raise Fail("fewer_than_4_feet")
            reps.sort(key=lambda i: J[i, 2])
            hind, fore = sorted(reps[:2], key=lambda i: -J[i, 0]), sorted(reps[2:], key=lambda i: -J[i, 0])
        par, cum = rooted(adj, 0, D)
        hips = trunk_node(path(par, hind[0], hind[1]), adj, D)
        chest = trunk_node(path(par, fore[0], fore[1]), adj, D)
        lowset = {k for g in groups for k in g}
        rest = [i for i in leaves if i not in lowset]
        if not rest:
            raise Fail("no_head")
        head_leaf = max(rest, key=lambda i: J[i, 2])
        if hips == chest or hips == head_leaf:
            raise Fail("degenerate_junction")
        out = dict(hips=hips)
        for k, s in enumerate("LR"):
            out[f"leg_{s}"] = path(par, hips, hind[k])[1:]
            out[f"arm_{s}"] = path(par, chest, fore[k])[1:]
        # the rigs put the hips on the midline level with the hind thighs; Anymate attaches the hind legs to a spine joint further forward,
        # so the hips joint is the predicted joint nearest the midpoint of the two thigh starts (outside the limb paths)
        if min(len(out["leg_L"]), len(out["leg_R"])) >= 2:
            mid = 0.5 * (J[out["leg_L"][0]] + J[out["leg_R"][0]])
            limbs = {j for k in ("leg_L", "leg_R", "arm_L", "arm_R") for j in out[k]}
            out["hips"] = min((i for i in range(n) if i not in limbs), key=lambda i: np.linalg.norm(J[i] - mid))
        out["spine"] = path(par, out["hips"], head_leaf)
    for k in ("leg_L", "leg_R", "arm_L", "arm_R"):
        if len(out[k]) < 3:
            raise Fail("short_" + k)
    out["D"] = D
    out["J"] = J
    out["y0"] = float(pts[:, 1].min()) if pts is not None else float(J[:, 1].min())   # ground level
    return out


# ----------------------------------------------------------------------------- calibration and assignment
def pick(p: list[int], D: np.ndarray, fr: list[float], J=None, y0: float = 0.0, h: float = 1.0, heights=None) -> list[int]:
    """Role joints on a limb path. Role i sits at arc-length fraction fr[i]; when heights[i] is not None it sits at that height above the ground,
    in units of the hips height h (a standing limb descends, so height pins the knee, ankle and wrist better than arc length does on a quadruped)."""
    c = cum_along(p, D)
    c = c / max(c[-1], 1e-9)
    ks = []
    for i, f in enumerate(fr):
        lo = ks[-1] + 1 if ks else 0
        if heights is not None and heights[i] is not None and J is not None and lo < len(p):
            hh = (J[p, 1] - y0) / h
            k = lo + int(np.argmin(np.abs(hh[lo:] - heights[i])))
        else:
            k = int(np.argmin(np.abs(c - f)))
            if ks and k <= ks[-1]:
                k = ks[-1] + 1
        ks.append(k)
    while ks and ks[-1] >= len(p):   # a short limb path: drop the last role (the toe) rather than fail
        ks.pop()
    if len(ks) < 3:
        raise Fail("chain_too_short")
    return [p[k] for k in ks]


def assign(paths: dict, cal: dict) -> dict:
    D = paths["D"]
    sp = paths["spine"]
    k = int(np.argmin(np.abs(cum_along(sp, D) / max(cum_along(sp, D)[-1], 1e-9) - cal["head"])))
    k = max(k, 1)
    roles = dict(hips=paths["hips"], head=sp[k], spine=sp[: k + 1])
    J, y0 = paths["J"], paths["y0"]
    h = max(float(J[paths["hips"], 1] - y0), 1e-3)
    for s in "LR":
        roles[f"leg_{s}"] = pick(paths[f"leg_{s}"], D, cal["leg"], J, y0, h, cal.get("leg_h"))
        roles[f"arm_{s}"] = pick(paths[f"arm_{s}"], D, cal["arm"], J, y0, h, cal.get("arm_h"))
    return roles


# ----------------------------------------------------------------------------- evaluation against the rig's own roles
def gt_roles(key: str):
    sys.path.insert(0, str(HERE))
    from retarget import clean_names, label_roles, load_skeleton, rig_of  # noqa: E402
    ds, rid = rig_of(key)
    sk = load_skeleton(ROOT / "data" / "characters" / key / "model.glb")
    return label_roles(sk, clean_names(ds, rid))


def role_joints(roles: dict) -> dict[str, int]:
    out = {"hips": roles["hips"], "head": roles["head"]}
    for s in "LR":
        for nm, i in zip(LEG, roles[f"leg_{s}"]):
            out[f"leg_{s}.{nm}"] = i
        for nm, i in zip(ARM, roles[f"arm_{s}"] or []):
            out[f"arm_{s}.{nm}"] = i
    return out


def load_char(d: Path):
    m = json.loads((d / "metrics.json").read_text())
    return m, np.load(d / "joints.npy"), np.load(d / "parents.npy"), np.load(d / "gt_joints.npy")


def cmd_eval(args) -> None:
    global HEIGHT_RULE
    HEIGHT_RULE = not args.no_height
    rows = []
    for d in sorted(OUT.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m, J, par, G = load_char(d)
        cls = "quadruped" if m["group"] == "quad" else "biped"
        try:
            gr = gt_roles(m["key"])
        except Exception as e:
            rows.append(dict(key=m["key"], cls=cls, status=m.get("status"), err=f"gt_{type(e).__name__}"))
            continue
        try:
            P = find_paths(J, par, cls, character_points(m["key"]) if cls == "quadruped" and not args.no_mesh_feet else None)
        except Fail as e:
            rows.append(dict(key=m["key"], cls=cls, status=m.get("status"), err=str(e)))
            continue
        rows.append(dict(key=m["key"], cls=cls, status=m.get("status"), J=J, G=G, gr=gr, P=P))
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    ok = [r for r in rows if "P" in r]
    print(f"{len(rows)} characters with predictions; {len(ok)} with paths found; failures: {dict(collections.Counter(r['err'] for r in rows if 'err' in r))}")
    train = [r for r in ok if zlib.crc32(r["key"].encode()) % 2 == 0]
    # calibration: where the rig's role joints sit along the predicted paths
    cal = {}
    for cls in ("biped", "quadruped"):
        acc = collections.defaultdict(list)
        for r in train:
            if r["cls"] != cls:
                continue
            D, G, gr, P = r["P"]["D"], r["G"], r["gr"], r["P"]
            for s in "LR":
                for kind, names in (("leg", LEG), ("arm", ARM)):
                    p = P[f"{kind}_{s}"]
                    c = cum_along(p, D)
                    c = c / max(c[-1], 1e-9)
                    for ri, gi in enumerate((gr[f"{kind}_{s}"] or [])[: len(names)]):
                        kk = int(np.argmin(np.linalg.norm(r["J"][p] - G[gi], axis=1)))
                        acc[(kind, ri)].append(float(c[kk]))
                        acc[(kind + "_h", ri)].append(float((r["J"][p[kk], 1] - P["y0"]) / max(r["J"][P["hips"], 1] - P["y0"], 1e-3)))
            sp = P["spine"]
            c = cum_along(sp, D)
            c = c / max(c[-1], 1e-9)
            acc[("head", 0)].append(float(c[int(np.argmin(np.linalg.norm(r["J"][sp] - G[gr["head"]], axis=1)))]))
        cal[cls] = dict(leg=[float(np.median(acc[("leg", i)])) for i in range(4) if len(acc[("leg", i)]) >= 0.5 * max(len(acc[("leg", 0)]), 1)],
                        arm=[float(np.median(acc[("arm", i)])) for i in range(3)], head=float(np.median(acc[("head", 0)])), n_train=len(train),
                        **({"leg_h": [None, float(np.median(acc[("leg_h", 1)])), float(np.median(acc[("leg_h", 2)])), None],
                            "arm_h": [None, float(np.median(acc[("arm_h", 1)])), float(np.median(acc[("arm_h", 2)]))]} if cls == "quadruped" and HEIGHT_RULE else {}))
        print(cls, {k: ([round(x, 2) if x is not None else None for x in v] if isinstance(v, list) else round(v, 2) if isinstance(v, float) else v) for k, v in cal[cls].items()})
    (OUT / "roles_calibration.json").write_text(json.dumps(cal, indent=1))
    # scoring
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    fails = collections.Counter()
    for r in ok:
        split = "train" if zlib.crc32(r["key"].encode()) % 2 == 0 else "val"
        try:
            roles = assign(r["P"], cal[r["cls"]])
        except Fail as e:
            fails[(r["cls"], str(e))] += 1
            continue
        G, gr = r["G"], r["gr"]
        h = max(G[gr["hips"], 1] - G[:, 1].min(), 1e-6)   # hips height above the lowest rig joint: the unit of the errors
        gj, pj = role_joints(gr), role_joints(roles)
        errs = {k: float(np.linalg.norm(r["J"][pj[k]] - G[gj[k]]) / h) for k in pj if k in gj}
        # floor: the best any predicted joint could do for that role (a dense skeleton nearly always has one close by)
        floor = {k: float(np.linalg.norm(r["J"] - G[gj[k]], axis=1).min() / h) for k in gj}
        for k, e in errs.items():
            stats[(r["cls"], split)][k.split(".")[-1] if "." in k else k].append(e)
            stats[(r["cls"], split, "floor")][k.split(".")[-1] if "." in k else k].append(floor[k])
        stats[(r["cls"], split)]["all_roles_mean"].append(float(np.mean(list(errs.values()))))
        stats[(r["cls"], split)]["_status"].append(r["status"])
        json.dump({"cls": r["cls"], "roles": {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in roles.items()}, "errors_over_hips_h": errs},
                  open(OUT / r["key"] / "roles.json", "w"), default=lambda o: int(o))
    print("role-assignment failures:", dict(fails))
    print("error = distance to the rig's own role joint, in units of the character's hips height (0.05 = 5% of it, about 5 cm for a human)")
    for key in sorted(k for k in stats if len(k) == 2):
        s = stats[key]
        n = len(s["all_roles_mean"])
        print(f"\n{key[0]} / {key[1]} ({n} characters)")
        print(f"  {'role':10s} {'median':>7s} {'<0.05':>6s} {'<0.10':>6s} {'<0.20':>6s}   (floor = best predicted joint: median)")
        for role in ["hips", "head", "thigh", "knee", "ankle", "toe", "upper_arm", "elbow", "wrist", "all_roles_mean"]:
            v = np.array(s.get(role, []))
            if not len(v):
                continue
            fl = np.array(stats[(*key, "floor")].get(role, [np.nan]))
            print(f"  {role:10s} {np.median(v):7.3f} {np.mean(v < .05):6.2f} {np.mean(v < .10):6.2f} {np.mean(v < .20):6.2f}   ({np.median(fl):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--status", help="score only characters with this retarget_targets.csv status (ok = standing rest pose)")
    e.add_argument("--no-mesh-feet", action="store_true", help="quadrupeds: find the feet from skeleton leaves only (the first version)")
    e.add_argument("--no-height", action="store_true", help="quadrupeds: arc-length fractions for every role joint (the first version)")
    args = ap.parse_args()
    {"eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
