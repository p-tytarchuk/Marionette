"""Fill a template's mesh: constrained Delaunay triangulation of its outline
+ v1 bone weights.

Triangulation: `triangle` (Shewchuk) on the outline as a PSLG, so concave
regions (gaps between legs, under the tail) stay empty -- scipy's Delaunay
would fill the convex hull.

Weights (v1, per CLAUDE.md): for each vertex, distance to each bone's
segment(s); keep the k_max nearest bones; w = 1/(d+eps)^p; normalise per
vertex. Bone j owns the segments rest_pos(j) -> rest_pos(child) for each
child, plus rest_pos(j) -> tip_pos(j) for leaves. Upgrade path: bounded
biharmonic weights (needs libigl -- ask first).

Writes mesh.vertices / mesh.triangles / mesh.weights back into the template
and a preview PNG next to the template (<template>.preview.png) showing the
dominant bone per triangle, one bone's weight field and a test pose.

    python tools/build_mesh.py templates/4leg/quadruped_generic.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import triangle as tr


def outline_is_simple(poly: np.ndarray) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent via wrap-around
            c, d = poly[j], poly[(j + 1) % n]
            if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
                return False
    return True


def triangulate(outline: np.ndarray, max_area: float, min_angle: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(outline)
    pslg = {"vertices": outline, "segments": np.array([[i, (i + 1) % n] for i in range(n)])}
    mesh = tr.triangulate(pslg, f"pq{min_angle}a{max_area}")
    return mesh["vertices"], mesh["triangles"]


def bone_segments(joints: list[dict]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    pos = {j["id"]: np.array(j["rest_pos"], float) for j in joints}
    segs: dict[str, list] = {j["id"]: [] for j in joints}
    for j in joints:
        if j["parent"] is not None:
            segs[j["parent"]].append((pos[j["parent"]], pos[j["id"]]))
        if "tip_pos" in j:
            segs[j["id"]].append((pos[j["id"]], np.array(j["tip_pos"], float)))
    for jid, s in segs.items():
        if not s:  # leaf without tip_pos: degenerate point bone
            s.append((pos[jid], pos[jid]))
    return segs


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    u = np.zeros(len(p)) if denom == 0 else np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + u[:, None] * ab), axis=1)


def inverse_distance_weights(verts: np.ndarray, joints: list[dict], k_max: int, power: float,
                             eps: float = 1.0) -> np.ndarray:
    segs = bone_segments(joints)
    ids = [j["id"] for j in joints]
    D = np.stack([np.min([point_segment_distance(verts, a, b) for a, b in segs[jid]], axis=0) for jid in ids], axis=1)
    W = 1.0 / (D + eps) ** power
    if k_max < len(ids):
        cutoff = np.sort(D, axis=1)[:, k_max - 1: k_max]
        W[D > cutoff] = 0.0
    return W / W.sum(axis=1, keepdims=True)


def deform(verts: np.ndarray, joints: list[dict], W: np.ndarray, angles: dict[str, float]) -> np.ndarray:
    """2D linear blend skinning. angles: joint id -> delta rotation (rad)
    around its rest_pos, accumulated down the hierarchy."""
    by_id = {j["id"]: j for j in joints}
    mats: dict[str, np.ndarray] = {}

    def world(jid: str) -> np.ndarray:
        if jid in mats:
            return mats[jid]
        j = by_id[jid]
        c, s = np.cos(angles.get(jid, 0.0)), np.sin(angles.get(jid, 0.0))
        px, py = j["rest_pos"]
        local = np.array([[c, -s, px - c * px + s * py], [s, c, py - s * px - c * py], [0, 0, 1]])
        m = local if j["parent"] is None else world(j["parent"]) @ local
        mats[jid] = m
        return m

    vh = np.column_stack([verts, np.ones(len(verts))])
    out = np.zeros_like(verts)
    for k, j in enumerate(joints):
        out += W[:, k: k + 1] * (vh @ world(j["id"]).T)[:, :2]
    return out


QUADRUPED_PREVIEW_POSE = {"leg_front_L_upper": 0.5, "leg_front_L_lower": -0.6, "leg_back_R_upper": -0.4,
                          "neck": -0.3, "tail_base": 0.5, "spine_front": 0.1}


def save_preview(path: Path, verts, tris, joints, W, pose: dict | None = None) -> None:
    """`pose`: joint id -> delta angle (rad); a template can set it as `mesh.preview_pose` (default: the quadruped's).
    The weight panel shows the first joint of the pose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pose = pose or QUADRUPED_PREVIEW_POSE
    shown = next(iter(pose))
    panels = [("rest: dominant bone", verts, W.argmax(1)),
              (f"weights: {shown}", verts, None),
              ("LBS test pose", deform(verts, joints, W, pose), W.argmax(1))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    k_leg = [j["id"] for j in joints].index(shown)
    for ax, (title, V, colour) in zip(axes, panels):
        if colour is None:
            ax.tripcolor(V[:, 0], V[:, 1], tris, W[:, k_leg], shading="gouraud", cmap="magma")
        else:
            ax.tripcolor(V[:, 0], V[:, 1], tris, facecolors=colour[tris].max(1), cmap="tab20", edgecolors="w", lw=0.2)
        if V is verts:
            for j in joints:
                if j["parent"]:
                    p = next(q for q in joints if q["id"] == j["parent"])
                    ax.plot([p["rest_pos"][0], j["rest_pos"][0]], [p["rest_pos"][1], j["rest_pos"][1]], "c-", lw=1)
                ax.plot(*j["rest_pos"], "co", ms=3)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", type=Path)
    ap.add_argument("--max-area", type=float, default=80.0, help="max triangle area, px^2")
    ap.add_argument("--min-angle", type=float, default=30.0, help="min triangle angle, degrees")
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--power", type=float, default=2.0)
    args = ap.parse_args()

    tpl = json.loads(args.template.read_text())
    joints = tpl["skeleton"]["joints"]
    outline = np.array(tpl["mesh"]["outline"], float)
    if not outline_is_simple(outline):
        raise SystemExit("mesh.outline self-intersects -- fix it before triangulating")

    verts, tris = triangulate(outline, args.max_area, args.min_angle)
    W = inverse_distance_weights(verts, joints, args.k_max, args.power)
    ids = [j["id"] for j in joints]

    # cross-limb leakage: weight a leg vertex receives from bones of a different leg
    dom = W.argmax(1)
    leak = 0.0
    for k, jid in enumerate(ids):
        m = re.match(r"(leg_(?:\w+_)?[LR])_", jid)  # leg_front_L_upper (4leg), leg_L_upper (humanoid)
        if m:
            other = [i for i, o in enumerate(ids) if o.startswith("leg_") and not o.startswith(m.group(1))]
            leak = max(leak, float(W[dom == k][:, other].sum(1).max(initial=0.0)))

    tpl["mesh"]["vertices"] = np.round(verts, 2).tolist()
    tpl["mesh"]["triangles"] = tris.tolist()
    tpl["mesh"]["weights"] = {
        str(v): {ids[k]: round(float(W[v, k]), 4) for k in np.nonzero(W[v] > 1e-3)[0]} for v in range(len(verts))
    }
    tpl["mesh"]["build"] = {"triangulator": "triangle (constrained Delaunay)", "max_area": args.max_area,
                            "min_angle": args.min_angle, "weights": "inverse-distance-to-bone-segment",
                            "k_max": args.k_max, "power": args.power}
    s = json.dumps(tpl, indent=2)
    s = re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", s)
    s = re.sub(r"\[\s+(\d+),\s+(\d+),\s+(\d+)\s+\]", r"[\1, \2, \3]", s)
    args.template.write_text(s + "\n")

    preview = args.template.with_suffix(".preview.png")
    save_preview(preview, verts, tris, joints, W, tpl["mesh"].get("preview_pose"))
    print(f"{len(verts)} vertices, {len(tris)} triangles; bones per vertex <= {args.k_max}; "
          f"max cross-limb weight leak = {leak:.3f}")
    print(f"wrote {args.template} and {preview}")


if __name__ == "__main__":
    main()
