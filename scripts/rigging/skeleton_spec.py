"""Role-labelled 2D skeleton spec ("skeleton-roles/1"): validate it, derive the body class and the movement-mode support from its part roles.

  python scripts/rigging/skeleton_spec.py validate FILE...          # structure and cross-reference checks; exit 1 on any error
  python scripts/rigging/skeleton_spec.py classify FILE...          # carriers (legs, arms, wings, fins, tail, ...) and the class derived from them
  python scripts/rigging/skeleton_spec.py modes FILE [--mode fly]   # per movement mode: native / stylised / unsupported, with the reason
  python scripts/rigging/skeleton_spec.py write-examples            # writes docs/skeleton_spec/examples/*.json from the examples below
  python scripts/rigging/skeleton_spec.py check-anymate             # roles of the Anymate characters -> spec -> derived class, against the characters' own class
  python scripts/rigging/skeleton_spec.py --self-test               # the examples, the broken cases, the derivation rules, the mode table

The format, the vocabulary and the reasons for it are in docs/SKELETON_SPEC.md. In short: every joint belongs to one part (trunk, head, tail, leg, arm, wing, fin, extra); the class is a
COUNT over the parts, never a guess from the geometry (docs/SKELETON_CLASSES.md showed that wings and snakes cannot be told from a bare tree).
Parts that exist on the body but have no artwork in this drawing (the far legs of a profile) stay in the spec with visible=false, so the class does not depend on the yaw.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
EXAMPLES = ROOT / "docs" / "skeleton_spec" / "examples"
VERSION = "skeleton-roles/1"
KINDS = ("trunk", "head", "tail", "leg", "arm", "wing", "fin", "extra")
CHAIN_KINDS = ("trunk", "head", "tail", "leg", "arm", "wing", "fin")        # a path; "extra" may be any subtree (horns, hair, feathers, cloth)
PAIRED = ("leg", "arm", "wing")                                              # need a side
CLASSES = ("biped", "quadruped", "limbless", "winged", "other")
SECONDARY_DEFAULT = {"tail": True, "extra": True, "head": False}            # follow-through (overlapping) motion unless the spec says otherwise


# ----------------------------------------------------------------------------- validation
def parts_of(sk):
    """{part id: ordered joint ids (top first)}; the order follows the parent links, not the order of the list"""
    by = {}
    for j in sk["joints"]:
        by.setdefault(j["part"], []).append(j["id"])
    return by


def validate(sk):
    """(errors, warnings) of one skeleton dict. Errors make the file unusable; warnings are things a generator should fix but a consumer can live with."""
    errs, warns = [], []
    if sk.get("spec") != VERSION:
        errs.append(f"spec must be {VERSION!r}")
    joints, parts = sk.get("joints"), sk.get("parts")
    if not isinstance(joints, list) or not isinstance(parts, list) or len(joints) < 2:
        return errs + ["joints and parts must be lists with at least 2 joints"], warns
    ids = [j.get("id") for j in joints]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        return errs + ["joint ids must be unique non-empty strings"], warns
    jid = {j["id"]: j for j in joints}
    pid = [p.get("id") for p in parts]
    if len(set(pid)) != len(pid) or not all(isinstance(i, str) and i for i in pid):
        return errs + ["part ids must be unique non-empty strings"], warns
    pm = {p["id"]: p for p in parts}
    for j in joints:
        pos = j.get("pos")
        if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)):
            errs.append(f"joint {j['id']}: pos must be two finite numbers [x, y] (y up)")
        if j.get("parent") is not None and j["parent"] not in jid:
            errs.append(f"joint {j['id']}: parent {j['parent']!r} does not exist")
        if j.get("part") not in pm:
            errs.append(f"joint {j['id']}: part {j.get('part')!r} does not exist")
    if errs:
        return errs, warns
    roots = [j["id"] for j in joints if j.get("parent") is None]
    if len(roots) != 1:
        errs.append(f"exactly one root joint (parent null) is required, found {len(roots)}: {roots[:4]}")
    for j in joints:                                                          # a cycle never reaches the root
        seen, cur = set(), j["id"]
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = jid[cur].get("parent")
        if cur is not None:
            errs.append(f"joint {j['id']}: the parent links form a cycle")
            break
    if errs:
        return errs, warns
    kinds = {p["id"]: p.get("kind") for p in parts}
    for p in parts:
        if p.get("kind") not in KINDS:
            errs.append(f"part {p['id']}: kind must be one of {KINDS}")
    if errs:
        return errs, warns
    members = parts_of(sk)
    for p in parts:
        if p["id"] not in members:
            errs.append(f"part {p['id']} has no joints")
    count = {k: sum(1 for p in parts if p["kind"] == k) for k in KINDS}
    if count["trunk"] != 1:
        errs.append(f"exactly one trunk part is required, found {count['trunk']}")
    for k in ("head", "tail"):
        if count[k] > 1:
            errs.append(f"at most one {k} part, found {count[k]}")
    if roots and kinds[jid[roots[0]]["part"]] != "trunk":
        errs.append("the root joint must belong to the trunk part")
    for p in parts:
        mem = members.get(p["id"], [])
        tops = [m for m in mem if jid[m].get("parent") is None or jid[jid[m]["parent"]]["part"] != p["id"]]
        if len(tops) != 1:
            errs.append(f"part {p['id']}: its joints must form one connected piece hanging from one attachment (found {len(tops)} attachments)")
            continue
        if p["kind"] in CHAIN_KINDS:
            for m in mem:
                inside = [c for c in mem if jid[c].get("parent") == m]
                if len(inside) > 1:
                    errs.append(f"part {p['id']} ({p['kind']}) must be a chain, but joint {m} has {len(inside)} children inside it (use an extra part for branches)")
                    break
        if p["kind"] != "trunk" and jid[tops[0]].get("parent") is None:
            errs.append(f"part {p['id']}: only the trunk may contain the root")
        if p["kind"] in PAIRED:
            if p.get("side") not in ("L", "R"):
                errs.append(f"part {p['id']} ({p['kind']}): side must be 'L' or 'R' (the body's own side, not the screen side)")
            if p["kind"] in ("leg", "arm", "wing") and p.get("girdle") not in ("pelvic", "pectoral", None):
                errs.append(f"part {p['id']}: girdle must be 'pelvic', 'pectoral' or null")
        if p["kind"] == "fin" and p.get("side") not in ("L", "R", "C"):
            errs.append(f"part {p['id']} (fin): side must be 'L', 'R' or 'C'")
        if p.get("layer") not in (None, "front", "behind"):
            errs.append(f"part {p['id']}: layer must be 'front', 'behind' or null")
        if not isinstance(p.get("visible", True), bool) or not isinstance(p.get("secondary", False), bool):
            errs.append(f"part {p['id']}: visible and secondary must be booleans")
    if errs:
        return errs, warns
    for k in PAIRED:
        nl = sum(1 for p in parts if p["kind"] == k and p["side"] == "L")
        nr = sum(1 for p in parts if p["kind"] == k and p["side"] == "R")
        if nl != nr:
            warns.append(f"{k}: {nl} on the left, {nr} on the right (an unpaired limb is allowed but unusual)")
    sk_class = sk.get("declared_class")
    if sk_class is not None:
        if sk_class not in CLASSES:
            errs.append(f"declared_class must be one of {CLASSES}")
        elif sk_class != derive_class(sk)[0]:
            warns.append(f"declared_class {sk_class!r} differs from the class derived from the parts ({derive_class(sk)[0]!r}); the parts win")
    if not any(p["kind"] == "head" for p in parts):
        warns.append("no head part: head turns and look-at motions have nothing to move")
    return errs, warns


# ----------------------------------------------------------------------------- derivation
def carriers(sk):
    """What the body can move with, counted over the parts (hidden parts count: they exist on the body)."""
    parts = sk["parts"]
    n = lambda kind, **kw: sum(1 for p in parts if p["kind"] == kind and all(p.get(k) == v for k, v in kw.items()))
    jid = {j["id"]: j for j in sk["joints"]}
    trunk = [j for j in sk["joints"] if next(p for p in parts if p["id"] == j["part"])["kind"] == "trunk"]
    head = [j for j in sk["joints"] if next(p for p in parts if p["id"] == j["part"])["kind"] == "head"]
    if len(trunk) >= 2 and head:                                              # upright / horizontal from the vector trunk root -> head
        (x0, y0), (x1, y1) = trunk[0]["pos"], head[-1]["pos"]
        axis = "upright" if abs(y1 - y0) >= 0.7 * math.hypot(x1 - x0, y1 - y0) else "horizontal"
    elif len(trunk) >= 2:
        (x0, y0), (x1, y1) = trunk[0]["pos"], trunk[-1]["pos"]
        axis = "upright" if abs(y1 - y0) >= 0.7 * math.hypot(x1 - x0, y1 - y0) else "horizontal"
    else:
        axis = "unknown"
    return dict(legs=n("leg"), legs_pelvic=n("leg", girdle="pelvic"), legs_pectoral=n("leg", girdle="pectoral"), arms=n("arm"), wings=n("wing"), fins=n("fin"),
                tail=n("tail") > 0, head=n("head") > 0, trunk_joints=len(trunk), trunk_axis=axis,
                secondary=[p["id"] for p in parts if p.get("secondary", SECONDARY_DEFAULT.get(p["kind"], False))],
                hidden=[p["id"] for p in parts if p.get("visible", True) is False])


def derive_class(sk):
    """(class, reason): a count over the parts. Wings decide first (a bird, a bat, a dragon, an angel are all winged); then the number of legs; fins are not limbs."""
    c = carriers(sk)
    if c["wings"] >= 1:
        return "winged", f"{c['wings']} wing part(s)"
    if c["legs"] == 2:
        return "biped", "2 legs, no wings"
    if c["legs"] == 4:
        return "quadruped", "4 legs, no wings"
    if c["legs"] == 0 and c["arms"] == 0:
        return "limbless", "no legs, arms or wings" + (f" ({c['fins']} fin part(s) are not limbs)" if c["fins"] else "")
    return "other", f"{c['legs']} legs and {c['arms']} arms, no wings"


# movement modes: (level, reason) from the carriers. A PROPOSAL to tune; whether "stylised" is shown or refused is a policy outside this table.
def _modes(c):
    legs, arms, wings, fins, limbless = c["legs"], c["arms"], c["wings"], c["fins"], c["legs"] == 0 and c["arms"] == 0 and c["wings"] == 0
    sup, sty, uns = "native", "stylised", "unsupported"
    return {
        "walk": (sup, f"{legs} legs") if legs >= 2 else (uns, "a body with no legs does not walk"),
        "run": (sup, f"{legs} legs") if legs >= 2 else (uns, "a body with no legs does not run"),
        "trot_gallop": (sup, "4 legs") if legs == 4 else ((sty, f"{legs} legs, borrowed gait") if legs >= 2 else (uns, "no legs")),
        "hop": (sup, f"{legs} legs") if legs >= 2 else (sty, "no legs: squash and stretch"),
        "jump": (sup, f"{legs} legs") if legs >= 2 else (sty, "no legs: squash and stretch"),
        "crawl": (sup, f"{legs} legs, low body") if legs >= 2 else (sty, "no legs: trunk pulls itself along"),
        "fly": (sup, f"{wings} wing part(s)") if wings else (sty, "no wings: float with a flying pose, limbs trail, hair and cloth follow"),
        "glide": (sup, f"{wings} wing part(s)") if wings else (sty, "no wings: float with a flying pose"),
        "slither": (sup, "single trunk chain, no limbs") if limbless and c["trunk_joints"] >= 3 else ((sty, "limbs present: reduced trunk wave, limbs tucked or trailing") if c["trunk_joints"] >= 3 else (uns, "the trunk is too short for a wave")),
        "swim": (sup, f"{fins} fin part(s)" if fins else "single trunk chain") if (fins or limbless) else (sty, "paddle with the limbs"),
        "wave": (sup, f"{arms} arm(s)") if arms else ((sty, "a wing or a lifted foreleg stands in for an arm") if (wings or c["legs_pectoral"]) else (uns, "no arm, wing or foreleg")),
        "dance": (sup, f"{legs} legs") if legs >= 2 else (sty, "trunk and head only"),
        "idle": (sup, "any body breathes and sways"),
    }


def mode_support(sk):
    return _modes(carriers(sk))


# ----------------------------------------------------------------------------- examples (also the test inputs and the files in docs/skeleton_spec/examples)
def _chain(part, start, pts, **extra):
    """joints along a polyline: [(id, parent, pos, part)]"""
    out, parent = [], start
    for k, p in enumerate(pts):
        jid = f"{part}{k + 1}"
        out.append(dict(id=jid, parent=parent, pos=[round(p[0], 4), round(p[1], 4)], part=part, **extra))
        parent = jid
    return out


def _limb(part, kind, side, girdle, attach, pts, layer="front", visible=True):
    return dict(id=part, kind=kind, side=side, girdle=girdle, layer=layer, visible=visible), _chain(part, attach, pts)


def _build(name, trunk, head, limbs, tail=None, fins=(), extras=(), declared=None, view=None):
    parts, joints = [dict(id="trunk", kind="trunk")], _chain("trunk", None, trunk)
    attach_head, attach_tail = joints[-1]["id"], joints[0]["id"]
    if head:
        parts.append(dict(id="head", kind="head"))
        joints += _chain("head", attach_head, head)
    if tail:
        parts.append(dict(id="tail", kind="tail", secondary=True))
        joints += _chain("tail", attach_tail, tail)
    for spec, chain in limbs:
        parts.append(spec)
        joints += chain
    for spec, chain in fins:
        parts.append(spec)
        joints += chain
    for spec, chain in extras:
        parts.append(spec)
        joints += chain
    sk = dict(spec=VERSION, id=name, view=view or dict(yaw_deg=0, facing="right"), parts=parts, joints=joints)
    if declared:
        sk["declared_class"] = declared
    return sk


def examples():
    """name -> skeleton dict. Coordinates are in body units (y up); the far-side limbs of a profile drawing sit a little behind the near ones."""
    E = {}
    leg = lambda pid, side, g, attach, x, y0, h, **kw: _limb(pid, "leg", side, g, attach, [(x, y0 - h * 0.45), (x + 0.02, y0 - h * 0.8), (x + 0.04, y0 - h)], **kw)
    # a person, drawn in 3/4 profile: pelvis -> chest, neck -> head, two arms, two legs
    E["biped_person"] = _build("biped_person", [(0, 0.5), (0.01, 0.62), (0.02, 0.74)], [(0.03, 0.80), (0.04, 0.90)], [
        _limb("arm_L", "arm", "L", "pectoral", "trunk3", [(0.03, 0.70), (0.06, 0.58), (0.08, 0.47)]),
        _limb("arm_R", "arm", "R", "pectoral", "trunk3", [(-0.02, 0.70), (0.00, 0.58), (0.02, 0.47)], layer="behind"),
        _limb("leg_L", "leg", "L", "pelvic", "trunk1", [(0.02, 0.28), (0.02, 0.10), (0.06, 0.0)]),
        _limb("leg_R", "leg", "R", "pelvic", "trunk1", [(-0.02, 0.28), (-0.02, 0.10), (0.02, 0.0)], layer="behind")], declared="biped")
    # a fox in profile: 4 legs, bushy tail (secondary), ears (extra)
    E["quadruped_fox"] = _build("quadruped_fox", [(0, 0.30), (0.22, 0.32), (0.44, 0.34), (0.60, 0.36)], [(0.70, 0.44), (0.82, 0.46)], [
        _limb("leg_hind_L", "leg", "L", "pelvic", "trunk1", [(0.0, 0.18), (0.03, 0.09), (0.05, 0.0)]),
        _limb("leg_hind_R", "leg", "R", "pelvic", "trunk1", [(-0.03, 0.18), (0.0, 0.09), (0.02, 0.0)], layer="behind"),
        _limb("leg_fore_L", "leg", "L", "pectoral", "trunk4", [(0.60, 0.20), (0.62, 0.10), (0.64, 0.0)]),
        _limb("leg_fore_R", "leg", "R", "pectoral", "trunk4", [(0.57, 0.20), (0.59, 0.10), (0.61, 0.0)], layer="behind")],
        tail=[(-0.15, 0.36), (-0.30, 0.42), (-0.42, 0.40)],
        extras=[(dict(id="ear_L", kind="extra", secondary=True, layer="front"), _chain("ear_L", "head1", [(0.72, 0.52)]))], declared="quadruped")
    # a snake: one trunk chain along a wave, a head; no limbs at all
    E["limbless_snake"] = _build("limbless_snake", [(i * 0.1, 0.05 * math.sin(i * 0.9)) for i in range(12)], [(1.12, 0.08)], [], declared="limbless")
    # a bird: 2 legs, 2 wings, tail, upright-ish trunk
    E["winged_bird"] = _build("winged_bird", [(0, 0.30), (0.10, 0.42)], [(0.16, 0.50), (0.22, 0.52)], [
        _limb("wing_L", "wing", "L", "pectoral", "trunk2", [(0.08, 0.50), (0.0, 0.62), (-0.12, 0.70), (-0.25, 0.72)]),
        _limb("wing_R", "wing", "R", "pectoral", "trunk2", [(0.07, 0.50), (-0.01, 0.62), (-0.13, 0.68), (-0.26, 0.69)], layer="behind"),
        _limb("leg_L", "leg", "L", "pelvic", "trunk1", [(0.01, 0.18), (0.03, 0.08), (0.05, 0.0)]),
        _limb("leg_R", "leg", "R", "pelvic", "trunk1", [(-0.01, 0.18), (0.01, 0.08), (0.03, 0.0)], layer="behind")],
        tail=[(-0.08, 0.24), (-0.20, 0.18)], declared="winged")
    # a dragon: 4 legs and 2 wings
    E["winged_dragon"] = _build("winged_dragon", [(0, 0.35), (0.25, 0.38), (0.5, 0.40), (0.7, 0.42)], [(0.82, 0.52), (0.95, 0.55)], [
        _limb("leg_hind_L", "leg", "L", "pelvic", "trunk1", [(0.0, 0.2), (0.03, 0.1), (0.06, 0.0)]),
        _limb("leg_hind_R", "leg", "R", "pelvic", "trunk1", [(-0.03, 0.2), (0.0, 0.1), (0.03, 0.0)], layer="behind"),
        _limb("leg_fore_L", "leg", "L", "pectoral", "trunk4", [(0.7, 0.22), (0.72, 0.1), (0.75, 0.0)]),
        _limb("leg_fore_R", "leg", "R", "pectoral", "trunk4", [(0.67, 0.22), (0.69, 0.1), (0.72, 0.0)], layer="behind"),
        _limb("wing_L", "wing", "L", "pectoral", "trunk3", [(0.45, 0.6), (0.35, 0.85), (0.2, 1.0), (0.05, 1.05)]),
        _limb("wing_R", "wing", "R", "pectoral", "trunk3", [(0.43, 0.6), (0.33, 0.83), (0.18, 0.98), (0.03, 1.02)], layer="behind")],
        tail=[(-0.2, 0.33), (-0.4, 0.30), (-0.6, 0.34)], declared="winged")
    # a bear in profile: the far legs exist on the body but have no artwork in this drawing (visible=false); the class must not change
    bear = _build("quadruped_bear_profile", [(0, 0.40), (0.25, 0.42), (0.5, 0.44)], [(0.62, 0.50), (0.74, 0.50)], [
        _limb("leg_hind_L", "leg", "L", "pelvic", "trunk1", [(0.0, 0.25), (0.02, 0.12), (0.05, 0.0)]),
        _limb("leg_hind_R", "leg", "R", "pelvic", "trunk1", [(0.0, 0.25), (0.02, 0.12), (0.05, 0.0)], layer="behind", visible=False),
        _limb("leg_fore_L", "leg", "L", "pectoral", "trunk3", [(0.5, 0.25), (0.52, 0.12), (0.55, 0.0)]),
        _limb("leg_fore_R", "leg", "R", "pectoral", "trunk3", [(0.5, 0.25), (0.52, 0.12), (0.55, 0.0)], layer="behind", visible=False)], declared="quadruped")
    E["quadruped_bear_profile"] = bear
    # a spider: 8 legs -> other (the carriers still say what it can do)
    legs = []
    for k in range(4):
        for side, dx in (("L", 0.0), ("R", -0.02)):
            legs.append(_limb(f"leg_{side}{k + 1}", "leg", side, None, f"trunk{1 + k % 2}", [(0.1 * k - 0.1 + dx, 0.18), (0.1 * k - 0.15 + dx, 0.10), (0.1 * k - 0.2 + dx, 0.0)],
                              layer="front" if side == "L" else "behind"))
    E["multileg_spider"] = _build("multileg_spider", [(0, 0.2), (0.15, 0.22)], [(0.22, 0.24)], legs, declared="other")
    # a fish: spine, tail, fins (fins are not limbs): class limbless, on purpose
    E["limbless_fish"] = _build("limbless_fish", [(0, 0.3), (0.2, 0.3), (0.4, 0.3), (0.6, 0.3)], [(0.7, 0.3)], [], tail=[(-0.15, 0.3), (-0.3, 0.38)], fins=[
        (dict(id="fin_dorsal", kind="fin", side="C"), _chain("fin_dorsal", "trunk2", [(0.2, 0.42), (0.3, 0.44)])),
        (dict(id="fin_pect_L", kind="fin", side="L"), _chain("fin_pect_L", "trunk4", [(0.58, 0.22), (0.5, 0.15)]))], declared="limbless")
    # an angel: a person with wings -> winged (the wing part decides, not the silhouette)
    p = json.loads(json.dumps(E["biped_person"]))
    p["id"] = "winged_angel"
    p["parts"] += [dict(id="wing_L", kind="wing", side="L", girdle="pectoral", layer="behind"), dict(id="wing_R", kind="wing", side="R", girdle="pectoral", layer="behind")]
    p["joints"] += _chain("wing_L", "trunk3", [(-0.08, 0.80), (-0.18, 0.92), (-0.30, 0.98)]) + _chain("wing_R", "trunk3", [(-0.10, 0.78), (-0.20, 0.90), (-0.32, 0.95)])
    p["declared_class"] = "winged"
    E["winged_angel"] = p
    return E


# ----------------------------------------------------------------------------- bridge from the existing roles (Anymate / retargeter): spine, leg_L, leg_R, arm_L, arm_R as joint-index lists
def from_roles(P3, parents, roles, cls, name):
    """A spec from 3D joints (n, 3), a parent array, and a roles dict as written by anymate_roles.py (hips, head, spine, leg_L, arm_L, leg_R, arm_R).
    The tree is a spanning tree of the parent links rooted at the hips (some predictions have cycles or no root). Quadruped forelimbs ("arm" in the retargeter) are legs
    with a pectoral girdle; a biped's arms are arms. Every joint not in a role becomes part of an `extra` part hanging from the nearest role joint."""
    import numpy as np
    import skeleton_classes as sc
    P3 = np.asarray(P3, float)
    adj = sc.tree_adjacency(list(parents), P3)
    root, par, order = int(roles["hips"]), {}, [int(roles["hips"])]
    par[root] = None
    for u in order:
        for v in adj[u]:
            if v not in par:
                par[v] = u
                order.append(v)
    Q = sc.project(P3, 0)                                                      # profile: x' = z, y up

    def path_to(j):
        out = [j]
        while par[out[-1]] is not None:
            out.append(par[out[-1]])
        return out[::-1]
    head_j = int(roles["head"])
    assign = {}
    trunk_path = path_to(head_j)                                              # hips -> head along the tree: helper joints between the listed spine joints belong to the trunk
    for j in trunk_path[:-1] or trunk_path:
        assign[j] = "trunk"
    assign[head_j] = "head" if len(trunk_path) > 1 else "trunk"
    limbs = []
    for key, kind, side, girdle in (("leg_L", "leg", "L", "pelvic"), ("leg_R", "leg", "R", "pelvic"),
                                    ("arm_L", "arm" if cls == "biped" else "leg", "L", "pectoral"), ("arm_R", "arm" if cls == "biped" else "leg", "R", "pectoral")):
        chain = [int(j) for j in roles.get(key, [])]
        if not chain:
            continue
        path = path_to(chain[-1])                                             # the role chains skip helper joints: a limb is every joint from the trunk to its last role joint
        k = max((i for i, j in enumerate(path) if assign.get(j) in ("trunk", "head")), default=-1)
        own = [j for j in path[k + 1:] if j not in assign]
        if own:
            limbs.append(dict(id=key, kind=kind, side=side, girdle=girdle))
            for j in own:
                assign[j] = key
    parts = [dict(id="trunk", kind="trunk")] + ([dict(id="head", kind="head")] if "head" in assign.values() else []) + limbs
    for j in order:                                                            # leftovers: one extra part per subtree that hangs off an assigned joint
        if j in assign:
            continue
        if par[j] in assign and not str(assign[par[j]]).startswith("extra_"):
            assign[j] = f"extra_{j}"
            parts.append(dict(id=f"extra_{j}", kind="extra", secondary=True))
        else:
            assign[j] = assign[par[j]]
    joints = [dict(id=f"j{j}", parent=(None if par[j] is None else f"j{par[j]}"), pos=[round(float(Q[j, 0]), 5), round(float(Q[j, 1]), 5)], part=assign[j]) for j in order]
    return dict(spec=VERSION, id=name, view=dict(yaw_deg=0, facing="unknown"), declared_class=cls if cls in CLASSES else None, parts=parts, joints=joints)


def check_anymate():
    import numpy as np
    rows, counts = [], {}
    for d in sorted((ROOT / "data" / "anymate").glob("*/")):
        if d.name.startswith("._") or not (d / "roles.json").exists():
            continue
        r = json.loads((d / "roles.json").read_text())
        for tag, jf, pf in (("pred", "joints.npy", "parents.npy"),):
            try:
                sk = from_roles(np.load(d / jf), np.load(d / pf), r["roles"], r["cls"], d.name)
            except Exception as e:                                          # a role chain that cannot be built into a tree
                counts[("build error", type(e).__name__)] = counts.get(("build error", type(e).__name__), 0) + 1
                continue
            errs, _ = validate(sk)
            if errs:
                counts[("invalid", errs[0][:70])] = counts.get(("invalid", errs[0][:70]), 0) + 1
                continue
            cls, _ = derive_class(sk)
            counts[("ok", f"{r['cls']} -> {cls}")] = counts.get(("ok", f"{r['cls']} -> {cls}"), 0) + 1
    for k in sorted(counts):
        print(f"{counts[k]:4d}  {k[0]:12s} {k[1]}")


# ----------------------------------------------------------------------------- self-test
_refused = []


def _must_fail(sk, text):
    errs, _ = validate(sk)
    assert any(text in e for e in errs), f"expected an error containing {text!r}, got {errs}"
    _refused.append(text)


def self_test():
    import copy
    ex = examples()
    for name, sk in ex.items():                                               # every example is valid and its declared class is the derived one
        errs, warns = validate(sk)
        assert not errs, (name, errs)
        assert derive_class(sk)[0] == sk["declared_class"], (name, derive_class(sk), sk["declared_class"])
    want = {"biped_person": "biped", "quadruped_fox": "quadruped", "limbless_snake": "limbless", "winged_bird": "winged", "winged_dragon": "winged",
            "quadruped_bear_profile": "quadruped", "multileg_spider": "other", "limbless_fish": "limbless", "winged_angel": "winged"}
    assert {n: derive_class(s)[0] for n, s in ex.items()} == want
    # hidden parts still count: the bear has 4 legs although two have no artwork
    c = carriers(ex["quadruped_bear_profile"])
    assert c["legs"] == 4 and len(c["hidden"]) == 2 and c["trunk_axis"] == "horizontal", c
    assert carriers(ex["biped_person"])["trunk_axis"] == "upright"
    assert "tail" in str(carriers(ex["quadruped_fox"])["secondary"]) and "ear_L" in carriers(ex["quadruped_fox"])["secondary"]
    # a wing decides before the legs; removing the wings gives the legs' class back
    sk = copy.deepcopy(ex["winged_angel"])
    sk["parts"] = [p for p in sk["parts"] if p["kind"] != "wing"]
    sk["joints"] = [j for j in sk["joints"] if not j["part"].startswith("wing")]
    assert derive_class(sk)[0] == "biped" and not validate(sk)[0]
    # broken skeletons are refused, each with its own message
    bad = copy.deepcopy(ex["biped_person"]); bad["joints"][3]["parent"] = bad["joints"][3]["id"]
    _must_fail(bad, "cycle")
    bad = copy.deepcopy(ex["biped_person"]); bad["joints"].append(dict(id="trunk1", parent=None, pos=[0, 0], part="trunk"))
    _must_fail(bad, "unique")
    bad = copy.deepcopy(ex["biped_person"]); bad["joints"][1]["parent"] = None
    _must_fail(bad, "exactly one root")
    bad = copy.deepcopy(ex["biped_person"]); bad["parts"] = [p for p in bad["parts"] if p["kind"] != "trunk"]
    _must_fail(bad, "does not exist")
    bad = copy.deepcopy(ex["biped_person"]); bad["parts"][[p["id"] for p in bad["parts"]].index("leg_L")]["side"] = "left"
    _must_fail(bad, "side must be")
    bad = copy.deepcopy(ex["biped_person"]); bad["joints"][next(i for i, j in enumerate(bad["joints"]) if j["id"] == "leg_L2")]["parent"] = "trunk1"
    _must_fail(bad, "connected piece")
    bad = copy.deepcopy(ex["limbless_snake"]); bad["joints"].append(dict(id="trunk_branch", parent="trunk3", pos=[0.2, 0.2], part="trunk"))
    _must_fail(bad, "must be a chain")
    bad = copy.deepcopy(ex["biped_person"]); bad["joints"][0]["pos"] = [0, float("nan")]
    _must_fail(bad, "finite")
    bad = copy.deepcopy(ex["biped_person"]); bad["parts"].append(dict(id="trunk2", kind="trunk")); bad["joints"].append(dict(id="x", parent="trunk1", pos=[0, 1], part="trunk2"))
    _must_fail(bad, "exactly one trunk")
    wrong = copy.deepcopy(ex["biped_person"]); wrong["declared_class"] = "quadruped"
    errs, warns = validate(wrong)
    assert not errs and any("differs" in w for w in warns), (errs, warns)
    # the movement modes: the cases discussed (snake slithers, bear slithers, person flies, bird flies)
    m = {n: mode_support(s) for n, s in ex.items()}
    lev = lambda n, mode: m[n][mode][0]
    assert lev("limbless_snake", "slither") == "native" and lev("limbless_snake", "walk") == "unsupported" and lev("limbless_snake", "fly") == "stylised"
    assert lev("quadruped_bear_profile", "slither") == "stylised" and lev("quadruped_bear_profile", "trot_gallop") == "native" and lev("quadruped_bear_profile", "wave") == "stylised"
    assert lev("biped_person", "fly") == "stylised" and lev("biped_person", "walk") == "native" and lev("biped_person", "wave") == "native" and lev("biped_person", "trot_gallop") == "stylised"
    assert lev("winged_bird", "fly") == "native" and lev("winged_bird", "walk") == "native" and lev("winged_bird", "hop") == "native" and lev("winged_bird", "wave") == "stylised"
    assert lev("limbless_fish", "swim") == "native" and lev("limbless_fish", "slither") == "native" and lev("multileg_spider", "walk") == "native"
    assert all(v[0] == "native" for v in [m[n]["idle"] for n in m]), "every body can idle"
    print(f"skeleton_spec self-test: {len(ex)} examples valid, class rules, {len(_refused)} broken cases refused (each with its own message), mode table OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["validate", "classify", "modes", "write-examples", "check-anymate"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--mode")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "write-examples":
        EXAMPLES.mkdir(parents=True, exist_ok=True)
        for name, sk in examples().items():
            (EXAMPLES / f"{name}.json").write_text(json.dumps(sk, indent=1) + "\n", encoding="utf-8")
        return print(f"wrote {len(examples())} examples to {EXAMPLES}")
    if a.cmd == "check-anymate":
        return check_anymate()
    if not a.cmd or not a.files:
        ap.error("give a command and at least one file")
    bad = False
    for f in a.files:
        sk = json.loads(Path(f).read_text(encoding="utf-8"))
        errs, warns = validate(sk)
        if a.cmd == "validate" or errs:
            print(f"{f}: {'INVALID' if errs else 'valid'}" + "".join(f"\n  error: {e}" for e in errs) + "".join(f"\n  warning: {w}" for w in warns))
        bad |= bool(errs)
        if errs:
            continue
        if a.cmd == "classify":
            c = carriers(sk)
            print(f"{f}: class {derive_class(sk)[0]} ({derive_class(sk)[1]}); legs {c['legs']} (pelvic {c['legs_pelvic']}, pectoral {c['legs_pectoral']}), arms {c['arms']}, wings {c['wings']}, fins {c['fins']}, "
                  f"tail {c['tail']}, head {c['head']}, trunk {c['trunk_joints']} joints {c['trunk_axis']}, hidden {c['hidden']}, secondary {c['secondary']}")
        elif a.cmd == "modes":
            for mode, (level, why) in mode_support(sk).items():
                if not a.mode or a.mode == mode:
                    print(f"  {mode:12s} {level:12s} {why}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
