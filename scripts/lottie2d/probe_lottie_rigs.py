"""Probe: do the Lottie animations that Qwen called a whole living character have a rig-like layer structure?

Question behind it: can Lottie layers be mapped to roles (hips, head, arms, legs) the way Anymate joints were, so that 3D (Mixamo etc.) and 2D (Lottie)
motion can share one representation? This script samples relevant rows (data/vlm), reads their Lottie JSON from ../Datasets/LottieAnimation-660K and counts:
parent links between layers, animated rotations (layer transforms and shape-group transforms), hinge chains (3 layers in a parent chain that all rotate),
expressions, and how many layer / group names are generic ("Layer 12", "Shape Layer 3", "Outlines") against body words ("arm", "leg", "head", ...).

    python scripts/lottie2d/probe_lottie_rigs.py --shards 12 --per-shard-2leg 17 --per-shard-4leg 9

Writes data/analysis/lottie_rig_probe.json and .csv. Needs numpy-free stdlib + orjson + zstandard.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import json
import random
import re
from pathlib import Path

import orjson
import zstandard

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT.parent / "Datasets" / "LottieAnimation-660K"
OUT = ROOT / "data" / "analysis"
BODY = {
    "head": r"head|face|skull", "neck": r"neck", "torso": r"torso|body|chest|trunk|spine|belly|back", "hip": r"hip|pelvis|waist",
    "arm": r"arm|shoulder|elbow|sleeve", "hand": r"hand|palm|finger|wrist", "leg": r"leg|thigh|knee|calf|shin|pant|trouser",
    "foot": r"foot|feet|shoe|boot|ankle|toe|sock", "tail": r"tail", "wing": r"wing",
}
BODY_RE = {k: re.compile(v, re.I) for k, v in BODY.items()}
GENERIC = re.compile(r"^\s*(layer|shape layer|null|null object|group|path|precomp|pre-comp|comp|outlines?|vector|rectangle|ellipse|fill|stroke|transform|solid|adjustment layer|\w+ \d+)\s*\d*(\s*copy\s*\d*)?\s*$", re.I)


def has_expression(obj, depth: int = 0) -> bool:
    if depth > 60:
        return False
    if isinstance(obj, dict):
        x = obj.get("x")
        if isinstance(x, str) and x.strip():
            return True
        return any(has_expression(v, depth + 1) for v in obj.values() if isinstance(v, (dict, list)))
    if isinstance(obj, list):
        return any(has_expression(v, depth + 1) for v in obj if isinstance(v, (dict, list)))
    return False


def animated(p) -> bool:
    return isinstance(p, dict) and p.get("a") == 1


def group_transforms(items, out, depth=0):
    """Walk shape items; collect (name, rotation animated, position animated) of every group transform."""
    if depth > 30 or not isinstance(items, list):
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("ty") == "sh" and animated(it.get("ks")):
            out.append(("__path__", False, False))        # an animated bezier path: the shape morphs instead of a transform moving it
        if it.get("ty") == "gr":
            sub = it.get("it") or []
            tr = next((x for x in sub if isinstance(x, dict) and x.get("ty") == "tr"), None)
            out.append((it.get("nm", ""), animated((tr or {}).get("r")), animated((tr or {}).get("p"))))
            group_transforms(sub, out, depth + 1)


def analyse(lj: dict) -> dict:
    comps = [lj.get("layers") or []] + [a["layers"] for a in lj.get("assets") or [] if isinstance(a, dict) and a.get("layers")]
    n_layers = n_null = n_parent = n_rot = n_pos = 0
    max_depth = hinge = 0
    best_chain: list[str] = []
    names: list[str] = []
    for layers in comps:
        by_ind = {l.get("ind"): l for l in layers if isinstance(l, dict) and l.get("ind") is not None}
        rot = {i: animated((l.get("ks") or {}).get("r")) or animated((l.get("ks") or {}).get("rx")) for i, l in by_ind.items()}
        for l in layers:
            if not isinstance(l, dict):
                continue
            n_layers += 1
            n_null += l.get("ty") == 3
            n_parent += l.get("parent") is not None
            ks = l.get("ks") or {}
            n_rot += animated(ks.get("r"))
            pp = ks.get("p")
            n_pos += bool(animated(pp) or (isinstance(pp, dict) and pp.get("s") and (animated(pp.get("x")) or animated(pp.get("y")))))
            names.append(l.get("nm", ""))
        for i, l in by_ind.items():   # depth of the parent chain and the longest run of rotating layers along it
            d, cur, seen, chain = 0, l, set(), [l.get("nm", "")]
            while cur is not None and cur.get("parent") in by_ind and cur.get("ind") not in seen:
                seen.add(cur.get("ind"))
                d += 1
                cur = by_ind[cur["parent"]]
                chain.append(cur.get("nm", ""))
            if d > max_depth:
                max_depth, best_chain = d, chain
            cur, seen, run, best = l, set(), 0, 0
            while cur is not None and cur.get("ind") not in seen:
                seen.add(cur.get("ind"))
                run = run + 1 if rot.get(cur.get("ind")) else 0
                best = max(best, run)
                cur = by_ind.get(cur.get("parent"))
            hinge = max(hinge, best)
    groups: list = []
    for layers in comps:
        for l in layers:
            if isinstance(l, dict) and l.get("shapes"):
                group_transforms(l["shapes"], groups)
    n_paths = sum(g[0] == "__path__" for g in groups)
    groups = [g for g in groups if g[0] != "__path__"]
    names += [g[0] for g in groups]
    n_grot = sum(g[1] for g in groups)
    words = {k for k, rx in BODY_RE.items() if any(rx.search(n) for n in names if n and not GENERIC.match(n))}
    generic = sum(bool(GENERIC.match(n)) or not n for n in names)
    return dict(n_layers=n_layers, n_null=n_null, n_with_parent=n_parent, max_parent_depth=max_depth, rot_layers=n_rot, rot_groups=n_grot, pos_layers=n_pos,
                hinge_run=hinge, n_groups=len(groups), generic_name_share=round(generic / max(len(names), 1), 2), body_words=len(words), words=sorted(words),
                expressions=has_expression(lj), precomps=len(comps) - 1, animated_paths=n_paths, chain=" > ".join(str(c) for c in best_chain[:6]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--per-shard-2leg", type=int, default=17)
    ap.add_argument("--per-shard-4leg", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rel: dict[int, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(str(ROOT / "data" / "vlm" / "animation" / "shard-*.jsonl.zst"))):
        if Path(f).name.startswith("._"):
            continue
        with open(f, "rb") as fh:
            for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                r = orjson.loads(line)
                if r.get("relevant"):
                    rel[r["shard"]][r["body_plan"]].append((str(r["id"]), r.get("subject")))
    shards = sorted(rel)
    rng.shuffle(shards)
    shards = shards[: args.shards]
    want: dict[str, tuple] = {}
    for s in shards:
        for bp, n in (("2leg", args.per_shard_2leg), ("4leg", args.per_shard_4leg)):
            for i, subj in rng.sample(rel[s][bp], min(n, len(rel[s][bp]))):
                want[i] = (bp, subj, s)
    print(f"{len(want)} animations sampled from {len(shards)} shards: {collections.Counter(v[0] for v in want.values())}")
    rows, rx = [], re.compile(rb'^\{"id":"(\d+)"')
    for s in shards:
        path = glob.glob(str(DATASETS / "snapshots" / "*" / "data" / f"metadata-{s:05d}-of-00068.jsonl.zst"))[0]
        with open(path, "rb") as fh:
            for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                m = rx.match(line[:40])
                if not m or m.group(1).decode() not in want:
                    continue
                r = orjson.loads(line)
                lj = r["lottie_json"]
                lj = orjson.loads(lj) if isinstance(lj, str) else lj
                bp, subj, _ = want[str(r["id"])]
                rows.append(dict(id=r["id"], body_plan=bp, subject=subj, shard=want[str(r["id"])][2], **analyse(lj)))
    print(f"analysed {len(rows)}")
    flags = {
        "has_parent_links (>=5 layers with a parent)": lambda r: r["n_with_parent"] >= 5,
        "parent chain >= 3 deep": lambda r: r["max_parent_depth"] >= 3,
        ">= 3 animated rotations (layers + groups)": lambda r: r["rot_layers"] + r["rot_groups"] >= 3,
        "hinge chain: >= 3 rotating layers in one parent chain": lambda r: r["hinge_run"] >= 3,
        ">= 3 distinct body words in layer/group names": lambda r: r["body_words"] >= 3,
        "rig-like by structure (depth >= 3 and >= 3 rotations)": lambda r: r["max_parent_depth"] >= 3 and r["rot_layers"] + r["rot_groups"] >= 3,
        "rig-like and named (structure + >= 3 body words)": lambda r: r["max_parent_depth"] >= 3 and r["rot_layers"] + r["rot_groups"] >= 3 and r["body_words"] >= 3,
        "mostly generic names (> 80%)": lambda r: r["generic_name_share"] > 0.8,
        "uses expressions": lambda r: r["expressions"],
        "morphing paths (>= 10 animated bezier paths)": lambda r: r["animated_paths"] >= 10,
        "morphing only: >= 10 animated paths, fewer than 3 rotations": lambda r: r["animated_paths"] >= 10 and r["rot_layers"] + r["rot_groups"] < 3,
        "any animated rotation at all": lambda r: r["rot_layers"] + r["rot_groups"] >= 1,
        "rotation only inside shape groups (no layer rotates)": lambda r: r["rot_layers"] == 0 and r["rot_groups"] >= 1,
    }
    groups = {"all": rows, "2leg": [r for r in rows if r["body_plan"] == "2leg"], "4leg": [r for r in rows if r["body_plan"] == "4leg"]}
    summ = {g: {k: round(sum(f(r) for r in rs) / max(len(rs), 1), 3) for k, f in flags.items()} | {"n": len(rs)} for g, rs in groups.items()}
    for g, rs in groups.items():
        for k in ("n_layers", "max_parent_depth", "rot_layers", "rot_groups", "n_groups", "hinge_run", "animated_paths"):
            v = sorted(r[k] for r in rs)
            summ[g][k + "_median"] = v[len(v) // 2] if v else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lottie_rig_probe.json").write_text(json.dumps(summ, indent=1))
    with open(OUT / "lottie_rig_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'':58s}" + "".join(f"{g:>9s}" for g in groups))
    for k in flags:
        print(f"{k:58s}" + "".join(f"{100 * summ[g][k]:8.0f}%" for g in groups))
    for k in ("n_layers", "max_parent_depth", "rot_layers", "rot_groups", "n_groups", "hinge_run", "animated_paths"):
        print(f"{'median ' + k:58s}" + "".join(f"{summ[g][k + '_median']:9}" for g in groups))
    print(f"{'n':58s}" + "".join(f"{summ[g]['n']:9d}" for g in groups))
    wc = collections.Counter(w for r in rows for w in r["words"])
    print("body words found in names:", dict(wc.most_common()))


if __name__ == "__main__":
    main()
