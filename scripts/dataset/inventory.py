#!/usr/bin/env python3
"""Full inventory of sources/unimate (UniML3D): every clip NPZ, caption file, quality list, rig and mesh.

Opens all export/<mixamo|objaverse>/motions/*.npz (animation arrays only; the large skin_matrix is
only header-read), reads the caption / quality-list / joint-name files, joins them with the
canonical meshes, data/manifest.csv and data/normalize_report.csv, and writes to data/inventory/:

  clips.csv     one row per clip: frames, fps, joints, rig signature, root-motion stats, loop guess,
                captions, quality-list status, and whether the clip is usable for Marionette (+ reason)
  rigs.csv      one row per rig (object type): clips, topology consistency, size/units hint, mesh and
                joint-name match with the canonical GLB, character normalization status, class
  summary.json  all headline numbers
  report.md     the same numbers as readable tables (also printed)

Truebones has annotations only (no motion files); it is counted from the index tables.

Usage:
  python3 scripts/dataset/inventory.py                 # everything (reads ~12.7k NPZ files; a few minutes)
  python3 scripts/dataset/inventory.py --limit 300     # quick trial
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import struct
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sources" / "unimate"
DATASETS = ("mixamo", "objaverse")


# ----------------------------------------------------------------------------- small readers
def npz_shape(zf, name):
    with zf.open(name + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        hdr = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        return hdr(f)[0]


def quat_to_mat(q):  # w first
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fk_rest(parents, pos, rot):
    """Global joint positions of the rest pose (parents may come in any order)."""
    J = len(parents)
    P, R, done = np.zeros((J, 3)), [None] * J, [False] * J

    def go(j):
        if done[j]:
            return
        p = parents[j]
        Rl = quat_to_mat(rot[j] / (np.linalg.norm(rot[j]) or 1))
        if p < 0:
            P[j], R[j] = pos[j], Rl
        else:
            go(p)
            P[j], R[j] = P[p] + R[p] @ pos[j], R[p] @ Rl
        done[j] = True

    for j in range(J):
        go(j)
    return P


def glb_joint_names(path):
    with open(path, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<I4s", f.read(8))
        js = json.loads(f.read(n))
    seen, names = set(), []
    for s in js.get("skins", []):
        for j in s["joints"]:
            if j not in seen:
                seen.add(j)
                names.append(js["nodes"][j].get("name", ""))
    return names


def read_list(path):
    """`name  # reason` or `name N  # reason` files -> {name: (rest_of_line_before_comment, reason)}"""
    out = {}
    if not Path(path).exists():
        return out
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        body, _, why = ln.partition("#")
        parts = body.split()
        out[parts[0]] = (parts[1] if len(parts) > 1 else "", why.strip())
    return out


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


# ----------------------------------------------------------------------------- per-clip analysis
def analyze_clip(job):
    ds, path = job
    stem = path.stem
    row = dict(dataset=ds, clip=stem, object_type="mixamo" if ds == "mixamo" else stem.split("-", 1)[0],
               action=stem if ds == "mixamo" else stem.split("-", 1)[1] if "-" in stem else "", path=str(path), error="")
    rig = None
    try:
        with zipfile.ZipFile(path) as zf:
            skin = npz_shape(zf, "skin_matrix")
        d = np.load(path, allow_pickle=False)
        pos, rot = d["anim_local_pos"].astype(float), d["anim_local_rot"].astype(float)
        names, parents = [str(n) for n in d["names"]], d["parents"].astype(int)
        T, J = pos.shape[:2]
        fps = int(d["fps"])
        rest_pos, rest_rot = d["rest_local_pos"].astype(float), d["rest_local_rot"].astype(float)
        sig = hashlib.sha1(json.dumps([names, parents.tolist()]).encode()).hexdigest()[:12]
        roots = np.flatnonzero(parents < 0)
        r0 = int(roots[0]) if len(roots) else 0
        P = fk_rest(parents, rest_pos, rest_rot)
        height = float(P[:, 1].max() - P[:, 1].min())
        rp = pos[:, r0, :]
        step = np.linalg.norm(np.diff(rp[:, [0, 2]], axis=0), axis=1) if T > 1 else np.zeros(1)
        qn = np.linalg.norm(rot, axis=2)
        q0, q1 = rot[0] / np.maximum(qn[0][:, None], 1e-9), rot[-1] / np.maximum(qn[-1][:, None], 1e-9)
        loop_gap = float(np.degrees(2 * np.arccos(np.clip(np.abs((q0 * q1).sum(1)), 0, 1))).mean())
        qa = rot / np.maximum(qn[..., None], 1e-9)
        spread = float(np.degrees(2 * np.arccos(np.clip(np.abs((qa * q0[None]).sum(2)), 0, 1))).mean())  # mean pose distance to frame 0
        row.update(frames=T, fps=fps, duration_s=round(T / fps, 3) if fps else "", joints=J, verts=int(skin[0]),
                   rig_sig=sig, rest_height_src=round(height, 4),
                   root_travel_rel=round(float(step.sum()) / height, 3) if height else "",
                   root_net_rel=round(float(np.linalg.norm(rp[-1, [0, 2]] - rp[0, [0, 2]])) / height, 3) if height else "",
                   root_dy_rel=round(float(np.ptp(rp[:, 1])) / height, 3) if height else "",
                   max_root_step_rel=round(float(step.max()) / height, 4) if height else "",
                   loop_gap_deg=round(loop_gap, 2), pose_spread_deg=round(spread, 2),
                   finite=int(bool(np.isfinite(pos).all() and np.isfinite(rot).all())),
                   max_quat_norm_err=round(float(np.abs(qn - 1).max()), 5))
        rig = dict(sig=sig, names=names, parents=parents.tolist(), height=height, verts=int(skin[0]),
                   rest_root_y=float(P[r0, 1]))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row, rig


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "inventory"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per dataset (trial run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--loop-deg", type=float, default=8.0, help="max mean joint angle between first and last frame for a loop guess")
    ap.add_argument("--loop-ratio", type=float, default=0.3, help="and that gap must be below this fraction of the clip's pose spread")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets.split(",")

    # ---- gather inputs
    jobs = []
    for ds in datasets:
        files = sorted(p for p in (SRC / "export" / ds / "motions").rglob("*.npz") if not p.name.startswith("._"))
        jobs += [(ds, p) for p in (files[:args.limit] if args.limit else files)]
    print(f"clips to read: {len(jobs)}", file=sys.stderr)

    rows, rigs_first = [], {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (row, rig) in enumerate(ex.map(analyze_clip, jobs), 1):
            rows.append(row)
            if rig:
                rigs_first.setdefault((row["dataset"], row["object_type"]), rig)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", file=sys.stderr)

    # ---- side files
    cap = {ds: {k: load_json(SRC / "export" / ds / f"motion_captions{s}.json", {})
                for k, s in (("caption", ""), ("generic", "_generic"), ("detail", "_detail"))} for ds in datasets}
    filtered = {ds: read_list(SRC / "export" / ds / "filtered_clips.txt") for ds in datasets}
    trims = read_list(SRC / "export" / "objaverse" / "clip_trims.txt")
    activity = read_list(SRC / "export" / "objaverse" / "activity_keep.txt")
    bad_rigs = {ds: read_list(SRC / "export" / ds / "filtered_objects.txt") for ds in datasets}
    cats = {}
    for ds in datasets:
        for cat, objs in load_json(SRC / "export" / ds / "category_groups.json", {}).items():
            for o in objs:
                cats[(ds, o)] = cat
    index_clips = collections.Counter()
    with open(SRC / "clips.csv", newline="") as f:
        for r in csv.DictReader(f):
            index_clips[r["dataset"]] += 1
    manifest, norm = {}, {}
    mp, npth = ROOT / "data" / "manifest.csv", ROOT / "data" / "normalize_report.csv"
    if mp.exists():
        with open(mp, newline="") as f:
            manifest = {(r["source"], r["id"]): r for r in csv.DictReader(f)}
    if npth.exists():
        with open(npth, newline="") as f:
            norm = {(r["source"], r["id"]): r for r in csv.DictReader(f)}
    n_norm_mixamo = sum(1 for (s, _), r in norm.items() if s == "mixamo" and r["status"] == "ok")

    # ---- rigs
    rig_rows = {}
    by_rig = collections.defaultdict(list)
    for r in rows:
        by_rig[(r["dataset"], r["object_type"])].append(r)
    for (ds, ot), clips in sorted(by_rig.items()):
        first = rigs_first.get((ds, ot))
        sigs = collections.Counter(c.get("rig_sig") for c in clips if c.get("rig_sig"))
        glb = SRC / "canonical_assets" / ds / (f"{ot}.glb" if ds == "objaverse" else "")
        has_glb = ds == "objaverse" and glb.exists()
        match = ""
        glb_j = 0
        if has_glb and first:
            gn = glb_joint_names(glb)
            glb_j = len(gn)
            match = round(sum(n in set(first["names"]) for n in gn) / max(len(gn), 1), 3)
        elif ds == "mixamo" and first:
            sample = next(iter(sorted((SRC / "canonical_assets" / "mixamo").glob("[!.]*.glb"))), None)
            if sample:
                gn = glb_joint_names(sample)
                glb_j = len(gn)
                match = round(sum(n in set(first["names"]) for n in gn) / max(len(gn), 1), 3)
        if ds == "mixamo":
            status = f"normalized x{n_norm_mixamo}" if n_norm_mixamo else "no_character"
        elif (ds, ot) in norm:
            nr = norm[(ds, ot)]
            status = "normalized" if nr["status"] == "ok" else f"skipped:{nr['reason'].split('(')[0]}"
        elif (ds, ot) in manifest:
            status = "dropped:" + (manifest[(ds, ot)]["reason"] or "kept_not_normalized")
        elif not has_glb:
            status = "no_canonical_mesh"
        else:
            status = "not_in_manifest"
        frames = [c["frames"] for c in clips if "frames" in c]
        rig_rows[(ds, ot)] = dict(
            dataset=ds, object_type=ot, category=cats.get((ds, ot), ""), clips=len(clips),
            clips_read=len(frames), total_frames=sum(frames), joints=len(first["names"]) if first else "",
            distinct_topologies=len(sigs), rig_sig=sigs.most_common(1)[0][0] if sigs else "",
            verts=first["verts"] if first else "", rest_height_src=round(first["height"], 3) if first else "",
            rest_root_y_src=round(first["rest_root_y"], 3) if first else "",
            canonical_glb=int(has_glb or ds == "mixamo"), glb_joints=glb_j, glb_names_in_npz=match,
            flagged_rig=int(ot in bad_rigs[ds]), character_status=status,
            manifest_keep=manifest.get((ds, ot), {}).get("keep", ""))

    # ---- clips: captions, quality lists, usable
    for r in rows:
        ds, stem = r["dataset"], r["clip"]
        for k in ("caption", "generic", "detail"):
            r[k] = cap[ds][k].get(stem, "")
        r["trim"] = int(trims[stem][0]) if stem in trims and ds == "objaverse" and trims[stem][0].isdigit() else 0
        r["quality_list"] = ""
        if stem in filtered[ds]:
            r["quality_list"] = "filtered:" + filtered[ds][stem][1].split(":")[0].strip()
        elif stem in activity:
            r["quality_list"] = "activity_keep"
        rg = rig_rows[(ds, r["object_type"])]
        r["category"] = rg["category"]
        r["character_status"] = rg["character_status"]
        # loop guess: first and last pose are close in absolute terms AND small compared with how far
        # the pose travels inside the clip (static idles count as loops)
        r["loop_guess"] = (int(r["loop_gap_deg"] <= args.loop_deg and (r["pose_spread_deg"] < 2 or r["loop_gap_deg"] <= args.loop_ratio * r["pose_spread_deg"]))
                           if "frames" in r else "")
        why = ""
        if r["error"]:
            why = "unreadable"
        elif not r["finite"]:
            why = "non_finite"
        elif r["quality_list"].startswith("filtered"):
            why = "quality_list"
        elif r["frames"] - r["trim"] < args.min_frames:
            why = "too_short"
        elif rg["flagged_rig"]:
            why = "flagged_rig"
        elif not rg["character_status"].startswith("normalized"):
            why = "no_normalized_character"
        elif rg["distinct_topologies"] > 1:
            why = "mixed_topology_in_rig"
        r["usable"] = int(not why)
        r["not_usable_reason"] = why
        r["character_class"] = {"bipedal": "biped", "quadrupedal": "quadruped"}.get(r["category"], r["category"])

    # ---- write csvs
    clip_cols = ["dataset", "clip", "object_type", "action", "category", "character_class", "frames", "trim", "fps",
                 "duration_s", "joints", "verts", "rig_sig", "rest_height_src", "root_travel_rel", "root_net_rel",
                 "root_dy_rel", "max_root_step_rel", "loop_gap_deg", "pose_spread_deg", "loop_guess", "finite", "max_quat_norm_err",
                 "quality_list", "character_status", "usable", "not_usable_reason", "caption", "generic", "detail",
                 "error", "path"]
    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, clip_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    rig_list = list(rig_rows.values())
    with open(out / "rigs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rig_list[0]))
        w.writeheader()
        w.writerows(rig_list)

    # ---- summary
    ok = [r for r in rows if "frames" in r]
    S = {}
    S["clips_read"] = {ds: sum(r["dataset"] == ds for r in rows) for ds in datasets}
    S["clips_in_index_csv"] = dict(index_clips)
    S["unreadable"] = [r["clip"] for r in rows if r["error"]][:20]
    S["fps"] = dict(collections.Counter(r["fps"] for r in ok))
    S["non_finite_clips"] = sum(not r["finite"] for r in ok)
    S["quat_norm_err_max"] = max((r["max_quat_norm_err"] for r in ok), default=0)
    for ds in datasets:
        fr = np.array([r["frames"] for r in ok if r["dataset"] == ds])
        if len(fr):
            S.setdefault("frames", {})[ds] = dict(total=int(fr.sum()), hours=round(float(fr.sum()) / 30 / 3600, 2),
                                                  min=int(fr.min()), median=float(np.median(fr)), p95=round(float(np.percentile(fr, 95)), 1), max=int(fr.max()))
    S["rigs"] = {ds: sum(1 for k in rig_rows if k[0] == ds) for ds in datasets}
    S["rigs_with_mixed_topology"] = [f"{k[0]}:{k[1]}" for k, v in rig_rows.items() if v["distinct_topologies"] > 1][:20]
    S["unique_skeletons"] = len({v["rig_sig"] for v in rig_list if v["rig_sig"]})
    S["rig_status"] = dict(collections.Counter(f"{v['dataset']}:{v['character_status'].split(' ')[0]}" for v in rig_list))
    objav = [v for v in rig_list if v["dataset"] == "objaverse"]
    S["objaverse_clips_per_rig"] = dict(collections.Counter(min(v["clips"], 10) for v in objav))
    S["objaverse_glb_name_match_lt_1"] = sum(1 for v in objav if v["glb_names_in_npz"] != "" and v["glb_names_in_npz"] < 1)
    canon = {p.stem for p in (SRC / "canonical_assets" / "objaverse").glob("[!.]*.glb")}
    S["canonical_objaverse_glb_without_clips"] = len(canon - {k[1] for k in rig_rows if k[0] == "objaverse"}) if not args.limit else "n/a (trial)"
    S["captions_missing"] = {k: sum(not r[k] for r in rows) for k in ("caption", "generic", "detail")}
    S["quality_list"] = dict(collections.Counter(r["quality_list"].split(":")[0] for r in rows if r["quality_list"]))
    S["trimmed_clips"] = sum(1 for r in rows if r["trim"])
    S["clips_by_character_status"] = dict(collections.Counter(f"{r['dataset']}:{r['character_status'].split(' ')[0]}" for r in rows))
    for ds in datasets:
        h = np.array([v["rest_height_src"] for v in rig_list if v["dataset"] == ds and v["rest_height_src"] != ""], float)
        if len(h):
            S.setdefault("rest_height_src_units", {})[ds] = dict(min=round(float(h.min()), 3), p10=round(float(np.percentile(h, 10)), 3),
                                                                 median=round(float(np.median(h)), 3), p90=round(float(np.percentile(h, 90)), 3), max=round(float(h.max()), 3))
    S["usable_total"] = sum(r["usable"] for r in rows)
    S["not_usable_reasons"] = dict(collections.Counter(r["not_usable_reason"] for r in rows if r["not_usable_reason"]))
    cls = collections.defaultdict(lambda: dict(clips=0, characters=set(), loops=0, rootmotion=0))
    for r in rows:
        if r["usable"]:
            c = cls[r["character_class"]]
            c["clips"] += 1
            c["characters"].add((r["dataset"], r["object_type"]))
            c["loops"] += r["loop_guess"]
            c["rootmotion"] += int(r["root_travel_rel"] > 0.5)
    S["usable_by_class"] = {k: dict(clips=v["clips"], rigs_with_clips=len(v["characters"]), loop_guess=v["loops"],
                                    root_motion_gt_half_body=v["rootmotion"]) for k, v in cls.items()}
    chars = [(k, v) for k, v in norm.items() if v["status"] == "ok"]
    have = collections.Counter()
    for (ds, oid), _ in chars:
        n = sum(r["usable"] for r in by_rig.get((ds, "mixamo" if ds == "mixamo" else oid), []))
        have[(ds, "0" if n == 0 else "1" if n == 1 else "2-4" if n < 5 else "5+")] += 1
    S["normalized_characters_by_usable_clips"] = {f"{a}:{b}": n for (a, b), n in sorted(have.items())}
    S["features_mixamo_clips"] = len(list((SRC / "features" / "mixamo" / "motions").glob("[!.]*.npz")))
    tb = load_json(SRC / "export" / "truebones" / "joint_names.json", {})
    S["truebones"] = dict(rigs=len(tb), clips_in_index=index_clips.get("truebones", 0), motion_files=0)
    S["settings"] = dict(min_frames=args.min_frames, loop_deg=args.loop_deg, loop_ratio=args.loop_ratio)
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))

    # ---- report
    L = ["# UniML3D inventory", "", f"{len(rows)} clips read, {len(rig_list)} rigs" + (" (trial run)" if args.limit else ""), ""]
    L += ["## Clips", "", "| dataset | clips | in index | hours | frames min / median / p95 / max |", "|---|---|---|---|---|"]
    for ds in datasets:
        f = S.get("frames", {}).get(ds, {})
        L.append(f"| {ds} | {S['clips_read'][ds]} | {index_clips.get(ds, '')} | {f.get('hours', '')} | "
                 f"{f.get('min', '')} / {f.get('median', '')} / {f.get('p95', '')} / {f.get('max', '')} |")
    L += ["", f"fps: {S['fps']}; non-finite clips: {S['non_finite_clips']}; max quaternion-norm error: {S['quat_norm_err_max']}; "
          f"unreadable: {len(S['unreadable'])}", "",
          "## Rigs and characters", "", f"- unique skeletons (names + hierarchy): {S['unique_skeletons']}",
          f"- rigs with more than one topology across their clips: {len(S['rigs_with_mixed_topology'])}",
          f"- Objaverse rigs whose canonical GLB joint names are not all in the clip NPZ: {S['objaverse_glb_name_match_lt_1']}",
          f"- canonical Objaverse GLBs without any clip: {S['canonical_objaverse_glb_without_clips']}",
          f"- Objaverse clips per rig (10 = 10 or more): {S['objaverse_clips_per_rig']}",
          f"- character status per rig: {S['rig_status']}", "",
          "## Captions and quality lists", "", f"- clips without caption / generic / detail: {S['captions_missing']}",
          f"- quality lists: {S['quality_list']}; clips with a head trim: {S['trimmed_clips']}", "",
          f"- clips by character status: {S['clips_by_character_status']}",
          f"- rig height in source units (clip NPZ are NOT metric; Mixamo is centimetres): {S['rest_height_src_units']}", "",
          "## Usable for Marionette", "",
          f"usable clips: **{S['usable_total']}** of {len(rows)}; not usable: {S['not_usable_reasons']}", "",
          "| class | usable clips | rigs with clips | loop guess | root travel > 0.5 body |", "|---|---|---|---|---|"]
    for k, v in sorted(S["usable_by_class"].items()):
        L.append(f"| {k} | {v['clips']} | {v['rigs_with_clips']} | {v['loop_guess']} | {v['root_motion_gt_half_body']} |")
    L += ["", f"normalized characters by number of usable clips: {S['normalized_characters_by_usable_clips']}", "",
          f"Truebones: {S['truebones']['rigs']} rigs, {S['truebones']['clips_in_index']} clips in the index, no motion files.",
          f"`features/mixamo` holds {S['features_mixamo_clips']} of 2,162 training clips.", "",
          f"Settings: min frames {args.min_frames}, loop gap {args.loop_deg} deg and ratio {args.loop_ratio}. Details: clips.csv, rigs.csv, summary.json."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
