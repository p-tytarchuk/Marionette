"""Character-level train/val split and per-character meta.json.

  python3 scripts/dataset/make_splits.py            # dry run: prints the counts, writes nothing
  python3 scripts/dataset/make_splits.py --write    # data/splits/{train,val}.txt, summary.json, data/characters/*/meta.json

Split: ~10% of the CHARACTERS held out (not clips), grouped by skeleton hash so that no held-out character shares a
rig with a training one, stratified by class (biped / quadruped) and source. Group assignment is by md5 of the key
(reproducible, independent of the order of the folder listing); the groups are ranked by hash and taken until the
class/source quota is filled.

Mixamo cannot be grouped: all 50 normalized Mixamo characters share ONE rig and the same 2,308 clips. They are held out
per character (a deterministic 10%), so for them the split tests mesh/skin generalisation, not motion.

This split is NOT the clip split of motion_ae.py / motion_sem.py (those hold out Objaverse rigs / object types and Mixamo
clip families by md5). The overlap is printed so the difference is visible.

meta.json follows data/characters/_example/meta.json: id, class, joint_map, shape_captions. joint_map is the file name in
data/joint_maps/ (only mixamo_biped exists; Objaverse characters get null, not a guess). shape_captions stay empty:
captioning from a static render was never tested. Extra fields: source, skeleton, retarget_target.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHARS = DATA / "characters"
FRAC = 0.10


def h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def load_characters():
    manifest = {(r["source"], r["id"]): r for r in csv.DictReader(open(DATA / "manifest.csv", newline=""))}
    targets = {r["key"]: r for r in csv.DictReader(open(DATA / "retarget_targets.csv", newline=""))}
    out = []
    for d in sorted(CHARS.iterdir()):
        if d.name.startswith(("._", "_")) or not (d / "normalization.json").exists():
            continue
        n = json.loads((d / "normalization.json").read_text())
        m = manifest.get((n["source"], n["id"]))
        out.append(dict(key=d.name, dir=d, source=n["source"], cls=n["class"], id=n["id"],
                        skeleton=m["skeleton"] if m else d.name,       # no manifest row: its own group
                        target=targets.get(d.name, {}).get("status") == "ok"))
    return out


def split(chars):
    val = set()
    for (source, cls), members in sorted(_strata(chars).items()):
        quota = round(FRAC * len(members))
        if source == "mixamo":                                      # one shared rig: hold out single characters
            groups = {c["key"]: [c] for c in members}
        else:
            groups = defaultdict(list)
            for c in members:
                groups[c["skeleton"]].append(c)
        taken = 0
        for g in sorted(groups, key=lambda k: h("g:" + k)):
            if taken >= quota:
                break
            # take a group while it does not overshoot the quota by more than half of the group's size
            if taken + len(groups[g]) - quota > len(groups[g]) / 2 and taken > 0:
                continue
            val.update(c["key"] for c in groups[g])
            taken += len(groups[g])
    return val


def _strata(chars):
    s = defaultdict(list)
    for c in chars:
        s[(c["source"], c["cls"])].append(c)
    return s


def clip_split_overlap(chars, val):
    """Objaverse characters whose rig is a validation rig in motion_ae.py's clip split (rig = skeleton hash of the clip)."""
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    try:
        from motion_ae import split_of
    except Exception as e:                                           # torch missing in this interpreter
        return f"not computed ({type(e).__name__})"
    ae_val = {c["key"] for c in chars if c["source"] == "objaverse" and split_of(dict(dataset="objaverse", clip="", rig=c["id"])) == "val"}
    mine = {k for k in val if k.startswith("objaverse_")}
    return f"{len(mine & ae_val)} of {len(mine)} held-out Objaverse characters are also held out by motion_ae.split_of"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    chars = load_characters()
    val = split(chars)
    by_key = {c["key"]: c for c in chars}
    # leakage check: no skeleton on both sides (Mixamo's single rig is exempt by design)
    both = {c["skeleton"] for c in chars if c["key"] in val and c["source"] != "mixamo"} & \
           {c["skeleton"] for c in chars if c["key"] not in val and c["source"] != "mixamo"}
    assert not both, f"{len(both)} skeletons on both sides"

    summary = dict(fraction=FRAC, rule="md5 of skeleton hash (Mixamo: of the character key), stratified by source and class",
                   n_characters=len(chars), n_val=len(val), per_stratum={}, retarget_targets_val=sum(by_key[k]["target"] for k in val),
                   retarget_targets_total=sum(c["target"] for c in chars),
                   note="Mixamo: one shared rig and shared clips, so held-out Mixamo characters test mesh/skin generalisation, not motion.")
    for (source, cls), members in sorted(_strata(chars).items()):
        v = [c for c in members if c["key"] in val]
        summary["per_stratum"][f"{source}/{cls}"] = dict(total=len(members), val=len(v), val_targets=sum(c["target"] for c in v),
                                                         skeletons=len({c["skeleton"] for c in members}))
    summary["overlap_with_clip_split"] = clip_split_overlap(chars, val)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.write:
        print("\n(dry run, nothing written; --write to write)")
        return

    sp = DATA / "splits"
    sp.mkdir(exist_ok=True)
    (sp / "train.txt").write_text("".join(c["key"] + "\n" for c in chars if c["key"] not in val))
    (sp / "val.txt").write_text("".join(c["key"] + "\n" for c in chars if c["key"] in val))
    (sp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for c in chars:
        meta = dict(id=c["key"], **{"class": c["cls"]}, joint_map="mixamo_biped" if c["source"] == "mixamo" else None,
                    shape_captions=[], source=c["source"], skeleton=c["skeleton"], retarget_target=c["target"])
        (c["dir"] / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {sp}/train.txt, val.txt, summary.json and {len(chars)} meta.json")


if __name__ == "__main__":
    main()
