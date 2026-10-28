"""Analyse the Lottie corpus tables copied from the server (data/vlm: Qwen labels, data/lottie: motion parse) and join them.

    python scripts/lottie2d/analyze_lottie.py vlm                  # what the Qwen labelling found (data/vlm/animation/shard-*.jsonl.zst)
    python scripts/lottie2d/analyze_lottie.py motion               # what the motion parse found (data/lottie/animation/shard-*.jsonl.zst) + the join
    python scripts/lottie2d/analyze_lottie.py motion --lottie-dir /tmp/x/animation --min-good-tracks 2

Both read finished shards only, stream them (a few GB of JSON never sit in memory), and write a JSON summary to data/analysis/. The join's output,
data/analysis/candidates.csv, is the set of animations that Qwen called a whole living 2- or 4-legged character AND whose motion parsed cleanly
(no expressions, no raster layers, at least --min-good-tracks rotation tracks whose sine fit has R^2 >= --r2): the training pool for text -> motion.
Needs numpy, zstandard, orjson (the scratchpad venv or ~/envs/marionette).
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import orjson
import zstandard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "analysis"


def rows(path):
    with open(path, "rb") as f:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 22):
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                continue


def shards(d: Path):
    return sorted(p for p in glob.glob(str(d / "shard-*.jsonl.zst")) if not Path(p).name.startswith("._"))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def q(v, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}


# ----------------------------------------------------------------------------- vlm
def load_vlm(d: Path):
    """id -> (relevant, body_plan, subject, full_body, caption_lottie) for every row of every finished shard."""
    out = {}
    for s in shards(d):
        for r in rows(s):
            out[str(r["id"])] = (bool(r.get("relevant")), r.get("body_plan"), r.get("subject"), r.get("full_body"), r.get("caption_lottie") or "")
    return out


def cmd_vlm(args) -> None:
    d = Path(args.vlm_dir)
    sh = shards(d)
    tot = collections.Counter()
    bp, subj, skip, models = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    rel_bp, rel_subj = collections.Counter(), collections.Counter()
    tags, words = collections.Counter(), collections.Counter()
    cap_len, per_shard = [], {}
    agree = collections.Counter()
    for s in sh:
        n = rel = proc = 0
        for r in rows(s):
            n += 1
            tot["rows"] += 1
            p, rl = bool(r.get("processed")), bool(r.get("relevant"))
            proc += p
            rel += rl
            tot["processed"] += p
            tot["relevant"] += rl
            skip[r.get("skip_reason")] += 1
            if p:
                bp[r.get("body_plan")] += 1
                subj[r.get("subject")] += 1
                models[r.get("qwen_model")] += 1
                tot["icon"] += bool(r.get("is_ui_icon"))
                tot["not_full_body"] += r.get("full_body") is False
                agree[(bool(r.get("relevant_2b")), rl)] += 1
            if rl:
                rel_bp[r.get("body_plan")] += 1
                rel_subj[r.get("subject")] += 1
                cap_len.append(len((r.get("caption_lottie") or "").split()))
                if len(tags) < 200000:
                    tags.update(t for t in (r.get("tag") or [])[:8])
                words.update(w for w in re.findall(r"[a-z]+", (r.get("caption_lottie") or "").lower()) if len(w) > 3)
        per_shard[Path(s).name.split(".")[0]] = dict(rows=n, processed=proc, relevant=rel)
    summ = dict(shards_finished=len(sh), shards_total=68, **tot, skip_reason={str(k): v for k, v in skip.items()}, body_plan_processed=dict(bp),
                subject_processed=dict(subj), qwen_model=dict(models), relevant_body_plan=dict(rel_bp), relevant_subject=dict(rel_subj),
                caption_words_relevant=q(cap_len), per_shard=per_shard)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vlm_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    n, p, r = tot["rows"], tot["processed"], tot["relevant"]
    print(f"{len(sh)} of 68 stage-2 shards finished: {n:,} rows; Qwen looked at {p:,} ({pct(p, n)}), skipped {skip.get('no_keyword', 0):,} with no person/animal word")
    print(f"relevant (whole living character, 2 or 4 legs, not an icon): {r:,} = {pct(r, p)} of the processed rows")
    print("  body plan of relevant rows:", dict(rel_bp.most_common()))
    print("  subject of relevant rows:  ", dict(rel_subj.most_common()))
    print("  all processed rows, body plan:", dict(bp.most_common()), "| icons:", tot["icon"], "| not full body:", tot["not_full_body"])
    print("  2B pre-filter vs 8B verdict (relevant_2b, relevant):", {str(k): v for k, v in agree.items()})
    print("  caption words (relevant rows):", q(cap_len))
    print("  most common tags:", ", ".join(f"{t} ({c})" for t, c in tags.most_common(12)))
    stop = set("with their that this from while shows into which have over then other being each also back forward".split())
    print("  most common caption words:", ", ".join(w for w, _ in words.most_common(60) if w not in stop)[:260])
    print(f"-> {OUT / 'vlm_summary.json'}")


# ----------------------------------------------------------------------------- motion
def cmd_motion(args) -> None:
    d = Path(args.lottie_dir)
    sh = shards(d)
    if not sh:
        sys.exit(f"no finished shards in {d} (copy data/lottie/animation from the server first)")
    vlm = load_vlm(Path(args.vlm_dir)) if Path(args.vlm_dir).exists() else {}
    c = collections.Counter()
    dur, ntr, r2s, acfs, rng = [], [], [], [], []
    fps = collections.Counter()
    fp_count = collections.Counter()
    cand = []
    for s in sh:
        for r in rows(s):
            c["rows"] += 1
            if "error" in r:
                c["error"] += 1
                continue
            c["expressions"] += bool(r["expressions"])
            c["raster"] += bool(r["raster"])
            fps[round(r["fr"])] += 1
            dur.append(r["duration_s"])
            tr = r["tracks"]
            ntr.append(r["n_tracks_fitted"])
            c["with_track"] += bool(tr)
            c["corrupt_tracks"] += r["n_corrupt"]
            if r.get("fingerprint"):
                fp_count[r["fingerprint"]] += 1
            good = [t for t in tr if t.get("r2", 0) >= args.r2]
            for t in tr:
                r2s.append(t.get("r2", 0))
                acfs.append(t["acf"])
                c["tracks"] += 1
                c["periodic"] += bool(t["periodic"])
                c["loop"] += bool(t["loop"])
                c["in_range"] += bool(t.get("in_range"))
                rng.append(np.degrees(t["range"]) if t["prop"].startswith("rotation") else 0)
            c["good_tracks"] += len(good)
            v = vlm.get(str(r["id"]))
            if v:
                c["joined"] += 1
                if v[0]:
                    c["joined_relevant"] += 1
                    ok = not r["expressions"] and not r["raster"] and len(good) >= args.min_good_tracks
                    c["relevant_clean"] += ok
                    if ok:
                        cand.append(dict(id=r["id"], body_plan=v[1], subject=v[2], full_body=v[3], fr=r["fr"], duration_s=round(r["duration_s"], 2),
                                         n_tracks=r["n_tracks_fitted"], n_good_tracks=len(good), periodic_tracks=sum(t["periodic"] for t in tr),
                                         fingerprint=r.get("fingerprint"), caption=v[4]))
    n = c["rows"]
    dupes = sum(v - 1 for v in fp_count.values() if v > 1)
    summ = dict(shards_finished=len(sh), **c, duration_s=q(dur), tracks_per_row=q(ntr), r2=q(r2s), acf=q(acfs), fps_top=dict(fps.most_common(5)),
                rows_sharing_a_fingerprint=dupes, unique_fingerprints=len(fp_count), candidates=len(cand))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motion_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"{len(sh)} of 68 motion shards: {n:,} rows, {c['error']} parse errors; expressions {pct(c['expressions'], n)}, raster layers {pct(c['raster'], n)}")
    print(f"rows with at least one fitted rotation track: {c['with_track']:,} ({pct(c['with_track'], n)}); tracks fitted: {c['tracks']:,} ({np.mean(ntr):.1f} per row)")
    print(f"tracks: periodic {pct(c['periodic'], c['tracks'])}, closed loops {pct(c['loop'], c['tracks'])}, inside the +-90 deg token range {pct(c['in_range'], c['tracks'])}, sine R2 >= {args.r2}: {pct(c['good_tracks'], c['tracks'])}")
    print("  sine-fit R2 percentiles:", q(r2s), "| duration s:", q(dur), "| frame rates:", dict(fps.most_common(4)))
    print(f"near-duplicates by fingerprint: {dupes:,} rows share a fingerprint with an earlier row ({len(fp_count):,} distinct)")
    if vlm:
        print(f"joined with the Qwen table: {c['joined']:,} rows, of which {c['joined_relevant']:,} relevant; clean + >= {args.min_good_tracks} good tracks: {c['relevant_clean']:,}")
        bp = collections.Counter(x["body_plan"] for x in cand)
        print("  candidates by body plan:", dict(bp.most_common()))
        with open(OUT / "candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(cand[0]) if cand else ["id"])
            w.writeheader()
            w.writerows(cand)
        print(f"-> {OUT / 'candidates.csv'} ({len(cand):,} rows)")
    print(f"-> {OUT / 'motion_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["vlm", "motion"])
    ap.add_argument("--vlm-dir", default=str(ROOT / "data" / "vlm" / "animation"))
    ap.add_argument("--lottie-dir", default=str(ROOT / "data" / "lottie" / "animation"))
    ap.add_argument("--r2", type=float, default=0.9, help="a track counts as well fitted at this sine-fit R^2")
    ap.add_argument("--min-good-tracks", type=int, default=2)
    args = ap.parse_args()
    {"vlm": cmd_vlm, "motion": cmd_motion}[args.cmd](args)


if __name__ == "__main__":
    main()
