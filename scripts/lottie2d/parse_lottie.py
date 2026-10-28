"""Parse the Lottie corpora (../Datasets/LottieAnimation-660K, LottieSVG-10M) into per-animation motion records.

One output row per animation: caption/tags, size/fps/duration, flags (expressions, raster assets), and for every
animated rotation/position/scale track the single-sine fit r + a*sin(2*pi*(s*t + phi)) (unclamped and clamped
variants, R^2, periodicity, quantized value tokens for rotation). The animation-level fingerprint is for dedupe.

Reads only the metadata shards (metadata-*.jsonl.zst); the MP4/PNG tar shards are not touched.
Resumable: a finished shard writes <out>/<dataset>/shard-NNNNN.jsonl.zst plus shard-NNNNN.done.json; rerun skips it.

    ~/envs/marionette/bin/python scripts/parse_lottie.py --dataset animation --limit-rows 300 --shards 0 --out /tmp/x   # benchmark
    ~/envs/marionette/bin/python scripts/parse_lottie.py --dataset animation --workers 18                              # full run
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker; the pool is the parallelism
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import orjson
import zstandard

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir())
for _p in (HERE / "lottie", ROOT / "lottie"):   # SciData layout: scripts/lottie; Marionette dev layout: lottie/
    if _p.exists():
        sys.path.insert(0, str(_p))
from curve_stats import MAX_ABS, MAX_SAMPLES, NEGLIGIBLE, autocorr_peak, curve_shape, fingerprint, is_loop  # noqa: E402
from lottie_curves import iter_tracks  # noqa: E402
from primitive import fit_clamped, fit_unclamped  # noqa: E402
from quantizer import DEFAULT, encode_curve  # noqa: E402

SCI = (ROOT / "datasets").exists()
DATASETS = ROOT / "datasets" if SCI else ROOT.parent / "Datasets"
SRC = {"animation": "LottieAnimation-660K", "svg": "LottieSVG-10M"}
warnings.filterwarnings("ignore")


_FIT_CACHE: dict = {}   # per worker: identical curves (re-exports under consecutive ids, mirrored limbs, instanced precomps) are fitted once
_FIT_CACHE_MAX = 200_000
STATS = {"fits": 0, "cache_hits": 0}


CLAMP_SKIP_R2 = 0.98   # the one-sided variant (85% of the fit time) is only tried when the plain sine explains < this; set > 1 to always try


def fit_both(t: np.ndarray, y: np.ndarray, fr: float, s_cap: float = 8.0):
    """primitive.fit(), except the clamped search is skipped for curves the plain sine already explains."""
    s_max = min(s_cap, 0.45 * fr)
    unc = fit_unclamped(t, y, s_max)
    if unc.r2 >= CLAMP_SKIP_R2:
        return unc, None
    return unc, fit_clamped(t, y, s_max, unc)


def cached_fit(t: np.ndarray, y: np.ndarray, fr: float):
    key = (round(fr, 3), len(y), hashlib.blake2b(np.round(y, 6).tobytes() + np.round(t, 6).tobytes(), digest_size=12).digest())
    hit = _FIT_CACHE.get(key)
    if hit is not None:
        STATS["cache_hits"] += 1
        return hit
    STATS["fits"] += 1
    res = fit_both(t, y, fr)
    if len(_FIT_CACHE) < _FIT_CACHE_MAX:
        _FIT_CACHE[key] = res
    return res


def has_expression(obj, depth: int = 0) -> bool:
    """A dict with a non-empty STRING under "x" (a bezier handle's "x" is a number/list)."""
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


def has_raster(lj: dict) -> bool:
    return any(isinstance(a, dict) and "p" in a and "layers" not in a for a in lj.get("assets") or [])


def count_layers(lj: dict) -> int:
    n = len(lj.get("layers") or [])
    for a in lj.get("assets") or []:
        if isinstance(a, dict):
            n += len(a.get("layers") or [])
    return n


def parse_row(row: dict, props: set[str], max_tracks: int, do_fit: bool) -> dict:
    lj = row.get("lottie_json")
    if isinstance(lj, str):
        lj = orjson.loads(lj)
    fr = float(lj.get("fr") or 30)
    ip, op = float(lj.get("ip", 0) or 0), float(lj.get("op", 0) or 0)
    out = {
        "id": row["id"], "caption": row.get("caption"), "tag": row.get("tag"), "name": row.get("name"),
        "w": row.get("width"), "h": row.get("height"), "fr": fr, "duration_s": max(op - ip, 0) / fr,
        "n_layers": count_layers(lj), "expressions": has_expression(lj), "raster": has_raster(lj),
        "n_tracks": 0, "n_negligible": 0, "n_corrupt": 0, "tracks": [], "fingerprint": None,
    }
    if "svg_code" in row:
        out["svg_len"] = len(row["svg_code"] or "")
        out["style"] = row.get("style")
    cand = []
    for j, tr in enumerate(iter_tracks(lj)):
        fam = tr.prop.split(".")[0]
        if fam not in props:
            continue
        out["n_tracks"] += 1
        if not np.all(np.isfinite(tr.y)) or np.abs(tr.y).max() > MAX_ABS:
            out["n_corrupt"] += 1
            continue
        if len(tr.y) < 4 or float(tr.y.max() - tr.y.min()) < NEGLIGIBLE[fam]:
            out["n_negligible"] += 1
            continue
        cand.append((j, tr, fam))
    cand.sort(key=lambda c: -(c[1].y.max() - c[1].y.min()) / NEGLIGIBLE[c[2]])   # largest visible motion first
    shapes = []
    for j, tr, fam in cand[:max_tracks]:
        t, y = tr.t, tr.y
        if len(y) > MAX_SAMPLES:
            idx = np.linspace(0, len(y) - 1, MAX_SAMPLES).round().astype(int)
            t, y = t[idx], y[idx]
        acf, lag = autocorr_peak(y)
        rec = {"j": j, "layer": tr.layer_path, "ind": tr.layer_index, "parent": tr.parent_index, "prop": tr.prop,
               "nkf": tr.n_keyframes, "n": int(len(y)), "dur": float(t[-1] - t[0]), "range": float(y.max() - y.min()),
               "loop": bool(is_loop(y)), "acf": acf, "periodic": bool(acf >= 0.8)}
        if fam == "rotation":
            shapes.append((tr.n_keyframes, round(rec["dur"], 2), curve_shape(t, y)))
        if do_fit:
            unc, cl = cached_fit(t, y, fr)
            best = cl if cl is not None and cl.r2 > unc.r2 else unc
            rec.update(unc=unc.as_dict(), cl=cl.as_dict() if cl else None, best="cl" if best is cl else "unc", r2=best.r2)
            if fam == "rotation":   # tokens only meaningful inside the template range (+-90 deg); out-of-range is flagged
                rec["q"] = encode_curve(best.as_dict(), DEFAULT)
                rec["in_range"] = bool(abs(best.r) <= DEFAULT.r_max and abs(best.a) <= DEFAULT.a_max)
        out["tracks"].append(rec)
    out["n_tracks_fitted"] = len(out["tracks"])
    out["n_tracks_dropped_cap"] = max(len(cand) - max_tracks, 0)
    out["fingerprint"] = fingerprint(shapes) if shapes else None
    return out


def worker(job: tuple) -> dict:
    clamp_skip, ds, shard, out_dir, limit, props, max_tracks, do_fit = job
    name = Path(shard).name.split("-of-")[0].replace("metadata-", "")
    dst = out_dir / f"shard-{name}.jsonl.zst"
    done = out_dir / f"shard-{name}.done.json"
    global CLAMP_SKIP_R2
    CLAMP_SKIP_R2 = clamp_skip
    if done.exists():
        return {"shard": name, "skipped": True}
    t0 = time.time()
    n = err = 0
    cctx = zstandard.ZstdCompressor(level=6, threads=1)
    tmp = dst.with_suffix(".tmp")
    with open(shard, "rb") as f, open(tmp, "wb") as fo, cctx.stream_writer(fo) as w:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 22):
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:   # truncated tail of a shard
                err += 1
                continue
            try:
                w.write(orjson.dumps(parse_row(row, props, max_tracks, do_fit)) + b"\n")
            except Exception as e:   # one bad file must not kill a shard
                err += 1
                w.write(orjson.dumps({"id": row.get("id"), "error": f"{type(e).__name__}: {e}"[:200]}) + b"\n")
            n += 1
            if limit and n >= limit:
                break
    tmp.rename(dst)
    stats = {"shard": name, "rows": n, "errors": err, "seconds": round(time.time() - t0, 1), **STATS}
    done.write_text(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=SRC, default="animation")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "lottie" if SCI else ROOT / "data" / "lottie")
    ap.add_argument("--shards", default="", help="comma list of shard numbers (default: all)")
    ap.add_argument("--limit-rows", type=int, default=0, help="rows per shard (0 = all); for benchmarks")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--props", default="rotation", help="rotation is what the template bones use; position,scale add ~2x fit time")
    ap.add_argument("--max-tracks", type=int, default=32, help="fit at most this many tracks per animation (largest motion first)")
    ap.add_argument("--no-fit", action="store_true", help="structure + periodicity only (fast)")
    ap.add_argument("--clamp-skip-r2", type=float, default=CLAMP_SKIP_R2)
    args = ap.parse_args()

    shards = sorted(glob.glob(str(DATASETS / SRC[args.dataset] / "snapshots" / "*" / "data" / "metadata-*.jsonl.zst")))
    if args.shards:
        want = {int(s) for s in args.shards.split(",")}
        shards = [s for s in shards if int(Path(s).name.split("-")[1]) in want]
    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.clamp_skip_r2, args.dataset, s, out_dir, args.limit_rows, set(args.props.split(",")), args.max_tracks, not args.no_fit) for s in shards]
    print(f"{len(jobs)} shards -> {out_dir}, {args.workers} workers", flush=True)
    t0 = time.time()
    rows = 0
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for st in pool.imap_unordered(worker, jobs):
            rows += st.get("rows", 0)
            print(f"[{time.time() - t0:7.0f}s] {st}  total rows {rows}", flush=True)


if __name__ == "__main__":
    main()
