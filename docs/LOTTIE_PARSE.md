# Lottie corpus parsing (text -> SVG/Lottie animation branch)

Pipeline code copied from ../Loomie so that Marionette does not import from it:
`lottie/{lottie_curves,primitive,quantizer}.py` (unchanged copies), `lottie/curve_stats.py` (loop / periodicity / dedupe
fingerprint from `fit_coverage.py` + `coverage_report.py`), driver `scripts/lottie2d/parse_lottie.py`.

## Environment (outside this drive)

```
python3 -m venv ~/envs/marionette && ~/envs/marionette/bin/pip install -r requirements.txt     # 560 MB, created 2026-10-24
```

## Run

```
~/envs/marionette/bin/python scripts/parse_lottie.py --dataset animation --workers 18      # LottieAnimation-660K, 68 shards
~/envs/marionette/bin/python scripts/parse_lottie.py --dataset svg --no-fit --workers 18   # LottieSVG-10M, 100 shards (static: no tracks)
```

Output `data/lottie/<dataset>/shard-NNNNN.jsonl.zst` (+ `.done.json`; a rerun skips finished shards, so Ctrl-C is safe).
One row per animation: `id, caption, tag, name, w, h, fr, duration_s, n_layers, expressions, raster, n_tracks, n_negligible,
n_corrupt, fingerprint, tracks[]`. Each track: layer path, parent, prop, keyframes, range, `loop`, `acf`/`periodic`,
`unc`/`cl` = sine fits `{r,a,s,phi,clamp,r2}`, `best`, `r2`, and for rotation `q` (value-token indices, Loomie quantizer)
and `in_range` (inside the +-90 deg token range). Options: `--props rotation,position,scale`, `--max-tracks 32`,
`--no-fit`, `--clamp-skip-r2 0.98` (`>1` = always run the clamped search), `--limit-rows N` (benchmark).

Defaults that differ from Loomie's `fit_coverage.py` (all for speed): rotation only (what template bones use), the 32 largest
tracks per animation, the clamped (one-sided) search skipped when the plain sine already has R^2 >= 0.98, and a per-worker
cache so identical curves (consecutive-id re-exports, mirrored limbs) are fitted once. Without these a dense region of
shard 1 cost ~3.8 s/row instead of ~0.8 s.

## Measurements (i9-10850K, 10 cores / 20 threads; 2x RTX 3090 are not used by this stage)

| | rows | per row | 
|---|---|---|
| read + parse JSON only (`--no-fit`) | 1000 | 14 ms |
| full fit, 18 shards x first 150 rows, 18 workers | 2,700 | 0.10 s CPU (0.013-0.36 per shard); 55 s wall |
| LottieSVG-10M, `--no-fit` | 500 | 1 ms (every row static: 0 tracks) |

## Estimates

| job | CPU | wall (about 12 effective cores) | output |
|---|---|---|---|
| LottieAnimation-660K, fit | 672k x 0.10-0.20 s = 19-37 CPU-h | **about 1.6-3 h** | about 0.3 GB |
| LottieSVG-10M, `--no-fit` | 9.78M x 1 ms = 2.7 CPU-h | about 15-30 min (reads 35 GB of metadata, exFAT limited) | about 2.5 GB |

The MP4 (129 GB) and PNG (57 GB) tars are not read. Disk: 593 GB free here, 114 GB free on `/` (the env uses 0.6 GB). RAM: 62 GB, a worker holds one
shard row at a time (< 1 GB each).

The sample is the first rows of 18 shards, so it is not random; treat the estimate as +-50%. Not done: the VLM "living character" filter
for the 480k rows of LottieAnimation-660K that I2V never scanned (needs the MP4s, Qwen3-VL-2B on the two 3090s; I2V scanned 190k of 672k rows
at that time). `data/lottie` rows can be joined to `../Datasets/LottieAnimation-small/manifest_v2.jsonl` by `id` for the already-classified 39.6k.
