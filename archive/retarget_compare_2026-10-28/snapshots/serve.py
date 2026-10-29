#!/usr/bin/env python3
"""Serve the character viewer.

  python3 viewer/serve.py [--port 8765] [--no-open]   opens http://127.0.0.1:8765/viewer/home.html (the menu: every page with a manual); the character browser is http://127.0.0.1:8765/viewer/

Serves the project folder (localhost only), /api/retargets (for viewer/compare.html) and /api/characters, which lists the
normalized characters from data/normalize_report.csv, data/manifest.csv and each
normalization.json.
GET /api/targets (characters that can be retarget targets) and GET /api/retarget (one clip) stay for the embedded player of compare.html. Motion generation (the Generate page, /api/generate) was removed on 2026-10-28; see archive/motion_generate_2026-10-28/.
Dataset viewers (one HTML page each, all served from here): viewer/vlm_table.html reads GET /api/vlm_table (the Qwen label table in data/vlm/animation, filtered and paged, loaded in the background on first
use) and plays rows with /api/lottie_item?raw=1; viewer/lottie_raw.html reads GET /api/lottie_raw (a window of rows of one metadata shard of ../Datasets/LottieAnimation-660K or LottieSVG-10M) and
GET /api/lottie_raw_row (one row: Lottie JSON and, for SVG, the SVG source). They replace the Qt viewer of the GPU server (SciData/Marionette/viewer/viewer.py), which stays there.
Skeleton viewer (viewer/skeletons.html): GET /api/skel_sources, /api/skel_list?group=real|synthetic&source=&cls=&q=&n=12&seed=&offset= (a filtered or shuffled page of skeletons, compact) and /api/skel_item?key= (one skeleton
with its derived class, carriers and movement-mode support); reads data/analysis/skeleton_classes and data/synthetic/skeletons, written by scripts/rigging/skeleton_classes.py and skeleton_synth.py.
Human rating (viewer/rate.html, protocol in eval/human_eval.md): GET /api/rate_sets (rating sets under data/eval), GET /api/rate_set?set=<test>/<run>/rate&rater=<name> (the items in their fixed
random order, hidden fields removed) and POST /api/rate_save (validated, appended to <set>/human.jsonl; nothing is ever rewritten). Only data/eval is readable or writable through these.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_counts = {}


def clip_count(adir):
    """Converted clips for a character (Mixamo characters share data/clips/mixamo/clips.json)."""
    f = adir / "anims.json"
    if not f.exists():
        return 0
    key = (str(f), f.stat().st_mtime)
    if key not in _counts:
        d = json.loads(f.read_text())
        if "_shared" in d:
            idx = ROOT / "data" / "clips" / "mixamo" / "clips.json"
            _counts[key] = (len(json.loads(idx.read_text())) - 1 if idx.exists() else 0) + sum(k.startswith("rt__") for k in d)
        else:
            _counts[key] = len(d)
    return _counts[key]


def listing():
    manifest = {}
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            manifest[(r["source"], r["id"])] = r
    out = []
    with open(ROOT / "data" / "normalize_report.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            key = f"{r['source']}_{r['id']}"
            model = ROOT / "data" / "characters" / key / "model.glb"
            if not model.exists():
                continue
            m = manifest.get((r["source"], r["id"]), {})
            norm = json.loads((model.parent / "normalization.json").read_text())
            out.append(dict(
                key=key, id=r["id"], source=r["source"], cls=r["cls"], category=m.get("category", ""),
                tris=int(m.get("tris") or 0), joints=int(m.get("joints") or 0), n_clips=int(m.get("n_clips") or 0),
                height=norm["height_m"], extent=norm["extent_m"], scale=norm["scale"], scale_mode=norm["scale_mode"],
                yaw=norm["yaw_deg"], flags=r["flags"].split(), mb=round(model.stat().st_size / 1e6, 2),
                clips=clip_count(model.parent / "anims"),
                url=f"/data/characters/{key}/model.glb"))
    return out


_src_out = {}
_rt_cache = {}


def src_out_map():
    """clip -> converted source file, from data/clips_report.csv (read once)"""
    if not _src_out:
        with open(ROOT / "data" / "clips_report.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    _src_out[r["clip"]] = r["out"]
    return _src_out


def mixamo_list():
    chars = ROOT / "data" / "characters"
    return sorted(p.name for p in chars.glob("mixamo_*") if not p.name.startswith("._") and (p / "model.glb").exists())


def retarget_item(glb):
    """What the compare viewer needs for one retargeted clip GLB (data/characters/<target>/anims/rt__*.glb): the target, the source clip's GLB and the character it is shown on (Mixamo clips are
    shared, so they play on a Mixamo character with that character's root scale). None if the clip has no anims.json entry."""
    anymate = glb.parent.parent.name == "rig"                                  # data/anymate/<key>/rig/anims/rt__*.glb: rigs predicted by Anymate (scripts/rigging/anymate_rig.py)
    key = ("anymate_" + glb.parent.parent.parent.name) if anymate else glb.parent.parent.name
    meta = json.loads((glb.parent / "anims.json").read_text()).get(glb.stem)
    if not meta:
        return None
    base = f"/data/anymate/{key[len('anymate_'):]}/rig" if anymate else f"/data/characters/{key}"
    so = src_out_map().get(meta["source_clip"], "") or meta.get("source_file", "")      # decoded latents carry their own source clip file
    shared = so.startswith(("data/clips/mixamo/", "data/decoded/"))
    return dict(
        target=key, name=glb.stem, source_clip=meta["source_clip"], target_cls=meta["retargeted"].get("target_class", ""),
        target_url=f"{base}/rig.glb" if anymate else f"{base}/model.glb", clip_url=f"{base}/anims/{glb.name}",
        source_clip_url="/" + so if so else "", source_shared=shared,
        source_model_url="" if shared or not so else "/" + str(Path(so).parent.parent / "model.glb"),
        fps=meta["fps"], frames=meta["frames"], duration_s=meta["duration_s"], captions=meta.get("motion_captions", []),
        metrics=meta["retargeted"])


def retargets(refresh=False):
    """Every retargeted clip. Scanning the 3,570 character folders took 13-14 s (external drive) and the generator page triggered it on every result, so the list is cached in memory
    (until ?refresh=1 or 10 minutes); retarget_one gives a single clip."""
    import time
    if not refresh and _rt_cache.get("t", 0) > time.time() - 600 and "data" in _rt_cache:
        return _rt_cache["data"]
    out = []
    for glb in sorted((ROOT / "data" / "characters").glob("*/anims/rt__*.glb")) + sorted((ROOT / "data" / "anymate").glob("*/rig/anims/rt__*.glb")):
        if glb.name.startswith("._"):
            continue
        try:
            item = retarget_item(glb)
        except (ValueError, KeyError, OSError) as e:   # a truncated anims.json (JSONDecodeError is a ValueError) must not empty the whole list
            print(f"skipped {glb}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        if item:
            out.append(item)
    _rt_cache.update(t=time.time(), data=dict(items=out, mixamo=mixamo_list()))
    return _rt_cache["data"]


def retarget_one(target, name):
    """the same structure as retargets() for one clip (a few milliseconds)"""
    if "/" in target or "/" in name or ".." in target + name:
        raise ValueError("bad name")
    if target.startswith("anymate_"):
        glb = ROOT / "data" / "anymate" / target[len("anymate_"):] / "rig" / "anims" / f"{name}.glb"
    else:
        glb = ROOT / "data" / "characters" / target / "anims" / f"{name}.glb"
    item = retarget_item(glb) if glb.exists() else None
    return dict(items=[item] if item else [], mixamo=mixamo_list())


def targets():
    """Characters that can be retarget targets (data/retarget_targets.csv, status ok), Mixamo first, then quadrupeds, then the other bipeds."""
    out = []
    with open(ROOT / "data" / "retarget_targets.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and (ROOT / "data" / "characters" / r["key"] / "model.glb").exists():
                out.append(dict(key=r["key"], source=r["source"], cls=r["cls"]))
    rank = lambda t: (0 if t["source"] == "mixamo" else 1 if t["cls"] == "quadruped" else 2, t["key"])
    return sorted(out, key=rank)


_lottie_cache: dict = {}


def lottie_item(item_id: str, shard: int, raw: bool = False) -> dict:
    """One Lottie animation of the corpus as the token viewer shows it: pulled out of its metadata shard with the zstd command (the shard is a stream, so the
    cost is reading up to that row: 1-3 s), then joint pivots and tokens are computed (scripts/lottie2d/lottie_tokens.py). The last 12 are kept in memory."""
    import glob
    import re
    import shutil
    import subprocess
    if not re.fullmatch(r"\d{1,12}", item_id) or not 0 <= shard < 68:
        raise ValueError("bad id or shard")
    key = (item_id, shard, raw)
    if key in _lottie_cache:
        return _lottie_cache[key]
    zstd = shutil.which("zstd") or str(Path(sys.executable).parent / "zstd")
    path = glob.glob(str(ROOT.parent / "Datasets" / "LottieAnimation-660K" / "snapshots" / "*" / "data" / f"metadata-{shard:05d}-of-00068.jsonl.zst"))
    if not path:
        raise FileNotFoundError("corpus shard not found")
    prefix = b'{"id":"' + item_id.encode() + b'"'
    line = None
    proc = subprocess.Popen([zstd, "-dc", os.path.realpath(path[0])], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
    try:
        for ln in proc.stdout:
            if ln.startswith(prefix):
                line = ln
                break
    finally:
        proc.kill()
        proc.wait()
    if line is None:
        raise KeyError("animation not found in that shard")
    row = json.loads(line)
    lj = row["lottie_json"]
    lj = json.loads(lj) if isinstance(lj, str) else lj
    if raw:                                                                   # just the animation, for players that do not need joints or tokens
        rec = dict(id=item_id, lottie=lj, caption=row.get("caption", ""))
        _lottie_cache[key] = rec
        return rec
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    import lottie_tokens as LT
    rec, status = LT.make_record(item_id, lj)
    if rec is None:
        raise KeyError("no usable layers in this animation")
    rec["caption"] = row.get("caption", "")
    if len(_lottie_cache) >= 12:
        _lottie_cache.pop(next(iter(_lottie_cache)))
    _lottie_cache[key] = rec
    return rec


_searchers: dict = {}
_search_lock = threading.Lock()


def lottie_search(q: str, k: int, rewrite: bool, tag: str) -> dict:
    """Text -> nearest Lottie animations and 3D clips with a trained joint model (scripts/lottie2d/lottie_search.py; index built by `lottie_search.py build`)."""
    import re
    if not q.strip() or len(q) > 200:
        raise ValueError("the prompt must be 1-200 characters")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,30}", tag):
        raise ValueError("bad index name")
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    import lottie_search as LS
    if not LS.index_path(tag).exists():
        raise FileNotFoundError(f"no search index for model '{tag}': run  python scripts/lottie2d/lottie_search.py build --tag {tag}")
    with _search_lock:                                                       # one CLIP text encoder, one query at a time
        if tag not in _searchers:
            _searchers[tag] = LS.Searcher(tag)
        return _searchers[tag].search(q, max(1, min(k, 48)), rewrite)


def lottie_search_tags() -> list:
    return sorted(p.stem[len("lottie_search_"):] for p in (ROOT / "data" / "embeddings").glob("lottie_search_*.npz") if not p.name.startswith("._"))


CORPUS = {"animation": ("LottieAnimation-660K", 68), "svg": ("LottieSVG-10M", 100)}   # dataset -> (folder in ../Datasets, number of metadata shards)


def corpus_shard(dataset: str, shard: int) -> str:
    """Real path of one metadata shard (zstd ignores the HF cache symlinks)."""
    import glob
    if dataset not in CORPUS or not 0 <= shard < CORPUS[dataset][1]:
        raise ValueError("bad dataset or shard")
    name, n = CORPUS[dataset]
    path = glob.glob(str(ROOT.parent / "Datasets" / name / "snapshots" / "*" / "data" / f"metadata-{shard:05d}-of-{n:05d}.jsonl.zst"))
    if not path:
        raise FileNotFoundError("corpus shard not found")
    return os.path.realpath(path[0])


def zstd_lines(path: str):
    """The lines of a .jsonl.zst file, streamed through the zstd command; stops the process when the caller stops reading."""
    import shutil
    import subprocess
    zstd = shutil.which("zstd") or str(Path(sys.executable).parent / "zstd")
    proc = subprocess.Popen([zstd, "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
    try:
        yield from proc.stdout
    finally:
        proc.kill()
        proc.wait()


def lottie_raw_list(dataset: str, shard: int, start: int, count: int) -> dict:
    """viewer/lottie_raw.html: a window of rows straight from one dataset metadata shard (no Qwen table needed). The shard is a stream, so rows before `start` are read and dropped."""
    path = corpus_shard(dataset, shard)
    start, count = max(0, start), max(1, min(count, 1000))
    rows = []
    for k, ln in enumerate(zstd_lines(path)):
        if k < start:
            continue
        if k >= start + count:
            break
        r = json.loads(ln)
        rows.append(dict(row=k, id=r["id"], name=r.get("name") or r.get("style") or "", caption=r.get("caption") or "", tag=r.get("tag") or []))
    return dict(dataset=dataset, shard=shard, start=start, shards=CORPUS[dataset][1], rows=rows)


_raw_cache: dict = {}


def lottie_raw_row(dataset: str, shard: int, item_id: str) -> dict:
    """One row of the raw browser: the Lottie JSON and, for the SVG dataset, the SVG source. The last 12 are kept."""
    import re
    if not re.fullmatch(r"\d{1,12}", item_id):
        raise ValueError("bad id")
    key = (dataset, shard, item_id)
    if key in _raw_cache:
        return _raw_cache[key]
    prefix = b'{"id":"' + item_id.encode() + b'"'
    line = next((ln for ln in zstd_lines(corpus_shard(dataset, shard)) if ln.startswith(prefix)), None)
    if line is None:
        raise KeyError("row not found in that shard")
    r = json.loads(line)
    lj = r["lottie_json"]
    rec = dict(id=item_id, lottie=json.loads(lj) if isinstance(lj, str) else lj, svg=r.get("svg_code"), caption=r.get("caption") or "", tag=r.get("tag") or [],
               name=r.get("name") or r.get("style") or "", width=r.get("width"), height=r.get("height"))
    if len(_raw_cache) >= 12:
        _raw_cache.pop(next(iter(_raw_cache)))
    _raw_cache[key] = rec
    return rec


_vlm = dict(rows=None, done=0, total=0, n_all=0, error=None, started=False)
_vlm_lock = threading.Lock()


def _vlm_load() -> None:
    """Background loader of the Qwen label table: every row Qwen looked at, from data/vlm/animation/shard-*.jsonl.zst (the 2B pre-filter's 'no keyword' rows carry no label and are only counted)."""
    try:
        import orjson
        loads = orjson.loads
    except ImportError:
        loads = json.loads
    try:
        files = sorted(p for p in (ROOT / "data" / "vlm" / "animation").glob("shard-*.jsonl.zst") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError("no label shards in data/vlm/animation (copy them from the server: see README)")
        _vlm["total"] = len(files)
        rows, n_all = [], 0
        keys = ("id", "shard", "processed", "relevant", "subject", "body_plan", "full_body", "is_ui_icon", "caption_qwen", "caption_lottie", "tag", "relevant_2b", "qwen_model", "skip_reason")
        for k, f in enumerate(files):
            for ln in zstd_lines(os.path.realpath(f)):
                d = loads(ln)
                n_all += 1
                if d.get("processed"):
                    rows.append({c: d.get(c) for c in keys})
            _vlm["done"], _vlm["n_all"] = k + 1, n_all
        _vlm["rows"] = rows
    except Exception as e:
        _vlm["error"] = f"{type(e).__name__}: {e}"


def vlm_table(relevant: str, plan: str, text: str, sort: str, desc: bool, offset: int, limit: int) -> dict:
    """viewer/vlm_table.html: filtered, sorted, paged rows of the Qwen label table."""
    with _vlm_lock:
        if not _vlm["started"]:
            _vlm["started"] = True
            threading.Thread(target=_vlm_load, daemon=True).start()
    if _vlm["error"]:
        raise FileNotFoundError(_vlm["error"])
    if _vlm["rows"] is None:
        return dict(loading=True, done=_vlm["done"], total=_vlm["total"])
    rows = _vlm["rows"]
    words = text.lower().split()

    def keep(d):
        if relevant != "all" and bool(d["relevant"]) != (relevant == "true"):
            return False
        if plan != "all" and (d["body_plan"] != plan if plan != "other" else d["body_plan"] in ("2leg", "4leg")):
            return False
        if words:
            blob = f"{d['id']} {d['caption_qwen'] or ''} {d['caption_lottie'] or ''} {' '.join(map(str, d['tag'] or []))}".lower()
            return all(w in blob for w in words)
        return True

    sel = [d for d in rows if keep(d)]
    if sort in ("id", "subject", "body_plan", "relevant", "full_body", "is_ui_icon"):
        sel.sort(key=(lambda d: int(d["id"])) if sort == "id" else (lambda d: (d[sort] is None, str(d[sort]))), reverse=desc)
    return dict(loading=False, loaded=len(rows), n_all=_vlm["n_all"], n_relevant=sum(1 for d in rows if d["relevant"]), matched=len(sel), offset=offset, rows=sel[offset:offset + limit])


# ---- human rating page (viewer/rate.html): a "set" is a folder data/eval/<test>/<run>/rate/ with set.json (axes, anchors), items.jsonl and human.jsonl (written here). Format: eval/human_eval.md ----
EVAL = ROOT / "data" / "eval"
RATER_RE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
_rate_lock = threading.Lock()


def rate_dir(rel: str) -> Path:
    """The set folder data/eval/<rel>; refuses anything that resolves outside data/eval."""
    if not rel or "\x00" in rel:
        raise ValueError("set path missing")
    base, d = EVAL.resolve(), (EVAL / rel).resolve()
    if base not in d.parents:
        raise ValueError("a set must be a folder inside data/eval")
    if not (d / "set.json").is_file():
        raise FileNotFoundError(f"no set.json in data/eval/{rel}")
    return d


def rate_rows(d: Path) -> list:
    """Every rating written so far. A truncated last line (a write that did not finish) is skipped, not fatal."""
    f, rows = d / "human.jsonl", []
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def rate_load(d: Path):
    meta = json.loads((d / "set.json").read_text(encoding="utf-8"))
    items = [json.loads(line) for line in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("items.jsonl has duplicate ids")
    if meta.get("mode", "scores") not in ("scores", "pairwise"):
        raise ValueError("set.json mode must be scores or pairwise")
    return meta, items


def rate_url(d: Path, m: dict) -> dict:
    """A media entry with a URL the page can load; the file must lie inside data/eval (a path inside the set folder, or one starting with /data/eval/)."""
    src = str(m["src"])
    p = (ROOT / src.lstrip("/") if src.startswith("/") else d / src).resolve()
    if EVAL.resolve() not in p.parents:
        raise ValueError(f"media outside data/eval: {src}")
    if m.get("type") not in ("image", "video"):
        raise ValueError("media type must be image or video")
    if not p.is_file():                                                       # fail when the set is loaded, not with a broken picture in front of a rater
        raise FileNotFoundError(f"media file missing: {src}")
    return dict(type=m["type"], label=str(m.get("label", "")), url="/" + p.relative_to(ROOT).as_posix())


def rate_sets() -> list:
    out = []
    for f in sorted(EVAL.glob("*/set.json")) + sorted(EVAL.glob("*/*/set.json")) + sorted(EVAL.glob("*/*/*/set.json")) if EVAL.exists() else []:
        if not f.name.startswith("._"):
            out.append(f.parent.relative_to(EVAL).as_posix())
    return out


def rate_set(rel: str, rater: str) -> dict:
    """The set for one rater: items in the set's fixed random order (seed in set.json), the ones for other raters dropped, and only what the rater may see
    (the prompt and the media). Hidden fields (meta: which system, control or baseline; the left/right mapping) never leave the server."""
    if not RATER_RE.match(rater or ""):
        raise ValueError("rater must be 1-24 letters, digits, - or _")
    d = rate_dir(rel)
    meta, items = rate_load(d)
    order = list(range(len(items)))
    if all("order" in it for it in items):
        order.sort(key=lambda i: items[i]["order"])
    else:
        random.Random(int(meta.get("seed", 0))).shuffle(order)
    rows = [r for r in rate_rows(d) if r.get("rater") == rater]
    revisions = {}
    for r in rows:
        revisions[r["id"]] = revisions.get(r["id"], 0) + 1
    out = []
    for i in order:
        it = items[i]
        if it.get("raters") and rater not in it["raters"]:
            continue
        e = dict(id=it["id"], prompt=it.get("prompt", ""), skip_axes=it.get("skip_axes", []), done=revisions.get(it["id"], 0))
        if meta.get("mode", "scores") == "pairwise":
            e["left"], e["right"] = [rate_url(d, m) for m in it["left"]], [rate_url(d, m) for m in it["right"]]
        else:
            e["media"] = [rate_url(d, m) for m in it["media"]]
        out.append(e)
    anchors = [dict(a, media=[rate_url(d, m) for m in a.get("media", [])]) for a in meta.get("anchor_items", [])]
    shown = {k: meta[k] for k in ("title", "test", "mode", "seed", "axes", "instructions") if k in meta}
    return dict(set=dict(shown, anchor_items=anchors), items=out, n_done=sum(1 for e in out if e["done"]))


def rate_save(req: dict) -> dict:
    """Validate one rating and append it to <set>/human.jsonl. Rows are never edited: a changed mind is a new row with a higher revision."""
    rater, item_id = str(req.get("rater", "")), str(req.get("id", ""))
    if not RATER_RE.match(rater):
        raise ValueError("rater must be 1-24 letters, digits, - or _")
    d = rate_dir(str(req.get("set", "")))
    meta, items = rate_load(d)
    item = next((it for it in items if it["id"] == item_id), None)
    if item is None:
        raise ValueError("unknown item id")
    if item.get("raters") and rater not in item["raters"]:
        raise ValueError("this item is not assigned to this rater")
    seconds, session = req.get("seconds"), req.get("session")
    if seconds is not None and not (isinstance(seconds, (int, float)) and 0 <= seconds < 36000):
        raise ValueError("seconds must be a number from 0 to 36000")
    if session is not None and not (isinstance(session, int) and 0 <= session < 10000):
        raise ValueError("session must be an integer")
    row = dict(id=item_id, rater=rater, session=session, seed=meta.get("seed"), cannot_judge=bool(req.get("cannot_judge")))
    if item.get("rig"):
        row["rig"] = item["rig"]
    if meta.get("mode", "scores") == "pairwise":
        row["choice"] = None if row["cannot_judge"] else req.get("choice")
        if not row["cannot_judge"] and row["choice"] not in ("left", "right", "tie"):
            raise ValueError("choice must be left, right or tie")
    else:
        axes = {a["key"]: a for a in meta["axes"]}
        given, skip, scores = req.get("scores") or {}, set(item.get("skip_axes", [])), {}
        if set(given) - set(axes):
            raise ValueError("unknown axis in scores")
        for key, a in axes.items():
            v = given.get(key)
            if row["cannot_judge"] or key in skip:
                scores[key] = None
            elif v is None and a.get("nullable"):
                scores[key] = None
            elif isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5:
                scores[key] = v
            else:
                raise ValueError(f"axis {key}: a score from 1 to 5 is required")
        row["scores"] = scores
    comment = str(req.get("comment", ""))[:300].strip()
    if comment:
        row["comment"] = comment
    if seconds is not None:
        row["seconds"] = round(float(seconds), 1)
    with _rate_lock:                                                        # one writer at a time; append only
        row["revision"] = sum(1 for r in rate_rows(d) if r.get("rater") == rater and r.get("id") == item_id)
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(d / "human.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return dict(ok=True, revision=row["revision"])


# ---- skeleton viewer (viewer/skeletons.html): the real skeletons (data/analysis/skeleton_classes) next to the synthetic ones (data/synthetic/skeletons) ----
SKEL_DIR = ROOT / "data" / "analysis" / "skeleton_classes"
SYN_FILE = ROOT / "data" / "synthetic" / "skeletons" / "skeletons.jsonl"
SKEL_CLASSES = ["biped", "quadruped", "limbless", "winged", "other"]
SKEL_SOURCES = {"manifest": "3D manifest (Objaverse, Mixamo)", "truebones": "AnyTop creatures (Truebones)", "anymate_pred": "Anymate predicted", "anymate_gt": "Anymate original", "lottie": "Lottie rigs"}
_skel = {}


def skel_load():
    if "real" in _skel:
        return
    if not (SKEL_DIR / "skeletons.json").exists():
        raise FileNotFoundError("no real skeletons yet: python scripts/rigging/skeleton_classes.py build")
    items = json.loads((SKEL_DIR / "skeletons.json").read_text())
    if (SKEL_DIR / "lottie_skeletons.json").exists():
        for r in json.loads((SKEL_DIR / "lottie_skeletons.json").read_text()):
            r["label"] = {"2leg": "biped", "4leg": "quadruped"}.get(r["label"], "other")      # Qwen's body plan, in the class names
            items.append(r)
    for r in items:
        r["key"] = f"{r['source']}:{r['id']}"
    syn = [json.loads(line) for line in SYN_FILE.read_text().splitlines() if line.strip()] if SYN_FILE.exists() else []
    for r in syn:                                                          # the same fields as the real ones, so the code below treats them alike
        r.update(key=f"synthetic:{r['id']}", source="synthetic", label=r["declared_class"], detail=r["meta"]["variant"])
    _skel.update(real=items, syn=syn, by_key={r["key"]: r for r in items + syn})


def skel_sources() -> dict:
    skel_load()
    count = lambda rows: {c: sum(1 for r in rows if r["label"] == c) for c in SKEL_CLASSES}
    real = [dict(id=k, name=v, n=sum(1 for r in _skel["real"] if r["source"] == k), classes=count([r for r in _skel["real"] if r["source"] == k])) for k, v in SKEL_SOURCES.items()]
    variants = sorted({(r["detail"], r["label"]) for r in _skel["syn"]})
    return dict(real=real, synthetic=dict(n=len(_skel["syn"]), classes=count(_skel["syn"]), variants=[dict(id=v, cls=c) for v, c in variants]), classes=SKEL_CLASSES)


def skel_view(r) -> dict:
    """one skeleton in the compact form the page draws: parents (index, -1 = root), P (3D [x, y, z], or 2D for Lottie), and for the synthetic ones the part of every joint"""
    if r["source"] == "synthetic":
        sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
        import skeleton_synth as sy
        body = sy.body3d(r["meta"]["variant"], r["meta"]["body_seed"])
        idx, pid = {j["id"]: i for i, j in enumerate(body.joints)}, {p["id"]: i for i, p in enumerate(body.parts)}
        return dict(key=r["key"], source="synthetic", label=r["label"], detail=r["detail"], nj=len(body.joints),
                    parents=[idx[j["parent"]] if j["parent"] else -1 for j in body.joints], P=[[round(float(v), 3) for v in j["P"]] for j in body.joints],
                    pj=[pid[j["part"]] for j in body.joints], parts=[dict(id=p["id"], kind=p["kind"], side=p.get("side")) for p in body.parts])
    return dict(key=r["key"], source=r["source"], label=r["label"], detail=r["detail"], nj=len(r["parents"]), parents=r["parents"], P=[[round(float(v), 3) for v in p] for p in r["P"]])


def skel_list(group: str, source: str, cls: str, q: str, n: int, seed, offset: int) -> dict:
    """source: a data source (real) or a variant (synthetic); cls: a class; q: words in the id or detail; seed: a fixed shuffle (None = file order)"""
    skel_load()
    pool = _skel["syn"] if group == "synthetic" else _skel["real"]
    pool = [r for r in pool if (not source or source == "all" or source in (r["source"], r["detail"]))
            and (not cls or cls == "all" or r["label"] == cls) and (not q or q.lower() in f"{r['id']} {r['detail']}".lower())]
    if seed is not None:
        pool = random.Random(seed).sample(pool, len(pool))
    return dict(total=len(pool), items=[skel_view(r) for r in pool[offset:offset + n]])


def skel_info(spec) -> dict:
    sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
    import skeleton_spec as ss
    errs, warns = ss.validate(spec)
    if errs:
        return dict(errors=errs)
    cls, why = ss.derive_class(spec)
    return dict(derived=cls, reason=why, carriers=ss.carriers(spec), modes={m: list(v) for m, v in ss.mode_support(spec).items()}, warnings=warns)


def skel_item(key: str) -> dict:
    skel_load()
    r = _skel["by_key"].get(key)
    if r is None:
        raise KeyError("unknown skeleton")
    if r["source"] == "synthetic":
        return dict(skel_view(r), info=skel_info(r), spec_view=True)
    if r["source"] == "anymate_pred" and (ROOT / "data" / "anymate" / r["id"].split("anymate_pred_", 1)[1] / "roles.json").exists():
        import numpy as np
        sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
        import skeleton_spec as ss
        d = ROOT / "data" / "anymate" / r["id"].split("anymate_pred_", 1)[1]
        roles = json.loads((d / "roles.json").read_text())
        P3 = np.load(d / "joints.npy")
        spec = ss.from_roles(P3, np.load(d / "parents.npy"), roles["roles"], roles["cls"], d.name)    # the real rig with the role labels of anymate_roles.py
        idx, pid = {j["id"]: i for i, j in enumerate(spec["joints"])}, {p["id"]: i for i, p in enumerate(spec["parts"])}
        return dict(key=r["key"], source=r["source"], label=r["label"], detail=r["detail"], nj=len(spec["joints"]), parents=[idx[j["parent"]] if j["parent"] else -1 for j in spec["joints"]],
                    P=[[round(float(v), 3) for v in P3[int(j["id"][1:])]] for j in spec["joints"]], pj=[pid[j["part"]] for j in spec["joints"]],
                    parts=[dict(id=p["id"], kind=p["kind"], side=p.get("side")) for p in spec["parts"]], info=skel_info(spec), spec_view=True)
    return dict(skel_view(r), info=None, spec_view=False)


_target_keys = set()


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".md": "text/plain; charset=utf-8"}   # the docs open in the browser instead of downloading

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/":                                                           # the menu page
            self.send_response(302)
            self.send_header("Location", "/viewer/home.html")
            self.end_headers()
            return
        if route in ("/api/vlm_table", "/api/lottie_raw", "/api/lottie_raw_row"):   # the two dataset viewers (viewer/vlm_table.html, viewer/lottie_raw.html)
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if route == "/api/vlm_table":                                     # ?relevant=all|true|false&plan=all|2leg|4leg|other&q=<words>&sort=<col>&desc=1&offset=0&limit=100
                    return self.send_json(vlm_table(q.get("relevant", "all"), q.get("plan", "all"), q.get("q", "")[:200], q.get("sort", ""), q.get("desc") == "1",
                                                    max(0, int(q.get("offset", 0))), max(1, min(int(q.get("limit", 100)), 500))))
                if route == "/api/lottie_raw":                                    # ?dataset=animation|svg&shard=0&start=0&count=300
                    return self.send_json(lottie_raw_list(q.get("dataset", "animation"), int(q.get("shard", 0)), int(q.get("start", 0)), int(q.get("count", 300))))
                return self.send_json(lottie_raw_row(q.get("dataset", "animation"), int(q.get("shard", -1)), q.get("id", "")))     # ?dataset=&shard=&id=
            except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=str(e)), 404 if isinstance(e, (KeyError, FileNotFoundError)) else 400)
        if route in ("/api/skel_sources", "/api/skel_list", "/api/skel_item"):          # the skeleton viewer (viewer/skeletons.html)
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if route == "/api/skel_sources":
                    return self.send_json(skel_sources())
                if route == "/api/skel_item":
                    return self.send_json(skel_item(q.get("key", "")))
                return self.send_json(skel_list(q.get("group", "real"), q.get("source", ""), q.get("cls", ""), q.get("q", "")[:80], max(1, min(int(q.get("n", 12)), 100)),
                                                int(q["seed"]) if q.get("seed", "") != "" else None, max(0, int(q.get("offset", 0)))))
            except ImportError as e:
                return self.send_json(dict(error=f"needs the conda env Mario ({e}); restart: conda activate Mario && python viewer/serve.py"), 500)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=f"{type(e).__name__}: {e}"), 400)
            except FileNotFoundError as e:
                return self.send_json(dict(error=str(e)), 404)
        if route in ("/api/rate_sets", "/api/rate_set"):                           # the human rating page (viewer/rate.html)
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                return self.send_json(rate_sets() if route == "/api/rate_sets" else rate_set(q.get("set", ""), q.get("rater", "")))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=f"{type(e).__name__}: {e}"), 400)
            except FileNotFoundError as e:
                return self.send_json(dict(error=str(e)), 404)
        if route == "/api/lottie_search":                                         # /api/lottie_search?q=<prompt>&k=12&rewrite=1&tag=mixed   (tags=1 lists the available indexes)
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            if q.get("tags"):
                return self.send_json(lottie_search_tags())
            try:
                return self.send_json(lottie_search(q.get("q", [""])[0], int(q.get("k", ["12"])[0]), q.get("rewrite", ["1"])[0] == "1", q.get("tag", ["mixed"])[0]))
            except (ValueError, FileNotFoundError) as e:
                return self.send_json(dict(error=str(e)), 400 if isinstance(e, ValueError) else 404)
        if route == "/api/lottie_item":                                           # one corpus animation with its joints and tokens: /api/lottie_item?id=<id>&shard=<n>
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            try:
                return self.send_json(lottie_item(q.get("id", [""])[0], int(q.get("shard", ["-1"])[0]), q.get("raw", ["0"])[0] == "1"))
            except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=str(e)), 404 if isinstance(e, (KeyError, FileNotFoundError)) else 400)
        if route == "/api/retarget":                                              # one clip: /api/retarget?target=<character>&name=<clip>
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            try:
                return self.send_json(retarget_one(q.get("target", [""])[0], q.get("name", [""])[0]))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=str(e)), 400)
        if route in ("/api/characters", "/api/retargets", "/api/targets"):
            if route == "/api/retargets":
                return self.send_json(retargets(refresh="refresh" in self.path))
            self.send_json({"/api/characters": listing, "/api/targets": targets}[route]())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] == "/api/rate_save":                              # one rating from viewer/rate.html, appended to data/eval/<set>/human.jsonl
            try:
                return self.send_json(rate_save(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=f"{type(e).__name__}: {e}"), 400)
            except FileNotFoundError as e:
                return self.send_json(dict(error=str(e)), 404)
        return self.send_json(dict(error="not found"), 404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser at the menu page")
    ap.add_argument("--no-preload", action="store_true", help="accepted and ignored (there is nothing to preload since the generator was removed)")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    home = f"http://127.0.0.1:{args.port}/viewer/home.html"
    print(f"viewer: {home}   (character browser: http://127.0.0.1:{args.port}/viewer/; Ctrl-C to stop)")
    if not args.no_open:
        import webbrowser
        threading.Timer(0.6, webbrowser.open, [home]).start()
    srv.serve_forever()
