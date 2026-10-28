#!/usr/bin/env python3
"""Serve the character viewer.

  python3 viewer/serve.py [--port 8765] [--no-open]   opens http://127.0.0.1:8765/viewer/home.html (the menu: every page with a manual); the character browser is http://127.0.0.1:8765/viewer/

Serves the project folder (localhost only), /api/retargets (for viewer/compare.html) and /api/characters, which lists the
normalized characters from data/normalize_report.csv, data/manifest.csv and each
normalization.json.
viewer/generate.html: GET /api/targets (characters that can be retarget targets) and POST /api/generate {prompt, target, guidance, seed, mode, rewrite, smooth, straight, tries, land, frontend (rules | llm), mobile, threads, auto_character}; the response has timing_ms
(scripts/generate/generate_motion.py, rewrite: false keeps the prompt as typed; writes a clip under data/characters/<target>/anims/). With auto_character: true the prompt is first split into a shape
part and a motion part (scripts/generate/text_to_character.py); if it names a shape the best-matching retarget target replaces `target` and only the motion part is generated; the response carries
shape: {shape, motion, hits: [{key, cls, cos}]}. Generation needs the conda env "Mario"; the rest does not.
Dataset viewers (one HTML page each, all served from here): viewer/vlm_table.html reads GET /api/vlm_table (the Qwen label table in data/vlm/animation, filtered and paged, loaded in the background on first
use) and plays rows with /api/lottie_item?raw=1; viewer/lottie_raw.html reads GET /api/lottie_raw (a window of rows of one metadata shard of ../Datasets/LottieAnimation-660K or LottieSVG-10M) and
GET /api/lottie_raw_row (one row: Lottie JSON and, for SVG, the SVG source). They replace the Qt viewer of the GPU server (SciData/Marionette/viewer/viewer.py), which stays there.
"""
import argparse
import csv
import json
import os
import sys
import threading
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
    (until a clip is generated through the page, or ?refresh=1, or 10 minutes); the generator page's embedded player asks for its single clip (retarget_one) instead."""
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
        if self.path.split("?")[0] == "/api/warm":                                   # the page ticked "mobile runtime": create the ONNX sessions now, in the background
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                threading.Thread(target=warm_mobile, args=(max(1, min(16, int(req.get("threads", 4)))),), daemon=True).start()
                return self.send_json(dict(ok=True))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self.send_json(dict(error=str(e)), 400)
        if self.path.split("?")[0] != "/api/generate":
            return self.send_json(dict(error="not found"), 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            prompt, target = str(req.get("prompt", "")).strip(), str(req.get("target", ""))
            guidance, seed, mode = float(req.get("guidance", 2.0)), int(req.get("seed", 0)), str(req.get("mode", "auto"))
            if not _target_keys:
                _target_keys.update(t["key"] for t in targets())
            if not prompt or len(prompt) > 200:
                raise ValueError("prompt must be 1-200 characters")
            if target not in _target_keys:
                raise ValueError("unknown target character")
            tries = int(req.get("tries", 4))
            if not 1 <= tries <= 16:
                raise ValueError("tries must be 1-16")
            land = bool(req.get("land", True))
            runtime = "mobile" if req.get("mobile") else "desktop"
            threads = int(req.get("threads", 4))
            if not 1 <= threads <= 16:
                raise ValueError("threads must be 1-16")
            frontend = str(req.get("frontend", "rules"))
            if frontend not in ("rules", "llm"):
                raise ValueError("frontend must be rules or llm")
            straight = 1 if req.get("straight") else 0                              # straight line: zero sideways velocity and yaw rate (generate_motion.straighten)
            smooth = float(req.get("smooth", 3.0))
            if not 0.0 <= smooth <= 8.0:
                raise ValueError("smooth must be 0-8")
            if not 1.0 <= guidance <= 6.0 or mode not in ("auto", "walk", "run", "plain"):
                raise ValueError("guidance must be 1-6; mode auto/walk/run/plain")
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self.send_json(dict(error=str(e)), 400)
        try:
            sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
            import generate_motion
            shape = None
            if req.get("auto_character"):                                          # choose the character from the shape words of the prompt (scripts/generate/text_to_character.py)
                import text_to_character
                shape = text_to_character.choose(prompt, frontend=frontend)
                if shape["hits"]:
                    target, prompt = shape["hits"][0]["key"], shape["motion"] or text_to_character.DEFAULT_MOTION
            out = generate_motion.generate(prompt, target, guidance, seed, mode, rewrite=bool(req.get("rewrite", True)), smooth=smooth, straight=straight, tries=tries, land=land, frontend=frontend, runtime=runtime, threads=threads)
            if shape:
                out["shape"] = shape
            _rt_cache.clear()                                                      # the clip list has a new entry
            self.send_json(out)
        except ImportError as e:
            self.send_json(dict(error=f"generation needs the conda env Mario ({e}); restart: conda activate Mario && python viewer/serve.py"), 500)
        except Exception as e:                                                   # report, keep the server alive
            self.send_json(dict(error=f"{type(e).__name__}: {e}"), 500)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


def warm_mobile(threads):
    try:
        sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
        import generate_motion
        import mobile_runtime
        with generate_motion._lock:
            S = generate_motion._load()
            if mobile_runtime.available():
                mobile_runtime.get(S, threads)
    except Exception as e:
        print(f"warm mobile skipped: {type(e).__name__}: {e}", flush=True)


def preload():
    """Load the generator's models in the background at start-up (about a minute: the Qwen model is 6 GB read from disk), so the first Generate click does not wait for them."""
    import time
    t0 = time.time()
    try:
        sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
        import generate_motion
        import text_to_character
        import shape_retrieval
        with generate_motion._lock:
            generate_motion._load()
        print(f"preload: generator models {time.time() - t0:.0f} s", flush=True)
        text_to_character.get_index()
        shape_retrieval.clip_model()
        if os.environ.get("MARIONETTE_PRELOAD_LLM"):                              # the LLM (frontend "llm") loads on first use unless asked for here
            text_to_character.shared_llm()
        print(f"preload: done, {time.time() - t0:.0f} s", flush=True)
    except Exception as e:                                                       # no conda env, missing files: generation will report its own error
        print(f"preload skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser at the menu page")
    ap.add_argument("--no-preload", action="store_true", help="do not load the generator's models at start-up (they load on the first Generate click)")
    args = ap.parse_args()
    if not args.no_preload:
        threading.Thread(target=preload, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    home = f"http://127.0.0.1:{args.port}/viewer/home.html"
    print(f"viewer: {home}   (character browser: http://127.0.0.1:{args.port}/viewer/; Ctrl-C to stop)")
    if not args.no_open:
        import webbrowser
        threading.Timer(0.6, webbrowser.open, [home]).start()
    srv.serve_forever()
