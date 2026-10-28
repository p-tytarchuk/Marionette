"""Describe and label LottieAnimation-660K / LottieSVG-10M with Qwen3-VL-8B (one model per GPU) -> two tables.

Target of the project: fully animated REAL objects with 2 legs (people, birds) or 4 legs (quadrupeds), seen whole.

For every row of a dataset the table has
    id, dataset, shard, caption_lottie, caption_qwen, processed, relevant,
    body_plan, subject, full_body, is_ui_icon, skip_reason, tag
  processed  true if Qwen looked at the image (animation: 3-frame filmstrip of the MP4 preview, svg: the PNG), false if not
  relevant   true if Qwen says: subject is a person/animal AND whole body visible AND not an icon/logo AND body_plan is 2leg/4leg
             (computed here from Qwen's sub-answers, not taken from the model's own verdict); false otherwise or if not processed
  skip_reason  why a row was not processed: no_keyword (caption/name has no person/animal word), sampled_out (--limit-per-shard),
             no_media, decode_failed (null when processed)

Pipeline per GPU process (one process per GPU, bf16 weights ~17 GB of the 24 GB):
    reader thread (tar.zst stream, only wanted members) -> decode threads (cv2 / PIL) -> batches -> Qwen3-VL generate -> table
Shards are pulled from a shared queue by the GPU processes. Resumable: finished shards are skipped, and a partially done shard
reuses its append-only partial file. Output: data/vlm/<dataset>/shard-NNNNN.jsonl.zst; --merge concatenates them into
data/vlm/<dataset>.jsonl (JSON Lines, one object per row).

    scripts/lottie2d/vlm_tables.py --count-only                    # CPU only: how many rows pass the prefilter (for the time estimate)
    scripts/lottie2d/vlm_tables.py --datasets animation --shards 0 --limit-per-shard 64     # smoke test / benchmark on the GPUs
    scripts/lottie2d/vlm_tables.py --datasets animation            # production: stage 1 (2B on all candidates) then stage 2 (8B on the relevant ones)
    scripts/lottie2d/vlm_tables.py --merge
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts").is_dir())
for _p in (HERE / "lottie", ROOT / "lottie"):   # SciData layout: scripts/lottie; Marionette dev layout: lottie/
    if _p.exists():
        sys.path.insert(0, str(_p))

import argparse
import glob
import io
import json
import multiprocessing as mp
import queue
import random
import re
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import orjson
import zstandard
from tqdm import tqdm

SCI = (ROOT / "datasets").exists()   # SciData/Marionette/{datasets,models,scripts,output,viewer}
DATASETS = ROOT / "datasets" if SCI else ROOT.parent / "Datasets"
MODELS = ROOT / "models" if SCI else ROOT.parent / "Models"
SRC = {"animation": "LottieAnimation-660K", "svg": "LottieSVG-10M"}
OUT = Path(os.environ.get("VLM_OUT_DIR", ROOT / "output" if SCI else ROOT / "data" / "vlm"))

PROMPT = """{context}You are building a dataset of fully animated REAL creatures (people, birds and animals that walk on 2 or 4 legs) for a rigging pipeline.

Caption (may be inaccurate): {caption}
Tags (noisy): {tags}

Look at the image and answer with ONLY one compact JSON object with these keys, in this order:
"subject": "person" | "animal" | "creature" (fantasy or anthropomorphic character) | "robot" | "object" | "none"
"body_plan": "2leg" if it stands/walks on two legs (people, birds, upright characters); "4leg" if it stands/walks on four legs (dog, cat, horse, deer ...); "other" for fish, snakes, insects, 6+ legs, no legs, or anything not clearly one of the two.
"full_body": true only if the ENTIRE body (head, torso, all limbs/legs) is visible; false for a lone head/face, a bust, a hand, or a cropped body.
"is_ui_icon": true if it is a generic icon, pictogram, logo or concept illustration (avatar glyph, org chart, person plus gears/charts/symbols) rather than a depicted character.
"description": 2-3 sentences that fully describe the MAIN subject: what it is (species / kind of person), body shape, colors, clothing or markings, pose, and what it does{motion}.

JSON only, no markdown."""

FILMSTRIP = ("You are looking at THREE video frames of one animation (25%, 50%, 75% of its timeline) tiled left-to-right "
             "in one image. Judge the subject using all three frames.\n\n")

_BODY_RE = None
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def body_regex():
    global _BODY_RE
    if _BODY_RE is None:
        import keywords
        words = sorted({re.escape(w) for w in keywords.PERSON_WORDS + keywords.ANIMAL_WORDS}, key=len, reverse=True)
        _BODY_RE = re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)
    return _BODY_RE


def prefilter_pass(row: dict, mode: str, use_tags: bool) -> bool:
    if mode == "none":
        return True
    text = (row.get("caption") or "") + " " + (row.get("name") or row.get("svg_filename") or "")
    if use_tags and isinstance(row.get("tag"), list):
        text += " " + " ".join(map(str, row["tag"]))
    return bool(body_regex().search(text))


def parse_answer(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    o = None
    if m:
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    if not isinstance(o, dict):   # cut off inside "description": the short fields come first, take them by regex
        o = {k: (g.group(1) if (g := re.search(rf'"{k}"\s*:\s*"?([A-Za-z0-9_]+)', text)) else None)
             for k in ("subject", "body_plan", "full_body", "is_ui_icon")}
        d = re.search(r'"description"\s*:\s*"(.*)', text, re.DOTALL)
        o["description"] = d.group(1).rstrip('"} \n') if d else ""
        if o["subject"] is None:
            return None
    subject = str(o.get("subject", "none")).strip().lower()
    plan = str(o.get("body_plan", "other")).strip().lower()
    full = o.get("full_body") is True or str(o.get("full_body")).lower() == "true"
    icon = o.get("is_ui_icon") is True or str(o.get("is_ui_icon")).lower() == "true"
    return {"caption_qwen": str(o.get("description", "")).strip(), "subject": subject, "body_plan": plan,
            "full_body": full, "is_ui_icon": icon,
            "relevant": subject in ("person", "animal") and plan in ("2leg", "4leg") and full and not icon}


# ------------------------------------------------------------------ media decoding (runs in threads)

def decode_mp4(data: bytes, frame_dim: int):
    import cv2
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/dev/shm") as tmp:
        tmp.write(data)
        tmp.flush()
        if hasattr(cv2, "CAP_PROP_N_THREADS"):   # FFmpeg otherwise starts one h264 decoder thread per core in every capture: 4x the CPU, measured
            cap = cv2.VideoCapture(tmp.name, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        else:
            cap = cv2.VideoCapture(tmp.name)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n <= 0:
                return None
            want = {min(round(n * pos), n - 1) for pos in (0.25, 0.5, 0.75)}
            got = {}
            for i in range(max(want) + 1):   # sequential grab(): seeking with CAP_PROP_POS_FRAMES was ~3x slower on these 1500x1500 clips
                if not cap.grab():
                    return None
                if i in want:
                    ok, fr = cap.retrieve()
                    if not ok:
                        return None
                    got[i] = fr
            frames = []
            for i in sorted(got):
                im = Image.fromarray(cv2.cvtColor(got[i], cv2.COLOR_BGR2RGB))
                s = frame_dim / max(im.size)
                if s < 1:
                    im = im.resize((max(round(im.width * s), 28), max(round(im.height * s), 28)))
                frames.append(im)
        finally:
            cap.release()
    h = max(f.height for f in frames)
    tile = Image.new("RGB", (sum(f.width for f in frames), h), (255, 255, 255))
    x = 0
    for f in frames:
        tile.paste(f, (x, 0))
        x += f.width
    return tile


def decode_png(data: bytes, image_dim: int):
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):   # icons are transparent: flatten on white, not black
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    s = image_dim / max(im.size)
    if s < 1:
        im = im.resize((max(round(im.width * s), 28), max(round(im.height * s), 28)))
    return im


# ------------------------------------------------------------------ GPU worker

class Qwen:
    def __init__(self, path: Path, mem_fraction: float):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        transformers.logging.set_verbosity_error()   # otherwise a warning per batch breaks the tqdm bars
        self.torch = torch
        torch.set_num_threads(2)
        if mem_fraction < 1.0:   # optional hard cap; allocations beyond it raise OOM (handled by halving the batch)
            torch.cuda.set_per_process_memory_fraction(mem_fraction)
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "left"
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
        except TypeError:   # transformers < 4.56 calls it torch_dtype
            self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16)
        self.model = self.model.to("cuda").eval()

    def run(self, images, prompts, max_new_tokens):
        torch = self.torch
        try:
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p}]}]
                    for im, p in zip(images, prompts)]
            inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
            return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(images) == 1:
                return [""]
            h = len(images) // 2
            return self.run(images[:h], prompts[:h], max_new_tokens) + self.run(images[h:], prompts[h:], max_new_tokens)


def iter_zst(path):
    with open(path, "rb") as f:
        yield from io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 22)


def read_meta(path: str, dataset: str, a) -> tuple[list[dict], list[dict]]:
    """All rows of a metadata shard as light records, plus the indices that should go to Qwen."""
    rows = []
    with open(path, "rb") as f:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 22):
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if dataset == "animation":
                tar, key = r.get("mp4_tar"), r.get("mp4_key")
            else:
                im = r.get("image") or {}
                tar, key = im.get("tar"), im.get("key")
            rec = {"id": r["id"], "dataset": dataset, "caption_lottie": r.get("caption"), "caption_qwen": None,
                   "processed": False, "relevant": False, "body_plan": None, "subject": None, "full_body": None,
                   "is_ui_icon": None, "skip_reason": None, "qwen_model": None, "relevant_2b": None, "tag": r.get("tag"), "shard": int(Path(path).name.split("-")[1])}
            if not tar or not key:
                rec["skip_reason"] = "no_media"
            elif not prefilter_pass(r, a.prefilter, a.prefilter_tags):
                rec["skip_reason"] = "no_keyword"
            rec["_tar"], rec["_key"] = tar, key
            rows.append(rec)
    cand = [i for i, r in enumerate(rows) if r["skip_reason"] is None]
    if a.limit_per_shard and len(cand) > a.limit_per_shard:
        keep = set(random.Random(a.seed).sample(cand, a.limit_per_shard))
        for i in cand:
            if i not in keep:
                rows[i]["skip_reason"] = "sampled_out"
        cand = sorted(keep)
    return rows, cand


def process_shard(dataset: str, shard: str, a, qwen: Qwen | None, prog) -> dict:
    name = Path(shard).name.split("-of-")[0].replace("metadata-", "")
    out_dir = (OUT / "stage1_2b" if a.stage == 1 else OUT) / dataset   # stage 1: 2B on every prefiltered row; stage 2 (final): 8B on the rows 2B called relevant
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"shard-{name}.jsonl.zst"
    if final.exists():
        return {"skipped": True}
    t0 = time.time()
    if a.stage == 1:
        rows, cand = read_meta(shard, dataset, a)
    else:
        src = OUT / "stage1_2b" / dataset / f"shard-{name}.jsonl.zst"
        if not src.exists():
            return {"skipped": "no stage-1 table"}
        rows = [orjson.loads(l) for l in iter_zst(src)]
        meta, _ = read_meta(shard, dataset, a)
        media = {m["id"]: (m["_tar"], m["_key"]) for m in meta}
        for r in rows:
            r["_tar"], r["_key"] = media.get(r["id"], (None, None))
            r["relevant_2b"] = r["relevant"]
        cand = [i for i, r in enumerate(rows) if r["relevant"] and r["_tar"]]
    prog.put(("scanned", len(rows)))
    partial = out_dir / f"partial-{name}.jsonl"
    done_ids = {}
    if partial.exists():
        for line in partial.read_bytes().splitlines():
            try:
                d = orjson.loads(line)
                done_ids[d["id"]] = d
            except orjson.JSONDecodeError:
                pass
    todo = []
    for i in cand:
        if rows[i]["id"] in done_ids:
            rows[i].update({k: v for k, v in done_ids[rows[i]["id"]].items() if k != "id"})
        else:
            todo.append(i)
    prog.put(("cand", len(todo)))

    if qwen is not None and todo:
        want: dict[str, dict[str, int]] = {}
        for i in todo:
            want.setdefault(rows[i]["_tar"], {})[rows[i]["_key"]] = i
        ready: queue.Queue = queue.Queue()
        sem = threading.Semaphore(a.decode_threads * 8)   # bounds the raw bytes held in memory
        pool = ThreadPoolExecutor(a.decode_threads)
        reader_done = threading.Event()
        n_inflight = [0]
        lock = threading.Lock()

        def job(i, data):
            try:
                img = decode_mp4(data, a.frame_dim) if dataset == "animation" else decode_png(data, a.image_dim)
            except Exception:
                img = None
            finally:
                sem.release()
            ready.put((i, img))

        def reader():
            try:
                for tar_name, keys in want.items():
                    p = next(iter(glob.glob(str(shard).rsplit("/", 1)[0] + "/" + tar_name + ".zst")), None)
                    if p is None:
                        for i in keys.values():
                            ready.put((i, None))
                        continue
                    left = dict(keys)
                    with open(p, "rb") as f, tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|") as tf:
                        for m in tf:
                            i = left.pop(m.name, None)
                            if i is None:
                                continue
                            data = tf.extractfile(m).read()
                            sem.acquire()
                            with lock:
                                n_inflight[0] += 1
                            pool.submit(job, i, data)
                            if not left:
                                break
                    for i in left.values():   # listed in metadata, absent from the tar
                        ready.put((i, None))
            finally:
                reader_done.set()

        threading.Thread(target=reader, daemon=True).start()
        remaining = len(todo)
        buf = []
        context = FILMSTRIP if dataset == "animation" else ""
        motion = ", and its motion across the three frames" if dataset == "animation" else ""
        pf = open(partial, "ab")

        def flush(batch):
            tg = time.time()
            prompts = []
            for i, _ in batch:
                r = rows[i]
                tags = ", ".join(map(str, (r["tag"] or [])[:12])) or "(none)"
                prompts.append(PROMPT.format(context=context, caption=(r["caption_lottie"] or "(none)")[:500],
                                             tags=tags[:300], motion=motion))
            texts = qwen.run([im for _, im in batch], prompts, a.max_new_tokens) if batch else []
            for (i, _), text in zip(batch, texts):
                r = rows[i]
                p = parse_answer(text)
                if p is None and a.stage == 2:   # 8B gave nothing usable: keep the 2B answer
                    continue
                r["processed"] = True
                r["qwen_model"] = a.model
                if p is None:
                    r.update(caption_qwen=text.strip()[:600], skip_reason=None, relevant=False, subject="unparsed")
                else:
                    r.update(p)
                pf.write(orjson.dumps({k: r[k] for k in ("id", "processed", "caption_qwen", "relevant", "subject", "body_plan",
                                                         "full_body", "is_ui_icon", "qwen_model")}) + b"\n")
            pf.flush()
            prog.put(("timing", (0.0, time.time() - tg)))   # seconds spent in generate + bookkeeping
            prog.put(("done", len(batch)))

        while remaining > 0:
            tw = time.time()
            try:
                i, img = ready.get(timeout=1.0)
                prog.put(("timing", (time.time() - tw, 0.0)))   # seconds the GPU loop waited for a decoded image
            except queue.Empty:
                prog.put(("timing", (time.time() - tw, 0.0)))
                if reader_done.is_set() and ready.empty() and n_inflight[0] == 0:
                    break
                continue
            remaining -= 1
            if img is None:
                if a.stage == 1:   # in stage 2 the row keeps its 2B answer
                    rows[i]["skip_reason"] = "decode_failed"
                prog.put(("done", 1))
                continue
            buf.append((i, img))
            if len(buf) >= a.batch_size:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
        pf.close()
        pool.shutdown()
        # rows the reader never delivered (truncated tar) are left unprocessed
        for i in todo:
            if not rows[i]["processed"] and rows[i]["skip_reason"] is None:
                rows[i]["skip_reason"] = "no_media"
    elif qwen is None:   # --count-only
        pass

    if qwen is not None:
        tmp = final.with_suffix(".tmp")
        with open(tmp, "wb") as fo, zstandard.ZstdCompressor(level=6).stream_writer(fo) as w:
            for r in rows:
                r.pop("_tar"), r.pop("_key")
                w.write(orjson.dumps(r) + b"\n")
        tmp.rename(final)
        partial.unlink(missing_ok=True)
    return {"rows": len(rows), "candidates": len(cand), "seconds": round(time.time() - t0, 1)}


def gpu_main(gpu: str | None, jobs, prog, a):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one worker must not grab every core: the decode threads and the GPU feeder share them
        os.environ[v] = "2"
    import cv2
    cv2.setNumThreads(1)   # cv2 otherwise spawns a full-size pool inside each of the decode threads (load average 24 on 20 threads)
    qwen = Qwen(MODELS / a.model, a.gpu_mem_fraction) if not a.count_only else None
    prog.put(("ready", gpu))
    while True:
        try:
            ds, shard = jobs.get(timeout=2)
        except queue.Empty:
            break
        res = process_shard(ds, shard, a, qwen, prog)
        prog.put(("shard", (ds, Path(shard).name, res)))
    prog.put(("exit", gpu))


# ------------------------------------------------------------------ driver

def merge():
    for ds in SRC:
        files = sorted(glob.glob(str(OUT / ds / "shard-*.jsonl.zst")))
        if not files:
            continue
        dst = OUT / f"{ds}.jsonl"
        n = n_proc = n_rel = 0
        with open(dst, "wb") as fo:
            for p in tqdm(files, desc=f"merge {ds}"):
                with open(p, "rb") as f:
                    for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 22):
                        fo.write(line)
                        d = orjson.loads(line)
                        n += 1
                        n_proc += d["processed"]
                        n_rel += d["relevant"]
        print(f"{dst}: {n} rows, {n_proc} processed by Qwen, {n_rel} relevant ({len(files)} shards)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="animation,svg", help="processed in this order")
    ap.add_argument("--shards", default="", help="comma list of shard numbers (default: all, in a seeded random order)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--stage", choices=("1", "2", "both"), default="both",
                    help="1 = 2B on every prefiltered row -> output/stage1_2b; 2 = 8B on the rows stage 1 called relevant -> output/<dataset> (final)")
    ap.add_argument("--model1", default="Qwen3-VL-2B-Instruct", help="folder under models/")
    ap.add_argument("--model2", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size1", type=int, default=16)
    ap.add_argument("--batch-size2", type=int, default=6)
    ap.add_argument("--gpu-mem-fraction", type=float, default=1.0, help="max share of each GPU's memory (1.0 = no cap)")
    ap.add_argument("--decode-threads", type=int, default=3, help="per GPU process")
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--frame-dim", type=int, default=384, help="animation: long edge of each of the 3 filmstrip frames")
    ap.add_argument("--image-dim", type=int, default=512, help="svg: long edge of the PNG")
    ap.add_argument("--prefilter", choices=("body", "none"), default="body",
                    help="body = caption/name must contain a person or animal word; none = send every row to Qwen")
    ap.add_argument("--prefilter-tags", action="store_true", help="also search the (noisy) tags")
    ap.add_argument("--limit-per-shard", type=int, default=0, help="random sample of this many Qwen images per shard (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count-only", action="store_true", help="no GPU: scan metadata and report how many rows pass the prefilter")
    ap.add_argument("--cpu-workers", type=int, default=16, help="--count-only")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()
    for st in ((1, 2) if a.stage == "both" else (int(a.stage),)):
        a.stage = st
        a.model, a.batch_size = (a.model1, a.batch_size1) if st == 1 else (a.model2, a.batch_size2)
        print(f"=== stage {st}: {a.model}", flush=True)
        run_stage(a)


def run_stage(a):
    jobs_l = []
    totals = {"rows": 0}
    for ds in a.datasets.split(","):
        sh = sorted(glob.glob(str(DATASETS / SRC[ds] / "snapshots" / "*" / "data" / "metadata-*.jsonl.zst")))
        if a.shards:
            want = {int(s) for s in a.shards.split(",")}
            sh = [s for s in sh if int(Path(s).name.split("-")[1]) in want]
        random.Random(a.seed).shuffle(sh)   # a partial run is then a random sample, not the first ids
        od = (OUT / "stage1_2b" if a.stage == 1 else OUT) / ds
        s1 = OUT / "stage1_2b" / ds
        jobs_l += [(ds, s) for s in sh if a.count_only or (not (od / f"shard-{Path(s).name.split('-')[1]}.jsonl.zst").exists()
                                                           and (a.stage == 1 or (s1 / f"shard-{Path(s).name.split('-')[1]}.jsonl.zst").exists()))]
        n_all = len(glob.glob(str(DATASETS / SRC[ds] / "snapshots" / "*" / "data" / "metadata-*.jsonl.zst")))
        n_rows = json.load(open(glob.glob(str(DATASETS / SRC[ds] / "snapshots" / "*" / "stats.json"))[0]))["num_examples"]
        totals["rows"] += n_rows * len(sh) // n_all
    if not jobs_l:
        print("nothing to do")
        return

    ctx = mp.get_context("spawn")
    jobs, prog = ctx.Queue(), ctx.Queue()
    for j in jobs_l:
        jobs.put(j)
    gpus = [None] * a.cpu_workers if a.count_only else a.gpus.split(",")
    procs = [ctx.Process(target=gpu_main, args=(g, jobs, prog, a), daemon=True) for g in gpus]
    for p in procs:
        p.start()
    b_shard = tqdm(total=len(jobs_l), desc="shards", position=0, unit="shard")
    b_scan = tqdm(total=totals["rows"], desc="rows scanned", position=1, unit="row", smoothing=0.05)
    b_qwen = tqdm(total=0, desc="Qwen images", position=2, unit="img", smoothing=0.05)
    cand_total = 0
    t_wait = t_gen = 0.0
    alive = len(procs)
    try:
        while alive:
            try:
                kind, v = prog.get(timeout=5)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            if kind == "scanned":
                b_scan.update(v)
            elif kind == "cand":
                cand_total += v
                b_qwen.total = cand_total
                b_qwen.refresh()
            elif kind == "done":
                b_qwen.update(v)
            elif kind == "timing":
                t_wait += v[0]
                t_gen += v[1]
                if t_wait + t_gen > 0:
                    b_qwen.set_postfix_str(f"GPU loop busy {t_gen / (t_wait + t_gen):.0%} (waiting for decode {t_wait / (t_wait + t_gen):.0%})")
            elif kind == "shard":
                b_shard.update(1)
                ds, nm, res = v
                tqdm.write(f"{ds} {nm}: {res}")
            elif kind == "exit":
                alive -= 1
    except KeyboardInterrupt:
        tqdm.write("interrupted: partial files are kept, rerun to resume")
        for p in procs:
            p.terminate()
    if a.count_only:
        tqdm.write(f"candidates for Qwen (prefilter={a.prefilter}): {cand_total} of {b_scan.n} rows scanned")


if __name__ == "__main__":
    main()
