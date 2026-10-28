"""Run the same prompt and images through two Qwen3-VL sizes and compare labels and speed (decides which model to use for vlm_tables.py).

    scripts/lottie2d/vlm_compare.py --dataset animation --gpu 0 --n 96
"""
import argparse
import gc
import glob
import json
import os
import sys
import tarfile
import time
from argparse import Namespace
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="animation")
ap.add_argument("--gpu", default="0")
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--shard", type=int, default=5)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import vlm_tables as v  # noqa: E402
import zstandard  # noqa: E402

shard = glob.glob(str(v.DATASETS / v.SRC[args.dataset] / "snapshots" / "*" / "data" / f"metadata-{args.shard:05d}-*.jsonl.zst"))[0]
rows, cand = v.read_meta(shard, args.dataset, Namespace(prefilter="body", prefilter_tags=False, limit_per_shard=0, seed=0))
cand = cand[: args.n]
keys = {rows[i]["_key"]: i for i in cand}
tar = glob.glob(str(Path(shard).parent / (rows[cand[0]]["_tar"] + ".zst")))[0]
imgs = {}
with open(tar, "rb") as f, tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|") as tf:
    for m in tf:
        i = keys.get(m.name)
        if i is None:
            continue
        data = tf.extractfile(m).read()
        imgs[i] = v.decode_mp4(data, 384) if args.dataset == "animation" else v.decode_png(data, 512)
        if len(imgs) == len(keys):
            break
ids = sorted(imgs)
film = args.dataset == "animation"
prompts = [v.PROMPT.format(context=v.FILMSTRIP if film else "", caption=(rows[i]["caption_lottie"] or "")[:500],
                           tags=", ".join(map(str, (rows[i]["tag"] or [])[:12]))[:300],
                           motion=", and its motion across the three frames" if film else "") for i in ids]
res = {}
for name, bs in (("Qwen3-VL-2B-Instruct", 16), ("Qwen3-VL-8B-Instruct", 4)):
    q = v.Qwen(v.MODELS / name, 0.8)
    t = time.time()
    out = []
    for k in range(0, len(ids), bs):
        out += q.run([imgs[i] for i in ids[k:k + bs]], prompts[k:k + bs], 260)
    dt = time.time() - t
    torch.cuda.synchronize()
    res[name] = {"seconds": dt, "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "bs": bs,
                 "answers": [v.parse_answer(o) or {"subject": "unparsed", "relevant": False, "caption_qwen": o} for o in out]}
    print(name, f"{dt:.1f}s for {len(ids)} images = {dt / len(ids):.2f} s/img, peak {res[name]['peak_gb']:.1f} GB", flush=True)
    del q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
res["rows"] = [{"id": rows[i]["id"], "caption_lottie": rows[i]["caption_lottie"]} for i in ids]
args.out.write_text(json.dumps(res))
