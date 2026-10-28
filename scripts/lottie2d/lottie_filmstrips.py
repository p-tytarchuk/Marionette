"""Contact sheets of Lottie animations: 6 frames per animation rendered with lottie-web (SVG), several animations per sheet.

    python scripts/lottie2d/lottie_filmstrips.py prep   --n-2leg 20 --n-4leg 10     # needs orjson + zstandard: copies the sampled animations' JSON out of the corpus
    /usr/local/bin/python3 scripts/lottie2d/lottie_filmstrips.py render             # needs playwright + Chrome (the system Python on this Mac)

The sample is the rig-like animations of data/analysis/lottie_rig_probe.csv (parent chain >= 3 deep and >= 3 animated rotations), picked at random.
Work files go to --work (default: a folder next to the sheets' output is not used; pass the scratchpad), sheets to docs/figures/lottie_rig_like_<k>.png.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "analysis" / "lottie_rig_probe.csv"
CORPUS = ROOT.parent / "Datasets" / "LottieAnimation-660K"
LOTTIE_JS = ROOT / "viewer" / "lottie.min.js"
FRAMES = 6
PER_SHEET = 10


def cmd_prep(args) -> None:
    import orjson
    import zstandard
    rows = [r for r in csv.DictReader(open(CSV)) if int(r["max_parent_depth"]) >= 3 and int(r["rot_layers"]) + int(r["rot_groups"]) >= 3]
    rng = random.Random(args.seed)
    pick = []
    for bp, n in (("2leg", args.n_2leg), ("4leg", args.n_4leg)):
        pool = [r for r in rows if r["body_plan"] == bp]
        pick += rng.sample(pool, min(n, len(pool)))
    print(f"{len(rows)} rig-like rows in the probe; sampling {len(pick)}")
    work = Path(args.work)
    (work / "json").mkdir(parents=True, exist_ok=True)
    by_shard: dict[int, dict[str, dict]] = {}
    for r in pick:
        by_shard.setdefault(int(r["shard"]), {})[r["id"]] = r
    rx, got = re.compile(rb'^\{"id":"(\d+)"'), 0
    for s, ids in by_shard.items():
        path = glob.glob(str(CORPUS / "snapshots" / "*" / "data" / f"metadata-{s:05d}-of-00068.jsonl.zst"))[0]
        with open(path, "rb") as fh:
            for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), 1 << 22):
                m = rx.match(line[:40])
                if m and m.group(1).decode() in ids:
                    row = orjson.loads(line)
                    lj = row["lottie_json"]
                    lj = orjson.loads(lj) if isinstance(lj, str) else lj
                    (work / "json" / f"{row['id']}.json").write_bytes(orjson.dumps(lj))
                    got += 1
    manifest = [dict(id=r["id"], body_plan=r["body_plan"], subject=r["subject"], depth=r["max_parent_depth"], rot=int(r["rot_layers"]) + int(r["rot_groups"]),
                     expr=r["expressions"], chain=r["chain"], words=r["words"]) for r in pick if (work / "json" / f"{r['id']}.json").exists()]
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"copied {got} animations to {work / 'json'}")


PAGE = """<!doctype html><meta charset=utf-8><style>
body{margin:0;background:#fff;font:12px system-ui;color:#222} .row{display:flex;align-items:center;border-bottom:1px solid #ccc;padding:3px 6px}
.lab{width:250px;flex:none;padding-right:8px;line-height:1.25;white-space:pre-line} .cell{width:170px;height:170px;flex:none;background:#f4f4f4;margin-right:4px;overflow:hidden}
.cell svg{width:100%;height:100%} </style><div id=root></div>"""


def cmd_render(args) -> None:
    from playwright.sync_api import sync_playwright
    work = Path(args.work)
    man = json.loads((work / "manifest.json").read_text())
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        for k in range(0, len(man), PER_SHEET):
            chunk = man[k: k + PER_SHEET]
            pg = b.new_page(viewport={"width": 1500, "height": 180 * len(chunk) + 10})
            pg.set_content(PAGE)
            pg.add_script_tag(path=str(LOTTIE_JS))
            for m in chunk:
                data = (work / "json" / f"{m['id']}.json").read_text()
                lab = f"{m['id']}  {m['body_plan']} · depth {m['depth']} · {m['rot']} rotations{' · EXPR' if m['expr'] == 'True' else ''}\nchain: {m['chain'][:70]}\nwords: {m['words']}"
                pg.evaluate("""([data, lab, n]) => {
                  const row = document.createElement('div'); row.className = 'row';
                  const l = document.createElement('div'); l.className = 'lab'; l.textContent = lab; row.appendChild(l);
                  document.getElementById('root').appendChild(row);
                  const ad = JSON.parse(data);
                  for (let i = 0; i < n; i++) {
                    const c = document.createElement('div'); c.className = 'cell'; row.appendChild(c);
                    try {
                      const a = lottie.loadAnimation({container: c, renderer: 'svg', loop: false, autoplay: false, animationData: JSON.parse(data)});
                      const f = a.totalFrames * i / (n - 1) * 0.999;
                      a.goToAndStop(f, true);
                    } catch (e) { c.textContent = 'render error'; }
                  }
                }""", [data, lab, FRAMES])
            pg.wait_for_timeout(1200)
            out = ROOT / "docs" / "figures" / f"lottie_rig_like_{k // PER_SHEET + 1}.png"
            pg.screenshot(path=str(out), full_page=True)
            print(out)
            pg.close()
        b.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "render"])
    ap.add_argument("--work", required=True, help="folder for the copied JSON and the manifest")
    ap.add_argument("--n-2leg", type=int, default=20)
    ap.add_argument("--n-4leg", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    {"prep": cmd_prep, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
