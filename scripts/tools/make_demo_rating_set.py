"""Write two small DEMO rating sets for viewer/rate.html, to try the page and to test it.

  python scripts/tools/make_demo_rating_set.py            # writes data/eval/_demo/motion/rate and data/eval/_demo/pairwise/rate
  python viewer/serve.py  ->  http://127.0.0.1:8765/viewer/rate.html

The media are placeholder pictures (and a short test-pattern video when ffmpeg is installed) that say DEMO: they show nothing
about any system. Real sets are built from a test run's outputs in the same format (eval/human_eval.md, "Rating set format").
Existing ratings (human.jsonl) are never touched; re-running only rewrites set.json, items.jsonl and the media.
"""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "eval" / "_demo"

AXES = [   # the motion rubric (eval/motion/rubric.md), anchors shortened
    dict(key="action", label="Action", question="Does the motion show the requested action (verb, direction, speed)?", nullable=False,
         anchors={"1": "wrong or no action", "3": "related action, wrong style", "5": "the requested action, clearly readable"}),
    dict(key="order", label="Order", question="Several steps: all there, in the right order?", nullable=True,
         anchors={"1": "steps missing or reversed", "3": "order or timing off", "5": "right order and timing"}),
    dict(key="body_use", label="Body use", question="Do the right body parts move?", nullable=False,
         anchors={"1": "wrong parts move", "3": "partly right", "5": "exactly the right parts"}),
    dict(key="naturalness", label="Naturalness", question="Is it plausible for this body (gait, weight, balance)?", nullable=False,
         anchors={"1": "broken or random", "3": "recognisable but stiff", "5": "natural"}),
    dict(key="smoothness", label="Smoothness", question="Free of popping, jitter, snapping?", nullable=False,
         anchors={"1": "constant artefacts", "3": "occasional artefacts", "5": "clean"}),
    dict(key="loop", label="Loop", question="Only for loops: does the end lead into the start?", nullable=True,
         anchors={"1": "visible jump", "3": "small pop", "5": "seamless"}),
]
PROMPTS = ["walks forward slowly", "waves with the right hand", "trots in a straight line and then stops", "runs in a wide circle", "does a backflip",
           "jumps three times and then bows", "looks left, then looks right, then nods", "slumps its shoulders and sighs", "stands still and breathes slowly",
           "gallops from left to right", "walks forward without moving its arms", "turns around and sits down"]
MULTI = {2, 5, 6, 11}          # prompts with several steps (the others get Order = n/a)
LOOPS = {0, 3, 8, 9, 10}       # prompts that expect a loop (the others get Loop = n/a)


def svg(path, title, sub, hue, frames=6):
    """a placeholder 'filmstrip': coloured tiles with a moving dot; says DEMO on it"""
    w, h, tiles = 1200, 240, []
    for k in range(frames):
        x = k * (w // frames)
        tiles.append(f'<rect x="{x + 4}" y="4" width="{w // frames - 8}" height="{h - 8}" rx="8" fill="hsl({(hue + 12 * k) % 360},45%,32%)"/>'
                     f'<circle cx="{x + 20 + k * 28}" cy="{70 + 55 * (k % 3)}" r="18" fill="#f2f2f2"/>'
                     f'<text x="{x + 12}" y="{h - 14}" font-size="14" fill="#fff" font-family="sans-serif">frame {k}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + 44}" width="{w}" height="{h + 44}"><rect width="{w}" height="{h + 44}" fill="#0e1013"/>'
                    f'<g transform="translate(0,40)">{"".join(tiles)}</g>'
                    f'<text x="12" y="26" font-size="20" fill="#e8b04a" font-family="sans-serif">DEMO · {title} · {sub} (placeholder, not a real output)</text></svg>', encoding="utf-8")


def video(path, hue):
    if not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=15", "-t", "2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", str(path)], capture_output=True)
    return r.returncode == 0


def write(d, meta, items):
    d.mkdir(parents=True, exist_ok=True)
    (d / "set.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (d / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")


def main():
    # ---- scores mode ----
    d = DEMO / "motion" / "rate"
    (d / "media").mkdir(parents=True, exist_ok=True)
    items, has_video = [], video(d / "media" / "v1.mp4", 0)
    for i, text in enumerate(PROMPTS):
        iid = f"m{i + 1:05d}"
        svg(d / "media" / f"{iid}.svg", f"item {i + 1}", "filmstrip", 30 * i)
        media = [dict(type="image", src=f"media/{iid}.svg", label="12-frame filmstrip")]
        if has_video and i in (0, 3):
            media.append(dict(type="video", src="media/v1.mp4", label="video (test pattern)"))
        skip = ([] if i in MULTI else ["order"]) + ([] if i in LOOPS else ["loop"])
        items.append(dict(id=iid, prompt=text, media=media, rig="demo_rig", skip_axes=skip, meta=dict(system="demo", control=False)))
    anchors = [dict(axis="action", score=s, caption=f"anchor example for score {s}", media=[dict(type="image", src=f"media/anchor_{s}.svg", label="")]) for s in (1, 3, 5)]
    for s in (1, 3, 5):
        svg(d / "media" / f"anchor_{s}.svg", f"anchor score {s}", "action", 100 + 40 * s, frames=4)
    write(d, dict(title="DEMO · motion test", test="motion", mode="scores", seed=1234, axes=AXES, anchor_items=anchors), items)

    # ---- pairwise mode ----
    d = DEMO / "pairwise" / "rate"
    (d / "media").mkdir(parents=True, exist_ok=True)
    items = []
    for i, text in enumerate(PROMPTS[:6]):
        iid = f"m{i + 1:05d}"
        svg(d / "media" / f"{iid}_a.svg", f"item {i + 1}", "output X", 30 * i, frames=4)
        svg(d / "media" / f"{iid}_b.svg", f"item {i + 1}", "output Y", 30 * i + 150, frames=4)
        swap = i % 2 == 1                                                         # the builder randomises sides; the mapping stays hidden in meta
        a, b = (f"media/{iid}_b.svg", f"media/{iid}_a.svg") if swap else (f"media/{iid}_a.svg", f"media/{iid}_b.svg")
        items.append(dict(id=iid, prompt=text, left=[dict(type="image", src=a, label="")], right=[dict(type="image", src=b, label="")],
                          meta=dict(left="baseline" if swap else "system", right="system" if swap else "baseline")))
    write(d, dict(title="DEMO · system vs baseline", test="motion", mode="pairwise", seed=99,
                  instructions="Which motion is the better answer to the prompt? Ignore how the character looks."), items)
    print(f"wrote {DEMO}  (video: {'yes' if has_video else 'no, ffmpeg not found'})")


if __name__ == "__main__":
    main()
