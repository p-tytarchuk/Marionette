"""How much real motion exists per body class and movement mode? Read-only; writes data/analysis/motion_coverage.md.

  python scripts/motion/motion_coverage.py

Counts clips whose motion caption names a movement mode (keyword rules, so a count is "mentions the mode", not a verified label): the 3D clips of the UniML3D export (data/inventory/clips.csv, by the rig's
category) and the tokenised Lottie clips (data/embeddings/lottie_feats_index.csv, by Qwen's body plan, with Qwen's caption). Why: a learned motion model can only learn the modes the data contains, and the
inventory marks every avian, marine, serpentine and insectoid clip "not usable" because those characters were never normalised (normalize.py handles only bipedal and quadrupedal), although their captions exist.
"""
import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "analysis" / "motion_coverage.md"
MODES = {
    "walk": r"walk|walks|walking|stroll|march|marches|pace|step|steps|stepping",
    "run": r"run|runs|running|jog|sprint|dash",
    "trot_gallop": r"trot|trots|gallop|gallops|canter|bound|bounds",
    "hop_jump": r"hop|hops|hopping|jump|jumps|jumping|leap|leaps|bounce|bounces",
    "fly": r"fly|flies|flying|flap|flaps|flapping|soar|soars|glide|glides|hover|hovers|hovering|wing|wings",
    "swim": r"swim|swims|swimming|paddle|dive|dives",
    "slither_crawl": r"slither|slithers|crawl|crawls|crawling|creep|creeps|wriggle|wriggles|slide|slides|sliding|undulate|undulates",
    "wave_gesture": r"wave|waves|waving|point|points|pointing|nod|nods|nodding|greet|greets|clap|claps",
    "dance": r"dance|dances|dancing",
    "idle": r"idle|stand|stands|standing|breathe|breathes|breathing|sway|sways|swaying|still|bobs|looks",
}
RX = {m: re.compile(rf"\b({p})\b", re.I) for m, p in MODES.items()}
CATS3D = ["bipedal", "quadrupedal", "avian", "marine", "serpentine", "insectoid", "articulated_rigid"]


# the four classes of docs/SKELETON_SPEC.md and the rig categories of the 3D export that belong to each (marine bodies without limbs are counted as limbless: they move by swimming / undulating)
CLASS_OF = {"bipedal": "biped", "quadrupedal": "quadruped", "avian": "winged", "marine": "limbless", "serpentine": "limbless", "insectoid": "other", "articulated_rigid": "other", "uncertain": "other"}
CLASSES = ["biped", "quadruped", "winged", "limbless", "other"]


def coverage():
    """The numbers behind viewer/motion_classes.html: per class (biped, quadruped, winged, limbless, other) the 3D clips of the export (data/inventory/clips.csv), how many are usable now, the movement modes their captions
    mention, the rig categories and reasons behind them and a few example captions; and the Lottie clips by Qwen's body plan. Stdlib only."""
    rows = list(csv.DictReader(open(ROOT / "data" / "inventory" / "clips.csv", newline="")))
    out = {c: dict(id=c, n=0, usable=0, hours=0.0, categories=collections.Counter(), reasons=collections.Counter(), modes=collections.Counter(), examples={m: [] for m in MODES}) for c in CLASSES}
    for r in rows:
        c = out[CLASS_OF.get(r["category"], "other")]
        c["n"] += 1
        c["categories"][r["category"]] += 1
        c["hours"] += float(r["duration_s"] or 0) / 3600
        if r["usable"] == "1":
            c["usable"] += 1
        else:
            c["reasons"][r["not_usable_reason"] or r["character_status"] or "unknown"] += 1
        text = r["caption"] or r["generic"] or r["detail"]
        for m, x in RX.items():
            if x.search(" ".join([r["caption"], r["generic"], r["detail"]])):
                c["modes"][m] += 1
                if len(c["examples"][m]) < 4 and text not in [e["caption"] for e in c["examples"][m]]:
                    c["examples"][m].append(dict(clip=r["clip"], caption=text[:140]))
    classes = []
    for c in out.values():
        top = [m for m, k in c["modes"].most_common(4) if k][:3]
        classes.append(dict(id=c["id"], n=c["n"], usable=c["usable"], hours=round(c["hours"], 2), categories=dict(c["categories"]), reasons=dict(c["reasons"].most_common(4)),
                            modes=dict(c["modes"]), top=top, examples=c["examples"]))
    lot = list(csv.DictReader(open(ROOT / "data" / "embeddings" / "lottie_feats_index.csv", newline="")))
    t, n = tally(lot, lambda r: r["object_type"], lambda r: r["caption"])
    lottie = [dict(id=k, n=n[k], modes=dict(t[k])) for k in ("2leg", "4leg") if k in n]
    return dict(modes=list(MODES), total=len(rows), classes=classes, lottie=lottie, lottie_total=len(lot))


def tally(rows, key, text):
    t, n = collections.defaultdict(collections.Counter), collections.Counter()
    for r in rows:
        k, s = key(r), text(r)
        n[k] += 1
        for m, x in RX.items():
            if x.search(s):
                t[k][m] += 1
    return t, n


def table(title, order, t, n, extra=None):
    cols = list(MODES)
    lines = [f"**{title}**", "", "| class | clips | " + " | ".join(cols) + " |" + (f" {extra[0]} |" if extra else ""), "|---|---|" + "---|" * len(cols) + ("---|" if extra else "")]
    for k in order:
        lines.append(f"| {k} | {n[k]} | " + " | ".join(str(t[k][c]) for c in cols) + " |" + (f" {extra[1](k)} |" if extra else ""))
    return "\n".join(lines) + "\n"


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "inventory" / "clips.csv", newline="")))
    text3 = lambda r: " ".join([r["caption"], r["generic"], r["detail"]])
    t_all, n_all = tally(rows, lambda r: r["category"], text3)
    usable = [r for r in rows if r["usable"] == "1"]
    t_use, n_use = tally(usable, lambda r: r["category"], text3)
    reasons = {c: collections.Counter(r["not_usable_reason"] or r["character_status"] for r in rows if r["category"] == c and r["usable"] != "1").most_common(1) for c in CATS3D}
    lot = list(csv.DictReader(open(ROOT / "data" / "embeddings" / "lottie_feats_index.csv", newline="")))
    t_lot, n_lot = tally(lot, lambda r: r["object_type"], lambda r: r["caption"])
    out = ["# Motion coverage: clips per body class and movement mode", "",
           "A clip counts for a mode when its motion caption mentions it (keyword rules). One clip can count for several modes. Read-only analysis; `scripts/motion/motion_coverage.py`.", "",
           table("3D clips of the export, all (the rig's own category)", CATS3D, t_all, n_all, ("usable now", lambda k: n_use[k])),
           "The rows below the first two are not usable because their characters were never normalised "
           f"(main reason per category: {', '.join(f'{c}: {r[0][0]}' for c, r in reasons.items() if r and c not in ('bipedal', 'quadrupedal'))}); the clips and their captions exist in the export.", "",
           table("Lottie clips (tokenised, Qwen-relevant), Qwen caption keywords", ["2leg", "4leg"], t_lot, n_lot)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
