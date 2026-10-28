#!/usr/bin/env python3
"""Action-focused captions for the Lottie animations, in the wording of the 3D captions ("An object waves its hand and points upward.").

The Lottie captions (Qwen, ~35 words) describe the whole picture: clothes, props, backgrounds, and somewhere in the middle what moves. The semantic motion model learns from
a caption's CLIP embedding, so scene words (a red shirt, a plain white background) are noise for it. This extracts the motion phrases by rules, no language model:

  1. find the verbs of the 3D captions (data/text_frontend.json, third-person forms) plus a short list of verbs the Lottie captions use for small movements (nods, types, blinks, ...);
  2. a phrase = the verb and the words after it, up to 6, cut at punctuation, "while", "as", "against", "with", "whose", "which", "that", "then", "showing", "displaying";
  3. dynamic verbs (gestures, moves, waves, points, raises, ...) come first; posture verbs (holds, stands, sits, wears, ...) only when nothing dynamic is found, or as the last phrase;
  4. colour and size words are dropped, "his / her / their" become "its", at most 2 phrases are kept: "An object gestures and raises its right hand."

    python scripts/lottie2d/lottie_actions.py show 30              # 30 random captions next to their action caption
    python scripts/lottie2d/lottie_actions.py build                # every row of data/embeddings/lottie_feats_index.csv -> data/embeddings/lottie_actions.csv
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "embeddings"

STATIC = {"holds", "stands", "sits", "wears", "stays", "lies", "rests", "remains", "carries", "poses", "leans", "kneels", "uses"}
JUNK = {"has", "shows", "appears", "contains", "displays", "features", "is", "wears", "makes", "takes", "performs", "sides", "blocks", "places", "brings", "transitions", "alternates", "stays", "poses"}
EXTRA = {"nods", "types", "blinks", "winks", "smiles", "laughs", "talks", "speaks", "presents", "writes", "scrolls", "taps", "claps", "shakes", "bobs", "wiggles", "twitches", "flaps",
         "waves", "gestures", "jumps", "bounces", "sways", "swings", "rotates", "spins", "turns", "tilts", "bends", "stretches", "dances", "walks", "runs", "points", "raises", "lowers",
         "lifts", "moves", "shifts", "reaches", "pushes", "pulls", "applies", "drinks", "eats", "pours", "throws", "catches", "kicks", "waggles", "nudges", "rocks", "hops", "floats",
         "rises", "falls", "flies", "opens", "closes", "swipes", "clicks", "presses", "pats", "scratches", "waters", "sweeps", "cleans", "cooks", "paints", "draws", "reads", "looks",
         "checks", "celebrates", "cheers", "greets", "thinks", "sneezes", "coughs", "yawns", "stretches", "exercises", "lifts", "pedals", "rides", "drives", "swims", "climbs", "crawls"}
STOP = {",", ".", ";", "while", "as", "against", "with", "whose", "which", "that", "then", "showing", "displaying", "set", "behind", "beside", "above", "below", "next", "before", "after",
        "creating", "revealing", "indicating", "symbolizing", "representing", "surrounded", "positioned", "placed", "featuring", "alongside", "near", "inside", "within", "throughout", "from"}
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "brown", "beige", "teal", "dark", "light", "golden", "silver", "cyan", "navy",
          "small", "large", "big", "tiny", "little", "huge", "bright", "colorful", "stylized", "animated", "cartoon", "flat", "minimalist", "illustrated"}
POSS = {"his": "its", "her": "its", "their": "its", "my": "its", "hers": "its"}
MAX_WORDS = 5


def third_of(b: str) -> str:
    if b.endswith(("s", "x", "z", "ch", "sh")):
        return b + "es"
    if b.endswith("y") and len(b) > 2 and b[-2] not in "aeiou":
        return b[:-1] + "ies"
    return b + "s"


def gerunds_of(b: str) -> set:
    out = {b + "ing"}
    if b.endswith("e") and not b.endswith("ee"):
        out.add(b[:-1] + "ing")
    if len(b) >= 3 and b[-1] not in "aeiouwxy" and b[-2] in "aeiou" and b[-3] not in "aeiou":
        out.add(b + b[-1] + "ing")
    return out


def load_verbs():
    """(third-person verbs, gerund -> third-person map), from clean base verbs only: the base forms in the 3D lexicon plus the verbs listed above, inflected by rule
    (the lexicon also holds over-generated forms such as 'pointes', which are ignored)."""
    d = json.loads((ROOT / "data" / "text_frontend.json").read_text())
    bases = {w for w in d["verbs"] if w.isalpha() and len(w) > 2 and not w.endswith(("s", "ed", "ing"))}
    for t in EXTRA | STATIC:
        bases.add(t[:-3] + "y" if t.endswith("ies") else t[:-2] if t.endswith(("sses", "xes", "zes", "ches", "shes")) else t[:-1])
    third, ger = set(), {}
    for b in bases:
        t = third_of(b)
        if t in JUNK:
            continue
        third.add(t)
        for g in gerunds_of(b):
            ger.setdefault(g, t)
    return third | ((EXTRA | STATIC) - JUNK), ger


VERBS, GERUNDS = load_verbs()


def phrases(text: str) -> list[tuple[str, str]]:
    """[(verb, phrase)] in caption order; a gerund ("pointing forward") becomes the third person ("points forward")."""
    toks = re.findall(r"[a-z']+|[,.;]", text.lower())
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        v = w if w in VERBS else GERUNDS.get(w)
        prev = toks[i - 1] if i else ""
        if v and prev not in ("a", "an", "the", "his", "her", "its", "their", "of", "to", "in", "on", "by", "for"):
            j, ws = i + 1, [v]
            while j < len(toks) and len(ws) < MAX_WORDS and toks[j] not in STOP:
                if (toks[j] in VERBS or toks[j] in GERUNDS) and ws[-1] == "and":
                    break                                              # "and points": a new phrase starts
                ws.append(toks[j])
                j += 1
            ws = [POSS.get(x, x) for x in ws if x not in COLORS]
            while ws and ws[-1] in ("and", "a", "an", "the", "of", "to", "in", "on", "at", "its", "up", "over", "into", "onto", "toward", "using", "by", "near"):
                ws.pop()
            out.append((v, " ".join(ws)))
            i = j
        else:
            i += 1
    return out


def action_caption(text: str, max_phrases: int = 2) -> str:
    ps = phrases(text)
    dyn = [p for v, p in ps if v not in STATIC]
    sta = [p for v, p in ps if v in STATIC]
    keep = (dyn[:max_phrases] + sta[: max(0, max_phrases - len(dyn))]) if dyn else sta[:1]
    keep = [p for p in keep if p]
    return "An object " + " and ".join(keep) + "." if keep else ""


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    rows = list(csv.DictReader(open(EMB / "lottie_feats_index.csv", newline="")))
    if cmd == "show":
        random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        for r in random.sample(rows, int(sys.argv[2]) if len(sys.argv) > 2 else 20):
            print(f"- {r['caption'][:260]}\n  => {action_caption(r['caption']) or '(none)'}\n")
    else:
        out = EMB / "lottie_actions.csv"
        n_ok = n_dyn = 0
        words = []
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "action_caption"])
            for r in rows:
                a = action_caption(r["caption"])
                w.writerow([r["clip"], a])
                n_ok += bool(a)
                n_dyn += bool(a) and any(v not in STATIC for v, _ in phrases(r["caption"]))
                words.append(len(a.split()))
        print(f"{len(rows):,} captions: action caption found for {n_ok:,} ({100 * n_ok / len(rows):.1f}%), with a dynamic verb for {n_dyn:,} ({100 * n_dyn / len(rows):.1f}%); "
              f"mean length {sum(words) / len(words):.1f} words (original {sum(len(r['caption'].split()) for r in rows) / len(rows):.1f}) -> {out}")


if __name__ == "__main__":
    main()
