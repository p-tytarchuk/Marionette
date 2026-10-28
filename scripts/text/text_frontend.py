#!/usr/bin/env python3
"""A prompt front end with no language model (small enough for a phone): split a prompt into shape and motion, and rewrite the motion into the dataset's caption wording.

  python3 scripts/text/text_frontend.py "a small brown dog doing a flip backwards" "person does a back flip"

split    rules: the first word that is a motion verb (a list read from the training captions, with its inflections) starts the MOTION; what comes before it is the SHAPE
         ("a small brown dog" | "doing a flip backwards"). No verb found: the whole prompt is the motion (or, with a leading article and noun phrase only, the shape).
rewrite  rules ("rules"): "doing / does / performs X" -> "An object performs X.", any other leading verb -> third person present ("running fast" -> "An object runs fast."), a few spelling
         normalisations ("back flip" -> "backflip"); optionally ("snap" / "hybrid") the nearest training caption by CLIP when it is close enough. Needs only the CLIP text tower and a word list.
Compared with the Qwen2.5 LLM versions in docs/EMBED.md "Small LLM / no LLM".
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(d) for d in (ROOT / "scripts").glob("[a-z]*") if d.suffix == ""]
EMB = ROOT / "data" / "embeddings"
LEXICON = ROOT / "data" / "text_frontend.json"
MIN_COS = 0.85                  # snap to the nearest training caption only above this centred-CLIP cosine (hybrid mode)
DO_VERBS = {"do", "does", "doing", "did", "perform", "performs", "performing", "performed", "make", "makes", "making"}
SPELL = [(r"\bback[ -]flips?\b", "backflip"), (r"\bbackwards\b", "backward"), (r"\bsomersault\b", "somersault"), (r"\bcartwheels?\b", "cartwheel")]
STOP_SHAPE = {"that", "who", "which", "is", "are", "was", "while", "then", "and", "to", "can", "will", "please", ","}
NOT_VERBS = {"object", "a", "an", "the", "in", "on", "at", "of", "with", "his", "her", "its", "their", "it", "is", "are", "be", "up", "down", "out", "off", "over", "into", "onto", "from",
             "forward", "backward", "backwards", "sideways", "slowly", "quickly", "fast", "high", "low", "left", "right", "around", "back", "front", "while", "then", "and", "to", "one", "two",
             "both", "hands", "hand", "arms", "arm", "legs", "leg", "feet", "foot", "head", "body", "air", "ground", "floor", "place", "circle", "position", "direction", "same", "other",
             "another", "each", "before", "after", "as", "by", "for", "its", "them", "him", "her", "this", "that", "no", "not", "new"}


NOT_GERUNDS = {"something", "anything", "nothing", "everything", "building", "ceiling", "morning", "evening", "clothing", "painting", "wedding", "feeling", "ring", "string"}


def forms(base):
    out = {base, base + "s", base + "es", base + "ed", base + "ing"}
    if base.endswith("e"):
        out |= {base[:-1] + "ing"}
    if base.endswith("y"):
        out |= {base[:-1] + "ies", base[:-1] + "ied"}
    if len(base) >= 3 and base[-1] not in "aeiouwxy" and base[-2] in "aeiou" and base[-3] not in "aeiou":
        out |= {base + base[-1] + "ing", base + base[-1] + "ed"}
    return out


def base_of(third):
    for suf, rep in (("ies", "y"), ("shes", "sh"), ("ches", "ch"), ("sses", "ss"), ("xes", "x"), ("zes", "ze"), ("oes", "o"), ("ses", "s"), ("es", "e"), ("s", "")):
        if third.endswith(suf) and len(third) > len(suf) + 1:
            return third[:-len(suf)] + rep
    return third


class Frontend:
    def __init__(self, rewriter=None, lexicon=LEXICON):
        """Rules only: the word lists are read from `lexicon` (data/text_frontend.json, about 40 KB), made once from the training captions. `rewriter` (a rewrite_prompt.Rewriter,
        no LLM is loaded) is needed only to build that file or for the nearest-caption modes of the evaluation."""
        self.rw = rewriter
        if rewriter is None and Path(lexicon).exists():
            d = json.loads(Path(lexicon).read_text())
            self.verbs, self.nouns, self.stems = set(d["verbs"]), set(d["nouns"]), tuple(d["stems"])
            return
        if rewriter is None:
            from rewrite_prompt import Rewriter
            rewriter = self.rw = Rewriter()
        verbs = {}
        self_nouns = set()                                               # words after "performs a" (backflip, handstand, cartwheel, front flip ...)
        for c in self.rw.caps:                                           # captions of the TRAIN split: "An object walks forward." -> walks
            m = re.match(r"an object ([a-z]+)", c.lower())
            if m and m.group(1) not in NOT_VERBS:
                verbs[m.group(1)] = verbs.get(m.group(1), 0) + 1
            for n in re.findall(r"(?:performs|does) (?:a|an|the) ([a-z]+(?: [a-z]+)?)", c.lower()):
                self_nouns.update(n.split())
        self.verbs = set()
        for v, n in verbs.items():
            if n >= 1:
                for f in forms(base_of(v)) | {v}:
                    self.verbs.add(f)
        for v in ("stroll trot gallop march jog sprint hop skip leap crawl walk run jump dance wave kick punch spin roll fall sit stand climb swim fly bow kneel crouch stretch throw catch turn bend lean "
                  "shake nod clap point salute stumble slide tumble flip dive lunge duck dodge roar howl bark limp sneak tiptoe waddle swing push pull lift carry hug fight attack block slash stab shoot aim").split():
            self.verbs |= forms(v)
        self.verbs |= {"do", "does", "doing", "did", "perform", "performs", "performing", "make", "makes", "making", "go", "goes", "going", "start", "starts", "starting", "back", "flip", "flips", "flipping"}
        self.verbs -= {"back", "forward"}
        import paraphrase as PP
        self.stems = tuple(PP.ACTIONS)
        self.nouns = self_nouns - NOT_VERBS
        Path(lexicon).write_text(json.dumps(dict(verbs=sorted(self.verbs), nouns=sorted(self.nouns), stems=list(self.stems))))

    def split(self, prompt):
        """-> (shape, motion), '' when missing"""
        words = prompt.strip().rstrip(".!").split()
        low = [w.lower().strip(",.") for w in words]
        for i, w in enumerate(low):
            if i > 0 and low[i - 1] in ("a", "an", "the", "his", "her", "its", "this", "another"):
                continue                                                 # "a duck waddles": a word right after an article is a noun, not the verb
            if (w in self.verbs or (w.endswith("ing") and len(w) > 5 and w not in NOT_GERUNDS)) and w not in NOT_VERBS:
                shape = [x for x in words[:i]]
                while shape and shape[-1].lower().strip(",") in STOP_SHAPE:
                    shape.pop()
                while shape and shape[-1].lower() in ("front", "back", "side", "high", "long"):
                    i -= 1                                               # "front flip": the direction word belongs to the motion
                    shape.pop()
                shape = " ".join(shape).strip(" ,")
                if shape.lower() in ("someone", "somebody", "something", "it", "he", "she", "they", "an object", "the object", "object", "the character", "a character"):
                    shape = ""
                return shape, " ".join(words[i:])
        for i, w in enumerate(low):                                      # no verb: a motion noun ("backflip", "a handstand") starts the motion, else it is all shape ("a robot dog")
            if w in self.nouns or (w not in NOT_VERBS and w.startswith(self.stems)) or any(st in w for st in ("flip", "kick", "jump", "punch", "cartwheel", "somersault")):
                j = i - 1 if i > 0 and low[i - 1] in ("front", "back", "side", "high", "long", "jumping") else i
                return " ".join(words[:j]).strip(" ,"), " ".join(words[j:])
        return " ".join(words), ""

    def third(self, w):
        """a verb form -> third person present: running -> runs, do -> does, swings -> swings"""
        w = w.lower()
        if w in ("do", "does", "doing", "did"):
            return "does"
        if w.endswith("ing") and len(w) > 4:
            stem = w[:-3]
            cands = ([stem[:-1]] if stem[-1] == stem[-2:-1] else []) + [stem, stem + "e"]
            w = next((c for c in cands if c in self.verbs), cands[0 if len(cands) == 2 else 1])
        elif w.endswith("s") and not w.endswith("ss"):
            return w
        if w.endswith(("s", "x", "z", "ch", "sh", "o")):
            return w + "es"
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        return w + "s"

    def caption_form(self, motion):
        """motion words -> a sentence in the dataset's style, by rules"""
        t = motion.strip().rstrip(".!")
        for pat, rep_ in SPELL:
            t = re.sub(pat, rep_, t, flags=re.I)
        words = t.split()
        if not words:
            return ""
        if words[0].lower() in DO_VERBS:
            rest = words[1:]
            while rest and rest[0].lower() in ("it", "them", "some", "this", "that"):
                rest = rest[1:]
            if rest and rest[0].lower() not in ("a", "an", "the", "some", "one", "two", "three"):
                rest = ["a"] + rest
            return "An object performs " + " ".join(rest) + "."
        if words[0].lower() in self.verbs:
            words[0] = self.third(words[0])
        elif words[0].lower() in self.nouns or any(k in " ".join(words[:2]).lower() for k in ("flip", "somersault", "cartwheel", "handstand")):
            words = ["performs", "a"] + words                         # a bare motion noun: "backflip" -> "performs a backflip"
        return "An object " + " ".join(words) + "."

    def motion_text(self, motion, mode="rules"):
        """mode rules: caption_form; snap: nearest training caption always; hybrid: nearest caption when its cosine is at least MIN_COS, else caption_form"""
        text = self.caption_form(motion)
        if mode == "rules" or not text:
            return text
        near, cos = self.rewrite(text, 0.0)
        return near if (mode == "snap" or cos >= MIN_COS) else text

    def rewrite(self, motion, min_cos=MIN_COS):
        """nearest training caption of the motion text, or 'An object ' + the text. -> (text, cosine)"""
        e = self.rw.embed(["An object " + motion])[0]
        s = self.rw.caps_emb @ e
        j = int(np.argmax(s))
        return (self.rw.caps[j], float(s[j])) if s[j] >= min_cos else ("An object " + motion[:1].lower() + motion[1:], float(s[j]))


def main():
    fe = Frontend()
    for p in sys.argv[1:]:
        shape, motion = fe.split(p)
        print(f"{p!r:52s} shape {shape!r:26s} motion {motion!r:30s} -> rules {fe.motion_text(motion) if motion else ''!r}   hybrid {fe.motion_text(motion, 'hybrid') if motion else ''!r}")


if __name__ == "__main__":
    main()


class Adapter:
    """drop-in for rewrite_prompt.Rewriter in wording_eval.py (embed + rewrite): split away the shape, then write the motion in dataset form"""
    def __init__(self, rewriter, mode="rules"):
        self.rw, self.mode = rewriter, mode
        self.fe = Frontend(rewriter)

    def embed(self, texts):
        return self.rw.embed(texts)

    def rewrite(self, prompts):
        out = []
        for p in prompts:
            shape, motion = self.fe.split(p)
            out.append(self.fe.motion_text(motion or p, self.mode) or p)
        return out
