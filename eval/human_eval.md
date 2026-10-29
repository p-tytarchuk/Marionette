# Human evaluation protocol (both tests)

There are two evaluations of the same outputs, and the report shows them side by side:

| | Model evaluation | Human evaluation |
|---|---|---|
| Who | open VLM judge (chosen by the bake-off, `README.md` rule 4) | you, and at least one more rater if possible |
| Covers | every prompt (about 10k per test) | a stratified sample (see "Sample") |
| Role | scale: gives numbers for every slice | ground truth: the headline numbers, and the check on the judge |
| Axes | the same, from `shape/rubric.md` and `motion/rubric.md` | the same, so the two are directly comparable |

The human scores are the primary result on the sample. The model scores extend the result to the full prompt set, and they are
only called validated for an axis where they agree with the humans (below). If they do not agree on an axis, the report says
so and gives that axis from the human sample only.

## Sample

- Size: 500 items per test as the target; 300 is the minimum for judge calibration. Rating one item takes about a minute, so
  500 items is roughly 8 to 10 hours: do it in sessions of 50 to avoid fatigue drift.
- Stratified over slices (subject / action class x body x difficulty) with a floor per slice, and over the model's own scores
  (clear good, clear bad, borderline), so the sample is not just the easy middle.
- The sample also contains the controls (baseline outputs, real dataset items, negative controls). They are mixed in, shuffled,
  and not labelled, so the raters cannot tell them apart.

## Rating rules

1. Blind: the rater does not see which system made the item, whether it is a control, or the judge's scores.
2. Same axes, same 1 to 5 scale and same images or video as the model judge. `null` axes are skipped.
3. An anchor set comes first: 3 items per score per axis taken from `dev` and rated by agreement, shown to the rater
   during the session so the scale stays the same from the first item to the last.
4. Order is random and fixed by a seed saved with the ratings. No rating is edited after the session; a changed mind is a new row.
5. A rater who cannot judge an item (video fails to play, image missing) marks `cannot_judge`; it is counted, not scored.

## Agreement

- **Human vs human:** at least 20% of the sample is rated by two raters. Report weighted kappa (or Krippendorff's alpha) per
  axis. If humans do not agree on an axis, the axis definition is unclear: fix `rubric.md` and re-rate, do not average it away.
- **Human vs model:** per axis, Spearman correlation, mean absolute difference, and the share of items that differ by 2 or
  more points. The acceptance threshold for "model validated on this axis" is set on `dev` before the `test` split is rated, not after.

## Optional: pairwise preference

Humans are more reliable at "which is better" than at an absolute score. For the headline comparison against the retrieval
baseline, show the system's output and the baseline's output for the same prompt side by side in random left / right order and
ask for a preference (A, B, tie). Report the win rate with a confidence interval.

## Rating file (one JSON object per line, `human.jsonl`)

```json
{"id": "m00001", "rig": "biped_01", "rater": "r1", "session": 3, "seed": 1234,
 "scores": {"action": 4, "order": null, "body_use": 4, "naturalness": 3, "smoothness": 3, "loop": 4},
 "cannot_judge": false, "comment": "optional, one line", "seconds": 52}
```

## Rating tool: `viewer/rate.html` (built 2026-10-28)

```
python viewer/serve.py            # then open http://127.0.0.1:8765/viewer/rate.html  (or the "Rate" card in the menu)
python scripts/tools/make_demo_rating_set.py   # two demo sets under data/eval/_demo/ to try it
```

Keyboard-first: `1`-`5` score the highlighted axis and move on, `0`/`n` not applicable, `Backspace` previous axis, `Enter` save and
next, `x` can't judge, `a` anchors, `c` comment, `[` `]` previous / next item, `space` play / pause. Pairwise sets: `1` left,
`2` right, `3` tie. The page resumes at the first unrated item, stores each rating as one appended line, and never edits a
row (re-rating an item adds a row with a higher `revision`). Items are shown in the set's fixed random order. Only the prompt
and the media reach the browser; the server strips every other field.

### Rating set format

A set is a folder `data/eval/<test>/<run>/rate/` with three files. The builder of a real set (written once the generator's
output format is known) produces the first two.

`set.json`
```json
{"title": "motion test, run 3", "test": "motion", "mode": "scores", "seed": 1234,
 "axes": [{"key": "action", "label": "Action", "question": "...", "nullable": false,
           "anchors": {"1": "...", "3": "...", "5": "..."}}],
 "anchor_items": [{"axis": "action", "score": 1, "caption": "...", "media": [{"type": "image", "src": "media/a1.png", "label": ""}]}]}
```
`mode` is `scores` (axes from the rubric) or `pairwise` (add `"instructions": "..."`, no axes). `seed` fixes the item order.

`items.jsonl`, one object per line
```json
{"id": "m00001", "prompt": "walks forward slowly", "media": [{"type": "video", "src": "media/m00001_a.mp4", "label": ""}],
 "rig": "biped_01", "skip_axes": ["order"], "raters": ["r1", "r2"], "meta": {"system": "...", "control": false}}
```
Pairwise items have `"left": [media...]` and `"right": [media...]` instead of `media`. `skip_axes` lists the axes that do not apply
(stored as null), `raters` (optional) restricts who rates the item (use it for the 20% double-rated overlap), `rig` is copied
into the rating row, and `meta` stays on the server: put the system, the control type and the left / right mapping there.
Media paths are relative to the set folder (or start with `/data/eval/`); use neutral file names, because the browser can
see them. A missing media file makes the set refuse to open.

`human.jsonl` is written by the page: one line per rating, in the format shown above (plus `revision` and `ts`; `choice` instead
of `scores` in pairwise sets).
