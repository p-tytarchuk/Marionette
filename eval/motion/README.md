# Test B: generated motion

How good or bad is the motion the system makes from an action description? To keep shape out of the result, every generated
motion is applied to the same small panel of known-good reference rigs. A failure here is then a motion failure, not a body failure.

## Pipeline

```
prompts/prompts_motion.jsonl                        1  action-only prompts + metadata (prompts/schema.json)    you write these
data/eval/motion/<run>/motion/<id>.*                2  generate: joint rotation tracks + root path            system under test
.../lottie/<id>__<rig>.json                         3  apply the motion to each panel rig (see "Panel")
.../checks.jsonl                                    4  deterministic checks (below)
.../video/<id>__<rig>.mp4  or  frames/*.png         5  video, or a filmstrip of 12 frames with index and time
.../judge.jsonl                                     6  model evaluation: VLM judge, scores per axis (rubric.md)
.../human.jsonl                                     7  human evaluation of the sample (../human_eval.md)
.../report.md + failures/                           8  per-slice results, human vs model, failure gallery
```

Step 2 needs an adapter that reads the generator's motion (open decision in `../README.md`).

## Panel of reference rigs (step 3)

A fixed set of 3 to 5 good rigs chosen once and never changed between runs: at least one biped, one standing quadruped and one
"other" body (the project's `data/characters/` and the Lottie rigs built by `rig_mesh.py` are the source). Each prompt is applied
only to the rigs its `applies_to` field allows. Applying a motion to a rig goes through role mapping (hips, spine, limbs) and adds
its own error, so the panel path is measured too: real dataset clips go through the identical path as a control (below).
Number of renders is prompts x applicable rigs, so the panel size is a cost decision.

## 4. Deterministic checks (pass / fail, with the measured value; thresholds to be set on `dev`)

| Check | What it measures |
|---|---|
| Valid output | tracks load, no NaN / infinite values, frame rate and duration in range, duration near `duration_hint_s` when given |
| Not static | total joint motion above a minimum; the clip is not a frozen pose |
| Joint limits | joint angles within plausible ranges for the role (no limb folded through itself) |
| Spikes and jerk | rotation steps above 60 degrees per frame (the retargeter's own spike metric), jerk above a limit |
| Foot contact | foot sliding on the ground during contact, ground penetration, floating (the retargeter's skate, penetration and reach metrics) |
| Root path | travel speed within the range for the action class, scaled by body size |
| Loop closure | for `expects_loop`: pose and velocity at the end match the start within a tolerance |
| Retrieval check | embed the motion with the project's motion model (`motion_sem`) and report R@1 / R@5 / R@10 of its own prompt among all prompts. Cheap and objective, but the model was trained on the same corpus, so treat it as a secondary signal, never as the headline |

## Judge input (step 6)

Video if the judge accepts it, otherwise a 12-frame filmstrip (frame index and time printed on every frame). A still-frame judge
cannot see smoothness between frames, which is why the checks above measure it directly.

## Controls

- **Retrieval baseline:** the nearest existing clip for the prompt (`lottie_search.py` for Lottie, the 3D clip search for 3D), through the same panel path.
- **Reference level:** real dataset clips with their own captions as prompts, through the same panel path. The gap between these and
  the raw clips is the error of the panel path itself.
- **Negative controls the scores must punish:**
  - mismatched prompt: the motion paired with a random other prompt; `action` and `order` must drop
  - frozen: the first frame repeated; `action` and `naturalness` must drop
  - scrambled: the same frames in random order; `smoothness` and `naturalness` must drop

## Slices that matter

action class x applicable body x difficulty. One slice is special: `adaptation_test = true`, an action that exists in the data
only for two-leg bodies, asked for on a four-leg rig (for example a backflip). It tests the two-leg -> four-leg adaptation that the
project is built around, and four-leg clips are only about 1% of the Lottie data, so it must be reported by itself.
