# Test A: generated mesh and skeleton

How good or bad is the character the system makes from a body description? The test looks only at the character in its rest
pose and at how it deforms under fixed poses. Motion is not involved, so a failure here is a shape failure.

## Pipeline

```
prompts/prompts_shape.jsonl                        1  body-only prompts + metadata (prompts/schema.json)       you write these
data/eval/shape/<run>/character/<id>/              2  generate: rest-pose Lottie + skeleton + skin weights    system under test
.../checks.jsonl                                   3  deterministic checks (below)
.../images/<id>_{rest,skeleton,pose1..3}.png       4  rest render, render with the skeleton drawn on it, three stress poses
.../judge.jsonl                                    5  model evaluation: VLM judge, scores per axis (rubric.md)
.../human.jsonl                                    6  human evaluation of the sample (../human_eval.md)
.../report.md + failures/                          7  per-slice results, judge-vs-human agreement, failure gallery
```

Step 2 needs an adapter that reads the character's joints, parent links and skin weights from whatever form the generator
emits (open decision in `../README.md`). Steps 3-7 only use that adapter's output.

## 3. Deterministic checks (pass / fail, with the measured value)

| Check | Pass rule (thresholds to be set on `dev`) |
|---|---|
| Renders | rest pose loads in lottie-web without console errors, non-blank, inside the canvas |
| Skeleton is a tree | one root, no cycles, no orphan joints; joint count inside the range for the class (the Lottie corpus median is 10 joints, 3D rigs about 49) |
| Bones | every bone has positive length; no two joints at the same position |
| Joints inside the body | each joint lies inside, or within a tolerance of, the character's silhouette |
| Roles found | the project's role finder (`anymate_roles` logic: hips, head, spine, legs, arms from tree geometry) labels the rig; coverage is reported |
| Limb count | biped: 2 legs; quadruped: 4 legs; compared with `expected_legs` when the prompt gives it |
| Skin weights | every shape or vertex is bound, weights sum to 1, every joint influences something |
| Symmetry | left / right limb pairs mirror each other within a tolerance (bilateral characters only) |
| Stress poses | pose the rig with a fixed set (each limb joint +-30 degrees, plus one standard walk clip) and measure: maximum edge strain, number of detached parts (connected components before vs after), self-overlap area. Fails if parts tear off or stretch past the limit |

The stress poses are the most informative check: they test "does it deform" without any learned motion, using the same fixed
poses for every system.

## 5. Judge (static images, so almost any good open VLM can do this)

Input: the prompt text, the rest render, the render with the skeleton drawn on it (joints as dots, bones as lines), and the three
stress-pose renders. Output: JSON scores per axis (see `rubric.md`).

## Controls

- **Retrieval baseline:** the nearest existing character for the prompt (SVG artwork rigged by `rig_mesh.py`, or a 3D rig
  drawn flat), scored the same way.
- **Reference level:** real artwork rigged by the existing chain (`svg_to_mesh` -> `rig_mesh`), with the prompt written by a human.
- **Mismatched prompt:** each character paired with a random other prompt. `subject` and `anatomy` must score these low.
- **Broken-rig control:** a good character with its joints shuffled or its weights randomised. `skeleton` and `deform` must
  score these low.
