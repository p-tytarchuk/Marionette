# Judge rubric, test A: mesh and skeleton (version v0, a draft: tune on `dev` against human ratings, then freeze)

The judge sees the prompt and five images of one character: rest pose, rest pose with the skeleton drawn on, and three stress
poses. Scores are integers from 1 to 5.

| Axis | Question | 1 | 3 | 5 |
|---|---|---|---|---|
| `subject` | Does the character match the described subject and attributes (species, build, features)? | wrong subject | right kind, details wrong | matches the description |
| `anatomy` | Are the body parts present, in the right number and in plausible proportion (head, torso, limbs, tail)? | parts missing or duplicated | present, odd proportions | plausible for this creature |
| `skeleton` | Are the joints where the real joints are (shoulder, elbow, knee, hip, neck) and is the bone structure sensible? | joints off the body or random | roughly right, some misplaced | every joint at the right place |
| `mesh` | Is the artwork clean and separable for rigging (parts distinct, no fused limbs, clean silhouette)? | one blob or broken shapes | usable with problems | clean, separate parts |
| `deform` | In the stress poses, does the body bend without tearing, stretching or collapsing? | parts detach or collapse | small distortions at some joints | bends cleanly at every joint |

## Output (JSON only)

```json
{"id": "s00001", "judge": "<model name>", "judge_prompt_version": "v0",
 "scores": {"subject": 4, "anatomy": 3, "skeleton": 3, "mesh": 4, "deform": 2},
 "reasons": {"subject": "one line", "anatomy": "one line", "skeleton": "one line", "mesh": "one line", "deform": "one line"},
 "cannot_judge": false}
```

`cannot_judge: true` (images missing or unreadable) is counted and reported, never scored as low.

## Judge prompt template

```
You are checking an AI-generated 2D character and its skeleton against the text description it was made from.
Description: "{prompt}"
Image 1: the character at rest. Image 2: the same with the skeleton drawn on it (dots = joints, lines = bones).
Images 3-5: the character bent into three test poses.
Score each axis from 1 (bad) to 5 (excellent) with the definitions below, judging only what is visible.
Do not reward the drawing for being attractive if it does not match the description. Return JSON only.
{axes definitions from the table above}
```

## Overall "good / bad" (proposal, to be fitted to the human ratings)

good = every deterministic check passed and `subject` >= 4 and `anatomy` >= 3 and `skeleton` >= 3 and `deform` >= 3.
Report the share of good per slice and also the per-axis means: one pass rule hides which axis fails.

## Calibration set (about 300 items, rated by you)

Stratified over slices and over the judge's own scores (clear good, clear bad, borderline). Same axes, same images, blind to the
judge. Report per axis: Spearman correlation, mean absolute difference, share of items differing by 2 or more points.
