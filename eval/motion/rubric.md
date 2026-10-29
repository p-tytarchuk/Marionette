# Rubric, test B: motion (version v0, a draft). Used by both the model judge and the human raters.

The rater sees the prompt and the motion on a reference rig, as video or a 12-frame filmstrip. Scores are integers from 1 to 5.
Rate only the motion, not how attractive the character is (the rig is the same for every system).

| Axis | Question | 1 | 3 | 5 |
|---|---|---|---|---|
| `action` | Does the motion show the requested action (verb, direction, speed words)? | wrong or no action | related action, wrong style | the requested action, clearly readable |
| `order` | For prompts with several steps: are all steps there, in the right order? `null` for single-step prompts | steps missing or reversed | all steps, order or timing off | all steps, right order and timing |
| `body_use` | Do the right body parts move (the named limb, the legs for locomotion, the head for "looks left")? | wrong parts move | partly right | exactly the right parts |
| `naturalness` | Is it plausible for this body (gait, weight, balance, anticipation, follow-through)? | broken or random | recognisable but stiff or odd | natural for this body |
| `smoothness` | Is it free of popping, jitter, snapping and sudden direction changes? | constant artefacts | occasional artefacts | clean |
| `loop` | Only when the prompt expects a loop: does the end lead into the start? `null` otherwise | visible jump | small pop | seamless |

## Judge output (JSON only)

```json
{"id": "m00001", "rig": "<panel rig id>", "judge": "<model name>", "judge_prompt_version": "v0",
 "scores": {"action": 4, "order": null, "body_use": 4, "naturalness": 3, "smoothness": 3, "loop": 4},
 "reasons": {"action": "one line", "body_use": "one line", "naturalness": "one line", "smoothness": "one line"},
 "cannot_judge": false}
```

`cannot_judge: true` (video missing or unreadable) is counted and reported, never scored as low.

## Judge prompt template

```
You are checking an AI-generated animation of a character against the text description of the motion it should show.
Motion description: "{prompt}"
The input is {a video | 12 frames, left to right, top to bottom, at times {times}} of the same character performing the motion.
Score each axis from 1 (bad) to 5 (excellent) with the definitions below, judging only the motion that is visible.
Ignore how the character looks. Return JSON only.
{axes definitions from the table above}
```

## Overall "good / bad" (proposal, to be fitted to the human ratings)

good = every deterministic check passed and `action` >= 4 and `body_use` >= 3 and `naturalness` >= 3 and `smoothness` >= 3
(and `order` >= 3 when it is not null). Report the share of good per slice and also the per-axis means.
