# Evaluation of the finished text -> 2D animation system (SVG / Lottie style)

Written 2026-10-28, before the generator exists. Nothing here runs yet. The system makes two things from text, and each is
tested on its own, because they fail in different ways and one bad character would otherwise look like bad motion:

| Test | Question | Prompt describes | System output | Judged on |
|---|---|---|---|---|
| [`shape/`](shape/README.md) | How good is the generated mesh and skeleton? | the body only ("a stocky goblin with long arms and short legs") | rest-pose Lottie character + skeleton + skin weights | still images (rest pose, skeleton overlay, stress poses) |
| [`motion/`](motion/README.md) | How good is the generated motion? | the motion only ("walks forward slowly, stops and looks left") | joint rotation tracks + root path | video or filmstrip of the motion on fixed reference rigs |

Each test has **two evaluations of the same outputs: a human one and a model one** ([`human_eval.md`](human_eval.md)). The model
judge scores every prompt; humans score a stratified sample and are the ground truth. The report puts them side by side.

This split follows the project's own convention (shape captions have no motion, motion captions have no body; see the plan in
`notes/` and `docs/DATASET.md`). A third test, end to end ("a fox trots forward" -> animation), comes later and reuses both.

Language: English only for the first stage.

## Shared rules (they apply to both tests)

1. **Checks first, judge second.** Deterministic checks are cheap and objective; an output that fails one is reported as failed
   and is not sent to the judge.
2. **The judge is not the labeler.** Qwen3-VL labelled the corpus the generator trains on, so it must not be the only judge.
3. **Humans are the ground truth; the model judge is validated against them.** Humans rate a stratified sample per test (500
   target, 300 minimum), blind to the judge, with the same axes (protocol in `human_eval.md`). Report per-axis Spearman
   correlation, mean absolute difference and the share of items where judge and human differ by 2 or more points, plus
   human-vs-human agreement. An axis with weak agreement is reported as "not validated", never dropped silently.
4. **Choose the judge by a bake-off, not by reputation.** Run 2-3 candidate models on the same human-rated items and pick, per
   test, the one that agrees best. Requirements: open weights, runs on the 2 x RTX 3090 server (24 GB each), not the labeler,
   and for `motion/` it must accept video (or be shown to work on a filmstrip). Locally available: Qwen3-VL 2B / 8B (the
   labeler: use only as a comparison, flagged), InternVL3.5-2B (probably too small). Newer open video-capable models would need
   downloading; which ones are current has not been checked yet.
5. **Held-out prompts.** No prompt may be a near-duplicate of a training caption (compare against the captions in `data/vlm/`
   and the 3D captions, by n-gram overlap and by embedding distance). `split` is `dev` (tune the judge prompt and thresholds) or
   `test` (touched once per system version).
6. **Controls in every report** (specific ones are listed in each test): a retrieval baseline, real dataset items as the
   reference level, and negative controls the judge must score low. If it does not, that axis cannot be used.
7. **Report per slice**, with at least 300 prompts per reported slice where possible; smaller slices are shown but marked.
   Four-leg characters are about 1% of the training data and the overall mean would hide how they do.
8. **Record everything**: system version, seed, judge model, judge prompt version, prompt file hash. Judge at temperature 0.

Generated outputs go to `data/eval/<test>/<run>/` (untracked, like the rest of `data/`). These folders hold only protocol,
prompt files and rubrics.

## Open decisions

- **How the generator is called** (command, input text, output files and format) so each test's "generate" step is a thin
  wrapper. To be decided by you; until then the retrieval baseline (`scripts/lottie2d/lottie_search.py`) is the stand-in
  system, so the harness and the judge can be built and calibrated first.
- **Which judge models go into the bake-off** (rule 4).
- **How many prompts per test.** "10k" can be 10k in total or per test; motion also multiplies by the number of reference rigs.
- The overall "good / bad" rule of each test is a draft in its `rubric.md`; it is to be fitted to the human ratings.
