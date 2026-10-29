# Marionette

Dissertation project, two papers (the plan is `docs/pipeline.html`: every step with the plan's wording and its status, and the original drawing at the bottom; the SVG file is `archive/plan_svg_2026-10-29/dissertation_pipeline_en.svg`):

- **Paper 1.** Vector rigs from Lottie and 3D → 2.5D motion transfer. Single character. Data: ~70k Lottie animations and UniML3D. Steps 1.1 to 1.10.
- **Paper 2.** Text-to-character and scene generation on device. Builds on the dataset, the motion model and the assembly of Paper 1 (1.4, 1.7, 1.8). Steps 2.1 to 2.8.

Last saved: 2026-10-29. On that day the project was cut down to this pipeline: everything that is not a step of the plan was moved to `archive/pipeline_svg_2026-10-29/`
(see **Removed**), and the UniMate checkpoint for step 1.7 was downloaded. The old README (state of 2026-10-28, with all the 3D retargeting, embedding and generator history)
is `archive/pipeline_svg_2026-10-29/snapshots/README.md`.

## Layout

```
Marionette/
  README.md, CLAUDE.md, environment.yml (conda env "Mario"), requirements.txt (pip env of the Lottie parser), .gitignore
  notes/      your own notes (not written by the scripts)
  docs/       pipeline.html (the plan with status and its drawing), DATASET.md, LOTTIE_PARSE.md, SKELETON_SPEC.md, SKELETON_CLASSES.md, skeleton_spec/, figures/ (Lottie figures)
  scripts/    pipeline code, by group (below)
  lottie/     Lottie curve / primitive / quantizer classes used by the Lottie scripts (copied from Loomie; nothing imports ../Loomie)
  viewer/     web viewers and the one server, serve.py
  eval/       evaluation plan: shape (mesh + skeleton) and motion, each scored by deterministic checks, a VLM judge and human raters (outputs go to data/eval/)
  archive/    everything moved out of the pipeline, by date, same relative paths
  data/       everything generated, about 11 GB (untracked)
  sources/    symlinks to ../Datasets and ../Models (read-only inputs)
```

### scripts/ by step of the plan

Run as `python scripts/<group>/<name>.py` from the project root.

| Step | Scripts |
|---|---|
| 1.1 Lottie parsing | `lottie2d/parse_lottie.py` (+ `run_parse.sh`), `lottie_tokens.py` (2D forward kinematics), `analyze_lottie.py` |
| 1.2 Character / background split | `lottie2d/vlm_tables.py`, `vlm_compare.py` (Qwen3-VL labels), `filter_side_view_animals.py` |
| 1.3 Rig extraction | `lottie2d/probe_lottie_rigs.py`, `lottie_tokens.py` |
| 1.5 Joint labeling | (not built) the part vocabulary is `rigging/skeleton_spec.py` |
| 1.6 UniML3D preprocessing | `dataset/{inventory, filter_dedupe, normalize, convert_clips, make_splits}.py`, `motion/motion_coverage.py` |
| 1.8 Lottie assembly | `lottie2d/{svg_to_mesh, build_mesh, rig_mesh, bake_lottie}.py` |
| 1.9 Benchmark, 2.8 Evaluation | `eval/` (protocol), `tools/make_demo_rating_set.py`, `viewer/rate.html` |
| 2.1 Timeline data | `lottie2d/lottie_actions.py` (action phrases from captions; the rest of the step is not built) |
| 2.2 Planner | `rigging/{skeleton_spec, skeleton_synth, skeleton_classes}.py` (skeleton format, synthetic skeletons) |
| support | `lottie2d/lottie_filmstrips.py` (contact sheets), `tools/verify_copy.py` |

Steps 1.4, 1.7, 1.10, 2.3 to 2.7 have no code yet. `parse_lottie`, `vlm_tables` and `run_parse.sh` also run from the GPU server's flat `scripts/` folder: they find the root by layout.

### viewer/

`python viewer/serve.py` opens the menu `home.html` (a card and manual per page): `index.html` (3D character browser with the original clips of all classes), `vlm_table.html` (Qwen labels),
`lottie_raw.html` (raw dataset), `lottie_tokens.html` (joints and tokens), `motion_classes.html` (clips per body class and movement mode), `skeletons.html` (real and synthetic skeletons, roles, derived class),
`rate.html` (human rating tool of `eval/`). `lottie.min.js` is the lottie-web player.

### data/

```
data/characters/ 3,570 normalised characters (model.glb, normalization.json, meta.json, anims/)     data/clips/  shared Mixamo clips
data/inventory/  clips.csv, rigs.csv, report.md, summary.json (UniML3D inventory)                    data/splits/ character-level split
data/vlm/        Qwen labels of LottieAnimation-660K (68 shards, from the server)                    data/lottie/ motion parse (68 shards)
data/analysis/   summaries, candidates.csv, skeleton_classes/, motion_coverage.md, lottie_browser_index.json
data/synthetic/  synthetic skeletons     data/eval/  rating sets     data/lottie_tokens_demo/  viewer samples     data/archive_44shards/  the first Lottie results
```

On 2026-10-29 (with your OK) the folders that only archived code used were moved to `archive/pipeline_svg_2026-10-29/data/` (same relative paths): `mobile`, `shape`, `decoded`, `logs`, `joint_maps`, `retarget_report.csv`, and 39 of the 42 entries of `data/embeddings`
(CLIP text, the motion and semantic models, diffusion models, paraphrase and rewrite files; 2.6 GB in all, `data/` is now 8.3 GB).
What is still in `data/` is read by live code (grep of `scripts/ viewer/ lottie/`): `data/embeddings/` with only `lottie_feats_index.csv` and `lottie_feats.npz` (written by `lottie_tokens.py build`; read by `motion_coverage.py`, hence the Motion classes page, and `lottie_actions.py`)
and `lottie_actions.csv` (written by `lottie_actions.py`), `text_frontend.json` (`lottie_actions.py`), `retarget_targets.csv` (`make_splits.py`, field `retarget_target`), `anymate/` (the skeleton tools and viewer read its output files), `characters`, `clips`, `inventory`, `manifest.csv`, `splits`, `vlm`, `lottie`, `analysis`, `synthetic`, `eval`.

### Outside this folder (read-only inputs)

- `../Datasets`: the UniML3D export, LottieAnimation-660K, LottieSVG-10M and -small, template rigs, Anymate and AnyTop files.
- `../Models`: Qwen3-VL 2B and 8B (step 1.2), Sam (SAM 2, 2.1, 3; step 1.2), **UniMate** (step 1.7, below). 
  Missing for the plan: GroundingDINO (1.2), a Qwen 0.5B (2.2), LPIPS weights (1.4).
- Moved on 2026-10-29 to `../archive_2026-10-29/` (a sibling of `Models`; move back to restore): `Anymate/` and `Anytop/` (cloned repos of the old pilots) and `Models/{InternVL3_5-2B, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct}` (13 GB; Anymate 1.8 GB).

#### UniMate (step 1.7), downloaded 2026-10-29

`../Models/UniMate/unimate_uniml3d_f60_v3/` from `huggingface.co/Linzhan/UniMate`: `checkpoints/checkpoint_step_{100000..150000}.pt` (six files, 1,185,836,040 bytes each, equal to the sizes in the hub listing),
`config.json`, `dataset_stats.npy`; plus the repo's `README.md` and `LICENSE` in `../Models/UniMate/`. It is the model the hub README marks recommended: graph attention with AdaLN text, 74.1M denoiser, trained on UniML3D
(150k steps, 8 H100, about 35 h), skeletons of 5 to 70 joints, 60 frames at 30 fps, per joint position (3), rotation relative to the T-pose (6) and velocity (3); text through a frozen `google/flan-t5-base`.
Use step 150000 (the hub says to use each model's latest checkpoint). License **CC-BY-NC-4.0**.
Not checked: that the files load (only names and sizes were compared). Not downloaded: the code (`github.com/Friedrich-M/UniMate`) and `google/flan-t5-base`, both needed to run it.

## Environment

```
conda env create -f environment.yml   # once; creates the env "Mario"
conda activate Mario
python viewer/serve.py                # menu at http://127.0.0.1:8765/viewer/home.html (--no-open to skip the browser; Ctrl-C to stop)
.claude/gate.sh full                  # syntax check of every .py + the self-tests of skeleton_spec and skeleton_synth
```

PyTorch comes from conda-forge on purpose (the pip wheel aborts with "OMP: Error #15"). `build_mesh.py` and `svg_to_mesh.py` need `triangle` (listed). Stop a leftover server with `pkill -f viewer/serve.py`.
macOS `._*` files appear everywhere on the external drive (macFUSE recreates them); the scripts skip them and `.gitignore` lists them.

## Status

Per step; the same table, with facts and open points, is `docs/pipeline.html`. Numbers are from runs recorded before 2026-10-29 (the removed 3D and embedding work is not repeated here).

| Step | State |
|---|---|
| 1.1 Lottie parsing | working: parse of 68/68 shards (17.4% rows use expressions, 49.8% have a fitted rotation track), 2D FK that nests precomps; no time remap |
| 1.2 Character / background split | pilot: Qwen3-VL labels (91,724 relevant of 142,412 looked at), side-view animal filter; no DINO + SAM grounding |
| 1.3 Rig extraction | pilot: layer = joint tokens (69,585 clips) and a probe (about 23% rig-like); no co-motion inference or bone-chain fit |
| 1.4 Reconstruction check | not built |
| 1.5 Joint labeling | not built |
| 1.6 UniML3D preprocessing | working for the 3D half (12,672 clips read, 7,225 converted, 3,570 characters, split 3,213 / 357); no 2D projection, view choice or out-of-plane filter |
| 1.7 2.5D motion model | not built; UniMate weights downloaded |
| 1.8 Lottie assembly | pilot: mesh → weights → baked Lottie ran once on one bear artwork with a hand-placed rig; no yaw flip, depth sort, keyframe reduction or easing fit |
| 1.9 Benchmark | protocol and rating page only |
| 1.10 Ablations | not started |
| 2.1 to 2.8 | not built; formats only: skeleton spec (`docs/SKELETON_SPEC.md`), synthetic skeletons, evaluation protocol |

### Checked on 2026-10-29 (this cleanup)

- `.claude/gate.sh full`: pass (syntax of every `.py`, `skeleton_spec.py --self-test`, `skeleton_synth.py --self-test`).
- A scan of the imports of every live script found none that import an archived module.
- `viewer/serve.py --no-open` on port 8765 (free before): 200 for every page of the menu, the README (the plan's SVG, then also served, has since been archived) and the docs linked from `home.html`, and for `/api/characters` (3,932), `/api/motion_classes`, `/api/skel_*` (including an Anymate-derived skeleton), `/api/rate_sets`, `/api/lottie_raw`; `/api/vlm_table` answered 200 with 40 bytes (it loads its table in the background on first use; its content was not checked). The server was stopped afterwards.
- `docs/pipeline.html` rendered in Chrome (headless): 18 boxes, no console errors, the manual opens from the address hash.
- The UniMate download: names and sizes against the hub listing. Not checked: loading the checkpoints.
- After the `data/` moves (second step): the gate, every page and the same API endpoints again gave the same results; `lottie_tokens.py selftest` passed; `data/embeddings` kept the three Lottie files and `motion_coverage` still serves.
- **One unintended write:** I ran `python scripts/lottie2d/lottie_actions.py -h` as a check; that script has no argument parser, so it rebuilt `data/embeddings/lottie_actions.csv` (rule-based, from the same inputs; the file has the same size as before, 3,369,063 bytes, but I had no backup and did not compare contents). It is the only file under `data/` written by the cleanup, apart from the moves. The `-h` commands now shown in `docs/pipeline.html` were checked: the others use argparse; `lottie_actions.py` is shown as `show 10`.
- Not run: `normalize.py`, `convert_clips.py`, any GPU job.

## Next steps

1. **1.7:** clone the UniMate code, fetch `google/flan-t5-base`, load `checkpoint_step_150000.pt` and generate one clip for a UniML3D skeleton (this also checks that the files load and that the output matches the format above).
2. **1.6:** write the 2D projection and view choice for the converted clips; decide how the winged, limbless and other classes get converted (the character browser shows them only as skeletons).
3. **1.3 / 1.4:** co-motion rule for layers without a parent, then the Lottie rebuild with per-frame LPIPS and the tiers A, B, C.
4. **1.2 / 1.5:** layer-level grounding with DINO + SAM (DINO weights are missing), then Qwen-VL part labelling with a highlighted part.
5. **1.8:** make 2D rig data (`data/side_view_animals/` does not exist) and add yaw flip, depth sorting, keyframe reduction and easing fit; take the motion from 1.7.
6. Align `eval/` with the metrics of step 1.9 (it was written for the earlier text-to-Lottie idea). It still cites archived code: `eval/shape/README.md` line 29 (the `anymate_roles` logic as a role check), `eval/motion/README.md` lines 36-37 (spike and skate metrics of the retargeter). Also `viewer/skeletons.html` and its `home.html` card cite `anymate_roles.py`, and `bake_lottie.py`'s usage lines mention the archived `motion_to_2d.py`.

## Removed

Moved, not deleted, with the same relative paths so any file can be put back. The 2026-10-28 steps (motion generator, Retarget compare page, motion model training code, Filmstrip and Portraits pages, non-original clips)
are in the other folders of `archive/`, with the detailed list in the snapshot README.

### 2026-10-29: everything outside the pipeline of the work plan → `archive/pipeline_svg_2026-10-29/`

- `scripts/rigging/{retarget, anymate_predict, anymate_roles, anymate_rig}.py`: the plan has no rule-based 3D retargeter and no 3D auto-rigging. (The "3D transfer" of ablation 1.10 is UniML3D training of UniMate, not retargeting.)
- `scripts/captioning/{qwen_caption, qwen_mesh_pilot, qwen_check_pilot}.py`, `scripts/motion/render_motion.py`, `scripts/text/embed.py`, `scripts/dataset/template_clips.py`: 3D motion-caption pilots and CLIP caption retrieval (UniMate uses flan-t5).
- `scripts/generate/shape_retrieval.py`, `scripts/tools/clean_generated.py` (had nothing left to clean).
- `docs/{RETARGET, EMBED, MOBILE}.md`, `docs/embed_pipeline.svg` and every figure in `docs/figures/` except the four `lottie_*` ones.
- `snapshots/`: README, CLAUDE.md, environment.yml, pipeline.html, DATASET.md, serve.py, home.html, gate.sh, test_guard.sh, quick SKILL.md as they were.
- Edited in place: `.claude/gate.sh` (the retargeter self-test is gone), `environment.yml` (onnx and Anymate packages dropped; this uninstalls nothing), `viewer/home.html` (documents list and intro), `docs/DATASET.md`, `docs/pipeline.html` (rewritten on the SVG's steps), `.claude/skills/quick/SKILL.md`, README and CLAUDE.md.
- Kept although they are not boxes of the plan: `viewer/index.html` (the only way to look at the UniML3D characters and clips for step 1.6), the skeleton tools (the format that steps 1.5 and 2.2 need; they read `data/anymate/` output files, which are data, not the archived scripts), `lottie_filmstrips.py`, `verify_copy.py`.
- Second step, same day, on your OK: the leftover `data/` folders and files listed under **data/** were moved to `archive/pipeline_svg_2026-10-29/data/`; `Anymate`, `Anytop` and three models went to `../archive_2026-10-29/`; the empty `scripts/generate/` and `scripts/text/` (only stale `__pycache__`) were deleted.
- Not touched: `sources/`, `notes/`, `eval/`, `../Datasets`, and the rest of `../Models` (Qwen3-VL 2B and 8B, Sam, UniMate).

### 2026-10-29, third step: the plan's SVG → `archive/plan_svg_2026-10-29/dissertation_pipeline_en.svg`

On your request. `docs/pipeline.html` now carries everything the SVG said: the plan's exact wording of every step ("In the plan" in each manual), the paper headings and the two output boxes, and the drawing itself (embedded as an image, minus the file's embedded provenance metadata block). The file was moved, not deleted, so it can be put back to `docs/`. References in README, CLAUDE.md, DATASET.md and `viewer/home.html` now point at `docs/pipeline.html`.
