# Marionette

Dissertation project: generating animatable characters (rest-pose mesh, skeleton, skin weights)
and motion for biped and quadruped classes.

Last saved: 2026-10-27, later that day: the project folder was cleaned and the scripts regrouped into folders (see **Session 2026-10-27: folder layout** under Status). Before that (all Lottie outputs rebuilt with the 68 finished Qwen shards: see **Update 2026-10-27** in the Lottie branch). The 2026-10-26 session added the Anymate rigging pilot, the Lottie text-to-motion branch (labels, motion parse, joint tokens, a joint 3D+2D model, a search page), AnyTop notes and a viewer menu: see **Session 2026-10-26** under Status and the matching Next steps. Everything older is the state of 2026-10-22; see **Status** for what is done, what is only partly tested, and what comes next.

## Layout

```
Marionette/
  README.md, environment.yml (conda env "Mario"), requirements.txt (pip env of the Lottie parser), .gitignore
  notes/      your own notes: todo.txt, "Text → Rigged, Animated Creature Pipeline Plan.{md,pdf}" (not written by the scripts)
  docs/       written records and figures (below)
  scripts/    all pipeline code, in 8 folders (below)
  lottie/     Lottie curve / primitive / quantizer classes used by the Lottie scripts (copied from Loomie)
  viewer/     web viewers and the one server, serve.py
  eval/       plan for testing the finished text -> Lottie system: shape (mesh + skeleton) and motion, each judged by humans and by a model (outputs go to data/eval/)
  data/       everything generated, about 11 GB (below)
  sources/    symlinks to ../Datasets and ../Models (read-only inputs)
```

### docs/

```
docs/pipeline.html   the pipeline as a map: every step of the three branches, its status and a manual per step (open via viewer/serve.py, or as a file)
docs/DATASET.md      what data exists, its gaps, and the target format (start here)
docs/RETARGET.md     the rule-based retargeter: method, parameters, metrics, results, limitations
docs/EMBED.md        caption + motion embeddings, the generator and every experiment around it (the detailed record; start at its section list); diagram in docs/embed_pipeline.svg
docs/MOBILE.md       running the generator on a phone: audit and measurements
docs/LOTTIE_PARSE.md the Lottie motion parse (written for the GPU server; its commands keep the server's flat scripts/ layout)
docs/SKELETON_CLASSES.md what a 2D skeleton alone can say about its body type (biped / quadruped / limbless / winged): method, results, caveats
docs/SKELETON_SPEC.md the role-labelled 2D skeleton format the shape stage should emit (parts: trunk, head, tail, leg, arm, wing, fin, extra); class and mode support are counted from it; schema and examples in docs/skeleton_spec/
docs/figures/        stick-figure sheets, backflip filmstrips (backflip_*.png), decoded-latent figures, Anymate and Lottie figures
```

### scripts/

Run as `python scripts/<folder>/<name>.py` from the project root. Each script finds the project root itself and puts every folder
below on `sys.path`, so scripts import each other by module name (`import retarget`) whichever folder they are in. Script names are unique.

```
scripts/dataset/     inventory, filter_dedupe, normalize, convert_clips, make_splits (splits + meta.json), template_clips
scripts/rigging/     retarget (the rule-based retargeter), anymate_predict / anymate_roles / anymate_rig (auto-rigging pilot), skeleton_classes (skeleton -> biped / quadruped / limbless / winged, exploratory), skeleton_spec (the role-labelled skeleton format: validate, derive class and movement-mode support), skeleton_synth (synthetic role-labelled skeletons of every class, any yaw)
scripts/motion/      motion_features, motion_ae, motion_sem, motion_sem_features, motion_sem_joint (3D + Lottie model, seeds, summary), decode_latents, render_motion,
                     diffusion_test, speed_sampler, walk_check, gait_check, motion_coverage (clips per body class and movement mode, from the captions)
scripts/text/        embed (CLIP captions), paraphrase, rewrite_prompt, text_frontend, wording_eval
scripts/generate/    shape_retrieval (the only script left here; the motion generator was removed on 2026-10-28, see "Removed" below)
scripts/captioning/  qwen_caption, qwen_mesh_pilot, qwen_check_pilot (pilots; none of these changes the dataset captions)
scripts/lottie2d/    the Lottie branch: parse_lottie, vlm_tables (+ vlm_compare, run_parse.sh), analyze_lottie, probe_lottie_rigs, lottie_filmstrips, lottie_tokens,
                     lottie_actions, lottie_search, filter_side_view_animals; 3D motion -> 2D rig -> Lottie: svg_to_mesh, build_mesh, rig_mesh, motion_to_2d, bake_lottie
scripts/tools/       clean_generated (removes generated clips, dry run by default), verify_copy, make_embed_diagram
```

parse_lottie, vlm_tables and run_parse.sh also run from the server's flat `scripts/` folder: they find the root by layout.

### viewer/

viewer/              local 3D viewer: home.html (the menu, opens when serve.py starts; a card and an "i" manual for every page below), index.html (character browser with clip playback), compare.html, strip.html (fixed-camera filmstrips of clips, for figures), portrait.html (rest-pose pictures); serve.py serves them
Pages added later: vlm_table.html (Qwen labels), lottie_raw.html (raw dataset browser), lottie_tokens.html (joints and tokens over the animation), lottie_search.html (retrieval), rate.html (human rating tool of the evaluation in `eval/`; `scripts/tools/make_demo_rating_set.py` writes demo sets), skeletons.html (real and synthetic skeletons side by side at any yaw, with roles, derived class and movement modes); compare.html also lists the Anymate rigs. lottie.min.js is the lottie-web player (copied from Loomie).

### data/

```
data/characters/<source>_<id>/   model.glb, normalization.json, meta.json, anims/ (+ anims.json)   3,570 normalized characters (5.2 GB)
data/clips/mixamo/               2,308 shared Mixamo clips (22-joint core)
data/inventory/                  clips.csv, rigs.csv, report.md, summary.json: full inventory of the UniML3D export
data/embeddings/                 CLIP text, motion embeddings and features, joint models, search indexes (see docs/EMBED.md; 3.4 GB)
data/decoded/                    decoded latents as human-skeleton clips (decoded_to_character.py) and gen_*.glb from the generator page
data/splits/                     train.txt, val.txt, summary.json: character-level split, 3,213 / 357 (scripts/dataset/make_splits.py)
data/joint_maps/                 only mixamo_biped.json (see Next steps)
data/shape/                      shape retrieval: portraits and CLIP image embeddings of the characters
data/mobile/                     ONNX files of the removed on-device generator (kept, not used)
data/anymate/<character>/        predicted joints, roles, skins, rig/rig.glb, retargeted clips (Anymate pilot)
data/vlm/                        Qwen labels of LottieAnimation-660K (68 shards, copied from the server)
data/lottie/animation/           Lottie motion parse (68 shards)
data/analysis/                   summaries, candidates.csv, cross_domain_*.json, lottie_browser_index.json
data/lottie_tokens_demo/         viewer samples for lottie_tokens.html
data/archive_44shards/           the first Lottie results (44 shards), kept for comparison
data/logs/                       logs and result JSONs of the experiments (names say what: runs_*, walks_*, wording_eval_*, qwen_*, motion_sem_joint_*, ...)
data/side_view_animals/          2D rig data (mesh/, rig_spec/, rig/): written by the Lottie rig scripts; the folder does not exist yet
data/manifest.csv, normalize_report.csv, clips_report.csv, retarget_report.csv   per-step reports
data/retarget_targets.csv        which characters can be retarget targets
data/text_frontend.json          word lists of the rule-based prompt front end
```

The loose reports in `data/` are read or written by several scripts: do not move them.

### Outside this folder

Nothing imports `../Loomie` any more (its code used here is copied in). Still outside, as read-only inputs: `../Datasets` (the UniML3D export `dataset`, the template rigs `glb` / `glb_mixamo` / `unique`, LottieAnimation-660K,
LottieSVG-10M and -small, Anymate, AnyTop files), `../Models` (Qwen3-VL 2B and 8B, Qwen2.5 1.5B and 3B, InternVL3.5-2B, Sam), `../Anymate` (cloned repo + checkpoints), `../Anytop` (cloned repo).

## Environment

```
conda env create -f environment.yml   # once; creates the env "Mario"
conda activate Mario
python viewer/serve.py                # opens the menu http://127.0.0.1:8765/viewer/home.html (--no-open to skip); character browser http://127.0.0.1:8765/viewer/   (Ctrl-C to stop)
                                      # retarget compare   http://127.0.0.1:8765/viewer/compare.html
```

PyTorch comes from conda-forge on purpose: the pip wheel aborts with "OMP: Error #15" next to conda's numpy/scipy.
`scripts/lottie2d/build_mesh.py` and `svg_to_mesh.py` need `pip install triangle` (listed in environment.yml; installed in the current Mario env on 2026-10-27). The Anymate scripts need `omegaconf`, `diffusers`, `termcolor`, `opencv-python-headless`, `point-cloud-utils` and `easydict`, also listed in environment.yml (installed in the current Mario env on 2026-10-27).
Note: stop a leftover viewer server with `pkill -f viewer/serve.py`. macOS `._*` files appear everywhere on the external drive (macFUSE recreates them
if deleted); the scripts skip them and `.gitignore` lists them.

## Scripts (run in this order)

```
python3 scripts/dataset/inventory.py       # every clip NPZ + captions + quality lists -> data/inventory/
python3 scripts/dataset/filter_dedupe.py   # triangle filter + dedupe               -> data/manifest.csv
python3 scripts/dataset/normalize.py       # scale / axes / ground                  -> data/characters/*/model.glb
python3 scripts/dataset/convert_clips.py   # clips -> GLB on the characters         -> data/characters/*/anims/, data/clips/mixamo/
python3 scripts/rigging/retarget.py ...    # rule-based cross-skeleton retargeting  -> data/characters/<target>/anims/rt__*.glb
```

`normalize.py` and `convert_clips.py` take `--dry-run` (compute and verify, write nothing) and `-h` for the options.
`retarget.py`: `--list-targets`, `--query <words>`, `--source-clip <name> --target <character folder>`, `--self-test`, `--batch`.

## Removed on 2026-10-28: the motion generator and its viewers

Decision: start the generation part again from the beginning. Moved (not deleted) to `archive/motion_generate_2026-10-28/`, same relative paths, so any file can be put back:
`scripts/generate/{generate_motion, text_to_character, decoded_to_character, export_onnx, mobile_runtime, trace_generation, flip_sweep, step_cap_check}.py`, `scripts/motion/planar_test.py`,
`scripts/lottie2d/motion_to_2d.py` (its only input was the generator), `viewer/generate.html`, `viewer/realgen.html`; `snapshots/` holds `serve.py`, `compare.html`, `home.html`, README, CLAUDE.md,
`gate.sh`, `environment.yml`, `EMBED.md`, `MOBILE.md` and `pipeline.html` as they were before the edit. Edited in place: `viewer/serve.py` (no `/api/generate`, `/api/warm`, `/api/real_*`, `/api/gen_metrics`,
no preload; `--no-preload` is still accepted and ignored), `viewer/compare.html` (no gate filter or generated-clip rows), `viewer/home.html` (cards and text), `.claude/gate.sh` (the generator self-test is gone),
`scripts/motion/diffusion_test.py` (no `--characters`), `docs/pipeline.html` (the generator cells are grey), `environment.yml` (comment on the onnx packages).
Kept: the retargeter, `shape_retrieval.py`, `text_frontend.py`, `clean_generated.py` (removes old generated clips), the motion model training/analysis scripts in `scripts/motion/`, the rating page, the skeleton
tools, the Lottie branch, and everything under `data/` (the old generated clips and the model files are still there). `GET /api/targets` and `/api/retarget` stay for compare.html. The sections below that
describe the generator (the Generator page, flip analysis, trace, sweeps, mobile) are history: their commands no longer run unless the archived files are copied back.

## Status

### Done and verified

| Step | Result |
|---|---|
| Dataset audit | `Datasets/dataset` (UniML3D), `glb`, `glb_mixamo`, `unique`, `Models` analysed; see `docs/DATASET.md` |
| Inventory | all 12,672 clip NPZ read: 30 fps, no NaN, 3,446 unique skeletons (matches `unique/`), 7,286 clips usable |
| Filter + dedupe | 6,700 meshes -> 3,688 kept (<= 20k triangles, biped/quadruped, max 10 per skeleton) |
| Normalize | 3,570 characters written; Y-up, +Z forward, metres, feet at y = 0, centred. All 3,567 with a hip pair face +Z within 8 degrees |
| Clip conversion | 7,225 clips -> GLB on the character skeleton, read back and checked: worst joint error 4e-6 body heights; 61 skipped (root in another frame) |
| Viewer | character browser + clip picker with playback, checked in headless Chrome (Mixamo, Objaverse biped, quadruped) |
| Environment | conda env `Mario` created and import-tested |

Normalization conventions: quadrupeds are all scaled to 0.9 m (placeholder, the real size is not
recoverable), Objaverse bipeds to 1.7 m, Mixamo characters share one scale (median 1.75 m). The clip files
are not metric in the source data (Mixamo is centimetres; Objaverse rig heights span 0 to 168,345 units),
so every clip gets a per-rig scale. Each clip's start is moved onto the character's rest x/z (the removed
offset is stored as `start_offset_m`).

### Rule-based retargeter (`scripts/rigging/retarget.py`): working, still being tuned

Goal: play a human clip (e.g. a backflip) on a quadruped. Method, in short: transfer the hips' world-space
rotation change (a flip is a rotation about the left-right axis on any body), rotate about the trunk centre,
resample the spine by arc length, move limb segments by their swing relative to a neutral pose, copy source
foot contacts (hind: same side; fore: the opposite side, a diagonal gait), pin the feet with two-bone IK, and
move the body only as much as the ground and reach require. Only joints outside these roles (tail, ears,
fingers) are carried rigidly. For quadruped targets the body pitch is reshaped: a flip is completed in the air,
the tilt is soft-limited on the ground (35 degrees; 12 degrees when both feet are down) and the spine bend
is scaled (0.6).

Tests and numbers:

- **Identity test** (a clip retargeted onto its own skeleton): PASS, errors around 1e-8 (`--self-test`).
- **Backflip on the stand-in quadruped** (`objaverse_a734df8176ca47c68b0a90d1a17b5132`, a rabbit-like creature,
  not a crocodile; there is no crocodile in the data): no ground penetration, 0 reach failures, foot
  sliding under 0.01 m/s, 6 rotation spikes up to 80 degrees per frame at take-off, body lift up to 0.16 m.
  Frames checked in the viewer: crouch, inverted mid-air, landing crouch.
- **Sweep, 6 quadrupeds x 7 clips** (backflip, walking, jogging, a wide turn, a run with a reload, two jumps)
  in the middle of tuning: walking, jogging and turning are clean (mean foot sliding 0.01-0.17 m/s,
  almost no spikes). Backflip on the other five quadrupeds: reach failures 16-71 frames (was 158-216 before
  the stance-tilt limit), mean sliding under 0.02 m/s. Some targets need a root lift of 0.2-1.0 m to clear
  the floor while flipping, because their trunks are long (reported as `lift_max`, not hidden).
  The jog and jump numbers in that sweep predate the stance-tilt limit and have not been re-run.
- **Biped target** (human to Mixamo character): essentially perfect (no sliding, no reach failures).
- **Usable targets** (`data/retarget_targets.csv`): 2,745 bipeds (all 50 Mixamo) and **47 quadrupeds**.
  640 characters have no usable body labels (449 lack hips/head names, 144 lack legs, 47 head not below hips).

### Re-run and verified on 2026-10-20

- `--self-test`: PASS (identity errors <= 6e-8 of body height).
- Stand-in backflip rewritten after the last tuning: no penetration, 0 reach failures, skate max 0.004 m/s,
  lift 0.16 m, 6 rotation steps over 60 degrees (max 81 at frame 37).
- `retarget.py --batch Backflip,Walking,Jog_In_Circle,Jump_3 --batch-targets 6` works (24 runs, writes
  `data/retarget_report.csv`). First run, before the tuning below: medians lift 0.185 m, skate 0.034 m/s,
  23 reach-fail frames (worst 97), 0 spikes (worst 16).
- **Tuning of jog/jump on the long quadrupeds (2026-10-20).** Cause of the reach failures: every grounded frame
  lifted the body for the thumb joints (`Finger0`), whose raw pose hangs about 0.1 m below the wrist, so the
  legs were pulled out of reach (vertical gap up to 1.3x the limb length). Changes in `retarget.py`:
  (1) digits below a limb end are exempt from the pre-IK floor lift on grounded frames (the post-IK residual
  still checks them); (2) the clearance IK for a limb end uses a floor raised by the digits' rest drop and its
  weight eases in and out over 4 frames (a hard switch gave 60-180 degree rotation spikes). Result on the same
  24 runs: medians lift 0.128 m, skate 0.009 m/s, reach-fail frames 5, penetration 0.002 m, spikes 2.
  Per clip (max over the 6 targets): Walking fails 3, skate 0.013; Jump_3 fails 6, skate 0.15, penetration 0.04;
  Jog_In_Circle fails 50, skate 0.15; Backflip fails 52 (unchanged, long flips). Biped targets (32 runs):
  identical to before (0 reach failures, skate <= 0.017). Identity self-test still PASS.
- Tried and dropped: raising the lift floor of the end joint by the digit drop and leaving digits out of the lift
  (cut Jog fails to 7-23 on two targets but added 3-9 cm of thumb penetration); flattening the foot twice as fast
  (more spikes, no gain).
- **Hind-leg flips fixed (`..64c84fab`, 2026-10-20).** Cause: two-bone IK (`ik_limb`) wrote each limb joint's local
  rotation as if its parent were the previous labelled joint. This rig has an unlabelled joint (`joint19`, rest
  rotation about 134 degrees) between knee and ankle, so the foot came out about 180 degrees wrong, and flipped
  as the IK weight ramped in. `ik_limb` now goes through any joints between the chain members (`between()`).
  65 of the 2,742 usable Objaverse targets have such a chain. Jog_In_Circle on `..64c84fab`: spikes 19 -> 3,
  Walking 4 -> 0 (penetration 0.055 -> 0.014); biped batch and the identity self-test unchanged. Batch medians now:
  spikes 1.5 (worst 13), penetration 0.002 (worst 0.048).
- Viewer (headless Chromium via Playwright, no console errors): `rt__` clips are listed with a `⇄` on both a Mixamo
  target (Aciana, 23/23 tracks bound) and the quadruped stand-in (28/28), metrics line shown, playback runs.
  `serve.py` clip counts and `convert_clips.py` keeping `rt__` entries were not exercised beyond that
  (the viewer count for the stand-in shows 2 clips, as expected).
- Side effect: `data/characters/mixamo_Aciana/anims/rt__Backflip.glb` now exists (written for the viewer test).

### Generator page (`viewer/generate.html`, 2026-10-21)

Type a prompt (it is first rewritten into the dataset's wording by a local Qwen2.5-3B unless the checkbox is off; the page shows the rewritten text), pick a character (2,792 usable targets, Mixamo first), choose guidance / seed / sampler mode, press Generate; the clip plays in the compare player (`compare.html?embed`). Behind it:
`POST /api/generate` in `serve.py` -> `scripts/generate/generate_motion.py` (also a CLI) -> diffusion sample -> IK proxy -> retargeter. Needs the conda env `Mario` for `serve.py`; the first call loads the models
(8 s here), then a sample takes 1-3 s. Mode `auto`: walk cap for "walk" prompts, duration band + 0.9-quantile cap for "run/jog/sprint", plain for everything else (the constraints were only validated for walks and runs).
The status line flags reach failures, penetration, sliding, rotation spikes and large body lift. Every generation writes `data/decoded/gen_*.glb` and `data/characters/<target>/anims/rt__dec_gen_*.glb`
(they accumulate; delete them to clean up). Checked in headless Chrome: a run on Aciana (0 reach failures, 23/23 tracks bound) and a backflip on a quadruped (flagged: 6 rotation spikes, 1.09 m lift,
as expected from the retargeter's known limits); 8 walk seeds on Aciana: 7 of 8 last 0.9-3.0 s, one lasted 13.7 s at 0.54 hips heights/s (seed 3; investigated 2026-10-21, see `docs/EMBED.md` "Long walks": a slow mode that is in the data, the duration band removes it).

**Paraphrase augmentation (2026-10-21, `docs/EMBED.md` "Paraphrase augmentation"):** Qwen2.5-3B paraphrases of the training captions (4,053 kept) + a held-out wording evaluation (`scripts/text/wording_eval.py`). Retrained model `data/embeddings/z_diffusion_para.pt`
(not the default yet): reworded-prompt R@10 +5 to +7 points, R@5 +3, canonical prompts -1 to -2 (noise level); the 41 hand-written prompts and flip rate did not change. The remaining gap is mostly CLIP's reading of the reworded text.

**Prompt rewriting (2026-10-21, `docs/EMBED.md` "Prompt rewriting at inference"):** `scripts/text/rewrite_prompt.py`, on by default in `generate_motion.py` / the generator page. Reworded prompts: generated R@5 24.4% -> 33.0% and 26.0% -> 30.0% on the two held-out sets
(3 seeds, ranges do not overlap), canonical prompts unchanged, hand-written precision@5 0.81 -> 0.865. Better than paraphrase training and not additive with it, so the default model is still `z_diffusion.pt`. It can invent details ("...and barks"); shown on the page.

**Qwen3-VL captions, pilot with mesh renders (2026-10-21, `docs/EMBED.md` "Qwen3-VL captions"):** the 2B model on mesh filmstrips of 60 Mixamo clips: retrieval R@5 30% (chance 8%; stick figures 10%), the right main action named for 62% of clips (stick figures 24%). Far above the old chance result but wrong for about 4 clips in 10,
so not a replacement for the dataset captions; full run about 7.7 h with the 2B, a 7-8B model would need a download and be slower. Not run on all clips. On 60 quadruped clips (each on its own character) it is much weaker: R@5 17% (chance 8%), the right gait named for walk/run clips but little else
(30% for the other actions, mostly by echoing the prompt's example sentence), so the quadruped captions stay the dataset's. 8B (downloaded, `Models/Qwen3-VL-8B-Instruct`, 16 GB; 5.5 s per clip): R@5 23%, R@10 42%, the right gait for 8 of 9 gait clips, 46% of the other actions by a shared word, 52 distinct captions of 60 and no template copying, but it misses tail wags and jaw opens
and invents sounds ("barks", "roars"); better than the 2B, still not a replacement for the dataset's captions. 8B as a caption CHECKER (`scripts/captioning/qwen_check_pilot.py`, true vs swapped captions on the 60+60 pilot clips): AUC 0.72-0.82, but it rejects 78-93% of the TRUE captions at its natural threshold and its most distrusted ones are
correct (a lion's swaying tail, a standing lemur); as a filter on 10% wrong captions only 21-33% of its flags would be real errors. Not usable as an automatic filter.

**Why generated backflips and other actions look bad (diagnosed 2026-10-21, scratch scripts not kept).** Tested on prompts and seeds, not on rendered video:
- *Wording matters:* "An object performs/does a backflip." inverts the body in 100% of samples at guidance 2; "A cat do it backflip" only 67% (min up-axis -0.31, a half flip). Use the dataset's wording ("An object ...").
- *Forced mode is not the cause:* walk / run / plain give the same inversion rate for backflips; the constraints only change durations (2.1-2.5 s).
- *Diffusion, decoder and IK keep the flip* (generated latents invert 100% at guidance 2-3, 67-75% at 1; after IK the proxy still inverts). Only 10 backflip clips (14 with "flips backward") are in the training pairs.
- *The roughness is in the generated features:* generated backflips have a median of 14 per-frame joint jumps over 60 degrees already in the IK proxy (real backflips decoded from their own latents: 2, original features: 0, real walks: 0);
  the retargeter carries them through unchanged (14 -> 14). Temporal smoothing of the decoded features does not remove them (sigma 3 frames: 14 -> 12 in the proxy, 8 after retarget). "Jumps" has 0, "kicks" 3-4. Cause of the jumps not found
  (candidates: the heading/yaw definition when the body is inverted, the 6D tilt that decodes slightly non-orthonormal, IK bend direction while inverted). **Investigated 2026-10-21 (`docs/EMBED.md` "Backflip spikes"):** the spikes are in the limb joints,
  not noise in one channel and not mainly folded limbs; in the body frame (limbs rotated back by the hips tilt) generated flips have occasional outlier frames (max jump 2x real). A body-frame low-pass of the decoded limbs (sigma 3 frames, `body_frame_filter`, on by default in the
  generator, field "Smooth" on the page) cuts spikes from 16.5 to 7.1 per generated flip (quadruped 14.5 -> 6.1) with limb movement kept at 99% (kicks 92.5%); the generator-side cause is not found (found 2026-10-22: the guided latent is too large, see "Session 2026-10-22").
- *Quadruped targets add the retargeter's known flip limits:* generated backflips on the 47 quadrupeds: median lift 0.32 m, 8 spikes, 99% flagged by the generator page (bipeds: lift 0.04-0.11 m, 14 spikes, 100% flagged by the spike rule).

### Compare viewer (`viewer/compare.html`, 2026-10-20)

Source clip and retargeted result side by side (or overlaid) on one synced timeline. The list is every
`rt__*` clip found under `data/characters/*/anims/` (served by `/api/retargets` in `serve.py`); `--write` on
`retarget.py` (or `--batch ... --write`) adds entries. Shared Mixamo source clips play on a chosen Mixamo
character (default Aciana, with its root scale). Controls: space play/pause, left/right one frame, 1/2/3 side /
front / three-quarter camera, `o` overlay (source drawn as a blue skeleton through the target), `s` skeletons,
`t` root trails, `f` follow, up/down next result. The panel shows the metrics (green/amber/red) and the
source captions; flagged list entries have many reach failures, sliding or penetration. Checked in headless
Chromium (no console errors, 23/23 and 28/28 tracks bound on the stand-in backflip); not checked on a GPU
browser. `data/characters/*/anims/rt__*.glb` now holds the 24 quadruped batch clips (4 clips x 6 targets) plus the
Mixamo Aciana backflip.

### Caption and motion embeddings (2026-10-20, `docs/EMBED.md`)

- CLIP ViT-B/32 text embeddings of all 17,384 distinct captions (`embed.py --text`, `--query`; `retarget.py --query` uses them).
- **Semantic motion embedding for any body** (`motion_sem*.py`): joint-set transformer, 12,638 clips including 156 insect,
  222 bird, 59 marine rigs, aligned with CLIP text. Held out (rigs never seen): text -> motion median rank 36 of 1,262
  (chance 631), R@5 18.5%. "walks", "jumps", "flies" retrieve matching motions on 2-, 4-, 6/8-legged and winged bodies.
  Transfer to a body type never seen in training is weak (insects: median rank 11 of 26, chance 13; a spider held out: common actions
  found among its own 9 clips, rarer ones not), so each body needs examples. 65 free template clips (spider, fox, horse, snake, fish...) were added.
- **Decodable latent** (`motion_ae.py`, biped and quadruped, 6,055 clips): `z` (256) for a diffusion model, plus `sem` (512). Held-out
  text -> motion R@5 36.4% (chance 0.9%); reconstructs pose well but root velocity and yaw barely better than the mean.
  Saved pairs (caption, CLIP text embedding, `z`, `sem`) are in `data/embeddings/motion.npz`; all-body `sem` in `motion_sem.npz`.
  Decoded latents were drawn (`decode_latents.py`, `docs/figures/decoded_latents_summary.png`): plausible poses, median 78% of the
  limb movement and 71% of the root travel kept on held-out clips (corrected features). **IK step built** (`decoded_to_character.py`): a decoded latent becomes a clip
  on the Mixamo skeleton (two-bone IK) and is retargeted onto any biped or quadruped; 9 latents x 2 characters, 13 of 18 runs with 0 reach failures; listed in
  `viewer/compare.html` (`rt__dec_*`), proxies in `data/decoded/`.
- **Why walking looked weak (`docs/EMBED.md`)**: measured, not weak by itself (reconstructions keep the gait). A data artifact made 12% of biped training clips
  "float" (rest height counted twice for some Objaverse rigs); hips height is now measured above the lowest foot, and the role and diffusion models were retrained
  (`*_v1` files kept). Bad generated walks (floating, too fast or not stepping) fell from 33% to 23% at guidance 2 (real clips 7%); guidance trades "too fast" against
  "not stepping", rescaled guidance did not help. Cost: clips whose feet never touch the floor (flips, hangs) now measure height above the feet; 3 of 9 decoded examples got new reach failures.
- **Speed-aware sampler** (`speed_sampler.py`, `docs/EMBED.md`): keeps a generated clip's speed in the range plausible for its prompt (prior from the nearest captions, Gauss-Newton
  step through the decoder, stride kept). "Too fast" walks 12.9-16.8% -> 0%, bad walks 22-25% -> 10-11% (real 7.1%); held-out adherence and variety unchanged. It mostly corrects the
  time scale (longer predicted durations); running durations are still too short at high guidance (0.4 s vs 0.8 s real), not fixed.
- **Quick diffusion test** (`diffusion_test.py`, `docs/EMBED.md`): a text-conditioned diffusion model on the saved pairs generates latents for
  held-out prompts; at guidance 2 it finds the prompt's clip with R@5 33.5% (direct retrieval 36.4%), decoded samples move like real clips, and a wave,
  a kick, a jump and a backflip look right by eye (walking looked weak at first; the cause turned out to be a data artifact, see above). Guidance trades variety (0.44 -> 0.06) and realism for adherence.
- Qwen3-VL-2B captions from stick-figure renders were at chance level against the dataset captions (`qwen_caption.py --eval`), so
  the dataset captions are used. (2026-10-21: mesh renders, the 8B model and an 8B caption checker were tried too, none is good enough; see the Qwen3-VL paragraph under Status.)

### Running durations (2026-10-21): mostly fixed by a duration band

**Result** (details and table in `docs/EMBED.md`, "Runs: a duration band"). Generated runs were too short because classifier-free guidance pushes the "runs" latent along the decoder's duration direction (plain median 1.06 / 0.64 / 0.47 s at
guidance 1 / 2 / 3, real 0.82 s). A per-prompt duration band (`speed_sampler.py --dur-k 1.0`, the real durations of the 40 nearest captions, mean +- k sd) with the speed cap taken as the 0.9 quantile of the neighbours' speeds (`--cap-q 0.9`)
gives, at guidance 2, median duration 0.77 s, median speed 2.96 hips heights/s (real 3.04), 4.7% of clips clearly shorter than the real 10th percentile (plain 43%). The old walking cap (mean + 0.5 sd) is wrong for runs (it
pulls the speed to about 2) and the band alone does nothing for walks (too fast 12.3% against 12.9% plain), so **walks keep the cap and runs use the band plus the quantile cap**; there is no single setting for both.
Side effects checked (`--check`): held-out adherence (R@5 33.5%) and variety (0.167) unchanged.

**Not solved:** the band's floor is the real 10th percentile, so clips pile up on it (25.5% of clips at guidance 2 sit exactly at 0.58 s, real 10%; the strict count of `--runs` shows it). A higher floor (`--dur-k 0.5`) helps but
trims long runs. Nothing was put on a character; "not stepping" does not work for runs (25.5% of real running clips fail it). The setting is chosen by prompt type, so the prompt must be known to be a run.

**Code state:** `speed_sampler.py --runs/--check` take `--dur-k`, `--cap-q`, `--max-step`; `--runs` prints a strict "short" count. `diffusion_test.py --samples ... --speed-aware` has new `--dur-band <k>` and `--cap-q <q>`
(parse, not run). Logs of this session: `data/logs/runs_*.log`, `walks_durband1.0.log`, `check_band1.0_capq0.9_g2.log`.

**Also open:** the hips-height fix (hips above the lowest foot) made clips whose feet never touch the floor, such as flips and hangs, measure height wrongly (3 of 9 decoded examples got new reach failures; the old pipeline is kept as `*_v1` files);
the speed-aware walks were only looked at as stick figures, not on characters; screenshots in `docs/figures/decoded_on_characters.png` and `diffusion_backflip_on_characters.png` are from the old features.

### Session 2026-10-22: splits, `meta.json`, cleanup script, flips

First took the mechanical leftovers of the requirements sheet (below, no research), then flips (next paragraph).

- **Flip roughness: cause found, partly fixed (`docs/EMBED.md` "Latent norm").** Classifier-free guidance pushes the sampled latent beyond the data: |z| about 22 at guidance 2 against 12.6-19.6 for real clips (flips 12-17.5), and the decoder is rough there (spikes rise with guidance and norm: 0.2 per clip unguided, 7.1 at g 2). `generate_motion.cap_latent_norm` (`--znorm`, default 0.9 since the held-out run, 1.0 for the numbers in this paragraph, **plain mode only**) scales the latent down to the mean norm of the 40 training clips nearest to the prompt. Backflips: smoothed spikes per clip 6.9 -> 2.5 (decoded proxy, 12 seeds), 8.0 -> 1.75 on Aciana and 6.1 -> 0.75 on the quadruped stand-in after retargeting (8 seeds), flips still invert (12/12) and land upright more often (15 -> 18 of 24 in the tilt); held-out adherence unchanged (R@5 33.5 -> 33.5, 555 prompts). Not helped: kicks and rolls on the ground (other cause). Walk and run modes are untouched because the cap changes their tuned speed/duration (it lengthens runs from 0.58 to 0.94 s, a lead for the duration-floor problem). Seed-1 failure (lying, then a pop, `docs/EMBED.md` "Seed 1"): the pop is in the decoded features (a back fall followed by a 4-frame get-up; the sample sits between the flip and fall-and-get-up modes), and 28% of capped backflip seeds never complete a turn (pitch range under 300 degrees; real flips 335-569), so the earlier "inverted 12/12" counts overstated flip completion. **New candidate gate** (`generate_motion.py`, `--tries 4`, plain mode only): draw up to 4 candidates, keep the first with at most 1 rotation step over 60 degrees, none over 100, and for flip prompts a completed turn and (added later, `LAND_UP` 0.8) an upright ending; on fresh seeds retargeted spikes fall 5.1 -> 1.3 (backflip), 4.8 -> 1.3 (kick), 4.7 -> 1.2 (rolls), 2.6 -> 0 (cartwheel), backflips completing a turn 62% -> 96%, front flips 42% -> 67% (gate passes only 33%). Seed 1 itself still fails all 4 candidates (smooth but incomplete). Ruled out on the way: the non-orthonormal decoded tilt (SVD projection and tilt smoothing change nothing). Looked at as frame strips (stills, no video; `docs/figures/flipcap_*.png`, `docs/EMBED.md` "Latent norm"): same overall shape with and without the cap, quadruped seed 3 clearly cleaner, but Aciana seed 1 is not a flip (lies on its back, then pops upright in one frame) and has a 161 degree step with the cap; the seed that lands on its back is not fixed; the quadruped's root lift went up slightly (0.28 -> 0.36 m max).

- **Character split** (`scripts/dataset/make_splits.py --write`, reproducible: md5 of the skeleton hash): `data/splits/train.txt` 3,213 and `val.txt` 357 characters (10%), stratified by source and class, grouped by skeleton so
  no held-out character shares a rig with a training one (asserted). Per stratum: Mixamo biped 5 of 50, Objaverse biped 331 of 3,307, Objaverse quadruped 21 of 213 (4 of the 47 usable quadruped targets; 284 of 2,792 targets overall).
  **Mixamo cannot be grouped**: all 50 share one rig and the same clips, so those 5 are held out per character and test mesh/skin generalisation, not motion. `splits/summary.json` has the numbers.
- **This split is not the clip split** used by `motion_ae.py` / `motion_sem.py` (those hold out Objaverse rigs / object types and Mixamo clip families by md5; Mixamo characters are never held out there). The two are independent: 28 of the 352
  held-out Objaverse characters are also held out by `motion_ae.split_of`, about chance. So **the embedding, decoder and diffusion models have seen clips of the characters in `val.txt`**; the held-out numbers in `docs/EMBED.md` do not refer to this split. Nothing is trained or evaluated on it yet.
- **`meta.json` in all 3,570 characters** (format of `_example`): `id`, `class`, `joint_map` (`mixamo_biped` for Mixamo; `null` for Objaverse, there is no map for them), `shape_captions: []` (empty on purpose: shape captioning is untested, nothing was made up),
  plus `source`, `skeleton`, `retarget_target`. `normalize.py` and `convert_clips.py` do not delete character folders, but rerunning `make_splits.py --write` rewrites the `meta.json` files (edit captions there only after a change to the script that keeps them).
- **`scripts/tools/clean_generated.py`**: removes `data/decoded/gen_*.glb`, `characters/*/anims/rt__dec_gen_*.glb` and their `anims.json` entries. Dry run by default; **not run with `--delete`**. Currently 119 files (5.8 MB) and 74 entries in 14 `anims.json` files would go.

### Prompt -> character by shape retrieval (2026-10-22, started)

`scripts/generate/text_to_character.py "a small brown dog doing a backflip"`: a local LLM (Qwen2.5-3B, few-shot) splits the prompt into SHAPE ("a small brown dog") and MOTION ("does a backflip"); the shape text is matched, in CLIP space, against CLIP **image** embeddings of rest-pose portraits of all 3,570 characters (`viewer/portrait.html`: front, side, three-quarter; `scripts/generate/shape_retrieval.py`; `data/shape/all/`), restricted to retarget targets; the motion text goes through `generate_motion.generate` onto the top hit. The result is a real dataset character (mesh, skeleton, skin) with a generated clip; no new body is created. Pilot on 150 characters (`data/shape/pilot`, `docs/figures/shape_retrieval_pilot.png`, `shape_specific_img.png`): looked at by eye, "a human character", "a dog", "a cat", "a cow", "a bear" return the right kind of character in the top 5, "a robot" mostly robots; the rig-category check (20/60/70 humanoid/biped/quadruped) is crude (quadruped queries 100% precision@5, human queries under-scored because Objaverse humans are labelled "bipedal"). **Qwen3-VL-2B captions of the portraits embedded as text were worse** (hubness: the same few characters for unrelated prompts), so the image route is used and no captioning model is needed. End to end on three prompts: `docs/figures/text_to_character_demo.png`, `data/logs/text_to_character_demo.log`. Limits: only retarget targets (47 quadrupeds, the biped targets) can be returned; cosines are about 0.30 with small margins; size and colour words only partly count ("small brown dog" gave a dark brown wolf); no new shapes; in `generate.html` since 2026-10-22 (checkbox "pick the character from the shape words in the prompt", on by default: the split prompt's shape part chooses the character, six candidate portraits are shown and a click puts the same motion on another one; no shape words keeps the character picked in the list; `POST /api/generate` takes `auto_character`; `docs/figures/generate_page_shape.png`; **no language model by default** (`scripts/text/text_frontend.py`: word lists and rules split the prompt into shape and motion and rewrite the motion into the dataset's wording; measured against the 3B rewriter on held-out reworded prompts it keeps about a quarter to a third of the gain on garbled paraphrases and three quarters on hand-written prompts, `docs/EMBED.md` "No language model"; `--frontend llm` / the page's "Text front end" selector switch to the LLM, which loads on first use); the LLM option is a text-only Qwen2.5-1.5B-Instruct (since 2026-10-22; the 3B gave the same results on a 24-prompt split test and a 90-prompt rewrite test, `docs/EMBED.md` "Small LLM") serves both splitting and rewriting, loaded lazily and preloaded in the background at server start; the server must run in the conda env: `conda run -n Mario python viewer/serve.py`); not evaluated on held-out characters (the index includes all 3,570).

### 2D (planar) test (2026-10-22)

`scripts/motion/planar_test.py`: autoencoder + diffusion retrained on features reduced to the sagittal plane (no sideways velocity, yaw, roll/twist, or lateral limb movement), compared with production and a control on flips, cartwheel, jump, walk, run (24 seeds x 4 candidates). The control reproduced production bit for bit (training is deterministic), so it adds no noise estimate. Result: exact straight-line motion (side and yaw 0); walk, run and jump unchanged (speed within 6%); a first planar run looked like a backflip landing gain (upright 47% vs 33%) and a front-flip loss (completes 41% vs 60%), but a second planar training seed moves the flip numbers by 10-25 points, so those are within training noise. With a second seed of the full model too (36.0 R@5, flip completion steady at 59-64%): the two variants are level on the gate (backflip all-three 12% vs 13.5%, front flip 9.5% vs 8.5%), planar is consistently rougher (more rotation spikes on flips, fewer smooth candidates), about 3 points lower on held-out R@5 (32.8 / 33.9 vs 36.4 / 36.0) and lands upright a bit more often (41% vs 34%, not established). Not adopted; a straight line comes for free from the full model by zeroing side velocity and yaw rate in its output (`generate_motion.py --straight 1`, default off; checkbox "straight line" on `generate.html`, a "Quality gate" block on `generate.html` (number of tries 1-16 and a checkbox "flips must land upright"; after each plain-mode clip a line shows gate passed/failed, which candidate was used, spikes, turn angle and landing value; `POST /api/generate` fields `tries`, `land`; `generate_motion.py --no-landing`; `docs/figures/generate_page_gate.png`), checkbox "show the source clip next to the result" on `generate.html` (a "Source" button / key `x` / `?nosource` in `compare.html`; off shows only the retargeted character, centred), `POST /api/generate` field `straight`, history/status show it; no change in any gate rate, root path deviation 0.07-0.29 -> 0 hips heights; level 2, pitch-only tilt, does not help); details and follow-ups in `docs/EMBED.md` "2D test".

### Running on a phone: audit (2026-10-22, `docs/MOBILE.md`)

Measured on the Mac, not on a phone. The three neural parts export to ONNX and match torch to 1e-6 (`scripts/generate/export_onnx.py`): denoiser 28.5 M parameters (28.6 MB int8), decoder 3.5 M (6.3 MB int8), CLIP text tower 63.4 M (254 MB fp32). Full 50-step guided sampling + decode runs in 60 ms (fp32) / 21 ms (int8) on the CPU against 158 ms in torch on the GPU, with the same gate metrics (int8 within noise). **CLIP is the blocker among the models:** ONNX dynamic int8 breaks it (centred embedding cosine 0.36); fp16 weights are lossless (127 MB) and weight-only int8 (about 63 MB) gives cosine 0.996 and generated motion within noise (emulated). int8 stack about 98 MB + tables about 10 MB, inside the 100-150 MB plan but with little margin. Memory (ONNX Runtime, measured): int8 denoiser + decoder 51-89 MB, CLIP fp32 494 MB (1.9x its file), so a naive port needs about 0.6 GB for the models; character textures dominate the characters (median about 0, p90 22 MB, worst 805 MB decoded). `viewer/generate.html` has a "Runtime" block with a **mobile checkbox** (int8 ONNX on the CPU, threads, assumed phone slowdown factor) that shows the time of each step (`docs/MOBILE.md` sections 6-7); the generator page used to wait 13-17 s for the player on every result (`/api/retargets` scanned all 3,570 character folders): the embedded player now asks for its single clip (`/api/retarget`, 0.08 s) and the full list is cached. Other blockers: about 2,300 lines of numpy / scipy Python to port (IK proxy, retargeter, walk/run constraint, CLIP tokenizer), a 4 GB character library (a subset must be chosen, licences unchecked), and the on-device player (not started).

### Session 2026-10-26: Anymate, the Lottie branch, joint 3D+2D model, search page

**Where things run.** Mac (this folder; Python envs: conda `Mario` for the viewer server and the older scripts, which has the `zstd` command but no `zstandard`/`orjson`) and the GPU server `ssh loomie@100.123.204.66`
(2x RTX 3090, 20 threads; Tailscale must be running on the Mac; project at `/media/loomie/SciData/Marionette`). The envs used this session (`amenv`, `venv`) were scratch virtualenvs under `/private/tmp` and may be gone:
recreate with `python3 -m venv ~/envs/marionette_mac && ~/envs/marionette_mac/bin/pip install torch numpy scipy trimesh scikit-learn einops omegaconf timm diffusers matplotlib termcolor opencv-python-headless point-cloud-utils pyyaml easydict transformers accelerate sentencepiece protobuf orjson zstandard`
(`anymate_*.py`, `motion_sem_joint.py`, `lottie_search.py build` need torch; the Lottie scripts need `orjson` + `zstandard`; call them with `python -I`). Re-copy server data with
`rsync -r --size-only --no-perms --no-owner --no-group -e "ssh -o BatchMode=yes" loomie@100.123.204.66:/media/loomie/SciData/Marionette/output/ data/vlm/` (no `-t`: the exFAT drive refuses to set times; add `--exclude lottie`; the motion parse is `output/lottie/animation/` -> `data/lottie/animation/`).

**Anymate (auto-rigging, `../Anymate`, https://github.com/yfde/Anymate).** Predicts joints, bone connections and skin weights from a mesh; the dataset has no motion. Pretrained checkpoints (1.8 GB) run on the Mac CPU through `scripts/rigging/anymate_predict.py`
(patches `torch.load` from outside; the repo is untouched; the 3.9 GB Michelangelo weights are not needed; `binvox` is replaced by trimesh voxelisation, IoU 0.84-0.96; model frame = bbox centred, longest side 2, Y up). 221 characters predicted (127 biped, 47 standing quadruped,
47 quadruped in a non-standing rest pose). `scripts/rigging/anymate_roles.py eval` maps the nameless predicted joints to the retargeter's roles (hips, head, spine, leg/arm chains) from tree geometry (class given); errors are in units of the rig's hips height.
Held-out half: bipeds 94% get roles, mean error 0.043; standing quadrupeds: coverage rose from 33 to 39 of 47 with mesh-based foot detection, mean error about 0.12 before and after (no clear accuracy gain), head is the weakest role (0.17-0.19, best possible 0.06). `scripts/rigging/anymate_rig.py run` builds
`data/anymate/<key>/rig/rig.glb` (named, skinned) and retargets Walking / Jump_3 onto it: 32/32 retargets on 16 held-out characters, metrics close to the same clips on the characters' own rigs (reach failures, skating, spikes); filmstrips in `docs/figures/anymate_rigs_*.png`; the rigs also play in `viewer/compare.html` (search "anymate").
Limits: geometry only, no textures on the rigs, quadruped limb choice is rough, leakage against Anymate's training set not checked (id schemes differ), judged from stills and the retargeter's own metrics, not from video.

**AnyTop (`../Anytop`, https://github.com/Anytop2025/Anytop).** Motion diffusion for any skeleton topology; no text prompt for motion (only T5 embeddings of joint names), 2.29 M parameters per checkpoint, needs CUDA. Its dataset is the paid Truebones Zoo and the processed version is not published, so it could not be downloaded;
everything the authors publish (5 checkpoints, `cond.npy`, `Truebones_skeletons.txt` with 68 creature types, research-only licence, 122 MB) is in `../Datasets/AnyTop`. Not run yet (planned: try the checkpoints on the server).

**Lottie branch (text -> 2D animation).**
- *Qwen labels* (`scripts/lottie2d/vlm_tables.py` on the server; stage 2 = Qwen3-VL-8B on rows the 2B stage kept): marks each animation relevant if it shows a whole living 2- or 4-legged character. Copied 44 shards at 14:50 -> `data/vlm/` (432,223 rows, 92,700 looked at, 60,412 relevant: 59,138 two-leg, 1,274 four-leg). **Synced again at 18:53 (rsync, all 53 finished shards verified with `zstd -t`; obsolete local `partial-*` files removed): `data/vlm` now has 53 of 68 stage-2 shards = 522,223 rows, 112,196 looked at, 72,396 relevant (70,758 two-leg, 1,638 four-leg, 2,067 animal subjects). The server job was still running (the progress bar sits near 90% of a total that keeps growing, so every finish estimate I gave was too short); only two partial files and the log keep changing.** The 44-shard figures above are what the tokens, browse index and models below were built from: 11,961 relevant animations (11,597 two-leg, 364 four-leg, from 9 corpus shards) are not in them yet. `output/animation.jsonl` (merged) is stale on both sides; nothing reads it.
- *Motion parse* (`scripts/lottie2d/parse_lottie.py`, rotation sine fits per layer): finished on the server, 68/68 shards, 672,223 rows, 0 errors, 2 h 8 min; copied to `data/lottie/animation`. `scripts/lottie2d/analyze_lottie.py motion`: 17.4% of rows use expressions, 49.8% have at least one fitted layer-level rotation track, 15% of tracks are periodic, 70% fit with R2 >= 0.9, 22% of rows repeat an earlier curve fingerprint.
  Relevant + clean + >= 2 good tracks: 28,008 candidates (27,752 two-leg, only 256 four-leg) in `data/analysis/candidates.csv`. The corpus has some duplicate ids (about 500 in the labelled part).
- *Rig-likeness probe* (`probe_lottie_rigs.py`, 298 animations; `lottie_filmstrips.py` for 30 of them): 23% have a parent chain >= 3 deep with >= 3 animated rotations, 70% have mostly generic layer names, so roles cannot come from names. Looked at: about 24 of 30 are real articulated characters, but the motion is upper-body gestures, head, face; legs are almost always static; many are the same template character; four-leg ones are sitting animals, no locomotion.
- *Joint tokens* (`scripts/lottie2d/lottie_tokens.py`): each layer (and each animated shape group) is a joint at its pivot, followed through parents, scale, precomps by 2D forward kinematics; output in the same format as `data/embeddings/sem_feats.npz` (same 4 channels, 8 static, 5 global, 8 scalars), so `motion_sem.py` can read it. Selftest passes; joints overlaid on lottie-web renders sit on the real limbs.
  `build-all` (8 workers, 5 min): `data/embeddings/lottie_feats.npz` (888 MB) + index with captions: 45,850 of 60,412 relevant animations tokenised (76%), 795,417 joint tokens, median 13 joints per clip; the rest have no moving layer or group transform. The tokens use the 44 label shards only.
  The skeleton shape is still separable from 3D (classifier AUC 0.996-1.000 on skeleton features alone, in every variant tried): the difference is structural (median joints 10 vs 49, leaf share 0.5-0.6 vs 0.2, bones 0.3 vs 0.07 of body size), not an artefact.
- *Action captions* (`scripts/lottie2d/lottie_actions.py`): rules cut the 35-word Qwen captions to motion phrases in the 3D wording ("An object points forward and gestures toward its chest."); found for 94% (dynamic verb for 72%), 7 words on average. Flaws: object motion counted as the character's, posture-only fallbacks.
- *Joint 3D + Lottie semantic model* (`scripts/motion/motion_sem_joint.py`, imports `motion_sem.py` unchanged): 3D train + 11,000 sampled Lottie clips, single-domain batches 50/50, per-domain text centring, 40 epochs, 13-22 min each on MPS. There are no paired 3D/2D clips, so the cross-domain test uses action words in captions:
  for held-out clips with the word, are the 5 nearest clips of the other domain also captioned with it (precision@5, chance in brackets)? Three seeds per version (a seed changes the sampled Lottie clips too), mean +- spread: Lottie->3D (7.9%): full captions 17.2 +- 0.5, mixed full+action texts 19.4 +- 2.4, 3D-only model 8.4-9.0; 3D->Lottie (2.7%): 20.3 +- 1.0, 23.2 +- 0.9, 3D-only 15.7-17.5.
  Mixed beat full in 5 of 6 paired comparisons. Action captions alone: Lottie->3D 15.1, 3D->Lottie 27.6 (better for run/jump/dance, worse for wave/point). Domain-classifier AUC on the clip embeddings stays 0.93-0.94 (3D-only 0.99): the domains are still separate. 3D held-out R@5 about 17.0 vs 18.5 for the production model (small consistent cost). Gestures (wave, point, nod) stay weak from the Lottie side. Caveats: a caption word is a crude truth, small counts per word, weighted by query count.
  Summary table: `python scripts/motion/motion_sem_joint.py summary --tags pilot full_s1 full_s2 mixed mixed_s1 mixed_s2 --group full=pilot,full_s1,full_s2 mixed=mixed,mixed_s1,mixed_s2`. Tags: `pilot` = full captions seed 0, `mixed` = mixed seed 0, `actions`.
- *Search page* (`viewer/lottie_search.html`, `scripts/lottie2d/lottie_search.py build --tag mixed --captions mixed`): text -> nearest Lottie animations (played with lottie-web) and nearest 3D clips. **Retrieval, not generation**: there is no Lottie generator yet; the generator page still makes 3D motion only.

**Viewer.** `python viewer/serve.py` now opens `viewer/home.html` (menu with an "i" manual for every page; `--no-open` to skip); new pages `lottie_tokens.html` (46 samples or the full 60,412-animation dataset, read from the corpus on click through `/api/lottie_item`, uses the real file path because `zstd` ignores symlinks)
and `lottie_search.html`; `compare.html` lists the Anymate rigs. The server skips an unreadable `anims.json`: `data/characters/objaverse_669a85df57bbceb71e84a51b_fbx/anims/anims.json` is truncated (NUL bytes from byte 8192, a write on 2026-10-22 that did not flush); its generated clips were hidden (repaired on 2026-10-27, see Session 2026-10-27).

**Update 2026-10-27: everything rebuilt with all 68 label shards.** The server finished both Qwen stages (68/68 shards in `output/stage1_2b/animation` and `output/animation`, 672,223 rows each; stage 1 looked at 142,412 rows and called 121,286 relevant, stage 2 re-ran those with the 8B model: **91,724 relevant**). `data/vlm` is a byte-identical copy (`rsync ... --exclude lottie/`), `data/lottie/animation` the motion parse. Rebuilt on the Mac (env `Mario`, now with scikit-learn, orjson, zstandard from conda-forge): `analyze_lottie.py vlm|motion` (**42,568 training candidates**, 42,092 two-leg / 476 four-leg; was 28,008), `lottie_tokens.py build-all --shards all --workers 8` (509 s: **69,585 of 91,724 tokenised (76%)**, 1,182,140 joint tokens, `lottie_feats.npz` 1.3 GB), `lottie_tokens.py browser-index` (91,724 animations), `lottie_actions.py build` (action caption for 94.2%), `lottie_search.py build --tag mixed --captions mixed` (69,585 Lottie + 12,703 3D clips; 720 s, much slower than before for an unclear reason).
  The 7 joint-model runs were trained on the server (2 x RTX 3090, about 7 min each instead of 13-22 on the Mac; two queues of runs in `/media/loomie/SciData/Marionette/work68`, a copy of `scripts/` with the device line patched to prefer CUDA, env `~/envs/marionette`; the Mac scripts still use MPS), same settings as before (11,000 Lottie clips sampled, 40 epochs, seeds 0/1/2), models and `cross_domain_*.json` copied back. Mean +- std over 3 seeds (query-weighted precision@5; chance 7.9% Lottie->3D, 2.7% 3D->Lottie): Lottie->3D full 17.8 +- 0.9, mixed 20.3 +- 0.8; 3D->Lottie full 20.4 +- 1.9, mixed 24.6 +- 0.4 (44-shard runs: 17.2 / 19.4 and 20.3 / 23.2); mixed beats full in all 3 paired seeds in both directions. Domain AUC 0.93 (full) / 0.94 (mixed), unchanged. In-domain held-out R@5: 3D 16.3 (full) / 16.5 (mixed) (production model 18.5); Lottie against the full captions 20.2 +- 2.3 (full) / 16.8 +- 1.6 (mixed), lower than the 44-shard runs (24.3 / 22.1): the held-out clips are a different slice and the CUDA run is not bit-comparable, so I did not chase it. `actions` (seed 0): Lottie->3D 18.6, 3D->Lottie 26.2, Lottie R@5 9.5 (judged against the full captions, so low by construction). The old 44-shard files are in `data/archive_44shards/` (features, indexes, models, results, logs). Not changed: the training still samples 11,000 Lottie clips (there are 69,585 now); the model was not retrained with more.

**Dataset viewers (2026-10-27).** The GPU server's Qt viewer (`SciData/Marionette/viewer/viewer.py`, PySide6, two modes) is replaced by two web pages in `viewer/`, one code file each, both served by the single `viewer/serve.py` (no second Python script; the Qt file stays on the server and was not copied): `vlm_table.html` = the Qwen label table (`/api/vlm_table`: processed rows of `data/vlm/animation`, 142,412 of 672,223, loaded in the background on first use, about 560 MB in the server, filter by relevant / body plan / words, sort, paging, plays the row through `/api/lottie_item?raw=1`) and `lottie_raw.html` = the corpus itself (`/api/lottie_raw`, `/api/lottie_raw_row`: LottieAnimation-660K or LottieSVG-10M, a shard and a window of rows, SVG source tab for the SVG set). Two cards with manuals were added to `home.html`. Tested in Chrome with playwright (no console errors, screenshots looked right); not done: the Qt viewer's `--screenshot` option, and the SVG label table (no SVG labels exist yet). Also pulled from the server: `scripts/vlm_tables.py` (newer, speed fixes), `run_parse.sh`, `verify_copy.py`.

**Not done / loose ends.** No Lottie generator (text -> new Lottie motion); the sine-fit tracks of the 42,568 candidates are not used by any model yet. `data/vlm/svg` is empty (the server has only a 2-shard SVG sample). The 3D->2D path (`motion_to_2d.py` + `bake_lottie.py`) needs `../Loomie/data/side_view_animals`, which was not found on this Mac (not tested here).
The AnyTop checkpoints were not run. Seed-0 models are the ones in `data/embeddings/` (`motion_sem_joint_{pilot,actions,mixed}.pt`); the seed runs left `motion_sem_joint_{full,mixed}_s{1,2}.pt` too.

### Session 2026-10-27: folder layout

- Deleted ~36,000 macOS `._*` files and the `__pycache__` folders; added `.gitignore`; moved the personal notes to `notes/`.
- `scripts/` is split into 8 folders (see Layout). Every script's root path and import path were updated; docs and viewer pages use the new paths. Checked: all scripts compile, every cross-folder import resolves, `retarget.py --self-test`, `lottie_tokens.py selftest`, the viewer server (home, generate, lottie search, lottie item), and the dry runs of `make_splits.py` and `clean_generated.py`. Also run from the new folders: `export_onnx.py` (all three models export, error vs torch about 3e-6, int8 sizes as in docs/MOBILE.md; needs `onnx` and `onnxscript`: listed in environment.yml and installed in the Mario env on 2026-10-27) and `generate_motion.py` ("An object runs forward.", seed 9001: gate passed, 0.58 s, 5.1 hips heights/s; the test clip was removed again). `anymate_predict.py` (`test`, `characters`), `anymate_roles.py eval` and `anymate_rig.py run` also run from `scripts/rigging/` (run on small samples, then `data/anymate/` and the figures were restored from a backup). They need packages the Mario env does not have (`omegaconf`, `diffusers`, `termcolor`, `opencv-python-headless`, `point-cloud-utils`, `easydict`); I ran them with those installed into a scratch folder on `PYTHONPATH`. Anymate's joint predictor is random from run to run (torch is not seeded: the same character gave 42, 42 and 45 joints), so new predictions never match the stored ones exactly. The training scripts (`motion_ae.py --train`, `motion_sem.py --train`, `motion_sem_joint.py train`, `diffusion_test.py --train`, `planar_test.py features/ae/diffusion`) were run for 1 epoch / 50-60 steps under throwaway names and then removed; they train, save and evaluate fine (note: `motion_sem_joint.py` needs at least about 40 total steps, otherwise OneCycleLR divides by zero). The production checkpoints are unchanged. The rest were run on 2026-10-27 as well, each in the safest mode available (dry run, `--limit`, scratch output, or a backup restored afterwards; 3,571 backed-up files compared byte for byte at the end: none differs). Exactly reproduced from the sources: `inventory.py` (clips.csv, rigs.csv, summary.json, report.md), `filter_dedupe.py` (manifest.csv), `motion_features.py` (motion_feats.npz), `motion_sem_features.py` (sem_feats.npz), `make_embed_diagram.py`, `analyze_lottie.py` (vlm and motion summaries, 42,568 candidates), `lottie_actions.py build`, `lottie_tokens.py browser-index`. Ran fine: normalize and convert_clips (dry run), template_clips, paraphrase, rewrite_prompt, text_frontend (library mode), wording_eval, shape_retrieval query, text_to_character (split, shape-only, full), qwen_caption, qwen_mesh_pilot, qwen_check_pilot, render_motion, decode_latents, walk_check, gait_check, speed_sampler, verify_copy, retarget (query and one clip), probe_lottie_rigs, lottie_filmstrips, lottie_tokens (build, demo), lottie_search (query, build), parse_lottie (benchmark), vlm_tables and vlm_compare (help only: they need the server's GPUs), generate_motion --mobile (ONNX Runtime: sampler 86 ms vs 910 ms on the desktop path). The whole 3D to 2D chain ran on real artwork in a scratch folder: filter_side_view_animals (extract, classify, select, verify) -> svg_to_mesh -> rig_mesh (with a hand-placed skeleton) -> motion_to_2d ("a bear walks forward") -> bake_lottie, and the baked Lottie plays in Chrome. Found: `text_frontend.py`'s demo command (`python scripts/text/text_frontend.py "..."`) crashes in its "hybrid" column because it builds the front end without a rewriter (an old bug, not from the move; the rules mode that the generators use works). Not run: `normalize.py` and `convert_clips.py` without `--dry-run` (they would rewrite every character and clip; their reports were rebuilt instead, see below), `retarget.py --batch`, `decoded_to_character.py --examples`, `paraphrase.py --split/--embed`, `shape_retrieval.py sample/render/clip/caption/eval`, `lottie_tokens.py build-all`, the full `vlm_tables.py` / `parse_lottie.py` runs (server).
- **Mistake and repair:** `normalize.py --dry-run --limit 10` and `convert_clips.py --dry-run --limit 20` still write their report (`--dry-run` skips only the model and clip files), and I had not redirected `--report`, so `data/normalize_report.csv` and `data/clips_report.csv` were overwritten with 10 and 43 rows. They were rebuilt the same day: full dry runs to a scratch report, the real `out` paths restored from `data/characters/*/normalization.json` and the `anims.json` files, and checked against the disk (3,570 characters; 7,225 clip rows matching the 7,225 clip files one to one); `inventory.py` regenerates an identical report.md from the rebuilt normalize report. Use `--report <scratch file>` with `--dry-run`. One regenerated file was overwritten without a backup: `rt__Backflip.glb` of `objaverse_135d9d148a8f4b388b5c915f42b1abec` (the retargeter is deterministic and its `anims.json` entry, all metrics, came out identical).
- The project no longer imports `../Loomie`: `build_mesh.py` was copied to `scripts/lottie2d/`, the 2D rig data will go to `data/side_view_animals/` (not created yet).
- `scripts/lottie2d/rig_mesh.py` had been cut off at byte 12,288 (in Loomie as well) and was rebuilt: the head is the original, the end of `check()` and the command-line entry are a reconstruction (`--k-max 4 --power 4` come from the usage line of its docstring). Tested on a synthetic rig only (the viewer's JavaScript deformation matches `build_mesh.deform` to 1e-14).
- `data/characters/objaverse_669a85df57bbceb71e84a51b_fbx/anims/anims.json` was damaged (second half NULs); three entries were rebuilt by re-running the retargeter (the GLBs came out byte-identical). One entry (`..._performs_a_back_flip_s5_..._t1`) lost its generation record and says so in a `repaired` field. `clean_generated.py` now skips unreadable `anims.json` files with a warning.
- Integrity scan of all 28,396 files in `data/` (JSON, GLB, zip/npz/pt, npy, PNG, zstd, NULs in text): no other damaged file.
- Open: `scripts/__pycache__` and `scripts/.___pycache__` are a corrupt directory entry on the drive (Disk Utility, First Aid); a code backup is in `../Marionette_backup_2026-10-27_pre-regroup/`.

### Session 2026-10-28: skeleton classes (exploratory, `docs/SKELETON_CLASSES.md`)

`scripts/rigging/skeleton_classes.py` tests whether a skeleton alone (tree shape and geometry, no names, 2D at several yaws) gives biped / quadruped / limbless / winged, on 4,306 3D
skeletons, AnyTop's 70 creatures, 221 Anymate characters and 557 Lottie rigs. Biped vs quadruped is readable (recall about 0.9 / 0.7-0.8); limbless has an exact "single chain" signature that
fish, ropes and chains share; winged is not recoverable from the rest pose (recall 0.4); real Lottie rigs carry almost no signal (AUC 0.4-0.6). Consequence: the shape stage should emit part roles
itself: done as `docs/SKELETON_SPEC.md` (format, validator, derived class and mode support, 9 examples, self-test in the gate; 192 of 192 Anymate role sets fit it) and `scripts/rigging/skeleton_synth.py` (29 body variants, any yaw, deterministic; a tree trained on 3,000 synthetic skeletons scores 0.56 on AnyTop's creatures against 0.49 for one trained on the real manifest). 97 of the 221 Anymate predictions have a cycle or no root in their parent array. The eval plan (`eval/`) and the human rating page (`viewer/rate.html`) were added the same day.

### Motion for the new body classes (2026-10-28, `data/analysis/motion_coverage.md`)

Decisions: motion is to be learned (not hand-built procedural modes); first modes walk, trot / gallop, flap, slither, swim, idle; a stylised pairing (a mode the body does not natively have) is shown and labelled, not hidden.
`scripts/motion/motion_coverage.py` counts real clips per body class and mode from the captions. Plenty: walk, run, idle, gestures (3D: 11,414 bipedal, 624 quadrupedal clips; Lottie: 69k, almost all two-leg gestures and
idles). Flap: 203 avian clips and swim: 33 marine clips exist in the export but are marked not usable, because normalize.py only handles bipedal and quadrupedal characters. Slither: 13 serpentine clips (2 mention slithering or crawling):
not enough to learn from. Quadruped trot / gallop: 39 clips. Lottie four-leg: 1,724 clips, mostly sitting. Not done: converting the unconverted categories (it would write into data/; to be done into a separate folder).

### Tracing one generated clip (2026-10-28, `scripts/generate/trace_generation.py`)

Runs a generated clip through text, sampler, decoder, IK and retargeter one stage at a time, next to a real clip through the same IK and retargeter, and writes figures and `trace.json` to `data/analysis/trace_generation/<label>/` (nothing in
`data/decoded` or `anims.json`). It reproduces an existing clip exactly (seed 110: maximum joint position difference 0.0 m). First case, `gen_an_object_does_a_front_flip_s11_g2_plain_sm3_zn0.9` (typed prompt, one candidate, no gate): the latent norm is 32.1 before the cap
(real flips 12-17.5), the decoded features turn the body only 130 degrees (real Front_Flip 380) and end lying down; IK and retarget reproduce it faithfully (1 forearm step of 95 degrees, 0 reach failures). With the production defaults (prompt rewritten, up to 4 candidates, quality gate) the same seed
gives a clean flip: candidates 0 and 1 are rejected, candidate 2 turns 359 degrees and lands upright. The generator page's `tries` and `rewrite` options decide which you get.
`scripts/generate/flip_sweep.py` (100 seeds per cell, production gate criteria, `data/analysis/flip_sweep.md`): one sample is a good flip (turn >= 300 degrees, upright landing, smooth) only 6% of the time for "An object does a front flip.", 12% for "performs a front flip", 15% for
"performs a backflip" (cap after sampling, guidance 2). No cap: 2-3%. Lower guidance does not help (guidance 1.5: 3-11%, 1.0: 1-4%; it lands upright but stops completing the turn) and rescaled guidance does not either. A cap applied to the clean-latent estimate inside every denoising step
(x0_fn) gives 8%, 22%, 21%: more completed turns and upright landings, but more spikes (smooth 36-49% against 51-53%). Held-out check (`scripts/generate/step_cap_check.py`, 555 prompts, guidance 2): no side effect (R@5 33.5 -> 35.1, prompt cosine 0.657 -> 0.653, spikes 0.16 -> 0.16 per clip, clips with any spike 7% -> 5%, median duration 2.22 -> 2.26 s). It is available as the opt-in `--step-cap 0.9` (and a checkbox on the generator page), off by default; `retarget compare` now shows each generated clip's gate verdict and has a filter for it. The gate (production setting) finds a good clip within 4 tries for 15%, 31%, 34% of seeds and within 8 for 38%, 54%, 63%
(the older "95% within 4" was measured before the landing criterion existed). Rewriting the prompt doubles the front-flip rate.

### Where we stopped (end of the 2026-10-21 session)

The generation chain is now: prompt -> **rewriter** (Qwen2.5-3B, dataset wording) -> CLIP -> **latent diffusion** (`z_diffusion.pt`) with the **walk cap / run band** sampler -> decoder -> **body-frame smoothing** -> IK proxy on the Mixamo skeleton -> **rule-based retargeter** -> any biped or quadruped
(`scripts/generate/generate_motion.py`, `viewer/generate.html`). What was done today, one line each (numbers in `docs/EMBED.md`, section in brackets):

| topic | outcome |
|---|---|
| runs too short ("Runs: a duration band") | fixed for runs with a duration band + 0.9-quantile speed cap; clips pile up on the band floor (25% at guidance 2), walks keep the plain cap |
| 13.7 s walk outlier ("Long walks") | a slow mode that is in the data; the band removes it (not adopted for walks yet) |
| generator page (`viewer/generate.html`, `strip.html`) | prompt -> clip on any of 2,792 characters; rewriting and smoothing on by default |
| wording robustness ("Paraphrase augmentation", "Prompt rewriting at inference") | rewriting at inference is the fix (reworded-prompt R@5 +4 to +9 points, canonical unchanged); paraphrase training adds little and does not stack; `z_diffusion_para.pt` exists but is not the default |
| backflip spikes ("Backflip spikes") | smoothing the decoded limbs in the body frame halves the spikes (mean over 7 seeds 18.7 -> 8.6) but 5 of 6 new seeds keep a jump over 130 degrees; the generator-side cause is not found |
| Qwen3-VL captions (2B and 8B) and an 8B caption checker ("Qwen3-VL captions") | mesh renders lift the 2B from chance to R@5 30% (humans) / 17% (quadrupeds), the 8B to 23% on quadrupeds; none is good enough to replace the dataset captions, the checker's strongest negatives are its own errors. The dataset captions stay |

**State of things**
- Defaults: diffusion model `z_diffusion.pt`; generator mode `auto`; rewriting on; smoothing sigma 3. Since 2026-10-22: candidate gate `--tries 4` in plain mode (generation 0.2 -> 0.3-0.6 s); latent norm cap on in plain mode (`--znorm 0.9`, switched from 1.0 the same day after the held-out run; the generator page uses it, there is no field for it). `z_diffusion_para.pt` unused.
- The viewer server is stopped (`conda activate Mario && python viewer/serve.py`, then `http://127.0.0.1:8765/viewer/compare.html`, `strip.html`).
- New models on disk: `Models/Qwen3-VL-8B-Instruct` (16 GB, downloaded for the pilots, can be deleted if the captioning idea is closed).
- Generated clips accumulate in `data/decoded/gen_*.glb` and `data/characters/*/anims/rt__dec_gen_*.glb` (and entries in each `anims.json`); `scripts/tools/clean_generated.py` removes them (dry run by default, not run yet). Remaining `rt__*` clips from the batch tests are in the same place.
- Scratch scripts of the investigations (spike diagnosis, seed tests) were not kept; their findings and numbers are in `docs/EMBED.md`. The reusable ones are in `scripts/` (listed above).
- Tip: `python3 -u` when redirecting a script to a log, otherwise the output only appears when it exits; `playwright` for screenshots is in the system Python (`/usr/local/bin/python3`, browser `channel="chrome"`), not in the `Mario` env.

**Open, roughly in the order I would take them tomorrow**
1. **Flips:** (2026-10-22: the guidance-norm cap cut the spikes to about 2.5 per clip, see "Session 2026-10-22"; the list below is what is left.) Flips were looked at as frame strips on 2026-10-22 (not video): the seed-1-type pop is now mostly gated out (see the candidate gate) but incomplete flips and clips that end in a deep crouch remain; the gate was not re-watched; a landing-upright criterion is in the gate but only 8-9% of single samples pass smooth + complete + upright, so 4 tries pass 7 of 24 seeds (returned clips upright 37% back / 50% front vs 33% / 34% single sample, `docs/EMBED.md` "Landing criterion"), more tries untested; `--znorm 0.9` was run on the held-out set (2026-10-22: R@5 33.5 unchanged, spikes 0.25 -> 0.15, durations closer to real; adopted as the default); seeds that land on the back; kicks and rolls still have 4 and 10 spikes (another cause); only 10 backflip clips are in the training pairs; the hips-height fix made flips/hangs measure
   height wrongly (`*_v1` files). The retargeter adds about 3.6 spikes even to real flips and needs a large root lift on long-bodied quadrupeds.
2. **Quadruped data:** 430 usable clips (below the roughly 500 asked for); 30% are "walks/runs in place". More data or a human-to-quadruped path for actions that quadrupeds do not have.
3. **Sampler choice by prompt type** (walk cap / run band / plain) is a regex on the rewritten prompt; the run band piles up on its floor; walks do not use the band yet.
4. **Requirements sheet leftovers** (see Next steps 4): shape captions per character (the 2B/8B model on static mesh renders would probably do for these, untested), loop flags, `joint_maps/` for Objaverse, rest poses. (`meta.json` and the character holdout in `splits/` done 2026-10-22.)
5. **Evaluation:** the held-out sets are small (101 / 765 prompts, 60-clip pilots); no perceptual metric and no user study; flips were judged on per-frame counts and small stills, not on video.

## Data findings worth remembering

- The Mixamo clip named `Running` is a roll on the ground ("crouches, rolls on the ground, then stands"):
  names can be wrong, check the captions.
- Objaverse clips: 6,908 of 7,355 rigs have a single clip; only 230 normalized characters have 2+ usable clips.
  Quadrupeds: 425 converted clips on 211 rigs (below the roughly 500 minimum in the requirements).
- Mesh coverage: 765 Objaverse rigs have clips but no mesh; 118 characters were skipped (morph targets).
- The loop flag is a guess (first/last pose similarity), not a label.
- The dataset's "biped" class includes odd entries (a T-rex, a 5-joint figure of floating spheres).
  A minimum joint count of about 15 for bipeds would remove the worst ones (not applied yet).
- Of the 213 quadrupeds, only 94 can be labelled at all and 47 stand in their rest pose (limb ends near the
  ground); the rest fail the stance audit (`retarget.py --list-targets`).

## Known limitations of the retargeter (v1)

- Shoulder, clavicle and toe motion is carried rigidly (a human shrug or toe bend is dropped).
- A quadruped's tail is rigid with the pelvis (no secondary motion).
- The forelimb source (`--forelimbs arms|legs|auto`) is a rule: legs for travelling clips, arms otherwise.
- Long flips on long-bodied targets need a large root lift; there is no physics check.
- Roles come from the dataset's clean joint names; rigs without them cannot be targets.
- Not yet compared with a learned retargeter or evaluated with a perceptual metric.

## Next steps (when work resumes)

1. Remaining retargeter issues: Jog_In_Circle on `..669a6881` / `..31a8aaef` (23-50 reach-fail frames, leg vertical
   gap), Backflip reach fails (16-52) and 5-13 rotation spikes, 3 spikes on `..64c84fab` (forelimb elbow `joint23_018`, 62-65 degrees,
   on the frames where the diagonal hind leg fails to reach), thumb penetration up to 5 cm in Walking on `..669a6881`.
2. (done 2026-10-20: `docs/RETARGET.md`; update its sweep table whenever the retargeter changes.)
3. Caption + motion embeddings and generation (see `docs/EMBED.md` and "Where we stopped" above): done: embeddings, decodable latent, IK onto characters, quick diffusion test, walking diagnosis and fix, speed-aware sampler.
   Also done 2026-10-21: generator page, prompt rewriting, paraphrase experiment, flip smoothing, Qwen3-VL caption pilots (negative). Open: see "Where we stopped" above (flips, quadruped data, sampler choice by prompt type), the pile-up of runs at the duration floor, the flip/hang regression of the hips-height fix,
   more non-biped data (Truebones ZOO is a paid pack; the 65 free template clips are in), embedding the `rt__` clips, one script that runs the whole chain, the long-bodied quadrupeds with generated clips.
4. Still missing from the requirements sheet (`meta.json` with class and the `splits/` holdout done 2026-10-22; its `joint_map` is null for Objaverse): 2-3 shape captions
   per character (draft with `Models/Qwen3-VL-8B-Instruct` or the 2B from `viewer/strip.html` renders, then review; the captioning pilots were on motion, shape from a static render is an easier task and untested), `anims.json` loop flags
   reviewed, more `joint_maps/`, one rest pose per class (T- or A-pose).
5. Decide the quadruped size policy (currently 0.9 m for all; `--overrides` or `--objaverse-scale canonical`).
6. Optional: biped minimum-joint filter; richer tail and shoulder handling; a learned retargeting baseline.
7. (2026-10-26) Let the Qwen job on the server finish (53/68 shards at 18:53, `data/vlm` already synced to that point), re-copy `data/vlm` once more, re-run `scripts/lottie2d/lottie_tokens.py build-all --shards all --workers 8`, `browser-index`, `lottie_search.py build`, and retrain the joint model on more than 11,000 Lottie clips.
8. (2026-10-26) Try to lower the domain separability (AUC 0.93-0.94): a domain-adversarial term, or role-map only the rig-like Lottie subset to body skeletons; improve `lottie_actions.py` (object motion, posture fallbacks).
9. (2026-10-26) Decide the generator: text -> Lottie motion. Options: sine-fit tracks of the 28,008 candidates as targets for a text-conditioned model on a 2D rig; or 3D generator -> `motion_to_2d.py` -> `bake_lottie.py` (find or rebuild `Loomie/data/side_view_animals`).
10. (2026-10-26) AnyTop: try its 5 checkpoints on the server; Anymate: improve quadruped limb choice, score skinning on the characters, look at the rigs as video.
