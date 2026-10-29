# Marionette

Dissertation project in two papers; the plan is `docs/pipeline.html` (steps 1.1 to 1.10 and 2.1 to 2.8, each with its status; the original drawing is embedded there and the SVG file is `archive/plan_svg_2026-10-29/dissertation_pipeline_en.svg`) and everything in this repo should belong to one of its steps.
Paper 1: vector rigs from Lottie and 3D -> 2.5D motion transfer (UniMate fine-tuned on UniML3D projections, then on Lottie curves). Paper 2: text-to-character and scene generation on device, built on Paper 1.
This is research code: "works" means the metrics and figures hold, not that tests pass.
Start with `README.md` (Layout, Status, Next steps) and `docs/DATASET.md`. `docs/pipeline.html` is the plan with the status of every step.

## Stack
Python 3.11 in the conda env `Mario` (`environment.yml`; PyTorch comes from conda-forge on purpose: the pip wheel aborts
with "OMP: Error #15"). Three.js/HTML viewers in `viewer/`. Heavy jobs (Qwen labelling, UniMate fine-tuning) run on a GPU
server; their outputs are copied into `data/`. Nothing here is deployed.

## Commands
Run everything from the project root: `python scripts/<group>/<name>.py` (the groups are listed in README "Layout").
- Env: `conda activate Mario`
- Viewer: `python viewer/serve.py` (menu at http://127.0.0.1:8765/viewer/home.html, `--no-open` to skip the browser)
- Quality gate: `.claude/gate.sh fast` (syntax check of every .py, about 1 s) and `.claude/gate.sh full` (+ the self-tests of `skeleton_spec.py` and `skeleton_synth.py`)
- Check the guard hook after editing it: `bash .claude/checks/test_guard.sh`
- Most scripts take `-h`; `normalize.py` and `convert_clips.py` have a dry run (see Gotchas)
- No build, deploy or CI.

## Architecture map
- `scripts/{dataset,rigging,motion,lottie2d,tools}/` : all pipeline code, grouped by step of the plan (README "scripts/ by step"), script names unique. `rigging/` now holds only the skeleton spec tools
- `lottie/` : Lottie curve/primitive classes used by the lottie2d scripts (copied from Loomie; nothing imports ../Loomie)
- `viewer/` : web pages and `serve.py`, the one server
- `docs/` : `pipeline.html` (the plan + status), `DATASET.md`, `LOTTIE_PARSE.md`, `SKELETON_SPEC.md`, `SKELETON_CLASSES.md`, `skeleton_spec/`, `figures/` (Lottie only)
- `eval/` : evaluation plan: `shape/` (mesh + skeleton) and `motion/`, each scored by deterministic checks, a VLM judge and human raters (`human_eval.md`); protocol, schemas and rubrics only, no code
  yet. Outputs go to `data/eval/`
- `archive/` : everything taken out of the pipeline, by date, same relative paths, with pre-edit snapshots (README "Removed"); nothing imports it. `pipeline_svg_2026-10-29/` is the cut to the SVG's steps (retargeter, Anymate pilot, CLIP embeddings, Qwen caption pilots, shape retrieval, 3D docs and figures); the 2026-10-28 folders are earlier removals
- `notes/` : the user's own notes, not written by scripts
- `data/` (untracked) : everything generated and still read by live code; what only archived code used is in `archive/pipeline_svg_2026-10-29/data/` (README "data/"). `sources/` (untracked) : symlinks to `../Datasets` and `../Models`
- `../archive_2026-10-29/` : Anymate and Anytop repos and three unused models, moved out of `..` on 2026-10-29
- `../Models/UniMate/unimate_uniml3d_f60_v3/` : step 1.7's starting checkpoint (CC-BY-NC-4.0); its code repo and `google/flan-t5-base` are not downloaded yet

## Conventions
- Scripts find the project root themselves (`ROOT = Path(__file__).resolve().parents[2]`) and put every `scripts/<group>`
  on `sys.path`, so they import each other by bare module name (`import skeleton_spec`). Copy that line from a neighbouring
  script when adding one; keep names unique across groups.
- `parse_lottie`, `vlm_tables`, `run_parse.sh` also run in the server's flat `scripts/` folder: they find the root by layout.
  Do not break that. Their usage lines, `verify_copy`'s and `docs/LOTTIE_PARSE.md` keep the flat `scripts/x.py` form on purpose.
- When behaviour or results change, update README "Status" / "Next steps", the matching `docs/*.md`, and the `CELLS` array
  in `docs/pipeline.html`. Report numbers only from a run you actually did.

## Boundaries
- Never edit by hand: anything under `data/`, `sources/` (read-only inputs), production checkpoints and embeddings.
- Never read or print: `.env*`, tokens, credentials.
- Ask first: anything that overwrites `data/`, training or embedding runs, ssh/scp/rsync to the server, new pip installs
  into `Mario`, deleting files. The guard hook enforces the destructive ones.

## Definition of done
- `.claude/gate.sh full` passes
- Production files in `data/` are unchanged unless the task was to regenerate them (compare before/after, or run into a scratch folder)
- README / docs updated as above, and any claim in them was checked against a fresh run

## Gotchas
- `normalize.py --dry-run` and `convert_clips.py --dry-run` STILL overwrite `data/normalize_report.csv` / `data/clips_report.csv`.
  Always pass `--report <scratch file>`. `lottie_tokens.py browser-index` ignores `--out` and writes `data/analysis/lottie_browser_index.json`.
  `lottie_actions.py` has no argument parser: any first word except `show` (even `-h`) rebuilds `data/embeddings/lottie_actions.csv`. `make_demo_rating_set.py` also has none and writes `data/eval/_demo`.
- `/Volumes/MyDrive` is macFUSE: `._*` sidecar files reappear after you delete them (ignored by `.gitignore`, skipped by the scripts).
  `scripts/__pycache__` is a ghost entry that breaks `find`, `tar` and `Path.is_dir` over `scripts/`: walk with `os.walk` and skip it.
  `.git/` also collects `._*` files, so `git fsck` reports `badRefName` for `refs/._heads`; harmless.
- `playwright` is installed in `/usr/local/bin/python3`, not in `Mario`, and its own browsers are not downloaded: use `p.chromium.launch(channel="chrome")` for a headless screenshot of a page.
- Port 8765 may already be running the user's own viewer server: check before starting or killing one.
- `scripts/lottie2d/rig_mesh.py` was truncated in both source copies and its tail was rebuilt on 2026-10-27; its `--k-max 4 --power 4`
  defaults are from the docstring, not known.
