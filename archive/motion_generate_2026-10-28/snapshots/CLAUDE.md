# Marionette

Dissertation project: generating animatable characters (rest-pose mesh, skeleton, skin weights) and motion for biped and
quadruped classes. Three branches: 3D motion retargeting and generation, auto-rigging (Anymate pilot), and Lottie 2D motion
(text -> motion). This is research code: "works" means the metrics and figures hold, not that tests pass.
Start with `README.md` (Layout, Status, Next steps) and `docs/DATASET.md`. `docs/pipeline.html` is the map of every step.

## Stack
Python 3.11 in the conda env `Mario` (`environment.yml`; PyTorch comes from conda-forge on purpose: the pip wheel aborts
with "OMP: Error #15"). Three.js/HTML viewers in `viewer/`. Heavy jobs (Qwen labelling, training on more data) run on a GPU
server; their outputs are copied into `data/`. Nothing here is deployed.

## Commands
Run everything from the project root: `python scripts/<group>/<name>.py` (the groups are listed in README "Layout").
- Env: `conda activate Mario`
- Viewer: `python viewer/serve.py` (menu at http://127.0.0.1:8765/viewer/home.html, `--no-open` to skip the browser)
- Quality gate: `.claude/gate.sh fast` (syntax check of every .py, about 1 s) and `.claude/gate.sh full` (+ the self-tests of `retarget.py`, `skeleton_spec.py`, `skeleton_synth.py` and the generator gate in `generate_motion.py`)
- Check the guard hook after editing it: `bash .claude/checks/test_guard.sh`
- Most scripts take `-h`; `normalize.py`, `convert_clips.py`, `clean_generated.py` have a dry run (see Gotchas)
- No build, deploy or CI.

## Architecture map
- `scripts/{dataset,rigging,motion,text,generate,captioning,lottie2d,tools}/` : all pipeline code, 8 groups, script names unique
- `lottie/` : Lottie curve/primitive classes used by the lottie2d scripts (copied from Loomie; nothing imports ../Loomie)
- `viewer/` : web pages and `serve.py`, the one server
- `docs/` : written records (`RETARGET.md`, `EMBED.md`, `MOBILE.md`, `LOTTIE_PARSE.md`), figures, `pipeline.html`
- `eval/` : plan for testing the finished text -> Lottie system on ~10k prompts: `shape/` (mesh + skeleton) and `motion/`, each
  scored by deterministic checks, a VLM judge and human raters (`human_eval.md`); protocol, schemas and rubrics only, no code
  yet. Outputs go to `data/eval/`
- `notes/` : the user's own notes, not written by scripts
- `data/` (about 11 GB, untracked) : everything generated. `sources/` (untracked) : symlinks to `../Datasets` and `../Models`

## Conventions
- Scripts find the project root themselves (`ROOT = Path(__file__).resolve().parents[2]`) and put every `scripts/<group>`
  on `sys.path`, so they import each other by bare module name (`import retarget`). Copy that line from a neighbouring
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
- `/Volumes/MyDrive` is macFUSE: `._*` sidecar files reappear after you delete them (ignored by `.gitignore`, skipped by the scripts).
  `scripts/__pycache__` is a ghost entry that breaks `find`, `tar` and `Path.is_dir` over `scripts/`: walk with `os.walk` and skip it.
  `.git/` also collects `._*` files, so `git fsck` reports `badRefName` for `refs/._heads`; harmless.
- `playwright` is installed in `/usr/local/bin/python3`, not in `Mario` (portrait rendering via `shape_retrieval.py render` needs it plus `viewer/serve.py`).
- Port 8765 may already be running the user's own viewer server: check before starting or killing one.
- Anymate joint prediction differs from run to run (unseeded torch).
- `scripts/lottie2d/rig_mesh.py` was truncated in both source copies and its tail was rebuilt on 2026-10-27; its `--k-max 4 --power 4`
  defaults are from the docstring, not known. `text_frontend.py` demo `main()` crashes in the hybrid column (pre-existing).
