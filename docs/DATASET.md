# Dataset guide

Short guide to the data available for Marionette (as audited 2026-10-19). Nothing here is
copied; `sources/` holds symlinks to `../Datasets` and `../Models`.

## Sources

| Symlink in `sources/` | Points to | What it is |
|---|---|---|
| `unimate` | `Datasets/dataset` | UniML3D: motion clips plus rigged meshes (Mixamo, Objaverse-XL, Truebones annotations) |
| `templates_mesh2motion` | `Datasets/glb_mixamo` | 9 template rigs (bird, dragon, fish, fox, horse, human, kaiju, snake, spider), with skinned mesh and named clips |
| `templates_skeleton_only` | `Datasets/glb` | the same 9 rigs, skeleton and clips only (kaiju also has a mesh) |
| `unique_skeletons` | `Datasets/unique` | index of 3,456 unique skeletons (JSON + tree), no mesh or motion |
| `models` | `Models` | Qwen3-VL-2B, InternVL3.5-2B, Qwen2.5 1.5B/3B, SAM 2/2.1/3 |

macOS `._*` files are everywhere on this drive. Always skip names starting with `._`.

## `sources/unimate` layout

```
clips.csv                         13,769 clips: dataset, clip, object_type, frames, fps, category, caption, ...
export/<mixamo|objaverse|truebones>/
  motions/<clip>.npz              clip in source rig (objaverse is hash-sharded: motions/<xx>/<file>)
  motion_captions*.json           caption, generic caption, detailed caption
  category_groups.json            body-plan category per skeleton
  clean_joint_names.json          unified joint vocabulary
  filtered_clips.txt              clips to skip
canonical_assets/mixamo/*.glb     110 characters, 22-joint core, rigged
canonical_assets/objaverse/*.glb  6,590 rigged meshes
features/mixamo/                  partial: 274 of 2,162 clips
```

Missing locally: `skeletons.csv`, Truebones motions/meshes (not redistributable), `features/` for
objaverse/truebones.

Clip NPZ fields: `rest_local_pos/rot`, `anim_local_pos/rot (T,J,3|4)`, `offsets`, `parents`, `names`,
`skin_matrix`, `fps` (30). Quaternions are w-first, Y-up.

Class names are `bipedal`, `quadrupedal`, `avian`, `insectoid`, `marine`, `serpentine`,
`articulated_rigid`, `uncertain`.

## What you can count on

| | Mixamo | Objaverse | Templates |
|---|---|---|---|
| Characters / skeletons | 110 meshes (1 shared rig) | 6,590 meshes, 7,355 rigs | 9 |
| Clips | 2,317 | 10,355 | 5 to 178 per rig |
| Bipedal / quadrupedal | all biped | 9,097 / 624 clips | human / fox, horse |
| Triangles (median) | 26k | 8.5k | 1k to 14k |
| Meshes over 20k triangles | 60 of 110 | 2,033 of 6,590 | 0 |

Quadrupeds are the weak spot (285 unique skeletons, 624 clips, fewer once meshes are filtered).
Biped data is plentiful, but 341 characters share one rig, so dedupe by skeleton.

## Gaps against the Marionette requirements

| Requirement | Status |
|---|---|
| Rest-pose mesh, under about 20k triangles | partial, filter by triangle count |
| Skeleton + skin weights | present (weights sum to 1 in the checked files) |
| Class label biped/quadruped | coarse VLM labels; avian, insectoid, marine, serpentine need mapping or exclusion |
| 2 to 3 shape captions | **missing** (existing captions are motion only) |
| Motion captions | present (3 styles), needs review |
| Loop flag | **missing** |
| Units in meters | **no**: canonical meshes are normalized to skeleton diameter 2 |
| Same rest pose per class | **no**: authored rest poses differ |
| Root motion kept in clip | not verified |
| `meta.json`, `anims.json`, `joint_map.json` | `meta.json` written for all 3,570 characters 2026-10-22 (`scripts/dataset/make_splits.py`; `shape_captions` empty, `joint_map` null except Mixamo); `anims.json` exists from `convert_clips.py` (loop is a guess); only `joint_maps/mixamo_biped.json` exists |
| Train/val split by character | `data/splits/{train,val}.txt` 2026-10-22, 10% by skeleton group (Mixamo per character, one shared rig); independent of the clip split in `motion_ae.py` |

## Target layout (in `data/`)

```
data/
  joint_maps/<rig>.json           e.g. mixamo_biped.json
  characters/<id>/model.glb
  characters/<id>/meta.json       id, class, joint_map, shape_captions[2-3]
  characters/<id>/anims/*.glb|bvh
  characters/<id>/anims/anims.json   per clip: fps, loop, motion_captions[1-3]
  splits/                         train/val lists; hold out ~10% of characters (not clips)
```

Conventions to enforce on every file: meters, Y-up, facing +Z, feet at y = 0, one rest pose per
class (pick T- or A-pose), fps recorded, root motion kept.

## Suggested order of work

1. Pick the conventions above and write `scripts/dataset/normalize.py` (scale to meters, orient, ground).
2. Filter meshes to under 20k triangles and dedupe by skeleton (`sources/unique_skeletons/index.csv`).
3. Build `joint_maps/` per source rig (Mixamo, mesh2motion templates, plus Objaverse rigs via `clean_joint_names.json`).
4. Render characters and draft shape captions with `sources/models/Qwen3-VL-2B-Instruct`, then review.
5. Write `meta.json` and `anims.json` (loop flag from clip name: walk, run, idle = true).
6. Create train/val splits by character.

## Embeddings (added 2026-10-20)

`data/embeddings/` holds CLIP text embeddings of all captions, joint-set and role features of the clips, and the motion embeddings
(semantic, any body; decodable latent, biped and quadruped). See `docs/EMBED.md`. CLIP weights are in the Hugging Face cache
(`openai/clip-vit-base-patch32`), not in the project.
