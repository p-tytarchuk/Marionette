# Skeleton classes: biped / quadruped / limbless / winged (exploratory, 2026-10-28)

Question: can a 2D skeleton drawn at an unknown yaw say which carriers a body has (legs, wings, a single spine chain), so the motion stage can pick a
movement mode? Code: `scripts/rigging/skeleton_classes.py` (`build`, `eval`, `lottie-build`, `lottie-eval`). Outputs: `data/analysis/skeleton_classes/` (`report.md`,
`lottie_report.md`, `summary.json`, the skeletons as JSON). Nothing in the dataset was changed.

## Method

- **Skeletons (4,982):** 4,306 distinct skeletons of the 3D manifest, AnyTop's 70 Truebones creatures, 221 Anymate characters (predicted and original skeleton), and
  557 real Lottie rigs. All 3D ones are read the same way (a raw GLB's node hierarchy with ancestor transforms), so a classifier cannot learn the pipeline.
- **Labels:** the manifest `category` (bipedal, quadrupedal, avian, serpentine; "other" = insectoid, marine, articulated_rigid, uncertain). It is the dataset's own
  topology-based label, not a human judgement; contact sheets of random samples looked right for bipedal and quadrupedal, and the avian ones are real birds in very
  varied rest poses. Truebones groups (FLYING 11, BIPEDS 8, SNAKES 2, QUADROPEDS 35, MILLIPEDS 12) are curated by the AnyTop authors.
- **Features (names are never used):** one undirected tree per skeleton (minimum spanning tree on the parent links, the idea of `anymate_roles.build_tree`; 97 of
  221 Anymate predictions have a cycle or no root), short twigs pruned (fingers, toes), then 14 scale-free features: limb counts by where the limb ends (low / mid /
  high), chain fraction, `diam_frac` (longest path / total bone length: 1 for a snake), `arm_leg_ratio`, height of the centroid, aspect. The root plays no part.
- **Yaw:** each 3D skeleton is projected at yaws 0, 30, 60, 90, 120, 150 (0 = profile). Shallow decision trees (depth 4, balanced class weights); folds are grouped by skeleton.

## Results (bal4 = mean recall over the four target classes)

| test | with aspect | without aspect (yaw-free) |
|---|---|---|
| cross-validation inside the 3D manifest | 0.66 | 0.63 |
| train on the manifest, test on AnyTop's 70 creatures | 0.50 | 0.43 |
| held-out skeletons, tree trained on all yaws, by view | 0.67 to 0.73 | 0.65 to 0.66 |
| held-out skeletons, tree trained on the profile only, by view | 0.65 at yaw 0, **0.25 at yaw 90** | 0.48 to 0.56 |

Per class (cross-validation, with aspect): biped recall 0.89 / precision 0.98; quadruped 0.70 / 0.72; limbless 0.60 / 0.20; winged 0.45 / 0.14.
Anymate characters (biped and quadruped only): 0.87 accuracy with the dataset's skeleton, 0.75 with the predicted one.

## What it says

1. **Biped vs quadruped is readable from structure** on full-body skeletons (about 0.9 and 0.7 to 0.8 recall). The aspect feature helps but breaks when the yaw changes;
   train on all yaws, or drop it and lose about 0.05.
2. **Limbless has an exact signature, but it is not specific.** `diam_frac` is 1.0 for 15 of 20 true-snake views (misses: one tangled rig, one cobra seen head-on).
   Robot arms, chains, fish and tails share it, so the skeleton says "single undulating chain", not "snake". For choosing a movement mode that is the useful notion.
3. **Winged cannot be recovered from the rest-pose skeleton** with these features (recall 0.4): wings versus arms is a semantic difference that is not in the tree, and the
   bird rigs are in very different rest poses. Only 86 + 11 labelled examples exist.
4. **Lottie corpus rigs carry almost no signal** for two-leg vs four-leg (feature AUCs 0.4 to 0.6; a tree trained on them reaches 0.67; the 3D-trained tree sends 70%
   of them to "other"). They are mostly upper-body layer groupings, not full-body skeletons, so the corpus is not a usable source of skeleton labels.

## Caveats

Labels are a dataset heuristic; only 10 limbless and 97 winged skeletons exist; trees are shallow and untuned; 12 hand-chosen features; the two Truebones snakes make
"limbless on AnyTop" a 2-skeleton result. Treat the numbers as a first map of what structure can and cannot say.

## Consequence for the design

Inferring wings or a snake from a bare tree is unreliable, so the shape stage should emit the part roles itself (head, spine chain, legs, wings, tail) and the class
should be a count over those roles. The classifier above stays useful as a check and for rigs that come without roles. Training data for it can be synthetic: structure
is the class, so rigs of each type with random proportions, rest poses and yaws are cheap to generate.
