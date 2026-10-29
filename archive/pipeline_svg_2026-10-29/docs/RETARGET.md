> **History:** the viewer pages named below (`viewer/compare.html`) were removed on 2026-10-28 (copy in `archive/retarget_compare_2026-10-28/`); the retargeter and its numbers are unchanged.

# Rule-based retargeter

`scripts/rigging/retarget.py` plays a clip made for one body (for example a human backflip) on another (for example a
quadruped). It uses no learning and no per-target tuning: the same rules run on every target. Written and
measured 2026-10-20; the numbers below come from `data/retarget_report.csv` and the commands at the end.

All geometry is in the normalized frame produced by `scripts/dataset/normalize.py`: metres, Y up, +Z forward, feet on
y = 0, the character's left is +X. Quaternions are w-first.

## Inputs and outputs

| | |
|---|---|
| Source clip | a clip GLB from `data/clips/mixamo/` (shared Mixamo clips) or a biped Objaverse clip, found by name or by keyword (`--query`, a TF-IDF-like match over the name and the three captions) |
| Target | a character folder in `data/characters/` that passes the role labelling and the stance audit (`--list-targets`, `data/retarget_targets.csv`): 2,745 bipeds (all 50 Mixamo) and 47 quadrupeds |
| Output | `data/characters/<target>/anims/rt__<clip>.glb` and an `rt__` entry in `anims.json` with the metrics (kept when `convert_clips.py` is rerun) |

Same frame rate and frame count as the source clip (30 fps).

## Roles

The method needs to know which joints are the hips, the spine path hips -> head, the two legs and the two arms.
They come from the dataset's clean joint names (`export/<dataset>/clean_joint_names.json`): `Hips`, `Head`,
`Left/Right Thigh, Shin, Foot, Toe` and `Left/Right Upper Arm, Forearm, Hand`. A limb chain may skip up to two
unlabelled helper joints between labelled ones (twist joints are common). A rig without these names cannot be a
target (449 lack hips or head names, 144 lack legs, 47 have the head not below the hips).

A target is a quadruped when its hips -> head vector is mostly horizontal. For a quadruped the "arms" are the
forelimbs. The stance audit rejects rigs whose rest pose is not standing (limb ends off the ground or above
their roots); of 213 quadrupeds, 94 can be labelled and 47 pass.

## Method

Let `Dp_s` be the source hips' world rotation change from the rest pose and `rho` the ratio of the hips heights
(target / source).

1. **Body rotation.** The target hips take the same world-space rotation change (`Dp = Dp_s`). A backflip is a
   rotation about the left-right axis, which is the same axis on every body. For a quadruped `Dp` is reshaped,
   see "Quadruped body pitch".
2. **Body position.** The horizontal path of the source hips is scaled by `rho * --stride-scale`. The height is
   `rest * (1 + --vertical-gain * (source height / source rest height - 1))`. The body rotates about its trunk
   centre (the mean of the four limb roots; the hips for a biped), not about the hips, so a long body does not
   sweep through the floor in a flip.
3. **Spine.** The source path hips -> head is resampled onto the target spine by normalized arc length. Each
   target joint takes the target body rotation plus its bend relative to the hips, with the bend scaled by
   `--spine-gain` (0.6) for a quadruped, whose trunk does not fold like a human's.
4. **Limbs.** Each limb segment's direction is compared with its neutral direction in the body frame (source
   legs: rest pose; source arms: hanging down, since Mixamo rests in a T-pose; target: its own rest pose), and
   the same swing is applied to the target segment. Legs drive the hind limbs. The forelimbs follow the arms
   (`--forelimbs arms`) or each foreleg copies the opposite hind leg (`legs`, a diagonal gait). `auto` (the
   default) picks `legs` when the hips travel more than 1.5 hips-heights, `arms` otherwise.
5. **Everything else** (tail, ears, jaw, fingers, shoulders, toes) is carried rigidly with its parent.
6. **Contacts.** A source foot is in contact when it is lower than 5% of the hips height above its rest height
   and slower than 0.6 hips-heights per second, for at least 3 frames (short runs are dropped, short gaps filled).
   Contacts are copied to the target feet: hind feet on the same side, forefeet on the opposite side.
7. **Foot pinning.** Each contact run gets a ground position (pin): the target's own stance under the body,
   turned by the body heading, plus how far the source foot is from its own stance scaled by `rho`. A run is cut
   (the foot takes a step) when the body has turned more than 35 degrees or travelled more than half a limb's
   reach. Analytic two-bone IK holds the foot on its pin, with a 5-frame ramp in and out and the foot kept flat
   (rest orientation turned by the heading; it flattens with the same weight). The bend direction follows the
   current knee or elbow and is pulled towards the rest-pose bend when the limb is nearly straight, so it does
   not flip. IK goes through any unlabelled joints between the limb's labelled joints.
8. **Reach and ground fix** (4 passes, in this order):
   - *Horizontal shift:* where the scaled source path would pull a limb root out of reach of a pinned foot (97%
     of the limb length), the root is moved just enough, smoothed (reported as `drift_max_m`).
   - *Vertical offset:* the root is lifted so that no joint that must stay off the floor is under it (limb ends
     at half their rest height, everything else at the floor) and lowered at most so that pinned feet stay
     reachable (98% of the limb length). Where both cannot hold, the floor wins and the difference shows up as
     a reach failure. After the IK the remaining penetration is fed back as extra lift.
   - *Clearance IK:* a grounded swinging foot that would go under the floor is lifted to it. Fingers and toes
     below a limb end hang at their rest offsets with the foot flat, so the end joint's floor is raised by that
     drop, and their raw pose does not push the body up. The weight eases in and out over 4 frames; a hard
     switch produced rotation spikes.

### Quadruped body pitch

A human body rotation is not workable on a body whose trunk lies horizontal, so for quadruped targets only the
pitch (rotation about the lateral axis, after the heading is taken out) is changed:

- while airborne, the rotation is stretched so each flight ends on a whole number of turns (a flip the human
  finishes after landing is finished in the air);
- while grounded, the remaining tilt is soft-limited with `L * tanh(angle / L)`: `--tilt-limit` 35 degrees,
  `--stance-tilt` 12 degrees when both feet are down;
- yaw and roll are left alone. `--no-shaping` turns all of this off.

## Parameters

| Option | Default | Effect |
|---|---|---|
| `--forelimbs` | auto | forelimb source: `arms`, `legs` (diagonal gait) or `auto` |
| `--stride-scale` | 1.0 | multiplies the scaled horizontal root travel |
| `--vertical-gain` | 1.0 | multiplies the source's vertical excursion (jump height) |
| `--tilt-limit` | 35 | quadruped: soft limit of body pitch while feet are down (degrees) |
| `--stance-tilt` | 12 | quadruped: the same, while both feet are down |
| `--spine-gain` | 0.6 | quadruped: spine bend relative to the hips |
| `--no-shaping` | off | transfer the body rotation as it is |
| `--no-ik` | off | skip foot pinning and the ground/reach correction |
| `--force` | off | try a target that fails the stance audit |

## Metrics

Written to each `rt__` entry in `anims.json` and, with `--batch`, to `data/retarget_report.csv`.

| Metric | Meaning |
|---|---|
| `skate_mean`, `skate_max` | horizontal speed (m/s) of a pinned foot during its contact run (the ramp frames before a step are excluded) |
| `reach_fail_frames` | pinned frames where the foot target is farther than 103% of the limb, or nearer than the minimum; the foot is clamped, so the leg is stretched or the foot slides |
| `penetration_max` | deepest joint below its floor (limb ends: half their rest height; others: 0), in metres |
| `lift_max`, `lift_mean` | extra root height the ground fix added (m); large values mean the body is long for its legs |
| `drift_max_m` | largest horizontal root shift made to keep pinned feet in reach |
| `rot_steps_over_60deg`, `rot_step_max_deg` | per-frame changes of a joint's local rotation above 60 degrees: a count of visible jumps (IK flips, spikes) |
| `root_travel_m`, `contact_frames`, `lift_binding_joints` | travel of the root, pinned frames, which joints forced the lift |

Reading them: the viewer (`viewer/compare.html`) colours sliding (<= 0.02 / 0.08 m/s), reach failures (<= 3 / 15
frames), penetration (<= 0.01 / 0.03 m), spikes (0 / <= 3) and lift (<= 0.1 / 0.3 m) green / amber / red.
These are the author's working thresholds, not values from a reference. The metrics are measured on the
retargeter's own output, so they say whether its constraints hold, not whether the motion looks right; that was
judged by eye in the viewer. There is no perceptual metric and no comparison with a learned retargeter yet.

## Results (2026-10-20)

Test sweep: four source clips on the first six usable quadruped targets in `data/retarget_targets.csv`
(trunk length / hips height 1.1-1.54; none is a crocodile, there is none in the data). Per cell: reach-fail
frames / mean sliding (m/s) / worst penetration (m); then spikes over 60 degrees.

| Target | Backflip | Walking | Jog_In_Circle | Jump_3 |
|---|---|---|---|---|
| `135d9d14` (trunk 1.54) | 16 / .004 / .005; 11 | 1 / .009 / 0; 0 | 8 / .009 / 0; 2 | 4 / .035 / .001; 4 |
| `31a8aaef` (1.44) | 35 / .001 / .001; 5 | 0 / .003 / .011; 0 | 46 / .060 / .005; 0 | 6 / .122 / .015; 2 |
| `50982988` (1.31) | 52 / .002 / .003; 13 | 2 / .013 / 0; 0 | 2 / .003 / 0; 0 | 2 / .023 / 0; 2 |
| `64c84fab` (1.10) | 2 / .001 / .002; 4 | 0 / .005 / .014; 0 | 13 / .032 / 0; 3 | 5 / .146 / .001; 1 |
| `664be084` (1.37) | 37 / .004 / 0; 4 | 0 / .001 / .012; 0 | 23 / .062 / .002; 0 | 5 / .075 / .010; 1 |
| `669a6881` (1.44) | 24 / .007 / .005; 4 | 3 / .010 / .048; 0 | 50 / .153 / .001; 0 | 5 / .139 / .037; 2 |

Over the 24 runs: median body lift 0.128 m (worst 1.01, a Backflip), mean sliding 0.009 m/s (worst 0.153),
reach-fail frames 5 (worst 52), penetration 0.002 m (worst 0.048), spikes 1.5 (worst 13). Walking is clean on
all six. The weak cases are the long flips (Backflip on long bodies needs a large root lift and still misses
reach for 16-52 frames), Jog_In_Circle on `31a8aaef` / `669a6881` (the legs are nearly straight at rest and the
scaled stride does not fit), and Jump_3 landings (sliding 0.07-0.15 m/s).

Other checks:

- **Identity:** a clip retargeted onto its own skeleton reproduces its hips and spine and the two upper
  segments of every limb: spine error <= 6e-8 m, limb direction error 0 degrees (`--self-test`, PASS).
- **Backflip on the stand-in quadruped** (`objaverse_a734df81...`, a rabbit-like creature): no penetration,
  0 reach failures, sliding <= 0.004 m/s, lift 0.16 m, 6 spikes (largest 81 degrees at take-off).
- **Biped targets** (human -> 8 usable bipeds, same 4 clips, 32 runs): 0 reach failures, sliding <= 0.017 m/s,
  penetration <= 0.029 m, spikes <= 6, median lift 0.

## What was tried (so it is not repeated)

- The stance-tilt limit (12 degrees with both feet down) cut Backflip reach failures from 158-216 to 16-71.
- Fingers and thumbs forced a body lift on every grounded frame (their raw pose hangs about 0.1 m below the
  wrist), pulling the legs out of reach. Exempting them from the pre-IK lift and using a digit-aware clearance
  IK halved the median reach failures (23 -> 5) and sliding (0.034 -> 0.009 m/s).
- Raising the lift floor of the end joint by the digit drop and dropping digits from the lift entirely: fewer
  jog reach failures on two targets but 3-9 cm of thumb penetration. Dropped.
- Flattening the foot twice as fast as the limb ramp: more spikes, no gain. Dropped.
- A hard on/off clearance IK produced 60-180 degree spikes; the 4-frame ease fixed it.
- On a rig with an unlabelled joint between knee and ankle (65 of 2,742 usable Objaverse targets) the IK wrote
  the foot's local rotation relative to the knee, leaving the foot about 180 degrees off; it flipped as the IK
  weight ramped in. `ik_limb` now goes through joints between the labelled ones (`64c84fab` Jog: 19 -> 3 spikes).

## Limitations

- Shoulder, clavicle and toe motion is carried rigidly (a shrug or toe bend is dropped). The tail has no
  secondary motion.
- Forelimb source is a rule (legs for travelling clips, arms otherwise), not learned or checked per clip.
- Long flips on long-bodied targets need a large root lift; there is no physics check.
- Targets need clean joint names; 640 characters have no usable body labels.
- Remaining glitches: 23-50 reach-fail frames in Jog_In_Circle on two targets, 16-52 in Backflip on long
  bodies, three 62-65 degree elbow steps on `64c84fab` on frames where the diagonal hind leg cannot reach, and
  thumb penetration up to 5 cm in Walking on `669a6881`.
- Quadruped sizes are a placeholder (all scaled to 0.9 m by `normalize.py`), so `rho` and with it stride and
  jump height are only as good as that scale.
- No comparison with a learned retargeter, no perceptual evaluation, no user study.

## Reproduce

```
conda activate Mario
python3 scripts/rigging/retarget.py --list-targets
python3 scripts/rigging/retarget.py --query backflip
python3 scripts/rigging/retarget.py --self-test
python3 scripts/rigging/retarget.py --source-clip Backflip --target objaverse_a734df8176ca47c68b0a90d1a17b5132
python3 scripts/rigging/retarget.py --batch Backflip,Walking,Jog_In_Circle,Jump_3 --batch-targets 6            # the sweep above
python3 scripts/rigging/retarget.py --batch Backflip,Walking,Jog_In_Circle,Jump_3 --batch-targets 8 --batch-class biped
python3 viewer/serve.py        # then http://127.0.0.1:8765/viewer/compare.html (add --write to the batch for GLBs)
```

`--batch` overwrites `data/retarget_report.csv` each time (the report in the repo is the quadruped sweep).
