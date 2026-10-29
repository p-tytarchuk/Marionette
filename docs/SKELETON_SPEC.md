# Role-labelled skeleton spec (`skeleton-roles/1`)

Written 2026-10-28. Code: `scripts/rigging/skeleton_spec.py` (validate, classify, modes, `--self-test`). Schema: `docs/skeleton_spec/schema.json`. Examples: `docs/skeleton_spec/examples/`.

## Why

The shape stage hands the motion stage a 2D skeleton, drawn at any yaw. The movement that looks right (walk, flap, slither) depends on which **carriers** the body has: legs, wings, a
single trunk chain, a tail. `docs/SKELETON_CLASSES.md` showed that these cannot be inferred reliably from a bare tree (winged recall 0.4; a snake looks like a fish or a rope; Lottie corpus
rigs carry almost no signal). So the shape stage states the roles itself, and everything downstream **counts** them: the class, the carriers and the movement-mode support are derived from the
parts, never guessed from geometry.

## The format

A JSON file. Every joint belongs to exactly one **part**; parts carry the meaning.

```json
{"spec": "skeleton-roles/1", "id": "fox_01", "view": {"yaw_deg": 0, "facing": "right"}, "declared_class": "quadruped",
 "parts":  [{"id": "trunk", "kind": "trunk"},
            {"id": "leg_hind_L", "kind": "leg", "side": "L", "girdle": "pelvic", "layer": "front"},
            {"id": "leg_hind_R", "kind": "leg", "side": "R", "girdle": "pelvic", "layer": "behind", "visible": false}, "..."],
 "joints": [{"id": "trunk1", "parent": null, "pos": [0, 0.30], "part": "trunk"},
            {"id": "leg_hind_L1", "parent": "trunk1", "pos": [0, 0.18], "part": "leg_hind_L"}, "..."]}
```

| part kind | meaning | rules |
|---|---|---|
| `trunk` | the spine from the root (pelvis) to the base of the head; for a snake or fish the whole body | exactly one; a chain; contains the root joint |
| `head` | neck and head | at most one; a chain |
| `tail` | a tail (secondary motion by default) | at most one; a chain |
| `leg` | a limb that carries the body on the ground | needs `side` L or R; `girdle` pelvic (hips) or pectoral (shoulders); a chain |
| `arm` | a limb that reaches (humanoid arms) | needs `side`, `girdle`; a chain |
| `wing` | a limb that flaps | needs `side`, `girdle`; a chain |
| `fin` | a fin | `side` L, R or C; a chain; **not a limb** for the class |
| `extra` | anything else that hangs on the body: ears, horns, hair, feathers, cloth | any subtree; secondary motion by default |

- **One root**, parent `null`, in the trunk. Every other joint has a parent; no cycles. A part's joints form one piece hanging from one attachment joint of another part.
- **Left and right are the body's own sides**, not the screen's, so the spec does not change with the yaw.
- **Hidden parts stay.** The far legs of a profile drawing exist on the body: keep them with `visible: false` (and `layer: "behind"`). Counting ignores `visible`, so the class cannot depend on the yaw.
- **Positions** are `[x, y]` in the rest pose, y up, any scale (consumers normalise). `view` records how the picture was drawn and is never used to count.
- `declared_class` is optional: the generator's own claim. If it disagrees with the parts, the parts win and the validator warns.

## What is derived

- **Class** (a count): any wing -> `winged` (a bird, a bat, a dragon and an angel are all winged); else 2 legs -> `biped`; 4 legs -> `quadruped`; no legs and no arms -> `limbless`
  (fins are not limbs, so a fish is limbless on purpose: one undulating chain); anything else -> `other` (a spider has 8 legs; the carriers still say what it can do).
- **Carriers:** legs (pelvic / pectoral), arms, wings, fins, tail, head, trunk length, trunk axis (upright or horizontal, from the geometry, never used for the class), which parts get
  follow-through motion, which parts are hidden.
- **Movement-mode support**, per mode, `native` / `stylised` / `unsupported` with a reason. A **proposal** to tune, from the carriers only:

| mode | native | stylised | unsupported |
|---|---|---|---|
| walk, run | 2 or more legs | | no legs |
| trot / gallop | 4 legs | 2 or 6+ legs (borrowed gait) | no legs |
| hop, jump, crawl | 2 or more legs | no legs: squash and stretch, the trunk pulls | |
| fly, glide | a wing | no wing: float in a flying pose, limbs trail, hair and cloth follow | |
| slither | no limbs, trunk of 3+ joints | limbs present: reduced trunk wave, limbs tucked | trunk too short |
| swim | a fin, or no limbs | paddle with the limbs | |
| wave | an arm | a wing or a lifted foreleg stands in | no arm, wing or foreleg |
| dance | 2 or more legs | trunk and head only | |
| idle | every body | | |

The cases we discussed come out as: a snake slithers (native) and does not walk (unsupported); a bear slithers (stylised: a reduced trunk wave, legs tucked); a person flies (stylised) and a bird
flies (native). **Whether a stylised pairing is shown or refused is a policy outside this table**; the table only reports the level.

## Checks that exist

`python scripts/rigging/skeleton_spec.py --self-test` (part of `.claude/gate.sh full`): the 9 examples (person, fox, snake, bird, dragon, bear in profile with hidden far legs, spider, fish, angel) are valid and
derive the intended class; 9 kinds of broken skeleton are each refused with their own message (cycle, duplicate id, two roots, missing trunk, bad side, disconnected part, branching chain, NaN position,
two trunks); the class rules, the hidden-limb rule and the mode cases above hold. The test was checked by sabotage: breaking a rule on purpose makes it fail.

`python scripts/rigging/skeleton_spec.py check-anymate` turns the role labels of the Anymate characters (`data/anymate/*/roles.json`, from `anymate_roles.py`) into specs: 192 of 192 give a valid spec and the
derived class is the characters' class (120 biped, 72 quadruped). Caveat: the role finder is given the class as input, so this shows that real rigs fit the spec and validate, not that the class was found
independently. 29 of the 221 characters have no roles and so no spec. `from_roles` is the bridge: the roles are subsets along a tree path (helper joints between them are skipped), so a limb is every joint on the path
from the trunk to its last role joint.

## Synthetic skeletons: `scripts/rigging/skeleton_synth.py`

Specs of the four classes plus "other", labelled by construction, deterministic from a seed: 29 body variants (humanoid, detailed humanoid with fingers and helper joints, chibi, tailed, long-armed, theropod;
dog, stocky, long-necked, short-legged, centaur; lying, coiled and raised snakes, fish, eel, worm; bird, bat, dragon, angel, winged insect, penguin; spider, insect, 3-legged, 1-legged, armed blob, octopus), each
with random proportions, joint counts, rest poses and a yaw. A body is built in 3D and drawn at a yaw, so near and far limbs, the drawing order and the far limbs that have no artwork (`visible: false`, only near
the profile) come from the geometry, while the class and the carriers never change with the yaw (tested). Commands: `write --n 2000` (to `data/synthetic/skeletons/skeletons.jsonl`), `sheet` (contact sheets
to look at; the viewer page `viewer/skeletons.html` shows them next to the real skeletons, at any yaw), `realism`, `--self-test` (in the gate). The generated specs validate and conform to the schema; `meta` records variant and seeds.

Plausibility was checked by looking at the sheets, which found and fixed zigzag legs that were too exaggerated and snake waves that were invisible from the side. The self-test was checked by sabotage: five
deliberate breaks, two of which first survived (wings without a near/far layer, a missing head), which closed two gaps in the test.

`realism` asks whether the structure features of `docs/SKELETON_CLASSES.md` see synthetic skeletons like real ones (balanced accuracy over the four classes):

| trained on | tested on | bal4 |
|---|---|---|
| synthetic | the 3D manifest (6 yaws) | 0.58 (biped 0.36, quadruped 0.63, limbless 0.75, winged 0.56) |
| synthetic | AnyTop's 70 creatures | 0.56 (limbless 0.92, winged 0.59, biped 0.12) |
| the 3D manifest | AnyTop's 70 creatures | 0.49 (the reference from SKELETON_CLASSES.md) |
| the 3D manifest | synthetic | 0.66 |

So synthetic data helps most where real data is thin (limbless, winged). It is weaker for humanoids: real bipeds have more branching joints and arms held higher, and AnyTop's "bipeds" (dinosaurs,
ostriches) are only partly covered by the theropod variant. Caveats: the pose weights (T / A / arms down) were adjusted once after seeing real-data feedback, so the transfer numbers are slightly optimistic; the
"other" class is not realistic (real "other" is marine, rigid objects, insects); there is no camera elevation, only yaw, so a flat coil never shows its spiral from the side.

## Not decided or not done

- The mode table is a first proposal; the stylise-or-refuse policy, the mode list and the thresholds are yours to set.
- `limits` (joint rotation limits) is in the schema but nothing reads it yet.
- No generator emits this format yet; the shape stage's output format is still open (`eval/README.md`). The eval plan should use it: the shape test's "roles found" check becomes "the spec validates and its derived class
  matches the prompt", and the motion test's `applies_to` becomes a body class plus the mode-support level.
