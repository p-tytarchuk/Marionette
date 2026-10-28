"""Remove the clips written by the generator page / generate_motion.py.

  python3 scripts/tools/clean_generated.py              # dry run (default): lists what would go
  python3 scripts/tools/clean_generated.py --delete     # really delete

Removes
  data/decoded/gen_*.glb                                   decoded human-skeleton proxies
  data/characters/*/anims/rt__dec_gen_*.glb                the retargeted clips
  the matching "rt__dec_gen_*" entries in each anims.json  (rewritten in place; all other entries are kept as they are)

Everything else is left alone: rt__dec_* clips from decoded_to_character.py (without "gen_"), rt__* batch clips, the
dataset's own clips. macOS ._* files are skipped.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PREFIX = "rt__dec_gen_"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true", help="really delete (default: dry run)")
    args = ap.parse_args()

    files = [p for p in sorted((DATA / "decoded").glob("gen_*.glb"))]
    entries = {}                                                     # anims.json path -> keys to drop
    unreadable = []                                                  # damaged anims.json: reported, never rewritten
    for adir in sorted(DATA.glob("characters/*/anims")):
        files += [p for p in sorted(adir.glob(PREFIX + "*.glb"))]
        f = adir / "anims.json"
        if f.exists():
            try:
                keys = [k for k in json.loads(f.read_text()) if k.startswith(PREFIX)]
            except (ValueError, OSError) as e:                       # JSONDecodeError and UnicodeDecodeError are ValueErrors
                unreadable.append((f, e))
                continue
            if keys:
                entries[f] = keys
    size = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files ({size / 1e6:.1f} MB), {sum(map(len, entries.values()))} anims.json entries in {len(entries)} files")
    for f, e in unreadable:
        print(f"WARNING: skipped unreadable {f.relative_to(ROOT)} ({type(e).__name__}); its entries are not counted and it is left untouched")
    for p in files[:5]:
        print("  ", p.relative_to(ROOT))
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more")
    if not args.delete:
        print("dry run, nothing deleted; --delete to delete")
        return

    for f, keys in entries.items():
        d = json.loads(f.read_text())
        for k in keys:
            d.pop(k)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        os.replace(tmp, f)
    for p in files:
        p.unlink()
    print("deleted")


if __name__ == "__main__":
    main()
