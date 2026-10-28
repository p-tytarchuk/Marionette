"""Verify a dataset copy against its source: same files and sizes, then (--bytes) every file byte-for-byte, and (--zstd) that each *.zst decompresses.

    ~/envs/marionette/bin/python scripts/verify_copy.py SRC_DIR DST_DIR [--bytes] [--zstd] [--workers 2]
Read-only on both sides. `._*` files (macOS resource forks) are ignored. Symlinks must point at the same relative target.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def scan(root):
    out = {}
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.startswith("._"):
                continue
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = os.readlink(p) if os.path.islink(p) else os.path.getsize(p)
    return out


def same_bytes(a, b, chunk=8 << 20):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bytes", action="store_true")
    ap.add_argument("--zstd", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    s, d = scan(a.src), scan(a.dst)
    bad = [f"MISSING {k}" for k in s if k not in d] + [f"EXTRA {k}" for k in d if k not in s]
    bad += [f"DIFFERENT size/target {k}: {s[k]} vs {d[k]}" for k in s if k in d and s[k] != d[k]]
    print(f"{len(s)} source files, {len(d)} copied; structure problems: {len(bad)}", flush=True)
    files = [k for k in s if k in d and not isinstance(s[k], str) and s[k] == d[k]]
    total = sum(s[k] for k in files)
    done, t0 = 0, time.time()

    def check(k):
        ok = same_bytes(os.path.join(a.src, k), os.path.join(a.dst, k)) if a.bytes else True
        z = True
        if ok and a.zstd and k.endswith(".zst"):
            z = subprocess.run(["zstd", "-tqf", os.path.join(a.dst, k)], capture_output=True).returncode == 0
        return k, ok, z

    if a.bytes or a.zstd:
        with ThreadPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(check, k) for k in files]):
                k, ok, z = fut.result()
                done += s[k]
                if not ok:
                    bad.append(f"BYTES DIFFER {k}")
                if not z:
                    bad.append(f"ZSTD FAILED {k}")
                print(f"\r{done / total:6.1%} {done / 1e9:7.1f}/{total / 1e9:.1f} GB  {time.time() - t0:6.0f}s  problems {len(bad)}", end="", flush=True)
        print()
    print("RESULT:", "OK, copy is identical" if not bad else f"{len(bad)} PROBLEMS")
    for b in bad[:50]:
        print(" ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
