#!/usr/bin/env python3
"""Syntax check of every Python file in the project (the "fast" quality gate).

Compiles each file in memory (nothing is written, so no __pycache__) and reports
file:line for the first syntax error in each broken file. Exit 0 = all files compile.

Skipped on purpose:
  data/, sources/        generated outputs and read-only inputs (11 GB, symlinks outside the folder)
  .git, .claude/runs     tool state
  __pycache__            on the external drive scripts/__pycache__ is a ghost entry (a file, not a directory)
  ._*                    macOS/macFUSE sidecar files, not source
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
SKIP_DIRS = {"data", "sources", ".git", "__pycache__", "node_modules"}


def python_files(root: Path):
    for base, dirs, files in os.walk(root):
        rel = Path(base).relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("._")
                         and not (rel.parts[:2] == (".claude", "runs")))
        for name in sorted(files):
            if name.endswith(".py") and not name.startswith("._"):
                yield Path(base) / name


def main() -> int:
    checked, failures = 0, []
    for path in python_files(ROOT):
        checked += 1
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: cannot read as source: {type(exc).__name__}: {exc}")
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} of {checked} Python files do not compile")
        return 1
    print(f"{checked} Python files compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
