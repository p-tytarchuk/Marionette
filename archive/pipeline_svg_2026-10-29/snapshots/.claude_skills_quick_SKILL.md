---
name: quick
description: Fast path for small, well-scoped changes to Marionette scripts, viewers or docs (about 3 files, no new data format). One agent, same quality gate. Use when the user types /quick.
disable-model-invocation: true
argument-hint: <small task>
---

# /quick: small-change path

Task: $ARGUMENTS

1. **Restate** the task in one sentence and name the files you expect to touch.
   If that is more than ~3 files, changes a file format under `data/`, or needs a training / embedding / server run,
   stop and tell the user: this path is not meant for it.
2. **Locate.** Read the relevant script and the docs that describe it (README "Status", the matching `docs/*.md`).
   Reuse existing helpers and the sys.path line from neighbouring scripts (see CLAUDE.md Conventions).
3. **Reproduce first when it is a bug.** There is no test suite, so reproduce on the smallest input that shows it:
   `retarget.py --self-test` for the retargeter, a dry run, or a one-clip / one-character run into a scratch folder
   (never into `data/` production files; pass `--report <scratch>` where the script writes one).
4. **Implement** the smallest change that satisfies the task and the CLAUDE.md conventions.
5. **Gate:** run `.claude/gate.sh full`. Fix until it passes (max 3 attempts, then stop and report what is failing).
   Then re-run the reproduction from step 3 and read its output; the gate does not cover behaviour.
6. **Self-review the diff** (`git diff`) against this checklist:
   - Does it do what was asked, and nothing else?
   - Are production files in `data/` unchanged (nothing overwritten by a dry run or a test)?
   - Any debug leftovers, hard-coded paths, secrets, commented-out code?
   - If behaviour or results changed: are README Status / the matching `docs/*.md` / `docs/pipeline.html` updated, with numbers from a run you did?
7. **Report** in at most 5 lines: what changed, gate result, what you re-ran and saw. Ask before committing or pushing.
