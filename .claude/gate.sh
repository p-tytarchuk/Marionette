#!/usr/bin/env bash
# Quality gate: the single source of truth for "is this change green?" in Marionette.
# Used by the Stop hook (fast) and by /quick (full).
#
# Usage: .claude/gate.sh fast | full | integration | format <file>
#   fast         syntax check of every .py file                       (about 0.2 s; runs when Claude finishes a turn with uncommitted changes)
#   full         fast + the retargeter identity self-test              (reads data/, writes nothing)
#   integration  same as full: the project has no e2e suite
#   format FILE  no formatter is configured: does nothing
#
# This project has no unit tests, linter or type checker. The checks below are the cheap, safe ones that exist.
# Deliberately NOT part of the gate:
#   normalize.py / convert_clips.py --dry-run   they still overwrite data/normalize_report.csv and data/clips_report.csv
#   anything that trains, embeds, calls an LLM or needs the GPU server

# The hooks may not inherit the conda activation: use the Mario env's interpreter explicitly.
PY="${MARIONETTE_PY:-/Users/paveltitarcuk/miniconda3/envs/Mario/bin/python3}"
[ -x "$PY" ] || PY=python3

FORMAT_CMD=""      # none configured
LINT_CMD="PYTHONDONTWRITEBYTECODE=1 $PY .claude/checks/syntax_check.py"
TYPECHECK_CMD=""   # none configured
TEST_CMD="PYTHONDONTWRITEBYTECODE=1 $PY scripts/rigging/retarget.py --self-test"
E2E_CMD=""         # none

set -o pipefail
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT" || exit 1

FAILED=""
RAN=0

run() {
  local label="$1" cmd="$2"
  [ -z "$cmd" ] && return 0
  RAN=$((RAN + 1))
  local start out rc
  start=$(date +%s)
  out=$(bash -c "$cmd" 2>&1)
  rc=$?
  if [ $rc -ne 0 ]; then
    printf '✗ %s failed (exit %d, %ss): %s\n' "$label" "$rc" "$(( $(date +%s) - start ))" "$cmd"
    [ -n "$out" ] && printf '%s\n' "$out" | tail -n 80 | sed 's/^/    /'
    FAILED="$FAILED $label"
  else
    printf '✓ %s (%ss)\n' "$label" "$(( $(date +%s) - start ))"
  fi
}

mode="${1:-full}"
case "$mode" in
  format)
    exit 0
    ;;
  fast)
    run lint "$LINT_CMD"; run typecheck "$TYPECHECK_CMD" ;;
  full)
    run lint "$LINT_CMD"; run typecheck "$TYPECHECK_CMD"; run test "$TEST_CMD" ;;
  integration)
    run lint "$LINT_CMD"; run typecheck "$TYPECHECK_CMD"; run test "$TEST_CMD"; run e2e "$E2E_CMD" ;;
  *)
    echo "usage: $0 fast|full|integration|format <file>" >&2; exit 64 ;;
esac

if [ "$RAN" -eq 0 ]; then
  echo "GATE SKIPPED ($mode): no commands configured in .claude/gate.sh"
  exit 0
fi
if [ -z "$FAILED" ]; then
  echo "GATE PASS ($mode)"
  exit 0
fi
echo "GATE FAIL ($mode):$FAILED"
exit 1
