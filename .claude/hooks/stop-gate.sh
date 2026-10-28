#!/usr/bin/env bash
# Stop: when Claude tries to finish a turn with uncommitted changes, run the fast gate.
# If it fails, block the stop and hand Claude the failures so it fixes them.
# Capped at PIPELINE_GATE_MAX_RETRIES (default 3) per turn, then lets the turn end
# and tells the human, so a stubborn failure can never loop forever.
# Disable with env PIPELINE_GATE_ON_STOP=0 (e.g. in .claude/settings.local.json).
INPUT=$(cat)
. "$(dirname "$0")/lib.sh"

[ "${PIPELINE_GATE_ON_STOP:-1}" = "0" ] && exit 0
cd "$PROJECT_DIR" || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# A /ship run is orchestrating (workers mid-build, the critic loop owns the gate).
# Its lock is ignored once it's older than 24h, so a crashed run can't disable the gate forever.
LOCK="$PROJECT_DIR/.claude/runs/.ship-active"
if [ -f "$LOCK" ] && [ -z "$(find "$LOCK" -mmin +1440 2>/dev/null)" ]; then
  exit 0
fi

# Nothing changed → nothing to check (pure conversation, or already committed)
[ -z "$(git status --porcelain 2>/dev/null)" ] && exit 0

SID=$(json_get .session_id)
ACTIVE=$(json_get .stop_hook_active)
KEY=$(printf '%s' "${SID:-nosession}" | tr -c 'A-Za-z0-9_-' '_')
STATE_DIR="${TMPDIR:-/tmp}/claude-pipeline"
mkdir -p "$STATE_DIR" 2>/dev/null
COUNTER="$STATE_DIR/gate-$KEY"
MAX="${PIPELINE_GATE_MAX_RETRIES:-3}"

# A fresh stop (not one we caused) starts a new retry budget
[ "$ACTIVE" != "true" ] && echo 0 > "$COUNTER"
N=$(cat "$COUNTER" 2>/dev/null || echo 0)

OUT=$("$PROJECT_DIR/.claude/gate.sh" fast 2>&1)
RC=$?

if [ $RC -eq 0 ]; then
  rm -f "$COUNTER"
  exit 0
fi

N=$((N + 1))
if [ "$N" -gt "$MAX" ]; then
  rm -f "$COUNTER"
  emit_system_message "Quality gate still failing after $MAX fix attempts. Stopping so you can take a look (.claude/gate.sh fast)."
  exit 0
fi
echo "$N" > "$COUNTER"

{
  echo "Quality gate (fast) failed. Fix these before finishing (attempt $N/$MAX):"
  printf '%s\n' "$OUT" | tail -n 60
} >&2
exit 2
