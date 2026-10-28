# Shared helpers for pipeline hooks. Sourced, not executed.
# Expects the hook's stdin JSON in $INPUT.

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# json_get .path.to.key  → prints the value ("" if missing). Uses jq, falls back to python3.
json_get() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$INPUT" | jq -r "$1 // empty" 2>/dev/null
  else
    printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for k in sys.argv[1].lstrip(".").split("."):
    d = d.get(k) if isinstance(d, dict) else None
if d is None:
    print("")
elif isinstance(d, bool):
    print("true" if d else "false")
else:
    print(d)
' "$1"
  fi
}

json_escape() {
  local s="$1"
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}
  printf '%s' "$s"
}

# PreToolUse decision: emit_decision allow|deny|ask "reason"
emit_decision() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' \
    "$1" "$(json_escape "$2")"
}

# Message shown to the user (any event)
emit_system_message() {
  printf '{"systemMessage":"%s"}\n' "$(json_escape "$1")"
}
