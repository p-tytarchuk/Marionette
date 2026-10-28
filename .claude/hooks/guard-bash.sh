#!/usr/bin/env bash
# PreToolUse(Bash): hard guardrails that hold no matter what any agent decides.
#   deny → force-push, push to main/master, rm -rf on / or home, recursive delete under sources/
#   ask  → discarding work (reset --hard, clean -f), production deploys,
#          recursive delete under data/, clean_generated --delete, ssh/scp/rsync to a remote host,
#          normalize.py / convert_clips.py without --report (they overwrite data/*_report.csv even with --dry-run)
# Extend PIPELINE_DEPLOY_REGEX in settings.json "env" to match your deploy commands.
INPUT=$(cat)
. "$(dirname "$0")/lib.sh"

CMD=$(json_get .tool_input.command)
[ -z "$CMD" ] && exit 0

S='[[:space:]]'
E='([[:space:]]|;|&|\||$)'   # end of a word

matches() { printf '%s' "$CMD" | grep -Eiq -- "$1"; }

PUSH="git${S}+push([[:space:]][^;&|]*)?"   # "git push" plus any args in the same command

if matches "${PUSH}${S}(--force(-with-lease)?|-[a-zA-Z]*f[a-zA-Z]*|\+[^[:space:]]+)${E}"; then
  emit_decision deny "Force-push is blocked by the pipeline guard. If it's really needed, the human should run it."
  exit 0
fi

if matches "${PUSH}(${S}|:)(main|master)${E}"; then
  emit_decision deny "Pushing to main/master is blocked. Push a feature branch and open a PR."
  exit 0
fi

if matches "rm${S}+(-[a-zA-Z]*${S}+)*-[a-zA-Z]*r[a-zA-Z]*${S}+(-[a-zA-Z]*${S}+)*(/|~|\\\$HOME|/\*)${E}"; then
  emit_decision deny "Recursive delete of / or the home directory is blocked."
  exit 0
fi

# --- Marionette rules -------------------------------------------------------------------------------
# A recursive rm: "rm" followed (anywhere in the same command) by -r / -R / -rf / --recursive
RM_REC="(^|[;&|]|${S})rm${S}+([^;&|]*${S})?(-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)"   # no trailing space: the path pattern below needs it

# sources/ holds symlinks to ../Datasets and ../Models. "rm -r sources/x/" follows a trailing-slash link
# into the read-only inputs (Qwen weights, the UniML3D export), which cannot be rebuilt from this folder.
if matches "${RM_REC}[^;&|]*(${S}|/)sources(/|${E})"; then
  emit_decision deny "Recursive delete under sources/ is blocked: it holds symlinks to the read-only inputs (../Datasets, ../Models). If it is really needed, the human should run it."
  exit 0
fi

# data/ is about 11 GB of generated results (embeddings, characters, Lottie parses) that take hours to rebuild.
if matches "${RM_REC}[^;&|]*(^|${S}|/)data(/|${E})"; then
  emit_decision ask "Recursive delete under data/: about 11 GB of generated results that take hours to rebuild. Confirm the exact path."
  exit 0
fi

if matches "clean_generated(\.py)?${S}+[^;&|]*--delete"; then
  emit_decision ask "clean_generated.py --delete removes generated clips and rewrites anims.json files. Run the dry run first and confirm."
  exit 0
fi

# Remote work runs on the GPU server: copying to or from it, or logging in, is a human decision.
if matches "(^|[;&|])${S}*(ssh|scp)${S}" || matches "(^|[;&|])${S}*rsync${S}[^;&|]*${S}[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?:[^[:space:]]"; then
  emit_decision ask "Remote access (ssh/scp/rsync to a host): server jobs and syncs of data/ need confirmation."
  exit 0
fi

# normalize.py and convert_clips.py write their --report CSV even with --dry-run (2026-10-27: this overwrote
# data/normalize_report.csv and data/clips_report.csv). Without --report they clobber the production reports.
if matches "scripts/dataset/(normalize|convert_clips)\.py" && ! matches "--report"; then
  emit_decision ask "normalize.py / convert_clips.py overwrite data/normalize_report.csv or data/clips_report.csv even with --dry-run. Pass --report <scratch file> or confirm."
  exit 0
fi
# ------------------------------------------------------------------------------------------------------

if matches "git${S}+(reset${S}+--hard|clean${S}+-[a-zA-Z]*f|checkout${S}+--${S}+\.|restore${S}+\.)"; then
  emit_decision ask "This discards uncommitted work. Confirm it's intended."
  exit 0
fi

DEPLOY_RE="${PIPELINE_DEPLOY_REGEX:-(deploy|release|promote).*(prod|production)|terraform${S}+(apply|destroy)|kubectl${S}+.*--context[= ]?[^[:space:]]*prod|helm${S}+(upgrade|install).*prod}"
if matches "$DEPLOY_RE"; then
  emit_decision ask "Production deploys need explicit human approval (staging + smoke checks first)."
  exit 0
fi

exit 0
