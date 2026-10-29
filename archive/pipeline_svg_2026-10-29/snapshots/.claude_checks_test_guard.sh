#!/usr/bin/env bash
# Re-run after any change to .claude/hooks/guard-bash.sh: bash .claude/checks/test_guard.sh   (needs jq)
G="$(cd "$(dirname "$0")/.." && pwd)/hooks/guard-bash.sh"
fail=0
t() { # expected, command
  local exp="$1" cmd="$2" out dec
  out=$(jq -n --arg c "$cmd" '{tool_input:{command:$c}}' | CLAUDE_PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)" "$G")
  dec=$(printf '%s' "$out" | jq -r '.hookSpecificOutput.permissionDecision // "allow"' 2>/dev/null); [ -z "$dec" ] && dec=allow
  if [ "$dec" = "$exp" ]; then printf 'ok   %-6s %s\n' "$exp" "$cmd"; else printf 'FAIL want %-5s got %-5s : %s\n' "$exp" "$dec" "$cmd"; fail=1; fi
}
t deny  'git push --force origin feat/x'
t deny  'git push origin main'
t deny  'rm -rf ~'
t deny  'rm -rf sources/Datasets/'
t deny  'rm -r sources/Models/Qwen'
t deny  'cd /Volumes/MyDrive/disertation/Marionette && rm -rf sources/x'
t ask   'rm -rf data/embeddings'
t ask   'rm -rf data'
t ask   'rm -Rf data/characters/foo'
t ask   'python3 scripts/tools/clean_generated.py --delete'
t ask   'ssh gpu-server "nvidia-smi"'
t ask   'scp results.tar.gz user@host:/tmp/'
t ask   'rsync -a data/vlm/ user@host:/work/data/vlm/'
t ask   'rsync -a host:/work/out/ data/out/'
t ask   'python3 scripts/dataset/normalize.py --dry-run'
t ask   'python scripts/dataset/convert_clips.py --limit 3'
t ask   'git reset --hard HEAD~1'
t allow 'ls data'
t allow 'ls -la sources'
t allow 'rm -rf /tmp/scratch_x'
t allow 'rm notes/old.txt'
t allow 'rm -rf scratch_sources'
t allow 'rsync -a a/ b/'
t allow 'echo ssh is a protocol'
t allow 'grep -r data scripts'
t allow 'python3 scripts/dataset/normalize.py --dry-run --report /tmp/r.csv'
t allow 'python3 scripts/tools/clean_generated.py'
t allow 'python3 scripts/rigging/retarget.py --self-test'
t allow 'git push origin feat/x'
t allow 'git status'
exit $fail
