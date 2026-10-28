#!/bin/sh
# Two-stage Qwen parse of LottieAnimation-660K. Run it in a terminal panel to see the tqdm bars.
#   stage 1: Qwen3-VL-2B on every row whose caption/name has a person or animal word  -> output/stage1_2b/animation/
#   stage 2: Qwen3-VL-8B on the rows stage 1 called relevant                          -> output/animation/  (final table)
# Safe to stop with Ctrl-C and rerun: finished shards are skipped and a half-done shard continues from its partial file.
D="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$D/../scripts" ]; then cd "$D/.." || exit 1; else cd "$D/../.." || exit 1; fi   # server: flat scripts/; Mac: scripts/lottie2d/
PY="${PY:-$HOME/envs/marionette/bin/python}"
"$PY" "$D/vlm_tables.py" --datasets animation --stage "${STAGE:-both}" "$@"
[ "${STAGE:-both}" != 1 ] && "$PY" "$D/vlm_tables.py" --merge   # output/animation.jsonl
