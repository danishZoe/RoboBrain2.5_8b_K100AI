#!/usr/bin/env bash
# Run selected public benchmarks sequentially; full datasets only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  echo 'Usage: bash eval.sh [cvbench] [refspatial]'
  echo 'No arguments: run cvbench then refspatial. Each uses its full dataset.'
  exit 0
fi
if [ "$#" -eq 0 ]; then set -- cvbench refspatial; fi
for name in "$@"; do
  case "$name" in cvbench|refspatial) ;; *) echo "Unsupported dataset: $name" >&2; exit 2;; esac
done
REPO_DIR="$(pwd)"
export WORK_DIR="${WORK_DIR:-$REPO_DIR}"
source ./env.sh
IMAGE_ROOT="${IMAGE_ROOT:-/opt/robo-image-v1}"
test -f "$IMAGE_ROOT/prepare.sh" || { echo "Missing $IMAGE_ROOT/prepare.sh" >&2; exit 1; }
bash "$IMAGE_ROOT/prepare.sh" "$REPO_DIR" "$WORK_DIR"
mkdir -p "$WORK_DIR/third_party/upstream_eval" "$WORK_DIR/outputs"
link_if_missing() {
  local source="$1" target="$2"
  test -e "$source" || { echo "Missing required input: $source" >&2; exit 1; }
  if [ ! -e "$target" ] && [ ! -L "$target" ]; then ln -s "$source" "$target"; fi
  test -e "$target" || { echo "Broken path: $target" >&2; exit 1; }
}
link_if_missing "$IMAGE_ROOT/eval_data/benchmarks" "$WORK_DIR/third_party/benchmarks"
link_if_missing "$IMAGE_ROOT/images_build/upstream_eval/FlagEvalMM" "$WORK_DIR/third_party/upstream_eval/FlagEvalMM"
for name in "$@"; do
  if [ "$name" = refspatial ]; then
    link_if_missing "$IMAGE_ROOT/images_build/upstream_eval/RoboRefer" "$WORK_DIR/third_party/upstream_eval/RoboRefer"
  fi
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$WORK_DIR/cache/torch" PIP_CACHE_DIR="$WORK_DIR/cache/pip"
mkdir -p "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
python3 -c 'import requests, numpy, PIL, pyarrow, qwen_vl_utils' || { echo 'Missing Python dependency in image' >&2; exit 1; }
curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo 'Service is not ready' >&2; exit 1; }
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$WORK_DIR/outputs/public_eval_$(date +%Y%m%d_%H%M%S)_$$}"
if [ -e "$EVAL_OUT_DIR" ]; then echo "Output path already exists: $EVAL_OUT_DIR" >&2; exit 1; fi
mkdir -p "$EVAL_OUT_DIR"
printf 'Git revision: %s\nDataset order: %s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)" "$*" | tee "$EVAL_OUT_DIR/run_info.txt"
for name in "$@"; do
  out="$EVAL_OUT_DIR/$name"
  mkdir -p "$out"
  echo "[eval] Starting full $name; output: $out"
  case "$name" in
    cvbench) python3 -u bench_public/run_cv_bench.py --concurrency 16 --out-dir "$out" --img-dir "$WORK_DIR/outputs/cvbench_eval_images" --run-id full-cvbench --no-resume 2>&1 | tee "$out/run.log" ;;
    refspatial) python3 -u bench_public/run_refspatial.py --variants upstream_scaled --splits Location,Placement --concurrency 16 --out-dir "$out" --run-id full-refspatial --no-resume 2>&1 | tee "$out/run.log" ;;
  esac
  python3 - "$name" "$out/summary.json" <<'PY'
import json, sys
name, path = sys.argv[1:]
with open(path, encoding='utf-8') as f:
    result = json.load(f)
if name == 'cvbench':
    score = result.get('scores', {})
    if result.get('n_planned') != 2638 or result.get('n_returned') != 2638 or result.get('terminal_states', {}).get('ok') != 2638 or score.get('returncode') != 0 or not isinstance(score.get('official_result'), dict):
        raise SystemExit('[eval] CV-Bench incomplete or scoring failed; see ' + path)
    print('[eval] CV-Bench accuracy:', score['official_result'].get('accuracy'))
else:
    splits = result.get('variants', {}).get('upstream_scaled', {}).get('splits', {})
    for name in ('Location', 'Placement'):
        item = splits.get(name, {})
        score = item.get('scores', {}).get('official_cli', {})
        if item.get('n') != 100 or item.get('terminal_states', {}).get('ok') != 100 or score.get('returncode') != 0 or score.get('official_evaluated') != 100 or score.get('official_accuracy') is None:
            raise SystemExit('[eval] RefSpatial ' + name + ' incomplete or scoring failed; see ' + path)
        print('[eval] RefSpatial', name, 'accuracy:', score['official_accuracy'])
PY
done
echo "[eval] All requested datasets complete: $EVAL_OUT_DIR"
