#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-/Users/usuario/venvs/m22tf}"
DATASET_ROOT="${DATASET_ROOT:-/Users/usuario/kenya/db/artificialvision/minidatasetdmd}"
DATASET_ROOTS="${DATASET_ROOTS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/dmd_processed}"
FRAME_SAMPLING_SECONDS="${FRAME_SAMPLING_SECONDS:-0.75}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-subject}"
BACKBONE_NAME="${BACKBONE_NAME:-MobileNetV3Small}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-10}"
FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-10}"
FINE_TUNE_LAYERS="${FINE_TUNE_LAYERS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
FINE_TUNE_LR="${FINE_TUNE_LR:-0.0001}"
MIN_PRECISION="${MIN_PRECISION:-0.90}"
SEED="${SEED:-42}"
MAX_REP_SAMPLES="${MAX_REP_SAMPLES:-500}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
USE_HANDS_SUPPORT="${USE_HANDS_SUPPORT:-0}"
OVERWRITE="${OVERWRITE:-0}"
DOWNSAMPLE_MAJORITY="${DOWNSAMPLE_MAJORITY:-0}"
OVERSAMPLE_MINORITY="${OVERSAMPLE_MINORITY:-0}"
CACHE_DATASET="${CACHE_DATASET:-0}"
NO_CLASS_WEIGHTS="${NO_CLASS_WEIGHTS:-0}"
ALLOW_INT8_FALLBACK="${ALLOW_INT8_FALLBACK:-1}"
RUN_SANITY="${RUN_SANITY:-1}"
USE_NEW_TFLITE_CONVERTER="${USE_NEW_TFLITE_CONVERTER:-0}"

usage() {
  cat <<'EOF'
Uso:
  ./run_all.sh

Variables de entorno utiles:
  VENV_PATH=/Users/usuario/venvs/m22tf
  DATASET_ROOT=/Users/usuario/kenya/db/artificialvision/minidatasetdmd
  DATASET_ROOTS=/path/dataset1;/path/dataset2
  OUTPUT_DIR=/Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed
  FRAME_SAMPLING_SECONDS=0.75
  SPLIT_STRATEGY=subject|session|sequence
  BACKBONE_NAME=MobileNetV3Small|MobileNetV2
  FREEZE_EPOCHS=10
  FINE_TUNE_EPOCHS=10
  FINE_TUNE_LAYERS=40
  MIN_PRECISION=0.90

Flags por env (0/1):
  INSTALL_DEPS=1
  USE_HANDS_SUPPORT=1
  OVERWRITE=1
  DOWNSAMPLE_MAJORITY=1
  OVERSAMPLE_MINORITY=1
  CACHE_DATASET=1
  NO_CLASS_WEIGHTS=1
  ALLOW_INT8_FALLBACK=1
  RUN_SANITY=1
  USE_NEW_TFLITE_CONVERTER=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[ERROR] No existe VENV_PATH: $VENV_PATH" >&2
  exit 1
fi

source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

dataset_roots=()
if [[ -n "$DATASET_ROOTS" ]]; then
  IFS=';' read -r -a dataset_roots <<< "$DATASET_ROOTS"
else
  dataset_roots=("$DATASET_ROOT")
fi
dataset_root_args=()
for root in "${dataset_roots[@]}"; do
  if [[ -n "$root" ]]; then
    dataset_root_args+=(--dataset-root "$root")
  fi
done
if [[ "${#dataset_root_args[@]}" -eq 0 ]]; then
  echo "[ERROR] No hay dataset roots validos. Usa DATASET_ROOT o DATASET_ROOTS." >&2
  exit 1
fi
echo "Dataset roots: ${dataset_roots[*]}"

run_step() {
  local name="$1"
  shift
  echo
  echo "========== $name =========="
  echo "CMD: $*"
  "$@"
}

if [[ "$INSTALL_DEPS" == "1" ]]; then
  run_step "Install dependencies" python -m pip install -r "$SCRIPT_DIR/requirements/distraction_pipeline.txt"
  run_step "Install package editable" python -m pip install -e "$SCRIPT_DIR"
fi

extract_cmd=(
  python -m mostacho.training.dmd_distraction.extract_frames
  "${dataset_root_args[@]}"
  --processed-output-dir "$OUTPUT_DIR"
  --frame-sampling-seconds "$FRAME_SAMPLING_SECONDS"
)
if [[ "$USE_HANDS_SUPPORT" == "1" ]]; then
  extract_cmd+=(--use-hands-support)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  extract_cmd+=(--overwrite)
fi
run_step "1) Extract frames + labels.csv" "${extract_cmd[@]}"

split_cmd=(
  python -m mostacho.training.dmd_distraction.build_splits
  --processed-output-dir "$OUTPUT_DIR"
  --split-strategy "$SPLIT_STRATEGY"
  --seed "$SEED"
)
if [[ "$DOWNSAMPLE_MAJORITY" == "1" ]]; then
  split_cmd+=(--downsample-majority)
fi
if [[ "$OVERSAMPLE_MINORITY" == "1" ]]; then
  split_cmd+=(--oversample-minority)
fi
run_step "2) Build leakage-safe splits" "${split_cmd[@]}"

if [[ "$RUN_SANITY" == "1" ]]; then
  run_step "3) Sanity check" \
    python -m mostacho.training.dmd_distraction.sanity_check \
    "${dataset_root_args[@]}" \
    --processed-output-dir "$OUTPUT_DIR"
fi

train_cmd=(
  python -m mostacho.training.dmd_distraction.train
  --processed-output-dir "$OUTPUT_DIR"
  --backbone-name "$BACKBONE_NAME"
  --batch-size "$BATCH_SIZE"
  --freeze-epochs "$FREEZE_EPOCHS"
  --fine-tune-epochs "$FINE_TUNE_EPOCHS"
  --fine-tune-layers "$FINE_TUNE_LAYERS"
  --learning-rate "$LEARNING_RATE"
  --fine-tune-learning-rate "$FINE_TUNE_LR"
  --seed "$SEED"
)
if [[ "$CACHE_DATASET" == "1" ]]; then
  train_cmd+=(--cache-dataset)
fi
if [[ "$NO_CLASS_WEIGHTS" == "1" ]]; then
  train_cmd+=(--no-class-weights)
fi
run_step "4) Train model" "${train_cmd[@]}"

run_step "5) Evaluate + threshold tuning" \
  python -m mostacho.training.dmd_distraction.evaluate \
  --processed-output-dir "$OUTPUT_DIR" \
  --min-precision "$MIN_PRECISION"

export_cmd=(
  python -m mostacho.training.dmd_distraction.export_tflite
  --processed-output-dir "$OUTPUT_DIR"
  --max-representative-samples "$MAX_REP_SAMPLES"
)
if [[ "$ALLOW_INT8_FALLBACK" == "1" ]]; then
  export_cmd+=(--allow-int8-fallback)
fi
if [[ "$USE_NEW_TFLITE_CONVERTER" == "1" ]]; then
  export_cmd+=(--use-new-converter)
fi
run_step "6) Export TFLite (float + int8)" "${export_cmd[@]}"

echo
echo "Pipeline completado."
echo "Output dir: $OUTPUT_DIR"
echo "Best model: $OUTPUT_DIR/models/dmd_distraction_best.keras"
echo "Metrics: $OUTPUT_DIR/reports/metrics.json"
echo "TFLite float: $OUTPUT_DIR/tflite/model_float32.tflite"
echo "TFLite int8: $OUTPUT_DIR/tflite/model_int8.tflite"
