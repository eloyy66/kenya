#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# 1) Entorno Python
source /Users/usuario/venvs/m22tf/bin/activate
cd "$PROJECT_ROOT"

# 2) Variables base
export PYTHONPATH="/Users/usuario/kenya/mostacho_tflite2.2/src"
export MOSTACHO_ROOT="/Users/usuario/kenya/mostacho_tflite2.2"
export MOSTACHO_ARTIFACTS_ROOT="/Users/usuario/kenya/mostacho_tflite2.2/artifacts"
export MOSTACHO_FACE_PYTHON="/Users/usuario/venvs/m22if/bin/python"
export MOSTACHO_TF_PYTHON="/Users/usuario/venvs/m22tf/bin/python"
export MOSTACHO_MICROSLEEP_CONFIRM_SECONDS="0.9"

# 3) Modelos runtime
export MOSTACHO_EYE_STATE_TFLITE="/Users/usuario/kenya/mostacho_tflite2.2/artifacts/models/eye_state_model_float32.tflite"
export MOSTACHO_DROWSINESS_TFLITE="/Users/usuario/kenya/mostacho_tflite2.2/artifacts/models/drowsiness_vision_int8.tflite"
export MOSTACHO_DISTRACTION_TFLITE="/Users/usuario/kenya/mostacho_tflite2.2/artifacts/models/distraction_model_float32.tflite"

# 4) Distraccion (always run + crop)
export MOSTACHO_DISTRACTION_ENABLED="1"
export MOSTACHO_DISTRACTION_ALWAYS_RUN="1"
export MOSTACHO_DISTRACTION_THRESHOLD="0.55"
export MOSTACHO_DISTRACTION_CROP_SCALE_X="1.80"
export MOSTACHO_DISTRACTION_CROP_SCALE_Y="2.20"
export MOSTACHO_DISTRACTION_CROP_CENTER_Y_OFFSET="0.20"

usage() {
  cat <<'EOF'
Uso:
  ./scripts/run_runtime.sh api
  ./scripts/run_runtime.sh monitor [camera_index]
  ./scripts/run_runtime.sh all [camera_index]
  ./scripts/run_runtime.sh env
  ./scripts/run_runtime.sh <comando personalizado...>

Ejemplos:
  ./scripts/run_runtime.sh api
  ./scripts/run_runtime.sh monitor 1
  ./scripts/run_runtime.sh all 1
  ./scripts/run_runtime.sh python -m mostacho.realtime_monitor --api-url http://127.0.0.1:8002
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

mode="$1"

if [[ "$mode" == "env" ]]; then
  echo "Entorno cargado:"
  echo "  PYTHONPATH=$PYTHONPATH"
  echo "  MOSTACHO_ROOT=$MOSTACHO_ROOT"
  echo "  MOSTACHO_ARTIFACTS_ROOT=$MOSTACHO_ARTIFACTS_ROOT"
  echo "  MOSTACHO_FACE_PYTHON=$MOSTACHO_FACE_PYTHON"
  echo "  MOSTACHO_TF_PYTHON=$MOSTACHO_TF_PYTHON"
  echo "  MOSTACHO_EYE_STATE_TFLITE=$MOSTACHO_EYE_STATE_TFLITE"
  echo "  MOSTACHO_DROWSINESS_TFLITE=$MOSTACHO_DROWSINESS_TFLITE"
  echo "  MOSTACHO_DISTRACTION_TFLITE=$MOSTACHO_DISTRACTION_TFLITE"
  exit 0
fi

if [[ "$mode" == "api" ]]; then
  exec uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002
fi

if [[ "$mode" == "monitor" ]]; then
  camera_index="${2:-1}"
  exec python -m mostacho.realtime_monitor \
    --api-url http://127.0.0.1:8002 \
    --camera-backend AVFOUNDATION \
    --camera-index "$camera_index" \
    --predict-every 2 \
    --max-fps 12
fi

if [[ "$mode" == "all" ]]; then
  camera_index="${2:-1}"
  uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002 &
  api_pid=$!
  trap 'kill "$api_pid" >/dev/null 2>&1 || true' EXIT INT TERM
  sleep 2
  python -m mostacho.realtime_monitor \
    --api-url http://127.0.0.1:8002 \
    --camera-backend AVFOUNDATION \
    --camera-index "$camera_index" \
    --predict-every 2 \
    --max-fps 12
  exit 0
fi

# Comando personalizado dentro del mismo entorno
exec "$@"

