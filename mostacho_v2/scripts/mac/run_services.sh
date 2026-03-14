#!/usr/bin/env bash
# Levanta servicios de face y tf en terminales separadas (macOS).
set -euo pipefail

# Define rutas por defecto de entornos.
FACE_VENV="${MOSTACHO_FACE_VENV:-$HOME/venvs/kenya_insightface}"
TF_VENV="${MOSTACHO_TF_VENV:-$HOME/venvs/kenya_tf}"

# Lanza face_service en una ventana nueva de Terminal.
osascript <<APPLESCRIPT
 tell application "Terminal"
   activate
   do script "source '$FACE_VENV/bin/activate'; uvicorn mostacho.services.face_api:app --host 127.0.0.1 --port 8001"
 end tell
APPLESCRIPT

# Lanza tf_service en otra ventana nueva de Terminal.
osascript <<APPLESCRIPT
 tell application "Terminal"
   activate
   do script "source '$TF_VENV/bin/activate'; uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002"
 end tell
APPLESCRIPT

# Mensaje para operador.
echo "Servicios iniciados en nuevas ventanas de Terminal."
