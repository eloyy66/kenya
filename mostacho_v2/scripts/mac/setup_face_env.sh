#!/usr/bin/env bash
# Crea entorno para servicio de InsightFace.
set -euo pipefail

# Define raíz del repo y del proyecto mostacho.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MOSTACHO_ROOT="$REPO_ROOT/mostacho"

# Permite override por variable de entorno.
VENV_PATH="${MOSTACHO_FACE_VENV:-$HOME/venvs/kenya_insightface}"

# Crea entorno virtual.
python -m venv "$VENV_PATH"
# Activa entorno virtual.
source "$VENV_PATH/bin/activate"
# Actualiza herramientas base de empaquetado.
python -m pip install --upgrade pip setuptools wheel
# Instala dependencias del servicio face.
pip install -r "$MOSTACHO_ROOT/requirements/face_service.txt"
# Instala paquete local en modo editable.
pip install -e "$MOSTACHO_ROOT"

# Mensaje final de éxito.
echo "Entorno face listo en: $VENV_PATH"
