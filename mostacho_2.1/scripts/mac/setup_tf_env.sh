#!/usr/bin/env bash
# Crea entorno para servicio TensorFlow.
set -euo pipefail

# Define raíz del repo y mostacho.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MOSTACHO_ROOT="$REPO_ROOT/mostacho"

# Permite override del path del entorno.
VENV_PATH="${MOSTACHO_TF_VENV:-$HOME/venvs/kenya_tf}"

# Crea y activa entorno virtual.
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
# Actualiza herramientas base.
python -m pip install --upgrade pip setuptools wheel
# Instala dependencias de TensorFlow.
pip install -r "$MOSTACHO_ROOT/requirements/tf_service.txt"
# Instala paquete local editable.
pip install -e "$MOSTACHO_ROOT"

# Mensaje final.
echo "Entorno tf listo en: $VENV_PATH"
