"""Configuracion centralizada para rutas y endpoints del proyecto."""

from __future__ import annotations

# Se usa dataclass para mantener una configuracion tipada y facil de transportar.
from dataclasses import dataclass
# Path facilita manejo robusto de rutas multiplataforma.
from pathlib import Path
# os permite leer variables de entorno sin dependencias externas.
import os


@dataclass(frozen=True)
class Settings:
    """Contenedor inmutable con rutas y parametros globales."""

    # Raiz del repositorio `kenya`.
    repo_root: Path
    # Raiz de la carpeta `mostacho`.
    mostacho_root: Path
    # Raiz de datasets compartidos.
    db_root: Path
    # Carpeta para artefactos de entrenamiento.
    artifacts_root: Path
    # Endpoint del servicio de InsightFace.
    face_service_url: str
    # Endpoint del servicio de TensorFlow.
    tf_service_url: str


def load_settings() -> Settings:
    """Carga configuracion desde entorno y aplica defaults razonables."""

    # Este archivo vive en `mostacho/src/mostacho/settings.py`.
    current_file = Path(__file__).resolve()
    # `parents[3]` apunta a la raiz del repo (`kenya`).
    repo_root = Path(os.getenv("MOSTACHO_REPO_ROOT", str(current_file.parents[3])))
    # Raiz de la carpeta actual `mostacho` o `mostacho_v2` según ubicación real.
    mostacho_root = Path(os.getenv("MOSTACHO_ROOT", str(current_file.parents[2])))
    # `db` tambien cuelga de la raiz del repo.
    db_root = Path(os.getenv("MOSTACHO_DB_ROOT", str(repo_root / "db")))
    # Los artefactos se centralizan en `mostacho/artifacts`.
    artifacts_root = Path(os.getenv("MOSTACHO_ARTIFACTS_ROOT", str(mostacho_root / "artifacts")))

    # URL local por defecto del servicio de vision.
    face_service_url = os.getenv("MOSTACHO_FACE_URL", "http://127.0.0.1:8001")
    # URL local por defecto del servicio de fusion neuronal.
    tf_service_url = os.getenv("MOSTACHO_TF_URL", "http://127.0.0.1:8002")

    # Se devuelve configuracion compacta e inmutable.
    return Settings(
        repo_root=repo_root,
        mostacho_root=mostacho_root,
        db_root=db_root,
        artifacts_root=artifacts_root,
        face_service_url=face_service_url,
        tf_service_url=tf_service_url,
    )
