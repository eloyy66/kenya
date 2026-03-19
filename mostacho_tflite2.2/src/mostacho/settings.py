"""Configuracion centralizada para Mostacho v2.2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class Settings:
    """Contenedor inmutable con rutas y parametros del runtime."""

    repo_root: Path
    mostacho_root: Path
    db_root: Path
    artifacts_root: Path
    face_python: str
    face_worker_path: Path
    face_timeout: float
    eye_python: str
    eye_worker_path: Path
    eye_timeout: float
    drowsiness_active_seconds: float
    eye_closed_confidence_threshold: float
    yawn_trigger_threshold: float
    eye_crop_padding: float
    microsleep_confirm_seconds: float
    eye_invert_output: bool


def load_settings() -> Settings:
    """Carga configuracion desde entorno y aplica defaults razonables."""

    current_file = Path(__file__).resolve()
    repo_root = Path(os.getenv("MOSTACHO_REPO_ROOT", str(current_file.parents[3])))
    mostacho_root = Path(os.getenv("MOSTACHO_ROOT", str(current_file.parents[2])))
    db_root = Path(os.getenv("MOSTACHO_DB_ROOT", str(repo_root / "db")))
    artifacts_root = Path(os.getenv("MOSTACHO_ARTIFACTS_ROOT", str(mostacho_root / "artifacts")))

    face_python = os.getenv("MOSTACHO_FACE_PYTHON", "python3")
    face_worker_path = Path(
        os.getenv(
            "MOSTACHO_FACE_WORKER",
            str(mostacho_root / "src" / "mostacho" / "services" / "face_worker.py"),
        )
    )
    face_timeout = float(os.getenv("MOSTACHO_FACE_TIMEOUT", "20"))

    eye_python = os.getenv("MOSTACHO_EYE_PYTHON", sys.executable)
    eye_worker_path = Path(
        os.getenv(
            "MOSTACHO_EYE_WORKER",
            str(mostacho_root / "src" / "mostacho" / "services" / "eye_state_worker.py"),
        )
    )
    eye_timeout = float(os.getenv("MOSTACHO_EYE_TIMEOUT", "20"))

    drowsiness_active_seconds = float(os.getenv("MOSTACHO_DROWSINESS_ACTIVE_SECONDS", "2.0"))
    eye_closed_confidence_threshold = float(os.getenv("MOSTACHO_EYE_CLOSED_CONFIDENCE", "0.60"))
    yawn_trigger_threshold = float(os.getenv("MOSTACHO_YAWN_TRIGGER_THRESHOLD", "0.32"))
    eye_crop_padding = float(os.getenv("MOSTACHO_EYE_CROP_PADDING", "0.25"))
    microsleep_confirm_seconds = float(os.getenv("MOSTACHO_MICROSLEEP_CONFIRM_SECONDS", "1.5"))
    eye_invert_output = os.getenv("MOSTACHO_EYE_INVERT_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        repo_root=repo_root,
        mostacho_root=mostacho_root,
        db_root=db_root,
        artifacts_root=artifacts_root,
        face_python=face_python,
        face_worker_path=face_worker_path,
        face_timeout=face_timeout,
        eye_python=eye_python,
        eye_worker_path=eye_worker_path,
        eye_timeout=eye_timeout,
        drowsiness_active_seconds=drowsiness_active_seconds,
        eye_closed_confidence_threshold=eye_closed_confidence_threshold,
        yawn_trigger_threshold=yawn_trigger_threshold,
        eye_crop_padding=eye_crop_padding,
        microsleep_confirm_seconds=microsleep_confirm_seconds,
        eye_invert_output=eye_invert_output,
    )
