"""Runtime de visión que integra InsightFace + landmarks + somnolencia."""

from __future__ import annotations

# dataclass para estructuras de salida claras.
from dataclasses import dataclass, field
# time para reloj monotónico en inferencia por frame.
import time
# typing para firmas explícitas.
from typing import Any, List, Tuple

# numpy para vectores numéricos.
import numpy as np

# utilidades de ojos y detector de somnolencia.
from mostacho.vision.eyes import compute_ear, get_eye_landmarks_from_face
from mostacho.vision.somnolence import SomnolenceDetector


@dataclass
class VisionRuntimeConfig:
    """Configuración de runtime visual (detección + EAR)."""

    # Tamaño de detección para InsightFace.
    detect_size: Tuple[int, int] = (256, 256)
    # Tamaño de ventana para suavizado EAR.
    window_size: int = 5
    # Segundos de ojos cerrados para declarar somnolencia.
    closed_seconds: float = 3.0
    # Segundos iniciales para calibrar baseline.
    calibration_seconds: float = 2.0
    # Offset para umbral dinámico.
    threshold_offset: float = 0.04
    # Umbral mínimo de seguridad.
    min_threshold: float = 0.15


@dataclass
class AnalyzedFace:
    """Estructura serializable de un rostro analizado."""

    # Bounding box [x1, y1, x2, y2].
    bbox: List[float]
    # Score de detección.
    score: float
    # Landmarks faciales principales.
    landmarks: List[List[float]] = field(default_factory=list)
    # Contorno de ojo izquierdo (6 puntos).
    left_eye: List[List[float]] = field(default_factory=list)
    # Contorno de ojo derecho (6 puntos).
    right_eye: List[List[float]] = field(default_factory=list)
    # EAR calculado para el rostro principal.
    ear: float | None = None


@dataclass
class VisionAnalysis:
    """Resultado de análisis de visión por frame."""

    # Lista de rostros detectados/anotados.
    detections: List[AnalyzedFace]
    # Estado visual final.
    vision_state: str
    # EAR promedio suavizado actual.
    avg_ear: float | None
    # Duración continua de ojos cerrados.
    closed_duration: float
    # Umbral actual usado por el detector.
    threshold: float | None
    # Baseline EAR calibrado.
    baseline: float | None


class VisionRuntime:
    """Motor reutilizable de visión para servicio HTTP y ejecución local."""

    def __init__(self, face_app: Any, config: VisionRuntimeConfig | None = None):
        # Instancia FaceAnalysis ya inicializada.
        self.face_app = face_app
        # Configuración del runtime visual.
        self.config = config or VisionRuntimeConfig()
        # Detector stateful de somnolencia por EAR.
        self.somnolence = SomnolenceDetector(
            window_size=self.config.window_size,
            closed_seconds=self.config.closed_seconds,
            calibration_seconds=self.config.calibration_seconds,
            threshold_offset=self.config.threshold_offset,
            min_threshold=self.config.min_threshold,
        )

    def reset_calibration(self) -> None:
        """Reinicia calibración EAR interna."""

        # Delegación al detector de somnolencia.
        self.somnolence.reset_calibration()

    def adjust_threshold(self, delta: float) -> float:
        """Ajusta umbral manual de EAR y retorna valor actualizado."""

        # Delegación de ajuste al detector stateful.
        return self.somnolence.adjust_threshold(delta)

    def analyze_image(self, image: np.ndarray, now: float | None = None) -> VisionAnalysis:
        """Analiza una imagen/frame y devuelve detecciones + estado visual."""

        # Usa reloj monotónico cuando no se provee timestamp.
        if now is None:
            now = time.monotonic()

        # Detección facial vía InsightFace.
        faces = self.face_app.get(image)

        # Serialización de detecciones para salida.
        serialized: List[AnalyzedFace] = []

        # Valores por defecto cuando no hay rostro/landmarks.
        vision_state = "NO_FACE"
        avg_ear: float | None = None
        closed_duration = 0.0
        threshold: float | None = None
        baseline: float | None = self.somnolence.baseline_ear

        # Se procesa cada rostro detectado.
        for index, face in enumerate(faces):
            # Bounding box normalizado a lista float.
            bbox = [float(value) for value in face.bbox.tolist()]
            # Score de detección normalizado.
            score = float(getattr(face, "det_score", 0.0))
            # Landmarks básicos (kps) para visualización general.
            kps = getattr(face, "kps", None)
            landmarks = [[float(x), float(y)] for x, y in kps.tolist()] if kps is not None else []

            # Extracción de contorno de ojos (6 puntos por ojo).
            left_eye, right_eye = get_eye_landmarks_from_face(face)
            # Valores default de ojos en lista vacía.
            left_eye_list: List[List[float]] = []
            right_eye_list: List[List[float]] = []
            ear_value: float | None = None

            # Si hay landmarks válidos de ojos, calcula EAR.
            if left_eye is not None and right_eye is not None:
                # Convertir puntos a listas serializables.
                left_eye_list = [[float(x), float(y)] for x, y in left_eye.tolist()]
                right_eye_list = [[float(x), float(y)] for x, y in right_eye.tolist()]
                # EAR promedio de ambos ojos.
                ear_value = float((compute_ear(left_eye) + compute_ear(right_eye)) / 2.0)

                # Solo el rostro principal actualiza estado de somnolencia.
                if index == 0:
                    somnolence = self.somnolence.update(ear_value, now=now)
                    vision_state = str(somnolence["state"])
                    avg_ear = float(somnolence["avg_ear"]) if somnolence["avg_ear"] is not None else None
                    closed_duration = float(somnolence["closed_duration"]) if somnolence["closed_duration"] is not None else 0.0
                    threshold = float(somnolence["threshold"]) if somnolence["threshold"] is not None else None
                    baseline = float(somnolence["baseline"]) if somnolence["baseline"] is not None else None
            else:
                # Si no hay ojos válidos en el rostro principal, estado específico.
                if index == 0:
                    vision_state = "NO_LANDMARKS"

            # Se agrega detección analizada a salida.
            serialized.append(
                AnalyzedFace(
                    bbox=bbox,
                    score=score,
                    landmarks=landmarks,
                    left_eye=left_eye_list,
                    right_eye=right_eye_list,
                    ear=ear_value,
                )
            )

        # Se retorna estructura de análisis completa.
        return VisionAnalysis(
            detections=serialized,
            vision_state=vision_state,
            avg_ear=avg_ear,
            closed_duration=closed_duration,
            threshold=threshold,
            baseline=baseline,
        )
