"""Runtime de visión que integra InsightFace, EAR y señal de bostezo."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, List, Tuple

import numpy as np

from mostacho.vision.eyes import (
    compute_ear,
    compute_mouth_open_ratio,
    get_eye_landmarks_from_face,
    get_mouth_landmarks_from_face,
    is_ear_plausible,
    is_eye_landmark_geometry_valid,
    is_mouth_ratio_plausible,
)
from mostacho.vision.somnolence import SomnolenceDetector


@dataclass
class VisionRuntimeConfig:
    """Configuración de runtime visual."""

    detect_size: Tuple[int, int] = (256, 256)
    window_size: int = 5
    closed_seconds: float = 3.0
    calibration_seconds: float = 5.0
    threshold_offset: float = 0.04
    min_threshold: float = 0.15
    reopen_threshold_offset: float = 0.015
    adaptive_baseline_alpha: float = 0.05
    threshold_std_factor: float = 2.0
    min_calibration_samples: int = 30
    reference_window_size: int = 180
    min_face_score: float = 0.55
    min_eye_width_px: float = 10.0
    max_eye_asymmetry: float = 0.12
    mouth_open_threshold: float = 0.32


@dataclass
class AnalyzedFace:
    """Estructura serializable de un rostro analizado."""

    bbox: List[float]
    score: float
    landmarks: List[List[float]] = field(default_factory=list)
    left_eye: List[List[float]] = field(default_factory=list)
    right_eye: List[List[float]] = field(default_factory=list)
    mouth: List[List[float]] = field(default_factory=list)
    ear: float | None = None
    mouth_open_ratio: float | None = None


@dataclass
class VisionAnalysis:
    """Resultado de análisis de visión por frame."""

    detections: List[AnalyzedFace]
    vision_state: str
    avg_ear: float | None
    closed_duration: float
    threshold: float | None
    baseline: float | None
    mouth_open_ratio: float | None
    yawn_candidate: bool


class VisionRuntime:
    """Motor reutilizable de visión para worker y ejecución local."""

    def __init__(self, face_app: Any, config: VisionRuntimeConfig | None = None):
        self.face_app = face_app
        self.config = config or VisionRuntimeConfig()
        self.somnolence = SomnolenceDetector(
            window_size=self.config.window_size,
            closed_seconds=self.config.closed_seconds,
            calibration_seconds=self.config.calibration_seconds,
            threshold_offset=self.config.threshold_offset,
            min_threshold=self.config.min_threshold,
            reopen_threshold_offset=self.config.reopen_threshold_offset,
            adaptive_baseline_alpha=self.config.adaptive_baseline_alpha,
            threshold_std_factor=self.config.threshold_std_factor,
            min_calibration_samples=self.config.min_calibration_samples,
            reference_window_size=self.config.reference_window_size,
        )

    def reset_calibration(self) -> None:
        """Reinicia calibración EAR interna."""

        self.somnolence.reset_calibration()

    def adjust_threshold(self, delta: float) -> float:
        """Ajusta umbral manual de EAR y retorna valor actualizado."""

        return self.somnolence.adjust_threshold(delta)

    def analyze_image(self, image: np.ndarray, now: float | None = None) -> VisionAnalysis:
        """Analiza una imagen/frame y devuelve detecciones + estado visual."""

        if now is None:
            now = time.monotonic()

        faces = self.face_app.get(image)
        serialized: List[AnalyzedFace] = []

        vision_state = "NO_FACE"
        avg_ear: float | None = None
        closed_duration = 0.0
        threshold: float | None = None
        baseline: float | None = self.somnolence.baseline_ear
        mouth_open_ratio: float | None = None
        yawn_candidate = False

        for index, face in enumerate(faces):
            bbox = [float(value) for value in face.bbox.tolist()]
            score = float(getattr(face, "det_score", 0.0))
            kps = getattr(face, "kps", None)
            landmarks = [[float(x), float(y)] for x, y in kps.tolist()] if kps is not None else []

            left_eye, right_eye = get_eye_landmarks_from_face(face)
            mouth = get_mouth_landmarks_from_face(face)

            left_eye_list: List[List[float]] = []
            right_eye_list: List[List[float]] = []
            mouth_list: List[List[float]] = []
            ear_value: float | None = None
            face_mouth_ratio: float | None = None

            if mouth is not None:
                mouth_list = [[float(x), float(y)] for x, y in mouth.tolist()]
                face_mouth_ratio = float(compute_mouth_open_ratio(mouth))
                if not is_mouth_ratio_plausible(face_mouth_ratio):
                    face_mouth_ratio = None

            if left_eye is not None and right_eye is not None:
                left_eye_list = [[float(x), float(y)] for x, y in left_eye.tolist()]
                right_eye_list = [[float(x), float(y)] for x, y in right_eye.tolist()]
                eyes_valid = is_eye_landmark_geometry_valid(left_eye, self.config.min_eye_width_px) and is_eye_landmark_geometry_valid(
                    right_eye,
                    self.config.min_eye_width_px,
                )

                if eyes_valid:
                    left_ear_value = float(compute_ear(left_eye))
                    right_ear_value = float(compute_ear(right_eye))
                    ear_value = float((left_ear_value + right_ear_value) / 2.0)

                    if index == 0:
                        ears_plausible = is_ear_plausible(left_ear_value) and is_ear_plausible(right_ear_value)
                        symmetry_ok = abs(left_ear_value - right_ear_value) <= self.config.max_eye_asymmetry
                        face_score_ok = score >= self.config.min_face_score

                        if face_score_ok and ears_plausible and symmetry_ok:
                            somnolence = self.somnolence.update(
                                ear_value,
                                now=now,
                                left_ear=left_ear_value,
                                right_ear=right_ear_value,
                            )
                            vision_state = str(somnolence["state"])
                            avg_ear = float(somnolence["avg_ear"]) if somnolence["avg_ear"] is not None else None
                            closed_duration = (
                                float(somnolence["closed_duration"]) if somnolence["closed_duration"] is not None else 0.0
                            )
                            threshold = float(somnolence["threshold"]) if somnolence["threshold"] is not None else None
                            baseline = float(somnolence["baseline"]) if somnolence["baseline"] is not None else None
                        else:
                            vision_state = "LOW_QUALITY"
                            avg_ear = ear_value
                            threshold = self.somnolence.current_threshold()
                            baseline = self.somnolence.baseline_ear
                elif index == 0:
                    vision_state = "NO_LANDMARKS"
                    threshold = self.somnolence.current_threshold()
                    baseline = self.somnolence.baseline_ear
            elif index == 0:
                vision_state = "NO_LANDMARKS"
                threshold = self.somnolence.current_threshold()
                baseline = self.somnolence.baseline_ear

            if index == 0 and face_mouth_ratio is not None:
                mouth_open_ratio = face_mouth_ratio
                yawn_candidate = face_mouth_ratio >= float(self.config.mouth_open_threshold)

            serialized.append(
                AnalyzedFace(
                    bbox=bbox,
                    score=score,
                    landmarks=landmarks,
                    left_eye=left_eye_list,
                    right_eye=right_eye_list,
                    mouth=mouth_list,
                    ear=ear_value,
                    mouth_open_ratio=face_mouth_ratio,
                )
            )

        return VisionAnalysis(
            detections=serialized,
            vision_state=vision_state,
            avg_ear=avg_ear,
            closed_duration=closed_duration,
            threshold=threshold,
            baseline=baseline,
            mouth_open_ratio=mouth_open_ratio,
            yawn_candidate=yawn_candidate,
        )
