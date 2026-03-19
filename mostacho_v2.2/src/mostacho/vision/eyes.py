"""Utilidades para extraer landmarks de ojos y boca y calcular razones geométricas."""

from __future__ import annotations

import numpy as np


def euclidean(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calcula distancia euclidiana entre dos puntos 2D."""

    return float(np.linalg.norm(point_a - point_b))


def compute_ear(eye_points: np.ndarray) -> float:
    """Calcula Eye Aspect Ratio (EAR) a partir de 6 puntos del ojo."""

    distance_a = euclidean(eye_points[1], eye_points[5])
    distance_b = euclidean(eye_points[2], eye_points[4])
    distance_c = euclidean(eye_points[0], eye_points[3])
    if distance_c <= 1e-9:
        return 0.0
    return float((distance_a + distance_b) / (2.0 * distance_c))


def eye_horizontal_span(eye_points: np.ndarray) -> float:
    """Retorna el ancho horizontal del ojo usando el eje externo-interno."""

    return euclidean(eye_points[0], eye_points[3])


def is_eye_landmark_geometry_valid(eye_points: np.ndarray, min_width_px: float = 10.0) -> bool:
    """Valida si la geometría del ojo es utilizable para EAR robusto."""

    if getattr(eye_points, "shape", (0, 0)) != (6, 2):
        return False
    if not np.isfinite(eye_points).all():
        return False
    return eye_horizontal_span(eye_points) >= float(min_width_px)


def is_ear_plausible(ear_value: float, min_ear: float = 0.08, max_ear: float = 0.60) -> bool:
    """Filtra EAR imposibles o muy ruidosos antes de actualizar estado."""

    return float(min_ear) <= float(ear_value) <= float(max_ear)


def get_eye_landmarks_from_face(face: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Obtiene 6 puntos por ojo desde landmarks 68 o 106 de InsightFace."""

    landmarks_68 = getattr(face, "landmark_3d_68", None)
    if landmarks_68 is not None and getattr(landmarks_68, "shape", (0,))[0] >= 68:
        kps = landmarks_68[:, :2]
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        right_eye_idx = [42, 43, 44, 45, 46, 47]
        left_eye = np.array([kps[index] for index in left_eye_idx], dtype=float)
        right_eye = np.array([kps[index] for index in right_eye_idx], dtype=float)
        return left_eye, right_eye

    landmarks_106 = getattr(face, "landmark_2d_106", None)
    if landmarks_106 is None or getattr(landmarks_106, "shape", (0,))[0] < 98:
        return None, None

    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]
    left_eye = np.array([landmarks_106[index] for index in left_eye_idx], dtype=float)
    right_eye = np.array([landmarks_106[index] for index in right_eye_idx], dtype=float)
    return left_eye, right_eye


def get_mouth_landmarks_from_face(face: object) -> np.ndarray | None:
    """Obtiene 20 puntos de boca desde landmarks 68 cuando están disponibles."""

    landmarks_68 = getattr(face, "landmark_3d_68", None)
    if landmarks_68 is not None and getattr(landmarks_68, "shape", (0,))[0] >= 68:
        return np.array(landmarks_68[48:68, :2], dtype=float)
    return None


def compute_mouth_open_ratio(mouth_points: np.ndarray) -> float:
    """Calcula una razón de apertura de boca usando el contorno interno."""

    if getattr(mouth_points, "shape", (0, 0)) != (20, 2):
        return 0.0

    inner_top_left = mouth_points[13]
    inner_top_mid = mouth_points[14]
    inner_top_right = mouth_points[15]
    inner_bottom_right = mouth_points[17]
    inner_bottom_mid = mouth_points[18]
    inner_bottom_left = mouth_points[19]
    inner_left = mouth_points[12]
    inner_right = mouth_points[16]

    vertical = (
        euclidean(inner_top_left, inner_bottom_left)
        + euclidean(inner_top_mid, inner_bottom_mid)
        + euclidean(inner_top_right, inner_bottom_right)
    ) / 3.0
    horizontal = euclidean(inner_left, inner_right)
    if horizontal <= 1e-9:
        return 0.0
    return float(vertical / horizontal)


def is_mouth_ratio_plausible(mouth_ratio: float, min_ratio: float = 0.02, max_ratio: float = 1.20) -> bool:
    """Filtra ratios de boca imposibles o muy ruidosos."""

    return float(min_ratio) <= float(mouth_ratio) <= float(max_ratio)
