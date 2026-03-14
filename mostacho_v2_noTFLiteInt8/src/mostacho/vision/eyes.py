"""Utilidades para extraer landmarks de ojos y calcular EAR."""

from __future__ import annotations

# numpy se usa para cálculo vectorial y distancias euclidianas.
import numpy as np


def euclidean(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calcula distancia euclidiana entre dos puntos 2D."""

    # Se usa norma L2 estándar para distancia geométrica.
    return float(np.linalg.norm(point_a - point_b))


def compute_ear(eye_points: np.ndarray) -> float:
    """Calcula Eye Aspect Ratio (EAR) a partir de 6 puntos del ojo."""

    # Distancia vertical 1 (p2-p6).
    distance_a = euclidean(eye_points[1], eye_points[5])
    # Distancia vertical 2 (p3-p5).
    distance_b = euclidean(eye_points[2], eye_points[4])
    # Distancia horizontal (p1-p4).
    distance_c = euclidean(eye_points[0], eye_points[3])

    # Evita división por cero en casos degenerados.
    if distance_c <= 1e-9:
        return 0.0

    # Fórmula clásica del EAR.
    return float((distance_a + distance_b) / (2.0 * distance_c))


def get_eye_landmarks_from_face(face: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Obtiene 6 puntos por ojo desde landmarks 68 o 106 de InsightFace."""

    # Prioridad a landmarks de 68 puntos por convención establecida.
    landmarks_68 = getattr(face, "landmark_3d_68", None)
    if landmarks_68 is not None and getattr(landmarks_68, "shape", (0,))[0] >= 68:
        # Se reduce a coordenadas 2D (x, y).
        kps = landmarks_68[:, :2]
        # Índices estándar para ojo izquierdo en malla de 68.
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        # Índices estándar para ojo derecho en malla de 68.
        right_eye_idx = [42, 43, 44, 45, 46, 47]
        # Se construyen arreglos de puntos por ojo.
        left_eye = np.array([kps[index] for index in left_eye_idx], dtype=float)
        right_eye = np.array([kps[index] for index in right_eye_idx], dtype=float)
        return left_eye, right_eye

    # Fallback a malla de 106 puntos cuando no hay 68.
    landmarks_106 = getattr(face, "landmark_2d_106", None)
    if landmarks_106 is None or getattr(landmarks_106, "shape", (0,))[0] < 98:
        return None, None

    # Índices equivalentes aproximados en malla 106 para ojo izquierdo.
    left_eye_idx = [33, 35, 37, 39, 41, 43]
    # Índices equivalentes aproximados en malla 106 para ojo derecho.
    right_eye_idx = [87, 89, 91, 93, 95, 97]

    # Construcción de arreglos de puntos por ojo.
    left_eye = np.array([landmarks_106[index] for index in left_eye_idx], dtype=float)
    right_eye = np.array([landmarks_106[index] for index in right_eye_idx], dtype=float)
    return left_eye, right_eye
