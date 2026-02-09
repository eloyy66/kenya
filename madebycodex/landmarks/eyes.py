"""Utilidades para extraer ojos y calcular EAR."""

import numpy as np


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_points):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    eye_points: 6 puntos del ojo en orden estandar.
    """
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


def get_eye_landmarks_from_face(face):
    """Obtiene 6 puntos por ojo desde landmarks 68 o 106."""
    # Preferimos 68 puntos (indices estandar)
    kps_68 = getattr(face, "landmark_3d_68", None)
    if kps_68 is not None and kps_68.shape[0] >= 68:
        kps = kps_68[:, :2]
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        right_eye_idx = [42, 43, 44, 45, 46, 47]
        left_eye = np.array([kps[i] for i in left_eye_idx])
        right_eye = np.array([kps[i] for i in right_eye_idx])
        return left_eye, right_eye

    # Fallback 106
    kps_106 = getattr(face, "landmark_2d_106", None)
    if kps_106 is None or kps_106.shape[0] < 98:
        return None, None

    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]
    left_eye = np.array([kps_106[i] for i in left_eye_idx])
    right_eye = np.array([kps_106[i] for i in right_eye_idx])
    return left_eye, right_eye
