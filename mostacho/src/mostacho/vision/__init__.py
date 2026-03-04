"""Subpaquete de visión base (InsightFace + EAR + somnolencia)."""

# Exportaciones públicas para reutilización entre servicio y runtime local.
from mostacho.vision.camera import Camera, list_available_cameras
from mostacho.vision.eyes import compute_ear, get_eye_landmarks_from_face
from mostacho.vision.somnolence import SomnolenceDetector
from mostacho.vision.runtime import VisionRuntime, VisionRuntimeConfig, AnalyzedFace, VisionAnalysis

__all__ = [
    "Camera",
    "list_available_cameras",
    "compute_ear",
    "get_eye_landmarks_from_face",
    "SomnolenceDetector",
    "VisionRuntime",
    "VisionRuntimeConfig",
    "AnalyzedFace",
    "VisionAnalysis",
]
