"""Esquemas Pydantic para Mostacho v2.2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ImageRequest(BaseModel):
    """Entrada base para inferencia por imagen."""

    image_b64: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)
    image_is_cropped: Optional[bool] = Field(default=False)
    eye_crop: Optional[bool] = Field(default=False)
    face_bbox: Optional[List[float]] = Field(default=None)


class FaceRequest(BaseModel):
    """Entrada para worker de InsightFace."""

    image_b64: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)


class FaceDetection(BaseModel):
    """Rostro analizado y metadatos locales."""

    bbox: List[float]
    score: float
    landmarks: List[List[float]]
    left_eye: List[List[float]] = Field(default_factory=list)
    right_eye: List[List[float]] = Field(default_factory=list)
    mouth: List[List[float]] = Field(default_factory=list)
    ear: Optional[float] = Field(default=None)
    mouth_open_ratio: Optional[float] = Field(default=None)


class FaceResponse(BaseModel):
    """Salida serializable del worker de InsightFace."""

    timestamp_utc: str
    detections: List[FaceDetection]
    vision_features: Dict[str, float]
    vision_state: str
    avg_ear: Optional[float] = Field(default=None)
    closed_duration: float = Field(default=0.0)
    threshold: Optional[float] = Field(default=None)
    baseline: Optional[float] = Field(default=None)
    mouth_open_ratio: Optional[float] = Field(default=None)
    yawn_candidate: bool = Field(default=False)


class EyeStateImageRequest(ImageRequest):
    """Entrada para inferencia del clasificador open/closed eyes."""


class EyeStateResponse(BaseModel):
    """Salida del clasificador open/closed eyes."""

    engine: str = Field(default="unknown")
    state: str
    confidence: float
    class_probabilities: Dict[str, float]
    regions_used: int = Field(default=1)
    eye_boxes: List[List[int]] = Field(default_factory=list)
    timestamp_utc: str


class DrowsinessImageRequest(ImageRequest):
    """Entrada para inferencia combinada de v2.2."""


class DrowsinessResponse(BaseModel):
    """Salida combinada con gating de drowsiness."""

    engine: str = Field(default="unknown")
    state: str
    confidence: float
    model_ran: bool
    run_reason: str
    model_state: str
    model_confidence: float
    class_probabilities: Dict[str, float]
    eye_state: str
    eye_confidence: float
    eye_class_probabilities: Dict[str, float]
    eye_regions_used: int = Field(default=0)
    eye_boxes: List[List[int]] = Field(default_factory=list)
    vision_state: str
    microsleep_duration: float = Field(default=0.0)
    microsleep_confirm_seconds: float = Field(default=1.5)
    avg_ear: Optional[float] = Field(default=None)
    closed_duration: float = Field(default=0.0)
    mouth_open_ratio: Optional[float] = Field(default=None)
    yawn_candidate: bool = Field(default=False)
    distraction_enabled: bool = Field(default=False)
    distraction_ran: bool = Field(default=False)
    distraction_run_reason: str = Field(default="disabled")
    distraction_state: str = Field(default="UNKNOWN")
    distraction_confidence: float = Field(default=0.0)
    distraction_class_probabilities: Dict[str, float] = Field(default_factory=dict)
    face_response: Optional[FaceResponse] = Field(default=None)
    timestamp_utc: str


class HealthResponse(BaseModel):
    """Respuesta estandar de health-check."""

    service: str
    status: str
    timestamp_utc: str


def utc_now_iso() -> str:
    """Retorna fecha/hora UTC en formato ISO 8601."""

    return datetime.now(tz=timezone.utc).isoformat()
