"""Esquemas Pydantic compartidos entre servicios y orquestador."""

from __future__ import annotations

# datetime permite sellar eventos con timestamp UTC.
from datetime import datetime, timezone
# typing aporta tipos declarativos para APIs.
from typing import Dict, List, Optional

# BaseModel valida y serializa datos de entrada/salida.
from pydantic import BaseModel, Field


class FaceRequest(BaseModel):
    """Entrada para el servicio de InsightFace."""

    # Se admite imagen en base64 para streaming desde cliente.
    image_b64: Optional[str] = Field(default=None, description="Imagen codificada en base64.")
    # Se admite ruta local para pruebas offline.
    image_path: Optional[str] = Field(default=None, description="Ruta local de imagen para pruebas.")


class FaceDetection(BaseModel):
    """Representa un rostro detectado con atributos clave."""

    # Bounding box en formato [x1, y1, x2, y2].
    bbox: List[float]
    # Score de confianza del detector.
    score: float
    # Landmarks 2D principales.
    landmarks: List[List[float]]
    # Contorno de ojo izquierdo (6 puntos), cuando está disponible.
    left_eye: List[List[float]] = Field(default_factory=list)
    # Contorno de ojo derecho (6 puntos), cuando está disponible.
    right_eye: List[List[float]] = Field(default_factory=list)
    # EAR calculado para el rostro, cuando está disponible.
    ear: Optional[float] = Field(default=None)


class FaceResponse(BaseModel):
    """Respuesta del servicio de vision."""

    # Timestamp UTC de procesamiento.
    timestamp_utc: str
    # Lista de detecciones.
    detections: List[FaceDetection]
    # Features resumidas para el modelo de fusion.
    vision_features: Dict[str, float]
    # Estado de visión basado en EAR/somnolencia.
    vision_state: str
    # EAR promedio suavizado.
    avg_ear: Optional[float] = Field(default=None)
    # Duración continua de ojos cerrados.
    closed_duration: float = Field(default=0.0)
    # Umbral actual de EAR.
    threshold: Optional[float] = Field(default=None)
    # Baseline EAR calibrado.
    baseline: Optional[float] = Field(default=None)


class SomnolenceControlResponse(BaseModel):
    """Respuesta para operaciones de control de calibración/umbral en visión."""

    # Estado de la operación solicitada.
    status: str
    # Mensaje breve para consumo de cliente.
    message: str
    # Umbral actual luego de una operación de ajuste (si aplica).
    threshold: Optional[float] = Field(default=None)
    # Timestamp UTC de la operación.
    timestamp_utc: str


class FusionRequest(BaseModel):
    """Entrada para el servicio de fusion TensorFlow."""

    # Features agregadas de vision.
    vision_features: Dict[str, float] = Field(default_factory=dict)
    # Features agregadas de voz.
    voice_features: Dict[str, float] = Field(default_factory=dict)
    # Features agregadas de biometria.
    biometric_features: Dict[str, float] = Field(default_factory=dict)


class FusionResponse(BaseModel):
    """Salida de clasificacion multimodal."""

    # Estado final estimado.
    state: str
    # Confianza global entre 0 y 1.
    confidence: float
    # Probabilidades por clase.
    class_probabilities: Dict[str, float]
    # Timestamp UTC de inferencia.
    timestamp_utc: str


class DrowsinessImageRequest(BaseModel):
    """Entrada para inferencia de somnolencia visual en TF service."""

    # Imagen codificada en base64 para envío por HTTP.
    image_b64: Optional[str] = Field(default=None)
    # Ruta local de imagen para pruebas offline.
    image_path: Optional[str] = Field(default=None)
    # Indica si la imagen ya es un recorte del rostro (evita crop extra).
    image_is_cropped: Optional[bool] = Field(default=False)
    # Bounding box del rostro principal [x1, y1, x2, y2] (opcional).
    face_bbox: Optional[List[float]] = Field(default=None)
    # Estado visual proveniente de face_service (opcional).
    vision_state: Optional[str] = Field(default=None)
    # EAR promedio suavizado desde face_service (opcional).
    avg_ear: Optional[float] = Field(default=None)
    # Duración de ojos cerrados desde face_service (opcional).
    closed_duration: float = Field(default=0.0)
    # Umbral EAR utilizado en face_service (opcional).
    threshold: Optional[float] = Field(default=None)
    # Baseline EAR utilizado en face_service (opcional).
    baseline: Optional[float] = Field(default=None)


class DrowsinessResponse(BaseModel):
    """Salida de inferencia de somnolencia fusionando red visual + señales EAR."""

    # Estado final de somnolencia fusionado.
    state: str
    # Confianza del estado final entre 0 y 1.
    confidence: float
    # Clase principal predicha por la red visual.
    model_state: str
    # Confianza de la clase principal de la red visual.
    model_confidence: float
    # Probabilidades por clase del modelo visual.
    class_probabilities: Dict[str, float]
    # Puntaje de riesgo de somnolencia fusionado entre 0 y 1.
    risk_score: float
    # Desglose de contribuciones del riesgo fusionado.
    components: Dict[str, float]
    # Timestamp UTC de inferencia.
    timestamp_utc: str


class HealthResponse(BaseModel):
    """Respuesta estandar de health-check."""

    # Nombre del servicio.
    service: str
    # Estado operacional.
    status: str
    # Hora UTC de respuesta.
    timestamp_utc: str


def utc_now_iso() -> str:
    """Retorna fecha/hora UTC en formato ISO 8601."""

    # Se normaliza a UTC para trazabilidad entre equipos y zonas horarias.
    return datetime.now(tz=timezone.utc).isoformat()
