"""Servicio FastAPI para visión base: InsightFace + EAR + somnolencia."""

from __future__ import annotations

# base64 para transportar imágenes por HTTP.
import base64
# logging para trazabilidad de errores y estado.
import logging
# os para leer variables de entorno.
import os
# Path para manejo robusto de rutas de archivo.
from pathlib import Path
# time para usar reloj monotónico en inferencia secuencial.
import time
# typing para anotaciones explícitas.
from typing import Any, List

# numpy para cálculo de métricas agregadas.
import numpy as np
# FastAPI para exponer endpoints de inferencia/control.
from fastapi import FastAPI, HTTPException, Query

# Esquemas compartidos de entrada/salida.
from mostacho.schemas import (
    FaceDetection,
    FaceRequest,
    FaceResponse,
    HealthResponse,
    SomnolenceControlResponse,
    utc_now_iso,
)
# Runtime visual que porta la base de madebycodex.
from mostacho.vision.runtime import VisionRuntime, VisionRuntimeConfig


# Logger de módulo para observabilidad en producción/desarrollo.
LOGGER = logging.getLogger(__name__)
# App FastAPI principal.
app = FastAPI(title="Mostacho Face Service", version="0.2.0")

# Cache global del objeto FaceAnalysis.
_FACE_APP: Any = None
# Cache global del runtime stateful de somnolencia.
_VISION_RUNTIME: VisionRuntime | None = None


def _load_cv2() -> Any:
    """Importa OpenCV de forma diferida para evitar side-effects globales."""

    # Import local para minimizar fallas al importar módulo completo.
    import cv2  # type: ignore

    # Retorna módulo cv2 listo para uso.
    return cv2


def _parse_det_size(raw_value: str | None, default: tuple[int, int] = (256, 256)) -> tuple[int, int]:
    """Parsea tamaño de detección desde env (ej: '256' o '320,320')."""

    if not raw_value:
        return default

    cleaned = raw_value.lower().replace("x", ",")
    parts = [value.strip() for value in cleaned.split(",") if value.strip()]
    try:
        if len(parts) == 1:
            size = int(parts[0])
            return (size, size)
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        return default

    return default


def _resolve_landmark_modules(raw_value: str | None) -> list[str]:
    """Resuelve módulos de landmarks para InsightFace."""

    normalized = (raw_value or "68").strip().lower()
    if normalized in {"106", "2d106", "landmark_2d_106", "l106"}:
        return ["detection", "landmark_2d_106"]
    if normalized in {"both", "all", "68+106", "106+68"}:
        return ["detection", "landmark_3d_68", "landmark_2d_106"]
    return ["detection", "landmark_3d_68"]


def _load_insightface_app() -> Any:
    """Inicializa FaceAnalysis una sola vez y lo deja en cache."""

    # Se usa variable global para singleton.
    global _FACE_APP

    # Si ya existe instancia, se reutiliza.
    if _FACE_APP is not None:
        return _FACE_APP

    # Import local para aislar dependencia de insightface al servicio de visión.
    from insightface.app import FaceAnalysis  # type: ignore

    # Lee configuración desde entorno para ajustar costo.
    model_name = os.getenv("MOSTACHO_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
    providers_raw = os.getenv("MOSTACHO_FACE_PROVIDERS", "CPUExecutionProvider")
    providers = [value.strip() for value in providers_raw.split(",") if value.strip()]
    allowed_modules = _resolve_landmark_modules(os.getenv("MOSTACHO_FACE_LANDMARKS"))
    detect_size = _parse_det_size(os.getenv("MOSTACHO_FACE_DET_SIZE"), default=(256, 256))

    # Configura modelo y módulos de landmarks requeridos.
    try:
        face_app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules,
        )
    except Exception:
        # Fallback seguro cuando el modelo solicitado no está disponible.
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=allowed_modules,
        )
    # Inicialización en CPU con tamaño de detección estable.
    face_app.prepare(ctx_id=-1, det_size=detect_size)

    # Cachea instancia para futuras requests.
    _FACE_APP = face_app
    return _FACE_APP


def _load_runtime() -> VisionRuntime:
    """Inicializa runtime de visión stateful y lo deja en cache."""

    # Se usa variable global para singleton runtime.
    global _VISION_RUNTIME

    # Si ya existe runtime, se devuelve de inmediato.
    if _VISION_RUNTIME is not None:
        return _VISION_RUNTIME

    # Se asegura inicialización del detector base.
    face_app = _load_insightface_app()

    # Config por defecto equivalente a base de madebycodex.
    detect_size = _parse_det_size(os.getenv("MOSTACHO_FACE_DET_SIZE"), default=(256, 256))
    config = VisionRuntimeConfig(
        detect_size=detect_size,
        window_size=5,
        closed_seconds=3.0,
        calibration_seconds=2.0,
        threshold_offset=0.04,
        min_threshold=0.15,
    )

    # Se crea runtime y se cachea.
    _VISION_RUNTIME = VisionRuntime(face_app=face_app, config=config)
    return _VISION_RUNTIME


def _decode_image_from_request(payload: FaceRequest) -> np.ndarray:
    """Decodifica imagen desde base64 o ruta local."""

    # Se carga OpenCV bajo demanda.
    cv2 = _load_cv2()

    # Ruta de decodificación por base64.
    if payload.image_b64:
        # Decodifica texto base64 a bytes binarios.
        raw_bytes = base64.b64decode(payload.image_b64)
        # Interpreta bytes como arreglo uint8.
        image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        # Decodifica buffer a imagen BGR.
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Valida decodificación correcta.
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar image_b64.")

        # Retorna imagen lista para inferencia.
        return image

    # Ruta de lectura por archivo local.
    if payload.image_path:
        # Normaliza path.
        image_path = Path(payload.image_path)

        # Verifica existencia de archivo.
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"No existe la ruta: {image_path}")

        # Lee archivo de imagen con OpenCV.
        image = cv2.imread(str(image_path))

        # Valida lectura correcta.
        if image is None:
            raise HTTPException(status_code=400, detail=f"No se pudo leer imagen: {image_path}")

        # Retorna imagen BGR.
        return image

    # Si no llega ningún input de imagen, se rechaza request.
    raise HTTPException(status_code=400, detail="Debe enviar image_b64 o image_path.")


def _vision_features_from_response(detections: List[FaceDetection], image_shape: tuple[int, ...], response: FaceResponse) -> dict[str, float]:
    """Construye features de visión para fusión multimodal."""

    # Extrae dimensiones para normalizar área.
    height, width = image_shape[:2]
    frame_area = float(height * width) if height and width else 1.0

    # Métricas base de detección facial.
    face_count = float(len(detections))
    avg_score = 0.0
    primary_face_area_ratio = 0.0

    # Cálculos cuando hay al menos un rostro.
    if detections:
        # Promedio de score de detección.
        avg_score = float(np.mean([det.score for det in detections]))

        # Área del rostro principal (primer rostro).
        x1, y1, x2, y2 = detections[0].bbox
        face_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        primary_face_area_ratio = float(face_area / frame_area)

    # Flags de estado visual para facilitar consumo por tf_service.
    state = response.vision_state
    is_calibrating = 1.0 if state == "CALIBRATING" else 0.0
    is_attentive = 1.0 if state == "ATTENTIVE" else 0.0
    is_eyes_closed = 1.0 if state == "EYES_CLOSED" else 0.0
    is_somnolent = 1.0 if state == "SOMNOLENT" else 0.0

    # Vector final de features visuales.
    features = {
        "vision_face_count": face_count,
        "vision_avg_score": avg_score,
        "vision_primary_face_area_ratio": primary_face_area_ratio,
        "vision_avg_ear": float(response.avg_ear) if response.avg_ear is not None else 0.0,
        "vision_closed_duration": float(response.closed_duration),
        "vision_is_calibrating": is_calibrating,
        "vision_is_attentive": is_attentive,
        "vision_is_eyes_closed": is_eyes_closed,
        "vision_is_somnolent": is_somnolent,
    }

    # Añade umbral actual si está disponible.
    if response.threshold is not None:
        features["vision_threshold"] = float(response.threshold)

    # Añade baseline si está disponible.
    if response.baseline is not None:
        features["vision_baseline"] = float(response.baseline)

    # Retorna vector de features listo para fusión.
    return features


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check liviano del servicio de visión."""

    # Respuesta estándar de disponibilidad.
    return HealthResponse(service="face_service", status="ok", timestamp_utc=utc_now_iso())


@app.post("/infer", response_model=FaceResponse)
def infer(payload: FaceRequest) -> FaceResponse:
    """Ejecuta detección facial + EAR/somnolencia en una imagen."""

    # Se carga runtime stateful de visión.
    runtime = _load_runtime()

    # Se decodifica imagen de entrada.
    image = _decode_image_from_request(payload)

    # Análisis visual completo (detección + EAR + estado).
    analysis = runtime.analyze_image(image, now=time.monotonic())

    # Serializa detecciones al esquema público API.
    detections: List[FaceDetection] = []
    for item in analysis.detections:
        detections.append(
            FaceDetection(
                bbox=item.bbox,
                score=item.score,
                landmarks=item.landmarks,
                left_eye=item.left_eye,
                right_eye=item.right_eye,
                ear=item.ear,
            )
        )

    # Construye respuesta base sin features (se rellenan después).
    response = FaceResponse(
        timestamp_utc=utc_now_iso(),
        detections=detections,
        vision_features={},
        vision_state=analysis.vision_state,
        avg_ear=analysis.avg_ear,
        closed_duration=analysis.closed_duration,
        threshold=analysis.threshold,
        baseline=analysis.baseline,
    )

    # Construye features visuales desde el análisis completo.
    response.vision_features = _vision_features_from_response(detections, image.shape, response)

    # Retorna respuesta final.
    return response


@app.post("/somnolence/reset", response_model=SomnolenceControlResponse)
def reset_somnolence() -> SomnolenceControlResponse:
    """Reinicia calibración y estado de somnolencia del runtime visual."""

    # Se obtiene runtime global.
    runtime = _load_runtime()
    # Se reinicia calibración y baseline.
    runtime.reset_calibration()

    # Respuesta de confirmación.
    return SomnolenceControlResponse(
        status="ok",
        message="Calibracion de somnolencia reiniciada.",
        threshold=None,
        timestamp_utc=utc_now_iso(),
    )


@app.post("/somnolence/threshold", response_model=SomnolenceControlResponse)
def adjust_somnolence_threshold(delta: float = Query(..., ge=-0.1, le=0.1)) -> SomnolenceControlResponse:
    """Ajusta umbral manual de EAR con delta positivo o negativo."""

    # Se obtiene runtime global.
    runtime = _load_runtime()
    # Se aplica ajuste incremental y se recupera nuevo umbral.
    threshold = runtime.adjust_threshold(delta)

    # Respuesta con umbral actualizado.
    return SomnolenceControlResponse(
        status="ok",
        message="Umbral de somnolencia actualizado.",
        threshold=threshold,
        timestamp_utc=utc_now_iso(),
    )
