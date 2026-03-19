"""Servicio principal de Mostacho v2.2 con gating y subprocesos."""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

from mostacho.eye_state import (
    crop_image_by_bbox,
    extract_eye_regions_from_detection,
    extract_eye_regions_from_face,
)
from mostacho.schemas import (
    DrowsinessImageRequest,
    DrowsinessResponse,
    EyeStateImageRequest,
    EyeStateResponse,
    FaceResponse,
    HealthResponse,
    utc_now_iso,
)
from mostacho.services.tf_subprocess import TfSubprocessClient
from mostacho.settings import Settings, load_settings
from mostacho.vision.face_subprocess import FaceSubprocessClient


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Mostacho v2.2 TF Service", version="0.1.0")

_FACE_SUBPROCESS_CLIENT: FaceSubprocessClient | None = None
_TF_SUBPROCESS_CLIENT: TfSubprocessClient | None = None
_DROWSINESS_ACTIVE_UNTIL: float = 0.0
_MICROSLEEP_SINCE: float | None = None
_DISTRACTION_LAST_RUN_AT: float = 0.0
_DISTRACTION_LAST_RESULT: Dict[str, Any] | None = None
_DISTRACTION_DISABLED_REASON: str | None = None


def _settings() -> Settings:
    """Retorna settings del runtime actual."""

    return load_settings()


def _combined_engine_label(tf_engine: str | None = None) -> str:
    """Describe el stack activo para inspección rápida."""

    return f"{(tf_engine or 'tf_subprocess')}+face_subprocess"


def _decode_image_bytes(image_b64: str | None, image_path: str | None) -> bytes:
    """Lee bytes de imagen desde base64 o ruta local."""

    if image_b64:
        return base64.b64decode(image_b64)

    if image_path:
        resolved = Path(image_path)
        if not resolved.exists():
            raise HTTPException(status_code=400, detail=f"No existe image_path: {resolved}")
        return resolved.read_bytes()

    raise HTTPException(status_code=400, detail="Debe enviar image_b64 o image_path.")


def _decode_rgb_image(image_b64: str | None, image_path: str | None) -> np.ndarray:
    """Decodifica una imagen RGB desde base64 o ruta local."""

    raw_bytes = _decode_image_bytes(image_b64=image_b64, image_path=image_path)
    image_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar una imagen RGB valida.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _normalize_eye_state(state: str) -> str:
    """Mapea etiquetas del clasificador a OPEN/CLOSED cuando aplica."""

    normalized = str(state or "").strip().upper()
    if normalized in {"OPEN", "OPEN_EYES"}:
        return "OPEN"
    if normalized in {"CLOSED", "CLOSED_EYES"}:
        return "CLOSED"
    return normalized or "UNKNOWN"


def _normalize_model_state(state: str) -> str:
    """Normaliza salida del modelo de somnolencia."""

    return str(state or "unknown").strip().upper()


def _normalize_distraction_state(state: str) -> str:
    """Normaliza salida del modelo de distraccion a NORMAL/DISTRACTED cuando aplica."""

    normalized = str(state or "").strip().upper().replace(" ", "_")
    if normalized in {"DISTRACTED", "DISTRACTION", "DISTRACT", "DRIVER_DISTRACTED"}:
        return "DISTRACTED"
    if normalized in {"NORMAL", "SAFE_DRIVING", "ATTENTIVE", "NOT_DISTRACTED"}:
        return "NORMAL"
    return normalized or "UNKNOWN"


def _distraction_enabled() -> bool:
    """Lee flag de habilitacion de distraccion desde entorno."""

    return os.getenv("MOSTACHO_DISTRACTION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def _distraction_interval_seconds() -> float:
    """Lee intervalo objetivo de inferencia de distraccion."""

    raw_value = os.getenv("MOSTACHO_DISTRACTION_INTERVAL_SECONDS", "0.40").strip()
    try:
        value = float(raw_value)
    except ValueError:
        value = 0.40
    return max(0.10, min(value, 5.0))


def _build_distraction_payload(
    enabled: bool,
    ran: bool,
    run_reason: str,
    state: str,
    confidence: float,
    class_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """Normaliza el payload de distraccion para la respuesta API."""

    normalized_state = _normalize_distraction_state(state)
    return {
        "distraction_enabled": bool(enabled),
        "distraction_ran": bool(ran),
        "distraction_run_reason": str(run_reason),
        "distraction_state": normalized_state,
        "distraction_confidence": float(confidence),
        "distraction_class_probabilities": {str(key): float(value) for key, value in class_probabilities.items()},
    }


def _resolve_distraction(face_crop: np.ndarray, face_response: FaceResponse | None, now: float) -> Dict[str, Any]:
    """Ejecuta o reutiliza inferencia de distraccion con frecuencia reducida."""

    global _DISTRACTION_LAST_RUN_AT, _DISTRACTION_LAST_RESULT, _DISTRACTION_DISABLED_REASON

    if not _distraction_enabled():
        return _build_distraction_payload(
            enabled=False,
            ran=False,
            run_reason="disabled_by_env",
            state="UNKNOWN",
            confidence=0.0,
            class_probabilities={},
        )

    if face_response is None or not face_response.detections:
        if _DISTRACTION_LAST_RESULT is not None:
            return _build_distraction_payload(
                enabled=True,
                ran=False,
                run_reason="no_face_cached",
                state=str(_DISTRACTION_LAST_RESULT.get("state", "UNKNOWN")),
                confidence=float(_DISTRACTION_LAST_RESULT.get("confidence", 0.0) or 0.0),
                class_probabilities=dict(_DISTRACTION_LAST_RESULT.get("class_probabilities", {})),
            )
        return _build_distraction_payload(
            enabled=True,
            ran=False,
            run_reason="no_face",
            state="UNKNOWN",
            confidence=0.0,
            class_probabilities={},
        )

    if _DISTRACTION_DISABLED_REASON:
        return _build_distraction_payload(
            enabled=False,
            ran=False,
            run_reason=f"disabled:{_DISTRACTION_DISABLED_REASON}",
            state="UNKNOWN",
            confidence=0.0,
            class_probabilities={},
        )

    interval_seconds = _distraction_interval_seconds()
    should_run = (_DISTRACTION_LAST_RESULT is None) or ((now - _DISTRACTION_LAST_RUN_AT) >= interval_seconds)

    if should_run:
        try:
            raw = _get_tf_subprocess_client().infer_distraction_image_array(face_crop)
        except Exception as exc:  # pragma: no cover - defensa runtime
            error_text = str(exc)
            if "No existe modelo de distraccion" in error_text:
                _DISTRACTION_DISABLED_REASON = error_text
                return _build_distraction_payload(
                    enabled=False,
                    ran=False,
                    run_reason=f"disabled:{error_text}",
                    state="UNKNOWN",
                    confidence=0.0,
                    class_probabilities={},
                )
            if _DISTRACTION_LAST_RESULT is not None:
                return _build_distraction_payload(
                    enabled=True,
                    ran=False,
                    run_reason=f"worker_error_cached:{error_text}",
                    state=str(_DISTRACTION_LAST_RESULT.get("state", "UNKNOWN")),
                    confidence=float(_DISTRACTION_LAST_RESULT.get("confidence", 0.0) or 0.0),
                    class_probabilities=dict(_DISTRACTION_LAST_RESULT.get("class_probabilities", {})),
                )
            return _build_distraction_payload(
                enabled=True,
                ran=False,
                run_reason=f"worker_error:{error_text}",
                state="UNKNOWN",
                confidence=0.0,
                class_probabilities={},
            )

        class_probabilities = {
            str(key): float(value)
            for key, value in dict(raw.get("class_probabilities", {})).items()
        }
        if class_probabilities:
            raw_state = max(class_probabilities, key=class_probabilities.get)
            raw_confidence = float(class_probabilities[raw_state])
        else:
            raw_state = str(raw.get("state", "UNKNOWN"))
            raw_confidence = float(raw.get("confidence", 0.0) or 0.0)

        _DISTRACTION_LAST_RUN_AT = now
        _DISTRACTION_LAST_RESULT = {
            "state": raw_state,
            "confidence": raw_confidence,
            "class_probabilities": class_probabilities,
        }
        return _build_distraction_payload(
            enabled=True,
            ran=True,
            run_reason="inference",
            state=raw_state,
            confidence=raw_confidence,
            class_probabilities=class_probabilities,
        )

    if _DISTRACTION_LAST_RESULT is not None:
        return _build_distraction_payload(
            enabled=True,
            ran=False,
            run_reason="cached",
            state=str(_DISTRACTION_LAST_RESULT.get("state", "UNKNOWN")),
            confidence=float(_DISTRACTION_LAST_RESULT.get("confidence", 0.0) or 0.0),
            class_probabilities=dict(_DISTRACTION_LAST_RESULT.get("class_probabilities", {})),
        )

    return _build_distraction_payload(
        enabled=True,
        ran=False,
        run_reason="cache_miss",
        state="UNKNOWN",
        confidence=0.0,
        class_probabilities={},
    )

def _get_face_subprocess_client() -> FaceSubprocessClient:
    """Obtiene el cliente del worker facial."""

    global _FACE_SUBPROCESS_CLIENT

    if _FACE_SUBPROCESS_CLIENT is None:
        _FACE_SUBPROCESS_CLIENT = FaceSubprocessClient.from_settings(_settings())
    return _FACE_SUBPROCESS_CLIENT


def _get_tf_subprocess_client() -> TfSubprocessClient:
    """Obtiene el cliente del worker TensorFlow unificado."""

    global _TF_SUBPROCESS_CLIENT

    if _TF_SUBPROCESS_CLIENT is None:
        _TF_SUBPROCESS_CLIENT = TfSubprocessClient.from_settings(_settings())
    return _TF_SUBPROCESS_CLIENT


def _face_request_kwargs(payload: EyeStateImageRequest | DrowsinessImageRequest) -> Dict[str, Any]:
    """Extrae solo los campos de imagen que entiende el worker facial."""

    return {
        "image_b64": payload.image_b64,
        "image_path": payload.image_path,
    }


def _run_face_subprocess(payload: EyeStateImageRequest | DrowsinessImageRequest) -> FaceResponse:
    """Ejecuta InsightFace en subproceso y valida su salida."""

    try:
        raw_response = _get_face_subprocess_client().infer(**_face_request_kwargs(payload))
    except Exception as exc:  # pragma: no cover - defensa runtime
        LOGGER.exception("Face subprocess failure")
        raise HTTPException(
            status_code=503,
            detail=(
                "Fallo el worker de InsightFace. Revisa MOSTACHO_FACE_PYTHON y sus dependencias. "
                f"Detalle: {exc}"
            ),
        ) from exc

    return FaceResponse.model_validate(raw_response)


def _merge_eye_predictions(predictions: Sequence[Dict[str, Any]]) -> EyeStateResponse:
    """Promedia probabilidades de varios recortes de ojo."""

    if not predictions:
        return EyeStateResponse(
            engine="tf_subprocess_eye",
            state="NO_EYES",
            confidence=0.0,
            class_probabilities={},
            regions_used=0,
            eye_boxes=[],
            timestamp_utc=utc_now_iso(),
        )

    accumulator: Dict[str, float] = {}
    for payload in predictions:
        for class_name, probability in payload.get("class_probabilities", {}).items():
            accumulator[class_name] = accumulator.get(class_name, 0.0) + float(probability)

    averaged = {
        class_name: float(value / float(len(predictions)))
        for class_name, value in accumulator.items()
    }
    state = max(averaged, key=averaged.get)
    return EyeStateResponse(
        engine=str(predictions[0].get("engine", "tf_subprocess_eye")),
        state=state,
        confidence=float(averaged[state]),
        class_probabilities=averaged,
        regions_used=len(predictions),
        eye_boxes=[],
        timestamp_utc=utc_now_iso(),
    )


def _infer_eye_regions(eye_regions: Sequence[np.ndarray]) -> EyeStateResponse:
    """Clasifica una o varias regiones oculares usando el worker dedicado."""

    if not eye_regions:
        return EyeStateResponse(
            engine="tf_subprocess_eye",
            state="NO_EYES",
            confidence=0.0,
            class_probabilities={},
            regions_used=0,
            eye_boxes=[],
            timestamp_utc=utc_now_iso(),
        )

    client = _get_tf_subprocess_client()
    predictions: List[Dict[str, Any]] = []
    last_error: Exception | None = None

    for region in list(eye_regions)[:2]:
        try:
            predictions.append(client.infer_eye_image_array(region))
        except Exception as exc:  # pragma: no cover - defensa runtime
            last_error = exc
            LOGGER.warning("Fallo worker TF (eye) en una region: %s", exc)

    if not predictions and last_error is not None:
        LOGGER.exception("TF subprocess (eye) failure", exc_info=last_error)
        raise HTTPException(status_code=503, detail=f"Fallo el worker TensorFlow (open/closed eyes): {last_error}")

    return _merge_eye_predictions(predictions)


def _extract_eye_regions_for_request(
    image: np.ndarray,
    payload: EyeStateImageRequest | DrowsinessImageRequest,
    face_response: FaceResponse | None,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Resuelve las regiones oculares a clasificar para una imagen dada."""

    if payload.eye_crop:
        return [image], []

    settings = _settings()

    if face_response is not None and face_response.detections:
        regions, boxes = extract_eye_regions_from_detection(
            image,
            face_response.detections[0].model_dump(),
            padding=float(settings.eye_crop_padding),
        )
        if regions:
            return regions, boxes

    if payload.image_is_cropped:
        regions = extract_eye_regions_from_face(image)
        return regions, []

    if payload.face_bbox:
        face_crop = crop_image_by_bbox(image, payload.face_bbox)
        regions = extract_eye_regions_from_face(face_crop) if face_crop.size > 0 else []
        return regions, []

    return [], []


def _resolve_face_crop(
    image: np.ndarray,
    payload: DrowsinessImageRequest,
    face_response: FaceResponse | None,
) -> np.ndarray:
    """Obtiene el recorte facial que alimentará drowsiness_vision."""

    if payload.image_is_cropped:
        return image

    if face_response is not None and face_response.detections:
        crop = crop_image_by_bbox(image, face_response.detections[0].bbox)
        if crop.size > 0:
            return crop

    if payload.face_bbox:
        crop = crop_image_by_bbox(image, payload.face_bbox)
        if crop.size > 0:
            return crop

    return image


def _should_run_drowsiness(
    face_response: FaceResponse | None,
    eye_response: EyeStateResponse,
    now: float,
) -> Tuple[bool, str]:
    """Decide si conviene disparar el modelo pesado en este frame."""

    global _DROWSINESS_ACTIVE_UNTIL

    settings = _settings()
    eye_state = _normalize_eye_state(eye_response.state)
    eye_closed_trigger = eye_state == "CLOSED" and float(eye_response.confidence) >= float(settings.eye_closed_confidence_threshold)
    yawn_trigger = bool(face_response.yawn_candidate) if face_response is not None else False

    if face_response is None or not face_response.detections:
        _DROWSINESS_ACTIVE_UNTIL = 0.0
        return False, "no_face"

    if eye_closed_trigger:
        _DROWSINESS_ACTIVE_UNTIL = now + float(settings.drowsiness_active_seconds)
        return True, "eye_closed_gate"

    if yawn_trigger:
        _DROWSINESS_ACTIVE_UNTIL = now + float(settings.drowsiness_active_seconds)
        return True, "yawn_gate"

    if now < _DROWSINESS_ACTIVE_UNTIL:
        return True, "active_window"

    return False, "eye_open_skip"


def _skip_state(face_response: FaceResponse | None, eye_response: EyeStateResponse, run_reason: str) -> Tuple[str, float]:
    """Construye un estado liviano cuando el modelo pesado no corre."""

    eye_state = _normalize_eye_state(eye_response.state)
    vision_state = str(face_response.vision_state).upper() if face_response is not None else "NO_FACE"

    if run_reason == "no_face" or (vision_state == "NO_FACE" and eye_state in {"NO_FACE", "NO_EYES", "UNKNOWN"}):
        return "NO_FACE", 0.0
    if eye_state == "OPEN":
        return "ALERT", float(eye_response.confidence)
    if eye_state == "CLOSED":
        return "EYES_CLOSED", float(eye_response.confidence)
    return "ALERT", 0.50


def _resolve_final_state(
    eye_response: EyeStateResponse,
    model_ran: bool,
    model_state: str,
    model_confidence: float,
    now: float,
) -> Tuple[str, float, float]:
    """Reduce la salida final a una lógica simple basada en eye-state y microsleep sostenido."""

    global _MICROSLEEP_SINCE

    settings = _settings()
    eye_state = _normalize_eye_state(eye_response.state)
    microsleep_duration = 0.0

    if eye_state != "CLOSED" or not model_ran:
        _MICROSLEEP_SINCE = None
        if eye_state == "OPEN":
            return "ALERT", float(eye_response.confidence), microsleep_duration
        if eye_state == "CLOSED":
            return "EYES_CLOSED", float(eye_response.confidence), microsleep_duration
        return "NO_FACE", 0.0, microsleep_duration

    if model_state == "MICROSLEEP":
        if _MICROSLEEP_SINCE is None:
            _MICROSLEEP_SINCE = now
        microsleep_duration = max(0.0, now - _MICROSLEEP_SINCE)
        if microsleep_duration >= float(settings.microsleep_confirm_seconds):
            return "SOMNOLENT", float(max(model_confidence, eye_response.confidence)), microsleep_duration
        return "EYES_CLOSED", float(eye_response.confidence), microsleep_duration

    _MICROSLEEP_SINCE = None
    return "EYES_CLOSED", float(eye_response.confidence), microsleep_duration


def _predict_eye_state_for_payload(
    image: np.ndarray,
    payload: EyeStateImageRequest | DrowsinessImageRequest,
    face_response: FaceResponse | None,
) -> EyeStateResponse:
    """Resuelve eye-state ya sea desde eye-crop directo o desde el rostro detectado."""

    eye_regions, eye_boxes = _extract_eye_regions_for_request(image, payload, face_response)
    if not eye_regions and face_response is None and payload.image_is_cropped:
        eye_regions = extract_eye_regions_from_face(image)
    response = _infer_eye_regions(eye_regions)
    response.eye_boxes = [[int(value) for value in box] for box in eye_boxes]
    return response


@app.on_event("shutdown")
def _shutdown_workers() -> None:
    """Cierra clientes de subprocess al apagar la API."""

    global _FACE_SUBPROCESS_CLIENT, _TF_SUBPROCESS_CLIENT, _MICROSLEEP_SINCE, _DROWSINESS_ACTIVE_UNTIL
    global _DISTRACTION_LAST_RUN_AT, _DISTRACTION_LAST_RESULT, _DISTRACTION_DISABLED_REASON

    if _FACE_SUBPROCESS_CLIENT is not None:
        _FACE_SUBPROCESS_CLIENT.close()
        _FACE_SUBPROCESS_CLIENT = None

    if _TF_SUBPROCESS_CLIENT is not None:
        _TF_SUBPROCESS_CLIENT.close()
        _TF_SUBPROCESS_CLIENT = None

    _MICROSLEEP_SINCE = None
    _DROWSINESS_ACTIVE_UNTIL = 0.0
    _DISTRACTION_LAST_RUN_AT = 0.0
    _DISTRACTION_LAST_RESULT = None
    _DISTRACTION_DISABLED_REASON = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check del servicio combinado."""

    return HealthResponse(service="mostacho_v2.2_tf_api", status="ok", timestamp_utc=utc_now_iso())


@app.post("/predict/eye_state_image", response_model=EyeStateResponse)
def predict_eye_state_image(payload: EyeStateImageRequest) -> EyeStateResponse:
    """Clasifica ojos abiertos/cerrados usando el worker dedicado."""

    image = _decode_rgb_image(image_b64=payload.image_b64, image_path=payload.image_path)

    if payload.eye_crop:
        return _infer_eye_regions([image])

    face_response = _run_face_subprocess(payload)
    if not face_response.detections and not payload.image_is_cropped and not payload.face_bbox:
        return EyeStateResponse(
            engine="tf_subprocess_eye",
            state="NO_FACE",
            confidence=0.0,
            class_probabilities={},
            regions_used=0,
            timestamp_utc=utc_now_iso(),
            eye_boxes=[],
        )

    return _predict_eye_state_for_payload(image, payload, face_response)


@app.post("/predict/drowsiness_image", response_model=DrowsinessResponse)
def predict_drowsiness_image(payload: DrowsinessImageRequest) -> DrowsinessResponse:
    """Orquesta InsightFace + eye-state + drowsiness + distraction con gating."""

    image = _decode_rgb_image(image_b64=payload.image_b64, image_path=payload.image_path)
    face_response = _run_face_subprocess(payload)
    eye_response = _predict_eye_state_for_payload(image, payload, face_response)
    face_crop = _resolve_face_crop(image, payload, face_response)

    now = time.monotonic()
    model_ran, run_reason = _should_run_drowsiness(face_response, eye_response, now)
    distraction_payload = _resolve_distraction(face_crop=face_crop, face_response=face_response, now=now)

    class_probabilities: Dict[str, float] = {}
    model_state = "NOT_RUN"
    model_confidence = 0.0
    microsleep_duration = 0.0
    drowsiness_engine = "tf_subprocess"

    if model_ran:
        try:
            drowsiness_result = _get_tf_subprocess_client().infer_drowsiness_image_array(face_crop)
        except Exception as exc:  # pragma: no cover - defensa runtime
            raise HTTPException(status_code=503, detail=f"Fallo el worker TensorFlow (drowsiness): {exc}") from exc

        drowsiness_engine = str(drowsiness_result.get("engine", "tf_subprocess"))
        class_probabilities = {
            str(key): float(value)
            for key, value in dict(drowsiness_result.get("class_probabilities", {})).items()
        }
        if not class_probabilities:
            raise HTTPException(status_code=503, detail="Worker TensorFlow no devolvio class_probabilities de drowsiness.")
        model_state = _normalize_model_state(max(class_probabilities, key=class_probabilities.get))
        model_confidence = float(class_probabilities[max(class_probabilities, key=class_probabilities.get)])
        state, confidence, microsleep_duration = _resolve_final_state(
            eye_response=eye_response,
            model_ran=model_ran,
            model_state=model_state,
            model_confidence=model_confidence,
            now=now,
        )
    else:
        global _MICROSLEEP_SINCE
        _MICROSLEEP_SINCE = None
        state, confidence = _skip_state(face_response, eye_response, run_reason)

    return DrowsinessResponse(
        engine=_combined_engine_label(drowsiness_engine),
        state=state,
        confidence=float(confidence),
        model_ran=bool(model_ran),
        run_reason=run_reason,
        model_state=model_state,
        model_confidence=float(model_confidence),
        class_probabilities=class_probabilities,
        eye_state=_normalize_eye_state(eye_response.state),
        eye_confidence=float(eye_response.confidence),
        eye_class_probabilities={str(key): float(value) for key, value in eye_response.class_probabilities.items()},
        eye_regions_used=int(eye_response.regions_used),
        eye_boxes=[[int(value) for value in box] for box in eye_response.eye_boxes],
        vision_state=str(face_response.vision_state),
        microsleep_duration=float(microsleep_duration),
        microsleep_confirm_seconds=float(_settings().microsleep_confirm_seconds),
        avg_ear=float(face_response.avg_ear) if face_response.avg_ear is not None else None,
        closed_duration=float(face_response.closed_duration or 0.0),
        mouth_open_ratio=float(face_response.mouth_open_ratio) if face_response.mouth_open_ratio is not None else None,
        yawn_candidate=bool(face_response.yawn_candidate),
        distraction_enabled=bool(distraction_payload["distraction_enabled"]),
        distraction_ran=bool(distraction_payload["distraction_ran"]),
        distraction_run_reason=str(distraction_payload["distraction_run_reason"]),
        distraction_state=str(distraction_payload["distraction_state"]),
        distraction_confidence=float(distraction_payload["distraction_confidence"]),
        distraction_class_probabilities={
            str(key): float(value)
            for key, value in dict(distraction_payload["distraction_class_probabilities"]).items()
        },
        face_response=face_response,
        timestamp_utc=utc_now_iso(),
    )
