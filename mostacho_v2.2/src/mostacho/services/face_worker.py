"""Worker por stdin/stdout para InsightFace sin HTTP.

Protocolo:
- Cada linea de stdin debe ser un JSON con `image_b64` o `image_path`.
- Se devuelve una linea JSON con la respuesta (misma estructura de FaceResponse).
- Se recomienda usar `python -u` para evitar buffering.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, List

import numpy as np

from mostacho.schemas import FaceDetection, FaceRequest, FaceResponse, utc_now_iso
from mostacho.vision.runtime import VisionRuntime, VisionRuntimeConfig


_FACE_APP: Any = None
_RUNTIME: VisionRuntime | None = None


def _load_cv2() -> Any:
    """Importa OpenCV de forma diferida."""

    import cv2  # type: ignore

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

    global _FACE_APP

    if _FACE_APP is not None:
        return _FACE_APP

    from insightface.app import FaceAnalysis  # type: ignore

    model_name = os.getenv("MOSTACHO_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
    providers_raw = os.getenv("MOSTACHO_FACE_PROVIDERS", "CPUExecutionProvider")
    providers = [value.strip() for value in providers_raw.split(",") if value.strip()]
    allowed_modules = _resolve_landmark_modules(os.getenv("MOSTACHO_FACE_LANDMARKS"))
    detect_size = _parse_det_size(os.getenv("MOSTACHO_FACE_DET_SIZE"), default=(256, 256))

    try:
        face_app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules,
        )
    except Exception:
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=allowed_modules,
        )

    face_app.prepare(ctx_id=-1, det_size=detect_size)

    _FACE_APP = face_app
    return _FACE_APP


def _load_runtime() -> VisionRuntime:
    """Inicializa runtime de visión stateful y lo deja en cache."""

    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    face_app = _load_insightface_app()
    detect_size = _parse_det_size(os.getenv("MOSTACHO_FACE_DET_SIZE"), default=(256, 256))
    config = VisionRuntimeConfig(
        detect_size=detect_size,
        window_size=5,
        closed_seconds=3.0,
        calibration_seconds=5.0,
        threshold_offset=0.04,
        min_threshold=0.15,
        mouth_open_threshold=float(os.getenv("MOSTACHO_YAWN_TRIGGER_THRESHOLD", "0.32") or 0.32),
    )

    _RUNTIME = VisionRuntime(face_app=face_app, config=config)
    return _RUNTIME


def _decode_image_from_request(payload: FaceRequest) -> np.ndarray:
    """Decodifica imagen desde base64 o ruta local."""

    cv2 = _load_cv2()

    if payload.image_b64:
        raw_bytes = base64.b64decode(payload.image_b64)
        image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo decodificar image_b64.")
        return image

    if payload.image_path:
        image_path = Path(payload.image_path)
        if not image_path.exists():
            raise ValueError(f"No existe la ruta: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo leer imagen: {image_path}")
        return image

    raise ValueError("Debe enviar image_b64 o image_path.")


def _vision_features_from_response(
    detections: List[FaceDetection],
    image_shape: tuple[int, ...],
    response: FaceResponse,
) -> dict[str, float]:
    """Construye features de visión para fusión multimodal."""

    height, width = image_shape[:2]
    frame_area = float(height * width) if height and width else 1.0

    face_count = float(len(detections))
    avg_score = 0.0
    primary_face_area_ratio = 0.0

    if detections:
        avg_score = float(np.mean([det.score for det in detections]))
        x1, y1, x2, y2 = detections[0].bbox
        face_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        primary_face_area_ratio = float(face_area / frame_area)

    state = response.vision_state
    is_calibrating = 1.0 if state == "CALIBRATING" else 0.0
    is_attentive = 1.0 if state == "ATTENTIVE" else 0.0
    is_eyes_closed = 1.0 if state == "EYES_CLOSED" else 0.0
    is_somnolent = 1.0 if state == "SOMNOLENT" else 0.0

    features = {
        "vision_face_count": face_count,
        "vision_avg_score": avg_score,
        "vision_primary_face_area_ratio": primary_face_area_ratio,
        "vision_avg_ear": float(response.avg_ear) if response.avg_ear is not None else 0.0,
        "vision_closed_duration": float(response.closed_duration),
        "vision_mouth_open_ratio": float(response.mouth_open_ratio) if response.mouth_open_ratio is not None else 0.0,
        "vision_yawn_candidate": 1.0 if response.yawn_candidate else 0.0,
        "vision_is_calibrating": is_calibrating,
        "vision_is_attentive": is_attentive,
        "vision_is_eyes_closed": is_eyes_closed,
        "vision_is_somnolent": is_somnolent,
    }

    if response.threshold is not None:
        features["vision_threshold"] = float(response.threshold)
    if response.baseline is not None:
        features["vision_baseline"] = float(response.baseline)

    return features


def _handle_request(raw_request: dict[str, Any]) -> dict[str, Any]:
    """Procesa un request JSON y devuelve respuesta serializable."""

    request_id = raw_request.get("id")

    try:
        payload = FaceRequest(**raw_request)
        image = _decode_image_from_request(payload)
        runtime = _load_runtime()
        analysis = runtime.analyze_image(image, now=time.monotonic())

        detections: List[FaceDetection] = []
        for item in analysis.detections:
            detections.append(
                FaceDetection(
                    bbox=item.bbox,
                    score=item.score,
                    landmarks=item.landmarks,
                    left_eye=item.left_eye,
                    right_eye=item.right_eye,
                    mouth=item.mouth,
                    ear=item.ear,
                    mouth_open_ratio=item.mouth_open_ratio,
                )
            )

        response = FaceResponse(
            timestamp_utc=utc_now_iso(),
            detections=detections,
            vision_features={},
            vision_state=analysis.vision_state,
            avg_ear=analysis.avg_ear,
            closed_duration=analysis.closed_duration,
            threshold=analysis.threshold,
            baseline=analysis.baseline,
            mouth_open_ratio=analysis.mouth_open_ratio,
            yawn_candidate=analysis.yawn_candidate,
        )
        response.vision_features = _vision_features_from_response(detections, image.shape, response)

        result = response.model_dump()
        if request_id is not None:
            result["id"] = request_id
        return result

    except Exception as exc:  # pragma: no cover - defensa runtime
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(limit=2),
        }
        if request_id is not None:
            error_payload["id"] = request_id
        return error_payload


def _write_response(payload: dict[str, Any]) -> None:
    """Escribe respuesta JSON en stdout."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Loop principal: lee stdin y responde por stdout."""

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _write_response({"error": "JSON invalido"})
            continue

        if isinstance(request, dict) and request.get("command") == "shutdown":
            _write_response({"status": "shutdown"})
            break

        if not isinstance(request, dict):
            _write_response({"error": "Formato invalido"})
            continue

        response = _handle_request(request)
        _write_response(response)


if __name__ == "__main__":
    main()
