"""Worker por stdin/stdout para el modelo open/closed eyes."""

from __future__ import annotations

import base64
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mostacho.eye_state import CLASS_NAMES, eye_state_probabilities_from_raw, prepare_eye_image_array
from mostacho.schemas import EyeStateResponse, utc_now_iso
from mostacho.settings import load_settings


_MODEL = None
_INPUT_SIZE = 96
_CLASS_NAMES = list(CLASS_NAMES)


def _model_paths() -> tuple[Path, Path]:
    """Resuelve modelo y archivo de clases del worker."""

    settings = load_settings()
    default_best = settings.artifacts_root / "models" / "eye_state_model_best.keras"
    default_final = settings.artifacts_root / "models" / "eye_state_model.keras"
    default_classes = settings.artifacts_root / "models" / "eye_state_classes.txt"

    model_path = Path(
        (os.getenv("MOSTACHO_EYE_STATE_MODEL", "").strip() or (str(default_best) if default_best.exists() else str(default_final)))
    )
    classes_path = Path(os.getenv("MOSTACHO_EYE_STATE_CLASSES", str(default_classes)))
    return model_path, classes_path


def _load_model_once() -> None:
    """Carga modelo Keras una sola vez."""

    global _MODEL, _INPUT_SIZE, _CLASS_NAMES

    if _MODEL is not None:
        return

    import tensorflow as tf  # type: ignore

    model_path, classes_path = _model_paths()
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo open/closed eyes: {model_path}")

    _MODEL = tf.keras.models.load_model(model_path, compile=False)

    input_shape = getattr(_MODEL, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        _INPUT_SIZE = int(input_shape[1] or 96)

    if classes_path.exists():
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if class_names:
            _CLASS_NAMES = class_names


def _decode_rgb_image(raw_request: Dict[str, Any]) -> np.ndarray:
    """Decodifica imagen RGB desde image_b64 o image_path."""

    import tensorflow as tf  # type: ignore

    if raw_request.get("image_b64"):
        raw_bytes = base64.b64decode(str(raw_request["image_b64"]))
    elif raw_request.get("image_path"):
        image_path = Path(str(raw_request["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"No existe la ruta: {image_path}")
        raw_bytes = image_path.read_bytes()
    else:
        raise ValueError("Debe enviar image_b64 o image_path.")

    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    array = image.numpy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("No se pudo decodificar una imagen RGB valida.")
    return array


def _handle_request(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    """Procesa request JSON y devuelve una respuesta serializable."""

    request_id = raw_request.get("id")
    if raw_request.get("command") == "shutdown":
        payload = {"status": "shutting_down"}
        if request_id is not None:
            payload["id"] = request_id
        return payload

    try:
        _load_model_once()
        image = _decode_rgb_image(raw_request)
        batch = prepare_eye_image_array(image, image_size=_INPUT_SIZE)
        raw = _MODEL(batch, training=False).numpy().reshape(-1)
        class_probabilities = eye_state_probabilities_from_raw(raw, class_names=_CLASS_NAMES)
        settings = load_settings()
        if settings.eye_invert_output and len(class_probabilities) >= 2:
            closed_value = float(class_probabilities.get("Closed_Eyes", 0.0))
            open_value = float(class_probabilities.get("Open_Eyes", 0.0))
            class_probabilities = {
                "Closed_Eyes": open_value,
                "Open_Eyes": closed_value,
            }
        state = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[state])

        response = EyeStateResponse(
            engine="keras_subprocess",
            state=state,
            confidence=confidence,
            class_probabilities=class_probabilities,
            regions_used=1,
            eye_boxes=[],
            timestamp_utc=utc_now_iso(),
        ).model_dump()
        if request_id is not None:
            response["id"] = request_id
        return response
    except Exception as exc:  # pragma: no cover - defensa runtime
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(limit=2),
        }
        if request_id is not None:
            error_payload["id"] = request_id
        return error_payload


def main() -> None:
    """Loop principal del worker stdin/stdout."""

    import sys

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"error": "JSON invalido"}, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            continue

        response = _handle_request(payload)
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()

        if payload.get("command") == "shutdown":
            break


if __name__ == "__main__":
    main()
