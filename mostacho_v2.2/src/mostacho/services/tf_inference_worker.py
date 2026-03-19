"""Worker TensorFlow unificado para eyes/drowsiness/distraction por stdin/stdout."""

from __future__ import annotations

import base64
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mostacho.eye_state import CLASS_NAMES as EYE_CLASS_NAMES
from mostacho.eye_state import eye_state_probabilities_from_raw, prepare_eye_image_array
from mostacho.schemas import EyeStateResponse, utc_now_iso
from mostacho.settings import load_settings


_EYE_MODEL = None
_EYE_INPUT_SIZE = 96
_EYE_CLASS_NAMES = list(EYE_CLASS_NAMES)

_DROWSINESS_MODEL = None
_DROWSINESS_ENGINE: str | None = None
_DROWSINESS_TFLITE = None
_DROWSINESS_TFLITE_INPUT_DETAILS = None
_DROWSINESS_TFLITE_OUTPUT_DETAILS = None
_DROWSINESS_CLASS_NAMES: List[str] = []
_DROWSINESS_INPUT_SIZE: Tuple[int, int] = (160, 160)

_DISTRACTION_MODEL = None
_DISTRACTION_INPUT_SIZE: Tuple[int, int] = (224, 224)
_DISTRACTION_CLASS_NAMES: List[str] = ["normal", "distracted"]


def _softmax(values: np.ndarray) -> np.ndarray:
    """Softmax estable numericamente."""

    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


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


def _eye_model_paths() -> tuple[Path, Path]:
    """Resuelve modelo y clases del clasificador open/closed eyes."""

    settings = load_settings()
    default_best = settings.artifacts_root / "models" / "eye_state_model_best.keras"
    default_final = settings.artifacts_root / "models" / "eye_state_model.keras"
    default_classes = settings.artifacts_root / "models" / "eye_state_classes.txt"

    model_path = Path(
        (os.getenv("MOSTACHO_EYE_STATE_MODEL", "").strip() or (str(default_best) if default_best.exists() else str(default_final)))
    )
    classes_path = Path(os.getenv("MOSTACHO_EYE_STATE_CLASSES", str(default_classes)))
    return model_path, classes_path


def _load_eye_model_once() -> None:
    """Carga el modelo de eyes una sola vez."""

    global _EYE_MODEL, _EYE_INPUT_SIZE, _EYE_CLASS_NAMES

    if _EYE_MODEL is not None:
        return

    import tensorflow as tf  # type: ignore

    model_path, classes_path = _eye_model_paths()
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo open/closed eyes: {model_path}")

    _EYE_MODEL = tf.keras.models.load_model(model_path, compile=False)

    input_shape = getattr(_EYE_MODEL, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        _EYE_INPUT_SIZE = int(input_shape[1] or 96)

    if classes_path.exists():
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if class_names:
            _EYE_CLASS_NAMES = class_names


def _infer_eye(image: np.ndarray) -> Dict[str, Any]:
    """Infiere estado OPEN/CLOSED y devuelve payload compatible con EyeStateResponse."""

    _load_eye_model_once()

    batch = prepare_eye_image_array(image, image_size=_EYE_INPUT_SIZE)
    raw = _EYE_MODEL(batch, training=False).numpy().reshape(-1)
    class_probabilities = eye_state_probabilities_from_raw(raw, class_names=_EYE_CLASS_NAMES)

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

    return EyeStateResponse(
        engine="tf_worker_eye",
        state=state,
        confidence=confidence,
        class_probabilities=class_probabilities,
        regions_used=1,
        eye_boxes=[],
        timestamp_utc=utc_now_iso(),
    ).model_dump()


def _drowsiness_model_paths() -> tuple[Path, Path]:
    """Resuelve rutas del modelo visual de somnolencia y clases."""

    settings = load_settings()
    default_best = settings.artifacts_root / "models" / "drowsiness_vision_best.keras"
    default_final = settings.artifacts_root / "models" / "drowsiness_vision.keras"
    default_classes = settings.artifacts_root / "models" / "drowsiness_vision_classes.txt"

    model_override = os.getenv("MOSTACHO_DROWSINESS_MODEL", "").strip()
    if model_override:
        model_path = Path(model_override)
    else:
        model_path = default_best if default_best.exists() else default_final

    classes_path = Path(os.getenv("MOSTACHO_DROWSINESS_CLASSES", str(default_classes)))
    return model_path, classes_path


def _load_drowsiness_model_once() -> None:
    """Carga modelo de somnolencia (Keras o TFLite) una sola vez."""

    global _DROWSINESS_MODEL, _DROWSINESS_CLASS_NAMES, _DROWSINESS_INPUT_SIZE
    global _DROWSINESS_ENGINE, _DROWSINESS_TFLITE, _DROWSINESS_TFLITE_INPUT_DETAILS, _DROWSINESS_TFLITE_OUTPUT_DETAILS

    if _DROWSINESS_ENGINE is not None:
        return

    import tensorflow as tf  # type: ignore

    model_path, classes_path = _drowsiness_model_paths()

    tflite_override = os.getenv("MOSTACHO_DROWSINESS_TFLITE", "").strip()
    if tflite_override:
        tflite_path = Path(tflite_override)
        if tflite_path.exists():
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            _DROWSINESS_TFLITE = interpreter
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            _DROWSINESS_TFLITE_INPUT_DETAILS = input_details[0] if input_details else None
            _DROWSINESS_TFLITE_OUTPUT_DETAILS = output_details[0] if output_details else None
            if _DROWSINESS_TFLITE_INPUT_DETAILS is not None:
                shape = _DROWSINESS_TFLITE_INPUT_DETAILS.get("shape")
                if shape is not None and len(shape) >= 3:
                    _DROWSINESS_INPUT_SIZE = (int(shape[1] or 160), int(shape[2] or 160))
            _DROWSINESS_ENGINE = "tflite"

    if _DROWSINESS_ENGINE != "tflite":
        if not model_path.exists():
            raise FileNotFoundError(f"No existe modelo de somnolencia: {model_path}")
        _DROWSINESS_MODEL = tf.keras.models.load_model(model_path, compile=False)
        _DROWSINESS_ENGINE = "keras"

        input_shape = getattr(_DROWSINESS_MODEL, "input_shape", None)
        if isinstance(input_shape, tuple) and len(input_shape) >= 3:
            _DROWSINESS_INPUT_SIZE = (int(input_shape[1] or 160), int(input_shape[2] or 160))

    if classes_path.exists():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        _DROWSINESS_CLASS_NAMES = names or ["alert", "yawning", "microsleep"]
    else:
        _DROWSINESS_CLASS_NAMES = ["alert", "yawning", "microsleep"]


def _prepare_tflite_input(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    input_details: Dict[str, Any],
) -> np.ndarray:
    """Prepara entrada para interpreter TFLite."""

    import tensorflow as tf  # type: ignore

    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (target_h, target_w))
    tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
    array = np.expand_dims(tensor.numpy(), axis=0)

    input_dtype = input_details.get("dtype", np.float32)
    if input_dtype in (np.uint8, np.int8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            array = np.round(array / scale + zero_point)
        info = np.iinfo(input_dtype)
        array = np.clip(array, info.min, info.max).astype(input_dtype)
    else:
        array = array.astype(input_dtype)
    return array


def _dequantize_output(raw: np.ndarray, output_details: Dict[str, Any]) -> np.ndarray:
    """De-cuantiza salida TFLite si aplica."""

    output_dtype = output_details.get("dtype", np.float32)
    if output_dtype in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def _infer_drowsiness(image: np.ndarray) -> Dict[str, Any]:
    """Infiere probabilidades del modelo de somnolencia."""

    _load_drowsiness_model_once()

    if _DROWSINESS_ENGINE == "tflite":
        if _DROWSINESS_TFLITE is None or _DROWSINESS_TFLITE_INPUT_DETAILS is None or _DROWSINESS_TFLITE_OUTPUT_DETAILS is None:
            raise RuntimeError("Interpreter TFLite de somnolencia no disponible.")
        target_h, target_w = _DROWSINESS_INPUT_SIZE
        input_data = _prepare_tflite_input(image, target_h, target_w, _DROWSINESS_TFLITE_INPUT_DETAILS)
        _DROWSINESS_TFLITE.set_tensor(_DROWSINESS_TFLITE_INPUT_DETAILS["index"], input_data)
        _DROWSINESS_TFLITE.invoke()
        raw = _DROWSINESS_TFLITE.get_tensor(_DROWSINESS_TFLITE_OUTPUT_DETAILS["index"]).reshape(-1)
        raw = _dequantize_output(raw, _DROWSINESS_TFLITE_OUTPUT_DETAILS)
    else:
        import tensorflow as tf  # type: ignore

        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        target_h, target_w = _DROWSINESS_INPUT_SIZE
        tensor = tf.image.resize(tensor, (target_h, target_w))
        tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
        batch = tf.expand_dims(tensor, axis=0)
        raw = _DROWSINESS_MODEL(batch, training=False).numpy().reshape(-1)

    if np.any(raw < 0.0) or not np.isclose(np.sum(raw), 1.0, atol=1e-2):
        probs = _softmax(raw.astype(np.float64)).astype(np.float32)
    else:
        probs = raw.astype(np.float32)

    class_names = _DROWSINESS_CLASS_NAMES if len(_DROWSINESS_CLASS_NAMES) == len(probs) else [f"class_{idx}" for idx in range(len(probs))]
    class_probabilities = {name: float(prob) for name, prob in zip(class_names, probs)}
    state = max(class_probabilities, key=class_probabilities.get)

    return {
        "engine": f"tf_worker_drowsiness_{_DROWSINESS_ENGINE}",
        "state": state,
        "confidence": float(class_probabilities[state]),
        "class_probabilities": class_probabilities,
        "timestamp_utc": utc_now_iso(),
    }


def _distraction_model_paths() -> tuple[Path | None, Path | None]:
    """Resuelve rutas candidatas del modelo de distraccion (opcional)."""

    settings = load_settings()
    model_override = os.getenv("MOSTACHO_DISTRACTION_MODEL", "").strip()
    classes_override = os.getenv("MOSTACHO_DISTRACTION_CLASSES", "").strip()

    if model_override:
        model_path: Path | None = Path(model_override)
    else:
        candidates = [
            settings.artifacts_root / "dmd_processed_dmd_r3" / "models" / "dmd_distraction_best.keras",
            settings.artifacts_root / "dmd_processed" / "models" / "dmd_distraction_best.keras",
            settings.artifacts_root / "models" / "distraction_model_best.keras",
        ]
        model_path = next((item for item in candidates if item.exists()), None)

    classes_path = Path(classes_override) if classes_override else None
    return model_path, classes_path


def _load_distraction_model_once() -> None:
    """Carga modelo de distraccion si existe."""

    global _DISTRACTION_MODEL, _DISTRACTION_CLASS_NAMES, _DISTRACTION_INPUT_SIZE

    if _DISTRACTION_MODEL is not None:
        return

    import tensorflow as tf  # type: ignore

    model_path, classes_path = _distraction_model_paths()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("No existe modelo de distraccion. Configure MOSTACHO_DISTRACTION_MODEL.")

    _DISTRACTION_MODEL = tf.keras.models.load_model(model_path, compile=False)
    input_shape = getattr(_DISTRACTION_MODEL, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        _DISTRACTION_INPUT_SIZE = (int(input_shape[1] or 224), int(input_shape[2] or 224))

    if classes_path and classes_path.exists():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            _DISTRACTION_CLASS_NAMES = names


def _infer_distraction(image: np.ndarray) -> Dict[str, Any]:
    """Infiere distraccion binaria: normal vs distracted."""

    import tensorflow as tf  # type: ignore

    _load_distraction_model_once()

    target_h, target_w = _DISTRACTION_INPUT_SIZE
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (target_h, target_w))
    batch = tf.expand_dims(tensor, axis=0)
    raw = _DISTRACTION_MODEL(batch, training=False).numpy().reshape(-1)

    if len(raw) == 1:
        distracted = float(np.clip(raw[0], 0.0, 1.0))
        class_probabilities = {
            "normal": float(1.0 - distracted),
            "distracted": distracted,
        }
    else:
        if np.any(raw < 0.0) or not np.isclose(np.sum(raw), 1.0, atol=1e-2):
            probs = _softmax(raw.astype(np.float64)).astype(np.float32)
        else:
            probs = raw.astype(np.float32)
        class_names = _DISTRACTION_CLASS_NAMES if len(_DISTRACTION_CLASS_NAMES) == len(probs) else [f"class_{idx}" for idx in range(len(probs))]
        class_probabilities = {name: float(prob) for name, prob in zip(class_names, probs)}

    state = max(class_probabilities, key=class_probabilities.get)
    return {
        "engine": "tf_worker_distraction_keras",
        "state": state,
        "confidence": float(class_probabilities[state]),
        "class_probabilities": class_probabilities,
        "timestamp_utc": utc_now_iso(),
    }


def _handle_request(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    """Procesa request JSON y devuelve una respuesta serializable."""

    request_id = raw_request.get("id")
    command = str(raw_request.get("command", "infer_eye")).strip().lower()

    if command == "shutdown":
        payload = {"status": "shutting_down"}
        if request_id is not None:
            payload["id"] = request_id
        return payload

    try:
        image = _decode_rgb_image(raw_request)

        if command == "infer_eye":
            response = _infer_eye(image)
        elif command == "infer_drowsiness":
            response = _infer_drowsiness(image)
        elif command == "infer_distraction":
            response = _infer_distraction(image)
        else:
            raise ValueError(f"Comando no soportado: {command}")

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
        line = line.strip()
        if not line:
            continue

        try:
            raw_request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = _handle_request(raw_request)
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()

        if response.get("status") == "shutting_down":
            break


if __name__ == "__main__":
    main()
