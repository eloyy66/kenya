"""Servicio FastAPI para fusión multimodal y somnolencia visual con TensorFlow."""

from __future__ import annotations

# base64 para decodificar imágenes enviadas por HTTP.
import base64
# json para cargar metadata de modelos.
import json
# logging para observabilidad operativa.
import logging
# os para variables de entorno.
import os
# Path para rutas de artefactos.
from pathlib import Path
# typing para contratos explícitos.
from typing import Any, Dict, List, Tuple

# numpy para vectores y operaciones numéricas.
import numpy as np
# FastAPI para exponer endpoints.
from fastapi import FastAPI, HTTPException

# Esquemas compartidos entre servicios.
from mostacho.schemas import (
    DrowsinessImageRequest,
    DrowsinessResponse,
    FusionRequest,
    FusionResponse,
    HealthResponse,
    utc_now_iso,
)
# Configuración global del proyecto.
from mostacho.settings import load_settings


# Logger del módulo para seguimiento de estado y errores.
LOGGER = logging.getLogger(__name__)
# Aplicación principal del servicio TF.
app = FastAPI(title="Mostacho TF Service", version="0.2.0")

# Cache global de modelo multimodal tabular.
_MODEL = None
# Cache del orden de features del modelo tabular.
_FEATURE_ORDER: List[str] = []
# Clases del modelo multimodal tabular.
CLASSES = ["alert", "somnolent", "stressed", "distracted"]

# Cache del modelo visual de somnolencia por imagen.
_DROWSINESS_MODEL = None
# Cache del motor en uso: "keras", "tflite" o "none".
_DROWSINESS_ENGINE: str | None = None
# Cache del intérprete TFLite cuando aplica.
_DROWSINESS_TFLITE = None
_DROWSINESS_TFLITE_INPUT_DETAILS = None
_DROWSINESS_TFLITE_OUTPUT_DETAILS = None
# Cache de clases del modelo visual de somnolencia.
_DROWSINESS_CLASS_NAMES: List[str] = []
# Tamaño de entrada del modelo visual (alto, ancho).
_DROWSINESS_INPUT_SIZE: Tuple[int, int] = (160, 160)


def _model_paths() -> Tuple[Path, Path]:
    """Resuelve rutas del modelo tabular de fusión y su metadata."""

    # Carga configuración base de rutas.
    settings = load_settings()
    # Modelo tabular por defecto.
    default_model = settings.artifacts_root / "models" / "multimodal_fusion.keras"
    # Orden de features tabular por defecto.
    default_order = settings.artifacts_root / "models" / "multimodal_feature_order.json"

    # Permite override por variables de entorno.
    model_path = Path(os.getenv("MOSTACHO_TF_MODEL", str(default_model)))
    feature_order_path = Path(os.getenv("MOSTACHO_TF_FEATURE_ORDER", str(default_order)))

    # Retorna rutas resueltas.
    return model_path, feature_order_path


def _drowsiness_model_paths() -> Tuple[Path, Path]:
    """Resuelve rutas del modelo visual de somnolencia y sus clases."""

    # Carga configuración base de rutas.
    settings = load_settings()

    # Ruta preferida para mejor checkpoint de validación.
    default_best = settings.artifacts_root / "models" / "drowsiness_vision_best.keras"
    # Ruta fallback del modelo final.
    default_final = settings.artifacts_root / "models" / "drowsiness_vision.keras"
    # Archivo de clases para mapping de salida.
    default_classes = settings.artifacts_root / "models" / "drowsiness_vision_classes.txt"

    # Permite override completo del modelo visual.
    model_override = os.getenv("MOSTACHO_DROWSINESS_MODEL", "").strip()
    if model_override:
        model_path = Path(model_override)
    else:
        model_path = default_best if default_best.exists() else default_final

    # Permite override del archivo de clases.
    classes_path = Path(os.getenv("MOSTACHO_DROWSINESS_CLASSES", str(default_classes)))

    # Retorna rutas resueltas.
    return model_path, classes_path


def _load_tf_model_once() -> None:
    """Carga una sola vez el modelo tabular de fusión y su metadata."""

    # Accede a caches globales.
    global _MODEL, _FEATURE_ORDER

    # Si ya está cargado, no repite trabajo.
    if _MODEL is not None:
        return

    # Import local para no romper entornos sin TensorFlow.
    import tensorflow as tf  # type: ignore

    # Obtiene rutas del modelo tabular.
    model_path, feature_order_path = _model_paths()

    # Si no existe modelo, habilita modo heurístico.
    if not model_path.exists():
        LOGGER.warning("No existe modelo tabular en %s. Se usara modo heuristico.", model_path)
        _MODEL = False
        _FEATURE_ORDER = []
        return

    # Carga modelo Keras tabular.
    _MODEL = tf.keras.models.load_model(model_path)

    # Carga orden de features si existe.
    if feature_order_path.exists():
        _FEATURE_ORDER = json.loads(feature_order_path.read_text(encoding="utf-8"))
    else:
        _FEATURE_ORDER = []


def _load_drowsiness_model_once() -> None:
    """Carga una sola vez el modelo visual de somnolencia y sus clases."""

    # Accede a caches globales.
    global _DROWSINESS_MODEL, _DROWSINESS_CLASS_NAMES, _DROWSINESS_INPUT_SIZE
    global _DROWSINESS_ENGINE, _DROWSINESS_TFLITE, _DROWSINESS_TFLITE_INPUT_DETAILS, _DROWSINESS_TFLITE_OUTPUT_DETAILS

    # Si ya está cargado, no repite.
    if _DROWSINESS_ENGINE is not None:
        return

    # Import local para mantener aislamiento de dependencia.
    import tensorflow as tf  # type: ignore

    # Resuelve rutas de modelo y clases.
    model_path, classes_path = _drowsiness_model_paths()

    # Permite uso de modelo TFLite si se provee explícitamente.
    tflite_override = os.getenv("MOSTACHO_DROWSINESS_TFLITE", "").strip()
    if tflite_override:
        tflite_path = Path(tflite_override)
        if tflite_path.exists():
            try:
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                _DROWSINESS_TFLITE = interpreter
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                _DROWSINESS_TFLITE_INPUT_DETAILS = input_details[0] if input_details else None
                _DROWSINESS_TFLITE_OUTPUT_DETAILS = output_details[0] if output_details else None

                # Inferir tamaño de entrada desde el modelo TFLite.
                if _DROWSINESS_TFLITE_INPUT_DETAILS is not None:
                    shape = _DROWSINESS_TFLITE_INPUT_DETAILS.get("shape")
                    if shape is not None and len(shape) >= 3:
                        height = int(shape[1] or 160)
                        width = int(shape[2] or 160)
                        _DROWSINESS_INPUT_SIZE = (height, width)

                _DROWSINESS_ENGINE = "tflite"
                LOGGER.info("Drowsiness engine: TFLite (%s)", tflite_path)
            except Exception as exc:
                LOGGER.warning("No se pudo cargar TFLite en %s (%s). Se usa Keras.", tflite_path, exc)

    if _DROWSINESS_ENGINE != "tflite":
        # Si no existe modelo, habilita fallback heurístico.
        if not model_path.exists():
            LOGGER.warning("No existe modelo visual de somnolencia en %s. Se usara fallback heuristico.", model_path)
            _DROWSINESS_MODEL = False
            _DROWSINESS_ENGINE = "none"
            _DROWSINESS_CLASS_NAMES = ["alert", "yawning", "microsleep"]
            _DROWSINESS_INPUT_SIZE = (160, 160)
            return

        # Carga modelo visual de somnolencia.
        model = tf.keras.models.load_model(model_path, compile=False)
        _DROWSINESS_MODEL = model
        _DROWSINESS_ENGINE = "keras"
        LOGGER.info("Drowsiness engine: Keras (%s)", model_path)

        # Intenta inferir tamaño de entrada del modelo.
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, tuple) and len(input_shape) >= 3:
            height = int(input_shape[1] or 160)
            width = int(input_shape[2] or 160)
            _DROWSINESS_INPUT_SIZE = (height, width)
        else:
            _DROWSINESS_INPUT_SIZE = (160, 160)

    # Carga clases desde archivo si existe.
    if classes_path.exists():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            _DROWSINESS_CLASS_NAMES = names
        else:
            _DROWSINESS_CLASS_NAMES = ["alert", "yawning", "microsleep"]
    else:
        _DROWSINESS_CLASS_NAMES = ["alert", "yawning", "microsleep"]


def _flatten_features(payload: FusionRequest) -> Dict[str, float]:
    """Une las modalidades en un único diccionario numérico."""

    # Contenedor final de features fusionadas.
    merged: Dict[str, float] = {}
    # Agrega features de visión.
    merged.update(payload.vision_features)
    # Agrega features de voz.
    merged.update(payload.voice_features)
    # Agrega features de biometría.
    merged.update(payload.biometric_features)

    # Filtra a valores estrictamente numéricos.
    numeric_only = {key: float(value) for key, value in merged.items() if isinstance(value, (int, float))}
    return numeric_only


def _heuristic_predict(features: Dict[str, float]) -> Dict[str, float]:
    """Heurística fallback para el modelo tabular multimodal."""

    # Puntaje base uniforme.
    scores = {
        "alert": 0.25,
        "somnolent": 0.25,
        "stressed": 0.25,
        "distracted": 0.25,
    }

    # Reglas simples por señales de visión y biometría.
    if features.get("vision_face_count", 0.0) < 0.5:
        scores["distracted"] += 0.25
    if features.get("vision_primary_face_area_ratio", 0.0) < 0.08:
        scores["distracted"] += 0.15
    if features.get("voice_rms_mean", 0.0) < 0.02:
        scores["somnolent"] += 0.20
    if features.get("bio_available", 0.0) > 0.5 and features.get("bio_num_rows", 0.0) > 10:
        scores["stressed"] += 0.10

    # Normalización a distribución de probabilidad.
    values = np.array([scores[label] for label in CLASSES], dtype=np.float32)
    probs = values / np.sum(values)

    # Retorna mapa clase -> probabilidad.
    return {label: float(prob) for label, prob in zip(CLASSES, probs)}


def _model_predict(features: Dict[str, float]) -> Dict[str, float]:
    """Predice con modelo tabular entrenado o fallback heurístico."""

    # Asegura carga del modelo tabular.
    _load_tf_model_once()

    # Si no hay modelo real, usa heurística.
    if _MODEL is False:
        return _heuristic_predict(features)

    # Si no hay orden de features, también usa heurística.
    if not _FEATURE_ORDER:
        return _heuristic_predict(features)

    # Vectoriza según orden de entrenamiento.
    vector = np.array([features.get(name, 0.0) for name in _FEATURE_ORDER], dtype=np.float32).reshape(1, -1)

    # Predicción del modelo tabular.
    raw = _MODEL.predict(vector, verbose=0)
    probabilities = np.asarray(raw).reshape(-1)

    # Valida dimensionalidad esperada.
    if probabilities.shape[0] != len(CLASSES):
        return _heuristic_predict(features)

    # Mapea salida a nombre de clase.
    return {label: float(prob) for label, prob in zip(CLASSES, probabilities)}


def _decode_rgb_image(payload: DrowsinessImageRequest) -> np.ndarray:
    """Decodifica imagen RGB desde base64 o ruta local."""

    # Import local de TensorFlow para decode de imagen.
    import tensorflow as tf  # type: ignore

    # Caso base64.
    if payload.image_b64:
        raw_bytes = base64.b64decode(payload.image_b64)
    # Caso ruta local.
    elif payload.image_path:
        image_path = Path(payload.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"No existe image_path: {image_path}")
        raw_bytes = image_path.read_bytes()
    else:
        raise HTTPException(status_code=400, detail="Debe enviar image_b64 o image_path.")

    # Decodifica a tensor RGB uint8.
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    # Convierte a numpy para manipulación de recorte.
    array = image.numpy()

    # Validación de forma mínima.
    if array.ndim != 3 or array.shape[2] != 3:
        raise HTTPException(status_code=400, detail="No se pudo decodificar imagen RGB válida.")

    # Retorna arreglo HxWx3.
    return array


def _crop_face(image: np.ndarray, face_bbox: List[float] | None) -> np.ndarray:
    """Recorta rostro principal según bbox; fallback a imagen completa."""

    # Si no hay bbox, usa imagen completa.
    if not face_bbox or len(face_bbox) != 4:
        return image

    # Dimensiones de imagen.
    height, width = image.shape[:2]

    # Extrae coordenadas y aplica redondeo seguro.
    x1, y1, x2, y2 = [int(round(value)) for value in face_bbox]

    # Clampea coordenadas al borde de imagen.
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Si bbox inválido, usa imagen completa.
    if x2 <= x1 or y2 <= y1:
        return image

    # Recorta región facial.
    crop = image[y1:y2, x1:x2]

    # Si recorte sale vacío por alguna razón, usa imagen completa.
    if crop.size == 0:
        return image

    # Retorna recorte válido.
    return crop


def _softmax(values: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable para fallback de normalización."""

    # Resta máximo para estabilidad numérica.
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _prepare_tflite_input(
    face_crop: np.ndarray,
    target_h: int,
    target_w: int,
    input_details: Dict[str, Any],
) -> np.ndarray:
    """Prepara entrada para intérprete TFLite, con cuantización opcional."""

    # Import local de TensorFlow para resize/preprocess.
    import tensorflow as tf  # type: ignore

    tensor = tf.convert_to_tensor(face_crop, dtype=tf.float32)
    tensor = tf.image.resize(tensor, (target_h, target_w))

    preprocess_raw = os.getenv("MOSTACHO_TFLITE_PREPROCESS", "1").strip().lower()
    if preprocess_raw not in {"0", "false", "no"}:
        tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)

    array = tensor.numpy()
    array = np.expand_dims(array, axis=0)

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


def _predict_drowsiness_image_tflite(face_crop: np.ndarray) -> np.ndarray:
    """Predice con motor TFLite y retorna vector de probabilidades crudas."""

    if _DROWSINESS_TFLITE is None or _DROWSINESS_TFLITE_INPUT_DETAILS is None or _DROWSINESS_TFLITE_OUTPUT_DETAILS is None:
        return np.array([0.60, 0.25, 0.15], dtype=np.float32)

    target_h, target_w = _DROWSINESS_INPUT_SIZE
    input_data = _prepare_tflite_input(
        face_crop=face_crop,
        target_h=target_h,
        target_w=target_w,
        input_details=_DROWSINESS_TFLITE_INPUT_DETAILS,
    )

    _DROWSINESS_TFLITE.set_tensor(_DROWSINESS_TFLITE_INPUT_DETAILS["index"], input_data)
    _DROWSINESS_TFLITE.invoke()
    output = _DROWSINESS_TFLITE.get_tensor(_DROWSINESS_TFLITE_OUTPUT_DETAILS["index"]).reshape(-1)
    output = _dequantize_output(output, _DROWSINESS_TFLITE_OUTPUT_DETAILS)
    return output


def _predict_drowsiness_image(face_crop: np.ndarray) -> Dict[str, float]:
    """Predice probabilidades de somnolencia desde imagen de rostro."""

    # Asegura carga del modelo visual.
    _load_drowsiness_model_once()

    # Si no hay modelo, retorna distribución heurística base.
    if _DROWSINESS_ENGINE == "none" or _DROWSINESS_MODEL is False:
        return {"alert": 0.60, "yawning": 0.25, "microsleep": 0.15}

    if _DROWSINESS_ENGINE == "tflite":
        raw = _predict_drowsiness_image_tflite(face_crop)
    else:
        # Import local de TensorFlow para preprocesamiento.
        import tensorflow as tf  # type: ignore

        # Convierte recorte a tensor float32.
        tensor = tf.convert_to_tensor(face_crop, dtype=tf.float32)

        # Redimensiona al tamaño esperado del modelo.
        target_h, target_w = _DROWSINESS_INPUT_SIZE
        tensor = tf.image.resize(tensor, (target_h, target_w))

        # Preprocesado MobileNetV2 (mismo usado en entrenamiento).
        tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)

        # Crea batch de tamaño 1.
        batch = tf.expand_dims(tensor, axis=0)

        # Predice con modelo Keras.
        raw = _DROWSINESS_MODEL(batch, training=False).numpy().reshape(-1)

    # Si salida no parece distribución válida, normaliza con softmax.
    if np.any(raw < 0.0) or not np.isclose(np.sum(raw), 1.0, atol=1e-2):
        probs = _softmax(raw.astype(np.float64)).astype(np.float32)
    else:
        probs = raw.astype(np.float32)

    # Ajusta nombres de clases si no coincide cantidad esperada.
    if len(_DROWSINESS_CLASS_NAMES) != len(probs):
        class_names = [f"class_{index}" for index in range(len(probs))]
    else:
        class_names = _DROWSINESS_CLASS_NAMES

    # Retorna mapa clase -> probabilidad.
    return {name: float(prob) for name, prob in zip(class_names, probs)}


def _fuse_drowsiness_signals(payload: DrowsinessImageRequest, class_probabilities: Dict[str, float]) -> Tuple[str, float, str, float, float, Dict[str, float]]:
    """Fusiona probabilidad de red visual con señales EAR/estado de visión."""

    # Estado principal del modelo visual.
    model_state = max(class_probabilities, key=class_probabilities.get)
    model_confidence = float(class_probabilities[model_state])

    # Probabilidades principales con fallback seguro.
    p_alert = float(class_probabilities.get("alert", 0.0))
    p_yawning = float(class_probabilities.get("yawning", 0.0))
    p_microsleep = float(class_probabilities.get("microsleep", 0.0))

    # Umbral opcional para reducir falsos positivos en microsleep.
    microsleep_min_prob = float(os.getenv("MOSTACHO_DROWSINESS_MICROSLEEP_MIN_PROB", "0.0") or 0.0)
    if p_microsleep < microsleep_min_prob:
        p_microsleep = 0.0

    # Pesos de contribución del modelo (configurables por entorno).
    weight_alert = float(os.getenv("MOSTACHO_DROWSINESS_ALERT_WEIGHT", "0.05") or 0.05)
    weight_yawning = float(os.getenv("MOSTACHO_DROWSINESS_YAWNING_WEIGHT", "0.55") or 0.55)
    weight_microsleep = float(os.getenv("MOSTACHO_DROWSINESS_MICROSLEEP_WEIGHT", "0.90") or 0.90)

    # Componente de riesgo proveniente del modelo visual.
    component_model = (weight_alert * p_alert) + (weight_yawning * p_yawning) + (weight_microsleep * p_microsleep)

    # Componente por duración de ojos cerrados.
    closed_duration = float(payload.closed_duration or 0.0)
    component_duration = min(closed_duration / 3.0, 1.0) * 0.40

    # Componente por estado discreto de face_service.
    vision_state = str(payload.vision_state or "").upper().strip()
    if vision_state == "SOMNOLENT":
        component_vision_state = 0.35
    elif vision_state == "EYES_CLOSED":
        component_vision_state = 0.20
    else:
        component_vision_state = 0.0

    # Componente por EAR respecto a umbral dinámico.
    avg_ear = payload.avg_ear
    threshold = payload.threshold
    if avg_ear is not None and threshold is not None and avg_ear < threshold:
        component_ear = 0.10
    else:
        component_ear = 0.0

    # Riesgo total acotado a [0, 1].
    risk_score = float(
        np.clip(component_model + component_duration + component_vision_state + component_ear, 0.0, 1.0)
    )

    # Mapeo de riesgo a estado final.
    warn_threshold = float(os.getenv("MOSTACHO_DROWSINESS_WARN_THRESHOLD", "0.45") or 0.45)
    sleep_threshold = float(os.getenv("MOSTACHO_DROWSINESS_SLEEP_THRESHOLD", "0.75") or 0.75)

    if risk_score >= sleep_threshold:
        state = "SOMNOLENT"
        confidence = risk_score
    elif risk_score >= warn_threshold:
        state = "DROWSY_WARNING"
        confidence = risk_score
    else:
        state = "ALERT"
        confidence = float(1.0 - risk_score)

    # Desglose de contribuciones para inspección.
    components = {
        "model": float(component_model),
        "closed_duration": float(component_duration),
        "vision_state": float(component_vision_state),
        "ear_threshold": float(component_ear),
    }

    # Retorna resultados de fusión completos.
    return state, confidence, model_state, model_confidence, risk_score, components


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check del servicio TF."""

    # Respuesta estándar de disponibilidad.
    return HealthResponse(service="tf_service", status="ok", timestamp_utc=utc_now_iso())


@app.post("/predict/fusion", response_model=FusionResponse)
def predict_fusion(payload: FusionRequest) -> FusionResponse:
    """Predice estado multimodal tabular (visión/voz/biometría)."""

    # Fusiona features de entrada.
    merged_features = _flatten_features(payload)
    # Obtiene probabilidades por clase del modelo tabular.
    class_probabilities = _model_predict(merged_features)

    # Selecciona clase de mayor probabilidad.
    state = max(class_probabilities, key=class_probabilities.get)
    confidence = float(class_probabilities[state])

    # Retorna respuesta del endpoint multimodal.
    return FusionResponse(
        state=state,
        confidence=confidence,
        class_probabilities=class_probabilities,
        timestamp_utc=utc_now_iso(),
    )


@app.post("/predict/drowsiness_image", response_model=DrowsinessResponse)
def predict_drowsiness_image(payload: DrowsinessImageRequest) -> DrowsinessResponse:
    """Predice somnolencia visual (alert/yawning/microsleep) y fusiona con EAR."""

    # Decodifica imagen RGB de request.
    image = _decode_rgb_image(payload)
    # Recorta rostro principal si se proporcionó bbox.
    if payload.image_is_cropped:
        face_crop = image
    else:
        face_crop = _crop_face(image, payload.face_bbox)

    # Predice distribución de clases visuales con modelo entrenado.
    class_probabilities = _predict_drowsiness_image(face_crop)

    # Fusiona modelo visual con señales EAR/estado de visión.
    state, confidence, model_state, model_confidence, risk_score, components = _fuse_drowsiness_signals(
        payload, class_probabilities
    )

    # Retorna salida completa de somnolencia fusionada.
    engine = _DROWSINESS_ENGINE or "unknown"
    return DrowsinessResponse(
        engine=engine,
        state=state,
        confidence=float(confidence),
        model_state=model_state,
        model_confidence=float(model_confidence),
        class_probabilities=class_probabilities,
        risk_score=float(risk_score),
        components=components,
        timestamp_utc=utc_now_iso(),
    )
