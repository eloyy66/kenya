"""Exporta modelo de somnolencia a TFLite INT8 con calibración representativa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Iterable, List

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """Argumentos CLI para exportar TFLite INT8."""

    parser = argparse.ArgumentParser(description="Exporta drowsiness_vision a TFLite INT8")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Ruta al modelo .keras (por ejemplo drowsiness_vision_best.keras)",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        help="Ruta al annotations_train.json",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        required=True,
        help="Ruta al annotations_val.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Raiz del dataset somnolent-db (default: carpeta del train.json)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Tamaño de entrada (lado) para el modelo",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Numero maximo de imagenes para calibracion",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Ruta de salida .tflite (default: artifacts/models/drowsiness_vision_int8.tflite)",
    )
    return parser.parse_args()


def _resolve_image_path(dataset_root: Path, json_key: str) -> Path | None:
    """Resuelve ruta de imagen desde la llave almacenada en anotaciones JSON."""

    cleaned = json_key.replace("\\", "/").strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]

    candidates: List[Path] = []
    candidates.append(dataset_root / cleaned)

    if cleaned.startswith("classification_frames/"):
        remainder = cleaned.split("/", 1)[1]
        candidates.append(dataset_root / remainder)
        candidates.append(dataset_root.parent.parent / "classification_frames" / remainder)
        candidates.append(dataset_root.parent / "classification_frames" / remainder)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_paths_from_json(dataset_root: Path, json_path: Path) -> List[Path]:
    """Carga rutas válidas de imágenes desde un JSON de anotaciones."""

    if not json_path.exists():
        raise FileNotFoundError(f"No existe JSON: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Formato JSON invalido en {json_path}")

    paths: List[Path] = []
    for json_key in payload.keys():
        image_path = _resolve_image_path(dataset_root, str(json_key))
        if image_path is not None:
            paths.append(image_path)

    return paths


def _sample_paths(paths: List[Path], max_samples: int) -> List[Path]:
    """Selecciona una muestra reproducible de rutas para calibración."""

    if max_samples <= 0 or len(paths) <= max_samples:
        return paths

    rng = np.random.default_rng(42)
    indices = rng.choice(len(paths), size=max_samples, replace=False)
    return [paths[idx] for idx in indices]


def _representative_dataset(paths: Iterable[Path], image_size: int):
    """Generador de dataset representativo para calibración INT8."""

    for path in paths:
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.image.convert_image_dtype(image, tf.float32)
        # El modelo de inferencia espera input preprocesado.
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
        image = tf.expand_dims(image, axis=0)
        yield [image]


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Encuentra backbone MobileNetV2 dentro del modelo entrenado."""

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    raise RuntimeError("No se encontro backbone MobileNetV2 en el modelo.")


def _build_inference_model(source_model: tf.keras.Model, image_size: int) -> tf.keras.Model:
    """Reconstruye un modelo de inferencia sin augmentation ni preprocess interno."""

    num_classes = int(source_model.output_shape[-1])

    # Backbone sin pesos; se cargan desde modelo fuente.
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=None,
    )
    backbone.trainable = False

    # Copia pesos del backbone entrenado.
    trained_backbone = _find_backbone(source_model)
    backbone.set_weights(trained_backbone.get_weights())

    # Construcción del grafo de inferencia (preprocess externo).
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    inference_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="drowsiness_inference")

    # Copia pesos de capas densas por unidades.
    source_dense = [layer for layer in source_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for layer in inference_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            match = next((src for src in source_dense if src.units == layer.units), None)
            if match is not None:
                layer.set_weights(match.get_weights())

    return inference_model


def main() -> None:
    """Entrada principal de exportación INT8."""

    args = parse_args()

    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No existe model_path: {model_path}")

    dataset_root = args.dataset_root or args.train_json.parent

    train_paths = _load_paths_from_json(dataset_root, args.train_json)
    val_paths = _load_paths_from_json(dataset_root, args.val_json)
    all_paths = train_paths + val_paths

    if not all_paths:
        raise RuntimeError("No se encontraron imagenes para calibracion.")

    sampled_paths = _sample_paths(all_paths, args.max_samples)

    output_path = args.output_path
    if output_path is None:
        output_path = model_path.parent / "drowsiness_vision_int8.tflite"

    model = tf.keras.models.load_model(model_path, compile=False)
    # Construye modelo solo-inferencia (sin augmentation).
    model = _build_inference_model(model, args.image_size)
    # Workaround para entornos donde TFLite no puede leer Keras directamente.
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        # Keras 3 requiere `export` para SavedModel.
        if hasattr(model, "export"):
            model.export(str(saved_model_dir))
        else:
            model.save(saved_model_dir, include_optimizer=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset(sampled_paths, args.image_size)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

    print(f"TFLite INT8 guardado en: {output_path}")
    print(f"Calibracion con {len(sampled_paths)} imagenes")


if __name__ == "__main__":
    main()
