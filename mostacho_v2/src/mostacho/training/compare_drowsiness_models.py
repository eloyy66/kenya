"""Compara salidas Keras vs TFLite para drowsiness_vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf


CLASS_NAMES = ["alert", "yawning", "microsleep"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara Keras vs TFLite en drowsiness_vision")
    parser.add_argument("--keras-model", type=Path, required=True, help="Ruta al .keras")
    parser.add_argument("--tflite-model", type=Path, required=True, help="Ruta al .tflite")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=200)
    return parser.parse_args()


def _resolve_image_path(dataset_root: Path, json_key: str) -> Path | None:
    cleaned = json_key.replace("\\", "/").strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]

    candidates: List[Path] = [dataset_root / cleaned]
    if cleaned.startswith("classification_frames/"):
        remainder = cleaned.split("/", 1)[1]
        candidates.append(dataset_root / remainder)
        candidates.append(dataset_root.parent.parent / "classification_frames" / remainder)
        candidates.append(dataset_root.parent / "classification_frames" / remainder)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_labeled_paths(dataset_root: Path, json_path: Path) -> List[Tuple[Path, int]]:
    if not json_path.exists():
        raise FileNotFoundError(f"No existe JSON: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Formato JSON invalido en {json_path}")

    rows: List[Tuple[Path, int]] = []
    for json_key, metadata in payload.items():
        if not isinstance(metadata, dict):
            continue
        driver_state = str(metadata.get("driver_state", "")).strip().lower()
        if driver_state not in CLASS_TO_INDEX:
            continue
        image_path = _resolve_image_path(dataset_root, str(json_key))
        if image_path is None:
            continue
        rows.append((image_path, CLASS_TO_INDEX[driver_state]))
    return rows


def _sample_rows(rows: List[Tuple[Path, int]], max_samples: int) -> List[Tuple[Path, int]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = np.random.default_rng(42)
    indices = rng.choice(len(rows), size=max_samples, replace=False)
    return [rows[idx] for idx in indices]


def _prepare_input(path: Path, image_size: int) -> np.ndarray:
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    raise RuntimeError("No se encontro backbone MobileNetV2 en el modelo.")


def _build_inference_model(source_model: tf.keras.Model, image_size: int) -> tf.keras.Model:
    num_classes = int(source_model.output_shape[-1])
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=None,
    )
    backbone.trainable = False
    backbone.set_weights(_find_backbone(source_model).get_weights())

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    inference_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    source_dense = [layer for layer in source_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for layer in inference_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            match = next((src for src in source_dense if src.units == layer.units), None)
            if match is not None:
                layer.set_weights(match.get_weights())

    return inference_model


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _tflite_predict(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    data = input_data
    if input_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            data = np.round(data / scale + zero_point)
        info = np.iinfo(input_details["dtype"])
        data = np.clip(data, info.min, info.max).astype(input_details["dtype"])
    else:
        data = data.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"]).reshape(-1)

    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            output = (output.astype(np.float32) - zero_point) * scale
        else:
            output = output.astype(np.float32)
    else:
        output = output.astype(np.float32)

    if np.any(output < 0.0) or not np.isclose(np.sum(output), 1.0, atol=1e-2):
        output = _softmax(output)
    return output


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root or args.train_json.parent
    rows = _load_labeled_paths(dataset_root, args.train_json) + _load_labeled_paths(dataset_root, args.val_json)
    rows = _sample_rows(rows, args.max_samples)
    if not rows:
        raise RuntimeError("No hay muestras para comparar.")

    keras_model = tf.keras.models.load_model(args.keras_model, compile=False)
    keras_model = _build_inference_model(keras_model, args.image_size)

    interpreter = tf.lite.Interpreter(model_path=str(args.tflite_model))
    interpreter.allocate_tensors()

    agree = 0
    keras_correct = 0
    tflite_correct = 0

    for path, label in rows:
        x = _prepare_input(path, args.image_size)
        keras_pred = keras_model(x, training=False).numpy().reshape(-1)
        if np.any(keras_pred < 0.0) or not np.isclose(np.sum(keras_pred), 1.0, atol=1e-2):
            keras_pred = _softmax(keras_pred)
        tflite_pred = _tflite_predict(interpreter, x)

        keras_idx = int(np.argmax(keras_pred))
        tflite_idx = int(np.argmax(tflite_pred))

        if keras_idx == tflite_idx:
            agree += 1
        if keras_idx == label:
            keras_correct += 1
        if tflite_idx == label:
            tflite_correct += 1

    total = len(rows)
    print(f"Muestras: {total}")
    print(f"Acuerdo Keras vs TFLite: {agree}/{total} ({agree/total:.2%})")
    print(f"Accuracy Keras: {keras_correct}/{total} ({keras_correct/total:.2%})")
    print(f"Accuracy TFLite: {tflite_correct}/{total} ({tflite_correct/total:.2%})")


if __name__ == "__main__":
    main()
