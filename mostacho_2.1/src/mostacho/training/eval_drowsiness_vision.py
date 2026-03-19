"""Evalua drowsiness_vision y genera matriz de confusion y F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from mostacho.settings import load_settings


DEFAULT_SPLITS = {
    "val": "annotations_val.json",
    "test": "annotations_test.json",
    "holdout": "annotations_holdout.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua drowsiness_vision con metricas por clase")
    parser.add_argument("--model-path", type=Path, default=None, help="Ruta al modelo .keras")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Raiz del somnolent-db")
    parser.add_argument("--split", type=str, choices=sorted(DEFAULT_SPLITS.keys()), default="holdout")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
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


def _load_split(dataset_root: Path, split_json_name: str, class_names: List[str]) -> Tuple[List[str], np.ndarray]:
    split_path = dataset_root / split_json_name
    if not split_path.exists():
        raise FileNotFoundError(f"No existe split JSON: {split_path}")

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Formato JSON invalido en {split_path}")

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    image_paths: List[str] = []
    labels: List[int] = []

    for json_key, metadata in payload.items():
        if not isinstance(metadata, dict):
            continue
        driver_state = str(metadata.get("driver_state", "")).strip().lower()
        if driver_state not in class_to_index:
            continue
        image_path = _resolve_image_path(dataset_root, str(json_key))
        if image_path is None:
            continue
        image_paths.append(str(image_path))
        labels.append(class_to_index[driver_state])

    return image_paths, np.asarray(labels, dtype=np.int32)


def _build_dataset(image_paths: List[str], labels: np.ndarray, image_size: int, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load_example(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        return image, label

    dataset = dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, object]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[int(true), int(pred)] += 1

    per_class = {}
    for idx, name in enumerate(class_names):
        tp = float(confusion[idx, idx])
        fp = float(np.sum(confusion[:, idx]) - tp)
        fn = float(np.sum(confusion[idx, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(confusion[idx, :])),
        }

    accuracy = float(np.trace(confusion) / np.sum(confusion)) if np.sum(confusion) > 0 else 0.0
    f1_macro = float(np.mean([per_class[name]["f1"] for name in class_names]))
    precision_macro = float(np.mean([per_class[name]["precision"] for name in class_names]))
    recall_macro = float(np.mean([per_class[name]["recall"] for name in class_names]))

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    settings = load_settings()

    model_path = args.model_path or (settings.artifacts_root / "models" / "drowsiness_vision_best.keras")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe model_path: {model_path}")

    dataset_root = args.dataset_root or (settings.db_root / "artificialvision" / "somnolent-db")
    split_json = DEFAULT_SPLITS[args.split]

    # Carga clases desde archivo si existe.
    classes_path = settings.artifacts_root / "models" / "drowsiness_vision_classes.txt"
    if classes_path.exists():
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        class_names = ["alert", "yawning", "microsleep"]

    image_paths, labels = _load_split(dataset_root, split_json, class_names)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]
        labels = labels[: args.max_samples]

    dataset = _build_dataset(image_paths, labels, args.image_size, args.batch_size)
    model = tf.keras.models.load_model(model_path, compile=False)

    preds = model.predict(dataset, verbose=0)
    y_pred = np.argmax(preds, axis=1).astype(np.int32)

    metrics = _classification_metrics(labels, y_pred, class_names)

    report = {
        "split": args.split,
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "image_size": args.image_size,
        "num_samples": int(len(labels)),
        "metrics": metrics,
    }

    output_path = settings.artifacts_root / "models" / f"drowsiness_vision_eval_{args.split}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Reporte guardado en: {output_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print("Recall por clase:")
    for name in class_names:
        recall = metrics["per_class"][name]["recall"]
        print(f"  {name}: {recall:.4f}")


if __name__ == "__main__":
    main()
