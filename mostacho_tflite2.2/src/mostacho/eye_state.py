"""Utilidades compartidas para el clasificador open/closed eyes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Closed_Eyes", "Open_Eyes"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_eye_state_samples(dataset_root: Path) -> List[Tuple[str, int]]:
    """Recolecta rutas etiquetadas desde la estructura Open_Eyes/Closed_Eyes."""

    samples: List[Tuple[str, int]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(image_path), CLASS_TO_INDEX[class_name]))

    if not samples:
        raise RuntimeError(f"No se encontraron imagenes en {dataset_root}")

    return samples


def split_eye_state_samples(
    samples: Sequence[Tuple[str, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    """Genera splits estratificados train/val/test."""

    if val_ratio <= 0.0 or test_ratio <= 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio y test_ratio deben ser positivos y sumar menos de 1.")

    image_paths = np.asarray([path for path, _ in samples], dtype=object)
    labels = np.asarray([label for _, label in samples], dtype=np.int32)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    val_fraction = val_ratio / (1.0 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_labels,
    )

    def _records(paths: np.ndarray, subset_labels: np.ndarray) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for path, label in zip(paths.tolist(), subset_labels.tolist()):
            rows.append(
                {
                    "path": str(path),
                    "label": int(label),
                    "class_name": CLASS_NAMES[int(label)],
                }
            )
        return rows

    return {
        "train": _records(train_paths, train_labels),
        "val": _records(val_paths, val_labels),
        "test": _records(test_paths, test_labels),
    }


def save_split_manifest(
    manifest: Dict[str, List[Dict[str, object]]],
    output_path: Path,
    dataset_root: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Guarda manifest JSON reutilizable para entrenamiento y comparacion."""

    payload = {
        "dataset_root": str(dataset_root),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "class_names": list(CLASS_NAMES),
        "splits": manifest,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_split_manifest(manifest_path: Path) -> Dict[str, object]:
    """Carga manifest JSON de splits."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe split manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError(f"Formato invalido de split manifest: {manifest_path}")
    return payload


def split_records_to_arrays(payload: Dict[str, object], split: str) -> Tuple[List[str], np.ndarray]:
    """Convierte un split del manifest a arreglos de rutas y etiquetas."""

    splits = payload.get("splits", {})
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"No existe split '{split}' en el manifest.")

    records = splits[split]
    if not isinstance(records, list):
        raise ValueError(f"Split '{split}' invalido en el manifest.")

    image_paths: List[str] = []
    labels: List[int] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        image_paths.append(str(record["path"]))
        labels.append(int(record["label"]))

    return image_paths, np.asarray(labels, dtype=np.int32)


def build_eye_state_dataset(
    image_paths: List[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Construye un tf.data.Dataset para Open_Eyes y Closed_Eyes."""

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    def _load_example(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        return image, label

    dataset = dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_eye_state_model(image_size: int, learning_rate: float) -> tf.keras.Model:
    """Construye un CNN ligero para open/closed eyes en escala de grises."""

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomTranslation(0.04, 0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="eye_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 1))
    x = augmentation(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eye_state_binary")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def class_weights_from_labels(labels: np.ndarray) -> Dict[int, float]:
    """Calcula class_weight para entrenamiento binario balanceado o desbalanceado."""

    counter = Counter(labels.tolist())
    total = float(len(labels))
    num_classes = float(len(CLASS_NAMES))
    return {
        class_index: total / (num_classes * float(counter[class_index]))
        for class_index in range(len(CLASS_NAMES))
        if counter[class_index] > 0
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str] | None = None) -> Dict[str, object]:
    """Calcula accuracy, precision, recall, f1 y matriz de confusion."""

    resolved_class_names = list(class_names or CLASS_NAMES)
    num_classes = len(resolved_class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        confusion[int(true), int(pred)] += 1

    per_class: Dict[str, Dict[str, float | int]] = {}
    for idx, name in enumerate(resolved_class_names):
        tp = float(confusion[idx, idx])
        fp = float(np.sum(confusion[:, idx]) - tp)
        fn = float(np.sum(confusion[idx, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(confusion[idx, :])),
        }

    accuracy = float(np.trace(confusion) / np.sum(confusion)) if np.sum(confusion) > 0 else 0.0
    precision_macro = float(np.mean([float(per_class[name]["precision"]) for name in resolved_class_names]))
    recall_macro = float(np.mean([float(per_class[name]["recall"]) for name in resolved_class_names]))
    f1_macro = float(np.mean([float(per_class[name]["f1"]) for name in resolved_class_names]))

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def prepare_eye_image_array(image: np.ndarray, image_size: int) -> np.ndarray:
    """Normaliza un recorte de ojo o rostro a batch grayscale NxHxWx1."""

    tensor = tf.convert_to_tensor(image)
    if tensor.shape.rank == 2:
        tensor = tf.expand_dims(tensor, axis=-1)
    elif tensor.shape.rank == 3 and tensor.shape[-1] == 3:
        tensor = tf.image.rgb_to_grayscale(tensor)
    elif tensor.shape.rank == 3 and tensor.shape[-1] != 1:
        raise ValueError("La imagen debe tener 1 o 3 canales.")

    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = tf.image.resize(tensor, (image_size, image_size))
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor.numpy()


def eye_state_probabilities_from_raw(raw: np.ndarray, class_names: Sequence[str] | None = None) -> Dict[str, float]:
    """Convierte salida cruda de Keras a probabilidades Closed_Eyes/Open_Eyes."""

    resolved_class_names = list(class_names or CLASS_NAMES)
    values = np.asarray(raw, dtype=np.float32).reshape(-1)

    if values.size == 1:
        open_prob = float(values[0])
        if open_prob < 0.0 or open_prob > 1.0:
            open_prob = float(1.0 / (1.0 + np.exp(-open_prob)))
        open_prob = float(np.clip(open_prob, 0.0, 1.0))
        closed_prob = 1.0 - open_prob
        return {
            resolved_class_names[0]: closed_prob,
            resolved_class_names[1]: open_prob,
        }

    if np.any(values < 0.0) or not np.isclose(np.sum(values), 1.0, atol=1e-2):
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        probs = exp_values / np.sum(exp_values)
    else:
        probs = values

    if len(resolved_class_names) != len(probs):
        resolved_class_names = [f"class_{index}" for index in range(len(probs))]

    return {name: float(prob) for name, prob in zip(resolved_class_names, probs)}


def extract_eye_regions_from_face(face_crop: np.ndarray) -> List[np.ndarray]:
    """Extrae dos ROIs heuristicas de ojos desde un recorte de rostro."""

    if face_crop.ndim < 2:
        return []

    height, width = face_crop.shape[:2]
    if height < 24 or width < 24:
        return []

    top = int(round(height * 0.18))
    bottom = int(round(height * 0.55))
    left_x1 = int(round(width * 0.10))
    left_x2 = int(round(width * 0.48))
    right_x1 = int(round(width * 0.52))
    right_x2 = int(round(width * 0.90))

    regions = [
        face_crop[top:bottom, left_x1:left_x2],
        face_crop[top:bottom, right_x1:right_x2],
    ]

    return [region for region in regions if region.size > 0]


def _normalize_bbox(bbox: Sequence[float], image_shape: Sequence[int]) -> Tuple[int, int, int, int] | None:
    """Normaliza una bbox al espacio valido de la imagen."""

    if len(bbox) != 4:
        return None

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return None

    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_image_by_bbox(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Recorta una bbox segura; si falla retorna arreglo vacio."""

    empty_crop = np.empty((0, 0, *image.shape[2:]), dtype=image.dtype) if image.ndim == 3 else np.empty((0, 0), dtype=image.dtype)
    normalized = _normalize_bbox(bbox, image.shape[:2])
    if normalized is None:
        return empty_crop

    x1, y1, x2, y2 = normalized
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else empty_crop


def heuristic_eye_boxes_from_face_bbox(face_bbox: Sequence[float], image_shape: Sequence[int]) -> List[List[int]]:
    """Genera dos cajas heuristicas de ojos dentro de una bbox facial."""

    normalized = _normalize_bbox(face_bbox, image_shape)
    if normalized is None:
        return []

    x1, y1, x2, y2 = normalized
    width = x2 - x1
    height = y2 - y1
    if width < 24 or height < 24:
        return []

    top = y1 + int(round(height * 0.18))
    bottom = y1 + int(round(height * 0.55))
    left_x1 = x1 + int(round(width * 0.10))
    left_x2 = x1 + int(round(width * 0.48))
    right_x1 = x1 + int(round(width * 0.52))
    right_x2 = x1 + int(round(width * 0.90))

    boxes = [
        [left_x1, top, left_x2, bottom],
        [right_x1, top, right_x2, bottom],
    ]
    valid_boxes: List[List[int]] = []
    for box in boxes:
        normalized_box = _normalize_bbox(box, image_shape)
        if normalized_box is not None:
            valid_boxes.append([int(value) for value in normalized_box])
    return valid_boxes


def _bbox_from_points(points: Sequence[Sequence[float]], image_shape: Sequence[int], padding: float) -> List[int] | None:
    """Convierte landmarks a bbox expandida y acotada a la imagen."""

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        return None

    x1 = float(np.min(array[:, 0]))
    y1 = float(np.min(array[:, 1]))
    x2 = float(np.max(array[:, 0]))
    y2 = float(np.max(array[:, 1]))

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * float(padding)
    pad_y = height * float(padding)

    bbox = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    normalized = _normalize_bbox(bbox, image_shape)
    if normalized is None:
        return None
    return [int(value) for value in normalized]


def _eye_bbox_from_points(points: Sequence[Sequence[float]], image_shape: Sequence[int], padding: float) -> List[int] | None:
    """Construye una bbox cuadrada de ojo con escala estable basada en el ancho ocular.

    Esto evita que el recorte se colapse verticalmente cuando el ojo se cierra.
    """

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        return None

    min_x = float(np.min(array[:, 0]))
    max_x = float(np.max(array[:, 0]))
    center_x = float(np.mean(array[:, 0]))
    center_y = float(np.mean(array[:, 1]))

    eye_width = max(1.0, max_x - min_x)
    crop_size = eye_width * (1.30 + (2.25 * float(padding)))

    # Ligero desplazamiento hacia abajo para incluir párpado superior e inferior.
    center_y += eye_width * 0.02

    bbox = [
        center_x - (crop_size / 2.0),
        center_y - (crop_size / 2.0),
        center_x + (crop_size / 2.0),
        center_y + (crop_size / 2.0),
    ]
    normalized = _normalize_bbox(bbox, image_shape)
    if normalized is None:
        return None
    return [int(value) for value in normalized]


def _extract_aligned_eye_crop(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    padding: float,
) -> np.ndarray:
    """Alinea el ojo horizontalmente antes de recortarlo para mayor robustez entre cámaras."""

    import cv2  # type: ignore

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 4 or array.shape[1] != 2 or not np.isfinite(array).all():
        return np.empty((0, 0, *image.shape[2:]), dtype=image.dtype) if image.ndim == 3 else np.empty((0, 0), dtype=image.dtype)

    outer = array[0]
    inner = array[3]
    center = np.mean(array, axis=0)
    angle_deg = float(np.degrees(np.arctan2(inner[1] - outer[1], inner[0] - outer[0])))
    rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation,
        (int(image.shape[1]), int(image.shape[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    ones = np.ones((array.shape[0], 1), dtype=np.float32)
    transformed = np.hstack([array, ones]) @ rotation.T
    transformed_center_x = float(np.mean(transformed[:, 0]))
    transformed_center_y = float(np.mean(transformed[:, 1]))
    transformed_width = max(1.0, float(np.max(transformed[:, 0]) - np.min(transformed[:, 0])))
    crop_size = transformed_width * (1.30 + (2.25 * float(padding)))
    transformed_center_y += transformed_width * 0.02

    x1 = int(round(transformed_center_x - (crop_size / 2.0)))
    y1 = int(round(transformed_center_y - (crop_size / 2.0)))
    x2 = int(round(transformed_center_x + (crop_size / 2.0)))
    y2 = int(round(transformed_center_y + (crop_size / 2.0)))

    normalized = _normalize_bbox([x1, y1, x2, y2], rotated.shape[:2])
    if normalized is None:
        return np.empty((0, 0, *image.shape[2:]), dtype=image.dtype) if image.ndim == 3 else np.empty((0, 0), dtype=image.dtype)

    rx1, ry1, rx2, ry2 = normalized
    crop = rotated[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return np.empty((0, 0, *image.shape[2:]), dtype=image.dtype) if image.ndim == 3 else np.empty((0, 0), dtype=image.dtype)
    return crop


def extract_eye_regions_from_landmarks(
    image: np.ndarray,
    left_eye: Sequence[Sequence[float]] | None,
    right_eye: Sequence[Sequence[float]] | None,
    padding: float = 0.25,
    min_size: int = 10,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Extrae ROIs de ojos usando landmarks precisos."""

    regions: List[np.ndarray] = []
    boxes: List[List[int]] = []
    for eye_points in (left_eye, right_eye):
        if eye_points is None or len(eye_points) == 0:
            continue
        bbox = _eye_bbox_from_points(eye_points, image.shape[:2], padding=padding)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < int(min_size) or (y2 - y1) < int(min_size):
            continue
        crop = _extract_aligned_eye_crop(image, eye_points, padding=padding)
        if crop.size == 0:
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        regions.append(crop)
        boxes.append(bbox)
    return regions, boxes


def extract_eye_regions_from_detection(
    image: np.ndarray,
    detection: Dict[str, Any] | Any,
    padding: float = 0.25,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Extrae ojos desde una detección facial serializada o un objeto similar."""

    if isinstance(detection, dict):
        left_eye = detection.get("left_eye") or []
        right_eye = detection.get("right_eye") or []
        face_bbox = detection.get("bbox") or []
    else:
        left_eye = getattr(detection, "left_eye", []) or []
        right_eye = getattr(detection, "right_eye", []) or []
        face_bbox = getattr(detection, "bbox", []) or []

    landmark_regions, landmark_boxes = extract_eye_regions_from_landmarks(image, left_eye, right_eye, padding=padding)
    if landmark_regions:
        return landmark_regions, landmark_boxes

    heuristic_boxes = heuristic_eye_boxes_from_face_bbox(face_bbox, image.shape[:2])
    heuristic_regions: List[np.ndarray] = []
    valid_boxes: List[List[int]] = []
    for bbox in heuristic_boxes:
        crop = crop_image_by_bbox(image, bbox)
        if crop.size == 0:
            continue
        heuristic_regions.append(crop)
        valid_boxes.append([int(value) for value in bbox])

    return heuristic_regions, valid_boxes
