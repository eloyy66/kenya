"""Genera reporte de metricas para modelos TFLite runtime (drowsiness/eye/distraction)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _apply_tflite_input_quantization(input_data: np.ndarray, input_details: Dict[str, Any]) -> np.ndarray:
    input_dtype = input_details.get("dtype", np.float32)
    if input_dtype in (np.uint8, np.int8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        quantized = input_data.astype(np.float32)
        if scale and scale > 0:
            quantized = np.round(quantized / scale + zero_point)
        info = np.iinfo(input_dtype)
        return np.clip(quantized, info.min, info.max).astype(input_dtype)
    return input_data.astype(input_dtype)


def _dequantize_tflite_output(raw: np.ndarray, output_details: Dict[str, Any]) -> np.ndarray:
    output_dtype = output_details.get("dtype", np.float32)
    if output_dtype in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            return (raw.astype(np.float32) - float(zero_point)) * float(scale)
    return raw.astype(np.float32)


def _plot_confusion(cm: np.ndarray, class_names: Sequence[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(6.2, 5.6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.045, pad=0.04)
    tick_idx = np.arange(len(class_names))
    plt.xticks(tick_idx, class_names, rotation=20, ha="right")
    plt.yticks(tick_idx, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    max_value = float(cm.max()) if cm.size > 0 else 0.0
    threshold = max_value / 2.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            color = "white" if value > threshold else "black"
            plt.text(col, row, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _binary_threshold_sweep(y_true: np.ndarray, score: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (score >= threshold).astype(np.int32)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "predicted_positive_rate": float(np.mean(y_pred)),
            }
        )
    return rows


def _plot_binary_threshold_curve(rows: Sequence[Dict[str, float]], output_path: Path, title: str) -> None:
    thresholds = [row["threshold"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["f1"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(thresholds, precision, label="precision")
    plt.plot(thresholds, recall, label="recall")
    plt.plot(thresholds, f1, label="f1")
    plt.plot(thresholds, accuracy, label="accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_roc(y_true: np.ndarray, score: np.ndarray, output_path: Path, title: str) -> float:
    auc = float(roc_auc_score(y_true, score))
    fpr, tpr, _ = roc_curve(y_true, score)
    plt.figure(figsize=(5.8, 5.0))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return auc


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, float | int]] = {}
    for idx, name in enumerate(class_names):
        per_class[str(name)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _resolve_somnolent_frame(dataset_root: Path, rel_path: str) -> Path:
    normalized = rel_path.replace("\\", "/").lstrip("./")
    direct = dataset_root / normalized
    if direct.exists():
        return direct

    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] == "classification_frames":
        candidate = dataset_root / "/".join(parts[1:])
        if candidate.exists():
            return candidate

    if len(parts) >= 2:
        fallback = dataset_root / parts[-2] / parts[-1]
        if fallback.exists():
            return fallback

    return direct


def _normalize_driver_state(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text in {"alert", "normal", "awake", "safe"}:
        return "alert"
    if text in {"yawn", "yawning"}:
        return "yawning"
    if text in {"microsleep", "sleep", "sleepy", "drowsy"}:
        return "microsleep"
    return text


def evaluate_eye_float32(
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    class_names = ["Closed_Eyes", "Open_Eyes"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details.get("shape")
    image_size = int(input_shape[1]) if input_shape is not None and len(input_shape) >= 3 else 96

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_name in class_names:
        class_dir = dataset_root / class_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                image_paths.append(path)
                labels.append(class_to_idx[class_name])

    if max_samples and max_samples > 0 and len(image_paths) > max_samples:
        rng = np.random.default_rng(42)
        selected_idx: List[int] = []
        per_class = max_samples // 2
        for class_idx in (0, 1):
            idxs = [idx for idx, value in enumerate(labels) if value == class_idx]
            if not idxs:
                continue
            take = min(len(idxs), per_class)
            chosen = rng.choice(np.asarray(idxs), size=take, replace=False).tolist()
            selected_idx.extend(int(item) for item in chosen)
        if len(selected_idx) < max_samples:
            remaining = [idx for idx in range(len(image_paths)) if idx not in set(selected_idx)]
            needed = min(max_samples - len(selected_idx), len(remaining))
            if needed > 0:
                chosen = rng.choice(np.asarray(remaining), size=needed, replace=False).tolist()
                selected_idx.extend(int(item) for item in chosen)
        selected_idx = sorted(selected_idx)
        image_paths = [image_paths[idx] for idx in selected_idx]
        labels = [labels[idx] for idx in selected_idx]

    y_true = np.asarray(labels, dtype=np.int32)
    score_open: List[float] = []
    y_pred: List[int] = []

    total = len(image_paths)
    for idx, path in enumerate(image_paths, start=1):
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=1, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
        input_data = _apply_tflite_input_quantization(batch, input_details)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"]).reshape(-1)
        output = _dequantize_tflite_output(raw_output, output_details).reshape(-1)
        open_prob = float(np.clip(output[0], 0.0, 1.0))
        score_open.append(open_prob)
        y_pred.append(1 if open_prob >= 0.5 else 0)
        if idx % 1000 == 0 or idx == total:
            print(f"  [eye] {idx}/{total}", flush=True)

    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    score_arr = np.asarray(score_open, dtype=np.float32)
    metrics = _multiclass_metrics(y_true, y_pred_arr, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)

    print("  [eye] plot confusion", flush=True)
    _plot_confusion(cm, class_names, output_dir / "eye_confusion_matrix.png", "Open/Closed Eyes (TFLite float32)")
    print("  [eye] threshold sweep", flush=True)
    threshold_rows = _binary_threshold_sweep(y_true=y_true, score=score_arr)
    print("  [eye] plot threshold curve", flush=True)
    _plot_binary_threshold_curve(threshold_rows, output_dir / "eye_threshold_curve.png", "Open/Closed Eyes Threshold Sweep")
    print("  [eye] plot roc", flush=True)
    eye_auc = _plot_roc(y_true, score_arr, output_dir / "eye_roc_curve.png", "Open/Closed Eyes ROC")
    print("  [eye] done", flush=True)

    return {
        "model": str(model_path),
        "dataset_root": str(dataset_root),
        "num_samples": int(len(y_true)),
        "metrics": metrics,
        "roc_auc_open_class": float(eye_auc),
        "threshold_sweep": threshold_rows,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "eye_confusion_matrix.png"),
            "threshold_curve_png": str(output_dir / "eye_threshold_curve.png"),
            "roc_curve_png": str(output_dir / "eye_roc_curve.png"),
        },
    }


def evaluate_distraction_float32(
    model_path: Path,
    split_csv: Path,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details.get("shape")
    image_h = int(input_shape[1]) if input_shape is not None and len(input_shape) >= 3 else 224
    image_w = int(input_shape[2]) if input_shape is not None and len(input_shape) >= 3 else 224

    df = pd.read_csv(split_csv)
    if "split" in df.columns and (df["split"] == "test").any():
        eval_df = df[df["split"] == "test"].copy()
        split_name = "test"
    else:
        eval_df = df.copy()
        split_name = "all"

    eval_df = eval_df[eval_df["image_path"].apply(lambda p: Path(str(p)).exists())].copy()
    eval_df["label"] = eval_df["label"].astype(int)
    if eval_df.empty:
        raise RuntimeError("No hay filas validas para evaluar distraccion.")

    if max_samples and max_samples > 0 and len(eval_df) > max_samples:
        rng = np.random.default_rng(42)
        selected_frames: List[pd.DataFrame] = []
        per_class = max_samples // 2
        for class_idx in (0, 1):
            part = eval_df[eval_df["label"] == class_idx]
            if part.empty:
                continue
            take = min(len(part), per_class)
            selected_frames.append(part.sample(n=take, random_state=42))
        reduced = pd.concat(selected_frames, axis=0)
        if len(reduced) < max_samples:
            missing = max_samples - len(reduced)
            remainder = eval_df.loc[~eval_df.index.isin(reduced.index)]
            if not remainder.empty:
                extra_take = min(missing, len(remainder))
                reduced = pd.concat([reduced, remainder.sample(n=extra_take, random_state=42)], axis=0)
        eval_df = reduced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    y_true = eval_df["label"].to_numpy(dtype=np.int32)
    score_distracted: List[float] = []

    paths = eval_df["image_path"].astype(str).tolist()
    total = len(paths)
    for idx, image_path in enumerate(paths, start=1):
        raw = tf.io.read_file(image_path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_h, image_w))
        batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
        input_data = _apply_tflite_input_quantization(batch, input_details)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"]).reshape(-1)
        output = _dequantize_tflite_output(raw_output, output_details).reshape(-1)
        distracted_prob = float(np.clip(output[0], 0.0, 1.0))
        score_distracted.append(distracted_prob)
        if idx % 500 == 0 or idx == total:
            print(f"  [distraction] {idx}/{total}", flush=True)

    score_arr = np.asarray(score_distracted, dtype=np.float32)
    y_pred = (score_arr >= 0.5).astype(np.int32)
    class_names = ["normal", "distracted"]
    metrics = _multiclass_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)

    print("  [distraction] plot confusion", flush=True)
    _plot_confusion(cm, class_names, output_dir / "distraction_confusion_matrix.png", "Distraction (TFLite float32)")
    print("  [distraction] threshold sweep", flush=True)
    threshold_rows = _binary_threshold_sweep(y_true=y_true, score=score_arr)
    print("  [distraction] plot threshold curve", flush=True)
    _plot_binary_threshold_curve(threshold_rows, output_dir / "distraction_threshold_curve.png", "Distraction Threshold Sweep")
    print("  [distraction] plot roc", flush=True)
    distraction_auc = _plot_roc(y_true, score_arr, output_dir / "distraction_roc_curve.png", "Distraction ROC")
    print("  [distraction] done", flush=True)

    return {
        "model": str(model_path),
        "split_csv": str(split_csv),
        "split_used": split_name,
        "num_samples": int(len(y_true)),
        "metrics": metrics,
        "roc_auc_distracted_class": float(distraction_auc),
        "threshold_sweep": threshold_rows,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "distraction_confusion_matrix.png"),
            "threshold_curve_png": str(output_dir / "distraction_threshold_curve.png"),
            "roc_curve_png": str(output_dir / "distraction_roc_curve.png"),
        },
    }


def evaluate_drowsiness_int8(
    model_path: Path,
    dataset_root: Path,
    split_json: Path,
    classes_path: Path,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    if not split_json.exists():
        raise FileNotFoundError(f"No existe split_json de drowsiness: {split_json}")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        class_names = ["alert", "yawning", "microsleep"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    payload = json.loads(split_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Formato invalido en split_json: {split_json}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details.get("shape")
    image_h = int(input_shape[1]) if input_shape is not None and len(input_shape) >= 3 else 160
    image_w = int(input_shape[2]) if input_shape is not None and len(input_shape) >= 3 else 160

    y_true: List[int] = []
    y_pred: List[int] = []
    missing_paths = 0
    unknown_labels: Dict[str, int] = {}

    entries = list(payload.items())
    if max_samples and max_samples > 0 and len(entries) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(np.arange(len(entries)), size=int(max_samples), replace=False)
        entries = [entries[int(idx)] for idx in sorted(indices.tolist())]
    total = len(entries)
    for idx, (rel_path, meta) in enumerate(entries, start=1):
        if not isinstance(meta, dict):
            continue
        raw_state = str(meta.get("driver_state", ""))
        normalized_state = _normalize_driver_state(raw_state)
        if normalized_state not in class_to_idx:
            unknown_labels[normalized_state] = unknown_labels.get(normalized_state, 0) + 1
            continue

        resolved_path = _resolve_somnolent_frame(dataset_root, str(rel_path))
        if not resolved_path.exists():
            missing_paths += 1
            continue

        raw = tf.io.read_file(str(resolved_path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_h, image_w))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
        input_data = _apply_tflite_input_quantization(batch, input_details)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"]).reshape(-1)
        output = _dequantize_tflite_output(raw_output, output_details).reshape(-1)

        if np.any(output < 0.0) or not np.isclose(np.sum(output), 1.0, atol=1e-2):
            probs = _softmax(output.astype(np.float64)).astype(np.float32)
        else:
            probs = output.astype(np.float32)

        y_true.append(int(class_to_idx[normalized_state]))
        y_pred.append(int(np.argmax(probs)))
        if idx % 500 == 0 or idx == total:
            print(f"  [drowsiness] {idx}/{total}", flush=True)

    if not y_true:
        raise RuntimeError("No se pudo construir ningun sample valido para evaluar drowsiness.")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    metrics = _multiclass_metrics(y_true_arr, y_pred_arr, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)
    print("  [drowsiness] plot confusion", flush=True)
    _plot_confusion(cm, class_names, output_dir / "drowsiness_confusion_matrix.png", "Drowsiness (TFLite int8)")
    print("  [drowsiness] done", flush=True)

    return {
        "model": str(model_path),
        "dataset_root": str(dataset_root),
        "split_json": str(split_json),
        "num_samples": int(len(y_true_arr)),
        "missing_paths": int(missing_paths),
        "unknown_labels": unknown_labels,
        "metrics": metrics,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "drowsiness_confusion_matrix.png"),
        },
    }


def _plot_summary_bars(model_rows: Sequence[Tuple[str, Dict[str, Any]]], output_path: Path) -> None:
    names = [name for name, _ in model_rows]
    accuracy = [float(data["metrics"]["accuracy"]) for _, data in model_rows]
    precision_macro = [float(data["metrics"]["precision_macro"]) for _, data in model_rows]
    recall_macro = [float(data["metrics"]["recall_macro"]) for _, data in model_rows]
    f1_macro = [float(data["metrics"]["f1_macro"]) for _, data in model_rows]

    x = np.arange(len(names))
    width = 0.19

    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, accuracy, width=width, label="accuracy")
    plt.bar(x - 0.5 * width, precision_macro, width=width, label="precision_macro")
    plt.bar(x + 0.5 * width, recall_macro, width=width, label="recall_macro")
    plt.bar(x + 1.5 * width, f1_macro, width=width, label="f1_macro")
    plt.ylim(0.0, 1.02)
    plt.xticks(x, names, rotation=10)
    plt.ylabel("Score")
    plt.title("TFLite Model Metrics Summary")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _best_threshold(rows: Sequence[Dict[str, float]], metric: str = "f1") -> Dict[str, float]:
    return max(rows, key=lambda row: float(row.get(metric, 0.0)))


def _write_markdown_summary(summary: Dict[str, Any], output_path: Path) -> None:
    timestamp = summary.get("generated_at_utc", "")
    lines: List[str] = []
    lines.append("# Mostacho TFLite 2.2 - Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{timestamp}`")
    lines.append("")

    for model_name in ("drowsiness_int8", "eye_float32", "distraction_float32"):
        model_data = summary.get(model_name, {})
        if not model_data:
            continue
        metrics = model_data.get("metrics", {})
        lines.append(f"## {model_name}")
        lines.append(f"- Accuracy: `{metrics.get('accuracy', 0.0):.4f}`")
        lines.append(f"- Precision macro: `{metrics.get('precision_macro', 0.0):.4f}`")
        lines.append(f"- Recall macro: `{metrics.get('recall_macro', 0.0):.4f}`")
        lines.append(f"- F1 macro: `{metrics.get('f1_macro', 0.0):.4f}`")
        lines.append(f"- Samples: `{model_data.get('num_samples', 0)}`")
        if "roc_auc_open_class" in model_data:
            lines.append(f"- ROC AUC (open): `{model_data['roc_auc_open_class']:.4f}`")
        if "roc_auc_distracted_class" in model_data:
            lines.append(f"- ROC AUC (distracted): `{model_data['roc_auc_distracted_class']:.4f}`")
        if "threshold_sweep" in model_data:
            best = _best_threshold(model_data["threshold_sweep"], metric="f1")
            lines.append(
                "- Best threshold by F1: "
                f"`{best.get('threshold', 0.5):.2f}` "
                f"(F1 `{best.get('f1', 0.0):.4f}`, P `{best.get('precision', 0.0):.4f}`, R `{best.get('recall', 0.0):.4f}`)"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de metricas para modelos TFLite de mostacho_tflite2.2")
    parser.add_argument("--project-root", type=Path, default=Path("/Users/usuario/kenya/mostacho_tflite2.2"))
    parser.add_argument("--eye-dataset-root", type=Path, default=Path("/Users/usuario/kenya/db/artificialvision/train"))
    parser.add_argument(
        "--distraction-split-csv",
        type=Path,
        default=Path("/Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed_dmd_r3/splits/labels_with_split.csv"),
    )
    parser.add_argument(
        "--drowsiness-dataset-root",
        type=Path,
        default=Path("/Users/usuario/kenya/db/artificialvision/somnolent-db"),
    )
    parser.add_argument(
        "--drowsiness-split-json",
        type=Path,
        default=Path("/Users/usuario/kenya/db/artificialvision/somnolent-db/annotations_holdout.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida para reportes. Default: <project-root>/artifacts/reports/tflite_metrics",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=("all", "eye", "distraction", "drowsiness"),
        help="Ejecuta solo una evaluacion para evitar corridas largas en un solo proceso.",
    )
    parser.add_argument("--max-eye-samples", type=int, default=0, help="Limita muestras de eye (0 = todas).")
    parser.add_argument("--max-distraction-samples", type=int, default=0, help="Limita muestras de distraction (0 = todas).")
    parser.add_argument("--max-drowsiness-samples", type=int, default=0, help="Limita muestras de drowsiness (0 = todas).")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    models_dir = project_root / "artifacts" / "models"
    output_dir = args.output_dir.resolve() if args.output_dir else (project_root / "artifacts" / "reports" / "tflite_metrics")
    _ensure_dir(output_dir)

    eye_model = models_dir / "eye_state_model_float32.tflite"
    distraction_model = models_dir / "distraction_model_float32.tflite"
    drowsiness_model = models_dir / "drowsiness_vision_int8.tflite"
    drowsiness_classes = models_dir / "drowsiness_vision_classes.txt"

    if not eye_model.exists():
        raise FileNotFoundError(f"No existe modelo eye float32: {eye_model}")
    if not distraction_model.exists():
        raise FileNotFoundError(f"No existe modelo distraction float32: {distraction_model}")
    if not drowsiness_model.exists():
        raise FileNotFoundError(f"No existe modelo drowsiness int8: {drowsiness_model}")
    if not drowsiness_classes.exists():
        raise FileNotFoundError(f"No existe archivo de clases drowsiness: {drowsiness_classes}")

    summary_json = output_dir / "summary_metrics.json"
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    else:
        summary = {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
        }
    summary["generated_at_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if args.only in {"all", "eye"}:
        print("[1/3] Evaluando Open/Closed (float32 TFLite)...")
        eye_result = evaluate_eye_float32(
            model_path=eye_model,
            dataset_root=args.eye_dataset_root,
            output_dir=output_dir,
            max_samples=int(args.max_eye_samples),
        )
        summary["eye_float32"] = eye_result

    if args.only in {"all", "distraction"}:
        print("[2/3] Evaluando Distraction (float32 TFLite)...")
        distraction_result = evaluate_distraction_float32(
            model_path=distraction_model,
            split_csv=args.distraction_split_csv,
            output_dir=output_dir,
            max_samples=int(args.max_distraction_samples),
        )
        summary["distraction_float32"] = distraction_result

    if args.only in {"all", "drowsiness"}:
        print("[3/3] Evaluando Drowsiness (int8 TFLite)...")
        drowsiness_result = evaluate_drowsiness_int8(
            model_path=drowsiness_model,
            dataset_root=args.drowsiness_dataset_root,
            split_json=args.drowsiness_split_json,
            classes_path=drowsiness_classes,
            output_dir=output_dir,
            max_samples=int(args.max_drowsiness_samples),
        )
        summary["drowsiness_int8"] = drowsiness_result

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for key in ("drowsiness_int8", "eye_float32", "distraction_float32"):
        if isinstance(summary.get(key), dict):
            rows.append((key, summary[key]))
    if rows:
        _plot_summary_bars(rows, output_dir / "metrics_summary_bar.png")

    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_markdown_summary(summary, summary_md)

    print(f"Reporte JSON: {summary_json}")
    print(f"Reporte MD: {summary_md}")
    print(f"Graficas en: {output_dir}")


if __name__ == "__main__":
    main()
