"""Reporte consolidado de modelos .keras para mostacho_v2.2.

Evalua:
- drowsiness_vision_best.keras (multiclase)
- eye_state_model_best.keras (binario open/closed)
- dmd_distraction_best.keras y dmd_distraction_final.keras (binario)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _normalize_driver_state(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text in {"alert", "normal", "awake", "safe"}:
        return "alert"
    if text in {"yawn", "yawning"}:
        return "yawning"
    if text in {"microsleep", "sleep", "sleepy", "drowsy"}:
        return "microsleep"
    return text


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


def _plot_confusion(cm: np.ndarray, class_names: Sequence[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(6.2, 5.6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.045, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=20, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    max_value = float(cm.max()) if cm.size else 0.0
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
    plt.title("Mostacho v2.2 - Metrics Summary")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _best_threshold(rows: Sequence[Dict[str, float]], metric: str = "f1") -> Dict[str, float]:
    return max(rows, key=lambda row: float(row.get(metric, 0.0)))


def _write_markdown_summary(summary: Dict[str, Any], output_path: Path) -> None:
    lines: List[str] = [
        "# Mostacho v2.2 - Model Report",
        "",
        f"- Generated at (UTC): `{summary.get('generated_at_utc', '')}`",
        "",
    ]
    keys = (
        "drowsiness_best_keras",
        "eye_best_keras",
        "distraction_best_keras",
        "distraction_final_keras",
    )
    for key in keys:
        payload = summary.get(key)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {})
        lines.append(f"## {key}")
        lines.append(f"- Accuracy: `{metrics.get('accuracy', 0.0):.4f}`")
        lines.append(f"- Precision macro: `{metrics.get('precision_macro', 0.0):.4f}`")
        lines.append(f"- Recall macro: `{metrics.get('recall_macro', 0.0):.4f}`")
        lines.append(f"- F1 macro: `{metrics.get('f1_macro', 0.0):.4f}`")
        lines.append(f"- Samples: `{payload.get('num_samples', 0)}`")
        if "roc_auc" in payload:
            lines.append(f"- ROC AUC: `{payload['roc_auc']:.4f}`")
        if "threshold_sweep" in payload:
            best = _best_threshold(payload["threshold_sweep"], metric="f1")
            lines.append(
                "- Best threshold by F1: "
                f"`{best.get('threshold', 0.5):.2f}` "
                f"(F1 `{best.get('f1', 0.0):.4f}`, "
                f"P `{best.get('precision', 0.0):.4f}`, "
                f"R `{best.get('recall', 0.0):.4f}`)"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def evaluate_eye_best_keras(
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    class_names = ["Closed_Eyes", "Open_Eyes"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = getattr(model, "input_shape", None)
    image_size = int(input_shape[1]) if isinstance(input_shape, tuple) and len(input_shape) >= 3 else 96

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"No existe carpeta: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                image_paths.append(path)
                labels.append(class_to_idx[class_name])

    if not image_paths:
        raise RuntimeError(f"No hay imagenes de eye-state en {dataset_root}")

    if max_samples and max_samples > 0 and len(image_paths) > max_samples:
        rng = np.random.default_rng(42)
        selected_idx: List[int] = []
        per_class = max_samples // 2
        for class_idx in (0, 1):
            idxs = [idx for idx, value in enumerate(labels) if value == class_idx]
            take = min(len(idxs), per_class)
            if take > 0:
                selected_idx.extend(rng.choice(np.asarray(idxs), size=take, replace=False).tolist())
        selected_idx = sorted({int(item) for item in selected_idx})
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
        batch = tf.expand_dims(image, axis=0)
        output = model(batch, training=False).numpy().reshape(-1)
        if output.size == 1:
            value = float(output[0])
            open_prob = _sigmoid(value) if value < 0.0 or value > 1.0 else float(np.clip(value, 0.0, 1.0))
        else:
            probs = _softmax(output.astype(np.float64))
            open_prob = float(probs[min(1, len(probs) - 1)])
        score_open.append(open_prob)
        y_pred.append(1 if open_prob >= 0.5 else 0)
        if idx % 1000 == 0 or idx == total:
            print(f"  [eye_best] {idx}/{total}", flush=True)

    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    score_arr = np.asarray(score_open, dtype=np.float32)
    metrics = _multiclass_metrics(y_true, y_pred_arr, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)
    _plot_confusion(cm, class_names, output_dir / "eye_best_confusion_matrix.png", "Eye State Best (.keras)")
    threshold_rows = _binary_threshold_sweep(y_true=y_true, score=score_arr)
    _plot_binary_threshold_curve(
        threshold_rows,
        output_dir / "eye_best_threshold_curve.png",
        "Eye State Best (.keras) - Threshold Sweep",
    )
    roc_auc = _plot_roc(y_true, score_arr, output_dir / "eye_best_roc_curve.png", "Eye State Best (.keras) ROC")

    return {
        "model": str(model_path),
        "dataset_root": str(dataset_root),
        "num_samples": int(len(y_true)),
        "metrics": metrics,
        "roc_auc": float(roc_auc),
        "threshold_sweep": threshold_rows,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "eye_best_confusion_matrix.png"),
            "threshold_curve_png": str(output_dir / "eye_best_threshold_curve.png"),
            "roc_curve_png": str(output_dir / "eye_best_roc_curve.png"),
        },
    }


def evaluate_distraction_keras(
    model_path: Path,
    split_csv: Path,
    output_dir: Path,
    tag: str,
    max_samples: int = 0,
) -> Dict[str, Any]:
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = getattr(model, "input_shape", None)
    image_h = int(input_shape[1]) if isinstance(input_shape, tuple) and len(input_shape) >= 3 else 224
    image_w = int(input_shape[2]) if isinstance(input_shape, tuple) and len(input_shape) >= 3 else 224

    df = pd.read_csv(split_csv)
    if "split" in df.columns and (df["split"] == "test").any():
        eval_df = df[df["split"] == "test"].copy()
        split_used = "test"
    else:
        eval_df = df.copy()
        split_used = "all"

    eval_df = eval_df[eval_df["image_path"].apply(lambda p: Path(str(p)).exists())].copy()
    eval_df["label"] = eval_df["label"].astype(int)
    if eval_df.empty:
        raise RuntimeError(f"No hay muestras validas para distraction ({tag}).")

    if max_samples and max_samples > 0 and len(eval_df) > max_samples:
        eval_df = (
            eval_df.groupby("label", group_keys=False)
            .apply(lambda part: part.sample(n=min(len(part), max_samples // 2), random_state=42))
            .reset_index(drop=True)
        )

    y_true = eval_df["label"].to_numpy(dtype=np.int32)
    scores: List[float] = []
    paths = eval_df["image_path"].astype(str).tolist()

    total = len(paths)
    for idx, image_path in enumerate(paths, start=1):
        raw = tf.io.read_file(image_path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_h, image_w))
        batch = tf.expand_dims(image, axis=0)
        output = model(batch, training=False).numpy().reshape(-1)
        if output.size == 1:
            value = float(output[0])
            distracted_prob = _sigmoid(value) if value < 0.0 or value > 1.0 else float(np.clip(value, 0.0, 1.0))
        else:
            probs = _softmax(output.astype(np.float64))
            distracted_prob = float(probs[min(1, len(probs) - 1)])
        scores.append(distracted_prob)
        if idx % 1000 == 0 or idx == total:
            print(f"  [{tag}] {idx}/{total}", flush=True)

    score_arr = np.asarray(scores, dtype=np.float32)
    y_pred = (score_arr >= 0.5).astype(np.int32)
    class_names = ["normal", "distracted"]
    metrics = _multiclass_metrics(y_true, y_pred, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)
    _plot_confusion(cm, class_names, output_dir / f"{tag}_confusion_matrix.png", f"Distraction {tag} (.keras)")
    threshold_rows = _binary_threshold_sweep(y_true=y_true, score=score_arr)
    _plot_binary_threshold_curve(
        threshold_rows,
        output_dir / f"{tag}_threshold_curve.png",
        f"Distraction {tag} (.keras) - Threshold Sweep",
    )
    roc_auc = _plot_roc(y_true, score_arr, output_dir / f"{tag}_roc_curve.png", f"Distraction {tag} (.keras) ROC")

    return {
        "model": str(model_path),
        "split_csv": str(split_csv),
        "split_used": split_used,
        "num_samples": int(len(y_true)),
        "metrics": metrics,
        "roc_auc": float(roc_auc),
        "threshold_sweep": threshold_rows,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / f"{tag}_confusion_matrix.png"),
            "threshold_curve_png": str(output_dir / f"{tag}_threshold_curve.png"),
            "roc_curve_png": str(output_dir / f"{tag}_roc_curve.png"),
        },
    }


def evaluate_drowsiness_best_keras(
    model_path: Path,
    dataset_root: Path,
    split_json: Path,
    classes_path: Path,
    output_dir: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"split_json invalido: {split_json}")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        class_names = ["alert", "yawning", "microsleep"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = getattr(model, "input_shape", None)
    image_h = int(input_shape[1]) if isinstance(input_shape, tuple) and len(input_shape) >= 3 else 160
    image_w = int(input_shape[2]) if isinstance(input_shape, tuple) and len(input_shape) >= 3 else 160

    entries = list(payload.items())
    if max_samples and max_samples > 0 and len(entries) > max_samples:
        rng = np.random.default_rng(42)
        idxs = rng.choice(np.arange(len(entries)), size=int(max_samples), replace=False).tolist()
        entries = [entries[int(idx)] for idx in sorted(idxs)]

    y_true: List[int] = []
    y_pred: List[int] = []
    missing_paths = 0
    unknown_labels: Dict[str, int] = {}

    total = len(entries)
    for idx, (rel_path, meta) in enumerate(entries, start=1):
        if not isinstance(meta, dict):
            continue
        raw_state = str(meta.get("driver_state", ""))
        state = _normalize_driver_state(raw_state)
        if state not in class_to_idx:
            unknown_labels[state] = unknown_labels.get(state, 0) + 1
            continue

        resolved = _resolve_somnolent_frame(dataset_root, str(rel_path))
        if not resolved.exists():
            missing_paths += 1
            continue

        raw = tf.io.read_file(str(resolved))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_h, image_w))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        batch = tf.expand_dims(image, axis=0)
        output = model(batch, training=False).numpy().reshape(-1)
        probs = _softmax(output.astype(np.float64)) if (np.any(output < 0.0) or not np.isclose(np.sum(output), 1.0, atol=1e-2)) else output

        y_true.append(int(class_to_idx[state]))
        y_pred.append(int(np.argmax(probs)))
        if idx % 500 == 0 or idx == total:
            print(f"  [drowsiness_best] {idx}/{total}", flush=True)

    if not y_true:
        raise RuntimeError("No hubo muestras validas para evaluar drowsiness.")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    metrics = _multiclass_metrics(y_true_arr, y_pred_arr, class_names=class_names)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int32)
    _plot_confusion(cm, class_names, output_dir / "drowsiness_best_confusion_matrix.png", "Drowsiness Best (.keras)")

    return {
        "model": str(model_path),
        "dataset_root": str(dataset_root),
        "split_json": str(split_json),
        "num_samples": int(len(y_true_arr)),
        "missing_paths": int(missing_paths),
        "unknown_labels": unknown_labels,
        "metrics": metrics,
        "artifacts": {
            "confusion_matrix_png": str(output_dir / "drowsiness_best_confusion_matrix.png"),
        },
    }


@dataclass
class ReportPaths:
    project_root: Path
    output_dir: Path
    eye_model: Path
    drowsiness_model: Path
    drowsiness_classes: Path
    distraction_best: Path
    distraction_final: Path
    distraction_split_csv: Path
    eye_dataset_root: Path
    drowsiness_dataset_root: Path
    drowsiness_split_json: Path


def _resolve_paths(args: argparse.Namespace) -> ReportPaths:
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (project_root / "artifacts" / "reports" / "model_metrics_v22")
    return ReportPaths(
        project_root=project_root,
        output_dir=output_dir,
        eye_model=project_root / "artifacts" / "models" / "eye_state_model_best.keras",
        drowsiness_model=project_root / "artifacts" / "models" / "drowsiness_vision_best.keras",
        drowsiness_classes=project_root / "artifacts" / "models" / "drowsiness_vision_classes.txt",
        distraction_best=project_root / "artifacts" / "dmd_processed_dmd_r3" / "models" / "dmd_distraction_best.keras",
        distraction_final=project_root / "artifacts" / "dmd_processed_dmd_r3" / "models" / "dmd_distraction_final.keras",
        distraction_split_csv=args.distraction_split_csv.resolve(),
        eye_dataset_root=args.eye_dataset_root.resolve(),
        drowsiness_dataset_root=args.drowsiness_dataset_root.resolve(),
        drowsiness_split_json=args.drowsiness_split_json.resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de modelos .keras de mostacho_v2.2")
    parser.add_argument("--project-root", type=Path, default=Path("/Users/usuario/kenya/mostacho_v2.2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eye-dataset-root", type=Path, default=Path("/Users/usuario/kenya/db/artificialvision/train"))
    parser.add_argument("--drowsiness-dataset-root", type=Path, default=Path("/Users/usuario/kenya/db/artificialvision/somnolent-db"))
    parser.add_argument("--drowsiness-split-json", type=Path, default=Path("/Users/usuario/kenya/db/artificialvision/somnolent-db/annotations_holdout.json"))
    parser.add_argument(
        "--distraction-split-csv",
        type=Path,
        default=Path("/Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed_dmd_r3/splits/labels_with_split.csv"),
    )
    parser.add_argument("--only", type=str, default="all", choices=("all", "drowsiness", "eye", "distraction", "distraction_best", "distraction_final"))
    parser.add_argument("--max-eye-samples", type=int, default=0)
    parser.add_argument("--max-drowsiness-samples", type=int, default=0)
    parser.add_argument("--max-distraction-samples", type=int, default=0)
    args = parser.parse_args()

    paths = _resolve_paths(args)
    _ensure_dir(paths.output_dir)

    required = [
        paths.eye_model,
        paths.drowsiness_model,
        paths.drowsiness_classes,
        paths.distraction_split_csv,
        paths.eye_dataset_root,
        paths.drowsiness_dataset_root,
        paths.drowsiness_split_json,
    ]
    for item in required:
        if not item.exists():
            raise FileNotFoundError(f"No existe ruta requerida: {item}")

    summary: Dict[str, Any] = {
        "project_root": str(paths.project_root),
        "output_dir": str(paths.output_dir),
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if args.only in {"all", "drowsiness"}:
        print("[1/4] Evaluando drowsiness best (.keras)...", flush=True)
        summary["drowsiness_best_keras"] = evaluate_drowsiness_best_keras(
            model_path=paths.drowsiness_model,
            dataset_root=paths.drowsiness_dataset_root,
            split_json=paths.drowsiness_split_json,
            classes_path=paths.drowsiness_classes,
            output_dir=paths.output_dir,
            max_samples=int(args.max_drowsiness_samples),
        )

    if args.only in {"all", "eye"}:
        print("[2/4] Evaluando eye best (.keras)...", flush=True)
        summary["eye_best_keras"] = evaluate_eye_best_keras(
            model_path=paths.eye_model,
            dataset_root=paths.eye_dataset_root,
            output_dir=paths.output_dir,
            max_samples=int(args.max_eye_samples),
        )

    if args.only in {"all", "distraction", "distraction_best"}:
        if paths.distraction_best.exists():
            print("[3/4] Evaluando distraction best (.keras)...", flush=True)
            summary["distraction_best_keras"] = evaluate_distraction_keras(
                model_path=paths.distraction_best,
                split_csv=paths.distraction_split_csv,
                output_dir=paths.output_dir,
                tag="distraction_best",
                max_samples=int(args.max_distraction_samples),
            )
        else:
            summary["distraction_best_keras"] = {"error": f"No existe {paths.distraction_best}"}

    if args.only in {"all", "distraction", "distraction_final"}:
        if paths.distraction_final.exists():
            print("[4/4] Evaluando distraction final (.keras)...", flush=True)
            summary["distraction_final_keras"] = evaluate_distraction_keras(
                model_path=paths.distraction_final,
                split_csv=paths.distraction_split_csv,
                output_dir=paths.output_dir,
                tag="distraction_final",
                max_samples=int(args.max_distraction_samples),
            )
        else:
            summary["distraction_final_keras"] = {"error": f"No existe {paths.distraction_final}"}

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for key in (
        "drowsiness_best_keras",
        "eye_best_keras",
        "distraction_best_keras",
        "distraction_final_keras",
    ):
        value = summary.get(key)
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
            rows.append((key, value))
    if rows:
        _plot_summary_bars(rows, paths.output_dir / "metrics_summary_bar.png")

    summary_json = paths.output_dir / "summary_metrics.json"
    summary_md = paths.output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_markdown_summary(summary, summary_md)
    print(f"Reporte JSON: {summary_json}")
    print(f"Reporte MD: {summary_md}")
    print(f"Graficas en: {paths.output_dir}")


if __name__ == "__main__":
    main()

