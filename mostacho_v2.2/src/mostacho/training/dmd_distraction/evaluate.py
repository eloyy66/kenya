from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
import seaborn as sns
import tensorflow as tf

from .config import PipelineConfig


AUTOTUNE = tf.data.AUTOTUNE


def _build_infer_dataset(df: pd.DataFrame, cfg: PipelineConfig) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label"].astype(np.float32).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (cfg.image_height, cfg.image_width))
        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32)
        return img, label

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE).batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds


def _predict_probs(model: tf.keras.Model, df: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    ds = _build_infer_dataset(df, cfg)
    probs = model.predict(ds, verbose=0).reshape(-1)
    return probs


def select_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    min_precision: float,
) -> dict[str, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        y_hat = (y_prob_val >= threshold).astype(np.int32)
        precision = precision_score(y_true_val, y_hat, zero_division=0)
        recall = recall_score(y_true_val, y_hat, zero_division=0)
        f1 = f1_score(y_true_val, y_hat, zero_division=0)
        beta = 0.5
        denom = (beta * beta * precision + recall)
        f_beta = ((1 + beta * beta) * precision * recall / denom) if denom > 0 else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f0_5": float(f_beta),
            }
        )

    valid = [c for c in candidates if c["precision"] >= min_precision]
    if valid:
        best = max(valid, key=lambda c: (c["recall"], c["f0_5"], -c["threshold"]))
        best["selection_reason"] = "max_recall_with_precision_constraint"
        return best
    best = max(candidates, key=lambda c: (c["f0_5"], c["precision"], c["recall"]))
    best["selection_reason"] = "max_f0_5_fallback"
    return best


def _save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["normal", "distracted"], yticklabels=["normal", "distracted"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar modelo DMD binario y ajustar umbral para reducir falsos positivos.")
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--min-precision", type=float, default=PipelineConfig().min_precision_for_threshold)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = PipelineConfig(
        processed_output_dir=args.processed_output_dir,
        min_precision_for_threshold=args.min_precision,
    )
    cfg.ensure_dirs()

    split_csv = args.split_csv or cfg.split_csv_path
    model_path = args.model_path or (cfg.models_dir / "dmd_distraction_best.keras")
    if not split_csv.exists():
        raise FileNotFoundError(f"No existe split CSV: {split_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo .keras: {model_path}")

    split_df = pd.read_csv(split_csv)
    required = {"image_path", "label", "split"}
    if not required.issubset(split_df.columns):
        raise RuntimeError(f"Split CSV requiere columnas: {sorted(required)}")

    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()
    if val_df.empty or test_df.empty:
        raise RuntimeError("Se requieren particiones val y test para evaluacion.")

    model = tf.keras.models.load_model(model_path)
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    val_probs = _predict_probs(model, val_df, cfg)
    test_probs = _predict_probs(model, test_df, cfg)

    if len(np.unique(y_val)) < 2:
        threshold = float(args.default_threshold)
        threshold_info = {
            "threshold": threshold,
            "selection_reason": "val_single_class_default",
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "f0_5": float("nan"),
        }
        print("[WARN] Val split tiene una sola clase; se usa default-threshold.")
    else:
        threshold_info = select_threshold(y_true_val=y_val, y_prob_val=val_probs, min_precision=cfg.min_precision_for_threshold)
        threshold = float(threshold_info["threshold"])

    test_pred = (test_probs >= threshold).astype(np.int32)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    report = classification_report(y_test, test_pred, labels=[0, 1], target_names=["normal", "distracted"], output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
    roc_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else float("nan")

    fp_mask = (test_pred == 1) & (y_test == 0)
    fn_mask = (test_pred == 0) & (y_test == 1)
    misclassified = test_df.loc[(test_pred != y_test)].copy()
    misclassified["predicted_label"] = test_pred[test_pred != y_test]
    misclassified["predicted_probability"] = test_probs[test_pred != y_test]
    misclassified["error_type"] = np.where(
        misclassified["label"].to_numpy() == 0,
        "false_positive",
        "false_negative",
    )

    metrics = {
        "model_path": str(model_path),
        "split_csv": str(split_csv),
        "threshold_selection": threshold_info,
        "test_metrics": {
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "accuracy": float((test_pred == y_test).mean()),
            "false_positives": int(fp_mask.sum()),
            "false_negatives": int(fn_mask.sum()),
            "n_test_samples": int(len(y_test)),
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = cfg.reports_dir / "metrics.json"
    misclassified_path = cfg.reports_dir / "misclassified_samples.csv"
    cm_path = cfg.reports_dir / "confusion_matrix.png"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    misclassified.to_csv(misclassified_path, index=False)
    _save_confusion_matrix(cm, cm_path)

    print(f"Selected threshold: {threshold:.3f} ({threshold_info['selection_reason']})")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Confusion matrix image: {cm_path}")
    print(f"Misclassified CSV: {misclassified_path}")


if __name__ == "__main__":
    main()
