"""Genera graficos y resumen del modelo drowsiness_vision (Keras)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


try:  # noqa: E402
    import seaborn as sns

    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False

from mostacho.settings import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera graficos y resumen del modelo drowsiness_vision.")
    parser.add_argument("--train-report", type=Path, default=None, help="Ruta al drowsiness_vision_report.json")
    parser.add_argument("--eval-report", type=Path, default=None, help="Ruta al drowsiness_vision_eval_<split>.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directorio de salida para graficos")
    parser.add_argument("--title", type=str, default="Drowsiness Vision - Reporte", help="Titulo base de graficos")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_history(history: Dict[str, List[float]], output_dir: Path, title: str) -> None:
    epochs = list(range(1, len(history.get("accuracy", [])) + 1))
    if not epochs:
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history.get("accuracy", []), label="train_acc")
    plt.plot(epochs, history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history.get("loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()


def _plot_confusion_matrix(confusion: np.ndarray, class_names: List[str], output_dir: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    if _HAS_SEABORN:
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(confusion, cmap="Blues")
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                plt.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def _plot_per_class_metrics(per_class: Dict[str, Dict[str, float]], output_dir: Path, title: str) -> None:
    class_names = list(per_class.keys())
    precision = [per_class[name]["precision"] for name in class_names]
    recall = [per_class[name]["recall"] for name in class_names]
    f1 = [per_class[name]["f1"] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, precision, width, label="precision")
    plt.bar(x, recall, width, label="recall")
    plt.bar(x + width, f1, width, label="f1")
    plt.xticks(x, class_names)
    plt.ylim(0.0, 1.0)
    plt.title(f"{title} - Per-class Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=160)
    plt.close()


def _interpretation(train_report: Dict, eval_report: Dict) -> Tuple[str, Dict[str, float]]:
    history = train_report.get("history", {})
    train_acc = float(history.get("accuracy", [0.0])[-1]) if history.get("accuracy") else 0.0
    val_acc = float(history.get("val_accuracy", [0.0])[-1]) if history.get("val_accuracy") else 0.0
    gap = max(0.0, train_acc - val_acc)
    gap_pct = gap * 100.0

    val_loss = history.get("val_loss", [])
    min_val_loss = float(min(val_loss)) if val_loss else 0.0
    last_val_loss = float(val_loss[-1]) if val_loss else 0.0
    loss_regret = (last_val_loss / min_val_loss) if min_val_loss > 0 else 1.0

    metrics = eval_report.get("metrics", {})
    per_class = metrics.get("per_class", {})
    microsleep_recall = float(per_class.get("microsleep", {}).get("recall", 0.0))

    lines: List[str] = []
    lines.append(f"- Gap train/val accuracy: {gap_pct:.2f}%")
    if gap_pct > 10.0:
        lines.append("- Overfitting: alto (gap > 10%)")
    else:
        lines.append("- Overfitting: bajo/moderado (gap <= 10%)")

    if loss_regret > 1.1:
        lines.append("- Val loss subio desde su minimo: posible sobreajuste.")
    else:
        lines.append("- Val loss estable respecto al minimo: entrenamiento estable.")

    if microsleep_recall >= 0.85:
        lines.append("- Recall de microsleep alto (>= 0.85).")
    elif microsleep_recall >= 0.70:
        lines.append("- Recall de microsleep moderado (0.70–0.85).")
    else:
        lines.append("- Recall de microsleep bajo (< 0.70).")

    return "\n".join(lines), {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap_pct": gap_pct,
        "microsleep_recall": microsleep_recall,
    }


def main() -> None:
    args = parse_args()
    settings = load_settings()

    train_report = args.train_report or (settings.artifacts_root / "models" / "drowsiness_vision_report.json")
    eval_report = args.eval_report or (settings.artifacts_root / "models" / "drowsiness_vision_eval_holdout.json")
    output_dir = args.output_dir or (settings.artifacts_root / "models" / "drowsiness_vision_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = _load_json(train_report)
    eval_data = _load_json(eval_report)

    history = train_data.get("history", {})
    metrics = eval_data.get("metrics", {})
    per_class = metrics.get("per_class", {})
    confusion = np.asarray(metrics.get("confusion_matrix", []), dtype=int)

    class_names = list(per_class.keys()) if per_class else ["alert", "yawning", "microsleep"]

    _plot_history(history, output_dir, args.title)
    if confusion.size:
        _plot_confusion_matrix(confusion, class_names, output_dir, args.title)
    if per_class:
        _plot_per_class_metrics(per_class, output_dir, args.title)

    interpretation, summary = _interpretation(train_data, eval_data)
    report_md = output_dir / "summary.md"
    report_md.write_text(
        "\n".join(
            [
                f"# {args.title}",
                "",
                "## Resumen",
                interpretation,
                "",
                "## Metrics (holdout)",
                f"- Accuracy: {metrics.get('accuracy', 0.0):.4f}",
                f"- F1 macro: {metrics.get('f1_macro', 0.0):.4f}",
                f"- Precision macro: {metrics.get('precision_macro', 0.0):.4f}",
                f"- Recall macro: {metrics.get('recall_macro', 0.0):.4f}",
                "",
                "## Overfitting",
                f"- Train acc (last): {summary['train_acc']:.4f}",
                f"- Val acc (last): {summary['val_acc']:.4f}",
                f"- Gap: {summary['gap_pct']:.2f}%",
                "",
                "## Archivos generados",
                "- accuracy_curve.png",
                "- loss_curve.png",
                "- confusion_matrix.png",
                "- per_class_metrics.png",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Graficos guardados en: {output_dir}")
    print(f"Resumen: {report_md}")


if __name__ == "__main__":
    main()
