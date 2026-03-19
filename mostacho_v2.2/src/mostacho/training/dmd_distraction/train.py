from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers

from .config import PipelineConfig


AUTOTUNE = tf.data.AUTOTUNE


def _build_backbone(backbone_name: str, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    name = backbone_name.strip().lower()
    if name in {"mobilenetv2", "mobilenet_v2"}:
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    if name in {"mobilenetv3small", "mobilenet_v3_small", "mobilenetv3"}:
        return tf.keras.applications.MobileNetV3Small(
            include_top=False,
            include_preprocessing=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    raise ValueError(f"Backbone no soportado: {backbone_name}")


def _build_model(cfg: PipelineConfig) -> tuple[tf.keras.Model, tf.keras.Model]:
    input_shape = (cfg.image_height, cfg.image_width, 3)
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name="rgb_face")
    # Equivalente al preprocess_input de MobileNetV2/V3: escalar a [-1, 1].
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    backbone = _build_backbone(cfg.backbone_name, input_shape=input_shape)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)
    x = layers.Dense(cfg.hidden_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(cfg.dropout_rate * 0.5, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="distracted_prob")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dmd_distraction_binary")
    return model, backbone


def _decode_resize(image_path: tf.Tensor, label: tf.Tensor, cfg: PipelineConfig) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, size=(cfg.image_height, cfg.image_width), method="bilinear")
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def _random_cutout(images: tf.Tensor, p: float = 0.25, size_frac: float = 0.18) -> tf.Tensor:
    if tf.random.uniform([]) > p:
        return images
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    cut_h = tf.cast(tf.cast(h, tf.float32) * size_frac, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * size_frac, tf.int32)
    cut_h = tf.maximum(cut_h, 1)
    cut_w = tf.maximum(cut_w, 1)
    y0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, h - cut_h), dtype=tf.int32)
    x0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, w - cut_w), dtype=tf.int32)
    mask = tf.ones_like(images)
    zeros = tf.zeros((tf.shape(images)[0], cut_h, cut_w, tf.shape(images)[-1]), dtype=images.dtype)
    paddings = tf.stack(
        [
            [0, 0],
            [y0, h - y0 - cut_h],
            [x0, w - x0 - cut_w],
            [0, 0],
        ]
    )
    cutout_mask = tf.pad(zeros, paddings, constant_values=1.0)
    mask = tf.minimum(mask, cutout_mask)
    return images * mask


def _build_train_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomBrightness(factor=0.15),
            layers.RandomContrast(factor=0.15),
            layers.RandomRotation(factor=0.04),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        ],
        name="train_augmentation",
    )


def _build_dataset(
    frame_df: pd.DataFrame,
    cfg: PipelineConfig,
    training: bool,
    augmenter: tf.keras.Sequential | None,
) -> tf.data.Dataset:
    image_paths = frame_df["image_path"].astype(str).to_numpy()
    labels = frame_df["label"].astype(np.float32).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(frame_df), 10_000), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _decode_resize(p, y, cfg), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (_random_cutout(x), y), num_parallel_calls=AUTOTUNE)
    if cfg.cache_dataset:
        ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )


def _merge_histories(first: tf.keras.callbacks.History, second: tf.keras.callbacks.History | None) -> dict[str, list[float]]:
    history = {k: list(v) for k, v in first.history.items()}
    if second is None:
        return history
    for key, values in second.history.items():
        history.setdefault(key, [])
        history[key].extend(list(values))
    return history


def _plot_history(history: dict[str, list[float]], output_path: Path, keys: tuple[str, str], title: str) -> None:
    plt.figure(figsize=(8, 4))
    train_key, val_key = keys
    plt.plot(history.get(train_key, []), label=train_key)
    plt.plot(history.get(val_key, []), label=val_key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(train_key.replace("val_", ""))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _load_or_compute_class_weights(split_df: pd.DataFrame, split_report_path: Path) -> dict[int, float]:
    if split_report_path.exists():
        with split_report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "class_weights" in data:
            return {int(k): float(v) for k, v in data["class_weights"].items()}
    train_y = split_df.loc[split_df["split"] == "train", "label"].astype(int).to_numpy()
    classes = np.array([0, 1], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar clasificador binario de distraccion (DMD RGB face).")
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--backbone-name", type=str, default=PipelineConfig().backbone_name)
    parser.add_argument("--batch-size", type=int, default=PipelineConfig().batch_size)
    parser.add_argument("--freeze-epochs", type=int, default=PipelineConfig().freeze_epochs)
    parser.add_argument("--fine-tune-epochs", type=int, default=PipelineConfig().fine_tune_epochs)
    parser.add_argument("--learning-rate", type=float, default=PipelineConfig().learning_rate)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=PipelineConfig().fine_tune_learning_rate)
    parser.add_argument("--fine-tune-layers", type=int, default=PipelineConfig().fine_tune_layers)
    parser.add_argument("--seed", type=int, default=PipelineConfig().seed)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    cfg = PipelineConfig(
        processed_output_dir=args.processed_output_dir,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        fine_tune_layers=args.fine_tune_layers,
        seed=args.seed,
        cache_dataset=args.cache_dataset,
    )
    cfg.ensure_dirs()

    split_csv = args.split_csv or cfg.split_csv_path
    if not split_csv.exists():
        raise FileNotFoundError(f"No existe split CSV: {split_csv}")
    split_df = pd.read_csv(split_csv)
    if split_df.empty:
        raise RuntimeError("Split CSV vacio.")
    needed = {"image_path", "label", "split"}
    if not needed.issubset(split_df.columns):
        raise RuntimeError(f"Split CSV requiere columnas: {sorted(needed)}")

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Split train/val/test incompleto.")

    augmenter = _build_train_augmenter()
    train_ds = _build_dataset(train_df, cfg=cfg, training=True, augmenter=augmenter)
    val_ds = _build_dataset(val_df, cfg=cfg, training=False, augmenter=None)
    test_ds = _build_dataset(test_df, cfg=cfg, training=False, augmenter=None)

    model, backbone = _build_model(cfg)
    _compile_model(model, learning_rate=cfg.learning_rate)

    best_model_path = cfg.models_dir / "dmd_distraction_best.keras"
    final_model_path = cfg.models_dir / "dmd_distraction_final.keras"
    history_json_path = cfg.models_dir / "dmd_distraction_history.json"
    history_csv_path = cfg.models_dir / "dmd_distraction_training_log.csv"

    val_unique = int(val_df["label"].nunique())
    if val_unique < 2:
        monitor_metric = "val_loss"
        monitor_mode = "min"
        print("[WARN] Val split tiene una sola clase; se usa val_loss para callbacks.")
    else:
        monitor_metric = "val_auc"
        monitor_mode = "max"
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=5,
        restore_best_weights=True,
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=2,
        factor=0.5,
        min_lr=1e-6,
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor=monitor_metric,
        mode=monitor_mode,
        save_best_only=True,
    )
    freeze_callbacks = [
        early_stopping_cb,
        reduce_lr_cb,
        checkpoint_cb,
        tf.keras.callbacks.CSVLogger(filename=str(history_csv_path), append=False),
    ]

    class_weight = None
    if not args.no_class_weights:
        class_weight = _load_or_compute_class_weights(split_df, cfg.split_report_path)

    history_freeze = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.freeze_epochs,
        class_weight=class_weight,
        callbacks=freeze_callbacks,
        verbose=1,
    )

    history_fine: tf.keras.callbacks.History | None = None
    if cfg.fine_tune_epochs > 0:
        backbone.trainable = True
        if cfg.fine_tune_layers > 0:
            freeze_until = max(0, len(backbone.layers) - cfg.fine_tune_layers)
            for layer in backbone.layers[:freeze_until]:
                layer.trainable = False
        _compile_model(model, learning_rate=cfg.fine_tune_learning_rate)
        fine_callbacks = [
            early_stopping_cb,
            reduce_lr_cb,
            checkpoint_cb,
            tf.keras.callbacks.CSVLogger(filename=str(history_csv_path), append=True),
        ]
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=cfg.freeze_epochs,
            epochs=cfg.freeze_epochs + cfg.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=fine_callbacks,
            verbose=1,
        )

    merged_history = _merge_histories(history_freeze, history_fine)
    with history_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    model.save(final_model_path)

    eval_metrics = model.evaluate(test_ds, verbose=0)
    eval_names = model.metrics_names
    eval_map = {name: float(value) for name, value in zip(eval_names, eval_metrics)}

    summary = {
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "history_json": str(history_json_path),
        "history_csv": str(history_csv_path),
        "class_weight": class_weight,
        "test_metrics_last_model": eval_map,
        "backbone_name": cfg.backbone_name,
    }
    with (cfg.models_dir / "dmd_distraction_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_history(
        merged_history,
        output_path=cfg.reports_dir / "train_accuracy.png",
        keys=("binary_accuracy", "val_binary_accuracy"),
        title="DMD Distraction - Binary Accuracy",
    )
    _plot_history(
        merged_history,
        output_path=cfg.reports_dir / "train_loss.png",
        keys=("loss", "val_loss"),
        title="DMD Distraction - Loss",
    )
    _plot_history(
        merged_history,
        output_path=cfg.reports_dir / "train_auc.png",
        keys=("auc", "val_auc"),
        title="DMD Distraction - AUC",
    )

    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"History JSON: {history_json_path}")
    print(f"Train summary: {cfg.models_dir / 'dmd_distraction_train_summary.json'}")
    print(f"Test metrics (last model): {eval_map}")


if __name__ == "__main__":
    main()
