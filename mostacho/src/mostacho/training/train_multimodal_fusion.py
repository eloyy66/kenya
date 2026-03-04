"""Entrena modelo de fusión multimodal para el servicio TensorFlow."""

from __future__ import annotations

# argparse para parámetros por consola.
import argparse
# json para guardar orden de features.
import json
# Path para rutas.
from pathlib import Path

# numpy para manipulación de arrays.
import numpy as np
# tensorflow para red de fusión.
import tensorflow as tf

# settings globales del proyecto.
from mostacho.settings import load_settings


# Clases objetivo del sistema multimodal.
CLASSES = ["alert", "somnolent", "stressed", "distracted"]


def parse_args() -> argparse.Namespace:
    """Argumentos del script de entrenamiento de fusión."""

    # Parser CLI.
    parser = argparse.ArgumentParser(description="Entrena modelo de fusión multimodal")
    # Ruta opcional a CSV de entrenamiento supervisado.
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV con columnas de features y columna `label`",
    )
    # Épocas de entrenamiento.
    parser.add_argument("--epochs", type=int, default=25, help="Numero de epocas")
    # Batch size.
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    # Retorno de argumentos.
    return parser.parse_args()


def _synthetic_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Genera dataset sintético de respaldo para dejar pipeline funcional."""

    # Orden estable de features mínimas esperadas por el orquestador.
    feature_order = [
        "vision_face_count",
        "vision_avg_score",
        "vision_primary_face_area_ratio",
        "voice_rms_mean",
        "voice_rms_std",
        "voice_zcr_mean",
        "voice_spectral_centroid_mean",
        "voice_duration_sec",
        "bio_available",
        "bio_num_rows",
    ]

    # Semilla fija para reproducibilidad.
    rng = np.random.default_rng(42)
    # Número de muestras sintéticas.
    n = 1200

    # Matriz base aleatoria controlada.
    x = rng.normal(size=(n, len(feature_order))).astype(np.float32)
    # Se fuerzan rangos más realistas para algunas features.
    x[:, 0] = np.clip(rng.normal(loc=0.9, scale=0.4, size=n), 0.0, 3.0)  # face_count
    x[:, 1] = np.clip(rng.normal(loc=0.7, scale=0.2, size=n), 0.0, 1.0)  # avg_score
    x[:, 2] = np.clip(rng.normal(loc=0.15, scale=0.06, size=n), 0.0, 1.0)  # face_area_ratio
    x[:, 3] = np.clip(rng.normal(loc=0.05, scale=0.03, size=n), 0.0, 1.0)  # rms_mean
    x[:, 8] = np.clip(rng.normal(loc=0.8, scale=0.3, size=n), 0.0, 1.0)  # bio_available
    x[:, 9] = np.clip(rng.normal(loc=700.0, scale=300.0, size=n), 0.0, 5000.0)  # bio_num_rows

    # Etiqueta sintética con reglas para clases.
    y = np.zeros((n,), dtype=np.int32)
    y[(x[:, 0] < 0.4) | (x[:, 2] < 0.08)] = 3  # distracted
    y[x[:, 3] < 0.02] = 1  # somnolent
    y[(x[:, 8] > 0.7) & (x[:, 9] > 1200)] = 2  # stressed

    # Se devuelve dataset sintético.
    return x, y, feature_order


def _load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga dataset real desde CSV con columna `label`."""

    # Import local para minimizar dependencia cuando no se usa CSV real.
    import pandas as pd

    # Se lee CSV completo.
    frame = pd.read_csv(csv_path)
    # Validación mínima de columna objetivo.
    if "label" not in frame.columns:
        raise ValueError("El CSV debe incluir columna 'label'.")

    # Se extraen nombres de features excluyendo label.
    feature_order = [col for col in frame.columns if col != "label"]
    # Matriz numérica X.
    x = frame[feature_order].to_numpy(dtype=np.float32)

    # Mapeo de etiquetas textuales a índice.
    label_to_index = {name: idx for idx, name in enumerate(CLASSES)}
    # Vector y codificado.
    y = frame["label"].map(label_to_index).fillna(0).to_numpy(dtype=np.int32)

    # Retorno de dataset real.
    return x, y, feature_order


def build_model(num_features: int) -> tf.keras.Model:
    """Construye red densa para clasificación multimodal."""

    # Entrada tabular de features fusionadas.
    inputs = tf.keras.Input(shape=(num_features,))
    # Bloque denso principal.
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    # Dropout para regularización.
    x = tf.keras.layers.Dropout(0.25)(x)
    # Segundo bloque denso.
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # Salida softmax con 4 clases.
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    # Modelo compilado.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Entrena modelo de fusión y persiste artefactos para tf_service."""

    # Se parsean argumentos CLI.
    args = parse_args()
    # Se cargan settings globales.
    settings = load_settings()

    # Si se pasa CSV real, se usa; de lo contrario se genera dataset sintético.
    if args.train_csv is not None:
        x, y, feature_order = _load_csv_dataset(args.train_csv)
    else:
        x, y, feature_order = _synthetic_dataset()

    # Construcción de la red de fusión.
    model = build_model(num_features=x.shape[1])
    # Entrenamiento baseline.
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, verbose=1)

    # Se asegura carpeta de artefactos.
    output_dir = settings.artifacts_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardado del modelo para tf_service.
    model_path = output_dir / "multimodal_fusion.keras"
    model.save(model_path)

    # Guardado del orden de features consumido por tf_service.
    feature_order_path = output_dir / "multimodal_feature_order.json"
    feature_order_path.write_text(json.dumps(feature_order, indent=2, ensure_ascii=True), encoding="utf-8")

    # Mensajes de confirmación.
    print(f"Modelo de fusion guardado en: {model_path}")
    print(f"Orden de features guardado en: {feature_order_path}")


if __name__ == "__main__":
    # Entrada directa por CLI.
    main()
