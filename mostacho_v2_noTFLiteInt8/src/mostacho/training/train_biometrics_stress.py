"""Entrenamiento baseline para estrés biométrico con TensorFlow."""

from __future__ import annotations

# argparse para CLI.
import argparse
# json para guardar metadata de features.
import json
# Path para manejo de rutas.
from pathlib import Path
# typing para hints de listas.
from typing import List, Tuple

# numpy para matrices numéricas.
import numpy as np
# tensorflow para modelo MLP.
import tensorflow as tf

# extractor reusable de biometría.
from mostacho.features.biometrics import extract_biometrics_features
# configuración global.
from mostacho.settings import load_settings


def _resolve_default_biometrics_dir(db_root: Path) -> Path:
    """Resuelve ruta por defecto para biometría con prioridad en estructura nueva."""

    # Se prioriza árbol nuevo y luego rutas legacy.
    candidates = [
        db_root / "biometrics" / "stress" / "dataset",
        db_root / "stress" / "dataset",
    ]

    # Retorna el primer candidato existente.
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Si no existe ninguno, retorna primera opción para error coherente.
    return candidates[0]


def parse_args() -> argparse.Namespace:
    """Argumentos para entrenamiento biométrico."""

    # Parser con descripción breve.
    parser = argparse.ArgumentParser(description="Entrena clasificador biométrico de estrés")
    # Carpeta base del dataset de stress.
    parser.add_argument("--data-dir", type=Path, default=None, help="Base del dataset de stress")
    # Épocas de entrenamiento.
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas")
    # Batch size.
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    # Limite opcional de archivos para pruebas rápidas.
    parser.add_argument("--max-files", type=int, default=0, help="Limite opcional de archivos")
    # Retorno de argumentos parseados.
    return parser.parse_args()


def _label_from_path(path: Path) -> int:
    """Heurística simple de etiqueta binaria (0=no stress, 1=stress)."""

    # Se normaliza ruta completa a minúsculas.
    normalized = str(path).lower()
    # Regla simple: archivos bajo `wesad` y `stress` se marcan como estrés potencial.
    if "wesad" in normalized:
        return 1
    if "stress" in normalized and "no stress" not in normalized:
        return 1
    # Caso por defecto: no stress.
    return 0


def load_dataset(data_dir: Path, max_files: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga archivos tabulares y crea dataset numérico para MLP."""

    # Se buscan CSV y TXT recursivamente.
    files = sorted([*data_dir.rglob("*.csv"), *data_dir.rglob("*.txt")])
    # Se limita si usuario lo solicita.
    if max_files > 0:
        files = files[:max_files]

    # Contenedores de features y etiquetas.
    rows = []
    labels = []

    # Se itera sobre archivos detectados.
    for path in files:
        # Se extraen features resumidas por archivo.
        features = extract_biometrics_features(path)
        # Se ignora si no hubo datos útiles.
        if features.get("bio_available", 0.0) < 0.5:
            continue
        # Se agrega fila de features.
        rows.append(features)
        # Se agrega etiqueta heurística.
        labels.append(_label_from_path(path))

    # Validación para evitar entrenamiento vacío.
    if not rows:
        raise RuntimeError("No se pudieron extraer features biométricas.")

    # Se fija orden de columnas por primera fila.
    feature_order = sorted(rows[0].keys())
    # Se arma matriz X.
    x = np.array([[row.get(name, 0.0) for name in feature_order] for row in rows], dtype=np.float32)
    # Se arma vector y.
    y = np.array(labels, dtype=np.int32)

    # Se devuelve dataset tabular.
    return x, y, feature_order


def build_model(num_features: int) -> tf.keras.Model:
    """Construye clasificador binario de estrés."""

    # Entrada tabular.
    inputs = tf.keras.Input(shape=(num_features,))
    # Primera capa densa.
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    # Regularización dropout.
    x = tf.keras.layers.Dropout(0.2)(x)
    # Segunda capa densa.
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    # Salida sigmoide binaria.
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Se crea y compila modelo.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Entrena baseline biométrico y guarda artefactos."""

    # Se parsean argumentos.
    args = parse_args()
    # Se cargan settings.
    settings = load_settings()

    # Se define carpeta base biométrica.
    data_dir = args.data_dir or _resolve_default_biometrics_dir(settings.db_root)
    # Se valida ruta.
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe data_dir: {data_dir}")

    # Se construye dataset.
    x, y, feature_order = load_dataset(data_dir=data_dir, max_files=args.max_files)

    # Se construye modelo.
    model = build_model(num_features=x.shape[1])
    # Entrenamiento baseline.
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, verbose=1)

    # Se prepara salida de artefactos.
    output_dir = settings.artifacts_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardado del modelo.
    model_path = output_dir / "biometrics_stress.keras"
    model.save(model_path)

    # Guardado de metadata de features.
    metadata_path = output_dir / "biometrics_stress_metadata.json"
    metadata_path.write_text(json.dumps({"feature_order": feature_order}, indent=2, ensure_ascii=True), encoding="utf-8")

    # Mensajes para operador.
    print(f"Modelo biométrico guardado en: {model_path}")
    print(f"Metadata biométrica guardada en: {metadata_path}")


if __name__ == "__main__":
    # Entrada script directa.
    main()
