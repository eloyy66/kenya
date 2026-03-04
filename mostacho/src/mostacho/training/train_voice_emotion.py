"""Entrenamiento base de emoción en voz usando CREMA-D + TensorFlow."""

from __future__ import annotations

# argparse para CLI.
import argparse
# json para metadata de clases.
import json
# Path para rutas portables.
from pathlib import Path
# typing para anotaciones.
from typing import List, Tuple

# numpy para matrices de features.
import numpy as np
# tensorflow para modelo neuronal.
import tensorflow as tf

# extractor de voz reusable.
from mostacho.features.voice import extract_voice_features
# settings globales del proyecto.
from mostacho.settings import load_settings


EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_args() -> argparse.Namespace:
    """Argumentos para entrenamiento de voz."""

    # Se crea parser CLI.
    parser = argparse.ArgumentParser(description="Entrena clasificador de emociones por voz")
    # Ruta de carpeta CREMA-D.
    parser.add_argument("--data-dir", type=Path, default=None, help="Carpeta con .wav de CREMA-D")
    # Épocas de entrenamiento.
    parser.add_argument("--epochs", type=int, default=15, help="Numero de epocas")
    # Tamaño de lote.
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    # Límite opcional para entrenamiento rápido.
    parser.add_argument("--max-files", type=int, default=0, help="Limite opcional de archivos")
    # Retorno de argumentos.
    return parser.parse_args()


def _label_from_filename(path: Path) -> str:
    """Extrae etiqueta de emoción desde el nombre de archivo CREMA-D."""

    # Se separa nombre por guion bajo.
    parts = path.stem.split("_")
    # Si formato inesperado, se marca unknown.
    if len(parts) < 3:
        return "unknown"
    # Se lee código de emoción.
    code = parts[2].upper()
    # Se mapea a etiqueta textual.
    return EMOTION_MAP.get(code, "unknown")


def load_dataset(audio_dir: Path, max_files: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Carga dataset de voz y genera matriz de features tabulares."""

    # Se listan wav ordenados para reproducibilidad.
    files = sorted(audio_dir.glob("*.wav"))
    # Se aplica límite si el usuario lo define.
    if max_files > 0:
        files = files[:max_files]

    # Contenedores de features y etiquetas.
    rows = []
    labels = []

    # Se recorren archivos para extracción de features.
    for path in files:
        # Se deriva etiqueta por nombre.
        label = _label_from_filename(path)
        # Se ignoran registros sin etiqueta válida.
        if label == "unknown":
            continue
        # Se extraen features acústicas.
        features = extract_voice_features(path)
        # Se guardan filas y etiquetas.
        rows.append(features)
        labels.append(label)

    # Si no hay datos suficientes, se aborta con error explícito.
    if not rows:
        raise RuntimeError("No se pudieron construir features de voz.")

    # Se define orden estable de columnas por primer registro.
    feature_order = sorted(rows[0].keys())
    # Se arma matriz X en orden fijo.
    x = np.array([[row.get(name, 0.0) for name in feature_order] for row in rows], dtype=np.float32)

    # Se define inventario de clases ordenado.
    class_names = sorted(set(labels))
    # Mapeo etiqueta->índice.
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    # Se codifican etiquetas a enteros.
    y = np.array([class_to_index[label] for label in labels], dtype=np.int32)

    # Se retorna dataset listo para entrenamiento.
    return x, y, feature_order, class_names


def build_model(num_features: int, num_classes: int) -> tf.keras.Model:
    """Construye MLP simple para features de voz."""

    # Capa de entrada tabular.
    inputs = tf.keras.Input(shape=(num_features,))
    # Bloque denso principal.
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    # Dropout para regularización.
    x = tf.keras.layers.Dropout(0.2)(x)
    # Segundo bloque denso.
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # Salida softmax por clase.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Se crea y compila modelo.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Entrena y guarda clasificador de voz."""

    # Se parsean argumentos.
    args = parse_args()
    # Se cargan settings globales.
    settings = load_settings()

    # Se define ruta de audio.
    audio_dir = args.data_dir or (settings.db_root / "audio" / "crema-d")
    # Validación de existencia de ruta.
    if not audio_dir.exists():
        raise FileNotFoundError(f"No existe audio_dir: {audio_dir}")

    # Se construye dataset tabular.
    x, y, feature_order, class_names = load_dataset(audio_dir=audio_dir, max_files=args.max_files)

    # Se construye modelo MLP.
    model = build_model(num_features=x.shape[1], num_classes=len(class_names))
    # Se entrena con split automático de validación.
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, verbose=1)

    # Se prepara carpeta de salida.
    output_dir = settings.artifacts_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Se guarda modelo entrenado.
    model_path = output_dir / "voice_emotion.keras"
    model.save(model_path)

    # Se guarda metadata de features y clases.
    metadata_path = output_dir / "voice_emotion_metadata.json"
    metadata_path.write_text(
        json.dumps({"feature_order": feature_order, "class_names": class_names}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # Mensajes finales para operador.
    print(f"Modelo voz guardado en: {model_path}")
    print(f"Metadata voz guardada en: {metadata_path}")


if __name__ == "__main__":
    # Entrada directa CLI.
    main()
