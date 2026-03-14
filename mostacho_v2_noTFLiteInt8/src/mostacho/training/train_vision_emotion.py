"""Entrenamiento base de emociones visuales con TensorFlow."""

from __future__ import annotations

# argparse para configuración por CLI.
import argparse
# Path para rutas robustas.
from pathlib import Path

# tensorflow para entrenamiento CNN.
import tensorflow as tf

# configuración global del proyecto.
from mostacho.settings import load_settings


def _resolve_default_vision_data_dir(db_root: Path) -> Path:
    """Resuelve directorio de entrenamiento visual con prioridad en estructura nueva."""

    # Orden de prioridad: estructura nueva primero, luego fallback legacy.
    candidates = [
        db_root / "artificialvision" / "generalcontext-emotions2",
        db_root / "artificialvision" / "generalcontext-emotions",
        db_root / "processed_data",
        db_root / "Data",
    ]

    # Retorna primer directorio existente.
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Si nada existe, retorna candidato principal para mensaje de error consistente.
    return candidates[0]


def parse_args() -> argparse.Namespace:
    """Define argumentos de entrenamiento visual."""

    # Se crea parser con descripción.
    parser = argparse.ArgumentParser(description="Entrena clasificador visual de emociones")
    # Carpeta de imágenes por clase.
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio con subcarpetas por clase (default: db/artificialvision/generalcontext-emotions2)",
    )
    # Tamaño de lote para entrenamiento.
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    # Épocas de entrenamiento.
    parser.add_argument("--epochs", type=int, default=5, help="Numero de epocas")
    # Tamaño de entrada cuadrado de imagen.
    parser.add_argument("--image-size", type=int, default=128, help="Lado de imagen")
    # Se retorna namespace.
    return parser.parse_args()


def build_model(num_classes: int, image_size: int) -> tf.keras.Model:
    """Construye CNN compacta para baseline visual."""

    # Capa de entrada RGB.
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    # Normalización [0,1].
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    # Primer bloque convolucional.
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # Segundo bloque convolucional.
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    # Tercer bloque convolucional.
    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Capa densa para representación intermedia.
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    # Salida softmax por clases emocionales.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Modelo final compilado.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    """Entrena y guarda modelo visual baseline."""

    # Se leen argumentos de CLI.
    args = parse_args()
    # Se carga configuración global.
    settings = load_settings()

    # Se resuelve ruta de entrenamiento.
    data_dir = args.data_dir or _resolve_default_vision_data_dir(settings.db_root)
    # Validación temprana de ruta.
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe data_dir: {data_dir}")

    # Dataset de entrenamiento con split interno.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    # Dataset de validación.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    # Cálculo de clases desde dataset.
    class_names = train_ds.class_names
    # Construcción de modelo.
    model = build_model(num_classes=len(class_names), image_size=args.image_size)
    # Entrenamiento baseline.
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Ruta de salida del modelo.
    output_path = settings.artifacts_root / "models" / "vision_emotion.keras"
    # Creación de carpeta destino.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Guardado del modelo.
    model.save(output_path)

    # Guardado de clases en texto para inferencia.
    classes_path = settings.artifacts_root / "models" / "vision_emotion_classes.txt"
    classes_path.write_text("\n".join(class_names), encoding="utf-8")

    # Log final de artefactos.
    print(f"Modelo visual guardado en: {output_path}")
    print(f"Clases visuales guardadas en: {classes_path}")


if __name__ == "__main__":
    # Entrada directa por CLI.
    main()
