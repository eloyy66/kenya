"""Entrena clasificador visual de somnolencia multiclase desde somnolent-db.

Clases objetivo (opcion B):
- alert
- yawning
- microsleep
"""

from __future__ import annotations

# argparse permite configurar entrenamiento desde CLI.
import argparse
# collections aporta Counter para conteos por clase.
from collections import Counter
# json se usa para leer anotaciones y escribir reportes.
import json
# Path facilita manejo robusto de rutas.
from pathlib import Path
# typing para anotaciones estáticas claras.
from typing import Dict, List, Tuple

# numpy para matrices y estadísticas.
import numpy as np
# tensorflow para construir y entrenar el modelo.
import tensorflow as tf

# Settings del proyecto para resolver rutas por defecto.
from mostacho.settings import load_settings


# Orden de clases objetivo para salida estable de logits/probabilidades.
CLASS_NAMES = ["alert", "yawning", "microsleep"]
# Mapeo nombre de clase -> índice entero.
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    """Define argumentos del entrenamiento de somnolencia visual."""

    # Parser principal con descripción breve.
    parser = argparse.ArgumentParser(description="Entrena modelo visual de somnolencia (alert/yawning/microsleep)")

    # Ruta base del dataset somnolent-db.
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Ruta a db/artificialvision/somnolent-db (default: autodetect)",
    )

    # Archivo JSON de entrenamiento.
    parser.add_argument("--train-json", type=str, default="annotations_train.json", help="Nombre del split train")
    # Archivo JSON de validación.
    parser.add_argument("--val-json", type=str, default="annotations_val.json", help="Nombre del split val")
    # Archivo JSON de prueba.
    parser.add_argument("--test-json", type=str, default="annotations_test.json", help="Nombre del split test")
    # Archivo JSON de holdout final.
    parser.add_argument("--holdout-json", type=str, default="annotations_holdout.json", help="Nombre del split holdout")

    # Tamaño cuadrado de entrada de imagen.
    parser.add_argument("--image-size", type=int, default=160, help="Lado de imagen de entrada")
    # Batch size de entrenamiento.
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    # Número de épocas máximas.
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas")
    # Learning rate base para Adam.
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate inicial")

    # Límite opcional de ejemplos de train para pruebas rápidas.
    parser.add_argument("--max-train", type=int, default=0, help="Limite opcional de ejemplos de train")
    # Semilla de reproducibilidad.
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    # Bandera opcional para intentar usar pesos ImageNet.
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Intenta usar pesos pretrained ImageNet (si no están disponibles, hace fallback)",
    )

    # Reduce agresividad del augmentation del modelo (util con augmentation extra en dataset).
    parser.add_argument(
        "--light-model-augmentation",
        action="store_true",
        help="Reduce agresividad del bloque de augmentation del modelo",
    )

    # Augmentation dirigida / oversampling.
    parser.add_argument(
        "--augment-microsleep",
        action="store_true",
        help="Aplica augmentation extra solo a microsleep en el dataset de train",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Usa oversampling balanceado por clase en train (requiere steps_per_epoch fijo)",
    )

    # Selección de función de pérdida.
    parser.add_argument(
        "--loss",
        type=str,
        default="sparse",
        choices=["sparse", "focal"],
        help="Loss a usar: sparse (crossentropy) o focal",
    )
    # Parámetros de focal loss (si --loss focal).
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma para focal loss")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha para focal loss")

    # Multiplicadores para ponderar clases específicas (útil para subir recall de microsleep).
    parser.add_argument(
        "--alert-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador extra para class_weight de alert",
    )
    parser.add_argument(
        "--yawning-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador extra para class_weight de yawning",
    )
    parser.add_argument(
        "--microsleep-weight-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador extra para class_weight de microsleep",
    )

    # Retorna argumentos parseados.
    return parser.parse_args()


def _resolve_default_dataset_root(db_root: Path) -> Path:
    """Resuelve dataset somnolent-db priorizando estructura actual."""

    # Candidato principal según estructura actual.
    primary = db_root / "artificialvision" / "somnolent-db"
    # Fallback antiguo por compatibilidad.
    fallback = db_root / "somnolent-db"

    # Retorna principal si existe.
    if primary.exists():
        return primary

    # Retorna fallback si existe.
    if fallback.exists():
        return fallback

    # Si no existe ninguno, retorna principal para error explícito posterior.
    return primary


def _resolve_image_path(dataset_root: Path, json_key: str) -> Path | None:
    """Resuelve ruta de imagen desde la llave almacenada en anotaciones JSON."""

    # Normaliza separadores de ruta para compatibilidad multiplataforma.
    cleaned = json_key.replace("\\", "/").strip()
    # Elimina prefijo relativo './' cuando aparece.
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]

    # Lista de candidatos posibles para la misma llave.
    candidates: List[Path] = []

    # Caso directo: dataset_root + llave completa.
    candidates.append(dataset_root / cleaned)

    # Caso común: llave comienza con classification_frames/.
    if cleaned.startswith("classification_frames/"):
        # Se extrae ruta relativa sin el prefijo del índice.
        remainder = cleaned.split("/", 1)[1]
        # En tu estructura actual las imágenes están en dataset_root/<subject>/frameN.jpg.
        candidates.append(dataset_root / remainder)
        # Fallback a raíz db/classification_frames para estructuras antiguas.
        candidates.append(dataset_root.parent.parent / "classification_frames" / remainder)
        # Fallback a db/artificialvision/classification_frames.
        candidates.append(dataset_root.parent / "classification_frames" / remainder)

    # Devolver primer candidato existente.
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Si no se encuentra, retorna None para filtrar ejemplo.
    return None


def _load_split(dataset_root: Path, split_json_name: str) -> Tuple[List[str], np.ndarray, Dict[str, int], int]:
    """Carga un split JSON y devuelve rutas + etiquetas + estadísticas."""

    # Ruta completa al archivo JSON del split.
    split_path = dataset_root / split_json_name
    # Verificación de existencia del split.
    if not split_path.exists():
        raise FileNotFoundError(f"No existe split JSON: {split_path}")

    # Carga del contenido JSON a diccionario.
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    # Validación básica del formato esperado.
    if not isinstance(payload, dict):
        raise ValueError(f"Formato JSON inválido en {split_path}; se esperaba un objeto clave->anotación")

    # Contenedores de salida del split.
    image_paths: List[str] = []
    labels: List[int] = []
    class_counter: Counter[str] = Counter()
    missing_images = 0

    # Recorre entradas del JSON para construir dataset.
    for json_key, metadata in payload.items():
        # Valida que metadata sea objeto con campos esperados.
        if not isinstance(metadata, dict):
            continue

        # Lee estado del conductor y lo normaliza.
        driver_state = str(metadata.get("driver_state", "")).strip().lower()

        # Ignora clases fuera de la opción B objetivo.
        if driver_state not in CLASS_TO_INDEX:
            continue

        # Resuelve ruta real de imagen para esta anotación.
        image_path = _resolve_image_path(dataset_root, str(json_key))
        if image_path is None:
            missing_images += 1
            continue

        # Agrega ruta absoluta serializada.
        image_paths.append(str(image_path))
        # Agrega etiqueta numérica.
        labels.append(CLASS_TO_INDEX[driver_state])
        # Cuenta clase para reporte.
        class_counter[driver_state] += 1

    # Conversión final a numpy para TensorFlow.
    labels_array = np.asarray(labels, dtype=np.int32)

    # Retorna datos del split + métricas de carga.
    return image_paths, labels_array, dict(class_counter), missing_images


def _build_dataset(
    image_paths: List[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augment_microsleep: bool = False,
) -> tf.data.Dataset:
    """Construye tf.data.Dataset con decode/resize y augmentation opcional."""

    # Dataset base desde tensores de rutas y etiquetas.
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle para entrenamiento, manteniendo reproducibilidad con seed.
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    # Función interna para lectura y preprocesado por ejemplo.
    def _microsleep_augment(image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # Blur leve simulado.
        image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), ksize=3, strides=1, padding="SAME")[0]
        return image

    def _load_example(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Lee bytes de archivo de imagen.
        image_bytes = tf.io.read_file(path)
        # Decodifica JPEG/PNG automáticamente a 3 canales RGB.
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        # Convierte tipo a float32 para red neuronal.
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Redimensiona a tamaño fijo cuadrado.
        image = tf.image.resize(image, (image_size, image_size))
        # Augmentation extra solo para microsleep.
        if augment_microsleep:
            image = tf.cond(tf.equal(label, CLASS_TO_INDEX["microsleep"]), lambda: _microsleep_augment(image), lambda: image)
        # Retorna imagen y etiqueta sin one-hot (sparse labels).
        return image, label

    # Mapea función de carga en paralelo.
    dataset = dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)
    # Agrupa por batch.
    dataset = dataset.batch(batch_size)
    # Prefetch para pipeline eficiente.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Retorna dataset listo para model.fit/evaluate.
    return dataset


def _build_balanced_dataset(
    image_paths: List[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    seed: int,
    augment_microsleep: bool,
) -> tf.data.Dataset:
    """Dataset balanceado por clase con oversampling uniforme."""

    def _load_and_augment(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))

        def microsleep_aug() -> tf.Tensor:
            img = tf.image.random_flip_left_right(image)
            img = tf.image.random_brightness(img, max_delta=0.3)
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
            return img

        if augment_microsleep:
            augmented = tf.cond(tf.equal(label, CLASS_TO_INDEX["microsleep"]), microsleep_aug, lambda: image)
        else:
            augmented = image
        return augmented, label

    paths_arr = np.asarray(image_paths)
    if len(paths_arr) != len(labels):
        raise ValueError("image_paths y labels no tienen la misma longitud.")

    datasets: List[tf.data.Dataset] = []
    weights: List[float] = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = labels == class_idx
        if not np.any(mask):
            raise RuntimeError(f"No hay muestras para la clase {CLASS_NAMES[class_idx]}")
        ds = tf.data.Dataset.from_tensor_slices((paths_arr[mask], labels[mask]))
        ds = ds.repeat().shuffle(500, seed=seed)
        datasets.append(ds)
        weights.append(1.0)

    # Muestreo uniforme entre clases.
    balanced = tf.data.Dataset.sample_from_datasets(datasets, weights=weights, seed=seed)

    return (
        balanced.map(_load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _class_weights_from_labels(labels: np.ndarray) -> Dict[int, float]:
    """Calcula pesos por clase para compensar desbalance en train split."""

    # Conteos por índice de clase.
    counter = Counter(labels.tolist())
    # Total de ejemplos de entrenamiento.
    total = float(len(labels))
    # Número de clases objetivo.
    num_classes = float(len(CLASS_NAMES))

    # Calcula peso inversamente proporcional a frecuencia por clase.
    weights = {
        class_index: total / (num_classes * float(counter[class_index]))
        for class_index in range(len(CLASS_NAMES))
        if counter[class_index] > 0
    }

    # Retorna diccionario class_index -> weight.
    return weights


def _sparse_focal_loss(gamma: float, alpha: float, class_weights: Dict[int, float] | None = None):
    """Crea focal loss para etiquetas sparse con salida softmax."""

    weights_tensor = None
    #if class_weights:
    #    weights_tensor = tf.constant(
    #        [float(class_weights[index]) for index in range(len(CLASS_NAMES))], dtype=tf.float32
    #    )
    
    # normalizar pesos para que su media sea 1 (evita que la loss se vuelva demasiado grande o pequeña)
    if class_weights:
        raw = [float(class_weights[index]) for index in range(len(CLASS_NAMES))]
        mean_w = sum(raw) / len(raw)
        normalized = [w / mean_w for w in raw]
        weights_tensor = tf.constant(normalized, dtype=tf.float32)

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true: (batch,), y_pred: (batch, num_classes)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Probabilidad del target verdadero.
        pt = tf.gather(y_pred, y_true, batch_dims=1)
        # Focal loss clásica para multiclass.
        loss = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        # Aplica pesos por clase si existen.
        if weights_tensor is not None:
            sample_weights = tf.gather(weights_tensor, y_true)
            loss = loss * sample_weights
        return tf.reduce_mean(loss)

    return _loss


def _build_model(
    image_size: int,
    learning_rate: float,
    pretrained: bool,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float,
    focal_class_weights: Dict[int, float] | None,
    light_augmentation: bool,
) -> tf.keras.Model:
    """Construye modelo CNN para clasificación de somnolencia multiclase."""

    # Capas de aumento de datos para robustez en conducción real.
    rotation_factor = 0.01 if light_augmentation else 0.03
    contrast_factor = 0.05 if light_augmentation else 0.15
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(rotation_factor),
            tf.keras.layers.RandomContrast(contrast_factor),
        ],
        name="augmentation",
    )

    # Selección opcional de pesos pretrained.
    weights_name = "imagenet" if pretrained else None

    # Intenta crear backbone MobileNetV2.
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=weights_name,
        )
    except Exception:
        # Fallback offline cuando no hay disponibilidad de pesos externos.
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=None,
        )

    # Se congela backbone para fase inicial estable.
    backbone.trainable = False

    # Entrada de imagen RGB normalizada [0,1].
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    # Aplica augmentación en entrenamiento.
    x = augmentation(inputs)
    # Preprocesado específico de MobileNetV2.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    # Extracción de features del backbone.
    x = backbone(x, training=False)
    # Pooling global para vector fijo.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Capa densa intermedia.
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    # Dropout para regularización.
    x = tf.keras.layers.Dropout(0.3)(x)
    # Salida softmax para 3 clases.
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    # Construye modelo final.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="drowsiness_vision_multiclass")
    # Compilación con pérdida sparse categórica.
    loss_fn: tf.keras.losses.Loss | str
    if loss_name == "focal":
        loss_fn = _sparse_focal_loss(gamma=focal_gamma, alpha=focal_alpha, class_weights=focal_class_weights)
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Retorna modelo compilado.
    return model


def _evaluate_split(model: tf.keras.Model, dataset: tf.data.Dataset) -> Dict[str, float]:
    """Evalúa un split y retorna métricas serializables."""

    # Ejecuta evaluación en modo no verboso.
    loss, accuracy = model.evaluate(dataset, verbose=0)
    # Retorna estructura simple de métricas.
    return {"loss": float(loss), "accuracy": float(accuracy)}


def main() -> None:
    """Entrena modelo multiclase de somnolencia y guarda artefactos."""

    # Parsea argumentos de ejecución.
    args = parse_args()
    # Carga settings globales del proyecto.
    settings = load_settings()

    # Semilla global para reproducibilidad aproximada.
    tf.keras.utils.set_random_seed(args.seed)


    # ── Validaciones de flags ──────────────────────────────────────────────
    if args.balanced_sampling and (
        args.microsleep_weight_multiplier != 1.0
        or args.alert_weight_multiplier != 1.0
        or args.yawning_weight_multiplier != 1.0
    ):
        print(
            "[WARN] --balanced-sampling activo: los multiplicadores de class_weight "
            "se ignorarán para evitar doble ponderación."
        )

    if args.light_model_augmentation and not args.augment_microsleep:
        print(
            "[WARN] --light-model-augmentation sin --augment-microsleep: "
            "reducirás augmentation del modelo sin añadir nada a microsleep."
        )
    # ── Fin validaciones ───────────────────────────────────────────────────


    # Resuelve dataset root con estructura nueva/fallback.
    dataset_root = args.dataset_root or _resolve_default_dataset_root(settings.db_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe dataset_root: {dataset_root}")

    # Carga splits oficiales desde JSON.
    train_paths, train_labels, train_counts, missing_train = _load_split(dataset_root, args.train_json)
    val_paths, val_labels, val_counts, missing_val = _load_split(dataset_root, args.val_json)
    test_paths, test_labels, test_counts, missing_test = _load_split(dataset_root, args.test_json)
    holdout_paths, holdout_labels, holdout_counts, missing_holdout = _load_split(dataset_root, args.holdout_json)

    # Limita train para pruebas rápidas si usuario lo indica.
    if args.max_train > 0:
        train_paths = train_paths[: args.max_train]
        train_labels = train_labels[: args.max_train]

    # Validaciones de seguridad para evitar entrenamiento vacío.
    if len(train_paths) == 0:
        raise RuntimeError("Split de entrenamiento vacío. Revisa rutas/anotaciones.")
    if len(val_paths) == 0:
        raise RuntimeError("Split de validación vacío. Revisa rutas/anotaciones.")

    # Construye datasets tf.data.
    if args.balanced_sampling:
        train_ds = _build_balanced_dataset(
            train_paths,
            train_labels,
            args.image_size,
            args.batch_size,
            args.seed,
            augment_microsleep=args.augment_microsleep,
        )
    else:
        train_ds = _build_dataset(
            train_paths,
            train_labels,
            args.image_size,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            augment_microsleep=args.augment_microsleep,
        )
    val_ds = _build_dataset(val_paths, val_labels, args.image_size, args.batch_size, shuffle=False, seed=args.seed)
    test_ds = _build_dataset(test_paths, test_labels, args.image_size, args.batch_size, shuffle=False, seed=args.seed)
    holdout_ds = _build_dataset(holdout_paths, holdout_labels, args.image_size, args.batch_size, shuffle=False, seed=args.seed)

    # Calcula pesos por clase para compensar desbalance del train split.
    class_weight = _class_weights_from_labels(train_labels) # ... multiplica por args.*_multiplier
    class_weight[CLASS_TO_INDEX["alert"]] *= float(args.alert_weight_multiplier)
    class_weight[CLASS_TO_INDEX["yawning"]] *= float(args.yawning_weight_multiplier)
    class_weight[CLASS_TO_INDEX["microsleep"]] *= float(args.microsleep_weight_multiplier)
    if args.balanced_sampling:
        # Oversampling ya balancea por clase; class_weight causaría doble sesgo.
        class_weight = None

    # Para focal loss, aplicar class_weight dentro de la loss (evita que se ignore).
    # focal_class_weights toma el valor DESPUÉS de haber resuelto balanced_sampling,
    # por lo que si balanced_sampling está activo, focal tampoco usará pesos.
    focal_class_weights = class_weight if args.loss == "focal" else None
    if args.loss == "focal":
        class_weight = None # focal ya los aplica internamente; evita doble ponderación en fit.

    auto_light_augmentation = bool(args.augment_microsleep and not args.light_model_augmentation)
    if auto_light_augmentation:
        print(
            "[INFO] --augment-microsleep activo: reduciendo agresividad del bloque de augmentation del modelo."
        )

    # Construye modelo de clasificación visual.
    model = _build_model(
        args.image_size,
        args.learning_rate,
        pretrained=args.pretrained,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        focal_class_weights=focal_class_weights,
        light_augmentation=bool(auto_light_augmentation or args.light_model_augmentation),
    )

    # Carpeta destino de artefactos.
    output_dir = settings.artifacts_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ruta del mejor checkpoint por validación.
    best_model_path = output_dir / "drowsiness_vision_best.keras"

    # Callbacks para estabilidad de entrenamiento.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True),
    ]

    # Entrenamiento principal.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        steps_per_epoch=int(np.ceil(len(train_paths) / float(args.batch_size))) if args.balanced_sampling else None,
        callbacks=callbacks,
        verbose=1,
    )

    # Guarda modelo final (con best weights restaurados por EarlyStopping).
    final_model_path = output_dir / "drowsiness_vision.keras"
    model.save(final_model_path)

    # Guarda nombres de clases para inferencia consistente.
    classes_path = output_dir / "drowsiness_vision_classes.txt"
    classes_path.write_text("\n".join(CLASS_NAMES), encoding="utf-8")

    # Evalúa splits de test y holdout para reporte final.
    test_metrics = _evaluate_split(model, test_ds)
    holdout_metrics = _evaluate_split(model, holdout_ds)

    # Construye reporte JSON de entrenamiento y métricas.
    report = {
        "dataset_root": str(dataset_root),
        "class_names": CLASS_NAMES,
        "train_size": int(len(train_paths)),
        "val_size": int(len(val_paths)),
        "test_size": int(len(test_paths)),
        "holdout_size": int(len(holdout_paths)),
        "missing_images": {
            "train": int(missing_train),
            "val": int(missing_val),
            "test": int(missing_test),
            "holdout": int(missing_holdout),
        },
        "class_distribution": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
            "holdout": holdout_counts,
        },
        "class_weight": {str(k): float(v) for k, v in class_weight.items()} if class_weight else None,
        "class_weight_multiplier": {
            "alert": float(args.alert_weight_multiplier),
            "yawning": float(args.yawning_weight_multiplier),
            "microsleep": float(args.microsleep_weight_multiplier),
        },
        "focal_class_weight": {str(k): float(v) for k, v in focal_class_weights.items()} if focal_class_weights else None,
        "training_flags": {
            "balanced_sampling": bool(args.balanced_sampling),
            "augment_microsleep": bool(args.augment_microsleep),
            "light_model_augmentation": bool(auto_light_augmentation or args.light_model_augmentation),
        },
        "loss": {
            "name": str(args.loss),
            "focal_gamma": float(args.focal_gamma),
            "focal_alpha": float(args.focal_alpha),
        },
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "metrics": {
            "test": test_metrics,
            "holdout": holdout_metrics,
        },
        "artifacts": {
            "final_model": str(final_model_path),
            "best_model": str(best_model_path),
            "classes": str(classes_path),
        },
    }

    # Guarda reporte de entrenamiento en artifacts/models.
    report_path = output_dir / "drowsiness_vision_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    # Imprime resumen útil en consola.
    print(f"Modelo final guardado en: {final_model_path}")
    print(f"Mejor checkpoint guardado en: {best_model_path}")
    print(f"Clases guardadas en: {classes_path}")
    print(f"Reporte guardado en: {report_path}")
    print(f"Test metrics: {test_metrics}")
    print(f"Holdout metrics: {holdout_metrics}")


if __name__ == "__main__":
    # Entrada CLI del script.
    main()
