"""Convierte modelos runtime (.keras) a TFLite (float32 + INT8) para mostacho_tflite2.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Callable, Iterator, List

import numpy as np
import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(root: Path, limit: int) -> List[Path]:
    if not root.exists():
        return []
    paths = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if limit > 0:
        return paths[:limit]
    return paths


def _load_rgb_image(path: Path) -> tf.Tensor:
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.cast(image, tf.float32)


def _load_eye_image(path: Path) -> tf.Tensor:
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=1, expand_animations=False)
    return tf.image.convert_image_dtype(image, tf.float32)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "export"):
        model.export(str(export_dir))
        return
    tf.saved_model.save(model, str(export_dir))


def _strip_augmentation_layers(model: tf.keras.Model) -> tf.keras.Model:
    """Reconstuye un modelo sin capas de augmentación para export TFLite."""

    input_shape = tuple(model.input_shape[1:])
    input_dtype = getattr(model, "input_dtype", "float32")
    inputs = tf.keras.Input(shape=input_shape, dtype=input_dtype, name="export_input")
    x = inputs

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        layer_name = str(getattr(layer, "name", "")).lower()
        if "augmentation" in layer_name or layer_name.startswith("random_") or "random" in layer_name:
            continue
        x = layer(x)

    stripped = tf.keras.Model(inputs=inputs, outputs=x, name=f"{model.name}_stripped")
    return stripped


def _make_converter(saved_model_dir: Path, use_new_converter: bool) -> tf.lite.TFLiteConverter:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.experimental_new_converter = bool(use_new_converter)
    return converter


def _convert_with_retry(
    saved_model_dir: Path,
    configure: Callable[[tf.lite.TFLiteConverter], None] | None = None,
    prefer_new_converter: bool = True,
) -> bytes:
    attempts = [prefer_new_converter, not prefer_new_converter]
    last_exc: Exception | None = None

    for use_new in attempts:
        converter = _make_converter(saved_model_dir, use_new_converter=use_new)
        if configure is not None:
            configure(converter)
        try:
            return converter.convert()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            mode = "new" if use_new else "legacy"
            print(f"[WARN] conversion fallo con converter {mode}: {type(exc).__name__}: {exc}")
            continue

    assert last_exc is not None
    raise last_exc


def _save_tensor_details(model_path: Path, output_json: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    payload = {
        "input_details": [
            {
                "name": item["name"],
                "shape": [int(v) for v in item["shape"]],
                "dtype": str(item["dtype"]),
                "quantization": [float(item["quantization"][0]), int(item["quantization"][1])],
            }
            for item in interpreter.get_input_details()
        ],
        "output_details": [
            {
                "name": item["name"],
                "shape": [int(v) for v in item["shape"]],
                "dtype": str(item["dtype"]),
                "quantization": [float(item["quantization"][0]), int(item["quantization"][1])],
            }
            for item in interpreter.get_output_details()
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _convert_float32(model_path: Path, output_path: Path) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    model_name = model_path.name.lower()
    if "eye_state" in model_name or "drowsiness_vision" in model_name:
        model = _strip_augmentation_layers(model)
    with tempfile.TemporaryDirectory(prefix="runtime_saved_model_") as tmp_dir:
        export_dir = Path(tmp_dir) / "saved_model"
        _export_saved_model(model, export_dir)
        tflite_bytes = _convert_with_retry(export_dir, configure=None, prefer_new_converter=True)
    _write_bytes(output_path, tflite_bytes)


def _convert_int8(
    model_path: Path,
    output_path: Path,
    representative_dataset_fn: Callable[[], Iterator[list[np.ndarray]]],
) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    model_name = model_path.name.lower()
    if "eye_state" in model_name or "drowsiness_vision" in model_name:
        model = _strip_augmentation_layers(model)
    with tempfile.TemporaryDirectory(prefix="runtime_saved_model_") as tmp_dir:
        export_dir = Path(tmp_dir) / "saved_model"
        _export_saved_model(model, export_dir)

        def _configure(converter: tf.lite.TFLiteConverter) -> None:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_fn
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_bytes = _convert_with_retry(export_dir, configure=_configure, prefer_new_converter=True)
    _write_bytes(output_path, tflite_bytes)


def _eye_representative(
    image_paths: List[Path],
    image_size: int,
    max_samples: int,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    def _generator() -> Iterator[list[np.ndarray]]:
        used = 0
        for path in image_paths:
            image = _load_eye_image(path)
            image = tf.image.resize(image, (image_size, image_size))
            batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
            yield [batch]
            used += 1
            if used >= max_samples:
                return
        while used < max_samples:
            batch = np.random.uniform(0.0, 1.0, size=(1, image_size, image_size, 1)).astype(np.float32)
            yield [batch]
            used += 1

    return _generator


def _drowsiness_representative(
    image_paths: List[Path],
    image_h: int,
    image_w: int,
    max_samples: int,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    def _generator() -> Iterator[list[np.ndarray]]:
        used = 0
        for path in image_paths:
            image = _load_rgb_image(path)
            image = tf.image.resize(image, (image_h, image_w))
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
            yield [batch]
            used += 1
            if used >= max_samples:
                return
        while used < max_samples:
            batch = np.random.uniform(-1.0, 1.0, size=(1, image_h, image_w, 3)).astype(np.float32)
            yield [batch]
            used += 1

    return _generator


def _distraction_representative(
    image_paths: List[Path],
    image_h: int,
    image_w: int,
    max_samples: int,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    def _generator() -> Iterator[list[np.ndarray]]:
        used = 0
        for path in image_paths:
            image = _load_rgb_image(path)
            image = tf.image.resize(image, (image_h, image_w))
            batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)
            yield [batch]
            used += 1
            if used >= max_samples:
                return
        while used < max_samples:
            batch = np.random.uniform(0.0, 255.0, size=(1, image_h, image_w, 3)).astype(np.float32)
            yield [batch]
            used += 1

    return _generator


def _input_hw(model_path: Path, default_hw: tuple[int, int]) -> tuple[int, int]:
    model = tf.keras.models.load_model(model_path, compile=False)
    model_name = model_path.name.lower()
    if "eye_state" in model_name or "drowsiness_vision" in model_name:
        model = _strip_augmentation_layers(model)
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) >= 3:
        return int(shape[1] or default_hw[0]), int(shape[2] or default_hw[1])
    return default_hw


def _maybe_convert(
    model_path: Path,
    float_out: Path,
    int8_out: Path,
    representative_fn: Callable[[], Iterator[list[np.ndarray]]],
    tensor_details_json: Path,
) -> None:
    if not model_path.exists():
        print(f"[SKIP] no existe: {model_path}")
        return

    print(f"[INFO] Convirtiendo: {model_path.name}")
    try:
        _convert_float32(model_path, float_out)
        print(f"[OK] float32: {float_out}")
        _convert_int8(model_path, int8_out, representative_fn)
        print(f"[OK] int8: {int8_out}")
        _save_tensor_details(int8_out, tensor_details_json)
        print(f"[OK] detalles: {tensor_details_json}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] conversion fallo para {model_path.name}: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export runtime models to TFLite INT8 for mostacho_tflite2.2")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eye-dataset-root", type=Path, default=Path("/Users/usuario/kenya/db/artificialvision/train"))
    parser.add_argument(
        "--face-representative-root",
        type=Path,
        default=Path("/Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed_dmd_r3/images"),
    )
    parser.add_argument("--max-representative-samples", type=int, default=300)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    models_dir = project_root / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    eye_model = models_dir / "eye_state_model_best.keras"
    drowsiness_model = models_dir / "drowsiness_vision_best.keras"
    distraction_model = models_dir / "dmd_distraction_best.keras"

    eye_h, eye_w = _input_hw(eye_model, (96, 96)) if eye_model.exists() else (96, 96)
    drows_h, drows_w = _input_hw(drowsiness_model, (160, 160)) if drowsiness_model.exists() else (160, 160)
    distract_h, distract_w = _input_hw(distraction_model, (224, 224)) if distraction_model.exists() else (224, 224)

    eye_images = _collect_images(args.eye_dataset_root / "Open_Eyes", args.max_representative_samples // 2) + _collect_images(
        args.eye_dataset_root / "Closed_Eyes",
        args.max_representative_samples // 2,
    )
    face_images = _collect_images(args.face_representative_root, args.max_representative_samples)

    print(f"[INFO] eye reps: {len(eye_images)} | face reps: {len(face_images)}")

    _maybe_convert(
        model_path=eye_model,
        float_out=models_dir / "eye_state_model_float32.tflite",
        int8_out=models_dir / "eye_state_model_int8.tflite",
        representative_fn=_eye_representative(eye_images, image_size=eye_h, max_samples=args.max_representative_samples),
        tensor_details_json=models_dir / "eye_state_tflite_details.json",
    )

    _maybe_convert(
        model_path=drowsiness_model,
        float_out=models_dir / "drowsiness_vision_float32.tflite",
        int8_out=models_dir / "drowsiness_vision_int8.tflite",
        representative_fn=_drowsiness_representative(
            face_images,
            image_h=drows_h,
            image_w=drows_w,
            max_samples=args.max_representative_samples,
        ),
        tensor_details_json=models_dir / "drowsiness_tflite_details.json",
    )

    _maybe_convert(
        model_path=distraction_model,
        float_out=models_dir / "distraction_model_float32.tflite",
        int8_out=models_dir / "distraction_model_int8.tflite",
        representative_fn=_distraction_representative(
            face_images,
            image_h=distract_h,
            image_w=distract_w,
            max_samples=args.max_representative_samples,
        ),
        tensor_details_json=models_dir / "distraction_tflite_details.json",
    )


if __name__ == "__main__":
    main()
