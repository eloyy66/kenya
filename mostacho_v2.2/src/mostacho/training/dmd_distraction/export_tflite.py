from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import PipelineConfig


def _load_image_for_tflite(image_path: str, image_size: tuple[int, int]) -> np.ndarray:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32).numpy()
    image = np.expand_dims(image, axis=0)
    return image


def _representative_dataset_generator(train_paths: list[str], image_size: tuple[int, int]) -> Iterator[list[np.ndarray]]:
    for image_path in train_paths:
        try:
            image = _load_image_for_tflite(image_path, image_size=image_size)
            yield [image]
        except Exception:
            continue


def _inspect_tflite_model(model_path: Path) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return {
        "model_path": str(model_path),
        "input_details": input_details,
        "output_details": output_details,
    }


def _export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "export"):
        model.export(str(export_dir))
        return
    tf.saved_model.save(model, str(export_dir))


def _make_converter(saved_model_dir: Path, use_new_converter: bool) -> tf.lite.TFLiteConverter:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.experimental_new_converter = bool(use_new_converter)
    return converter


def _converter_name(use_new_converter: bool) -> str:
    return "new" if use_new_converter else "legacy"


def _short_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _convert_with_retry(
    saved_model_dir: Path,
    use_new_converter: bool,
    configure: Callable[[tf.lite.TFLiteConverter], None] | None = None,
) -> tuple[bytes, bool]:
    attempts: list[bool] = [use_new_converter]
    alternate = not use_new_converter
    if alternate not in attempts:
        attempts.append(alternate)

    last_exc: Exception | None = None
    for mode in attempts:
        converter = _make_converter(saved_model_dir, use_new_converter=mode)
        if configure is not None:
            configure(converter)
        try:
            return converter.convert(), mode
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[WARN] Conversion fallo con converter {_converter_name(mode)}: {_short_exc(exc)}")
            continue

    assert last_exc is not None
    raise last_exc


def _to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32)):
        return int(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if isinstance(value, np.dtype):
        return str(value)
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return getattr(value, "name")
    if hasattr(value, "__dict__"):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar modelo .keras a TFLite float32 e INT8 (NNAPI-friendly).")
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--max-representative-samples", type=int, default=PipelineConfig().max_representative_samples)
    parser.add_argument("--allow-int8-fallback", action="store_true")
    parser.add_argument("--use-new-converter", action="store_true", help="Usar MLIR converter nuevo (puede fallar en algunos entornos macOS).")
    args = parser.parse_args()

    cfg = PipelineConfig(
        processed_output_dir=args.processed_output_dir,
        max_representative_samples=args.max_representative_samples,
    )
    cfg.ensure_dirs()

    model_path = args.model_path or (cfg.models_dir / "dmd_distraction_best.keras")
    split_csv = args.split_csv or cfg.split_csv_path
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo .keras: {model_path}")
    if not split_csv.exists():
        raise FileNotFoundError(f"No existe split CSV: {split_csv}")

    split_df = pd.read_csv(split_csv)
    train_df = split_df[split_df["split"] == "train"]
    if train_df.empty:
        raise RuntimeError("No hay datos train para representative dataset.")
    train_paths = train_df["image_path"].astype(str).tolist()[: cfg.max_representative_samples]
    if not train_paths:
        raise RuntimeError("No se pudieron obtener rutas para representative dataset.")

    model = tf.keras.models.load_model(model_path, compile=False)

    float_tflite_path = cfg.tflite_dir / "model_float32.tflite"
    int8_tflite_path = cfg.tflite_dir / "model_int8.tflite"
    inspect_path = cfg.tflite_dir / "tflite_tensor_details.json"

    with tempfile.TemporaryDirectory(prefix="dmd_saved_model_") as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model_for_tflite"
        _export_saved_model(model, saved_model_dir)

        float_bytes, float_used_new = _convert_with_retry(
            saved_model_dir=saved_model_dir,
            use_new_converter=args.use_new_converter,
            configure=None,
        )
        float_tflite_path.write_bytes(float_bytes)

        rep_dataset = lambda: _representative_dataset_generator(train_paths, cfg.image_size)

        def _configure_int8(converter: tf.lite.TFLiteConverter) -> None:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        int8_status = "full_integer"
        try:
            int8_bytes, int8_used_new = _convert_with_retry(
                saved_model_dir=saved_model_dir,
                use_new_converter=args.use_new_converter,
                configure=_configure_int8,
            )
        except Exception as exc:  # noqa: BLE001
            if not args.allow_int8_fallback:
                raise RuntimeError(
                    "Fallo conversion INT8 full-integer. Usa --allow-int8-fallback para permitir fallback."
                ) from exc

            def _configure_dynamic_range(converter: tf.lite.TFLiteConverter) -> None:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            int8_status = f"fallback_dynamic_range ({_short_exc(exc)})"
            int8_bytes, int8_used_new = _convert_with_retry(
                saved_model_dir=saved_model_dir,
                use_new_converter=args.use_new_converter,
                configure=_configure_dynamic_range,
            )

        int8_tflite_path.write_bytes(int8_bytes)

    float_info = _inspect_tflite_model(float_tflite_path)
    int8_info = _inspect_tflite_model(int8_tflite_path)
    export_summary = {
        "keras_model_path": str(model_path),
        "float_tflite_path": str(float_tflite_path),
        "int8_tflite_path": str(int8_tflite_path),
        "int8_status": int8_status,
        "float_converter_used": _converter_name(float_used_new),
        "int8_converter_used": _converter_name(int8_used_new),
        "float_model_size_bytes": int(float_tflite_path.stat().st_size),
        "int8_model_size_bytes": int(int8_tflite_path.stat().st_size),
        "float_tensors": float_info,
        "int8_tensors": int8_info,
    }
    with inspect_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_safe(export_summary), f, indent=2)

    print(f"Float32 model: {float_tflite_path}")
    print(f"INT8 model: {int8_tflite_path}")
    print(f"Tensor details: {inspect_path}")
    print(f"INT8 status: {int8_status}")
    print(f"Converters usados: float={_converter_name(float_used_new)}, int8={_converter_name(int8_used_new)}")
    print(f"Input details: {int8_info['input_details']}")
    print(f"Output details: {int8_info['output_details']}")


if __name__ == "__main__":
    main()
