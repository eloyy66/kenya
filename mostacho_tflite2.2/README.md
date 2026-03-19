# Mostacho TFLite 2.2

Runtime mínimo para inferencia en cámara con:

- `InsightFace` en subproceso (`m22if`)
- worker TensorFlow/TFLite unificado para:
  - `open/closed eyes`
  - `drowsiness`
  - `distraction` (opcional)

Objetivo: usar modelos `INT8 TFLite` cuando existan, con fallback automático a `.keras`.

## Estructura

- `src/mostacho`: API, monitor y workers runtime
- `artifacts/models`: modelos `.keras` y `.tflite`
- `scripts/export_runtime_tflite_int8.py`: conversión a TFLite INT8

## Conversión de modelos a TFLite INT8

```bash
source /Users/usuario/venvs/m22tf/bin/activate
cd /Users/usuario/kenya/mostacho_tflite2.2
PYTHONPATH=src python scripts/export_runtime_tflite_int8.py \
  --eye-dataset-root /Users/usuario/kenya/db/artificialvision/train \
  --face-representative-root /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed_dmd_r3/images \
  --max-representative-samples 300
```

Archivos esperados en `artifacts/models`:

- `eye_state_model_int8.tflite`
- `drowsiness_vision_int8.tflite`
- `distraction_model_int8.tflite`

## Ejecutar API + monitor

Terminal 1 (API):

```bash
source /Users/usuario/venvs/m22tf/bin/activate
cd /Users/usuario/kenya/mostacho_tflite2.2
export PYTHONPATH=/Users/usuario/kenya/mostacho_tflite2.2/src
export MOSTACHO_FACE_PYTHON=/Users/usuario/venvs/m22if/bin/python
export MOSTACHO_TF_PYTHON=/Users/usuario/venvs/m22tf/bin/python
export MOSTACHO_DISTRACTION_ENABLED=1
export MOSTACHO_DISTRACTION_INTERVAL_SECONDS=0.40
uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002
```

Terminal 2 (monitor):

```bash
source /Users/usuario/venvs/m22tf/bin/activate
cd /Users/usuario/kenya/mostacho_tflite2.2
export PYTHONPATH=/Users/usuario/kenya/mostacho_tflite2.2/src
python -m mostacho.realtime_monitor --api-url http://127.0.0.1:8002 --camera-backend AVFOUNDATION --camera-index 1 --max-fps 12
```

## Nota de rendimiento

Sí es rentable: INT8 reduce memoria y suele mejorar latencia en CPU móvil.  
En macOS con Metal el beneficio depende del backend, pero para despliegue Android/TFLite normalmente conviene mantener INT8.
