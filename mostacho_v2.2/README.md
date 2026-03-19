# Mostacho v2.2

Carpeta de trabajo local para combinar:

- `InsightFace` desde `mostacho_2.1`
- modelo `Open/Closed Eyes`
- modelo `drowsiness_vision_best.keras`

La idea de esta versión es:

- `face_worker` corre en subproceso y entrega cara, landmarks, EAR y candidato de bostezo
- `tf_inference_worker` corre en subproceso y unifica `open/closed eyes` + `drowsiness` (y `distraction` opcional)
- la API principal solo orquesta estado y dispara inferencia pesada cuando detecta ojos cerrados o bostezo

Esto reduce inferencia constante del modelo de somnolencia y deja el proyecto listo para trabajo local, no para empaquetado final de GitHub.

## Flujo recomendado

- un venv TensorFlow para `mostacho_v2.2`
- un venv separado para `InsightFace`
- `MOSTACHO_FACE_PYTHON` apuntando al Python del venv con `insightface`
- `MOSTACHO_TF_PYTHON` apuntando al Python del venv TensorFlow (si no usas el mismo)

### Variables útiles

- `MOSTACHO_FACE_PYTHON=/ruta/al/python/del/venv_insightface`
- `MOSTACHO_TF_PYTHON=/ruta/al/python/del/venv_tensorflow`
- `MOSTACHO_FACE_PROVIDERS=CPUExecutionProvider`
- `MOSTACHO_FACE_MODEL=buffalo_l`
- `MOSTACHO_YAWN_TRIGGER_THRESHOLD=0.32`
- `MOSTACHO_EYE_CLOSED_CONFIDENCE=0.60`
- `MOSTACHO_DROWSINESS_ACTIVE_SECONDS=2.0`
- `MOSTACHO_DISTRACTION_ENABLED=1`
- `MOSTACHO_DISTRACTION_INTERVAL_SECONDS=0.40`
- `MOSTACHO_DISTRACTION_MODEL=/ruta/al/dmd_distraction_best.keras` (opcional si quieres forzar ruta)

### API principal

```bash
uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002
```

### Monitor de cámara

```bash
python -m mostacho.realtime_monitor --camera-backend AVFOUNDATION
```

## Pipeline DMD distraction (entrenamiento + TFLite)

### Instalar dependencias

```bash
source /Users/usuario/venvs/m22tf/bin/activate
cd /Users/usuario/kenya/mostacho_v2.2
python -m pip install -r requirements/distraction_pipeline.txt
python -m pip install -e .
```

### 1) Preprocesamiento (descubrir dataset + extraer frames con labels)

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.extract_frames \
  --dataset-root /Users/usuario/kenya/db/artificialvision/minidatasetdmd \
  --dataset-root /Users/usuario/kenya/db/artificialvision/dmd-distraction-rgb-ir \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed \
  --frame-sampling-seconds 0.75
```

### 2) Construir splits 80/10/10 sin leakage

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.build_splits \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed \
  --split-strategy subject
```

### 3) Entrenar clasificador binario (normal vs distracted)

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.train \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed \
  --backbone-name MobileNetV3Small \
  --freeze-epochs 10 \
  --fine-tune-epochs 10 \
  --fine-tune-layers 40
```

### 4) Evaluar + ajustar umbral para minimizar falsos positivos

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.evaluate \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed \
  --min-precision 0.90
```

### 5) Exportar TensorFlow Lite (float + INT8)

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.export_tflite \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed \
  --allow-int8-fallback
```

### 6) Sanity check utilitario

```bash
cd /Users/usuario/kenya/mostacho_v2.2
PYTHONPATH=src python -m mostacho.training.dmd_distraction.sanity_check \
  --dataset-root /Users/usuario/kenya/db/artificialvision/minidatasetdmd \
  --dataset-root /Users/usuario/kenya/db/artificialvision/dmd-distraction-rgb-ir \
  --processed-output-dir /Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed
```

### Ejecutar todo con un solo comando

```bash
cd /Users/usuario/kenya/mostacho_v2.2
./run_all.sh
```

Ejemplo con overwrite e instalacion de deps:

```bash
cd /Users/usuario/kenya/mostacho_v2.2
DATASET_ROOTS="/Users/usuario/kenya/db/artificialvision/minidatasetdmd;/Users/usuario/kenya/db/artificialvision/dmd-distraction-rgb-ir" \
INSTALL_DEPS=1 OVERWRITE=1 ./run_all.sh
```

### Archivos clave generados

- `artifacts/dmd_processed/labels.csv`
- `artifacts/dmd_processed/splits/labels_with_split.csv`
- `artifacts/dmd_processed/models/dmd_distraction_best.keras`
- `artifacts/dmd_processed/reports/confusion_matrix.png`
- `artifacts/dmd_processed/reports/metrics.json`
- `artifacts/dmd_processed/tflite/model_float32.tflite`
- `artifacts/dmd_processed/tflite/model_int8.tflite`
- ejemplo Android: `android_inference_example.kt`
