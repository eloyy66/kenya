# Runbook

## 1) Indexar datasets

```bash
python -m mostacho.cli index-db --write-json
```

## 2) Levantar Face Service (entorno face)

```bash
uvicorn mostacho.services.face_api:app --host 127.0.0.1 --port 8001 --reload
```

### Opcional: correr solo la base visual en tiempo real

```bash
python -m mostacho.vision.realtime --camera-backend AUTO
```

Controles runtime:
- `q`: salir
- `u`: subir umbral EAR
- `j`: bajar umbral EAR
- `r`: recalibrar baseline

Controles equivalentes por API:

```bash
curl -X POST "http://127.0.0.1:8001/somnolence/threshold?delta=0.01"
curl -X POST "http://127.0.0.1:8001/somnolence/threshold?delta=-0.01"
curl -X POST "http://127.0.0.1:8001/somnolence/reset"
```

## 3) Levantar TF Service (entorno tf)

```bash
uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002 --reload
```

### Probar endpoint de somnolencia visual fusionada

```bash
curl -X POST "http://127.0.0.1:8002/predict/drowsiness_image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "db/artificialvision/generalcontext-emotions2/happy/happy_01932.jpg",
    "vision_state": "ATTENTIVE",
    "avg_ear": 0.24,
    "closed_duration": 0.1,
    "threshold": 0.19
  }'
```

## 4) Monitor en tiempo real (face + tf)

Listar cámaras disponibles (interna/iPhone, etc.):

```bash
python -m mostacho.orchestrator.realtime_monitor --list-cameras --camera-backend AUTO --max-camera-index 12
```

Ejecutar monitor eligiendo índice:

```bash
python -m mostacho.orchestrator.realtime_monitor --camera-backend AUTO --camera-index 0
```

## 5) Probar inferencia integrada

```bash
python -m mostacho.orchestrator.runner \
  --image db/artificialvision/generalcontext-emotions2/happy/happy_01932.jpg \
  --audio db/audio/crema-d/1008_TAI_HAP_XX.wav \
  --biometrics "db/biometrics/stress/dataset/1. processed/eda/wesad/raw/chest/S2.csv"
```

## 6) Entrenar modelos base

```bash
python -m mostacho.training.train_vision_emotion
python -m mostacho.training.train_drowsiness_vision
python -m mostacho.training.train_voice_emotion
python -m mostacho.training.train_biometrics_stress
python -m mostacho.training.train_multimodal_fusion
```

## Nota

Cada script permite ajustar rutas y tamaños de muestra por argumentos CLI.
