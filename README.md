# Mostacho

Arquitectura multimodal para detección de somnolencia/estrés/emoción del conductor usando:

- **Visión**: InsightFace (detección facial y landmarks)
- **Voz**: extracción de features desde micrófono/audio (CREMA-D)
- **Biométricos**: señales EDA/HRV desde wearables (WESAD + stress)
- **Fusión neuronal**: TensorFlow para combinar modalidades

La base visual de legacy (detección facial, contorno de landmarks de ojos y EAR para somnolencia) está adaptada en:

- `mostacho.vision`
- `mostacho.services.face_api`

## Objetivo técnico

Este proyecto está diseñado para colaborar en GitHub con equipos de **macOS** y **Windows 11** sin conflictos de dependencias entre InsightFace y TensorFlow.

Por eso se separa en procesos:

1. `face_service` (entorno InsightFace)
2. `tf_service` (entorno TensorFlow)
3. `orchestrator` (cliente que une todo)

## Estructura

```text
mostacho/
  artifacts/                 # modelos e índices generados
  docs/                      # arquitectura y datasets
  requirements/              # entornos separados
  scripts/                   # setup/run macOS y Windows
  src/mostacho/
    data/                    # indexado/carga de datasets
    features/                # extracción de features (voz/biométricos)
    services/                # APIs de InsightFace y TensorFlow
    orchestrator/            # cliente de inferencia multimodal
    training/                # scripts de entrenamiento
```

## Datasets ya contemplados

- Visión (estructura actual):
  - `db/artificialvision/generalcontext-emotions2`
  - `db/artificialvision/generalcontext-emotions`
  - `db/artificialvision/DrivFace`
  - `db/artificialvision/train` (Open/Closed Eyes)
- Visión (fallback legacy):
  - `db/processed_data`, `db/Data`, `db/DrivFace`
- Voz: `db/audio/crema-d`
- Biométricos:
  - `db/biometrics/WESAD`
  - `db/biometrics/stress/dataset`

## Instalación recomendada (dos entornos)

### 1) Entorno de InsightFace

```bash
python -m venv ~/venvs/mostacho_face
source ~/venvs/mostacho_face/bin/activate
pip install -r mostacho/requirements/face_service.txt
pip install -e mostacho
```

### 2) Entorno de TensorFlow

```bash
python -m venv ~/venvs/mostacho_tf
source ~/venvs/mostacho_tf/bin/activate
pip install -r mostacho/requirements/tf_service.txt
pip install -e mostacho
```

### 3) Entorno del orquestador (opcional)

```bash
python -m venv ~/venvs/mostacho_orch
source ~/venvs/mostacho_orch/bin/activate
pip install -r mostacho/requirements/orchestrator.txt
pip install -e mostacho
```

## Flujo de ejecución

1. Levanta `face_service` en el entorno face.
2. Levanta `tf_service` en el entorno tf.
3. Ejecuta `orchestrator` para enviar features multimodales a `tf_service`.
4. (Opcional) Ejecuta monitor en vivo que fusiona `face_service` + `tf_service`.

Comandos detallados en `docs/ARCHITECTURE.md` y `docs/RUNBOOK.md`.

Monitor en vivo:

```bash
python -m mostacho.orchestrator.realtime_monitor --camera-backend AUTO
```

## Ejecutar base visual (sin TensorFlow)

En el entorno `face_service` puedes ejecutar solo visión en tiempo real, equivalente a la base legacy (ya no disponible):

```bash
python -m mostacho.vision.realtime --camera-backend AUTO
```

Controles:
- `q`: salir
- `u`: subir umbral EAR
- `j`: bajar umbral EAR
- `r`: recalibrar baseline EAR
