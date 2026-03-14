# Datasets soportados

## Visión artificial

- Principal (estructura actual):
  - `db/artificialvision/generalcontext-emotions2/<clase>/*.jpg`
  - `db/artificialvision/generalcontext-emotions/<clase>/*`
  - `db/artificialvision/DrivFace/DrivImages/*`
  - `db/artificialvision/train/{Open_Eyes,Closed_Eyes}/*`
- Fallback legacy:
  - `db/processed_data/<clase>/*.jpg`
  - `db/Data/<clase>/*`
  - `db/DrivFace/DrivImages/*`

## Voz (CREMA-D)

- `db/audio/crema-d/*.wav`
- Emoción codificada en nombre de archivo:
  - `ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`

## Biométricos

- WESAD bruto: `db/biometrics/WESAD/S*/S*.pkl`, `S*_quest.csv`
- Stress dataset procesado/intermedio:
  - `db/biometrics/stress/dataset/1. processed/eda/...`
  - `db/biometrics/stress/dataset/1. processed/hrv/...`
  - `db/biometrics/stress/dataset/0. interim/...`

## Recomendación de ingeniería

Crear un `dataset_index.json` versionado por fecha para trazabilidad del entrenamiento.
