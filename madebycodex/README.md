# Kenya Vision (madebycodex)

Estructura modular para deteccion de somnolencia con InsightFace y espacio para fusion multimodal.

## Carpetas
- `camera/`: captura de video
- `facedetection/`: wrapper de InsightFace
- `landmarks/`: extraccion de ojos + EAR
- `distraction_logic/`: somnolencia + fusion multimodal
- `voice/`: modulo de voz (placeholder)
- `biometrics/`: modulo de biometria (placeholder)

## Ejecutar
```bash
cd /Users/usuario/Desktop/kenya
source venv/bin/activate
python madebycodex/main.py
```

## Controles
- `q`: salir
- `u`: subir umbral
- `j`: bajar umbral
- `r`: recalibrar

## Configuracion
Ajusta valores en `config.py`.
