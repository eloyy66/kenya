# Arquitectura Mostacho

## Principio clave: separación por procesos

`insightface` y `tensorflow` pueden entrar en conflicto por `numpy` y `ml-dtypes`.
Para evitarlo de forma robusta y multiplataforma, el sistema se divide en servicios:

- **Face Service** (`mostacho.services.face_api`)
  - Dependencias: InsightFace + ONNX Runtime + OpenCV
  - Entrada: frame o imagen
  - Salida: bbox, score, landmarks, contorno de ojos, EAR y features visuales agregados
  - Control runtime: recalibración y ajuste de umbral de somnolencia

- **TF Service** (`mostacho.services.tf_api`)
  - Dependencias: TensorFlow + utilidades de entrenamiento
  - Entrada: features de visión/voz/biométricos
  - Salida: estado emocional/estrés/somnolencia + confianza

- **Orchestrator** (`mostacho.orchestrator.runner`)
  - Dependencias ligeras
  - Toma cámara/audio/biométricos
  - Llama a ambos servicios y unifica salida en tiempo real

## Beneficios

- Sin choques de dependencias en el mismo entorno
- Equipo mixto macOS + Windows 11 funciona igual
- Deploy modular (local, edge, cloud)

## Rutas de datos

1. Cámara -> `face_service`
2. Micrófono/archivo -> extractor de voz
3. Wearable/CSV -> extractor biométrico
4. Features fusionadas -> `tf_service`
5. Resultado final -> UI / alertas en cabina
