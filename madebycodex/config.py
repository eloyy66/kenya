"""Configuracion global del proyecto Kenya Vision (madebycodex)."""

# Camara
CAMERA_INDEX = 0
CAMERA_BACKEND = "AVFOUNDATION"  # "AVFOUNDATION" en macOS, "DEFAULT" en otros
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# InsightFace
DETECT_SIZE = (320, 320)

# Somnolencia (EAR)
WINDOW_SIZE = 5
CLOSED_SECONDS = 3.0
CALIBRATION_SECONDS = 2.0
THRESHOLD_OFFSET = 0.04
MIN_THRESHOLD = 0.15

# Umbral de referencia (no se usa si hay calibracion)
EAR_THRESHOLD = 0.22

# Fusion multimodal
FUSION_WEIGHTS = {"vision": 0.6, "voice": 0.2, "biometrics": 0.2}
FUSION_MIN_CONFIDENCE = 0.3
