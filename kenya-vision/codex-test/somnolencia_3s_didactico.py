"""
Kenya Vision - Somnolencia 3s (versión didáctica paso a paso)

Objetivo:
- Detectar somnolencia cuando los ojos permanecen cerrados >= 3 segundos.
- Usar InsightFace para detectar rostro y landmarks.
- Calcular EAR (Eye Aspect Ratio) para estimar si los ojos están abiertos/cerrados.

Cómo leer este archivo:
1) Configuración (parámetros ajustables).
2) Utilidades matemáticas (distancia, EAR).
3) Extracción de landmarks (ojos) desde InsightFace.
4) Lógica de calibración (baseline con ojos abiertos).
5) Lógica principal de detección en bucle de video.
6) Controles de teclado para ajustar umbral en vivo.
"""

# =======================
# 1) IMPORTS
# =======================
import time  # Para medir tiempos de ojos cerrados y FPS
from collections import deque  # Para suavizar el EAR con una ventana

import cv2  # OpenCV: cámara, dibujar en pantalla, ventanas
import numpy as np  # Numpy: operaciones numéricas
from insightface.app import FaceAnalysis  # InsightFace: detección + landmarks


# =======================
# 2) CONFIGURACIÓN (AJUSTA AQUÍ)
# =======================
EAR_THRESHOLD = 0.22  # (Referencia) umbral base si no hubiera calibración
CLOSED_SECONDS = 3.0  # Segundos con ojos cerrados para declarar somnolencia
WINDOW_SIZE = 5  # Tamaño de ventana para suavizar EAR
DETECT_SIZE = (320, 320)  # Resolución interna del detector (baja = más FPS)
FRAME_WIDTH = 640  # Ancho de captura
FRAME_HEIGHT = 360  # Alto de captura
CALIBRATION_SECONDS = 2.0  # Tiempo para calibrar baseline con ojos abiertos
THRESHOLD_OFFSET = 0.04  # Resta al baseline para crear umbral dinámico
MIN_THRESHOLD = 0.15  # Umbral mínimo permitido


# =======================
# 3) UTILIDADES MATEMÁTICAS
# =======================

def euclidean(p1, p2):
    """
    Calcula distancia euclidiana entre dos puntos 2D.
    p1, p2: np.array([x, y])
    """
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_points):
    """
    Calcula Eye Aspect Ratio (EAR).

    eye_points: 6 puntos del contorno del ojo en orden:
      p1---p2---p3
      |         |
      p6---p5---p4

    Fórmula EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    """
    A = euclidean(eye_points[1], eye_points[5])  # vertical 1
    B = euclidean(eye_points[2], eye_points[4])  # vertical 2
    C = euclidean(eye_points[0], eye_points[3])  # horizontal
    return (A + B) / (2.0 * C)


# =======================
# 4) LANDMARKS DE OJOS
# =======================

def get_eye_landmarks_from_face(face):
    """
    Extrae los 6 puntos de cada ojo.

    InsightFace puede devolver:
    - face.landmark_3d_68: 68 puntos (índices estándar tipo dlib)
    - face.landmark_2d_106: 106 puntos (índices propios)

    Preferimos 68 puntos porque los índices de ojos son estándar.
    Si no están disponibles, usamos 106 como fallback.
    """
    # --- Caso 68 puntos ---
    kps_68 = getattr(face, "landmark_3d_68", None)
    if kps_68 is not None and kps_68.shape[0] >= 68:
        kps = kps_68[:, :2]  # Solo x,y
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        right_eye_idx = [42, 43, 44, 45, 46, 47]
        left_eye = np.array([kps[i] for i in left_eye_idx])
        right_eye = np.array([kps[i] for i in right_eye_idx])
        return left_eye, right_eye

    # --- Caso 106 puntos ---
    kps_106 = getattr(face, "landmark_2d_106", None)
    if kps_106 is None or kps_106.shape[0] < 98:
        return None, None

    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]
    left_eye = np.array([kps_106[i] for i in left_eye_idx])
    right_eye = np.array([kps_106[i] for i in right_eye_idx])
    return left_eye, right_eye


# =======================
# 5) FUNCIÓN PRINCIPAL
# =======================

def main():
    """
    Paso a paso:
    1) Inicializar InsightFace.
    2) Abrir cámara.
    3) Calibrar EAR baseline con ojos abiertos.
    4) En cada frame: detectar rostro, calcular EAR.
    5) Si EAR < umbral por >= 3s => SOMNOLENT.
    """

    # 1) Inicializar InsightFace con módulos mínimos
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"],
    )
    app.prepare(ctx_id=-1, det_size=DETECT_SIZE)

    # 2) Abrir cámara
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 3) Variables de estado
    ear_window = deque(maxlen=WINDOW_SIZE)  # Suavizado
    closed_start = None  # Momento en el que se cierran los ojos
    prev_time = 0.0  # Para FPS

    # Calibración
    calibration_start = None
    calibration_ears = []
    baseline_ear = None

    # Umbral manual
    manual_threshold = None

    print("Presiona 'q' para salir | 'u' subir umbral | 'j' bajar umbral | 'r' recalibrar")

    # Calentar cámara (descartar frames iniciales)
    for _ in range(10):
        cap.read()

    # 4) Bucle principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir resolución si se configuró (mejor FPS)
        if FRAME_WIDTH and FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Detectar rostros
        faces = app.get(frame)
        state = "NO_FACE"
        closed_duration = 0.0

        # Procesar el primer rostro (conductor)
        for face in faces:
            # Dibujar bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Obtener puntos de ojos
            left_eye, right_eye = get_eye_landmarks_from_face(face)
            if left_eye is None:
                state = "NO_LANDMARKS"
                break

            # Dibujar puntos de ojos
            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

            # Calcular EAR
            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            ear_value = (ear_left + ear_right) / 2.0
            ear_window.append(ear_value)
            avg_ear = float(np.mean(ear_window))

            # 4.1) Calibración con ojos abiertos
            if baseline_ear is None:
                if calibration_start is None:
                    calibration_start = time.monotonic()
                calibration_ears.append(avg_ear)

                if (time.monotonic() - calibration_start) >= CALIBRATION_SECONDS and len(calibration_ears) >= 5:
                    baseline_ear = float(np.median(calibration_ears))

                state = "CALIBRATING"
                closed_start = None
                closed_duration = 0.0
                break

            # 4.2) Umbral dinámico
            if manual_threshold is not None:
                dynamic_threshold = manual_threshold
            else:
                dynamic_threshold = max(MIN_THRESHOLD, baseline_ear - THRESHOLD_OFFSET)

            # 4.3) Decidir estado
            if avg_ear < dynamic_threshold:
                if closed_start is None:
                    closed_start = time.monotonic()
                closed_duration = time.monotonic() - closed_start

                if closed_duration >= CLOSED_SECONDS:
                    state = "SOMNOLENT"
                else:
                    state = "EYES_CLOSED"
            else:
                closed_start = None
                state = "ATTENTIVE"

            # 4.4) Mostrar métricas
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Closed: {closed_duration:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"THR: {dynamic_threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if baseline_ear is not None:
                cv2.putText(frame, f"BASE: {baseline_ear:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            break  # Solo un rostro

        # 5) FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        # 6) Mostrar estado general
        color = (0, 255, 0)
        if state == "EYES_CLOSED":
            color = (0, 165, 255)
        elif state == "SOMNOLENT":
            color = (0, 0, 255)

        cv2.putText(frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 7) Mostrar ventana
        cv2.imshow("Kenya Vision - Somnolencia 3s (Didáctico)", frame)

        # 8) Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            if manual_threshold is None:
                manual_threshold = max(MIN_THRESHOLD, (baseline_ear or 0.2) - THRESHOLD_OFFSET)
            manual_threshold = min(0.35, manual_threshold + 0.01)
        if key == ord("j"):
            if manual_threshold is None:
                manual_threshold = max(MIN_THRESHOLD, (baseline_ear or 0.2) - THRESHOLD_OFFSET)
            manual_threshold = max(0.10, manual_threshold - 0.01)
        if key == ord("r"):
            calibration_start = None
            calibration_ears = []
            baseline_ear = None
            manual_threshold = None

    # 9) Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


# =======================
# 6) EJECUCIÓN
# =======================

if __name__ == "__main__":
    main()
