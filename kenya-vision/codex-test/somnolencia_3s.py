import time  # manejo de tiempos
from collections import deque  # cola de tamaño fijo para suavizar

import cv2  # OpenCV para video y dibujo
import numpy as np  # cálculos numéricos
from insightface.app import FaceAnalysis  # modelo de InsightFace

# =======================
# Configuración
# =======================
EAR_THRESHOLD = 0.22  # umbral base (referencia)
CLOSED_SECONDS = 3.0  # segundos con ojos cerrados para somnolencia
WINDOW_SIZE = 5  # suavizado de EAR (frames)
DETECT_SIZE = (320, 320)  # resolución de detección (más bajo = más FPS)
FRAME_WIDTH = 640  # ancho de captura
FRAME_HEIGHT = 360  # alto de captura
CALIBRATION_SECONDS = 2.0  # segundos para calibrar con ojos abiertos
THRESHOLD_OFFSET = 0.04  # resta al baseline para crear umbral dinámico
MIN_THRESHOLD = 0.15  # mínimo permitido para el umbral


# =======================
# Utilidades EAR
# =======================

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)  # distancia euclidiana


def compute_ear(eye_points):
    # eye_points: 6 puntos (x, y) del ojo
    A = euclidean(eye_points[1], eye_points[5])  # distancia vertical 1
    B = euclidean(eye_points[2], eye_points[4])  # distancia vertical 2
    C = euclidean(eye_points[0], eye_points[3])  # distancia horizontal
    return (A + B) / (2.0 * C)  # Eye Aspect Ratio


def get_eye_landmarks_from_face(face):
    # Preferimos 68 puntos porque el índice de ojos es estándar (dlib)
    kps_68 = getattr(face, "landmark_3d_68", None)  # landmarks 68
    if kps_68 is not None and kps_68.shape[0] >= 68:  # verificar tamaño
        kps = kps_68[:, :2]  # usar solo x,y
        left_eye_idx = [36, 37, 38, 39, 40, 41]  # ojo izquierdo
        right_eye_idx = [42, 43, 44, 45, 46, 47]  # ojo derecho
        left_eye = np.array([kps[i] for i in left_eye_idx])  # puntos izq
        right_eye = np.array([kps[i] for i in right_eye_idx])  # puntos der
        return left_eye, right_eye  # devolver puntos

    # Fallback 106 (si está disponible)
    kps_106 = getattr(face, "landmark_2d_106", None)  # landmarks 106
    if kps_106 is None or kps_106.shape[0] < 98:  # validar existencia
        return None, None  # no hay landmarks suficientes

    left_eye_idx = [33, 35, 37, 39, 41, 43]  # ojo izquierdo 106
    right_eye_idx = [87, 89, 91, 93, 95, 97]  # ojo derecho 106
    left_eye = np.array([kps_106[i] for i in left_eye_idx])  # puntos izq
    right_eye = np.array([kps_106[i] for i in right_eye_idx])  # puntos der
    return left_eye, right_eye  # devolver puntos


# =======================
# Main
# =======================

def main():
    app = FaceAnalysis(  # inicializar InsightFace
        name="buffalo_l",  # modelo preentrenado
        providers=["CPUExecutionProvider"],  # correr en CPU
        allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"],  # módulos
    )
    app.prepare(ctx_id=-1, det_size=DETECT_SIZE)  # preparar modelo

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # abrir cámara (Mac)
    if not cap.isOpened():  # validar apertura
        print("Error: no se pudo abrir la cámara")  # mensaje de error
        return  # salir
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  # fijar ancho
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)  # fijar alto

    ear_window = deque(maxlen=WINDOW_SIZE)  # ventana para suavizar EAR
    closed_start = None  # inicio de ojos cerrados
    prev_time = 0.0  # para FPS
    calibration_start = None  # inicio de calibración
    calibration_ears = []  # lista de EAR durante calibración
    baseline_ear = None  # EAR base con ojos abiertos
    manual_threshold = None  # umbral manual

    print("Presiona 'q' para salir")  # instrucción

    # Calentamiento de cámara
    for _ in range(10):  # leer algunos frames iniciales
        cap.read()  # descartar frame

    while True:  # loop principal
        ret, frame = cap.read()  # capturar frame
        if not ret:  # validar captura
            break  # salir si falla

        # Opcional: reducir resolución para mayor FPS
        if FRAME_WIDTH and FRAME_HEIGHT:  # validar tamaños
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # resize

        faces = app.get(frame)  # detectar rostros
        state = "NO_FACE"  # estado por defecto
        ear_value = None  # EAR actual
        closed_duration = 0.0  # duración de ojos cerrados

        for face in faces:  # iterar rostros
            x1, y1, x2, y2 = face.bbox.astype(int)  # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # dibujar

            left_eye, right_eye = get_eye_landmarks_from_face(face)  # ojos
            if left_eye is None:  # sin landmarks
                state = "NO_LANDMARKS_106"  # estado
                break  # salir del loop de rostros

            for (x, y) in left_eye:  # dibujar puntos ojo izq
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)  # punto
            for (x, y) in right_eye:  # dibujar puntos ojo der
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)  # punto

            ear_left = compute_ear(left_eye)  # EAR ojo izq
            ear_right = compute_ear(right_eye)  # EAR ojo der
            ear_value = (ear_left + ear_right) / 2.0  # EAR promedio
            ear_window.append(ear_value)  # guardar en ventana
            avg_ear = float(np.mean(ear_window))  # promedio suavizado

            # Calibración automática con ojos abiertos (primeros segundos)
            if baseline_ear is None:  # si aún no calibró
                if calibration_start is None:  # iniciar tiempo
                    calibration_start = time.monotonic()  # tiempo actual
                calibration_ears.append(avg_ear)  # agregar EAR
                if (time.monotonic() - calibration_start) >= CALIBRATION_SECONDS and len(calibration_ears) >= 5:
                    baseline_ear = float(np.median(calibration_ears))  # baseline
                state = "CALIBRATING"  # estado
                closed_start = None  # reset de cerrado
                closed_duration = 0.0  # reset duración
                break  # esperar siguiente frame

            if manual_threshold is not None:  # si usuario ajustó
                dynamic_threshold = manual_threshold  # usar manual
            else:
                dynamic_threshold = max(MIN_THRESHOLD, baseline_ear - THRESHOLD_OFFSET)  # umbral dinámico

            if avg_ear < dynamic_threshold:  # ojos cerrados
                if closed_start is None:  # iniciar conteo
                    closed_start = time.monotonic()  # tiempo de inicio
                closed_duration = time.monotonic() - closed_start  # duración
                if closed_duration >= CLOSED_SECONDS:  # si supera 3s
                    state = "SOMNOLENT"  # somnolencia
                else:
                    state = "EYES_CLOSED"  # cerrados pero no 3s
            else:
                closed_start = None  # reset
                state = "ATTENTIVE"  # atento

            # Métricas en pantalla
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Closed: {closed_duration:.1f}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"THR: {dynamic_threshold:.2f}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            if baseline_ear is not None:
                cv2.putText(
                    frame,
                    f"BASE: {baseline_ear:.2f}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            break  # solo un rostro (conductor)

        # FPS
        current_time = time.time()  # tiempo actual
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0  # cálculo
        prev_time = current_time  # actualizar

        # Estado visual
        color = (0, 255, 0)  # color por defecto
        if state == "EYES_CLOSED":  # ojos cerrados
            color = (0, 165, 255)  # naranja
        elif state == "SOMNOLENT":  # somnolencia
            color = (0, 0, 255)  # rojo

        cv2.putText(
            frame,
            f"STATE: {state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Kenya Vision - Somnolencia 3s", frame)  # mostrar ventana

        key = cv2.waitKey(1) & 0xFF  # leer tecla
        if key == ord("q"):  # salir
            break
        if key == ord("u"):  # subir umbral
            if manual_threshold is None:
                manual_threshold = max(MIN_THRESHOLD, (baseline_ear or 0.2) - THRESHOLD_OFFSET)
            manual_threshold = min(0.35, manual_threshold + 0.01)
        if key == ord("j"):  # bajar umbral
            if manual_threshold is None:
                manual_threshold = max(MIN_THRESHOLD, (baseline_ear or 0.2) - THRESHOLD_OFFSET)
            manual_threshold = max(0.10, manual_threshold - 0.01)
        if key == ord("r"):  # recalibrar
            calibration_start = None
            calibration_ears = []
            baseline_ear = None
            manual_threshold = None

    cap.release()  # liberar cámara
    cv2.destroyAllWindows()  # cerrar ventanas


if __name__ == "__main__":
    main()  # ejecutar
