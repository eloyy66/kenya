import time
from collections import deque

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =======================
# Configuración
# =======================
EAR_THRESHOLD = 0.22
CLOSED_SECONDS = 3.0
WINDOW_SIZE = 5  # suavizado de EAR (frames)
DETECT_SIZE = (320, 320)  # menor = más FPS
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CALIBRATION_SECONDS = 2.0
THRESHOLD_OFFSET = 0.04  # se resta al baseline
MIN_THRESHOLD = 0.15


# =======================
# Utilidades EAR
# =======================

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_points):
    # eye_points: array de 6 puntos (x, y)
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


def get_eye_landmarks_from_face(face):
    # Preferimos 68 puntos porque el índice de ojos es estándar (dlib).
    kps_68 = getattr(face, "landmark_3d_68", None)
    if kps_68 is not None and kps_68.shape[0] >= 68:
        kps = kps_68[:, :2]  # usar x,y
        left_eye_idx = [36, 37, 38, 39, 40, 41]
        right_eye_idx = [42, 43, 44, 45, 46, 47]
        left_eye = np.array([kps[i] for i in left_eye_idx])
        right_eye = np.array([kps[i] for i in right_eye_idx])
        return left_eye, right_eye

    # Fallback 106 (si está disponible)
    kps_106 = getattr(face, "landmark_2d_106", None)
    if kps_106 is None or kps_106.shape[0] < 98:
        return None, None

    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]
    left_eye = np.array([kps_106[i] for i in left_eye_idx])
    right_eye = np.array([kps_106[i] for i in right_eye_idx])
    return left_eye, right_eye


# =======================
# Main
# =======================

def main():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"],
    )
    app.prepare(ctx_id=-1, det_size=DETECT_SIZE)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ear_window = deque(maxlen=WINDOW_SIZE)
    closed_start = None
    prev_time = 0.0
    calibration_start = None
    calibration_ears = []
    baseline_ear = None
    manual_threshold = None

    print("Presiona 'q' para salir")

    # Calentamiento de cámara
    for _ in range(10):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Opcional: reducir resolución para mayor FPS
        if FRAME_WIDTH and FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        faces = app.get(frame)
        state = "NO_FACE"
        ear_value = None
        closed_duration = 0.0

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            left_eye, right_eye = get_eye_landmarks_from_face(face)
            if left_eye is None:
                state = "NO_LANDMARKS_106"
                break

            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            ear_value = (ear_left + ear_right) / 2.0
            ear_window.append(ear_value)
            avg_ear = float(np.mean(ear_window))

            # Calibración automática con ojos abiertos (primeros segundos)
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

            if manual_threshold is not None:
                dynamic_threshold = manual_threshold
            else:
                dynamic_threshold = max(MIN_THRESHOLD, baseline_ear - THRESHOLD_OFFSET)

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

            # Métricas
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
            cv2.putText(
                frame,
                f"Closed: {closed_duration:.1f}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            break  # Solo un rostro (conductor)

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        # Estado visual
        color = (0, 255, 0)
        if state == "EYES_CLOSED":
            color = (0, 165, 255)
        elif state == "SOMNOLENT":
            color = (0, 0, 255)

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

        cv2.imshow("Kenya Vision - Somnolencia 3s", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
