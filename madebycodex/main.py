"""Entry point del prototipo Kenya Vision (madebycodex)."""

import time
import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_BACKEND,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DETECT_SIZE,
    WINDOW_SIZE,
    CLOSED_SECONDS,
    CALIBRATION_SECONDS,
    THRESHOLD_OFFSET,
    MIN_THRESHOLD,
    FUSION_WEIGHTS,
    FUSION_MIN_CONFIDENCE,
)
from camera.capture import Camera
from facedetection.insightface_detector import InsightFaceDetector
from landmarks.eyes import compute_ear, get_eye_landmarks_from_face
from distraction_logic.somnolence import SomnolenceDetector
from distraction_logic.fusion import MultimodalFusion
from voice.stub import VoiceAnalyzer
from biometrics.stub import BiometricsAnalyzer


def main():
    # 1) Inicializar detector y camara
    detector = InsightFaceDetector(detect_size=DETECT_SIZE)
    camera = Camera(
        index=CAMERA_INDEX,
        backend=CAMERA_BACKEND,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
    )

    if not camera.open():
        print("Error: no se pudo abrir la camara")
        return
    camera.warmup()

    # 2) Inicializar logica de somnolencia
    somnolence = SomnolenceDetector(
        window_size=WINDOW_SIZE,
        closed_seconds=CLOSED_SECONDS,
        calibration_seconds=CALIBRATION_SECONDS,
        threshold_offset=THRESHOLD_OFFSET,
        min_threshold=MIN_THRESHOLD,
    )
    fusion = MultimodalFusion(weights=FUSION_WEIGHTS, min_confidence=FUSION_MIN_CONFIDENCE)
    voice = VoiceAnalyzer()
    bio = BiometricsAnalyzer()

    print("Presiona 'q' para salir | 'u' subir umbral | 'j' bajar umbral | 'r' recalibrar")

    prev_time = 0.0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Resize para consistencia
        if FRAME_WIDTH and FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        faces = detector.detect(frame)
        state = "NO_FACE"
        avg_ear = None
        closed_duration = 0.0
        threshold = None
        baseline = None
        vision_state = "NO_FACE"
        fused_state = "UNKNOWN"
        fused_confidence = 0.0

        for face in faces:
            # Bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Landmarks ojos
            left_eye, right_eye = get_eye_landmarks_from_face(face)
            if left_eye is None:
                state = "NO_LANDMARKS"
                break

            # Dibujar ojos
            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

            # EAR
            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            ear_value = (ear_left + ear_right) / 2.0

            result = somnolence.update(ear_value, now=time.monotonic())
            vision_state = result["state"]
            avg_ear = result["avg_ear"]
            closed_duration = result["closed_duration"]
            threshold = result["threshold"]
            baseline = result["baseline"]

            voice_result = voice.update()
            bio_result = bio.update()

            fused = fusion.fuse(
                {
                    "vision": result,
                    "voice": voice_result,
                    "biometrics": bio_result,
                }
            )
            fused_state = fused["state"]
            fused_confidence = fused["confidence"]

            # Si la fusion no tiene suficiente confianza, usar vision como respaldo
            state = fused_state if fused_state != "UNKNOWN" else vision_state

            # Mostrar metricas
            if avg_ear is not None:
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Closed: {closed_duration:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if threshold is not None:
                cv2.putText(frame, f"THR: {threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if baseline is not None:
                cv2.putText(frame, f"BASE: {baseline:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"VISION: {vision_state}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
            cv2.putText(frame, f"FUSED: {fused_state} ({fused_confidence:.2f})", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

            break  # solo un rostro

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

        cv2.putText(frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Kenya Vision - madebycodex", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            somnolence.adjust_threshold(0.01)
        if key == ord("j"):
            somnolence.adjust_threshold(-0.01)
        if key == ord("r"):
            somnolence.reset_calibration()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
