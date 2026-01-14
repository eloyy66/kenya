import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def main():
    # Inicializar InsightFace (CPU)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Abrir webcam (0 suele ser la cámara por defecto)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        return

    prev_time = 0.0

    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # InsightFace trabaja en BGR (OpenCV default)
        faces = app.get(frame)

        for face in faces:
            # Bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Landmarks (106 puntos)
            if face.kps is not None:
                for (x, y) in face.kps:
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        1,
                        (0, 0, 255),
                        -1
                    )

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow("Kenya Vision - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Este es un código funcional de solo detección facial con insightface y opencv2
"""
