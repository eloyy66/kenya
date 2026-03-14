"""Runner en tiempo real (webcam) para la base visual de Mostacho."""

from __future__ import annotations

# argparse para CLI local.
import argparse
# time para cálculo de FPS.
import time
# os para leer variables de entorno.
import os
# typing para tipos explícitos.
from typing import Any

# numpy para manejo básico de arrays.
import numpy as np

# cámara multiplataforma y runtime visual.
from mostacho.vision.camera import Camera, list_available_cameras
from mostacho.vision.runtime import VisionRuntime, VisionRuntimeConfig


def _load_cv2() -> Any:
    """Carga OpenCV de forma diferida para ejecución local."""

    # Import local de OpenCV.
    import cv2  # type: ignore

    # Retorna módulo cargado.
    return cv2


def _create_face_app(detect_size: tuple[int, int]) -> Any:
    """Crea instancia FaceAnalysis para ejecución local en CPU."""

    # Import local de InsightFace.
    from insightface.app import FaceAnalysis  # type: ignore

    model_name = os.getenv("MOSTACHO_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
    providers_raw = os.getenv("MOSTACHO_FACE_PROVIDERS", "CPUExecutionProvider")
    providers = [value.strip() for value in providers_raw.split(",") if value.strip()]
    landmarks_raw = os.getenv("MOSTACHO_FACE_LANDMARKS", "68").strip().lower()
    if landmarks_raw in {"106", "2d106", "landmark_2d_106", "l106"}:
        allowed_modules = ["detection", "landmark_2d_106"]
    elif landmarks_raw in {"both", "all", "68+106", "106+68"}:
        allowed_modules = ["detection", "landmark_3d_68", "landmark_2d_106"]
    else:
        allowed_modules = ["detection", "landmark_3d_68"]

    # Se crea app con módulos necesarios para detección y landmarks.
    try:
        face_app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules,
        )
    except Exception:
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=allowed_modules,
        )
    # Se prepara detector en CPU.
    face_app.prepare(ctx_id=-1, det_size=detect_size)
    return face_app


def _draw_overlay(frame: np.ndarray, analysis: Any, fps: float) -> np.ndarray:
    """Dibuja overlays de bbox, landmarks de ojos, estado y métricas."""

    # Se carga OpenCV.
    cv2 = _load_cv2()

    # Copia de frame para dibujar encima sin alterar referencia original.
    output = frame.copy()

    # Dibuja cada rostro detectado.
    for detection in analysis.detections:
        # Bounding box principal del rostro.
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Landmarks faciales generales (si existen).
        for x, y in detection.landmarks:
            cv2.circle(output, (int(x), int(y)), 1, (255, 180, 0), -1)

        # Contorno de ojo izquierdo.
        for x, y in detection.left_eye:
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 255), -1)

        # Contorno de ojo derecho.
        for x, y in detection.right_eye:
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 255), -1)

    # Color por estado visual.
    color = (0, 255, 0)
    if analysis.vision_state == "EYES_CLOSED":
        color = (0, 165, 255)
    elif analysis.vision_state == "SOMNOLENT":
        color = (0, 0, 255)

    # Texto principal de estado visual.
    cv2.putText(output, f"STATE: {analysis.vision_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Texto de EAR si existe.
    if analysis.avg_ear is not None:
        cv2.putText(output, f"EAR: {analysis.avg_ear:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Duración de ojos cerrados.
    cv2.putText(output, f"Closed: {analysis.closed_duration:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Umbral actual si existe.
    if analysis.threshold is not None:
        cv2.putText(output, f"THR: {analysis.threshold:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Baseline calibrado si existe.
    if analysis.baseline is not None:
        cv2.putText(output, f"BASE: {analysis.baseline:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # FPS en tiempo real.
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    # Se retorna frame anotado.
    return output


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de ejecución en tiempo real."""

    # Parser CLI de runner local.
    parser = argparse.ArgumentParser(description="Mostacho Vision Realtime")
    # Índice de cámara.
    parser.add_argument("--camera-index", type=int, default=0, help="Indice de camara")
    # Backend de cámara (AUTO recomendado).
    parser.add_argument("--camera-backend", type=str, default="AUTO", help="AUTO|AVFOUNDATION|DSHOW|MSMF|V4L2")
    # Ancho de frame.
    parser.add_argument("--width", type=int, default=640, help="Ancho de captura")
    # Alto de frame.
    parser.add_argument("--height", type=int, default=360, help="Alto de captura")
    # Tamaño de detección de InsightFace.
    parser.add_argument("--detect-width", type=int, default=256, help="Ancho de deteccion")
    # Tamaño de detección de InsightFace.
    parser.add_argument("--detect-height", type=int, default=256, help="Alto de deteccion")
    # Ejecuta detección completa cada N frames (reutiliza último análisis entre medias).
    parser.add_argument("--detect-every", type=int, default=1, help="Frames entre detecciones completas")
    # Lista cámaras detectadas y termina.
    parser.add_argument("--list-cameras", action="store_true", help="Lista cámaras disponibles y sale")
    # Índice máximo a probar al listar cámaras.
    parser.add_argument("--max-camera-index", type=int, default=10, help="Indice maximo a probar al listar cámaras")
    # Retorna argumentos parseados.
    return parser.parse_args()


def main() -> None:
    """Ejecuta pipeline visual local equivalente a la base de madebycodex."""

    # Carga argumentos de CLI.
    args = parse_args()

    # Modo descubrimiento de cámaras sin iniciar inferencia.
    if args.list_cameras:
        cameras = list_available_cameras(
            max_index=args.max_camera_index,
            backend=args.camera_backend,
            width=args.width,
            height=args.height,
        )
        if not cameras:
            print("No se detectaron camaras en el rango solicitado.")
            return
        print("Camaras detectadas:")
        for camera_index, backend_name in cameras:
            print(f"  index={camera_index} backend={backend_name}")
        return

    # Carga OpenCV para captura y ventana.
    cv2 = _load_cv2()

    # Configuración del runtime visual.
    config = VisionRuntimeConfig(detect_size=(args.detect_width, args.detect_height))
    # Inicializa detector InsightFace.
    face_app = _create_face_app(config.detect_size)
    # Inicializa runtime de análisis visual.
    runtime = VisionRuntime(face_app=face_app, config=config)

    # Inicializa cámara con backend AUTO o explícito.
    camera = Camera(index=args.camera_index, backend=args.camera_backend, width=args.width, height=args.height)
    if not camera.open():
        raise RuntimeError("No se pudo abrir la camara en este equipo.")

    # Se descartan frames iniciales para estabilizar sensor.
    camera.warmup(frames=10)
    print(f"Camara abierta con backend: {camera.active_backend}")
    print("Controles: q=salir | u=subir umbral | j=bajar umbral | r=recalibrar")

    # Inicializa contador de FPS.
    previous_time = 0.0
    frame_index = 0
    last_analysis = None

    # Bucle principal de lectura + inferencia.
    while True:
        # Captura frame actual.
        ok, frame = camera.read()
        if not ok or frame is None:
            break

        # Analiza frame con runtime visual (decimación opcional).
        frame_index += 1
        if args.detect_every <= 1 or frame_index % args.detect_every == 0 or last_analysis is None:
            analysis = runtime.analyze_image(frame, now=time.monotonic())
            last_analysis = analysis
        else:
            analysis = last_analysis

        # Cálculo de FPS instantáneo.
        current_time = time.time()
        fps = 1.0 / (current_time - previous_time) if previous_time > 0 else 0.0
        previous_time = current_time

        # Dibuja overlays de depuración.
        output = _draw_overlay(frame, analysis, fps)

        # Muestra ventana principal.
        cv2.imshow("Mostacho Vision Realtime", output)

        # Lee tecla para controles.
        key = cv2.waitKey(1) & 0xFF

        # Salida del programa.
        if key == ord("q"):
            break

        # Ajuste de umbral hacia arriba.
        if key == ord("u"):
            runtime.adjust_threshold(0.01)

        # Ajuste de umbral hacia abajo.
        if key == ord("j"):
            runtime.adjust_threshold(-0.01)

        # Recalibración completa.
        if key == ord("r"):
            runtime.reset_calibration()

    # Libera cámara al terminar.
    camera.release()
    # Cierra ventanas de OpenCV.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Entrada principal del script.
    main()
