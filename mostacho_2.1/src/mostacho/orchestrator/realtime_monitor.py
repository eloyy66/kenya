"""Monitor en tiempo real: combina face_service + tf_service para somnolencia."""

from __future__ import annotations

# argparse para configuración desde CLI.
import argparse
# base64 para enviar frames por HTTP JSON.
import base64
# time para cálculo de FPS.
import time
# Path para rutas opcionales.
from pathlib import Path
# typing para anotaciones explícitas.
from typing import Any, Dict, List, Optional

# httpx para cliente HTTP síncrono.
import httpx

# Configuración central de endpoints.
from mostacho.settings import load_settings
# Wrapper de cámara compatible macOS/Windows/Linux.
from mostacho.vision.camera import Camera, list_available_cameras
from mostacho.vision.face_subprocess import FaceSubprocessClient


def _load_cv2() -> Any:
    """Carga OpenCV de forma diferida."""

    # Import local para evitar side-effects en import del módulo.
    import cv2  # type: ignore

    # Retorna módulo cv2 listo para usar.
    return cv2


def _encode_frame_to_b64(frame: Any, quality: int = 85) -> str:
    """Codifica un frame BGR como JPEG base64."""

    # Carga OpenCV para compresión JPEG.
    cv2 = _load_cv2()

    # Codifica imagen con calidad configurable.
    quality = max(30, min(int(quality), 95))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("No se pudo codificar frame a JPEG.")

    # Convierte bytes JPEG a cadena base64.
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _crop_face(frame: Any, face_bbox: Optional[List[float]]) -> Any:
    """Recorta rostro principal si bbox es válida; fallback a frame completo."""

    if not face_bbox or len(face_bbox) != 4:
        return frame

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in face_bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return frame

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else frame


def _draw_overlay(frame: Any, face_payload: Dict[str, Any], drowsiness_payload: Dict[str, Any], fps: float) -> Any:
    """Dibuja overlays de detección y estado final en el frame."""

    # Carga OpenCV para dibujo.
    cv2 = _load_cv2()

    # Copia de frame para no alterar buffer original.
    output = frame.copy()

    # Detecciones faciales retornadas por face_service.
    detections: List[Dict[str, Any]] = list(face_payload.get("detections", []))

    # Dibuja bbox y contorno de ojos por detección.
    for detection in detections:
        bbox = detection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for x, y in detection.get("left_eye", []):
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 255), -1)

        for x, y in detection.get("right_eye", []):
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 255), -1)

    # Campos principales de salida fusionada.
    fused_state = str(drowsiness_payload.get("state", "UNKNOWN"))
    fused_conf = float(drowsiness_payload.get("confidence", 0.0))
    model_state = str(drowsiness_payload.get("model_state", "unknown"))
    model_conf = float(drowsiness_payload.get("model_confidence", 0.0))
    risk_score = float(drowsiness_payload.get("risk_score", 0.0))

    # Estado de visión proveniente de EAR/landmarks.
    vision_state = str(face_payload.get("vision_state", "NO_FACE"))
    avg_ear = face_payload.get("avg_ear")
    closed_duration = float(face_payload.get("closed_duration", 0.0) or 0.0)

    # Color por severidad del estado final.
    color = (0, 255, 0)
    if fused_state == "DROWSY_WARNING":
        color = (0, 165, 255)
    elif fused_state == "SOMNOLENT":
        color = (0, 0, 255)

    # Texto de estado final fusionado.
    cv2.putText(output, f"FUSED: {fused_state} ({fused_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Texto de clase del modelo visual.
    cv2.putText(output, f"MODEL: {model_state} ({model_conf:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Texto de estado visual por EAR.
    cv2.putText(output, f"VISION: {vision_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
    # Texto de EAR.
    if avg_ear is not None:
        cv2.putText(output, f"EAR: {float(avg_ear):.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Texto de duración de ojos cerrados.
    cv2.putText(output, f"Closed: {closed_duration:.2f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Texto de riesgo fusionado.
    cv2.putText(output, f"Risk: {risk_score:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
    # Texto de FPS.
    cv2.putText(output, f"FPS: {fps:.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Retorna frame anotado.
    return output


def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI para monitor en tiempo real."""

    # Configuración básica del parser.
    parser = argparse.ArgumentParser(description="Realtime monitor (face_service + tf_service)")

    # Índice de cámara local.
    parser.add_argument("--camera-index", type=int, default=0, help="Indice de camara")
    # Backend de cámara recomendado AUTO para multiplataforma.
    parser.add_argument("--camera-backend", type=str, default="AUTO", help="AUTO|AVFOUNDATION|DSHOW|MSMF|V4L2")
    # Resolución de captura.
    parser.add_argument("--width", type=int, default=640, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=360, help="Alto de captura")
    # Lista cámaras detectadas y termina.
    parser.add_argument("--list-cameras", action="store_true", help="Lista cámaras disponibles y sale")
    # Índice máximo a probar en descubrimiento.
    parser.add_argument("--max-camera-index", type=int, default=10, help="Indice maximo a probar al listar cámaras")

    # Endpoint opcional de face_service.
    parser.add_argument("--face-url", type=str, default=None, help="URL base de face_service")
    # Modo de inferencia facial (http/subprocess).
    parser.add_argument("--face-mode", type=str, default=None, help="http|subprocess")
    # Python del entorno InsightFace (modo subprocess).
    parser.add_argument("--face-python", type=str, default=None, help="Python del entorno InsightFace")
    # Script worker de InsightFace (modo subprocess).
    parser.add_argument("--face-worker", type=str, default=None, help="Ruta al worker de InsightFace")
    # Timeout de worker (modo subprocess).
    parser.add_argument("--face-timeout", type=float, default=None, help="Timeout de worker (segundos)")
    # Endpoint opcional de tf_service.
    parser.add_argument("--tf-url", type=str, default=None, help="URL base de tf_service")

    # Timeout de requests HTTP.
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout HTTP en segundos")
    # Control de frecuencia para reducir carga.
    parser.add_argument("--face-every", type=int, default=1, help="Frames entre llamadas a face_service")
    parser.add_argument("--tf-every", type=int, default=1, help="Frames entre llamadas a tf_service")
    # Limita FPS totales del monitor para controlar consumo.
    parser.add_argument("--max-fps", type=float, default=20.0, help="FPS maximos del monitor (0 para sin limite)")
    # Enviar recorte de rostro al tf_service (reduce decode y resize).
    parser.add_argument(
        "--crop-to-face",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enviar recorte de rostro al tf_service",
    )
    # Calidad JPEG para reducir tamaño de payload.
    parser.add_argument("--jpeg-quality", type=int, default=85, help="Calidad JPEG (30-95)")

    # Retorna argumentos parseados.
    return parser.parse_args()


def main() -> None:
    """Ejecuta monitor de somnolencia en tiempo real."""

    # Argumentos del usuario.
    args = parse_args()

    # Modo descubrimiento de cámaras sin levantar monitor.
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

    # Carga configuración global de endpoints.
    settings = load_settings()
    face_mode = (args.face_mode or settings.face_mode).strip().lower()
    face_url = args.face_url or settings.face_service_url
    face_python = args.face_python or settings.face_python
    face_worker = Path(args.face_worker) if args.face_worker else settings.face_worker_path
    face_timeout = args.face_timeout or settings.face_timeout
    tf_url = args.tf_url or settings.tf_service_url

    # Inicializa cámara local.
    camera = Camera(index=args.camera_index, backend=args.camera_backend, width=args.width, height=args.height)
    if not camera.open():
        raise RuntimeError("No se pudo abrir la camara local.")

    # Calentamiento para estabilizar exposición/foco.
    camera.warmup(frames=10)

    # Carga OpenCV para ventana de visualización.
    cv2 = _load_cv2()

    # Cliente HTTP para llamadas a servicios locales.
    timeout = httpx.Timeout(args.timeout)

    # Buffers de última respuesta para tolerar errores transitorios de red.
    last_face_payload: Dict[str, Any] = {"detections": [], "vision_state": "NO_FACE", "closed_duration": 0.0}
    last_drowsiness_payload: Dict[str, Any] = {
        "state": "UNKNOWN",
        "confidence": 0.0,
        "model_state": "unknown",
        "model_confidence": 0.0,
        "risk_score": 0.0,
    }

    # Variables para medición de FPS.
    previous_time = 0.0
    fps = 0.0
    frame_index = 0
    target_frame_time = 0.0 if args.max_fps <= 0 else 1.0 / float(args.max_fps)

    print("Monitor en vivo iniciado. Tecla 'q' para salir.")
    print(f"face_mode:    {face_mode}")
    print(f"max_fps:      {args.max_fps:.1f}" if args.max_fps > 0 else "max_fps:      sin limite")
    if face_mode == "subprocess":
        print(f"face_python:  {face_python}")
        print(f"face_worker:  {face_worker}")
    else:
        print(f"face_service: {face_url}")
    print(f"tf_service:   {tf_url}")

    face_client: FaceSubprocessClient | None = None

    if face_mode == "subprocess":
        face_client = FaceSubprocessClient.from_settings(
            settings,
            python_executable=face_python,
            worker_path=face_worker,
            timeout=face_timeout,
        )
        face_client.start()

    try:
        # Bucle principal de captura e inferencia.
        with httpx.Client(timeout=timeout) as client:
            while True:
                # Marca inicio de iteración para aplicar throttle de FPS.
                loop_started = time.monotonic()

                # Captura frame local.
                ok, frame = camera.read()
                if not ok or frame is None:
                    break

                frame_index += 1

                # Decide si llama a face_service en este frame.
                call_face = args.face_every <= 1 or frame_index % args.face_every == 0

                face_payload = last_face_payload
                full_frame_b64: Optional[str] = None

                if call_face:
                    full_frame_b64 = _encode_frame_to_b64(frame, quality=args.jpeg_quality)
                    try:
                        if face_mode == "subprocess":
                            if face_client is None:
                                raise RuntimeError("face_client no inicializado")
                            face_payload = face_client.infer(image_b64=full_frame_b64)
                        else:
                            face_response = client.post(f"{face_url}/infer", json={"image_b64": full_frame_b64})
                            face_response.raise_for_status()
                            face_payload = face_response.json()

                        last_face_payload = face_payload
                    except Exception:
                        # Reusa última respuesta válida si falla request.
                        face_payload = last_face_payload

                # Toma bbox del rostro principal si existe.
                detections = face_payload.get("detections", []) or []
                primary_bbox = detections[0].get("bbox") if detections else None

                # Decide si llama a tf_service en este frame.
                call_tf = args.tf_every <= 1 or frame_index % args.tf_every == 0
                drowsiness_payload = last_drowsiness_payload

                if call_tf:
                    # Construye payload para tf_service.
                    if args.crop_to_face and primary_bbox is not None:
                        crop = _crop_face(frame, primary_bbox)
                        crop_b64 = _encode_frame_to_b64(crop, quality=args.jpeg_quality)
                        image_b64 = crop_b64
                        image_is_cropped = True
                        face_bbox = None
                    else:
                        if full_frame_b64 is None:
                            full_frame_b64 = _encode_frame_to_b64(frame, quality=args.jpeg_quality)
                        image_b64 = full_frame_b64
                        image_is_cropped = False
                        face_bbox = primary_bbox

                    drowsiness_request = {
                        "image_b64": image_b64,
                        "image_is_cropped": image_is_cropped,
                        "face_bbox": face_bbox,
                        "vision_state": face_payload.get("vision_state"),
                        "avg_ear": face_payload.get("avg_ear"),
                        "closed_duration": face_payload.get("closed_duration", 0.0),
                        "threshold": face_payload.get("threshold"),
                        "baseline": face_payload.get("baseline"),
                    }

                    # Llama a tf_service para estado de somnolencia fusionado.
                    try:
                        tf_response = client.post(f"{tf_url}/predict/drowsiness_image", json=drowsiness_request)
                        tf_response.raise_for_status()
                        drowsiness_payload = tf_response.json()
                        last_drowsiness_payload = drowsiness_payload
                    except Exception:
                        # Reusa última salida válida si falla request.
                        drowsiness_payload = last_drowsiness_payload

                # Dibuja overlays con resultados actuales.
                output = _draw_overlay(frame, face_payload, drowsiness_payload, fps)

                # Muestra ventana de monitor.
                cv2.imshow("Mostacho Drowsiness Monitor", output)

                # Espera restante para sostener FPS máximos configurados.
                elapsed = time.monotonic() - loop_started
                remaining = max(0.0, target_frame_time - elapsed)
                wait_ms = max(1, int(round(remaining * 1000.0))) if target_frame_time > 0 else 1

                # Salida por teclado.
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break

                # Cálculo de FPS reales incluyendo inferencia y throttle.
                current_time = time.monotonic()
                fps = 1.0 / (current_time - previous_time) if previous_time > 0 else 0.0
                previous_time = current_time
    finally:
        if face_client is not None:
            face_client.close()

        # Libera cámara y recursos gráficos.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Entrada directa del script.
    main()
