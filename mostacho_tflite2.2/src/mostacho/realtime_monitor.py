"""Monitor local para probar mostacho v2.2 con cámara."""

from __future__ import annotations

import argparse
import base64
import time
from typing import Any, Dict, List

import httpx

from mostacho.vision.camera import Camera, list_available_cameras


def _load_cv2() -> Any:
    """Carga OpenCV de forma diferida."""

    import cv2  # type: ignore

    return cv2


def _encode_frame_to_b64(frame: Any, quality: int = 85) -> str:
    """Codifica un frame BGR como JPEG base64."""

    cv2 = _load_cv2()
    quality = max(30, min(int(quality), 95))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("No se pudo codificar frame a JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _draw_polyline(output: Any, points: List[List[float]], color: tuple[int, int, int]) -> None:
    """Dibuja landmarks simples si existen."""

    cv2 = _load_cv2()
    if len(points) < 2:
        return
    for x, y in points:
        cv2.circle(output, (int(x), int(y)), 2, color, -1)


def _draw_overlay(frame: Any, payload: Dict[str, Any], fps: float) -> Any:
    """Dibuja el estado combinado sobre el frame."""

    cv2 = _load_cv2()
    output = frame.copy()

    face_payload = payload.get("face_response") or {}
    detections = face_payload.get("detections") or []
    for detection in detections[:1]:
        bbox = detection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)
        _draw_polyline(output, detection.get("left_eye") or [], (0, 255, 255))
        _draw_polyline(output, detection.get("right_eye") or [], (0, 255, 255))
        _draw_polyline(output, detection.get("mouth") or [], (255, 0, 255))
    for eye_box in payload.get("eye_boxes", []) or []:
        if len(eye_box) == 4:
            x1, y1, x2, y2 = [int(value) for value in eye_box]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    state = str(payload.get("state", "UNKNOWN"))
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    eye_state = str(payload.get("eye_state", "UNKNOWN"))
    eye_conf = float(payload.get("eye_confidence", 0.0) or 0.0)
    model_state = str(payload.get("model_state", "NOT_RUN"))
    model_conf = float(payload.get("model_confidence", 0.0) or 0.0)
    model_ran = bool(payload.get("model_ran", False))
    run_reason = str(payload.get("run_reason", ""))
    microsleep_duration = float(payload.get("microsleep_duration", 0.0) or 0.0)
    microsleep_confirm_seconds = float(payload.get("microsleep_confirm_seconds", 1.5) or 1.5)
    distraction_enabled = bool(payload.get("distraction_enabled", False))
    distraction_ran = bool(payload.get("distraction_ran", False))
    distraction_state = str(payload.get("distraction_state", "UNKNOWN"))
    distraction_conf = float(payload.get("distraction_confidence", 0.0) or 0.0)
    distraction_reason = str(payload.get("distraction_run_reason", "disabled"))
    distraction_probs = payload.get("distraction_class_probabilities", {}) or {}
    d_normal = float(distraction_probs.get("normal", 0.0) or 0.0)
    d_distracted = float(distraction_probs.get("distracted", 0.0) or 0.0)

    color = (0, 255, 0)
    if state in {"EYES_CLOSED"} or eye_state == "CLOSED":
        color = (0, 165, 255)
    if state in {"SOMNOLENT"}:
        color = (0, 0, 255)
    if distraction_state == "DISTRACTED":
        color = (0, 0, 255)

    cv2.putText(output, f"STATE: {state} ({confidence:.2f})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(output, f"EYES: {eye_state} ({eye_conf:.2f})", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if model_ran:
        cv2.putText(output, f"MODEL: {model_state} ({model_conf:.2f})", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            output,
            f"MICROSLEEP: {microsleep_duration:.2f}s / {microsleep_confirm_seconds:.2f}s",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 220),
            2,
        )
    else:
        cv2.putText(output, "MODEL: OFF", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    distraction_status = "ON" if distraction_enabled else "OFF"
    distraction_source = "RUN" if distraction_ran else "CACHE"
    cv2.putText(
        output,
        f"DISTRACTION: {distraction_state} ({distraction_conf:.2f}) [{distraction_status}/{distraction_source}]",
        (15, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 255, 180) if distraction_state != "DISTRACTED" else (0, 120, 255),
        2,
    )
    cv2.putText(
        output,
        f"D-PROB N:{d_normal:.2f} D:{d_distracted:.2f}",
        (15, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        2,
    )
    cv2.putText(output, f"D-REASON: {distraction_reason}", (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)
    cv2.putText(output, f"REASON: {run_reason}", (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 180), 2)
    cv2.putText(output, f"FPS: {fps:.1f}", (15, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if state in {"SOMNOLENT"} or distraction_state == "DISTRACTED":
        cv2.rectangle(output, (0, 0), (output.shape[1], output.shape[0]), color, 6)

    return output


def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI del monitor."""

    parser = argparse.ArgumentParser(description="Monitor de cámara para Mostacho v2.2")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8002", help="URL base de la API")
    parser.add_argument("--camera-index", type=int, default=0, help="Indice de camara")
    parser.add_argument("--camera-backend", type=str, default="AUTO", help="AUTO|AVFOUNDATION|DSHOW|MSMF|V4L2")
    parser.add_argument("--width", type=int, default=640, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=360, help="Alto de captura")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="Calidad JPEG (30-95)")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout HTTP")
    parser.add_argument("--predict-every", type=int, default=1, help="Llamar a la API cada N frames")
    parser.add_argument("--max-fps", type=float, default=20.0, help="FPS maximos del loop (0 sin limite)")
    parser.add_argument("--list-cameras", action="store_true", help="Lista las camaras detectadas y sale")
    parser.add_argument("--max-camera-index", type=int, default=10, help="Indice maximo a revisar al listar")
    return parser.parse_args()


def main() -> None:
    """Ejecuta monitor de cámara con la API combinada."""

    args = parse_args()
    if args.list_cameras:
        cameras = list_available_cameras(
            max_index=args.max_camera_index,
            backend=args.camera_backend,
            width=args.width,
            height=args.height,
        )
        if not cameras:
            print("No se detectaron camaras.")
            return
        for camera_index, backend_name in cameras:
            print(f"index={camera_index} backend={backend_name}")
        return

    camera = Camera(index=args.camera_index, backend=args.camera_backend, width=args.width, height=args.height)
    if not camera.open():
        raise RuntimeError("No se pudo abrir la camara local.")
    camera.warmup(frames=10)

    cv2 = _load_cv2()
    frame_index = 0
    previous_time = 0.0
    fps = 0.0
    last_payload: Dict[str, Any] = {
        "state": "UNKNOWN",
        "confidence": 0.0,
        "eye_state": "UNKNOWN",
        "model_state": "NOT_RUN",
        "model_ran": False,
        "distraction_enabled": False,
        "distraction_ran": False,
        "distraction_state": "UNKNOWN",
        "distraction_confidence": 0.0,
        "distraction_run_reason": "init",
    }
    target_frame_time = 0.0 if args.max_fps <= 0 else 1.0 / float(args.max_fps)

    print("Monitor v2.2 iniciado. Tecla 'q' para salir.")
    print(f"API: {args.api_url}")

    with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
        while True:
            loop_started = time.monotonic()
            ok, frame = camera.read()
            if not ok or frame is None:
                break

            frame_index += 1
            if args.predict_every <= 1 or frame_index % args.predict_every == 0:
                try:
                    response = client.post(
                        f"{args.api_url}/predict/drowsiness_image",
                        json={
                            "image_b64": _encode_frame_to_b64(frame, quality=args.jpeg_quality),
                            "image_is_cropped": False,
                        },
                    )
                    response.raise_for_status()
                    last_payload = response.json()
                except httpx.HTTPStatusError as exc:
                    try:
                        detail_payload = exc.response.json()
                    except Exception:
                        detail_payload = exc.response.text
                    last_payload = {
                        "state": "ERROR",
                        "confidence": 0.0,
                        "eye_state": "UNKNOWN",
                        "model_state": "NOT_RUN",
                        "model_ran": False,
                        "distraction_enabled": False,
                        "distraction_ran": False,
                        "distraction_state": "UNKNOWN",
                        "distraction_confidence": 0.0,
                        "distraction_run_reason": "http_error",
                        "run_reason": f"HTTP {exc.response.status_code}: {detail_payload}",
                    }
                except httpx.ConnectError as exc:
                    last_payload = {
                        "state": "ERROR",
                        "confidence": 0.0,
                        "eye_state": "UNKNOWN",
                        "model_state": "NOT_RUN",
                        "model_ran": False,
                        "distraction_enabled": False,
                        "distraction_ran": False,
                        "distraction_state": "UNKNOWN",
                        "distraction_confidence": 0.0,
                        "distraction_run_reason": "connect_error",
                        "run_reason": f"ConnectError: {exc}",
                    }
                except Exception as exc:
                    last_payload = {
                        "state": "ERROR",
                        "confidence": 0.0,
                        "eye_state": "UNKNOWN",
                        "model_state": "NOT_RUN",
                        "model_ran": False,
                        "distraction_enabled": False,
                        "distraction_ran": False,
                        "distraction_state": "UNKNOWN",
                        "distraction_confidence": 0.0,
                        "distraction_run_reason": "exception",
                        "run_reason": str(exc),
                    }

            output = _draw_overlay(frame, last_payload, fps)
            cv2.imshow("Mostacho v2.2", output)

            elapsed = time.monotonic() - loop_started
            remaining = max(0.0, target_frame_time - elapsed)
            wait_ms = max(1, int(round(remaining * 1000.0))) if target_frame_time > 0 else 1

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break

            current_time = time.monotonic()
            fps = 1.0 / (current_time - previous_time) if previous_time > 0 else 0.0
            previous_time = current_time

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
