"""Capa de captura de cámara compatible con macOS, Windows y Linux."""

from __future__ import annotations

# platform permite detectar sistema operativo en tiempo de ejecución.
import platform
# typing para tipos explícitos y mantenibles.
from typing import Any, Iterable, List, Tuple


def _load_cv2() -> Any:
    """Carga OpenCV de forma diferida para reducir side-effects de import."""

    # Import local para evitar cargar OpenCV al importar el módulo completo.
    import cv2  # type: ignore

    # Retorna módulo OpenCV listo para usar.
    return cv2


def _backend_candidates(backend: str) -> List[Tuple[str, int | None]]:
    """Resuelve lista ordenada de backends candidatos por plataforma."""

    # Se carga OpenCV para acceder a constantes CAP_*.
    cv2 = _load_cv2()
    # Se normaliza string de backend.
    normalized = backend.strip().upper()

    # Si el usuario fija backend explícito, se respeta y se agrega fallback default.
    if normalized not in {"", "AUTO", "DEFAULT"}:
        mapping = {
            "AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", None),
            "DSHOW": getattr(cv2, "CAP_DSHOW", None),
            "MSMF": getattr(cv2, "CAP_MSMF", None),
            "V4L2": getattr(cv2, "CAP_V4L2", None),
            "GSTREAMER": getattr(cv2, "CAP_GSTREAMER", None),
        }
        selected = mapping.get(normalized)
        if selected is None:
            return [("DEFAULT", None)]
        return [(normalized, selected), ("DEFAULT", None)]

    # Detección de OS para orden de intentos.
    system_name = platform.system().lower()

    # macOS: AVFoundation suele ser el backend correcto.
    if system_name == "darwin":
        return [("AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", None)), ("DEFAULT", None)]

    # Windows: DSHOW suele ser estable; MSMF como fallback.
    if system_name == "windows":
        return [
            ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
            ("MSMF", getattr(cv2, "CAP_MSMF", None)),
            ("DEFAULT", None),
        ]

    # Linux: V4L2 es opción primaria, luego default.
    return [("V4L2", getattr(cv2, "CAP_V4L2", None)), ("DEFAULT", None)]


class Camera:
    """Encapsula captura de video con fallback automático de backend."""

    def __init__(self, index: int = 0, backend: str = "AUTO", width: int = 640, height: int = 360):
        # Índice de dispositivo de cámara.
        self.index = index
        # Nombre de backend preferido o AUTO.
        self.backend = backend
        # Ancho esperado de captura.
        self.width = width
        # Alto esperado de captura.
        self.height = height
        # Objeto VideoCapture de OpenCV.
        self.cap: Any = None
        # Nombre del backend que finalmente abrió.
        self.active_backend = "UNOPENED"

    def open(self) -> bool:
        """Abre la cámara probando backends en orden de compatibilidad."""

        # Se carga OpenCV cuando realmente se necesita abrir cámara.
        cv2 = _load_cv2()

        # Se recorre lista de backends candidatos.
        for backend_name, backend_code in _backend_candidates(self.backend):
            # Si el código de backend no existe en esta plataforma, se ignora.
            if backend_name != "DEFAULT" and backend_code is None:
                continue

            # Se intenta abrir cámara con backend específico.
            if backend_name == "DEFAULT":
                cap = cv2.VideoCapture(self.index)
            else:
                cap = cv2.VideoCapture(self.index, backend_code)

            # Si no abrió, se libera y se prueba siguiente backend.
            if not cap.isOpened():
                cap.release()
                continue

            # Se fija resolución objetivo cuando aplica.
            if self.width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

            # Se conserva captura abierta y backend ganador.
            self.cap = cap
            self.active_backend = backend_name
            return True

        # Si ningún backend abrió, se deja estado cerrado.
        self.cap = None
        self.active_backend = "FAILED"
        return False

    def warmup(self, frames: int = 10) -> None:
        """Descarta frames iniciales para estabilizar exposición/foco."""

        # Si no hay cámara abierta, no hace nada.
        if self.cap is None:
            return

        # Lee y descarta primeros frames.
        for _ in range(max(frames, 0)):
            self.cap.read()

    def read(self) -> tuple[bool, Any]:
        """Lee frame actual de la cámara."""

        # Si no hay cámara abierta, retorna fallo seguro.
        if self.cap is None:
            return False, None

        # Delega en OpenCV.
        return self.cap.read()

    def release(self) -> None:
        """Libera el dispositivo de cámara."""

        # Si existe captura activa, se libera recurso.
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.active_backend = "UNOPENED"


def list_available_cameras(max_index: int = 10, backend: str = "AUTO", width: int = 640, height: int = 360) -> List[Tuple[int, str]]:
    """Descubre índices de cámara disponibles y backend efectivo por índice."""

    # Lista de resultados (indice, backend_efectivo).
    available: List[Tuple[int, str]] = []

    # Recorre rango de índices solicitado.
    for camera_index in range(max(0, int(max_index)) + 1):
        # Crea instancia temporal de cámara.
        camera = Camera(index=camera_index, backend=backend, width=width, height=height)
        # Si abre correctamente, agrega resultado.
        if camera.open():
            available.append((camera_index, camera.active_backend))
            camera.release()

    # Retorna lista de cámaras detectadas.
    return available
