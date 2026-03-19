"""Cliente IPC por subprocess para worker TensorFlow unificado."""

from __future__ import annotations

import base64
import json
import os
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mostacho.settings import Settings


@dataclass(frozen=True)
class TfSubprocessConfig:
    """Configuracion para levantar el worker TensorFlow."""

    python_executable: str
    worker_path: Path
    timeout: float
    pythonpath: Path


class TfSubprocessClient:
    """Cliente sincronico para worker TensorFlow (eye + drowsiness + distraction)."""

    def __init__(
        self,
        config: TfSubprocessConfig,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._config = config
        self._env_overrides = env_overrides or {}
        self._proc: subprocess.Popen[str] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        python_executable: Optional[str] = None,
        worker_path: Optional[Path] = None,
        timeout: Optional[float] = None,
    ) -> "TfSubprocessClient":
        """Construye cliente desde Settings y overrides de entorno."""

        env_python = os.getenv("MOSTACHO_TF_PYTHON", "").strip()
        env_worker = os.getenv("MOSTACHO_TF_WORKER", "").strip()
        env_timeout = os.getenv("MOSTACHO_TF_TIMEOUT", "").strip()

        resolved_timeout = timeout or settings.eye_timeout
        if env_timeout:
            try:
                resolved_timeout = float(env_timeout)
            except ValueError:
                pass

        cfg = TfSubprocessConfig(
            python_executable=python_executable or env_python or settings.eye_python,
            worker_path=worker_path
            or Path(env_worker or str(settings.mostacho_root / "src" / "mostacho" / "services" / "tf_inference_worker.py")),
            timeout=float(resolved_timeout),
            pythonpath=settings.mostacho_root / "src",
        )

        env_overrides = {
            "MOSTACHO_REPO_ROOT": str(settings.repo_root),
            "MOSTACHO_ROOT": str(settings.mostacho_root),
            "MOSTACHO_DB_ROOT": str(settings.db_root),
            "MOSTACHO_ARTIFACTS_ROOT": str(settings.artifacts_root),
        }
        return cls(cfg, env_overrides=env_overrides)

    def start(self) -> None:
        """Arranca el worker si no esta iniciado."""

        if self._proc is not None and self._proc.poll() is None:
            return

        env = os.environ.copy()
        env.update(self._env_overrides)
        env["PYTHONUNBUFFERED"] = "1"

        pythonpath_entries = [str(self._config.pythonpath)]
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        cmd = [self._config.python_executable, "-u", str(self._config.worker_path)]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )

        if self._proc.stdout is None or self._proc.stdin is None:
            raise RuntimeError("No se pudieron abrir pipes del worker TensorFlow.")

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)

    def close(self) -> None:
        """Cierra el worker."""

        if self._proc is None:
            return

        try:
            if self._proc.stdin:
                self._proc.stdin.write(json.dumps({"command": "shutdown"}, ensure_ascii=True) + "\n")
                self._proc.stdin.flush()
                self._read_line(timeout=1.0)
        except Exception:
            pass

        try:
            self._proc.terminate()
        except Exception:
            pass

        try:
            self._proc.wait(timeout=2.0)
        except Exception:
            pass

        if self._selector is not None and self._proc.stdout is not None:
            try:
                self._selector.unregister(self._proc.stdout)
            except Exception:
                pass

        self._proc = None
        self._selector = None

    def infer_eye(self, image_b64: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Clasifica open/closed eyes."""

        return self._infer("infer_eye", image_b64=image_b64, image_path=image_path)

    def infer_drowsiness(self, image_b64: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Predice probabilidades del modelo de somnolencia."""

        return self._infer("infer_drowsiness", image_b64=image_b64, image_path=image_path)

    def infer_distraction(self, image_b64: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Predice probabilidades del modelo de distraccion (opcional)."""

        return self._infer("infer_distraction", image_b64=image_b64, image_path=image_path)

    def infer_eye_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Codifica ndarray a PNG base64 y lo envia para inferencia de ojos."""

        return self.infer_eye(image_b64=self._encode_png_b64(image))

    def infer_drowsiness_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Codifica ndarray a PNG base64 y lo envia para inferencia de somnolencia."""

        return self.infer_drowsiness(image_b64=self._encode_png_b64(image))

    def infer_distraction_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Codifica ndarray a PNG base64 y lo envia para inferencia de distraccion."""

        return self.infer_distraction(image_b64=self._encode_png_b64(image))

    def _infer(self, command: str, image_b64: Optional[str], image_path: Optional[str]) -> Dict[str, Any]:
        """Envio sincronico de un comando al worker."""

        if not image_b64 and not image_path:
            raise ValueError("Debe enviar image_b64 o image_path.")

        self.start()
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Worker TensorFlow no inicializado.")

        with self._lock:
            request_id = self._next_id
            self._next_id += 1

            payload: Dict[str, Any] = {"id": request_id, "command": command}
            if image_b64:
                payload["image_b64"] = image_b64
            if image_path:
                payload["image_path"] = image_path

            self._proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._proc.stdin.flush()

            deadline = time.monotonic() + self._config.timeout
            while True:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    raise TimeoutError(f"Timeout esperando respuesta del worker TensorFlow ({command}).")

                line = self._read_line(timeout=remaining)
                if not line:
                    if self._proc.poll() is None:
                        continue
                    raise RuntimeError("Worker TensorFlow termino inesperadamente.")

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue

                response_id = response.get("id")
                if response_id is None or response_id == request_id:
                    if "error" in response:
                        raise RuntimeError(response.get("error", f"Error en worker TensorFlow ({command})"))
                    return response

    def _read_line(self, timeout: float) -> str:
        """Lee una linea de stdout con timeout."""

        if self._selector is None or self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Selector del worker no inicializado.")

        events = self._selector.select(timeout)
        if not events:
            return ""
        return self._proc.stdout.readline().strip()

    @staticmethod
    def _encode_png_b64(image: np.ndarray) -> str:
        """Convierte ndarray RGB/gray en PNG base64."""

        import cv2  # type: ignore

        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Imagen vacia para inferencia.")

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                max_value = float(np.nanmax(array)) if array.size > 0 else 0.0
                if max_value <= 1.5:
                    array = np.clip(array, 0.0, 1.0) * 255.0
                else:
                    array = np.clip(array, 0.0, 255.0)
                array = np.rint(array).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 3 and array.shape[2] == 3:
            # OpenCV codifica en BGR; convertimos desde RGB para preservar colores.
            array_to_encode = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        elif array.ndim == 3 and array.shape[2] == 1:
            array_to_encode = array[:, :, 0]
        else:
            array_to_encode = array

        ok, encoded = cv2.imencode(".png", array_to_encode)
        if not ok:
            raise RuntimeError("No se pudo codificar PNG para el worker TensorFlow.")

        return base64.b64encode(encoded.tobytes()).decode("utf-8")
