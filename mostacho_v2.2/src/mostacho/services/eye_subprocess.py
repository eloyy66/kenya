"""Cliente IPC por subprocess para el modelo open/closed eyes."""

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
class EyeSubprocessConfig:
    """Configuracion para levantar el worker del clasificador de ojos."""

    python_executable: str
    worker_path: Path
    timeout: float
    pythonpath: Path


class EyeSubprocessClient:
    """Cliente sincronico para el worker open/closed eyes por pipes."""

    def __init__(
        self,
        config: EyeSubprocessConfig,
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
    ) -> "EyeSubprocessClient":
        """Construye cliente usando Settings y overrides opcionales."""

        cfg = EyeSubprocessConfig(
            python_executable=python_executable or settings.eye_python,
            worker_path=worker_path or settings.eye_worker_path,
            timeout=timeout or settings.eye_timeout,
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
        """Arranca el proceso worker si no esta iniciado."""

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
            raise RuntimeError("No se pudieron abrir pipes del worker open/closed eyes.")

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)

    def close(self) -> None:
        """Cierra el proceso worker."""

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

    def infer(self, image_b64: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Envía una imagen al worker y retorna respuesta JSON."""

        if not image_b64 and not image_path:
            raise ValueError("Debe enviar image_b64 o image_path.")

        self.start()
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Worker open/closed eyes no inicializado.")

        with self._lock:
            request_id = self._next_id
            self._next_id += 1

            payload: Dict[str, Any] = {"id": request_id}
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
                    raise TimeoutError("Timeout esperando respuesta del worker open/closed eyes.")

                line = self._read_line(timeout=remaining)
                if not line:
                    if self._proc.poll() is None:
                        continue
                    raise RuntimeError("Worker open/closed eyes termino inesperadamente.")

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response_id = response.get("id")
                if response_id is None or response_id == request_id:
                    if "error" in response:
                        raise RuntimeError(response.get("error", "Error en worker open/closed eyes"))
                    return response

    def infer_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Codifica un ndarray a PNG base64 y lo envia al worker."""

        import tensorflow as tf  # type: ignore

        tensor = tf.convert_to_tensor(image)
        if tensor.shape.rank == 2:
            tensor = tf.expand_dims(tensor, axis=-1)

        if tensor.dtype != tf.uint8:
            if tensor.dtype.is_floating:
                max_value = float(tf.reduce_max(tensor).numpy()) if tf.size(tensor) > 0 else 0.0
                if max_value <= 1.5:
                    tensor = tf.clip_by_value(tensor, 0.0, 1.0) * 255.0
                else:
                    tensor = tf.clip_by_value(tensor, 0.0, 255.0)
                tensor = tf.cast(tf.round(tensor), tf.uint8)
            else:
                tensor = tf.cast(tensor, tf.uint8)

        encoded = tf.io.encode_png(tensor).numpy()
        image_b64 = base64.b64encode(encoded).decode("utf-8")
        return self.infer(image_b64=image_b64)

    def _read_line(self, timeout: float) -> str:
        """Lee una linea del stdout del worker con timeout."""

        if self._selector is None or self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Selector del worker no inicializado.")

        events = self._selector.select(timeout)
        if not events:
            return ""
        return self._proc.stdout.readline().strip()
