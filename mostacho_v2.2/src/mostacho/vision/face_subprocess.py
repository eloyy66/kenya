"""Cliente IPC por subprocess para InsightFace (stdin/stdout)."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mostacho.settings import Settings


@dataclass(frozen=True)
class FaceSubprocessConfig:
    """Configuración para levantar el worker de InsightFace."""

    python_executable: str
    worker_path: Path
    timeout: float
    pythonpath: Path


class FaceSubprocessClient:
    """Cliente síncrono para worker de InsightFace por pipes."""

    def __init__(
        self,
        config: FaceSubprocessConfig,
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
    ) -> "FaceSubprocessClient":
        """Construye cliente usando Settings y overrides opcionales."""

        py_exec = python_executable or settings.face_python
        worker = worker_path or settings.face_worker_path
        cfg = FaceSubprocessConfig(
            python_executable=py_exec,
            worker_path=worker,
            timeout=timeout or settings.face_timeout,
            pythonpath=settings.mostacho_root / "src",
        )

        env_overrides = {
            "MOSTACHO_REPO_ROOT": str(settings.repo_root),
            "MOSTACHO_ROOT": str(settings.mostacho_root),
            "MOSTACHO_DB_ROOT": str(settings.db_root),
            "MOSTACHO_ARTIFACTS_ROOT": str(settings.artifacts_root),
        }

        return cls(cfg, env_overrides=env_overrides)

    def __enter__(self) -> "FaceSubprocessClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Arranca el proceso worker si no está iniciado."""

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

        cmd = [
            self._config.python_executable,
            "-u",
            str(self._config.worker_path),
        ]

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
            raise RuntimeError("No se pudieron abrir pipes del worker.")

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)

    def close(self) -> None:
        """Cierra el proceso worker."""

        if self._proc is None:
            return

        try:
            if self._proc.stdin:
                shutdown_payload = json.dumps({"command": "shutdown"}, ensure_ascii=True)
                self._proc.stdin.write(shutdown_payload + "\n")
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
        """Envía imagen al worker y retorna respuesta JSON."""

        if not image_b64 and not image_path:
            raise ValueError("Debe enviar image_b64 o image_path.")

        self.start()

        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Worker no inicializado.")

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
                    raise TimeoutError("Timeout esperando respuesta del worker.")

                line = self._read_line(timeout=remaining)
                if not line:
                    if self._proc.poll() is None:
                        continue
                    raise RuntimeError("Worker terminó inesperadamente.")

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response_id = response.get("id")
                if response_id is None or response_id == request_id:
                    if "error" in response:
                        raise RuntimeError(response.get("error", "Error en worker"))
                    return response

    def _read_line(self, timeout: float) -> str:
        """Lee una línea de stdout con timeout."""

        if self._selector is None or self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Selector no inicializado.")

        events = self._selector.select(timeout)
        if not events:
            return ""
        return self._proc.stdout.readline().strip()
