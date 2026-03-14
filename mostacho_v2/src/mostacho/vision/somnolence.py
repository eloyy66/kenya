"""Lógica de somnolencia basada en EAR y duración de ojos cerrados."""

from __future__ import annotations

# time provee reloj monotónico para duraciones robustas.
import time
# deque mantiene ventana deslizante de EAR.
from collections import deque
# typing para contratos explícitos.
from typing import Dict

# numpy se usa para estadística de ventana.
import numpy as np


class SomnolenceDetector:
    """Detecta estados visuales de atención/somnolencia usando EAR."""

    def __init__(
        self,
        window_size: int = 5,
        closed_seconds: float = 3.0,
        calibration_seconds: float = 2.0,
        threshold_offset: float = 0.04,
        min_threshold: float = 0.15,
    ) -> None:
        # Tamaño de ventana de suavizado de EAR.
        self.window_size = window_size
        # Tiempo continuo mínimo de ojos cerrados para somnolencia.
        self.closed_seconds = closed_seconds
        # Duración de calibración inicial del baseline.
        self.calibration_seconds = calibration_seconds
        # Desplazamiento respecto al baseline para umbral dinámico.
        self.threshold_offset = threshold_offset
        # Umbral mínimo de seguridad.
        self.min_threshold = min_threshold

        # Buffer de EAR reciente para media móvil.
        self.ear_window: deque[float] = deque(maxlen=window_size)
        # Inicio del periodo de ojos cerrados continuo.
        self.closed_start: float | None = None

        # Estado interno de calibración.
        self.calibration_start: float | None = None
        self.calibration_ears: list[float] = []
        self.baseline_ear: float | None = None
        self.manual_threshold: float | None = None

    def reset_calibration(self) -> None:
        """Reinicia calibración, baseline y umbral manual."""

        # Limpia temporizador de calibración.
        self.calibration_start = None
        # Limpia histórico de EAR de calibración.
        self.calibration_ears = []
        # Elimina baseline previo.
        self.baseline_ear = None
        # Elimina ajuste manual previo.
        self.manual_threshold = None
        # Reinicia contador de ojos cerrados.
        self.closed_start = None
        # Limpia ventana de suavizado para empezar limpio.
        self.ear_window.clear()

    def adjust_threshold(self, delta: float) -> float:
        """Ajusta umbral manual de detección y retorna nuevo valor."""

        # Si no existe umbral manual, inicializa desde baseline.
        if self.manual_threshold is None:
            base = self.baseline_ear or 0.2
            self.manual_threshold = max(self.min_threshold, base - self.threshold_offset)

        # Se aplica delta acotado para evitar valores extremos.
        self.manual_threshold = float(np.clip(self.manual_threshold + delta, 0.10, 0.35))
        # Se retorna valor actualizado para UI/log.
        return self.manual_threshold

    def _dynamic_threshold(self) -> float | None:
        """Retorna umbral dinámico efectivo (manual o por baseline)."""

        # Prioridad al umbral manual si existe.
        if self.manual_threshold is not None:
            return self.manual_threshold

        # Si no hay baseline aún, no hay umbral válido.
        if self.baseline_ear is None:
            return None

        # Umbral derivado de baseline con mínimo de seguridad.
        return max(self.min_threshold, self.baseline_ear - self.threshold_offset)

    def update(self, ear_value: float, now: float | None = None) -> Dict[str, float | str | Dict[str, float] | None]:
        """Actualiza estado con nuevo EAR y retorna métricas de visión."""

        # Si no se pasa tiempo, se usa reloj monotónico.
        if now is None:
            now = time.monotonic()

        # Se agrega muestra actual al buffer deslizante.
        self.ear_window.append(float(ear_value))
        # Se calcula EAR suavizado.
        avg_ear = float(np.mean(self.ear_window))

        # Etapa de calibración inicial hasta fijar baseline.
        if self.baseline_ear is None:
            # Si es primera muestra, marca inicio de calibración.
            if self.calibration_start is None:
                self.calibration_start = now

            # Acumula EAR de calibración.
            self.calibration_ears.append(avg_ear)

            # Cuando se cumple tiempo mínimo y muestras suficientes, fija baseline.
            elapsed = now - self.calibration_start
            if elapsed >= self.calibration_seconds and len(self.calibration_ears) >= 5:
                self.baseline_ear = float(np.median(self.calibration_ears))

            # Respuesta mientras calibra.
            return {
                "state": "CALIBRATING",
                "confidence": 0.0,
                "scores": {"CALIBRATING": 1.0},
                "avg_ear": avg_ear,
                "closed_duration": 0.0,
                "threshold": None,
                "baseline": self.baseline_ear,
            }

        # Umbral efectivo actual.
        threshold = self._dynamic_threshold()

        # Seguridad adicional: si algo raro sucede y no hay umbral.
        if threshold is None:
            return {
                "state": "NO_THRESHOLD",
                "confidence": 0.0,
                "scores": {"NO_THRESHOLD": 1.0},
                "avg_ear": avg_ear,
                "closed_duration": 0.0,
                "threshold": None,
                "baseline": self.baseline_ear,
            }

        # Determina estado visual según EAR frente a umbral.
        if avg_ear < threshold:
            # Inicia contador de cierre si aún no estaba cerrando.
            if self.closed_start is None:
                self.closed_start = now

            # Tiempo continuo de ojos cerrados.
            closed_duration = now - self.closed_start

            # Clasificación según duración acumulada.
            if closed_duration >= self.closed_seconds:
                state = "SOMNOLENT"
            else:
                state = "EYES_CLOSED"
        else:
            # Reinicia contador cuando vuelve a abrir ojos.
            self.closed_start = None
            # Duración cerrados es cero.
            closed_duration = 0.0
            # Estado atencional normal.
            state = "ATTENTIVE"

        # Confianza heurística simple por estado.
        if state == "SOMNOLENT":
            confidence = 0.9
        elif state == "EYES_CLOSED":
            confidence = 0.7
        else:
            confidence = 0.6

        # Formato de scores compatible con pipeline multimodal.
        scores = {state: 1.0}

        # Retorna paquete completo de métricas.
        return {
            "state": state,
            "confidence": confidence,
            "scores": scores,
            "avg_ear": avg_ear,
            "closed_duration": float(closed_duration),
            "threshold": float(threshold),
            "baseline": self.baseline_ear,
        }
