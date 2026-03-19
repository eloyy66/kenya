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
        calibration_seconds: float = 5.0,
        threshold_offset: float = 0.04,
        min_threshold: float = 0.15,
        reopen_threshold_offset: float = 0.015,
        adaptive_baseline_alpha: float = 0.05,
        threshold_std_factor: float = 2.0,
        min_calibration_samples: int = 30,
        reference_window_size: int = 180,
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
        # Histéresis para evitar alternancia entre abierto/cerrado.
        self.reopen_threshold_offset = reopen_threshold_offset
        # Peso de recalibración online durante estado atento.
        self.adaptive_baseline_alpha = adaptive_baseline_alpha
        # Multiplicador de dispersión para umbral dinámico.
        self.threshold_std_factor = threshold_std_factor
        # Muestras mínimas para fijar baseline inicial.
        self.min_calibration_samples = min_calibration_samples
        # Ventana de referencias para recalibración lenta.
        self.reference_window_size = reference_window_size

        # Buffer de EAR reciente para media móvil.
        self.ear_window: deque[float] = deque(maxlen=window_size)
        # Inicio del periodo de ojos cerrados continuo.
        self.closed_start: float | None = None

        # Estado interno de calibración.
        self.calibration_start: float | None = None
        self.calibration_ears: list[float] = []
        self.baseline_ear: float | None = None
        self.baseline_spread: float | None = None
        self.manual_threshold: float | None = None
        self.reference_ears: deque[float] = deque(maxlen=reference_window_size)

    def reset_calibration(self) -> None:
        """Reinicia calibración, baseline y umbral manual."""

        # Limpia temporizador de calibración.
        self.calibration_start = None
        # Limpia histórico de EAR de calibración.
        self.calibration_ears = []
        # Elimina baseline previo.
        self.baseline_ear = None
        # Elimina dispersión previa.
        self.baseline_spread = None
        # Elimina ajuste manual previo.
        self.manual_threshold = None
        # Reinicia contador de ojos cerrados.
        self.closed_start = None
        # Limpia ventana de suavizado para empezar limpio.
        self.ear_window.clear()
        # Limpia referencias de recalibración online.
        self.reference_ears.clear()

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

        # Umbral derivado de baseline con offset fijo + dispersión observada.
        spread = self.baseline_spread or 0.0
        close_offset = max(self.threshold_offset, spread * self.threshold_std_factor)
        return max(self.min_threshold, self.baseline_ear - close_offset)

    def current_threshold(self) -> float | None:
        """Expone el umbral actual para overlays y APIs."""

        return self._dynamic_threshold()

    def _open_threshold(self, close_threshold: float) -> float:
        """Define umbral de reapertura para histéresis."""

        # Se usa margen fijo pequeño para no reabrir con ruido de un frame.
        reopen_offset = max(self.reopen_threshold_offset, (self.baseline_spread or 0.0) * 0.75)
        # Se evita ubicar el umbral de reapertura por encima del baseline.
        baseline_cap = (self.baseline_ear or close_threshold) - 0.002
        return min(baseline_cap, close_threshold + reopen_offset)

    def _estimate_spread(self, samples: np.ndarray) -> float:
        """Calcula dispersión robusta de EAR usando MAD y desvío estándar."""

        # Con muy pocas muestras se devuelve un ruido base pequeño.
        if samples.size < 2:
            return 0.005

        # MAD es más robusto ante parpadeos o outliers cortos.
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        robust_sigma = mad * 1.4826
        std = float(np.std(samples))
        return max(0.005, robust_sigma, std)

    def _set_baseline_from_samples(self, samples: list[float]) -> None:
        """Fija baseline y dispersión robusta desde muestras atentas."""

        sample_array = np.asarray(samples, dtype=float)
        self.baseline_ear = float(np.median(sample_array))
        self.baseline_spread = self._estimate_spread(sample_array)

        self.reference_ears.clear()
        self.reference_ears.extend(float(value) for value in sample_array[-self.reference_window_size :])

    def _update_online_baseline(self, avg_ear: float) -> None:
        """Recalibra lentamente el baseline solo en frames claramente atentos."""

        if self.baseline_ear is None:
            return

        # Se aceptan solo muestras cercanas al baseline abierto, no somnolientas.
        spread = self.baseline_spread or 0.005
        attentive_gate = max(self.reopen_threshold_offset, spread * 1.5)
        if avg_ear < (self.baseline_ear - attentive_gate):
            return

        self.reference_ears.append(float(avg_ear))
        sample_array = np.asarray(self.reference_ears, dtype=float)
        target_baseline = float(np.median(sample_array))
        target_spread = self._estimate_spread(sample_array)

        alpha = float(np.clip(self.adaptive_baseline_alpha, 0.0, 1.0))
        self.baseline_ear = ((1.0 - alpha) * self.baseline_ear) + (alpha * target_baseline)

        if self.baseline_spread is None:
            self.baseline_spread = target_spread
        else:
            self.baseline_spread = ((1.0 - alpha) * self.baseline_spread) + (alpha * target_spread)

    def update(
        self,
        ear_value: float,
        now: float | None = None,
        left_ear: float | None = None,
        right_ear: float | None = None,
    ) -> Dict[str, float | str | Dict[str, float] | None]:
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
            if elapsed >= self.calibration_seconds and len(self.calibration_ears) >= self.min_calibration_samples:
                self._set_baseline_from_samples(self.calibration_ears)

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

        # Umbral de reapertura para histéresis.
        open_threshold = self._open_threshold(threshold)

        # Regla principal: para somnolencia se prioriza que ambos ojos caigan bajo umbral.
        eyes_closed_candidate = avg_ear < threshold
        eyes_open_candidate = avg_ear > open_threshold
        if left_ear is not None and right_ear is not None:
            eyes_closed_candidate = float(left_ear) < threshold and float(right_ear) < threshold
            eyes_open_candidate = float(left_ear) > open_threshold and float(right_ear) > open_threshold

        # Determina estado visual según umbral de cierre e histéresis de reapertura.
        if eyes_closed_candidate:
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
        elif self.closed_start is not None and not eyes_open_candidate:
            # Mantiene estado cerrado mientras EAR siga en banda ambigua.
            closed_duration = now - self.closed_start
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
            # Recalibra lentamente solo al estar claramente atento.
            self._update_online_baseline(avg_ear)

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
