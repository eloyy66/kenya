"""Logica de somnolencia usando EAR y tiempo de ojos cerrados."""

import time
from collections import deque
import numpy as np


class SomnolenceDetector:
    """Detecta somnolencia por ojos cerrados >= N segundos."""

    def __init__(
        self,
        window_size=5,
        closed_seconds=3.0,
        calibration_seconds=2.0,
        threshold_offset=0.04,
        min_threshold=0.15,
    ):
        self.window_size = window_size
        self.closed_seconds = closed_seconds
        self.calibration_seconds = calibration_seconds
        self.threshold_offset = threshold_offset
        self.min_threshold = min_threshold

        self.ear_window = deque(maxlen=window_size)
        self.closed_start = None

        self.calibration_start = None
        self.calibration_ears = []
        self.baseline_ear = None
        self.manual_threshold = None

    def reset_calibration(self):
        """Reinicia la calibracion y umbral manual."""
        self.calibration_start = None
        self.calibration_ears = []
        self.baseline_ear = None
        self.manual_threshold = None
        self.closed_start = None

    def adjust_threshold(self, delta):
        """Ajusta manualmente el umbral."""
        if self.manual_threshold is None:
            base = self.baseline_ear or 0.2
            self.manual_threshold = max(self.min_threshold, base - self.threshold_offset)
        self.manual_threshold = float(np.clip(self.manual_threshold + delta, 0.10, 0.35))

    def _get_dynamic_threshold(self):
        if self.manual_threshold is not None:
            return self.manual_threshold
        if self.baseline_ear is None:
            return None
        return max(self.min_threshold, self.baseline_ear - self.threshold_offset)

    def update(self, ear_value, now=None):
        """
        Actualiza el estado con un nuevo EAR.
        Retorna un dict con estado y metricas.
        """
        if now is None:
            now = time.monotonic()

        self.ear_window.append(ear_value)
        avg_ear = float(np.mean(self.ear_window))

        # Calibracion inicial
        if self.baseline_ear is None:
            if self.calibration_start is None:
                self.calibration_start = now
            self.calibration_ears.append(avg_ear)
            if (now - self.calibration_start) >= self.calibration_seconds and len(self.calibration_ears) >= 5:
                self.baseline_ear = float(np.median(self.calibration_ears))
            return {
                "state": "CALIBRATING",
                "confidence": 0.0,
                "scores": {"CALIBRATING": 1.0},
                "avg_ear": avg_ear,
                "closed_duration": 0.0,
                "threshold": None,
                "baseline": self.baseline_ear,
            }

        dynamic_threshold = self._get_dynamic_threshold()

        if avg_ear < dynamic_threshold:
            if self.closed_start is None:
                self.closed_start = now
            closed_duration = now - self.closed_start
            if closed_duration >= self.closed_seconds:
                state = "SOMNOLENT"
            else:
                state = "EYES_CLOSED"
        else:
            self.closed_start = None
            closed_duration = 0.0
            state = "ATTENTIVE"

        if state == "SOMNOLENT":
            confidence = 0.9
        elif state == "EYES_CLOSED":
            confidence = 0.7
        elif state == "ATTENTIVE":
            confidence = 0.6
        else:
            confidence = 0.3

        scores = {state: 1.0}

        return {
            "state": state,
            "confidence": confidence,
            "scores": scores,
            "avg_ear": avg_ear,
            "closed_duration": closed_duration,
            "threshold": dynamic_threshold,
            "baseline": self.baseline_ear,
        }
