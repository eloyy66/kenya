"""Analizador de biometria (placeholder)."""


class BiometricsAnalyzer:
    """Stub de biometria: devuelve estado UNKNOWN hasta integrar wearable real."""

    def __init__(self):
        self.last_state = "UNKNOWN"

    def update(self, biometrics_sample=None):
        """
        Retorna un dict compatible con el fuser multimodal.
        En el futuro, aqui se procesa HR/HRV para detectar estres/fatiga.
        """
        return {
            "state": "UNKNOWN",
            "confidence": 0.0,
            "scores": {"UNKNOWN": 1.0},
            "meta": {"source": "biometrics_stub"},
        }
