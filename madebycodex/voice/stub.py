"""Analizador de voz (placeholder)."""


class VoiceAnalyzer:
    """Stub de voz: devuelve estado UNKNOWN hasta integrar audio real."""

    def __init__(self):
        self.last_state = "UNKNOWN"

    def update(self, audio_chunk=None):
        """
        Retorna un dict compatible con el fuser multimodal.
        En el futuro, aqui se procesa audio para detectar estres/atencion.
        """
        return {
            "state": "UNKNOWN",
            "confidence": 0.0,
            "scores": {"UNKNOWN": 1.0},
            "meta": {"source": "voice_stub"},
        }
