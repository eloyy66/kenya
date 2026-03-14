"""Extraccion de features de voz para inferencia y entrenamiento."""

from __future__ import annotations

# logging permite reportar caidas de librerias opcionales.
import logging
# Path se usa para resolver rutas de audio de forma robusta.
from pathlib import Path
# typing define contrato de salida.
from typing import Dict

# numpy se usa para estadisticas basicas de señal.
import numpy as np


# Logger del modulo para mensajes de depuracion.
LOGGER = logging.getLogger(__name__)


def extract_voice_features(audio_path: Path, target_sr: int = 16000) -> Dict[str, float]:
    """Extrae features robustas desde un archivo de audio."""

    # Se normaliza entrada a Path por seguridad.
    audio_path = Path(audio_path)
    # Si el archivo no existe, se retorna vector neutro.
    if not audio_path.exists():
        return {
            "voice_rms_mean": 0.0,
            "voice_rms_std": 0.0,
            "voice_zcr_mean": 0.0,
            "voice_spectral_centroid_mean": 0.0,
            "voice_duration_sec": 0.0,
        }

    # Se intenta usar librosa para extracción espectral rica.
    try:
        # Import local para mantener carga perezosa.
        import librosa  # type: ignore

        # Se carga audio mono remuestreado a frecuencia objetivo.
        signal, sample_rate = librosa.load(str(audio_path), sr=target_sr, mono=True)
        # Si la señal viene vacía, se retorna vector neutro.
        if signal.size == 0:
            return {
                "voice_rms_mean": 0.0,
                "voice_rms_std": 0.0,
                "voice_zcr_mean": 0.0,
                "voice_spectral_centroid_mean": 0.0,
                "voice_duration_sec": 0.0,
            }

        # RMS por frame para energía de voz.
        rms = librosa.feature.rms(y=signal)[0]
        # Tasa de cruces por cero para dinamica temporal.
        zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
        # Centroide espectral para balance frecuencial.
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]

        # Se entrega diccionario float listo para fusion.
        return {
            "voice_rms_mean": float(np.mean(rms)),
            "voice_rms_std": float(np.std(rms)),
            "voice_zcr_mean": float(np.mean(zcr)),
            "voice_spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "voice_duration_sec": float(signal.shape[0] / sample_rate),
        }
    except Exception as exc:  # pragma: no cover - ruta de contingencia
        # Se registra warning y se aplica fallback sin librosa.
        LOGGER.warning("Fallo extraccion avanzada de voz (%s). Se usa fallback simple.", exc)

    # Fallback simple usando soundfile para minimizar dependencia.
    try:
        # Import local para evitar fallo de modulo al importar paquete.
        import soundfile as sf  # type: ignore

        # Se lee señal completa.
        signal, sample_rate = sf.read(str(audio_path), always_2d=False)
        # Si el audio es estéreo, se promedia a mono.
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        # Se convierte a float32 para cálculo estable.
        signal = signal.astype(np.float32)
        # Se calcula RMS global como energia media.
        rms_value = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        # Se calcula ZCR global aproximada.
        zcr_value = float(np.mean(np.abs(np.diff(np.signbit(signal))))) if signal.size > 1 else 0.0

        # Se retorna vector reducido de contingencia.
        return {
            "voice_rms_mean": rms_value,
            "voice_rms_std": 0.0,
            "voice_zcr_mean": zcr_value,
            "voice_spectral_centroid_mean": 0.0,
            "voice_duration_sec": float(signal.shape[0] / sample_rate) if sample_rate else 0.0,
        }
    except Exception:
        # Si todo falla, se retorna neutro para no bloquear pipeline.
        return {
            "voice_rms_mean": 0.0,
            "voice_rms_std": 0.0,
            "voice_zcr_mean": 0.0,
            "voice_spectral_centroid_mean": 0.0,
            "voice_duration_sec": 0.0,
        }
