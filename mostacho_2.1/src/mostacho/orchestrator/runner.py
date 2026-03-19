"""Orquestador: une vision (InsightFace), voz y biometria contra servicio TF."""

from __future__ import annotations

# argparse para CLI simple.
import argparse
# base64 para enviar imagen por HTTP.
import base64
# json para imprimir salida legible.
import json
# Path para rutas de archivos.
from pathlib import Path
# typing para contratos explícitos.
from typing import Any, Dict

# httpx para cliente HTTP síncrono.
import httpx

# Extractores locales de features.
from mostacho.features.voice import extract_voice_features
from mostacho.features.biometrics import extract_biometrics_features
# Configuración central de endpoints.
from mostacho.settings import load_settings
from mostacho.vision.face_subprocess import FaceSubprocessClient


def _read_image_as_b64(image_path: Path) -> str:
    """Lee un archivo de imagen y lo convierte a base64."""

    # Se leen bytes crudos del archivo.
    raw = image_path.read_bytes()
    # Se codifica a base64 para payload JSON.
    return base64.b64encode(raw).decode("utf-8")


def run_once(image_path: Path, audio_path: Path, biometrics_path: Path) -> Dict[str, Any]:
    """Ejecuta una corrida de inferencia multimodal de extremo a extremo."""

    # Se cargan URLs de servicios desde settings/env.
    settings = load_settings()
    # Timeout para llamadas HTTP de inferencia.
    timeout = httpx.Timeout(30.0)

    # Imagen codificada una sola vez para ambos modos.
    image_b64 = _read_image_as_b64(image_path)

    # Se crea cliente HTTP reutilizable (para tf_service).
    with httpx.Client(timeout=timeout) as client:
        if settings.face_mode == "subprocess":
            with FaceSubprocessClient.from_settings(settings) as face_client:
                face_payload = face_client.infer(image_b64=image_b64)
        else:
            # Se envía imagen al servicio de cara.
            face_response = client.post(
                f"{settings.face_service_url}/infer",
                json={"image_b64": image_b64},
            )
            # Se valida respuesta HTTP.
            face_response.raise_for_status()
            # Se decodifica JSON de salida.
            face_payload = face_response.json()

        # Se extraen features de voz desde archivo.
        voice_features = extract_voice_features(audio_path)
        # Se extraen features biométricas desde archivo.
        biometric_features = extract_biometrics_features(biometrics_path)

        # Se arma payload multimodal para el servicio TF.
        fusion_payload = {
            "vision_features": face_payload.get("vision_features", {}),
            "voice_features": voice_features,
            "biometric_features": biometric_features,
        }

        # Se envía al servicio de fusión neuronal.
        tf_response = client.post(
            f"{settings.tf_service_url}/predict/fusion",
            json=fusion_payload,
        )
        # Se valida respuesta HTTP.
        tf_response.raise_for_status()
        # Se parsea JSON final.
        tf_payload = tf_response.json()

    # Se retorna paquete completo para logging/diagnóstico.
    return {
        "face": face_payload,
        "voice_features": voice_features,
        "biometric_features": biometric_features,
        "fusion": tf_payload,
    }


def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI del orquestador."""

    # Parser con descripción de uso.
    parser = argparse.ArgumentParser(description="Orquestador multimodal Mostacho")
    # Ruta de imagen para análisis facial.
    parser.add_argument("--image", required=True, type=Path, help="Ruta a imagen de entrada")
    # Ruta de audio para features de voz.
    parser.add_argument("--audio", required=True, type=Path, help="Ruta a audio de entrada")
    # Ruta de biometría tabular.
    parser.add_argument("--biometrics", required=True, type=Path, help="Ruta a CSV/TXT biométrico")
    # Se devuelven argumentos parseados.
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI para una inferencia integrada."""

    # Se leen argumentos del usuario.
    args = parse_args()
    # Se ejecuta corrida única.
    result = run_once(args.image, args.audio, args.biometrics)
    # Se imprime JSON legible para depuración.
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    # Ejecución directa como script.
    main()
