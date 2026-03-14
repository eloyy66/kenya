"""Indexador de datasets para vision, voz y biometria."""

from __future__ import annotations

# json se usa para exportar indices reproducibles.
import json
# dataclass simplifica estructuras de resumen.
from dataclasses import dataclass, asdict
# Path facilita recorrido de carpetas.
from pathlib import Path
# Counter permite contar etiquetas de forma compacta.
from collections import Counter
# typing para contratos claros.
from typing import Dict, Iterable, List

# Se reutiliza configuracion global de rutas.
from mostacho.settings import load_settings


# Extensiones validas para imagenes de entrenamiento/inferencia.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Extensiones validas para audio.
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
# Extensiones candidatas para datos tabulares/serializados biométricos.
BIOMETRIC_EXTENSIONS = {".csv", ".txt", ".pkl", ".zip", ".parquet"}

# Mapeo de codigo CREMA-D -> etiqueta semantica.
CREMA_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


@dataclass
class DatasetSection:
    """Resumen de una seccion del dataset."""

    # Nombre legible de la seccion.
    name: str
    # Total de archivos detectados.
    total_files: int
    # Conteo por clase/etiqueta.
    labels: Dict[str, int]
    # Muestra corta de archivos para inspeccion manual.
    sample_files: List[str]


@dataclass
class DatasetCatalog:
    """Catalogo global de datasets usados por Mostacho."""

    # Informacion de vision artificial.
    vision: DatasetSection
    # Informacion de voz.
    voice: DatasetSection
    # Informacion de biometria/wearables.
    biometrics: DatasetSection


def _iter_files(base_dirs: Iterable[Path], allowed_ext: set[str]) -> List[Path]:
    """Recorre directorios y devuelve archivos por extension."""

    # Se acumulan rutas para garantizar orden estable.
    collected: List[Path] = []
    # Se recorre cada directorio candidato.
    for base_dir in base_dirs:
        # Si no existe, se ignora para tolerar datasets parciales.
        if not base_dir.exists():
            continue
        # rglob permite recorrer recursivamente subdirectorios.
        for path in base_dir.rglob("*"):
            # Solo se agregan archivos con extension permitida.
            if path.is_file() and path.suffix.lower() in allowed_ext:
                collected.append(path)
    # Orden alfabetico para resultados deterministas.
    return sorted(collected)


def _existing_dirs(candidates: Iterable[Path]) -> List[Path]:
    """Filtra y retorna solo directorios existentes en orden."""

    # Se conserva orden de prioridad definido por el llamador.
    return [path for path in candidates if path.exists()]


def build_vision_section(db_root: Path) -> DatasetSection:
    """Construye resumen de datasets de vision."""

    # Se prioriza estructura nueva y se mantiene fallback a estructura anterior.
    base_dirs = _existing_dirs(
        [
            db_root / "artificialvision" / "generalcontext-emotions2",
            db_root / "artificialvision" / "generalcontext-emotions",
            db_root / "artificialvision" / "DrivFace",
            db_root / "processed_data",
            db_root / "Data",
            db_root / "DrivFace",
        ]
    )
    # Se recolectan archivos de imagen.
    files = _iter_files(base_dirs, IMAGE_EXTENSIONS)

    # Se cuentan etiquetas a partir del nombre del directorio padre.
    labels = Counter(path.parent.name for path in files)
    # Se arma muestra corta relativa a `db_root`.
    sample = [str(path.relative_to(db_root)) for path in files[:20]]

    # Se devuelve seccion compacta y serializable.
    return DatasetSection(
        name="vision",
        total_files=len(files),
        labels=dict(labels),
        sample_files=sample,
    )


def _crema_label_from_filename(path: Path) -> str:
    """Extrae etiqueta de emocion desde nombre de archivo CREMA-D."""

    # Ejemplo: 1008_TAI_HAP_XX.wav -> codigo HAP.
    parts = path.stem.split("_")
    # Si el formato no coincide, se marca como desconocido.
    if len(parts) < 3:
        return "unknown"
    # Se normaliza codigo para mapa de etiquetas.
    code = parts[2].upper()
    # Se devuelve etiqueta textual o unknown.
    return CREMA_EMOTION_MAP.get(code, "unknown")


def build_voice_section(db_root: Path) -> DatasetSection:
    """Construye resumen de dataset de voz."""

    # CREMA-D vive en esta ruta en la estructura actual.
    crema_dir = db_root / "audio" / "crema-d"
    # Se recolectan archivos de audio soportados.
    files = _iter_files([crema_dir], AUDIO_EXTENSIONS)

    # Se cuenta cada emocion inferida por nombre de archivo.
    labels = Counter(_crema_label_from_filename(path) for path in files)
    # Se prepara muestra de rutas relativas.
    sample = [str(path.relative_to(db_root)) for path in files[:20]]

    # Se devuelve estructura tipada de salida.
    return DatasetSection(
        name="voice",
        total_files=len(files),
        labels=dict(labels),
        sample_files=sample,
    )


def _biometric_label_from_path(path: Path) -> str:
    """Genera etiqueta coarse para archivos biométricos."""

    # Se normaliza ruta a minúsculas para reglas simples.
    normalized = str(path).lower()
    # Regla para identificar WESAD.
    if "wesad" in normalized:
        return "wesad"
    # Regla para identificar SWELL u otros de stress dataset.
    if "swell" in normalized:
        return "swell"
    # Regla para identificar datos EDA.
    if "eda" in normalized:
        return "eda"
    # Regla para identificar datos HRV.
    if "hrv" in normalized:
        return "hrv"
    # Si no aplica regla, se marca como generic.
    return "generic"


def build_biometrics_section(db_root: Path) -> DatasetSection:
    """Construye resumen de biometria (WESAD + stress)."""

    # Se prioriza estructura nueva y se mantiene fallback a la anterior.
    base_dirs = _existing_dirs(
        [
            db_root / "biometrics" / "WESAD",
            db_root / "biometrics" / "stress" / "dataset",
            db_root / "WESAD",
            db_root / "stress" / "dataset",
        ]
    )
    # Se recolectan extensiones tabulares/serializadas.
    files = _iter_files(base_dirs, BIOMETRIC_EXTENSIONS)

    # Se contabilizan etiquetas coarse por ruta.
    labels = Counter(_biometric_label_from_path(path) for path in files)
    # Se arma muestra para revision manual.
    sample = [str(path.relative_to(db_root)) for path in files[:20]]

    # Se retorna seccion consolidada.
    return DatasetSection(
        name="biometrics",
        total_files=len(files),
        labels=dict(labels),
        sample_files=sample,
    )


def build_catalog() -> DatasetCatalog:
    """Construye catalogo completo de datasets desde la configuracion activa."""

    # Se cargan rutas globales del proyecto.
    settings = load_settings()
    # Se crea resumen de vision.
    vision = build_vision_section(settings.db_root)
    # Se crea resumen de voz.
    voice = build_voice_section(settings.db_root)
    # Se crea resumen de biometria.
    biometrics = build_biometrics_section(settings.db_root)

    # Se retorna catalogo global.
    return DatasetCatalog(vision=vision, voice=voice, biometrics=biometrics)


def write_catalog_json(output_path: Path) -> Path:
    """Escribe el catalogo a JSON y retorna la ruta final."""

    # Se construye el catalogo actual.
    catalog = build_catalog()
    # Se garantiza existencia de la carpeta padre.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Se serializa dataclass a JSON legible.
    output_path.write_text(json.dumps(asdict(catalog), indent=2, ensure_ascii=True), encoding="utf-8")
    # Se devuelve ruta para logging superior.
    return output_path
