"""Utilidades de extracción de features biométricas desde CSV/TXT."""

from __future__ import annotations

# Path para manejo de rutas portable.
from pathlib import Path
# typing para contratos de salida.
from typing import Dict

# numpy para estadistica numerica.
import numpy as np
# pandas para carga tabular flexible.
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    """Carga tabla con separador inferido y fallback de texto plano."""

    # Primer intento: CSV estándar con coma.
    try:
        return pd.read_csv(path)
    except Exception:
        pass

    # Segundo intento: delimitador por espacios o tabuladores.
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        pass

    # Tercer intento: archivo de una columna numérica.
    data = np.loadtxt(path, dtype=float)
    # Si el array es 1D, se convierte a DataFrame con nombre base.
    if data.ndim == 1:
        return pd.DataFrame({"value": data})
    # Si es 2D, se crean columnas automáticas.
    return pd.DataFrame(data)


def extract_biometrics_features(path: Path, max_columns: int = 8) -> Dict[str, float]:
    """Extrae estadisticas resumidas para fusion multimodal."""

    # Se normaliza entrada de ruta.
    path = Path(path)
    # Si no existe archivo, se retorna vector neutro.
    if not path.exists():
        return {"bio_available": 0.0}

    # Se carga la tabla de señales.
    table = _read_table(path)
    # Se seleccionan columnas numéricas únicamente.
    numeric = table.select_dtypes(include=[np.number])
    # Si no hay columnas numéricas, se retorna indicador mínimo.
    if numeric.empty:
        return {"bio_available": 0.0}

    # Se limita cantidad de columnas para mantener vector manejable.
    selected_cols = list(numeric.columns[:max_columns])
    # Se prepara diccionario de salida.
    features: Dict[str, float] = {"bio_available": 1.0, "bio_num_rows": float(len(numeric))}

    # Se agregan estadisticas por columna seleccionada.
    for column in selected_cols:
        # Serie limpia sin NaNs para estadística robusta.
        values = pd.to_numeric(numeric[column], errors="coerce").dropna().to_numpy(dtype=float)
        # Si no hay datos válidos, se continúa.
        if values.size == 0:
            continue
        # Prefijo normalizado por nombre de columna.
        key = f"bio_{str(column).lower()}"
        # Se calculan métricas de primer orden.
        features[f"{key}_mean"] = float(np.mean(values))
        features[f"{key}_std"] = float(np.std(values))
        features[f"{key}_min"] = float(np.min(values))
        features[f"{key}_max"] = float(np.max(values))

    # Se entrega vector final.
    return features
