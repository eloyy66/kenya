"""CLI utilitaria para indexado de datasets y tareas rápidas de proyecto."""

from __future__ import annotations

# argparse para definir subcomandos.
import argparse
# json para salida legible en terminal.
import json

# indexador central de datasets.
from mostacho.data.catalog import build_catalog, write_catalog_json
# settings globales para rutas de salida.
from mostacho.settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    """Construye parser principal con subcomandos."""

    # Parser raíz.
    parser = argparse.ArgumentParser(description="CLI de utilidades Mostacho")
    # Registro de subcomandos.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando para indexar datasets.
    index_cmd = subparsers.add_parser("index-db", help="Indexa datasets de db/")
    # Flag opcional para escribir JSON en artifacts.
    index_cmd.add_argument("--write-json", action="store_true", help="Escribe dataset_index.json")

    # Se retorna parser completo.
    return parser


def cmd_index_db(write_json: bool) -> None:
    """Ejecuta indexado de dataset y emite resumen por consola/archivo."""

    # Se construye catálogo desde rutas actuales.
    catalog = build_catalog()
    # Se imprime resumen por consola en JSON.
    print(json.dumps({
        "vision": catalog.vision.__dict__,
        "voice": catalog.voice.__dict__,
        "biometrics": catalog.biometrics.__dict__,
    }, indent=2, ensure_ascii=True))

    # Si se solicitó persistencia, se guarda índice en artifacts.
    if write_json:
        settings = load_settings()
        output_path = settings.artifacts_root / "indexes" / "dataset_index.json"
        write_catalog_json(output_path)
        print(f"Indice guardado en: {output_path}")


def main() -> None:
    """Entrada principal de la CLI."""

    # Parseo de argumentos y subcomando.
    args = build_parser().parse_args()

    # Dispatch de subcomandos soportados.
    if args.command == "index-db":
        cmd_index_db(write_json=args.write_json)


if __name__ == "__main__":
    # Ejecuta CLI cuando se llama como módulo/script.
    main()
