from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import PipelineConfig


@dataclass(slots=True)
class SequenceRecord:
    sequence_id: str
    dataset_root: str
    sequence_dir: str
    subject_id: str
    session_id: str
    rgb_face_path: str
    annotation_path: str
    rgb_hands_path: str
    has_hands: bool


_SUBJECT_PATTERNS = (
    re.compile(r"(subject|subj|driver|participant|user)[_\- ]?(\d+)", re.IGNORECASE),
)
_SESSION_PATTERNS = (
    re.compile(r"(session|seq|sequence|drive|trip|run)[_\- ]?(\d+)", re.IGNORECASE),
)
_SESSION_TOKEN_PATTERN = re.compile(r"^s(\d{1,3})$", re.IGNORECASE)
_ACTOR_TOKEN_PATTERN = re.compile(r"^\d{1,4}$")
_ACTOR_FROM_STEM_PATTERN = re.compile(r"(?:^|_)(\d+)_s\d+(?:_|$)", re.IGNORECASE)
_SESSION_FROM_STEM_PATTERN = re.compile(r"(?:^|_)s(\d+)(?:_|$)", re.IGNORECASE)


def _infer_identifier(parts: Iterable[str], patterns: tuple[re.Pattern[str], ...], fallback: str) -> str:
    for part in parts:
        for pattern in patterns:
            match = pattern.search(part)
            if match:
                if match.lastindex and match.lastindex >= 2:
                    return f"{match.group(1).lower()}_{match.group(2)}"
                return match.group(1).lower()
    return fallback


def infer_subject_session(sequence_dir: Path, dataset_root: Path, face_video: Path | None = None) -> tuple[str, str]:
    relative_parts = list(sequence_dir.relative_to(dataset_root).parts)

    session_token: str | None = None
    session_part_idx: int | None = None
    for idx in range(len(relative_parts) - 1, -1, -1):
        part = relative_parts[idx]
        if _SESSION_TOKEN_PATTERN.match(part):
            session_token = part.lower()
            session_part_idx = idx
            break

    if session_token is None and face_video is not None:
        m_session = _SESSION_FROM_STEM_PATTERN.search(face_video.stem)
        if m_session:
            session_token = f"s{int(m_session.group(1))}"

    session_fallback = relative_parts[-1] if relative_parts else "unknown_session"
    if session_token is not None:
        session_id = f"session_{session_token}"
    else:
        session_id = _infer_identifier(relative_parts, _SESSION_PATTERNS, fallback=f"session_{session_fallback}")

    actor_token: str | None = None
    if face_video is not None:
        m_actor = _ACTOR_FROM_STEM_PATTERN.search(face_video.stem)
        if m_actor:
            actor_token = str(int(m_actor.group(1)))

    if actor_token is None:
        search_parts = relative_parts if session_part_idx is None else relative_parts[:session_part_idx]
        for part in reversed(search_parts):
            if _ACTOR_TOKEN_PATTERN.match(part):
                actor_token = str(int(part))
                break

    if actor_token is not None:
        subject_id = f"actor_{actor_token}"
    else:
        subject_id = _infer_identifier(relative_parts, _SUBJECT_PATTERNS, fallback=f"subject_{relative_parts[0] if relative_parts else 'unknown'}")
    return subject_id, session_id


def _sanitize_token(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    return value.strip("_")


def build_sequence_id(sequence_dir: Path, dataset_root: Path, face_video: Path | None = None) -> str:
    relative = sequence_dir.relative_to(dataset_root).as_posix()
    safe = re.sub(r"[^a-zA-Z0-9/_-]+", "_", relative)
    safe = safe.replace("/", "__")
    base = safe.strip("_")
    if face_video is None:
        return base
    return f"{base}__{_sanitize_token(face_video.stem)}"


def _find_best_sibling(sequence_dir: Path, suffix: str, face_stem: str) -> Path | None:
    candidates = sorted(sequence_dir.glob(f"*{suffix}"))
    if not candidates:
        return None
    expected_prefix = face_stem[: -len("rgb_face")] if face_stem.endswith("rgb_face") else face_stem
    if expected_prefix:
        for candidate in candidates:
            if candidate.stem.startswith(expected_prefix):
                return candidate
    return candidates[0]


def _as_dataset_roots(dataset_roots: Path | Iterable[Path]) -> list[Path]:
    if isinstance(dataset_roots, Path):
        roots = [dataset_roots]
    else:
        roots = list(dataset_roots)
    if not roots:
        raise ValueError("Debes proveer al menos un dataset root.")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def discover_sequences(dataset_roots: Path | Iterable[Path]) -> list[SequenceRecord]:
    roots = _as_dataset_roots(dataset_roots)
    missing = [root for root in roots if not root.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset root(s) no existe(n): {missing}")

    records: list[SequenceRecord] = []
    seen_sequence_ids: set[str] = set()
    for root_idx, dataset_root in enumerate(roots):
        root_tag = _sanitize_token(dataset_root.name) or f"dataset_{root_idx + 1}"
        for face_video in dataset_root.rglob("*rgb_face.mp4"):
            sequence_dir = face_video.parent
            annotation = _find_best_sibling(sequence_dir, suffix="rgb_ann_distraction.json", face_stem=face_video.stem)
            if annotation is None:
                continue
            hands_path = _find_best_sibling(sequence_dir, suffix="rgb_hands.mp4", face_stem=face_video.stem)
            subject_id, session_id = infer_subject_session(sequence_dir, dataset_root, face_video=face_video)
            sequence_id = f"{root_tag}__{build_sequence_id(sequence_dir, dataset_root, face_video=face_video)}"
            if sequence_id in seen_sequence_ids:
                suffix = 1
                while f"{sequence_id}__dup_{suffix}" in seen_sequence_ids:
                    suffix += 1
                sequence_id = f"{sequence_id}__dup_{suffix}"
            seen_sequence_ids.add(sequence_id)
            records.append(
                SequenceRecord(
                    sequence_id=sequence_id,
                    dataset_root=str(dataset_root),
                    sequence_dir=str(sequence_dir),
                    subject_id=subject_id,
                    session_id=session_id,
                    rgb_face_path=str(face_video),
                    annotation_path=str(annotation),
                    rgb_hands_path=str(hands_path) if hands_path is not None else "",
                    has_hands=hands_path is not None,
                )
            )
    records.sort(key=lambda r: r.sequence_id)
    return records


def build_index_df(records: list[SequenceRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def main() -> None:
    parser = argparse.ArgumentParser(description="Descubrir secuencias validas DMD (face + annotation).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Ruta de dataset DMD. Repetible para combinar multiples roots.",
    )
    parser.add_argument("--output-index", type=Path, default=None)
    args = parser.parse_args()

    dataset_roots = args.dataset_root or [PipelineConfig().dataset_root]
    cfg = PipelineConfig(dataset_root=dataset_roots[0])
    records = discover_sequences(dataset_roots)
    if not records:
        raise RuntimeError("No se encontraron secuencias validas con rgb_face.mp4 + rgb_ann_distraction.json")
    df = build_index_df(records)

    output_index = args.output_index
    if output_index is None:
        cfg.ensure_dirs()
        output_index = cfg.processed_output_dir / "dataset_index.csv"
    output_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_index, index=False)

    print(f"Secuencias validas: {len(df)}")
    print(f"Subjects unicos: {df['subject_id'].nunique()}")
    print(f"Sessions unicas: {df['session_id'].nunique()}")
    print(f"Con rgb_hands.mp4: {int(df['has_hands'].sum())}")
    print(f"Dataset roots combinados: {len(set(df['dataset_root'].astype(str).tolist()))}")
    print(f"Index guardado en: {output_index}")


if __name__ == "__main__":
    main()
