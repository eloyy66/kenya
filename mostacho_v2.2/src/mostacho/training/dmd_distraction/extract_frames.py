from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .discover_dataset import SequenceRecord, build_index_df, discover_sequences
from .parse_annotations import ParsedAnnotation, parse_annotation_file


@dataclass(slots=True)
class VideoMeta:
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


def read_video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    duration_sec = float(total_frames / fps) if total_frames > 0 else 0.0
    return VideoMeta(fps=fps, total_frames=total_frames, duration_sec=duration_sec, width=width, height=height)


def _frame_signature(frame_bgr: np.ndarray, hash_size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).flatten()


def _is_near_duplicate(prev_sig: Optional[np.ndarray], current_sig: np.ndarray, threshold: float) -> bool:
    if prev_sig is None:
        return False
    diff = np.mean(np.abs(prev_sig - current_sig))
    return bool(diff < threshold)


def _hands_motion_score(cap_hands: cv2.VideoCapture, timestamp_sec: float) -> float:
    cap_hands.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec - 0.15) * 1000.0)
    ok0, frame0 = cap_hands.read()
    cap_hands.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ok1, frame1 = cap_hands.read()
    if not ok0 or not ok1:
        return -1.0
    g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g0, g1)
    return float(np.mean(diff))


def _sample_timestamps(duration_sec: float, step_sec: float) -> np.ndarray:
    if duration_sec <= 0:
        return np.array([], dtype=np.float32)
    ts = np.arange(0.0, duration_sec, step_sec, dtype=np.float32)
    if len(ts) == 0:
        ts = np.array([0.0], dtype=np.float32)
    return ts


def _load_existing_labels(labels_csv_path: Path) -> pd.DataFrame:
    if not labels_csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(labels_csv_path)
    required = {"source_video", "sequence_id"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df


def extract_sequence_frames(
    record: SequenceRecord,
    cfg: PipelineConfig,
    overwrite: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sequence_dir = Path(record.sequence_dir)
    face_video = Path(record.rgb_face_path)
    ann_path = Path(record.annotation_path)

    image_out_dir = cfg.processed_output_dir / "images" / record.sequence_id
    image_out_dir.mkdir(parents=True, exist_ok=True)

    cap_face = cv2.VideoCapture(str(face_video))
    if not cap_face.isOpened():
        raise RuntimeError(f"No se pudo abrir rgb_face.mp4: {face_video}")

    meta = read_video_meta(face_video)
    parsed: ParsedAnnotation = parse_annotation_file(
        annotation_path=ann_path,
        cfg=cfg,
        video_duration_sec=meta.duration_sec,
        fps=meta.fps,
        default_segment_len=cfg.frame_sampling_seconds,
    )

    cap_hands: Optional[cv2.VideoCapture] = None
    has_hands = bool(record.rgb_hands_path)
    if cfg.use_hands_support and has_hands:
        cap_hands = cv2.VideoCapture(record.rgb_hands_path)
        if not cap_hands.isOpened():
            cap_hands = None

    timestamps = _sample_timestamps(meta.duration_sec, cfg.frame_sampling_seconds)
    rows: list[dict[str, object]] = []
    skipped_unknown = 0
    skipped_dup = 0
    skipped_unreadable = 0
    prev_sig: Optional[np.ndarray] = None

    for timestamp in timestamps:
        cap_face.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame_bgr = cap_face.read()
        if not ok or frame_bgr is None:
            skipped_unreadable += 1
            continue
        binary_label, original_label = parsed.label_at(float(timestamp))
        if binary_label is None:
            skipped_unknown += 1
            continue

        signature = _frame_signature(frame_bgr, hash_size=cfg.duplicate_hash_size)
        if _is_near_duplicate(prev_sig, signature, threshold=cfg.duplicate_diff_threshold):
            skipped_dup += 1
            continue
        prev_sig = signature

        frame_name = f"t_{float(timestamp):08.3f}_label_{binary_label}.jpg"
        image_path = image_out_dir / frame_name
        if image_path.exists() and not overwrite:
            pass
        else:
            write_ok = cv2.imwrite(str(image_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
            if not write_ok:
                skipped_unreadable += 1
                continue

        hands_motion = -1.0
        if cap_hands is not None:
            hands_motion = _hands_motion_score(cap_hands, float(timestamp))

        rows.append(
            {
                "image_path": str(image_path),
                "label": int(binary_label),
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "sequence_id": record.sequence_id,
                "timestamp_sec": float(timestamp),
                "source_video": str(face_video),
                "source_json": str(ann_path),
                "original_label": original_label,
                "normalized_label": cfg.map_label_to_binary(original_label)[1],
                "has_hands_video": has_hands,
                "hands_motion_score": float(hands_motion),
            }
        )

    cap_face.release()
    if cap_hands is not None:
        cap_hands.release()

    stats = {
        "sequence_id": record.sequence_id,
        "source_video": str(face_video),
        "saved_frames": len(rows),
        "skipped_unknown_labels": skipped_unknown,
        "skipped_duplicates": skipped_dup,
        "skipped_unreadable": skipped_unreadable,
        "unknown_labels": sorted(parsed.unknown_labels),
        "warnings": parsed.warnings,
        "video_meta": asdict(meta),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraer frames etiquetados desde DMD rgb_face.mp4 + annotations.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Ruta de dataset DMD. Repetible para combinar multiples roots.",
    )
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--frame-sampling-seconds", type=float, default=PipelineConfig().frame_sampling_seconds)
    parser.add_argument("--duplicate-diff-threshold", type=float, default=PipelineConfig().duplicate_diff_threshold)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-hands-support", action="store_true")
    args = parser.parse_args()

    dataset_roots = args.dataset_root or [PipelineConfig().dataset_root]
    cfg = PipelineConfig(
        dataset_root=dataset_roots[0],
        processed_output_dir=args.processed_output_dir,
        frame_sampling_seconds=args.frame_sampling_seconds,
        duplicate_diff_threshold=args.duplicate_diff_threshold,
        use_hands_support=args.use_hands_support,
    )
    cfg.ensure_dirs()

    records = discover_sequences(dataset_roots)
    if not records:
        raise RuntimeError("No se encontraron secuencias validas para extraer.")

    dataset_index = build_index_df(records)
    dataset_index_path = cfg.processed_output_dir / "dataset_index.csv"
    dataset_index.to_csv(dataset_index_path, index=False)

    existing = _load_existing_labels(cfg.labels_csv_path)
    processed_videos = set(existing["source_video"].tolist()) if not existing.empty and not args.overwrite else set()

    all_rows: list[dict[str, object]] = []
    all_stats: list[dict[str, object]] = []
    global_unknown_labels: set[str] = set()
    skipped_sequences = 0

    for record in tqdm(records, desc="Extracting DMD face frames"):
        if not args.overwrite and record.rgb_face_path in processed_videos:
            skipped_sequences += 1
            continue
        try:
            rows, stats = extract_sequence_frames(record=record, cfg=cfg, overwrite=args.overwrite)
            all_rows.extend(rows)
            all_stats.append(stats)
            global_unknown_labels.update(stats["unknown_labels"])
        except Exception as exc:
            all_stats.append(
                {
                    "sequence_id": record.sequence_id,
                    "source_video": record.rgb_face_path,
                    "saved_frames": 0,
                    "error": str(exc),
                }
            )

    if existing.empty or args.overwrite:
        final_df = pd.DataFrame(all_rows)
    else:
        final_df = pd.concat([existing, pd.DataFrame(all_rows)], ignore_index=True)

    if final_df.empty:
        raise RuntimeError("No se generaron filas etiquetadas. Revisa mapeo de etiquetas y JSON.")

    final_df = final_df.drop_duplicates(subset=["image_path"], keep="last")
    final_df.to_csv(cfg.labels_csv_path, index=False)

    extraction_summary = {
        "dataset_roots": [str(root.expanduser().resolve()) for root in dataset_roots],
        "processed_output_dir": str(cfg.processed_output_dir),
        "sequences_discovered": len(records),
        "sequences_skipped_resumable": skipped_sequences,
        "rows_new": len(all_rows),
        "rows_total_labels_csv": len(final_df),
        "unknown_labels": sorted(global_unknown_labels),
        "per_sequence_stats": all_stats,
    }
    summary_path = cfg.processed_output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(extraction_summary, f, indent=2, ensure_ascii=False)

    print(f"Dataset index: {dataset_index_path}")
    print(f"Labels CSV: {cfg.labels_csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows total: {len(final_df)}")
    print(f"Unknown labels detectados: {sorted(global_unknown_labels)}")


if __name__ == "__main__":
    main()
