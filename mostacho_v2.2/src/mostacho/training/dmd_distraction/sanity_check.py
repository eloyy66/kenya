from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from .config import PipelineConfig
from .discover_dataset import discover_sequences
from .extract_frames import read_video_meta
from .parse_annotations import parse_annotation_file


def print_annotation_label_summary(cfg: PipelineConfig, dataset_roots: list[Path], max_sequences: int = 200) -> None:
    records = discover_sequences(dataset_roots)
    label_counts: dict[str, int] = {}
    unknown_labels: set[str] = set()
    scanned = 0
    for record in records:
        if scanned >= max_sequences:
            break
        scanned += 1
        meta = read_video_meta(Path(record.rgb_face_path))
        parsed = parse_annotation_file(
            annotation_path=Path(record.annotation_path),
            cfg=cfg,
            video_duration_sec=meta.duration_sec,
            fps=meta.fps,
            default_segment_len=cfg.frame_sampling_seconds,
        )
        for interval in parsed.intervals:
            key = interval.normalized_label or interval.original_label
            label_counts[key] = label_counts.get(key, 0) + 1
        unknown_labels.update(parsed.unknown_labels)
    print(f"[Sanity] Secuencias escaneadas para labels: {scanned}")
    print("[Sanity] Top labels detectados:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:40]:
        print(f"  - {label}: {count}")
    print(f"[Sanity] Labels desconocidos: {sorted(unknown_labels)}")


def make_preview_montage(labels_csv: Path, output_path: Path, num_samples: int = 12) -> None:
    df = pd.read_csv(labels_csv)
    if df.empty:
        raise RuntimeError("labels.csv vacio, no se puede crear preview.")
    sample_df = (
        df.sample(n=min(num_samples, len(df)), random_state=42)
        .sort_values("label")
        .reset_index(drop=True)
    )

    tile_w, tile_h = 320, 180
    cols = 4
    rows = (len(sample_df) + cols - 1) // cols
    canvas = 255 * (cv2.UMat(rows * tile_h, cols * tile_w, cv2.CV_8UC3).get())

    for idx, row in sample_df.iterrows():
        image_path = str(row["image_path"])
        img = cv2.imread(image_path)
        if img is None:
            img = 255 * (cv2.UMat(tile_h, tile_w, cv2.CV_8UC3).get())
            cv2.putText(img, "Unreadable", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            img = cv2.resize(img, (tile_w, tile_h))

        lbl = int(row["label"])
        text = f"{'DISTRACTED' if lbl == 1 else 'NORMAL'} | {row.get('original_label', '')}"
        color = (0, 0, 255) if lbl == 1 else (0, 180, 0)
        cv2.rectangle(img, (0, 0), (tile_w - 1, 25), (0, 0, 0), -1)
        cv2.putText(img, text[:42], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        r = idx // cols
        c = idx % cols
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        canvas[y0:y1, x0:x1] = img

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[Sanity] Preview montage guardado en: {output_path}")


def verify_split_leakage(split_csv: Path) -> None:
    if not split_csv.exists():
        print(f"[Sanity] Split CSV no existe, se omite leakage check: {split_csv}")
        return
    df = pd.read_csv(split_csv)
    required_cols = {"split", "subject_id", "sequence_id"}
    if not required_cols.issubset(df.columns):
        print("[Sanity] Split CSV sin columnas suficientes para leakage check.")
        return

    def _values(split: str, col: str) -> set[str]:
        return set(df.loc[df["split"] == split, col].astype(str))

    train_subjects = _values("train", "subject_id")
    val_subjects = _values("val", "subject_id")
    test_subjects = _values("test", "subject_id")
    train_sequences = _values("train", "sequence_id")
    val_sequences = _values("val", "sequence_id")
    test_sequences = _values("test", "sequence_id")

    subject_overlap = {
        "train_val": sorted(train_subjects.intersection(val_subjects)),
        "train_test": sorted(train_subjects.intersection(test_subjects)),
        "val_test": sorted(val_subjects.intersection(test_subjects)),
    }
    sequence_overlap = {
        "train_val": sorted(train_sequences.intersection(val_sequences)),
        "train_test": sorted(train_sequences.intersection(test_sequences)),
        "val_test": sorted(val_sequences.intersection(test_sequences)),
    }
    print("[Sanity] Subject overlap:", {k: len(v) for k, v in subject_overlap.items()})
    print("[Sanity] Sequence overlap:", {k: len(v) for k, v in sequence_overlap.items()})
    if any(len(v) > 0 for v in sequence_overlap.values()):
        raise RuntimeError(f"Leakage por sequence detectado: {sequence_overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks: labels descubiertos, preview y leakage de splits.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Ruta de dataset DMD. Repetible para combinar multiples roots.",
    )
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--max-sequences", type=int, default=200)
    parser.add_argument("--num-preview", type=int, default=12)
    args = parser.parse_args()

    dataset_roots = args.dataset_root or [PipelineConfig().dataset_root]
    cfg = PipelineConfig(dataset_root=dataset_roots[0], processed_output_dir=args.processed_output_dir)
    cfg.ensure_dirs()
    print_annotation_label_summary(cfg, dataset_roots=dataset_roots, max_sequences=args.max_sequences)

    if not cfg.labels_csv_path.exists():
        raise FileNotFoundError(f"No existe labels.csv para preview: {cfg.labels_csv_path}")
    preview_path = cfg.reports_dir / "sanity_preview.jpg"
    make_preview_montage(cfg.labels_csv_path, preview_path, num_samples=args.num_preview)
    verify_split_leakage(cfg.split_csv_path)


if __name__ == "__main__":
    main()
