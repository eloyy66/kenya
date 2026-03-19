from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Optional


def normalize_label_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _alias_set(values: Iterable[str]) -> set[str]:
    return {normalize_label_text(v) for v in values}


def _alias_phrase_matches(normalized_label: str, normalized_alias: str) -> bool:
    if not normalized_label or not normalized_alias:
        return False
    if normalized_label == normalized_alias:
        return True

    label_tokens = normalized_label.split()
    alias_tokens = normalized_alias.split()
    if not label_tokens or not alias_tokens:
        return False

    # Single-word aliases require exact token match (avoid substring bugs like "eat" in "backseat").
    if len(alias_tokens) == 1:
        return alias_tokens[0] in label_tokens

    # Multi-word aliases must appear as contiguous phrase tokens.
    n = len(alias_tokens)
    for idx in range(0, len(label_tokens) - n + 1):
        if label_tokens[idx : idx + n] == alias_tokens:
            return True
    return False


@dataclass(slots=True)
class PipelineConfig:
    dataset_root: Path = Path("/Users/usuario/kenya/db/artificialvision/dmd")
    processed_output_dir: Path = Path("/Users/usuario/kenya/mostacho_v2.2/artifacts/dmd_processed")
    frame_sampling_seconds: float = 0.75
    image_width: int = 224
    image_height: int = 224
    jpeg_quality: int = 95
    duplicate_diff_threshold: float = 4.5
    duplicate_hash_size: int = 16
    split_strategy: str = "subject"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    backbone_name: str = "MobileNetV3Small"
    batch_size: int = 32
    freeze_epochs: int = 10
    fine_tune_epochs: int = 10
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    fine_tune_layers: int = 40
    dropout_rate: float = 0.25
    hidden_units: int = 128
    min_precision_for_threshold: float = 0.90
    max_representative_samples: int = 500
    use_hands_support: bool = False
    hands_motion_threshold: float = 4.0
    cache_dataset: bool = False
    positive_label_aliases: set[str] = field(
        default_factory=lambda: _alias_set(
            [
                "phone",
                "mobile phone",
                "texting",
                "calling",
                "talking on phone",
                "phone usage",
                "using phone",
                "phonecall left",
                "phonecall right",
                "texting left",
                "texting right",
                "makeup",
                "applying makeup",
                "hair and makeup",
                "eating",
                "drinking",
                "drink",
                "eat",
                "driver actions drinking",
                "driver actions eating",
            ]
        )
    )
    negative_label_aliases: set[str] = field(
        default_factory=lambda: _alias_set(
            [
                "normal driving",
                "attentive driving",
                "safe driving",
                "no distraction",
                "normal",
                "attentive",
                "safe",
                "focused driving",
                "driver actions safe drive",
                "gaze on road",
                "hands using wheel",
                "hand on gear",
                "change gear",
                "reach side",
                "reach backseat",
                "talking to passenger",
                "talking talking",
                "standstill or waiting",
                "radio",
                "unclassified",
            ]
        )
    )

    def ensure_dirs(self) -> None:
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.processed_output_dir / "splits").mkdir(parents=True, exist_ok=True)
        (self.processed_output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.processed_output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.processed_output_dir / "tflite").mkdir(parents=True, exist_ok=True)

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)

    @property
    def labels_csv_path(self) -> Path:
        return self.processed_output_dir / "labels.csv"

    @property
    def split_csv_path(self) -> Path:
        return self.processed_output_dir / "splits" / "labels_with_split.csv"

    @property
    def split_report_path(self) -> Path:
        return self.processed_output_dir / "splits" / "split_report.json"

    @property
    def models_dir(self) -> Path:
        return self.processed_output_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.processed_output_dir / "reports"

    @property
    def tflite_dir(self) -> Path:
        return self.processed_output_dir / "tflite"

    def map_label_to_binary(self, raw_label: Optional[str]) -> tuple[Optional[int], str]:
        if raw_label is None:
            return None, ""
        normalized = normalize_label_text(str(raw_label))
        if not normalized:
            return None, normalized
        if normalized in self.positive_label_aliases:
            return 1, normalized
        if normalized in self.negative_label_aliases:
            return 0, normalized

        pos_matches = [alias for alias in self.positive_label_aliases if _alias_phrase_matches(normalized, alias)]
        neg_matches = [alias for alias in self.negative_label_aliases if _alias_phrase_matches(normalized, alias)]

        if pos_matches and not neg_matches:
            return 1, normalized
        if neg_matches and not pos_matches:
            return 0, normalized
        if pos_matches and neg_matches:
            # Resolve ambiguous hits by preferring the most specific phrase.
            # If tie, bias to negative to reduce false positives in production.
            pos_specificity = max(len(alias.split()) for alias in pos_matches)
            neg_specificity = max(len(alias.split()) for alias in neg_matches)
            if neg_specificity >= pos_specificity:
                return 0, normalized
            return 1, normalized
        return None, normalized
