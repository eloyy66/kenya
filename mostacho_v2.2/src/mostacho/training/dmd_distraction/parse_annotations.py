from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from .config import PipelineConfig


@dataclass(slots=True)
class AnnotationInterval:
    start_sec: float
    end_sec: float
    original_label: str
    normalized_label: str
    binary_label: Optional[int]


@dataclass(slots=True)
class ParsedAnnotation:
    intervals: list[AnnotationInterval] = field(default_factory=list)
    discovered_labels: set[str] = field(default_factory=set)
    unknown_labels: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def _is_driver_action(interval: AnnotationInterval) -> bool:
        normalized = (interval.normalized_label or "").strip().lower()
        original = (interval.original_label or "").strip().lower()
        return normalized.startswith("driver actions ") or "driver_actions/" in original

    @staticmethod
    def _pick_most_specific(intervals: list[AnnotationInterval]) -> AnnotationInterval:
        # Prefer narrower temporal segments (usually more specific than broad context labels),
        # then longer normalized text as a soft tie-breaker.
        def _sort_key(interval: AnnotationInterval) -> tuple[float, int]:
            duration = max(0.0, interval.end_sec - interval.start_sec)
            label_len = len(interval.normalized_label or "")
            return (duration, -label_len)

        return sorted(intervals, key=_sort_key)[0]

    def label_at(self, timestamp_sec: float) -> tuple[Optional[int], str]:
        overlaps = [interval for interval in self.intervals if interval.start_sec <= timestamp_sec <= interval.end_sec]
        if not overlaps:
            return None, ""

        known = [interval for interval in overlaps if interval.binary_label is not None]
        if not known:
            return None, ""

        positives = [interval for interval in known if interval.binary_label == 1]
        if positives:
            chosen = self._pick_most_specific(positives)
            return chosen.binary_label, chosen.original_label

        driver_actions = [interval for interval in known if self._is_driver_action(interval)]
        if driver_actions:
            chosen = self._pick_most_specific(driver_actions)
            return chosen.binary_label, chosen.original_label

        chosen = self._pick_most_specific(known)
        return chosen.binary_label, chosen.original_label


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _find_first_key(obj: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obj:
            return obj[key]
    return None


def _extract_label(obj: dict[str, Any]) -> Optional[str]:
    label = _find_first_key(
        obj,
        (
            "label",
            "class",
            "action",
            "activity",
            "state",
            "distraction",
            "event",
            "name",
            "type",
            "value",
        ),
    )
    if isinstance(label, str):
        return label
    if isinstance(label, dict):
        candidate = _find_first_key(label, ("name", "label", "class"))
        if isinstance(candidate, str):
            return candidate
    return None


def _extract_time_fields(obj: dict[str, Any], fps: float) -> tuple[Optional[float], Optional[float]]:
    start_sec = _to_float(
        _find_first_key(
            obj,
            (
                "start_sec",
                "start_time",
                "start",
                "from",
                "begin",
                "timestamp_start",
            ),
        )
    )
    end_sec = _to_float(
        _find_first_key(
            obj,
            (
                "end_sec",
                "end_time",
                "end",
                "to",
                "stop",
                "timestamp_end",
            ),
        )
    )
    frame_start = _to_float(_find_first_key(obj, ("start_frame", "frame_start")))
    frame_end = _to_float(_find_first_key(obj, ("end_frame", "frame_end")))
    timestamp = _to_float(_find_first_key(obj, ("timestamp", "time", "t_sec", "sec")))

    if start_sec is None and frame_start is not None and fps > 0:
        start_sec = frame_start / fps
    if end_sec is None and frame_end is not None and fps > 0:
        end_sec = frame_end / fps
    if start_sec is None and timestamp is not None:
        start_sec = timestamp
    if end_sec is None and timestamp is not None:
        end_sec = timestamp
    return start_sec, end_sec


def _collect_event_dicts(node: Any, sink: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        label = _extract_label(node)
        has_time_like = any(
            k in node
            for k in (
                "start",
                "start_sec",
                "start_time",
                "from",
                "begin",
                "timestamp",
                "time",
                "end",
                "end_sec",
                "end_time",
                "to",
                "stop",
                "start_frame",
                "end_frame",
            )
        )
        if label is not None and has_time_like:
            sink.append(node)
        for value in node.values():
            _collect_event_dicts(value, sink)
        return
    if isinstance(node, list):
        for value in node:
            _collect_event_dicts(value, sink)


def parse_annotation_file(
    annotation_path: Path,
    cfg: PipelineConfig,
    video_duration_sec: float,
    fps: float,
    default_segment_len: float = 0.5,
) -> ParsedAnnotation:
    parsed = ParsedAnnotation()
    if not annotation_path.exists():
        raise FileNotFoundError(f"No existe JSON de anotacion: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON invalido en {annotation_path}: {exc}") from exc

    event_dicts: list[dict[str, Any]] = []
    _collect_event_dicts(data, event_dicts)

    # DMD OpenLABEL common path: openlabel.actions[*] with frame_intervals + type.
    openlabel = data.get("openlabel") if isinstance(data, dict) else None
    if isinstance(openlabel, dict):
        actions = openlabel.get("actions")
        if isinstance(actions, dict):
            for action in actions.values():
                if not isinstance(action, dict):
                    continue
                action_label = action.get("type") or action.get("name")
                if not isinstance(action_label, str):
                    ad = action.get("action_data", {})
                    if isinstance(ad, dict):
                        text_entries = ad.get("text", [])
                        if isinstance(text_entries, list):
                            for item in text_entries:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("name") in {"label", "class", "name", "type", "action"} and isinstance(item.get("val"), str):
                                    action_label = item["val"]
                                    break
                frame_intervals = action.get("frame_intervals", [])
                if not isinstance(frame_intervals, list) or not frame_intervals:
                    continue
                for fi in frame_intervals:
                    if not isinstance(fi, dict):
                        continue
                    fs = _to_float(fi.get("frame_start"))
                    fe = _to_float(fi.get("frame_end"))
                    if fs is None and fe is None:
                        continue
                    entry: dict[str, Any] = {"label": action_label}
                    if fs is not None:
                        entry["start_frame"] = fs
                    if fe is not None:
                        entry["end_frame"] = fe
                    event_dicts.append(entry)

    if not event_dicts and isinstance(data, dict):
        top_label = _extract_label(data)
        if top_label is not None:
            event_dicts.append({"label": top_label, "start_sec": 0.0, "end_sec": video_duration_sec})
            parsed.warnings.append("No se encontraron eventos temporales; se uso etiqueta global para todo el video.")

    for event in event_dicts:
        label = _extract_label(event)
        if label is None:
            continue
        start_sec, end_sec = _extract_time_fields(event, fps=fps)
        if start_sec is None and end_sec is None:
            continue
        if start_sec is None:
            start_sec = max(0.0, end_sec - default_segment_len)
        if end_sec is None:
            end_sec = start_sec + default_segment_len
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        start_sec = max(0.0, start_sec)
        end_sec = min(max(start_sec, end_sec), video_duration_sec) if video_duration_sec > 0 else max(start_sec, end_sec)

        binary_label, normalized = cfg.map_label_to_binary(label)
        parsed.discovered_labels.add(normalized or str(label))
        if binary_label is None:
            parsed.unknown_labels.add(normalized or str(label))
        parsed.intervals.append(
            AnnotationInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                original_label=str(label),
                normalized_label=normalized,
                binary_label=binary_label,
            )
        )

    parsed.intervals.sort(key=lambda x: (x.start_sec, x.end_sec))
    if not parsed.intervals:
        parsed.warnings.append("No se pudieron construir intervalos validos.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeccion de un rgb_ann_distraction.json")
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--video-duration-sec", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    cfg = PipelineConfig()
    parsed = parse_annotation_file(
        annotation_path=args.json_path,
        cfg=cfg,
        video_duration_sec=args.video_duration_sec,
        fps=args.fps,
    )
    print(f"Intervalos: {len(parsed.intervals)}")
    print(f"Labels detectados ({len(parsed.discovered_labels)}): {sorted(parsed.discovered_labels)}")
    if parsed.unknown_labels:
        print(f"Labels desconocidos ({len(parsed.unknown_labels)}): {sorted(parsed.unknown_labels)}")
    for warning in parsed.warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
