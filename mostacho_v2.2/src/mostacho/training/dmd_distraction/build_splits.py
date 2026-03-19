from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import PipelineConfig


def _resolve_group_key(df: pd.DataFrame, strategy: str) -> pd.Series:
    strategy = strategy.lower().strip()
    if strategy == "subject":
        if "subject_id" not in df.columns:
            raise RuntimeError("labels.csv no contiene columna subject_id")
        return df["subject_id"].astype(str)
    if strategy == "session":
        if not {"subject_id", "session_id"}.issubset(df.columns):
            raise RuntimeError("labels.csv no contiene columnas subject_id/session_id")
        return df["subject_id"].astype(str) + "::" + df["session_id"].astype(str)
    if strategy == "sequence":
        if "sequence_id" not in df.columns:
            raise RuntimeError("labels.csv no contiene columna sequence_id")
        return df["sequence_id"].astype(str)
    raise ValueError(f"split_strategy invalida: {strategy}")


def _split_groups(groups_df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str], set[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio debe sumar 1.0")

    groups = groups_df["group_id"].tolist()
    group_labels = groups_df["group_label"].tolist()
    if len(groups) < 3:
        raise RuntimeError("No hay suficientes grupos para crear split 80/10/10 sin leakage.")

    def _safe_stratify(labels: list[int], n_holdout: int) -> list[int] | None:
        if len(set(labels)) <= 1:
            return None
        if n_holdout < len(set(labels)):
            return None
        min_class_count = min(labels.count(c) for c in set(labels))
        if min_class_count < 2:
            return None
        return labels

    n_groups = len(groups)
    # For subject-level evaluation, keep at least 2 groups in val and 2 in test
    # whenever the dataset size allows it.
    min_holdout_groups = 4 if n_groups >= 8 else 2
    n_temp = int(round(n_groups * (1.0 - train_ratio)))
    n_temp = max(min_holdout_groups, min(n_groups - 1, n_temp))

    stratify_main = _safe_stratify(group_labels, n_holdout=n_temp)
    if stratify_main is not None:
        train_groups, temp_groups = train_test_split(
            groups,
            test_size=n_temp,
            random_state=seed,
            stratify=stratify_main,
        )
    else:
        shuffled = groups_df.sample(frac=1.0, random_state=seed)["group_id"].tolist()
        temp_groups = shuffled[:n_temp]
        train_groups = shuffled[n_temp:]

    temp_df = groups_df[groups_df["group_id"].isin(temp_groups)].copy()
    temp_groups_list = temp_df["group_id"].tolist()
    if len(temp_groups_list) < 2:
        raise RuntimeError("Split invalido: no hay suficientes grupos para separar val/test.")

    if len(temp_groups_list) >= 4:
        n_val = int(round(n_groups * val_ratio))
        n_val = max(2, min(len(temp_groups_list) - 2, n_val))
        n_test = len(temp_groups_list) - n_val
        if n_test < 2:
            n_test = 2
            n_val = len(temp_groups_list) - n_test
    else:
        n_val = int(round(n_groups * val_ratio))
        n_val = max(1, min(len(temp_groups_list) - 1, n_val))
        n_test = len(temp_groups_list) - n_val
        if n_test < 1:
            n_test = 1
            n_val = len(temp_groups_list) - 1

    temp_labels = temp_df["group_label"].tolist()
    stratify_temp = _safe_stratify(temp_labels, n_holdout=n_test)
    if stratify_temp is not None:
        val_groups, test_groups = train_test_split(
            temp_groups_list,
            test_size=n_test,
            random_state=seed,
            stratify=stratify_temp,
        )
    else:
        shuffled_temp = temp_df.sample(frac=1.0, random_state=seed)["group_id"].tolist()
        test_groups = shuffled_temp[:n_test]
        val_groups = shuffled_temp[n_test:]
        if not val_groups:
            val_groups = test_groups[:1]
            test_groups = test_groups[1:]

    return set(train_groups), set(val_groups), set(test_groups)


def verify_no_group_overlap(df: pd.DataFrame, group_col: str) -> dict[str, bool]:
    train_groups = set(df.loc[df["split"] == "train", group_col].astype(str))
    val_groups = set(df.loc[df["split"] == "val", group_col].astype(str))
    test_groups = set(df.loc[df["split"] == "test", group_col].astype(str))
    return {
        "train_val_disjoint": len(train_groups.intersection(val_groups)) == 0,
        "train_test_disjoint": len(train_groups.intersection(test_groups)) == 0,
        "val_test_disjoint": len(val_groups.intersection(test_groups)) == 0,
    }


def _maybe_balance_training(df: pd.DataFrame, seed: int, downsample_majority: bool, oversample_minority: bool) -> pd.DataFrame:
    if not downsample_majority and not oversample_minority:
        return df
    train_df = df[df["split"] == "train"].copy()
    others_df = df[df["split"] != "train"].copy()
    class0 = train_df[train_df["label"] == 0]
    class1 = train_df[train_df["label"] == 1]
    if class0.empty or class1.empty:
        return df
    if downsample_majority:
        n = min(len(class0), len(class1))
        class0 = class0.sample(n=n, random_state=seed)
        class1 = class1.sample(n=n, random_state=seed)
        train_df = pd.concat([class0, class1], ignore_index=True)
    elif oversample_minority:
        if len(class0) > len(class1):
            class1 = class1.sample(n=len(class0), replace=True, random_state=seed)
        else:
            class0 = class0.sample(n=len(class1), replace=True, random_state=seed)
        train_df = pd.concat([class0, class1], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pd.concat([train_df, others_df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir split train/val/test sin leakage por sujeto o sesion.")
    parser.add_argument("--processed-output-dir", type=Path, default=PipelineConfig().processed_output_dir)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--split-strategy", type=str, default=PipelineConfig().split_strategy, choices=["subject", "session", "sequence"])
    parser.add_argument("--seed", type=int, default=PipelineConfig().seed)
    parser.add_argument("--downsample-majority", action="store_true")
    parser.add_argument("--oversample-minority", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        processed_output_dir=args.processed_output_dir,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )
    cfg.ensure_dirs()
    labels_csv_path = args.labels_csv or cfg.labels_csv_path
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"No existe labels.csv: {labels_csv_path}")

    df = pd.read_csv(labels_csv_path)
    if df.empty:
        raise RuntimeError("labels.csv esta vacio.")
    if "label" not in df.columns:
        raise RuntimeError("labels.csv no tiene columna label")
    df["label"] = df["label"].astype(int)

    df["group_id"] = _resolve_group_key(df, cfg.split_strategy)
    group_stats = (
        df.groupby("group_id", as_index=False)
        .agg(group_label=("label", "max"), rows=("label", "count"))
        .sort_values("group_id")
    )
    train_groups, val_groups, test_groups = _split_groups(
        groups_df=group_stats,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    df["split"] = "train"
    df.loc[df["group_id"].isin(val_groups), "split"] = "val"
    df.loc[df["group_id"].isin(test_groups), "split"] = "test"

    df = _maybe_balance_training(
        df=df,
        seed=cfg.seed,
        downsample_majority=args.downsample_majority,
        oversample_minority=args.oversample_minority,
    )

    overlap_checks = verify_no_group_overlap(df, "group_id")
    if not all(overlap_checks.values()):
        raise RuntimeError(f"Leakage detectado en split: {overlap_checks}")

    class_dist = {}
    for split_name in ("train", "val", "test"):
        subset = df[df["split"] == split_name]
        class_dist[split_name] = {
            "rows": int(len(subset)),
            "class_0": int((subset["label"] == 0).sum()),
            "class_1": int((subset["label"] == 1).sum()),
        }

    train_labels = df.loc[df["split"] == "train", "label"].to_numpy()
    classes = np.array([0, 1], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    output_csv = cfg.split_csv_path
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["split", "group_id", "sequence_id", "timestamp_sec"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    report = {
        "split_strategy": cfg.split_strategy,
        "ratios": {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio},
        "n_groups": int(group_stats.shape[0]),
        "n_rows": int(df.shape[0]),
        "group_counts": {"train": len(train_groups), "val": len(val_groups), "test": len(test_groups)},
        "class_distribution": class_dist,
        "class_weights": class_weights,
        "overlap_checks": overlap_checks,
        "downsample_majority": bool(args.downsample_majority),
        "oversample_minority": bool(args.oversample_minority),
    }
    with cfg.split_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Split CSV: {output_csv}")
    print(f"Split report: {cfg.split_report_path}")
    print(json.dumps(report["class_distribution"], indent=2))


if __name__ == "__main__":
    main()
