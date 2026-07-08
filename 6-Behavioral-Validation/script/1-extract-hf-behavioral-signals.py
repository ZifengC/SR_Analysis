from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "6-Behavioral-Validation" / "data"
STEP1_DIR = PROJECT_ROOT / "Data" / "Step1"
HF_DATASET = "THUIR/Qilin"

CONFIGS = [
    {
        "config": "search_train",
        "channel": "S",
        "detail_col": "search_result_details_with_idx",
        "timestamp_key": "search_timestamp",
        "split_group": "train",
    },
    {
        "config": "search_test",
        "channel": "S",
        "detail_col": "search_result_details_with_idx",
        "timestamp_key": "search_timestamp",
        "split_group": "test",
    },
    {
        "config": "recommendation_train",
        "channel": "R",
        "detail_col": "rec_result_details_with_idx",
        "timestamp_key": "request_timestamp",
        "split_group": "train",
    },
    {
        "config": "recommendation_test",
        "channel": "R",
        "detail_col": "rec_result_details_with_idx",
        "timestamp_key": "request_timestamp",
        "split_group": "test",
    },
]


def scalar(value: Any, default=np.nan):
    if value is None:
        return default
    return value


def load_local_id_maps() -> tuple[dict[int, int], dict[int, int]]:
    user_dict = pickle.load(open(STEP1_DIR / "user_dict.pkl", "rb"))
    note_dict = pickle.load(open(STEP1_DIR / "note_dict.pkl", "rb"))
    return dict(user_dict), dict(note_dict)


def map_id(mapping: dict[int, int], raw_value: Any):
    if raw_value is None or pd.isna(raw_value):
        return np.nan
    try:
        return mapping.get(int(raw_value), np.nan)
    except Exception:
        return np.nan


def explode_config(
    config_spec: dict[str, str],
    user_map: dict[int, int],
    note_map: dict[int, int],
    max_rows: int | None = None,
) -> pd.DataFrame:
    config = config_spec["config"]
    detail_col = config_spec["detail_col"]
    timestamp_key = config_spec["timestamp_key"]

    print(f"Loading {HF_DATASET} / {config}")
    ds = load_dataset(HF_DATASET, config, split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    df = ds.to_pandas()
    print(f"  source rows: {len(df):,}")

    rows: list[dict[str, Any]] = []
    optional_source_cols = ["query", "session_idx", "search_idx", "request_idx", "query_from_type"]
    for row_idx, row in df.iterrows():
        details = row.get(detail_col)
        if details is None:
            continue
        source_values = {col: row.get(col, np.nan) for col in optional_source_cols if col in df.columns}
        for detail_pos, detail in enumerate(details):
            if not isinstance(detail, dict):
                continue
            raw_user_id = scalar(row.get("user_idx"))
            raw_item_id = scalar(detail.get("note_idx"))
            rows.append(
                {
                    "source_config": config,
                    "split_group": config_spec["split_group"],
                    "channel": config_spec["channel"],
                    "source_row": int(row_idx),
                    "detail_pos": int(detail_pos),
                    "raw_user_id": raw_user_id,
                    "raw_item_id": raw_item_id,
                    "user_id": map_id(user_map, raw_user_id),
                    "item_id": map_id(note_map, raw_item_id),
                    "timestamp": scalar(detail.get(timestamp_key)),
                    "timestamp_source": timestamp_key,
                    "page_time": scalar(detail.get("page_time")),
                    "position": scalar(detail.get("position")),
                    "click": scalar(detail.get("click"), 0),
                    "like": scalar(detail.get("like"), 0),
                    "collect": scalar(detail.get("collect"), 0),
                    "comment": scalar(detail.get("comment"), 0),
                    "share": scalar(detail.get("share"), 0),
                    **source_values,
                }
            )

    out = pd.DataFrame(rows)
    print(f"  item-level rows: {len(out):,}")
    return out


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "source_row",
        "detail_pos",
        "raw_user_id",
        "raw_item_id",
        "user_id",
        "item_id",
        "timestamp",
        "page_time",
        "position",
        "click",
        "like",
        "collect",
        "comment",
        "share",
        "session_idx",
        "search_idx",
        "request_idx",
        "query_from_type",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["raw_user_id", "raw_item_id", "user_id", "item_id", "source_row", "detail_pos"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in ["click", "like", "collect", "comment", "share"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("Int64")

    sort_cols = ["channel", "user_id", "timestamp", "source_config", "source_row", "detail_pos"]
    sort_cols = [col for col in sort_cols if col in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qilin item-level behavioral signals from Hugging Face.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows-per-config", type=int, default=None, help="Debug option; process only first N source rows.")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    user_map, note_map = load_local_id_maps()
    print(f"Loaded local id maps: users={len(user_map):,}, notes={len(note_map):,}")

    parts = [explode_config(spec, user_map, note_map, args.max_rows_per_config) for spec in CONFIGS]
    raw = normalize_types(pd.concat(parts, ignore_index=True))
    clicked = raw[pd.to_numeric(raw["click"], errors="coerce") == 1].copy().reset_index(drop=True)

    raw_path = args.data_dir / "raw_behavior_events.parquet"
    clicked_path = args.data_dir / "clicked_behavior_samples.parquet"
    raw.to_parquet(raw_path, index=False)
    clicked.to_parquet(clicked_path, index=False)

    print(f"Saved {raw_path} rows={len(raw):,}")
    print(f"Saved {clicked_path} rows={len(clicked):,}")


if __name__ == "__main__":
    main()
