from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "6-Behavioral-Validation" / "data"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "6-Behavioral-Validation" / "output" / "tables"
STEP4_DIR = PROJECT_ROOT / "Data" / "Step4"
KEY = ["channel", "user_id", "item_id", "timestamp"]
FINAL_MATCHED_NAME = "final_matched_samples_with_step4.parquet"


def load_step4() -> pd.DataFrame:
    src = pd.read_pickle(STEP4_DIR / "src_all.pkl").copy()
    rec = pd.read_pickle(STEP4_DIR / "rec_all.pkl").copy()
    src["channel"] = "S"
    rec["channel"] = "R"
    common = ["channel", "user_id", "item_id", "timestamp", "rec_his", "src_session_his"]
    if "search_session_id" in src.columns:
        common_src = common + ["search_session_id"]
    else:
        common_src = common
    src = src[common_src].copy()
    rec["search_session_id"] = np.nan
    rec = rec[common + ["search_session_id"]].copy()
    step4 = pd.concat([src, rec], ignore_index=True)
    for col in ["user_id", "item_id", "timestamp"]:
        step4[col] = pd.to_numeric(step4[col], errors="coerce")
    return step4


def normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["user_id", "item_id", "timestamp", "page_time", "position", "click"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def distribution(prefix: str, df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["page_time", "position"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        out[f"{prefix}_{col}_missing_rate"] = float(vals.isna().mean()) if len(vals) else np.nan
        out[f"{prefix}_{col}_zero_rate"] = float((vals == 0).mean()) if len(vals) else np.nan
        out[f"{prefix}_{col}_min"] = float(vals.min()) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_p25"] = float(vals.quantile(0.25)) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_p75"] = float(vals.quantile(0.75)) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_p95"] = float(vals.quantile(0.95)) if vals.notna().any() else np.nan
        out[f"{prefix}_{col}_max"] = float(vals.max()) if vals.notna().any() else np.nan
    return out


def match_summary(step4: pd.DataFrame, behavior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_keep_cols = KEY + [
        "source_config",
        "split_group",
        "source_row",
        "detail_pos",
        "raw_user_id",
        "raw_item_id",
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
        "query",
    ]
    behavior_keep_cols = [col for col in behavior_keep_cols if col in behavior.columns]
    behavior_unique = behavior[behavior_keep_cols].drop_duplicates(KEY, keep="first").copy()

    merged = step4.merge(
        behavior_unique,
        on=KEY,
        how="left",
        indicator=True,
        suffixes=("", "_behavior"),
    )

    rows = []
    for channel, g in merged.groupby("channel", dropna=False, sort=True):
        matched = g["_merge"].eq("both")
        row = {
            "channel": channel,
            "step4_rows": len(g),
            "matched_rows": int(matched.sum()),
            "unmatched_rows": int((~matched).sum()),
            "match_rate": float(matched.mean()) if len(g) else np.nan,
            "unique_users": g["user_id"].nunique(dropna=True),
            "unique_items": g["item_id"].nunique(dropna=True),
        }
        row.update(distribution("matched", g[matched]))
        rows.append(row)

    all_matched = merged["_merge"].eq("both")
    row = {
        "channel": "ALL",
        "step4_rows": len(merged),
        "matched_rows": int(all_matched.sum()),
        "unmatched_rows": int((~all_matched).sum()),
        "match_rate": float(all_matched.mean()) if len(merged) else np.nan,
        "unique_users": merged["user_id"].nunique(dropna=True),
        "unique_items": merged["item_id"].nunique(dropna=True),
    }
    row.update(distribution("matched", merged[all_matched]))
    rows.append(row)
    return pd.DataFrame(rows), merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Match extracted Qilin behavioral signals to local Step4 clicked samples.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    args = parser.parse_args()

    args.table_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.data_dir / "clicked_behavior_samples.parquet"
    if not behavior_path.exists():
        raise FileNotFoundError(f"Missing {behavior_path}. Run 1-extract-hf-behavioral-signals.py first.")

    step4 = load_step4()
    behavior = normalize_behavior(pd.read_parquet(behavior_path))

    summary, merged = match_summary(step4, behavior)

    summary.to_csv(args.table_dir / "step4_match_summary.csv", index=False)
    final_path = args.data_dir / FINAL_MATCHED_NAME
    merged.to_parquet(final_path, index=False)

    print(f"Step4 rows: {len(step4):,}")
    print(f"Behavior clicked rows: {len(behavior):,}")
    print(summary.to_string(index=False))
    print(f"Saved final matched table: {final_path}")
    print(f"Saved match summary: {args.table_dir / 'step4_match_summary.csv'}")


if __name__ == "__main__":
    main()
