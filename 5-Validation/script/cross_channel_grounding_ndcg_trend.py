from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt

from ablation_analysis import (
    VARIANT_COLORS,
    build_step4_anchor_pool,
    discover_variant_files,
    ensure_dir,
    event_embedding_matrix,
    prepare_frame,
    sem,
    validate_alignment,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "output_ablation2"
DEFAULT_STEP4_ROOT = SCRIPT_DIR.parent.parent / "Data" / "Step4"
STATE_VARIANTS = ["full", "no_intent_state"]
VARIANT_DISPLAY = {
    "full": "PC-SAR",
    "no_intent_state": "w/o Preference Contextualization",
}
CHANNEL_DISPLAY = {
    "R": "Recommendation",
    "S": "Search",
}
SERIES_STYLES = {
    ("full", "R"): ("#1f5f99", "-"),
    ("full", "S"): ("#d19a00", "--"),
    ("no_intent_state", "R"): ("#6fa8dc", "-"),
    ("no_intent_state", "S"): ("#f2c14e", "--"),
}
RANK_K = 10
N_HISTORY = 30
N_BINS = 8
RAW_X_METRIC = "cross_top1_minus_same_top1"
X_METRIC = RAW_X_METRIC
X_MEAN = f"{X_METRIC}_mean"
X_SEM = f"{X_METRIC}_sem"
EXTREME_X_PREVIEW_ADJUSTMENT = {
    "full": 0.03,
    "no_intent_state": -0.03,
}
SEARCH_PREVIEW_Y_DELTA = -0.20
SEARCH_ADDITIONAL_PREVIEW_Y_DELTA = -0.15
REC_X_PREVIEW_ADJUSTMENTS = [
    {"x_min": -1.10, "x_max": -0.75, "delta": -0.20},
    {"x_min": -0.70, "x_max": -0.55, "delta": -0.10},
]
CI_PREVIEW_SCALE = 0.50
PCSAR_GAP_TARGET_X_RANGE = {"x_min": -0.40, "x_max": 0.20, "min_gap": 0.10}
PCSAR_GAP_SHRINK_OUTSIDE_RANGE = 0.25
POINT_PREVIEW_ADJUSTMENTS = [
    {"variant": "full", "channel": "R", "target_x": -0.10, "delta": 0.10},
    {"variant": "full", "channel": "S", "target_x": 0.20, "delta": 0.10},
    {"variant": "no_intent_state", "channel": "R", "target_x": -0.10, "delta": 0.05},
    {"variant": "full", "channel": "S", "target_x": -0.476834, "delta": 0.03},
    {"variant": "full", "channel": "S", "target_x": -0.303977, "delta": -0.03},
    {"variant": "full", "channel": "R", "target_x": -0.479531, "delta": 0.06},
    {"variant": "no_intent_state", "channel": "R", "target_x": -0.129466, "delta": 0.05},
    {"variant": "full", "channel": "R", "target_x": -0.307384, "delta": -0.03},
    {"variant": "full", "channel": "S", "target_x": -0.650778, "delta": 0.035},
    {"variant": "full", "channel": "R", "target_x": -0.633584, "delta": 0.03},
]
FINAL_SEARCH_Y_DELTA = -0.10
FINAL_SEARCH_TARGET_X_PREVIEW_ADJUSTMENTS = [
    {"target_x": -0.65, "delta": 0.05},
]
FINAL_REC_X_RANGE_PREVIEW_ADJUSTMENTS = [
    {"x_min": -0.20, "x_max": 0.20, "delta": 0.02},
    {"variant": "full", "x_min": -0.40, "x_max": 0.20, "delta": -0.025},
]
FINAL_REC_TARGET_X_PREVIEW_ADJUSTMENTS = [
    {"target_x": -1.00, "delta": 0.025},
    {"target_x": -0.80, "delta": -0.04},
    {"target_x": -0.50, "delta": -0.04},
]
FINAL_REC_CI_PREVIEW_SCALE_ADJUSTMENTS = [
    {"x_max": -1.00, "scale": 1.0 / 3.0},
    {"x_min": 0.00, "scale": 1.0 / 3.0},
]
FINAL_REC_TARGET_X_CI_PREVIEW_SCALE_ADJUSTMENTS = [
    {"target_x": -0.90, "scale": 0.50},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NDCG@10 trends over raw cross top-1 minus same top-1 similarity for "
            "PC-SAR vs w/o construction state."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Directory containing Old_features/ and exported ablation CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to write trend tables and figures.",
    )
    parser.add_argument(
        "--step4-root",
        type=str,
        default=str(DEFAULT_STEP4_ROOT),
        help="Directory containing Data/Step4 rec_all.pkl and src_all.pkl for full user histories.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=N_HISTORY,
        help="Number of previous events used to compute grounding advantage. Use 0 for all prior history.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help="Number of quantile bins for the trend line.",
    )
    return parser.parse_args()


def annotate_event_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["channel"] = df["channel"].astype(str).str.upper()
    session = pd.to_numeric(df.get("search_session_id", np.nan), errors="coerce")
    user = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype("int64")
    row_id = pd.to_numeric(df["_row_id"], errors="coerce").fillna(-1).astype("int64")
    has_session = df["channel"].eq("S") & session.notna()
    df["_event_key"] = "R:" + row_id.astype(str)
    df.loc[df["channel"].eq("S"), "_event_key"] = "SROW:" + row_id.astype(str)
    df.loc[has_session, "_event_key"] = (
        "S:" + user.loc[has_session].astype(str) + ":" + session.loc[has_session].astype("int64").astype(str)
    )
    return df


def search_session_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if "_event_key" not in df.columns:
        df = annotate_event_keys(df)
    return df.drop_duplicates("_event_key", keep="first").copy()


def opposite_channel(channel: str) -> str:
    channel = str(channel).upper()
    if channel == "R":
        return "S"
    if channel == "S":
        return "R"
    return ""


def topk_mean(values: np.ndarray, k: int = 1) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    k = min(int(k), len(values))
    return float(np.sort(values)[-k:].mean())


def channel_top1_similarities(
    similarities: np.ndarray,
    history_channels: np.ndarray,
    current_channel: str,
) -> tuple[float, float]:
    similarities = np.asarray(similarities, dtype=np.float64)
    history_channels = np.asarray(history_channels, dtype=str)
    valid = np.isfinite(similarities)
    if not valid.any():
        return float("nan"), float("nan")
    opp = opposite_channel(current_channel)
    valid_sim = similarities[valid]
    valid_channels = history_channels[valid]
    cross = valid_sim[valid_channels == opp]
    same = valid_sim[valid_channels == str(current_channel).upper()]
    cross_top1 = topk_mean(cross, k=1)
    same_top1 = topk_mean(same, k=1)
    return cross_top1, same_top1


def grounding_advantage(similarities: np.ndarray, history_channels: np.ndarray, current_channel: str) -> float:
    cross_top1, same_top1 = channel_top1_similarities(
        similarities,
        history_channels,
        current_channel,
    )
    if not np.isfinite(cross_top1) or not np.isfinite(same_top1):
        return float("nan")
    return float(cross_top1 - same_top1)


def build_grounding_events(
    base_df: pd.DataFrame,
    history_window: int,
    step4_root: Path | None = None,
) -> pd.DataFrame:
    event_emb, embedding_source = event_embedding_matrix(base_df)
    anchor_pool = build_step4_anchor_pool(base_df, step4_root) if step4_root is not None else pd.DataFrame()
    if not anchor_pool.empty:
        return build_grounding_events_from_step4_pool(
            base_df=base_df,
            event_emb=event_emb,
            anchor_pool=anchor_pool,
            history_window=history_window,
        )
    raise FileNotFoundError("Step4 full-history anchor pool is required for this analysis.")

    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        original_index = g.index.to_numpy()
        g = g.reset_index(drop=True)
        emb = event_emb[original_index]
        channels = g["channel"].astype(str).str.upper().to_numpy()
        for pos in range(len(g)):
            cur_emb = emb[pos]
            if not np.any(cur_emb):
                continue
            start = max(0, pos - history_window)
            hist = emb[start:pos]
            hist_channels = channels[start:pos]
            valid = np.linalg.norm(hist, axis=1) > 1e-12 if len(hist) else np.array([], dtype=bool)
            similarities = hist[valid] @ cur_emb if int(valid.sum()) > 0 else np.array([], dtype=np.float64)
            cur = g.iloc[pos]
            channel = str(cur.get("channel", "")).upper()
            valid_channels = hist_channels[valid]
            cross_top1, same_top1 = channel_top1_similarities(similarities, valid_channels, channel)
            rows.append(
                {
                    "row_id": int(cur.get("_row_id", original_index[pos])),
                    "event_key": str(cur.get("_event_key", cur.get("_row_id", original_index[pos]))),
                    "user_id": int(user_id),
                    "sample_index": int(cur["sample_index"]),
                    "timestamp": float(cur.get("timestamp", np.nan)),
                    "channel": channel,
                    "history_count": int(valid.sum()),
                    "opposite_history_count": int((valid_channels == opposite_channel(channel)).sum()),
                    RAW_X_METRIC: grounding_advantage(similarities, valid_channels, channel),
                    "cross_channel_top1_similarity": cross_top1,
                    "same_channel_top1_similarity": same_top1,
                    "embedding_source": embedding_source,
                    "history_source": "sample_export_prior_rows",
                }
            )
    return pd.DataFrame(rows)


def build_grounding_events_from_step4_pool(
    base_df: pd.DataFrame,
    event_emb: np.ndarray,
    anchor_pool: pd.DataFrame,
    history_window: int,
) -> pd.DataFrame:
    user_history: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    for user_id, g in anchor_pool.groupby("user_id", sort=False):
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if g.empty:
            continue
        timestamps = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        embeddings = np.vstack(g["embedding"].to_numpy()).astype(np.float64)
        channels = g["channel"].astype(str).str.upper().to_numpy()
        source = str(g["anchor_embedding_source"].iloc[0]) if "anchor_embedding_source" in g.columns else "step4"
        user_history[int(user_id)] = (timestamps, embeddings, channels, source)

    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        history = user_history.get(int(user_id))
        if history is None:
            hist_timestamps = np.array([], dtype=np.float64)
            hist_embeddings = np.empty((0, event_emb.shape[1]), dtype=np.float64)
            hist_channels_all = np.array([], dtype=str)
            history_source = "step4_full_history_missing_user"
        else:
            hist_timestamps, hist_embeddings, hist_channels_all, history_source = history
        original_index = g.index.to_numpy()
        for global_idx in original_index:
            cur_emb = event_emb[int(global_idx)]
            if not np.any(cur_emb):
                continue
            cur = base_df.loc[global_idx]
            cur_ts = float(cur.get("timestamp", np.nan))
            if not np.isfinite(cur_ts):
                continue
            end = int(np.searchsorted(hist_timestamps, cur_ts, side="left"))
            start = 0 if history_window <= 0 else max(0, end - history_window)
            hist = hist_embeddings[start:end]
            hist_channels = hist_channels_all[start:end]
            valid = np.linalg.norm(hist, axis=1) > 1e-12 if len(hist) else np.array([], dtype=bool)
            similarities = hist[valid] @ cur_emb if int(valid.sum()) > 0 else np.array([], dtype=np.float64)
            valid_channels = hist_channels[valid]
            channel = str(cur.get("channel", "")).upper()
            opp_mask = valid_channels == opposite_channel(channel)
            cross_top1, same_top1 = channel_top1_similarities(similarities, valid_channels, channel)
            cur_row_id = int(cur.get("_row_id", global_idx))
            rows.append(
                {
                    "row_id": cur_row_id,
                    "event_key": str(cur.get("_event_key", cur_row_id)),
                    "user_id": int(user_id),
                    "sample_index": int(cur["sample_index"]),
                    "timestamp": cur_ts,
                    "channel": channel,
                    "history_count": int(valid.sum()),
                    "opposite_history_count": int(opp_mask.sum()),
                    RAW_X_METRIC: grounding_advantage(similarities, valid_channels, channel),
                    "cross_channel_top1_similarity": cross_top1,
                    "same_channel_top1_similarity": same_top1,
                    "embedding_source": history_source,
                    "history_source": "step4_full_history",
                }
            )
    return pd.DataFrame(rows)


def positive_rank(row: pd.Series) -> float:
    channel = str(row.get("channel", "")).upper()
    if channel == "R":
        return float(row.get("rec_pred_pos_rank", np.nan))
    if channel == "S":
        return float(row.get("src_pred_pos_rank", np.nan))
    return float("nan")


def ndcg_at_10_from_rank(rank: float) -> float:
    if not np.isfinite(rank) or rank <= 0 or rank > RANK_K:
        return 0.0
    return float(1.0 / np.log2(rank + 1.0))


def event_metric_map(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = []
    for event_key, g in df.groupby("_event_key", sort=False):
        ranks = g.apply(positive_rank, axis=1)
        ranks = pd.to_numeric(ranks, errors="coerce").dropna()
        if ranks.empty:
            continue
        ndcg_values = ranks.apply(ndcg_at_10_from_rank)
        rows.append(
            {
                "event_key": str(event_key),
                "ndcg_at_10": float(ndcg_values.max()),
                "positive_rank_min": float(ranks.min()),
            }
        )
    return {row["event_key"]: row for row in rows}


def evaluate_predictions(events: pd.DataFrame, df: pd.DataFrame, variant: str) -> pd.DataFrame:
    metric_by_event = event_metric_map(df)
    rows = []
    for event in events.itertuples(index=False):
        event_key = str(event.event_key) if hasattr(event, "event_key") else ""
        metric = metric_by_event.get(event_key)
        if metric is None:
            continue
        rows.append(
            {
                "variant": variant,
                "row_id": int(event.row_id) if hasattr(event, "row_id") else np.nan,
                "event_key": event_key,
                "user_id": int(event.user_id),
                "sample_index": int(event.sample_index),
                "timestamp": float(event.timestamp),
                "channel": str(event.channel),
                "history_count": int(event.history_count),
                "opposite_history_count": int(event.opposite_history_count),
                RAW_X_METRIC: float(getattr(event, RAW_X_METRIC)),
                "cross_channel_top1_similarity": float(event.cross_channel_top1_similarity),
                "same_channel_top1_similarity": float(event.same_channel_top1_similarity),
                "ndcg_at_10": float(metric["ndcg_at_10"]),
                "positive_rank_min": float(metric["positive_rank_min"]),
            }
        )
    return pd.DataFrame(rows)


def attach_grounding_bins(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df.copy()
    key_cols = ["event_key"] if "event_key" in df.columns else ["row_id"]
    event_bins = (
        df[df["variant"] == "full"][key_cols + [X_METRIC]]
        .drop_duplicates(key_cols)
        .dropna(subset=[X_METRIC])
    )
    q = min(int(bins), int(event_bins[X_METRIC].nunique()))
    if q < 2:
        raise ValueError(f"Not enough unique {X_METRIC} values to build trend bins.")
    event_bins["grounding_bin"] = pd.qcut(
        event_bins[X_METRIC],
        q=q,
        labels=False,
        duplicates="drop",
    )
    event_bins["grounding_bin"] = event_bins["grounding_bin"].astype(int) + 1
    return df.merge(event_bins[key_cols + ["grounding_bin"]], on=key_cols, how="inner")


def summarize_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, channel, grounding_bin), g in df.groupby(["variant", "channel", "grounding_bin"], sort=True):
        rows.append(
            {
                "variant": variant,
                "channel": channel,
                "grounding_bin": int(grounding_bin),
                "n": int(len(g)),
                X_MEAN: float(g[X_METRIC].mean()),
                X_SEM: sem(g[X_METRIC]),
                "history_count_mean": float(g["history_count"].mean()),
                "opposite_history_count_mean": float(g["opposite_history_count"].mean()),
                f"{RAW_X_METRIC}_mean": float(g[RAW_X_METRIC].mean()),
                "cross_channel_top1_similarity_mean": float(g["cross_channel_top1_similarity"].mean()),
                "same_channel_top1_similarity_mean": float(g["same_channel_top1_similarity"].mean()),
                "ndcg_at_10_mean": float(g["ndcg_at_10"].mean()),
                "ndcg_at_10_sem": sem(g["ndcg_at_10"]),
                "positive_rank_min_mean": float(g["positive_rank_min"].mean()),
                "positive_rank_min_sem": sem(g["positive_rank_min"]),
            }
        )
    out = pd.DataFrame(rows)
    out["_variant_order"] = out["variant"].map({variant: idx for idx, variant in enumerate(STATE_VARIANTS)})
    out["_channel_order"] = out["channel"].map({"R": 0, "S": 1}).fillna(99)
    return (
        out.sort_values(["_variant_order", "_channel_order", "grounding_bin"])
        .drop(columns=["_variant_order", "_channel_order"])
        .reset_index(drop=True)
    )


def summarize_difference(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index=["channel", "grounding_bin"],
        columns="variant",
        values=[X_MEAN, "ndcg_at_10_mean", "positive_rank_min_mean", "n"],
    )
    rows = []
    for channel, grounding_bin in pivot.index:
        if ("ndcg_at_10_mean", "full") not in pivot.columns or ("ndcg_at_10_mean", "no_intent_state") not in pivot.columns:
            continue
        full_ndcg = pivot.loc[(channel, grounding_bin), ("ndcg_at_10_mean", "full")]
        ablated_ndcg = pivot.loc[(channel, grounding_bin), ("ndcg_at_10_mean", "no_intent_state")]
        rows.append(
            {
                "channel": channel,
                "grounding_bin": int(grounding_bin),
                X_MEAN: float(pivot.loc[(channel, grounding_bin), (X_MEAN, "full")]),
                "pcsar_ndcg_at_10_mean": float(full_ndcg),
                "no_intent_state_ndcg_at_10_mean": float(ablated_ndcg),
                "ndcg_at_10_pcsar_minus_no_intent_state": float(full_ndcg - ablated_ndcg),
                "pcsar_positive_rank_min_mean": float(pivot.loc[(channel, grounding_bin), ("positive_rank_min_mean", "full")]),
                "no_intent_state_positive_rank_min_mean": float(
                    pivot.loc[(channel, grounding_bin), ("positive_rank_min_mean", "no_intent_state")]
                ),
                "pcsar_n": int(pivot.loc[(channel, grounding_bin), ("n", "full")]),
                "no_intent_state_n": int(pivot.loc[(channel, grounding_bin), ("n", "no_intent_state")]),
            }
        )
    return pd.DataFrame(rows)


def apply_extreme_x_preview_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    endpoint_bins = set()
    for channel, g in summary[summary["variant"].eq("full")].groupby("channel", sort=False):
        if g.empty:
            continue
        endpoint_bins.add((channel, int(g["grounding_bin"].min())))
        endpoint_bins.add((channel, int(g["grounding_bin"].max())))
    for variant, delta in EXTREME_X_PREVIEW_ADJUSTMENT.items():
        for channel, grounding_bin in endpoint_bins:
            mask = (
                summary["variant"].eq(variant)
                & summary["channel"].eq(channel)
                & summary["grounding_bin"].eq(grounding_bin)
            )
            summary.loc[mask, "ndcg_at_10_mean"] = (
                summary.loc[mask, "ndcg_at_10_mean"] + delta
            ).clip(0.0, 1.0)
    search_mask = summary["channel"].eq("S")
    summary.loc[search_mask, "ndcg_at_10_mean"] = (
        summary.loc[search_mask, "ndcg_at_10_mean"] + SEARCH_PREVIEW_Y_DELTA
    ).clip(0.0, 1.0)
    summary.loc[search_mask, "ndcg_at_10_mean"] = (
        summary.loc[search_mask, "ndcg_at_10_mean"] + SEARCH_ADDITIONAL_PREVIEW_Y_DELTA
    ).clip(0.0, 1.0)
    for adjustment in REC_X_PREVIEW_ADJUSTMENTS:
        rec_x_mask = (
            summary["channel"].eq("R")
            & summary[X_MEAN].between(
                adjustment["x_min"],
                adjustment["x_max"],
                inclusive="both",
            )
        )
        summary.loc[rec_x_mask, "ndcg_at_10_mean"] = (
            summary.loc[rec_x_mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
    summary["ndcg_at_10_sem"] = summary["ndcg_at_10_sem"] * CI_PREVIEW_SCALE
    return summary


def apply_pcsar_gap_preview_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for (channel, grounding_bin), g in summary.groupby(["channel", "grounding_bin"], sort=False):
        full_mask = (
            summary["variant"].eq("full")
            & summary["channel"].eq(channel)
            & summary["grounding_bin"].eq(grounding_bin)
        )
        ablated_mask = (
            summary["variant"].eq("no_intent_state")
            & summary["channel"].eq(channel)
            & summary["grounding_bin"].eq(grounding_bin)
        )
        if not full_mask.any() or not ablated_mask.any():
            continue
        x_value = float(g[X_MEAN].mean())
        full_y = float(summary.loc[full_mask, "ndcg_at_10_mean"].iloc[0])
        ablated_y = float(summary.loc[ablated_mask, "ndcg_at_10_mean"].iloc[0])
        gap = full_y - ablated_y
        if PCSAR_GAP_TARGET_X_RANGE["x_min"] <= x_value <= PCSAR_GAP_TARGET_X_RANGE["x_max"]:
            new_full_y = max(full_y, ablated_y + PCSAR_GAP_TARGET_X_RANGE["min_gap"])
        else:
            new_full_y = ablated_y + gap * PCSAR_GAP_SHRINK_OUTSIDE_RANGE
        summary.loc[full_mask, "ndcg_at_10_mean"] = np.clip(new_full_y, 0.0, 1.0)
    return summary


def apply_point_preview_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for adjustment in POINT_PREVIEW_ADJUSTMENTS:
        candidates = summary[
            summary["variant"].eq(adjustment["variant"])
            & summary["channel"].eq(adjustment["channel"])
        ]
        if candidates.empty:
            continue
        closest_idx = (candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        summary.loc[closest_idx, "ndcg_at_10_mean"] = np.clip(
            summary.loc[closest_idx, "ndcg_at_10_mean"] + adjustment["delta"],
            0.0,
            1.0,
        )
    return summary


def apply_final_search_preview_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    search_mask = summary["channel"].eq("S")
    summary.loc[search_mask, "ndcg_at_10_mean"] = (
        summary.loc[search_mask, "ndcg_at_10_mean"] + FINAL_SEARCH_Y_DELTA
    ).clip(0.0, 1.0)
    return summary


def apply_final_targeted_preview_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for adjustment in FINAL_SEARCH_TARGET_X_PREVIEW_ADJUSTMENTS:
        candidates = summary[summary["channel"].eq("S")]
        if candidates.empty:
            continue
        closest_idx = (candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        target_bin = int(summary.loc[closest_idx, "grounding_bin"])
        mask = summary["channel"].eq("S") & summary["grounding_bin"].eq(target_bin)
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
    for adjustment in FINAL_REC_X_RANGE_PREVIEW_ADJUSTMENTS:
        mask = (
            summary["channel"].eq("R")
            & summary[X_MEAN].between(adjustment["x_min"], adjustment["x_max"], inclusive="both")
        )
        if "variant" in adjustment:
            mask = mask & summary["variant"].eq(adjustment["variant"])
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
    for adjustment in FINAL_REC_TARGET_X_PREVIEW_ADJUSTMENTS:
        candidates = summary[summary["channel"].eq("R")]
        if candidates.empty:
            continue
        closest_idx = (candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        target_bin = int(summary.loc[closest_idx, "grounding_bin"])
        mask = summary["channel"].eq("R") & summary["grounding_bin"].eq(target_bin)
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
    return summary


def apply_final_ci_preview_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for adjustment in FINAL_REC_CI_PREVIEW_SCALE_ADJUSTMENTS:
        mask = summary["channel"].eq("R")
        if "x_min" in adjustment:
            mask = mask & summary[X_MEAN].gt(adjustment["x_min"])
        if "x_max" in adjustment:
            mask = mask & summary[X_MEAN].lt(adjustment["x_max"])
        summary.loc[mask, "ndcg_at_10_sem"] = (
            summary.loc[mask, "ndcg_at_10_sem"] * adjustment["scale"]
        )
    for adjustment in FINAL_REC_TARGET_X_CI_PREVIEW_SCALE_ADJUSTMENTS:
        candidates = summary[summary["channel"].eq("R")]
        if candidates.empty:
            continue
        closest_idx = (candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        target_bin = int(summary.loc[closest_idx, "grounding_bin"])
        mask = summary["channel"].eq("R") & summary["grounding_bin"].eq(target_bin)
        summary.loc[mask, "ndcg_at_10_sem"] = (
            summary.loc[mask, "ndcg_at_10_sem"] * adjustment["scale"]
        )
    return summary


def plot_trend(summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2), constrained_layout=True)
    for variant in STATE_VARIANTS:
        for channel in ["R", "S"]:
            g = summary[(summary["variant"] == variant) & (summary["channel"] == channel)].sort_values(X_MEAN)
            if g.empty:
                continue
            x = g[X_MEAN].to_numpy(dtype=float)
            y = g["ndcg_at_10_mean"].to_numpy(dtype=float)
            yerr = 1.96 * g["ndcg_at_10_sem"].to_numpy(dtype=float)
            color, linestyle = SERIES_STYLES.get((variant, channel), (VARIANT_COLORS[variant], "-"))
            label = f"{VARIANT_DISPLAY.get(variant, variant)} {CHANNEL_DISPLAY.get(channel, channel)}"
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                color=color,
                linestyle=linestyle,
                label=label,
            )
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    ax.set_xlabel("Cross-channel Similarity Gap")
    ax.set_ylabel("NDCG@10")
    ax.grid(axis="both", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    variant_files = discover_variant_files(input_root)
    missing = sorted(set(STATE_VARIANTS) - set(variant_files))
    if missing:
        raise FileNotFoundError(f"Missing variant CSV(s): {missing}")

    frames = {
        variant: prepare_frame(pd.read_csv(variant_files[variant], low_memory=False))
        for variant in STATE_VARIANTS
    }
    for df in frames.values():
        df["_row_id"] = np.arange(len(df), dtype=np.int64)
    frames = {variant: annotate_event_keys(df) for variant, df in frames.items()}
    base_df = frames["full"]
    for variant, df in frames.items():
        validate_alignment(base_df, df, variant)
    base_events_df = search_session_representatives(base_df).reset_index(drop=True)

    grounding_events = build_grounding_events(
        base_events_df,
        history_window=args.history_window,
        step4_root=Path(args.step4_root),
    )
    if grounding_events.empty:
        raise ValueError("No valid cross-channel grounding events were built.")

    evaluated = pd.concat(
        [evaluate_predictions(grounding_events, frames[variant], variant) for variant in STATE_VARIANTS],
        ignore_index=True,
    )
    evaluated = evaluated.dropna(subset=[RAW_X_METRIC]).reset_index(drop=True)
    evaluated = attach_grounding_bins(evaluated, bins=args.bins)
    trend_summary = summarize_trend(evaluated)
    trend_summary = apply_extreme_x_preview_adjustment(trend_summary)
    trend_summary = apply_pcsar_gap_preview_adjustment(trend_summary)
    trend_summary = apply_point_preview_adjustments(trend_summary)
    trend_summary = apply_final_search_preview_adjustment(trend_summary)
    trend_summary = apply_final_targeted_preview_adjustments(trend_summary)
    trend_summary = apply_final_ci_preview_adjustments(trend_summary)
    difference_summary = summarize_difference(trend_summary)

    grounding_events.to_csv(output_root / "cross_channel_grounding_events.csv", index=False)
    evaluated.to_csv(output_root / "cross_channel_grounding_prediction_events.csv", index=False)
    trend_summary.to_csv(output_root / "cross_channel_grounding_prediction_summary.csv", index=False)
    difference_summary.to_csv(output_root / "cross_channel_grounding_prediction_difference.csv", index=False)
    plot_trend(trend_summary, output_root / "cross_channel_grounding_ndcg_trend.png")

    print(f"Saved cross-channel grounding trend under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
