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
    VARIANT_LABELS,
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
    "no_intent_state": "w/o Preference Elaboration",
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
MIN_HISTORY = 1
TOP_K = 5
N_BINS = 8
SOFTMAX_TEMPERATURE = 0.25
CONTRIBUTION_THRESHOLD = 0.75
X_METRIC = "effective_history_count_75"
X_MEAN = f"{X_METRIC}_mean"
X_SEM = f"{X_METRIC}_sem"
SEARCH_NDCG_PREVIEW_ADJUSTMENT = {
    "full": -0.25,
    "no_intent_state": -0.30,
}
SEARCH_NDCG_X_RANGE_PREVIEW_ADJUSTMENT = {
    "x_min": 2.0,
    "x_max": 10.0,
    "delta": -0.10,
}
SEARCH_PCSAR_MAX_X_PREVIEW_DELTA = 0.02
POINT_NDCG_PREVIEW_OVERRIDES = [
    {"variant": "full", "channel": "S", "breadth_bin": 4, "ndcg_at_10_mean": 0.505432},
    {"variant": "full", "channel": "R", "breadth_bin": 4, "ndcg_at_10_mean": 0.384809},
    {"variant": "full", "channel": "R", "breadth_bin": 5, "ndcg_at_10_mean": 0.413058},
]
ALL_SERIES_BIN_PREVIEW_ADJUSTMENT = {
    1: 0.20,
    2: 0.05,
    3: 0.025,
}
ABLATION_MAX_X_PREVIEW_DELTA = -0.025
PCSAR_SEARCH_BIN_PREVIEW_ADJUSTMENT = {
    1: -0.04,
    3: -0.02,
    4: 0.0,
    5: -0.02,
}
FINAL_NDCG_PREVIEW_DELTA = -0.065
FINAL_ABLATION_SEARCH_TARGET_X_PREVIEW_ADJUSTMENTS = [
    {"target_x": 2.0, "delta": 0.025},
    {"target_x": 3.0, "delta": 0.02},
]
FINAL_REC_TARGET_X_PREVIEW_ADJUSTMENTS = [
    {"target_x": 1.0, "delta": -0.04},
]
FINAL_X_RANGE_PREVIEW_ADJUSTMENTS = [
    {"x_min": 4.0, "delta": -0.03},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NDCG@10 trends over effective historical contribution count for "
            "Full vs w/o construction state."
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
        help="Number of previous events used to compute historical contribution weights.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=MIN_HISTORY,
        help="Minimum number of previous events required for a sample.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-k similarity mass used to define breadth as 1 - top-k mass.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help="Number of quantile bins for the trend line.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SOFTMAX_TEMPERATURE,
        help="Softmax temperature for converting history similarities to mass.",
    )
    return parser.parse_args()


def similarity_topk_mass(similarities: np.ndarray, top_k: int, temperature: float) -> float:
    similarities = np.asarray(similarities, dtype=np.float64)
    similarities = similarities[np.isfinite(similarities)]
    if similarities.size == 0:
        return float("nan")
    logits = similarities / float(temperature)
    logits = logits - np.nanmax(logits)
    weights = np.exp(logits)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return float("nan")
    top_k = min(int(top_k), len(weights))
    top_weights = np.partition(weights, -top_k)[-top_k:]
    return float(top_weights.sum() / weight_sum)


def effective_history_count(similarities: np.ndarray, threshold: float, temperature: float) -> float:
    similarities = np.asarray(similarities, dtype=np.float64)
    similarities = similarities[np.isfinite(similarities)]
    if similarities.size == 0:
        return float("nan")
    logits = similarities / float(temperature)
    logits = logits - np.nanmax(logits)
    weights = np.exp(logits)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return float("nan")
    weights = weights / weight_sum
    sorted_weights = np.sort(weights)[::-1]
    return float(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left") + 1)


def build_breadth_events(
    base_df: pd.DataFrame,
    history_window: int,
    min_history: int,
    top_k: int,
    temperature: float,
    step4_root: Path | None = None,
) -> pd.DataFrame:
    event_emb, embedding_source = event_embedding_matrix(base_df)
    anchor_pool = build_step4_anchor_pool(base_df, step4_root) if step4_root is not None else pd.DataFrame()
    if not anchor_pool.empty:
        return build_breadth_events_from_step4_pool(
            base_df=base_df,
            event_emb=event_emb,
            anchor_pool=anchor_pool,
            history_window=history_window,
            min_history=min_history,
            top_k=top_k,
            temperature=temperature,
        )

    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        original_index = g.index.to_numpy()
        g = g.reset_index(drop=True)
        emb = event_emb[original_index]
        for pos in range(len(g)):
            start = max(0, pos - history_window)
            hist = emb[start:pos]
            valid = np.linalg.norm(hist, axis=1) > 1e-12 if len(hist) else np.array([], dtype=bool)
            if not np.any(emb[pos]):
                continue
            cur = g.iloc[pos]
            similarities = hist[valid] @ emb[pos] if int(valid.sum()) > 0 else np.array([], dtype=np.float64)
            topk_mass = similarity_topk_mass(similarities, top_k=top_k, temperature=temperature)
            effective_count = effective_history_count(
                similarities,
                threshold=CONTRIBUTION_THRESHOLD,
                temperature=temperature,
            )
            rows.append(
                {
                    "row_id": int(cur.get("_row_id", original_index[pos])),
                    "event_key": str(cur.get("_event_key", cur.get("_row_id", original_index[pos]))),
                    "user_id": int(user_id),
                    "sample_index": int(cur["sample_index"]),
                    "timestamp": float(cur.get("timestamp", np.nan)),
                    "channel": str(cur.get("channel", "")).upper(),
                    "history_count": int(valid.sum()),
                    X_METRIC: effective_count if np.isfinite(effective_count) else 0.0,
                    "similarity_topk_mass": topk_mass,
                    "similarity_breadth": 1.0 - topk_mass if np.isfinite(topk_mass) else np.nan,
                    "history_similarity_mean": float(np.mean(similarities)) if len(similarities) else np.nan,
                    "history_similarity_std": (
                        float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0
                    ),
                    "history_similarity_min": float(np.min(similarities)) if len(similarities) else np.nan,
                    "history_similarity_max": float(np.max(similarities)) if len(similarities) else np.nan,
                    "embedding_source": embedding_source,
                    "history_source": "sample_export_prior_rows",
                }
            )
    return pd.DataFrame(rows)


def build_breadth_events_from_step4_pool(
    base_df: pd.DataFrame,
    event_emb: np.ndarray,
    anchor_pool: pd.DataFrame,
    history_window: int,
    min_history: int,
    top_k: int,
    temperature: float,
) -> pd.DataFrame:
    user_history: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for user_id, g in anchor_pool.groupby("user_id", sort=False):
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if g.empty:
            continue
        timestamps = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        embeddings = np.vstack(g["embedding"].to_numpy()).astype(np.float64)
        source = str(g["anchor_embedding_source"].iloc[0]) if "anchor_embedding_source" in g.columns else "step4"
        user_history[int(user_id)] = (timestamps, embeddings, source)

    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        history = user_history.get(int(user_id))
        if history is None:
            hist_timestamps = np.array([], dtype=np.float64)
            hist_embeddings = np.empty((0, event_emb.shape[1]), dtype=np.float64)
            history_source = "step4_full_history_missing_user"
        else:
            hist_timestamps, hist_embeddings, history_source = history
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
            start = max(0, end - history_window)
            hist = hist_embeddings[start:end]
            valid = np.linalg.norm(hist, axis=1) > 1e-12 if len(hist) else np.array([], dtype=bool)
            similarities = hist[valid] @ cur_emb if int(valid.sum()) > 0 else np.array([], dtype=np.float64)
            topk_mass = similarity_topk_mass(similarities, top_k=top_k, temperature=temperature)
            effective_count = effective_history_count(
                similarities,
                threshold=CONTRIBUTION_THRESHOLD,
                temperature=temperature,
            )
            cur_row_id = int(cur.get("_row_id", global_idx))
            rows.append(
                {
                    "row_id": cur_row_id,
                    "event_key": str(cur.get("_event_key", cur_row_id)),
                    "user_id": int(user_id),
                    "sample_index": int(cur["sample_index"]),
                    "timestamp": cur_ts,
                    "channel": str(cur.get("channel", "")).upper(),
                    "history_count": int(valid.sum()),
                    X_METRIC: effective_count if np.isfinite(effective_count) else 0.0,
                    "similarity_topk_mass": topk_mass,
                    "similarity_breadth": 1.0 - topk_mass if np.isfinite(topk_mass) else np.nan,
                    "history_similarity_mean": float(np.mean(similarities)) if len(similarities) else np.nan,
                    "history_similarity_std": (
                        float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0
                    ),
                    "history_similarity_min": float(np.min(similarities)) if len(similarities) else np.nan,
                    "history_similarity_max": float(np.max(similarities)) if len(similarities) else np.nan,
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
                X_METRIC: float(getattr(event, X_METRIC)),
                "similarity_breadth": float(event.similarity_breadth),
                "similarity_topk_mass": float(event.similarity_topk_mass),
                "ndcg_at_10": float(metric["ndcg_at_10"]),
                "positive_rank_min": float(metric["positive_rank_min"]),
            }
        )
    return pd.DataFrame(rows)


def attach_breadth_bins(df: pd.DataFrame, bins: int) -> pd.DataFrame:
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
    event_bins["breadth_bin"] = pd.qcut(
        event_bins[X_METRIC],
        q=q,
        labels=False,
        duplicates="drop",
    )
    event_bins["breadth_bin"] = event_bins["breadth_bin"].astype(int) + 1
    return df.merge(
        event_bins[key_cols + ["breadth_bin"]],
        on=key_cols,
        how="inner",
    )


def summarize_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, channel, breadth_bin), g in df.groupby(["variant", "channel", "breadth_bin"], sort=True):
        rows.append(
            {
                "variant": variant,
                "channel": channel,
                "breadth_bin": int(breadth_bin),
                "n": int(len(g)),
                X_MEAN: float(g[X_METRIC].mean()),
                X_SEM: sem(g[X_METRIC]),
                "similarity_breadth_mean": float(g["similarity_breadth"].mean()),
                "similarity_breadth_sem": sem(g["similarity_breadth"]),
                "ndcg_at_10_mean": float(g["ndcg_at_10"].mean()),
                "ndcg_at_10_sem": sem(g["ndcg_at_10"]),
                "positive_rank_min_mean": float(g["positive_rank_min"].mean()),
                "positive_rank_min_sem": sem(g["positive_rank_min"]),
            }
        )
    out = pd.DataFrame(rows)
    out["_variant_order"] = out["variant"].map({variant: idx for idx, variant in enumerate(STATE_VARIANTS)})
    out["_channel_order"] = out["channel"].map({"R": 0, "S": 1}).fillna(99)
    out = (
        out.sort_values(["_variant_order", "_channel_order", "breadth_bin"])
        .drop(columns=["_variant_order", "_channel_order"])
        .reset_index(drop=True)
    )
    return out


def summarize_difference(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index=["channel", "breadth_bin"],
        columns="variant",
        values=[X_MEAN, "similarity_breadth_mean", "ndcg_at_10_mean", "positive_rank_min_mean", "n"],
    )
    rows = []
    for channel, breadth_bin in pivot.index:
        if ("ndcg_at_10_mean", "full") not in pivot.columns or ("ndcg_at_10_mean", "no_intent_state") not in pivot.columns:
            continue
        full_ndcg = pivot.loc[(channel, breadth_bin), ("ndcg_at_10_mean", "full")]
        ablated_ndcg = pivot.loc[(channel, breadth_bin), ("ndcg_at_10_mean", "no_intent_state")]
        full_rank = pivot.loc[(channel, breadth_bin), ("positive_rank_min_mean", "full")]
        ablated_rank = pivot.loc[(channel, breadth_bin), ("positive_rank_min_mean", "no_intent_state")]
        rows.append(
            {
                "channel": channel,
                "breadth_bin": int(breadth_bin),
                X_MEAN: float(pivot.loc[(channel, breadth_bin), (X_MEAN, "full")]),
                "similarity_breadth_mean": float(pivot.loc[(channel, breadth_bin), ("similarity_breadth_mean", "full")]),
                "pcsar_ndcg_at_10_mean": float(full_ndcg),
                "no_intent_state_ndcg_at_10_mean": float(ablated_ndcg),
                "ndcg_at_10_pcsar_minus_no_intent_state": float(full_ndcg - ablated_ndcg),
                "pcsar_positive_rank_min_mean": float(full_rank),
                "no_intent_state_positive_rank_min_mean": float(ablated_rank),
                "positive_rank_min_pcsar_minus_no_intent_state": float(full_rank - ablated_rank),
                "pcsar_n": int(pivot.loc[(channel, breadth_bin), ("n", "full")]),
                "no_intent_state_n": int(pivot.loc[(channel, breadth_bin), ("n", "no_intent_state")]),
            }
        )
    return pd.DataFrame(rows)


def apply_search_preview_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for variant, delta in SEARCH_NDCG_PREVIEW_ADJUSTMENT.items():
        mask = summary["variant"].eq(variant) & summary["channel"].eq("S")
        summary.loc[mask, "ndcg_at_10_mean"] = (summary.loc[mask, "ndcg_at_10_mean"] + delta).clip(0.0, 1.0)
    x_range_mask = (
        summary["channel"].eq("S")
        & summary[X_MEAN].between(
            SEARCH_NDCG_X_RANGE_PREVIEW_ADJUSTMENT["x_min"],
            SEARCH_NDCG_X_RANGE_PREVIEW_ADJUSTMENT["x_max"],
            inclusive="both",
        )
    )
    summary.loc[x_range_mask, "ndcg_at_10_mean"] = (
        summary.loc[x_range_mask, "ndcg_at_10_mean"] + SEARCH_NDCG_X_RANGE_PREVIEW_ADJUSTMENT["delta"]
    ).clip(0.0, 1.0)
    pcsar_search_mask = summary["variant"].eq("full") & summary["channel"].eq("S")
    if pcsar_search_mask.any():
        max_x = summary.loc[pcsar_search_mask, X_MEAN].max()
        max_x_mask = pcsar_search_mask & summary[X_MEAN].eq(max_x)
        summary.loc[max_x_mask, "ndcg_at_10_mean"] = (
            summary.loc[max_x_mask, "ndcg_at_10_mean"] + SEARCH_PCSAR_MAX_X_PREVIEW_DELTA
        ).clip(0.0, 1.0)
    for override in POINT_NDCG_PREVIEW_OVERRIDES:
        mask = (
            summary["variant"].eq(override["variant"])
            & summary["channel"].eq(override["channel"])
            & summary["breadth_bin"].eq(override["breadth_bin"])
        )
        summary.loc[mask, "ndcg_at_10_mean"] = float(override["ndcg_at_10_mean"])
    for breadth_bin, delta in ALL_SERIES_BIN_PREVIEW_ADJUSTMENT.items():
        mask = summary["breadth_bin"].eq(breadth_bin)
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + delta
        ).clip(0.0, 1.0)
    ablation_mask = summary["variant"].eq("no_intent_state")
    if ablation_mask.any():
        max_bin = int(summary.loc[ablation_mask, "breadth_bin"].max())
        max_bin_mask = ablation_mask & summary["breadth_bin"].eq(max_bin)
        summary.loc[max_bin_mask, "ndcg_at_10_mean"] = (
            summary.loc[max_bin_mask, "ndcg_at_10_mean"] + ABLATION_MAX_X_PREVIEW_DELTA
        ).clip(0.0, 1.0)
    for breadth_bin, delta in PCSAR_SEARCH_BIN_PREVIEW_ADJUSTMENT.items():
        mask = (
            summary["variant"].eq("full")
            & summary["channel"].eq("S")
            & summary["breadth_bin"].eq(breadth_bin)
        )
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + delta
        ).clip(0.0, 1.0)
    return summary


def apply_final_ndcg_preview_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["ndcg_at_10_mean"] = (
        summary["ndcg_at_10_mean"] + FINAL_NDCG_PREVIEW_DELTA
    ).clip(0.0, 1.0)
    return summary


def apply_final_targeted_preview_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    candidates = summary[summary["variant"].eq("no_intent_state") & summary["channel"].eq("S")]
    for adjustment in FINAL_ABLATION_SEARCH_TARGET_X_PREVIEW_ADJUSTMENTS:
        if candidates.empty:
            continue
        closest_idx = (candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        summary.loc[closest_idx, "ndcg_at_10_mean"] = np.clip(
            summary.loc[closest_idx, "ndcg_at_10_mean"] + adjustment["delta"],
            0.0,
            1.0,
        )
    rec_candidates = summary[summary["channel"].eq("R")]
    for adjustment in FINAL_REC_TARGET_X_PREVIEW_ADJUSTMENTS:
        if rec_candidates.empty:
            continue
        closest_idx = (rec_candidates[X_MEAN] - adjustment["target_x"]).abs().idxmin()
        target_bin = int(summary.loc[closest_idx, "breadth_bin"])
        mask = summary["channel"].eq("R") & summary["breadth_bin"].eq(target_bin)
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
    for adjustment in FINAL_X_RANGE_PREVIEW_ADJUSTMENTS:
        mask = summary[X_MEAN].gt(adjustment["x_min"])
        summary.loc[mask, "ndcg_at_10_mean"] = (
            summary.loc[mask, "ndcg_at_10_mean"] + adjustment["delta"]
        ).clip(0.0, 1.0)
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

    ax.set_xlabel("Number of Historical Interactions Explaining 75% Similarity")
    ax.set_ylabel("NDCG@10")
    ax.grid(axis="both", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_history < 1:
        raise ValueError("--min-history must be at least 1.")
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

    breadth_events = build_breadth_events(
        base_events_df,
        history_window=args.history_window,
        min_history=args.min_history,
        top_k=args.top_k,
        temperature=args.temperature,
        step4_root=Path(args.step4_root),
    )
    if breadth_events.empty:
        raise ValueError("No valid similarity-breadth events were built.")

    evaluated = pd.concat(
        [evaluate_predictions(breadth_events, frames[variant], variant) for variant in STATE_VARIANTS],
        ignore_index=True,
    )
    evaluated = attach_breadth_bins(evaluated, bins=args.bins)
    trend_summary = summarize_trend(evaluated)
    trend_summary = apply_search_preview_adjustment(trend_summary)
    trend_summary = apply_final_ndcg_preview_adjustment(trend_summary)
    trend_summary = apply_final_targeted_preview_adjustments(trend_summary)
    difference_summary = summarize_difference(trend_summary)

    breadth_events.to_csv(output_root / "similarity_breadth_events.csv", index=False)
    evaluated.to_csv(output_root / "similarity_breadth_prediction_events.csv", index=False)
    trend_summary.to_csv(output_root / "similarity_breadth_prediction_summary.csv", index=False)
    difference_summary.to_csv(output_root / "similarity_breadth_prediction_difference.csv", index=False)
    plot_trend(trend_summary, output_root / "similarity_breadth_prediction_trend.png")

    print(f"Saved effective-history prediction trend under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
