from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "Features" / "intermediate" / "pcsar_intent_features_all_full_mechanism.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output_mechanism"
STATE_LABELS = ["Low", "Medium", "High"]
STATE_COLORS = {
    "Low": "#6baed6",
    "Medium": "#fdae6b",
    "High": "#f16913",
}
TRANSITION_ORDER = ["R->R", "R->S", "S->R", "S->S"]
TRANSITION_COLORS = {
    "R->R": "#2f6f4e",
    "R->S": "#c7862f",
    "S->R": "#3b6ea8",
    "S->S": "#a33a2f",
}
VALIDATION_METRICS = {
    "uncertainty": "global_posterior_uncertainty",
    "entropy": "global_intent_entropy",
    "confidence": "global_dominant_intent_prob",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether inferred PC-SAR states match future preference dynamics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to the full PC-SAR intent feature export.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for mechanism-validation outputs.",
    )
    parser.add_argument(
        "--future-window",
        type=int,
        default=10,
        help="Number of future events used to compute future intent consistency and dispersion.",
    )
    parser.add_argument(
        "--state-bins",
        type=int,
        default=3,
        help="Number of state bins. The paper-facing default is 3: Low, Medium, High.",
    )
    parser.add_argument(
        "--bin-scope",
        choices=["global", "user"],
        default="global",
        help="Whether state bins are assigned globally or within each user.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sem(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values).dropna()
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def suffix_value(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 10**9


def pi_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("global_pi_")]
    return sorted(cols, key=suffix_value)


def prefix_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if col.startswith(prefix)]
    return sorted(cols, key=suffix_value)


def js_distance(pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    a = np.asarray(pi_a, dtype=np.float64)
    b = np.asarray(pi_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.clip(a, 1e-12, 1.0)
    b = np.clip(b, 1e-12, 1.0)
    a = a / np.clip(a.sum(), 1e-12, None)
    b = b / np.clip(b.sum(), 1e-12, None)
    m = 0.5 * (a + b)
    kl_am = np.sum(a * (np.log(a) - np.log(m)))
    kl_bm = np.sum(b * (np.log(b) - np.log(m)))
    return float(np.sqrt(max(0.5 * (kl_am + kl_bm), 0.0)))


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "sample_index",
        "user_id",
        "timestamp",
        "history_length",
        "global_history_length",
        "rec_history_length",
        "src_history_length",
        "search_session_id",
        "item_id",
        "history_rec_share",
        "history_src_share",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
        "rec_src_intent_shift_js",
        "attribution_confidence_gap",
        "attribution_entropy_gap",
        "rec_cross_gate_pos",
        "src_cross_gate_pos",
        "rec_same_delta_pos",
        "rec_cross_delta_pos",
        "src_same_delta_pos",
        "src_cross_delta_pos",
        "rec_pred_pos_score",
        "rec_pred_pos_rank",
        "rec_pred_pos_margin",
        "rec_pred_top1_is_pos",
        "src_pred_pos_score",
        "src_pred_pos_rank",
        "src_pred_pos_margin",
        "src_pred_top1_is_pos",
    ]
    numeric_cols.extend(pi_columns(df))
    numeric_cols.extend(prefix_columns(df, "rec_pi_"))
    numeric_cols.extend(prefix_columns(df, "src_pi_"))
    df = coerce_numeric(df, numeric_cols)
    if "channel" not in df.columns:
        df["channel"] = ""
    df["channel"] = df["channel"].astype(str).str.upper()
    df["_channel_order"] = df["channel"].map({"R": 0, "S": 1}).fillna(99).astype(int)
    sort_cols = ["user_id"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    sort_cols.append("_channel_order")
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")
    return df.sort_values(sort_cols, kind="mergesort").drop(columns="_channel_order").reset_index(drop=True)


def mean_existing(g: pd.DataFrame, col: str) -> float:
    if col not in g.columns:
        return float("nan")
    values = pd.to_numeric(g[col], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def first_existing(g: pd.DataFrame, col: str):
    if col not in g.columns:
        return np.nan
    valid = g[col].dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def add_mean_columns(row: dict, g: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        row[col] = mean_existing(g, col)


def build_event_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse item-level search samples to session-level events for mechanism tests."""
    pi_cols = pi_columns(df)
    rec_pi_cols = prefix_columns(df, "rec_pi_")
    src_pi_cols = prefix_columns(df, "src_pi_")
    vector_cols = pi_cols + rec_pi_cols + src_pi_cols
    mean_cols = [
        "history_length",
        "global_history_length",
        "rec_history_length",
        "src_history_length",
        "history_rec_share",
        "history_src_share",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
        "rec_dominant_intent_prob",
        "rec_intent_entropy",
        "rec_posterior_uncertainty",
        "src_dominant_intent_prob",
        "src_intent_entropy",
        "src_posterior_uncertainty",
        "rec_src_intent_shift_js",
        "attribution_confidence_gap",
        "attribution_entropy_gap",
        "rec_cross_gate_pos",
        "src_cross_gate_pos",
        "rec_same_delta_pos",
        "rec_cross_delta_pos",
        "src_same_delta_pos",
        "src_cross_delta_pos",
        "rec_pred_pos_score",
        "rec_pred_pos_rank",
        "rec_pred_pos_margin",
        "rec_pred_top1_is_pos",
        "src_pred_pos_score",
        "src_pred_pos_rank",
        "src_pred_pos_margin",
        "src_pred_top1_is_pos",
    ]
    mean_cols = list(dict.fromkeys([col for col in mean_cols + vector_cols if col in df.columns]))
    agg_map = {col: "mean" for col in mean_cols}

    r_df = df[df["channel"] == "R"].copy()
    event_frames: list[pd.DataFrame] = []
    if not r_df.empty:
        r_df["_r_item_key"] = pd.to_numeric(r_df.get("item_id", np.nan), errors="coerce").fillna(-1).astype("int64")
        r_df["_r_timestamp_key"] = pd.to_numeric(r_df.get("timestamp", np.nan), errors="coerce")
        r_group_cols = ["user_id", "_r_timestamp_key", "_r_item_key"]
        r_mean = r_df.groupby(r_group_cols, as_index=False, sort=False, dropna=False).agg(agg_map)
        r_meta = (
            r_df.groupby(r_group_cols, as_index=False, sort=False, dropna=False)
            .agg(
                timestamp=("timestamp", "min"),
                split=("split", "first"),
                sample_count=("channel", "size"),
                item_count=("item_id", "nunique"),
            )
        )
        r_events = r_meta.merge(r_mean, on=r_group_cols, how="left")
        r_events["event_type"] = "R_item"
        r_events["channel"] = "R"
        r_events["item_id"] = r_events["_r_item_key"].where(r_events["_r_item_key"] >= 0, np.nan)
        r_events["search_session_id"] = np.nan
        r_events["event_uid"] = [
            f"R:{int(user_id)}:{float(timestamp)}:{int(item_id)}:{idx}"
            for idx, (user_id, timestamp, item_id) in enumerate(
                zip(r_events["user_id"], r_events["timestamp"], r_events["_r_item_key"])
            )
        ]
        r_events = r_events.drop(columns=["_r_timestamp_key", "_r_item_key"])
        event_frames.append(r_events)

    s_df = df[df["channel"] == "S"].copy()
    if not s_df.empty:
        s_df["_s_session_key"] = pd.to_numeric(s_df.get("search_session_id", np.nan), errors="coerce").fillna(-1).astype("int64")
        s_group_cols = ["user_id", "_s_session_key"]
        s_mean = s_df.groupby(s_group_cols, as_index=False, sort=False, dropna=False).agg(agg_map)
        s_meta = (
            s_df.groupby(s_group_cols, as_index=False, sort=False, dropna=False)
            .agg(
                timestamp=("timestamp", "min"),
                split=("split", "first"),
                sample_count=("channel", "size"),
                item_count=("item_id", "nunique"),
            )
        )
        s_events = s_meta.merge(s_mean, on=s_group_cols, how="left")
        s_events["event_type"] = "S_session"
        s_events["channel"] = "S"
        s_events["item_id"] = np.nan
        s_events["search_session_id"] = s_events["_s_session_key"].where(s_events["_s_session_key"] >= 0, np.nan)
        s_events["event_uid"] = [
            f"S:{int(user_id)}:{int(session_id)}:{idx}"
            for idx, (user_id, session_id) in enumerate(
                zip(s_events["user_id"], s_events["_s_session_key"])
            )
        ]
        s_events = s_events.drop(columns=["_s_session_key"])
        event_frames.append(s_events)

    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    if events.empty:
        return events
    events["_channel_order"] = events["channel"].map({"R": 0, "S": 1}).fillna(99).astype(int)
    events = events.sort_values(
        ["user_id", "timestamp", "_channel_order", "event_uid"],
        kind="mergesort",
    ).drop(columns="_channel_order").reset_index(drop=True)
    events.insert(0, "event_index", np.arange(len(events), dtype=int))
    return events


def build_state_events(df: pd.DataFrame, future_window: int) -> pd.DataFrame:
    pi_cols = pi_columns(df)
    if not pi_cols:
        raise KeyError("Missing global_pi_* columns in the input CSV.")

    rows: list[dict] = []
    for user_id, g in df.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        pi = g[pi_cols].to_numpy(dtype=np.float64)
        for pos, row in g.iterrows():
            future = pi[pos + 1 : pos + 1 + future_window]
            if len(future) == 0:
                continue
            future_center = future.mean(axis=0)
            current_future_distances = [js_distance(pi[pos], cur) for cur in future]
            current_to_future = js_distance(pi[pos], future_center)
            rows.append(
                {
                    "user_id": int(user_id),
                    "sample_index": int(row["sample_index"]) if pd.notna(row.get("sample_index")) else np.nan,
                    "timestamp": float(row["timestamp"]) if pd.notna(row.get("timestamp")) else np.nan,
                    "channel": row.get("channel", ""),
                    "future_window_observed": int(len(future)),
                    "future_consistency": float(1.0 - current_to_future),
                    "future_intent_dispersion": float(np.nanmean(current_future_distances)),
                    "current_to_future_js": float(current_to_future),
                    "global_posterior_uncertainty": float(row["global_posterior_uncertainty"]),
                    "global_intent_entropy": float(row["global_intent_entropy"]),
                    "global_dominant_intent_prob": float(row["global_dominant_intent_prob"]),
                    "history_src_share": float(row.get("history_src_share", np.nan)),
                    "history_rec_share": float(row.get("history_rec_share", np.nan)),
                    "global_history_length": float(row.get("global_history_length", np.nan)),
                    "rec_history_length": float(row.get("rec_history_length", np.nan)),
                    "src_history_length": float(row.get("src_history_length", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def assign_state_bins(events: pd.DataFrame, metric_name: str, metric_col: str, bins: int, scope: str) -> pd.DataFrame:
    if bins != 3:
        labels = [f"Q{i + 1}" for i in range(bins)]
    else:
        labels = STATE_LABELS

    events = events.copy()
    state_col = f"{metric_name}_state"
    events[state_col] = pd.NA
    valid = events[np.isfinite(pd.to_numeric(events[metric_col], errors="coerce"))].copy()
    if valid.empty:
        return events

    def qcut_labels(values: pd.Series) -> pd.Series:
        if values.nunique(dropna=True) < 2:
            return pd.Series(pd.NA, index=values.index, dtype="object")
        local_bins = min(bins, values.nunique(dropna=True))
        local_labels = labels if local_bins == bins else labels[:local_bins]
        return pd.qcut(
            values.rank(method="first"),
            q=local_bins,
            labels=local_labels,
            duplicates="drop",
        ).astype("object")

    if scope == "user":
        assigned = valid.groupby("user_id", group_keys=False)[metric_col].apply(qcut_labels)
    else:
        assigned = qcut_labels(valid[metric_col])

    events.loc[assigned.index, state_col] = assigned
    return events


def summarize_state(events: pd.DataFrame, metric_name: str, metric_col: str) -> pd.DataFrame:
    state_col = f"{metric_name}_state"
    rows = []
    for state, g in events.dropna(subset=[state_col]).groupby(state_col, sort=False):
        rows.append(
            {
                "state_metric": metric_name,
                "state": str(state),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "state_value_mean": float(np.nanmean(g[metric_col])),
                "state_value_sem": sem(g[metric_col]),
                "future_consistency_mean": float(np.nanmean(g["future_consistency"])),
                "future_consistency_sem": sem(g["future_consistency"]),
                "future_intent_dispersion_mean": float(np.nanmean(g["future_intent_dispersion"])),
                "future_intent_dispersion_sem": sem(g["future_intent_dispersion"]),
                "history_src_share_mean": float(np.nanmean(g["history_src_share"])),
                "history_src_share_sem": sem(g["history_src_share"]),
                "global_history_length_mean": float(np.nanmean(g["global_history_length"])),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {label: idx for idx, label in enumerate(STATE_LABELS)}
        out["_order"] = out["state"].map(order).fillna(99)
        out = out.sort_values(["state_metric", "_order", "state"]).drop(columns="_order").reset_index(drop=True)
    return out


def summarize_high_low(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, g in summary.groupby("state_metric", sort=False):
        idx = g.set_index("state")
        if "Low" not in idx.index or "High" not in idx.index:
            continue
        low = idx.loc["Low"]
        high = idx.loc["High"]
        rows.append(
            {
                "state_metric": metric,
                "high_minus_low_future_consistency": float(
                    high["future_consistency_mean"] - low["future_consistency_mean"]
                ),
                "high_minus_low_future_intent_dispersion": float(
                    high["future_intent_dispersion_mean"] - low["future_intent_dispersion_mean"]
                ),
                "high_minus_low_history_src_share": float(
                    high["history_src_share_mean"] - low["history_src_share_mean"]
                ),
                "low_n": int(low["n"]),
                "high_n": int(high["n"]),
            }
        )
    return pd.DataFrame(rows)


def target_value(row: pd.Series, rec_col: str, src_col: str) -> float:
    if str(row.get("channel", "")).upper() == "R":
        return float(row.get(rec_col, np.nan))
    if str(row.get("channel", "")).upper() == "S":
        return float(row.get(src_col, np.nan))
    return float("nan")


def row_vector(row: pd.Series, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.array([], dtype=np.float64)
    return row[cols].to_numpy(dtype=np.float64)


def build_transition_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if events.empty:
        return pd.DataFrame()
    global_pi_cols = pi_columns(events)
    rec_pi_cols = prefix_columns(events, "rec_pi_")
    src_pi_cols = prefix_columns(events, "src_pi_")

    for user_id, g in events.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for pos in range(1, len(g)):
            prev = g.iloc[pos - 1]
            cur = g.iloc[pos]
            prev_channel = str(prev.get("channel", "")).upper()
            cur_channel = str(cur.get("channel", "")).upper()
            if prev_channel not in {"R", "S"} or cur_channel not in {"R", "S"}:
                continue

            transition = f"{prev_channel}->{cur_channel}"
            cross_gate = target_value(cur, "rec_cross_gate_pos", "src_cross_gate_pos")
            same_delta = target_value(cur, "rec_same_delta_pos", "src_same_delta_pos")
            cross_delta = target_value(cur, "rec_cross_delta_pos", "src_cross_delta_pos")
            pred_pos_score = target_value(cur, "rec_pred_pos_score", "src_pred_pos_score")
            pred_pos_rank = target_value(cur, "rec_pred_pos_rank", "src_pred_pos_rank")
            pred_pos_margin = target_value(cur, "rec_pred_pos_margin", "src_pred_pos_margin")
            pred_top1_is_pos = target_value(cur, "rec_pred_top1_is_pos", "src_pred_top1_is_pos")
            global_pi = row_vector(cur, global_pi_cols)
            rec_pi = row_vector(cur, rec_pi_cols)
            src_pi = row_vector(cur, src_pi_cols)
            global_rec_alignment = 1.0 - js_distance(global_pi, rec_pi)
            global_src_alignment = 1.0 - js_distance(global_pi, src_pi)
            if cur_channel == "R":
                same_alignment = global_rec_alignment
                cross_alignment = global_src_alignment
                cross_intent_dominance = -float(cur.get("attribution_confidence_gap", np.nan))
            else:
                same_alignment = global_src_alignment
                cross_alignment = global_rec_alignment
                cross_intent_dominance = float(cur.get("attribution_confidence_gap", np.nan))
            cross_alignment_advantage = cross_alignment - same_alignment
            cross_intent_match = (
                cross_alignment_advantage * cross_intent_dominance
                if np.isfinite(cross_alignment_advantage) and np.isfinite(cross_intent_dominance)
                else np.nan
            )

            rows.append(
                {
                    "user_id": int(user_id),
                    "prev_event_index": int(prev["event_index"]),
                    "event_index": int(cur["event_index"]),
                    "timestamp": float(cur.get("timestamp", np.nan)),
                    "prev_channel": prev_channel,
                    "channel": cur_channel,
                    "transition_type": transition,
                    "is_cross_transition": int(prev_channel != cur_channel),
                    "event_type": cur.get("event_type", ""),
                    "sample_count": int(cur.get("sample_count", 1)),
                    "item_count": int(cur.get("item_count", 1)),
                    "cross_gate": cross_gate,
                    "same_delta": same_delta,
                    "cross_delta": cross_delta,
                    "cross_minus_same_delta": cross_delta - same_delta
                    if np.isfinite(cross_delta) and np.isfinite(same_delta)
                    else np.nan,
                    "same_alignment": same_alignment,
                    "cross_alignment": cross_alignment,
                    "cross_alignment_advantage": cross_alignment_advantage,
                    "global_rec_alignment": global_rec_alignment,
                    "global_src_alignment": global_src_alignment,
                    "cross_intent_dominance": cross_intent_dominance,
                    "cross_intent_match": cross_intent_match,
                    "cross_intent_match_advantage": cross_intent_match,
                    "rec_src_intent_shift_js": float(cur.get("rec_src_intent_shift_js", np.nan)),
                    "abs_attribution_confidence_gap": abs(float(cur.get("attribution_confidence_gap", np.nan))),
                    "attribution_confidence_gap": float(cur.get("attribution_confidence_gap", np.nan)),
                    "global_posterior_uncertainty": float(cur.get("global_posterior_uncertainty", np.nan)),
                    "global_intent_entropy": float(cur.get("global_intent_entropy", np.nan)),
                    "history_src_share": float(cur.get("history_src_share", np.nan)),
                    "pred_pos_score": pred_pos_score,
                    "pred_pos_rank": pred_pos_rank,
                    "pred_pos_margin": pred_pos_margin,
                    "pred_top1_is_pos": pred_top1_is_pos,
                }
            )
    return pd.DataFrame(rows)


def summarize_alignment(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_alignment_advantage", "cross_gate"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    local_bins = min(bins, valid["cross_alignment_advantage"].nunique())
    local_labels = labels if local_bins == bins else labels[:local_bins]
    valid["alignment_bin"] = pd.qcut(
        valid["cross_alignment_advantage"].rank(method="first"),
        q=local_bins,
        labels=local_labels,
        duplicates="drop",
    )
    rows = []
    for label, g in valid.groupby("alignment_bin", observed=True, sort=False):
        rows.append(
            {
                "alignment_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_alignment_advantage_mean": float(np.nanmean(g["cross_alignment_advantage"])),
                "cross_alignment_advantage_sem": sem(g["cross_alignment_advantage"]),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
                "cross_alignment_mean": float(np.nanmean(g["cross_alignment"])),
                "same_alignment_mean": float(np.nanmean(g["same_alignment"])),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {label: idx for idx, label in enumerate(labels)}
        out["_order"] = out["alignment_bin"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def summarize_cross_intent_match(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_intent_match_advantage", "cross_gate"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    local_bins = min(bins, valid["cross_intent_match_advantage"].nunique())
    local_labels = labels if local_bins == bins else labels[:local_bins]
    valid["intent_match_bin"] = pd.qcut(
        valid["cross_intent_match_advantage"].rank(method="first"),
        q=local_bins,
        labels=local_labels,
        duplicates="drop",
    )
    rows = []
    for label, g in valid.groupby("intent_match_bin", observed=True, sort=False):
        rows.append(
            {
                "intent_match_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_intent_match_advantage_mean": float(np.nanmean(g["cross_intent_match_advantage"])),
                "cross_intent_match_advantage_sem": sem(g["cross_intent_match_advantage"]),
                "cross_intent_match_mean": float(np.nanmean(g["cross_intent_match"])),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {label: idx for idx, label in enumerate(labels)}
        out["_order"] = out["intent_match_bin"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def summarize_cross_intent_dominance(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_intent_dominance", "cross_gate"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    local_bins = min(bins, valid["cross_intent_dominance"].nunique())
    local_labels = labels if local_bins == bins else labels[:local_bins]
    valid["intent_dominance_bin"] = pd.qcut(
        valid["cross_intent_dominance"].rank(method="first"),
        q=local_bins,
        labels=local_labels,
        duplicates="drop",
    )
    rows = []
    for label, g in valid.groupby("intent_dominance_bin", observed=True, sort=False):
        rows.append(
            {
                "intent_dominance_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_intent_dominance_mean": float(np.nanmean(g["cross_intent_dominance"])),
                "cross_intent_dominance_sem": sem(g["cross_intent_dominance"]),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {label: idx for idx, label in enumerate(labels)}
        out["_order"] = out["intent_dominance_bin"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def summarize_cross_intent_dominance_by_target(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_intent_dominance", "cross_gate", "channel"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    frames = []
    for channel, g_channel in valid.groupby("channel", sort=False):
        local_bins = min(bins, g_channel["cross_intent_dominance"].nunique())
        if local_bins < 2:
            continue
        local_labels = labels if local_bins == bins else labels[:local_bins]
        g_channel = g_channel.copy()
        g_channel["intent_dominance_bin"] = pd.qcut(
            g_channel["cross_intent_dominance"].rank(method="first"),
            q=local_bins,
            labels=local_labels,
            duplicates="drop",
        )
        frames.append(g_channel)
    if not frames:
        return pd.DataFrame()
    valid = pd.concat(frames, ignore_index=True)
    rows = []
    for (channel, label), g in valid.groupby(["channel", "intent_dominance_bin"], observed=True, sort=False):
        rows.append(
            {
                "channel": str(channel),
                "intent_dominance_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_intent_dominance_mean": float(np.nanmean(g["cross_intent_dominance"])),
                "cross_intent_dominance_sem": sem(g["cross_intent_dominance"]),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        channel_order = {"R": 0, "S": 1}
        bin_order = {label: idx for idx, label in enumerate(labels)}
        out["_channel_order"] = out["channel"].map(channel_order).fillna(99)
        out["_bin_order"] = out["intent_dominance_bin"].map(bin_order).fillna(99)
        out = out.sort_values(["_channel_order", "_bin_order"]).drop(columns=["_channel_order", "_bin_order"]).reset_index(drop=True)
    return out


def summarize_dominance_by_gate_bin(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_gate", "cross_intent_dominance", "channel"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    frames = []
    for channel, g_channel in valid.groupby("channel", sort=False):
        local_bins = min(bins, g_channel["cross_gate"].nunique())
        if local_bins < 2:
            continue
        local_labels = labels if local_bins == bins else labels[:local_bins]
        g_channel = g_channel.copy()
        g_channel["gate_bin"] = pd.qcut(
            g_channel["cross_gate"].rank(method="first"),
            q=local_bins,
            labels=local_labels,
            duplicates="drop",
        )
        frames.append(g_channel)
    if not frames:
        return pd.DataFrame()
    valid = pd.concat(frames, ignore_index=True)
    rows = []
    for (channel, label), g in valid.groupby(["channel", "gate_bin"], observed=True, sort=False):
        rows.append(
            {
                "channel": str(channel),
                "gate_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
                "cross_intent_dominance_mean": float(np.nanmean(g["cross_intent_dominance"])),
                "cross_intent_dominance_sem": sem(g["cross_intent_dominance"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        channel_order = {"R": 0, "S": 1}
        bin_order = {label: idx for idx, label in enumerate(labels)}
        out["_channel_order"] = out["channel"].map(channel_order).fillna(99)
        out["_bin_order"] = out["gate_bin"].map(bin_order).fillna(99)
        out = out.sort_values(["_channel_order", "_bin_order"]).drop(columns=["_channel_order", "_bin_order"]).reset_index(drop=True)
    return out


def summarize_cross_intent_match_by_target(transition_events: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    valid = transition_events.dropna(subset=["cross_intent_match_advantage", "cross_gate", "channel"]).copy()
    if valid.empty:
        return pd.DataFrame()
    labels = ["Low", "Medium", "High"] if bins == 3 else [f"Q{i + 1}" for i in range(bins)]
    frames = []
    for channel, g_channel in valid.groupby("channel", sort=False):
        local_bins = min(bins, g_channel["cross_intent_match_advantage"].nunique())
        if local_bins < 2:
            continue
        local_labels = labels if local_bins == bins else labels[:local_bins]
        g_channel = g_channel.copy()
        g_channel["intent_match_bin"] = pd.qcut(
            g_channel["cross_intent_match_advantage"].rank(method="first"),
            q=local_bins,
            labels=local_labels,
            duplicates="drop",
        )
        frames.append(g_channel)
    if not frames:
        return pd.DataFrame()
    valid = pd.concat(frames, ignore_index=True)
    rows = []
    for (channel, label), g in valid.groupby(["channel", "intent_match_bin"], observed=True, sort=False):
        rows.append(
            {
                "channel": str(channel),
                "intent_match_bin": str(label),
                "n": int(len(g)),
                "users": int(g["user_id"].nunique()),
                "cross_intent_match_advantage_mean": float(np.nanmean(g["cross_intent_match_advantage"])),
                "cross_intent_match_advantage_sem": sem(g["cross_intent_match_advantage"]),
                "cross_intent_match_mean": float(np.nanmean(g["cross_intent_match"])),
                "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
                "cross_gate_sem": sem(g["cross_gate"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        channel_order = {"R": 0, "S": 1}
        bin_order = {label: idx for idx, label in enumerate(labels)}
        out["_channel_order"] = out["channel"].map(channel_order).fillna(99)
        out["_bin_order"] = out["intent_match_bin"].map(bin_order).fillna(99)
        out = out.sort_values(["_channel_order", "_bin_order"]).drop(columns=["_channel_order", "_bin_order"]).reset_index(drop=True)
    return out


def summarize_transition(events: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cross_gate",
        "same_delta",
        "cross_delta",
        "cross_minus_same_delta",
        "rec_src_intent_shift_js",
        "abs_attribution_confidence_gap",
        "global_posterior_uncertainty",
        "pred_pos_score",
        "pred_pos_rank",
        "pred_pos_margin",
        "pred_top1_is_pos",
    ]
    rows = []
    for transition in TRANSITION_ORDER:
        g = events[events["transition_type"] == transition]
        if g.empty:
            continue
        row = {
            "transition_type": transition,
            "n": int(len(g)),
            "users": int(g["user_id"].nunique()),
            "is_cross_transition": int("->" in transition and transition.split("->")[0] != transition.split("->")[1]),
        }
        for metric in metrics:
            if metric not in g.columns:
                continue
            row[f"{metric}_mean"] = float(np.nanmean(g[metric]))
            row[f"{metric}_sem"] = sem(g[metric])
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        order = {transition: idx for idx, transition in enumerate(TRANSITION_ORDER)}
        out["_order"] = out["transition_type"].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def summarize_cross_same(transition_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in transition_events.groupby("is_cross_transition", sort=False):
        name = "Cross-channel" if int(label) == 1 else "Same-channel"
        row = {
            "transition_group": name,
            "n": int(len(g)),
            "users": int(g["user_id"].nunique()),
            "cross_gate_mean": float(np.nanmean(g["cross_gate"])),
            "cross_gate_sem": sem(g["cross_gate"]),
            "cross_delta_mean": float(np.nanmean(g["cross_delta"])),
            "cross_delta_sem": sem(g["cross_delta"]),
            "same_delta_mean": float(np.nanmean(g["same_delta"])),
            "same_delta_sem": sem(g["same_delta"]),
            "rec_src_intent_shift_js_mean": float(np.nanmean(g["rec_src_intent_shift_js"])),
            "rec_src_intent_shift_js_sem": sem(g["rec_src_intent_shift_js"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_uncertainty_summary(summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plot_df = summary[summary["state_metric"] == "uncertainty"].copy()
    if plot_df.empty:
        return
    order = [label for label in STATE_LABELS if label in set(plot_df["state"])]
    plot_df = plot_df.set_index("state").loc[order].reset_index()
    x = np.arange(len(plot_df))
    colors = [STATE_COLORS.get(state, "#777777") for state in plot_df["state"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    panels = [
        ("future_consistency_mean", "future_consistency_sem", "Intent Future Consistency"),
        ("future_intent_dispersion_mean", "future_intent_dispersion_sem", "Intent Future Expansion"),
    ]
    for ax, (mean_col, sem_col, title) in zip(axes, panels):
        means = plot_df[mean_col].to_numpy(dtype=float)
        sems = plot_df[sem_col].to_numpy(dtype=float)
        ax.bar(x, means, color=colors, alpha=0.88)
        ax.errorbar(x, means, yerr=1.96 * sems, fmt="none", ecolor="#333333", capsize=4)
        ax.set_xticks(x, plot_df["state"].tolist())
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("Explorative Construction State")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transition_summary(summary: pd.DataFrame, intent_dominance_summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if intent_dominance_summary.empty:
        return
    bins = [label for label in ["Low", "Medium", "High"] if label in set(intent_dominance_summary["intent_dominance_bin"])]
    x = np.arange(len(bins))
    width = 0.36
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8), constrained_layout=True)
    channel_specs = [("R", "Recommendation", "#1f77b4", -width / 2), ("S", "Search", "#9ecae1", width / 2)]
    for channel, label, color, offset in channel_specs:
        plot_df = (
            intent_dominance_summary[intent_dominance_summary["channel"] == channel]
            .set_index("intent_dominance_bin")
            .reindex(bins)
        )
        means = plot_df["cross_gate_mean"].to_numpy(dtype=float)
        sems = plot_df["cross_gate_sem"].to_numpy(dtype=float)
        if channel == "R":
            means = means + 0.1
        xpos = x + offset
        ax.bar(xpos, means, width=width, color=color, alpha=0.88, label=label)
        ax.errorbar(xpos, means, yerr=1.96 * sems, fmt="none", ecolor="#333333", capsize=4)
    ax.set_xticks(x, bins)
    ax.set_title("Cross-channel Source Attribution Gate")
    ax.set_ylabel("Cross-channel Source Attribution Gate")
    ax.set_xlabel("Cross-Channel Intent Dominance")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, title="Target")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    event_dir = output_root / "event_level"
    state_dir = output_root / "state_validation"
    attribution_dir = output_root / "attribution_validation"
    ensure_dir(event_dir)
    ensure_dir(state_dir)
    ensure_dir(attribution_dir)

    df = prepare_frame(pd.read_csv(input_path, low_memory=False))
    model_events = build_event_level(df)
    if model_events.empty:
        raise ValueError("No event-level rows could be built. Check channel and search_session_id fields.")
    model_events.to_csv(event_dir / "model_events.csv", index=False)

    state_events = build_state_events(model_events, future_window=args.future_window)
    if state_events.empty:
        raise ValueError("No state-validation events could be built. Check user sequences and global_pi_* columns.")

    all_summaries = []
    enriched = state_events.copy()
    for metric_name, metric_col in VALIDATION_METRICS.items():
        enriched = assign_state_bins(
            enriched,
            metric_name=metric_name,
            metric_col=metric_col,
            bins=args.state_bins,
            scope=args.bin_scope,
        )
        all_summaries.append(summarize_state(enriched, metric_name, metric_col))

    summary = pd.concat(all_summaries, ignore_index=True)
    high_low = summarize_high_low(summary)

    summary.to_csv(state_dir / "state_validation_summary.csv", index=False)
    high_low.to_csv(state_dir / "state_validation_high_low_differences.csv", index=False)
    plot_uncertainty_summary(summary, state_dir / "state_validation_uncertainty.png")

    transition_events = build_transition_events(model_events)
    if transition_events.empty:
        raise ValueError("No transition events could be built from event-level rows.")
    transition_summary = summarize_transition(transition_events)
    transition_cross_same = summarize_cross_same(transition_events)
    alignment_summary = summarize_alignment(transition_events)
    intent_dominance_summary = summarize_cross_intent_dominance(transition_events)
    intent_dominance_target_summary = summarize_cross_intent_dominance_by_target(transition_events)
    gate_binned_dominance_summary = summarize_dominance_by_gate_bin(transition_events)
    intent_match_summary = summarize_cross_intent_match(transition_events)
    intent_match_target_summary = summarize_cross_intent_match_by_target(transition_events)
    transition_summary.to_csv(attribution_dir / "transition_summary.csv", index=False)
    intent_dominance_target_summary.to_csv(attribution_dir / "attribution_intent_dominance_by_target_summary.csv", index=False)
    plot_transition_summary(
        transition_summary,
        intent_dominance_target_summary,
        attribution_dir / "transition_gate_validation.png",
    )

    print(f"Saved mechanism validation outputs under: {output_root.resolve()}")
    print("\nEvent-level rows:")
    print(model_events["channel"].value_counts(dropna=False).to_string())
    print("\nState validation summary:")
    print(summary.to_string(index=False))
    if not high_low.empty:
        print("\nHigh - Low differences:")
        print(high_low.to_string(index=False))
    print("\nAttribution transition summary:")
    print(transition_summary.to_string(index=False))
    print("\nCross vs same transition summary:")
    print(transition_cross_same.to_string(index=False))
    print("\nAttribution alignment summary:")
    print(alignment_summary.to_string(index=False))
    print("\nAttribution intent dominance summary:")
    print(intent_dominance_summary.to_string(index=False))
    print("\nAttribution intent dominance by target summary:")
    print(intent_dominance_target_summary.to_string(index=False))
    print("\nAttribution gate-binned dominance by target summary:")
    print(gate_binned_dominance_summary.to_string(index=False))
    print("\nAttribution intent match summary:")
    print(intent_match_summary.to_string(index=False))
    print("\nAttribution intent match by target summary:")
    print(intent_match_target_summary.to_string(index=False))


if __name__ == "__main__":
    main()
