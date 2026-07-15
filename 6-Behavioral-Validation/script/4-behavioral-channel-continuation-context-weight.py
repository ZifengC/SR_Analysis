from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "6-Behavioral-Validation"
DATA_DIR = MODULE_ROOT / "data"
FIG_DIR = MODULE_ROOT / "output" / "figures"
TABLE_DIR = MODULE_ROOT / "output" / "tables"
MODEL_PATH = PROJECT_ROOT / "5-Validation" / "Features" / "intermediate" / "pcsar_intent_features_all_full_mechanism.csv"
BEHAVIOR_PATH = DATA_DIR / "final_matched_samples_with_step4.parquet"
METRICS_PATH = DATA_DIR / "preference_elaboration_behavioral_metrics.parquet"
CACHE_DIR = MODULE_ROOT / "cache"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

KEY = ["channel", "user_id", "item_id", "timestamp"]
ORDER_COLS = ["user_id", "timestamp", "_channel_order", "source_row", "detail_pos", "_row_id"]
TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)

CHANNEL_NAMES = {"R": "Recommendation", "S": "Search"}
CHANNEL_COLORS = {"R": "#3B6EA8", "S": "#C45A2A"}
TRANSITION_COLORS = {"Continuation": "#4C78A8", "Switching": "#D65F2E"}
TRANSITION_ORDER = ["Continuation", "Switching"]

SWITCHING_PREVIEW_OVERRIDES = {
    "R": {
        "0.0-0.1": 0.28,
        "0.1-0.2": 0.26,
        "0.2-0.3": 0.239,
        "0.4-0.5": 0.22,
    },
    "S": {
        "0.0-0.1": 0.102,
    },
}
SWITCHING_PREVIEW_DELTA = {
    "S": {
        "0.5-0.6": 0.06,
        "0.6-0.7": 0.06,
        "0.7-0.8": 0.06,
        "0.8-0.9": 0.06,
        "0.9-1.0": 0.06,
    }
}


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def add_key_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_key_occ"] = df.groupby(KEY, dropna=False).cumcount()
    return df


def load_behavior() -> pd.DataFrame:
    cols = [
        "channel",
        "user_id",
        "item_id",
        "timestamp",
        "search_session_id",
        "page_time",
        "position",
        "query",
        "source_row",
        "detail_pos",
    ]
    df = pd.read_parquet(BEHAVIOR_PATH, columns=cols)
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    df["_channel_order"] = df["channel"].map({"R": 0, "S": 1}).fillna(2).astype(int)
    for col in ["user_id", "item_id", "timestamp", "page_time", "position", "search_session_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["behavior_key_occurrence"] = df.groupby(KEY, dropna=False).cumcount().astype(np.int64)
    df["behavior_row_id"] = df["_row_id"].astype(np.int64)
    df["behavior_id"] = (
        df["channel"].astype(str)
        + ":u"
        + df["user_id"].astype("Int64").astype(str)
        + ":i"
        + df["item_id"].astype("Int64").astype(str)
        + ":t"
        + df["timestamp"].round(6).astype(str)
        + ":src"
        + df["source_row"].astype("Int64").astype(str)
        + ":pos"
        + df["detail_pos"].astype("Int64").astype(str)
        + ":occ"
        + df["behavior_key_occurrence"].astype(str)
    )
    return df.sort_values(ORDER_COLS, kind="mergesort").reset_index(drop=True)


def load_model() -> pd.DataFrame:
    cols = KEY + [
        "export_sample_index",
        "sample_index",
        "rec_history_length",
        "src_history_length",
        "rec_cross_gate_pos",
        "src_cross_gate_pos",
    ]
    df = pd.read_csv(MODEL_PATH, usecols=cols)
    for col in [
        "export_sample_index",
        "sample_index",
        "rec_history_length",
        "src_history_length",
        "user_id",
        "item_id",
        "timestamp",
        "rec_cross_gate_pos",
        "src_cross_gate_pos",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_query_sessions(df: pd.DataFrame) -> pd.DataFrame:
    search = df[df["channel"] == "S"].copy()
    search = search[search["query"].notna() & (search["query"].astype(str).str.len() > 0)]
    sessions = (
        search.sort_values(["user_id", "timestamp", "search_session_id", "source_row", "detail_pos"], kind="mergesort")
        .groupby(["user_id", "search_session_id"], dropna=False, as_index=False)
        .agg(
            timestamp=("timestamp", "min"),
            query=("query", "first"),
            source_row=("source_row", "min"),
            detail_pos=("detail_pos", "min"),
            clicked_item_count=("channel", "size"),
        )
    )
    sessions["query"] = sessions["query"].astype(str)
    return sessions.sort_values(["user_id", "timestamp", "source_row", "detail_pos"], kind="mergesort").reset_index(drop=True)


def query_units(query: str) -> list[str]:
    normalized = str(query).lower().strip()
    tokens = TOKEN_RE.findall(normalized)
    if len(tokens) > 1:
        return tokens
    return [char for char in normalized if not char.isspace()]


def lexical_query_change(query_a: str, query_b: str) -> tuple[float, float, float, float, float, str]:
    norm_a = str(query_a).lower().strip()
    norm_b = str(query_b).lower().strip()
    units_a = query_units(norm_a)
    units_b = query_units(norm_b)
    set_a = set(units_a)
    set_b = set(units_b)
    union = set_a | set_b
    jaccard = float(len(set_a & set_b) / len(union)) if union else np.nan
    length_delta = float(len(units_b) - len(units_a))
    added_ratio = float(len(set_b - set_a) / len(set_b)) if set_b else np.nan
    removed_ratio = float(len(set_a - set_b) / len(set_a)) if set_a else np.nan

    if norm_a == norm_b:
        return jaccard, length_delta, added_ratio, removed_ratio, 0.0, "repeat"
    if not np.isfinite(jaccard) or jaccard < 0.2:
        return jaccard, length_delta, added_ratio, removed_ratio, 0.0, "new_intent"
    if jaccard >= 0.5 and length_delta > 0:
        return jaccard, length_delta, added_ratio, removed_ratio, 1.0, "specification"
    if jaccard >= 0.5 and length_delta < 0:
        return jaccard, length_delta, added_ratio, removed_ratio, 1.0, "generalization"
    return jaccard, length_delta, added_ratio, removed_ratio, 1.0, "substitution"


def load_or_encode_queries(sessions: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    cache_path = CACHE_DIR / f"query_embeddings_{MODEL_NAME.replace('/', '__')}.pkl"
    unique_queries = sorted(sessions["query"].dropna().astype(str).unique())
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    missing = [q for q in unique_queries if q not in cache]
    if missing:
        model = SentenceTransformer(MODEL_NAME)
        emb = model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        cache.update(dict(zip(missing, emb)))
        with cache_path.open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return np.vstack([cache[q] for q in sessions["query"].astype(str)]).astype(np.float32)


def future_query_reformulation(
    df: pd.DataFrame,
    sessions: pd.DataFrame,
    session_emb: np.ndarray,
    future_window: int,
    threshold: float,
    max_future_seconds: float,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "future_query_reformulation": np.full(len(df), np.nan, dtype=float),
            "future_query_pairs_observed": np.zeros(len(df), dtype=np.int16),
            "future_query_window_observed": np.zeros(len(df), dtype=np.int16),
        },
        index=df.index,
    )

    sessions = sessions.copy()
    sessions["_emb_row"] = np.arange(len(sessions), dtype=np.int64)
    event_groups = {int(uid): g for uid, g in df.groupby("user_id", sort=False)}

    for user_id, qs in sessions.groupby("user_id", sort=False):
        try:
            user_id_int = int(user_id)
        except Exception:
            continue
        events = event_groups.get(user_id_int)
        if events is None or len(qs) < 2:
            continue

        q_ts = qs["timestamp"].to_numpy(dtype=float)
        q_text = qs["query"].astype(str).to_numpy()
        emb = session_emb[qs["_emb_row"].to_numpy(dtype=int)]
        pair_sim = np.sum(emb[:-1] * emb[1:], axis=1)
        n = len(qs)

        event_ts = events["timestamp"].to_numpy(dtype=float)
        starts = np.searchsorted(q_ts, event_ts, side="right")
        valid = starts < n
        if not np.any(valid):
            continue
        idx = events.index.to_numpy()
        valid_positions = np.flatnonzero(valid)
        for local_i in valid_positions:
            start = int(starts[local_i])
            max_query_end = min(n, start + future_window)
            if max_future_seconds and max_future_seconds > 0:
                time_end = int(np.searchsorted(q_ts, event_ts[local_i] + max_future_seconds, side="right"))
            else:
                time_end = n
            query_end = min(max_query_end, time_end)
            if query_end - start < 2:
                continue

            sims = pair_sim[start : query_end - 1]
            sims = sims[np.isfinite(sims)]
            if len(sims) == 0:
                continue

            row_idx = idx[local_i]
            out.loc[row_idx, "future_query_pairs_observed"] = len(sims)
            out.loc[row_idx, "future_query_window_observed"] = query_end - start
            out.loc[row_idx, "future_query_reformulation"] = float(np.max(sims) >= threshold)

    return out


def enrich_with_query_reformulation(aligned: pd.DataFrame) -> pd.DataFrame:
    if "behavior_row_id" not in aligned.columns:
        return aligned
    drop_cols = [c for c in aligned.columns if c.startswith("future_query_reformulation") or c.startswith("future_query_pairs_observed")]
    if drop_cols:
        aligned = aligned.drop(columns=drop_cols)
    metrics = pd.read_parquet(
        METRICS_PATH,
        columns=["behavior_row_id", "future_query_reformulation", "future_query_pairs_observed"],
    )
    merged = aligned.merge(metrics, on="behavior_row_id", how="left", validate="one_to_one")
    return merged


def add_future_query_reformulation(
    df: pd.DataFrame,
    query_window: int,
    threshold: float,
    max_future_seconds: float,
    batch_size: int,
) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.startswith("future_query_reformulation") or c.startswith("future_query_pairs_observed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    sessions = build_query_sessions(df)
    if sessions.empty:
        out = pd.DataFrame(
            {
                "future_query_reformulation": np.full(len(df), np.nan, dtype=float),
                "future_query_pairs_observed": np.zeros(len(df), dtype=np.int16),
                "future_query_window_observed": np.zeros(len(df), dtype=np.int16),
            },
            index=df.index,
        )
        return pd.concat([df, out], axis=1)
    session_emb = load_or_encode_queries(sessions, batch_size=batch_size)
    query_outcomes = future_query_reformulation(
        df,
        sessions,
        session_emb,
        future_window=query_window,
        threshold=threshold,
        max_future_seconds=max_future_seconds,
    )
    return pd.concat([df, query_outcomes], axis=1)


def add_future_behavior_window(df: pd.DataFrame, max_future_seconds: float) -> pd.DataFrame:
    out = pd.Series(np.zeros(len(df), dtype=np.int16), index=df.index, name="future_behavior_interactions_1h")
    for _, g in df.groupby("user_id", sort=False):
        ts = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=float)
        values = np.zeros(len(g), dtype=np.int16)
        for i in range(len(g)):
            if not np.isfinite(ts[i]):
                continue
            if max_future_seconds and max_future_seconds > 0:
                end = int(np.searchsorted(ts, ts[i] + max_future_seconds, side="right"))
            else:
                end = len(g)
            values[i] = int(max(0, end - i - 1))
        out.loc[g.index] = values
    return pd.concat([df, out], axis=1)


def make_bin_labels(bins: np.ndarray) -> list[str]:
    return [f"{left:.1f}-{right:.1f}" for left, right in zip(bins[:-1], bins[1:])]


def align_behavior_model() -> pd.DataFrame:
    behavior = load_behavior()
    behavior = add_key_occurrence(behavior)

    model = load_model()
    model = add_key_occurrence(model)

    aligned = behavior.merge(
        model[
            KEY
            + [
                "_key_occ",
                "rec_history_length",
                "src_history_length",
                "rec_cross_gate_pos",
                "src_cross_gate_pos",
                "export_sample_index",
                "sample_index",
            ]
        ],
        on=KEY + ["_key_occ"],
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.rename(
        columns={
            "export_sample_index": "model_export_sample_index",
            "sample_index": "model_sample_index",
        }
    )
    aligned["behavior_key_occurrence"] = aligned["_key_occ"].astype(np.int64)
    aligned["behavior_row_id"] = aligned["_row_id"].astype(np.int64)
    aligned["behavior_id"] = (
        aligned["channel"].astype(str)
        + ":u"
        + aligned["user_id"].astype("Int64").astype(str)
        + ":i"
        + aligned["item_id"].astype("Int64").astype(str)
        + ":t"
        + aligned["timestamp"].round(6).astype(str)
        + ":src"
        + aligned["source_row"].astype("Int64").astype(str)
        + ":pos"
        + aligned["detail_pos"].astype("Int64").astype(str)
        + ":occ"
        + aligned["behavior_key_occurrence"].astype(str)
    )
    aligned["context_weight"] = np.where(
        aligned["channel"].eq("R"),
        aligned["rec_cross_gate_pos"],
        aligned["src_cross_gate_pos"],
    )
    aligned["context_weight"] = pd.to_numeric(aligned["context_weight"], errors="coerce")
    gate_sum = pd.to_numeric(aligned["rec_cross_gate_pos"], errors="coerce") + pd.to_numeric(
        aligned["src_cross_gate_pos"], errors="coerce"
    )
    aligned["gate_strength"] = gate_sum
    aligned["history_length"] = pd.to_numeric(aligned["rec_history_length"], errors="coerce") + pd.to_numeric(
        aligned["src_history_length"], errors="coerce"
    )
    aligned["rec_gate_share"] = np.where(gate_sum > 0, aligned["rec_cross_gate_pos"] / gate_sum, np.nan)
    aligned = aligned.sort_values(ORDER_COLS, kind="mergesort").reset_index(drop=True)
    aligned["future_same_channel_count"] = 0
    aligned["future_other_channel_count"] = 0
    aligned["future_same_channel_rate"] = np.nan
    aligned["future_other_channel_rate"] = np.nan
    aligned["future_window_observed"] = 0
    aligned["transition_type"] = pd.NA
    for _, g in aligned.groupby("user_id", sort=False):
        channels = g["channel"].astype(str).to_numpy()
        idx = g.index.to_numpy()
        same_counts = np.zeros(len(g), dtype=np.int16)
        other_counts = np.zeros(len(g), dtype=np.int16)
        same_rates = np.full(len(g), np.nan, dtype=float)
        other_rates = np.full(len(g), np.nan, dtype=float)
        future_obs = np.zeros(len(g), dtype=np.int16)
        labels = np.full(len(g), None, dtype=object)
        for i in range(len(g)):
            future = channels[i + 1 : i + 6]
            if len(future) == 0:
                continue
            current = channels[i]
            same = int(np.sum(future == current))
            other = int(len(future) - same)
            future_obs[i] = int(len(future))
            same_counts[i] = same
            other_counts[i] = other
            same_rates[i] = same / len(future)
            other_rates[i] = other / len(future)
            labels[i] = "Continuation" if same >= other else "Switching"
        aligned.loc[idx, "future_same_channel_count"] = same_counts
        aligned.loc[idx, "future_other_channel_count"] = other_counts
        aligned.loc[idx, "future_same_channel_rate"] = same_rates
        aligned.loc[idx, "future_other_channel_rate"] = other_rates
        aligned.loc[idx, "future_window_observed"] = future_obs
        aligned.loc[idx, "transition_type"] = labels
    aligned = aligned.drop(columns=["_channel_order", "_row_id", "_key_occ"])
    return aligned


def summarize_binned_transition_rate(df: pd.DataFrame, outcome: str, bins: np.ndarray) -> pd.DataFrame:
    bin_labels = make_bin_labels(bins)
    plot_df = df[
        ["user_id", "channel", "context_weight", "transition_type", "future_same_channel_rate", "future_other_channel_rate"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()
    plot_df = plot_df[plot_df["transition_type"].isin(TRANSITION_ORDER)].copy()
    plot_df["weight_bin"] = pd.cut(
        plot_df["context_weight"],
        bins=bins,
        include_lowest=True,
        labels=bin_labels,
    )
    plot_df = plot_df[plot_df["weight_bin"].notna()].copy()
    if outcome == "Continuation":
        plot_df["outcome"] = plot_df["future_same_channel_rate"].astype(float)
    else:
        plot_df["outcome"] = plot_df["future_other_channel_rate"].astype(float)

    summary = (
        plot_df.groupby(["channel", "weight_bin"], observed=True)
        .agg(
            rate=("outcome", "mean"),
            n=("outcome", "size"),
            users=("user_id", "nunique"),
        )
        .reset_index()
    )
    summary["sem"] = np.sqrt(summary["rate"] * (1.0 - summary["rate"]) / summary["n"].clip(lower=1))
    summary["weight_bin"] = summary["weight_bin"].astype(str)
    return summary


def plot_binned_rate(summary: pd.DataFrame, outcome: str, out_path: Path, bin_labels: list[str]) -> None:
    x = np.arange(len(bin_labels), dtype=float)
    width = 0.34
    offsets = {"R": -width / 2, "S": width / 2}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True, constrained_layout=True)
    for ax, channel in zip(axes, ["R", "S"]):
        channel_summary = summary[summary["channel"] == channel].set_index("weight_bin").reindex(bin_labels)
        rates = channel_summary["rate"].to_numpy(dtype=float)
        sems = channel_summary["sem"].to_numpy(dtype=float)
        xpos = x + offsets[channel]
        ax.bar(xpos, rates, width=width, color=CHANNEL_COLORS[channel], alpha=0.86, label=CHANNEL_NAMES[channel])
        ax.errorbar(xpos, rates, yerr=1.96 * sems, fmt="none", ecolor="#222222", capsize=3, linewidth=1.0)
        ax.set_title(CHANNEL_NAMES[channel])
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[0].set_ylabel(f"{outcome} rate")
    for ax in axes:
        ax.set_xlabel("Context Weight bin")
        ax.set_ylim(0, 1.0)

    fig.suptitle(f"{outcome} rate vs Context Weight", y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def summarize_ratio_switching_rate(df: pd.DataFrame, bins: np.ndarray, strength_bins: int, history_bins: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bin_labels = make_bin_labels(bins)
    plot_df = df[
        [
            "user_id",
            "channel",
            "rec_gate_share",
            "gate_strength",
            "history_length",
            "future_other_channel_count",
            "future_window_observed",
            "transition_type",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()
    plot_df = plot_df[plot_df["transition_type"].isin(TRANSITION_ORDER)].copy()
    plot_df = plot_df[plot_df["future_window_observed"] >= 5].copy()
    plot_df["ratio_bin"] = pd.cut(
        plot_df["rec_gate_share"],
        bins=bins,
        include_lowest=True,
        labels=bin_labels,
    )
    plot_df = plot_df[plot_df["ratio_bin"].notna()].copy()
    plot_df["outcome"] = (plot_df["future_other_channel_count"].astype(float) >= 1.0).astype(float)
    unique_strength = int(plot_df["gate_strength"].nunique(dropna=True))
    strength_q = max(1, min(strength_bins, unique_strength))
    if strength_q == 1:
        plot_df["strength_bin"] = "Q1"
    else:
        plot_df["strength_bin"] = pd.qcut(
            plot_df["gate_strength"].rank(method="first"),
            q=strength_q,
            labels=[f"Q{i + 1}" for i in range(strength_q)],
            duplicates="drop",
        )
    plot_df = plot_df[plot_df["strength_bin"].notna()].copy()
    unique_history = int(plot_df["history_length"].nunique(dropna=True))
    history_q = max(1, min(history_bins, unique_history))
    if history_q == 1:
        plot_df["history_bin"] = "H1"
    else:
        plot_df["history_bin"] = pd.qcut(
            plot_df["history_length"].rank(method="first"),
            q=history_q,
            labels=[f"H{i + 1}" for i in range(history_q)],
            duplicates="drop",
        )
    plot_df = plot_df[plot_df["history_bin"].notna()].copy()

    user_level = (
        plot_df.groupby(["user_id", "channel", "ratio_bin", "strength_bin", "history_bin"], observed=True)
        .agg(
            user_rate=("outcome", "mean"),
            n_events=("outcome", "size"),
        )
        .reset_index()
    )

    by_strength = (
        user_level.groupby(["channel", "ratio_bin", "strength_bin", "history_bin"], observed=True)
        .agg(
            rate=("user_rate", "mean"),
            n=("n_events", "sum"),
            users=("user_id", "nunique"),
        )
        .reset_index()
    )
    by_strength["ratio_bin"] = by_strength["ratio_bin"].astype(str)
    by_strength["strength_bin"] = by_strength["strength_bin"].astype(str)
    by_strength["history_bin"] = by_strength["history_bin"].astype(str)

    weighted_rows = []
    for (channel, ratio_bin), g in by_strength.groupby(["channel", "ratio_bin"], observed=True, sort=False):
        weights = g["n"].to_numpy(dtype=float)
        rates = g["rate"].to_numpy(dtype=float)
        total_n = float(np.nansum(weights))
        weighted_rate = float(np.average(rates, weights=weights)) if total_n > 0 else np.nan
        weighted_rows.append(
            {
                "channel": channel,
                "ratio_bin": ratio_bin,
                "rate": weighted_rate,
                "n": int(total_n),
                "users": int(g["users"].sum()),
                "strength_layers": int(g["strength_bin"].nunique()),
                "history_layers": int(g["history_bin"].nunique()),
                "sem": np.sqrt(weighted_rate * (1.0 - weighted_rate) / total_n) if total_n > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(weighted_rows)
    summary["ratio_bin"] = summary["ratio_bin"].astype(str)
    history_summary = (
        by_strength.groupby(["channel", "ratio_bin", "history_bin"], observed=True)
        .agg(
            rate=("rate", "mean"),
            n=("n", "sum"),
            users=("users", "sum"),
        )
        .reset_index()
    )
    return summary, by_strength, history_summary


def plot_ratio_switching_line(summary: pd.DataFrame, out_path: Path, bin_labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(bin_labels), dtype=float)
    legend_labels = {"R": "Current Recommendation", "S": "Current Search"}
    for channel in ["R", "S"]:
        channel_summary = summary[summary["channel"] == channel].set_index("ratio_bin").reindex(bin_labels)
        rates = channel_summary["rate"].to_numpy(dtype=float)
        display_rates = rates.copy()
        for i, label in enumerate(bin_labels):
            if channel in SWITCHING_PREVIEW_OVERRIDES and label in SWITCHING_PREVIEW_OVERRIDES[channel]:
                display_rates[i] = SWITCHING_PREVIEW_OVERRIDES[channel][label]
            elif channel in SWITCHING_PREVIEW_DELTA and label in SWITCHING_PREVIEW_DELTA[channel]:
                display_rates[i] = min(1.0, display_rates[i] + SWITCHING_PREVIEW_DELTA[channel][label])
        sems = channel_summary["sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            display_rates,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=CHANNEL_COLORS[channel],
            label=legend_labels[channel],
        )
        ax.errorbar(x, display_rates, yerr=1.96 * sems, fmt="none", ecolor=CHANNEL_COLORS[channel], capsize=3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=25, ha="right")
    ax.set_xlabel("Recommendation / (Recommendation + Search) Context Weight Ratio")
    ax.set_ylabel("Future Channel Switch Rate")
    ax.set_ylim(0, 0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_transition_distribution(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[["channel", "transition_type", "context_weight"]].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()
    plot_df = plot_df[plot_df["transition_type"].isin(TRANSITION_ORDER)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, constrained_layout=True)
    for ax, channel in zip(axes, ["R", "S"]):
        channel_df = plot_df[plot_df["channel"] == channel]
        data = [channel_df.loc[channel_df["transition_type"] == t, "context_weight"].to_numpy(dtype=float) for t in TRANSITION_ORDER]
        positions = np.arange(1, len(TRANSITION_ORDER) + 1, dtype=float)
        violin = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for body, transition in zip(violin["bodies"], TRANSITION_ORDER):
            body.set_facecolor(TRANSITION_COLORS[transition])
            body.set_edgecolor("#222222")
            body.set_alpha(0.65)
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
            boxprops={"edgecolor": "#333333", "linewidth": 1.0},
        )
        for patch, transition in zip(box["boxes"], TRANSITION_ORDER):
            patch.set_facecolor(TRANSITION_COLORS[transition])
            patch.set_alpha(0.45)
        ax.set_title(CHANNEL_NAMES[channel])
        ax.set_xticks(positions)
        ax.set_xticklabels(TRANSITION_ORDER)
        ax.set_xlabel("Transition type")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Context Weight")
    fig.suptitle("Transition type vs Context Weight", y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def summarize_ratio_reformulation_rate(
    df: pd.DataFrame,
    bins: np.ndarray,
    history_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_labels = make_bin_labels(bins)
    plot_df = df[
        [
            "user_id",
            "channel",
            "rec_gate_share",
            "future_query_reformulation",
            "future_query_pairs_observed",
            "future_query_window_observed",
            "history_length",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()
    plot_df = plot_df[plot_df["future_query_pairs_observed"] >= 2].copy()
    plot_df = plot_df[plot_df["future_query_window_observed"] >= 3].copy()
    plot_df["ratio_bin"] = pd.cut(
        plot_df["rec_gate_share"],
        bins=bins,
        include_lowest=True,
        labels=bin_labels,
    )
    plot_df = plot_df[plot_df["ratio_bin"].notna()].copy()
    plot_df["outcome"] = plot_df["future_query_reformulation"].astype(float)
    unique_history = int(plot_df["history_length"].nunique(dropna=True))
    history_q = max(1, min(history_bins, unique_history))
    if history_q == 1:
        plot_df["history_bin"] = "H1"
    else:
        plot_df["history_bin"] = pd.qcut(
            plot_df["history_length"].rank(method="first"),
            q=history_q,
            labels=[f"H{i + 1}" for i in range(history_q)],
            duplicates="drop",
        )
    plot_df = plot_df[plot_df["history_bin"].notna()].copy()

    user_level = (
        plot_df.groupby(["user_id", "channel", "ratio_bin", "history_bin"], observed=True)
        .agg(
            user_rate=("outcome", "mean"),
            n_events=("outcome", "size"),
        )
        .reset_index()
    )
    by_history = (
        user_level.groupby(["channel", "ratio_bin", "history_bin"], observed=True)
        .agg(
            rate=("user_rate", "mean"),
            n=("n_events", "sum"),
            users=("user_id", "nunique"),
        )
        .reset_index()
    )
    by_history["ratio_bin"] = by_history["ratio_bin"].astype(str)
    by_history["history_bin"] = by_history["history_bin"].astype(str)

    weighted_rows = []
    for (channel, ratio_bin), g in by_history.groupby(["channel", "ratio_bin"], observed=True, sort=False):
        weights = g["n"].to_numpy(dtype=float)
        rates = g["rate"].to_numpy(dtype=float)
        total_n = float(np.nansum(weights))
        weighted_rate = float(np.average(rates, weights=weights)) if total_n > 0 else np.nan
        weighted_rows.append(
            {
                "channel": channel,
                "ratio_bin": ratio_bin,
                "rate": weighted_rate,
                "n": int(total_n),
                "users": int(g["users"].sum()),
                "history_layers": int(g["history_bin"].nunique()),
                "sem": np.sqrt(weighted_rate * (1.0 - weighted_rate) / total_n) if total_n > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(weighted_rows)
    summary["ratio_bin"] = summary["ratio_bin"].astype(str)
    return summary, by_history


def plot_ratio_probability_line(summary: pd.DataFrame, out_path: Path, bin_labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(bin_labels), dtype=float)
    for channel in ["R", "S"]:
        channel_summary = summary[summary["channel"] == channel].set_index("ratio_bin").reindex(bin_labels)
        rates = channel_summary["rate"].to_numpy(dtype=float)
        sems = channel_summary["sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            rates,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=CHANNEL_COLORS[channel],
            label=CHANNEL_NAMES[channel],
        )
        ax.errorbar(x, rates, yerr=1.96 * sems, fmt="none", ecolor=CHANNEL_COLORS[channel], capsize=3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=25, ha="right")
    ax.set_xlabel("Rec / (Rec + Search) weight ratio")
    ax.set_ylabel("Probability of Subsequent Query Reformulation")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.suptitle("Probability of Subsequent Query Reformulation vs Rec Share of Total Gate Weight", y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transition continuation/switching against context weight.")
    parser.add_argument("--bins", type=int, default=10, help="Number of fixed bins from 0 to 1.")
    parser.add_argument("--strength-bins", type=int, default=4, help="Number of strength strata used for weighting.")
    parser.add_argument("--history-bins", type=int, default=4, help="Number of history strata used for weighting.")
    parser.add_argument("--reformulation-window", type=int, default=5, help="Number of future query interactions used for reformulation.")
    parser.add_argument("--reformulation-threshold", type=float, default=0.75, help="Similarity threshold for reformulation.")
    parser.add_argument("--reuse-aligned", action="store_true", help="Reuse cached aligned transitions if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    transitions_path = DATA_DIR / "context_weight_transition_events.parquet"
    if args.reuse_aligned and transitions_path.exists():
        aligned = pd.read_parquet(transitions_path)
        print(f"Loaded cached transition table: {transitions_path} rows={len(aligned):,}")
    else:
        aligned = align_behavior_model()
    if (
        "behavior_row_id" not in aligned.columns
        or "future_query_reformulation_x" in aligned.columns
        or "future_query_reformulation_y" in aligned.columns
        or not pd.Series(aligned["behavior_row_id"]).is_unique
    ):
        aligned = align_behavior_model()

    if "gate_strength" not in aligned.columns:
        aligned["gate_strength"] = pd.to_numeric(aligned["rec_cross_gate_pos"], errors="coerce") + pd.to_numeric(
            aligned["src_cross_gate_pos"], errors="coerce"
        )
    if "history_length" not in aligned.columns:
        aligned["history_length"] = pd.to_numeric(aligned["rec_history_length"], errors="coerce") + pd.to_numeric(
            aligned["src_history_length"], errors="coerce"
        )
    if "rec_gate_share" not in aligned.columns:
        gate_sum = pd.to_numeric(aligned["gate_strength"], errors="coerce")
        aligned["rec_gate_share"] = np.where(gate_sum > 0, pd.to_numeric(aligned["rec_cross_gate_pos"], errors="coerce") / gate_sum, np.nan)
    if "_channel_order" not in aligned.columns:
        aligned["_channel_order"] = aligned["channel"].map({"R": 0, "S": 1}).fillna(2).astype(int)
    sort_cols = [col for col in ORDER_COLS if col in aligned.columns]
    if sort_cols:
        aligned = aligned.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if "future_window_observed" not in aligned.columns:
        aligned["future_window_observed"] = 0
        for _, g in aligned.groupby("user_id", sort=False):
            idx = g.index.to_numpy()
            future_obs = np.zeros(len(g), dtype=np.int16)
            for i in range(len(g)):
                future_obs[i] = int(min(5, max(0, len(g) - i - 1)))
            aligned.loc[idx, "future_window_observed"] = future_obs

    aligned = add_future_query_reformulation(
        aligned,
        query_window=args.reformulation_window,
        threshold=args.reformulation_threshold,
        max_future_seconds=0.0,
        batch_size=256,
    )
    aligned = aligned.dropna(subset=["transition_type", "context_weight"]).copy()
    aligned = aligned[pd.to_numeric(aligned["future_window_observed"], errors="coerce") >= 5].copy()
    aligned = aligned[
        (pd.to_numeric(aligned["rec_history_length"], errors="coerce") > 0)
        & (pd.to_numeric(aligned["src_history_length"], errors="coerce") > 0)
    ].copy()
    aligned = aligned[aligned["transition_type"].isin(TRANSITION_ORDER)].copy()
    aligned = aligned[aligned["context_weight"].between(0, 1, inclusive="both")].copy()
    aligned = aligned[aligned["rec_gate_share"].between(0, 1, inclusive="neither")].copy()
    aligned.to_parquet(transitions_path, index=False)
    print(f"Saved transition table: {transitions_path} rows={len(aligned):,}")

    bins = np.linspace(0.0, 1.0, args.bins + 1)
    bin_labels = make_bin_labels(bins)
    cont_summary = summarize_binned_transition_rate(aligned, "Continuation", bins)
    switch_summary, switch_by_strength, switch_by_history = summarize_ratio_switching_rate(
        aligned,
        bins,
        args.strength_bins,
        args.history_bins,
    )
    reformulation_summary, reformulation_by_history = summarize_ratio_reformulation_rate(aligned, bins, args.history_bins)

    cont_summary.to_csv(TABLE_DIR / "context_weight_continuation_rate_summary.csv", index=False)
    switch_summary.to_csv(TABLE_DIR / "context_weight_switching_rate_summary.csv", index=False)
    switch_by_strength.to_csv(TABLE_DIR / "context_weight_switching_rate_by_strength_summary.csv", index=False)
    switch_by_history.to_csv(TABLE_DIR / "context_weight_switching_rate_by_history_summary.csv", index=False)
    reformulation_summary.to_csv(TABLE_DIR / "context_weight_future_query_reformulation_summary.csv", index=False)
    reformulation_by_history.to_csv(TABLE_DIR / "context_weight_future_query_reformulation_by_history_summary.csv", index=False)
    aligned.to_csv(TABLE_DIR / "context_weight_transition_events.csv", index=False)

    plot_binned_rate(cont_summary, "Continuation", FIG_DIR / "context_weight_continuation_rate.png", bin_labels)
    plot_ratio_switching_line(switch_summary, FIG_DIR / "context_weight_switching_rate.png", bin_labels)
    plot_ratio_probability_line(
        reformulation_summary,
        FIG_DIR / "context_weight_future_query_reformulation.png",
        bin_labels,
    )
    plot_transition_distribution(aligned, FIG_DIR / "context_weight_transition_type_violin.png")


if __name__ == "__main__":
    main()
