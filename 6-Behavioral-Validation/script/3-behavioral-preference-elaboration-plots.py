from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "6-Behavioral-Validation"
DATA_DIR = MODULE_ROOT / "data"
FIG_DIR = MODULE_ROOT / "output" / "figures"
CACHE_DIR = MODULE_ROOT / "cache"

BEHAVIOR_PATH = DATA_DIR / "final_matched_samples_with_step4.parquet"
MODEL_PATH = PROJECT_ROOT / "5-Validation" / "Features" / "intermediate" / "pcsar_intent_features_all_full_mechanism.csv"
METRICS_PATH = DATA_DIR / "preference_elaboration_behavioral_metrics.parquet"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

KEY = ["channel", "user_id", "item_id", "timestamp"]
ORDER_COLS = ["user_id", "timestamp", "_channel_order", "source_row", "detail_pos", "_row_id"]
CHANNEL_COLORS = {"R": "#3B6EA8", "S": "#C45A2A"}
STATE_LABELS = ["Low", "Medium", "High"]
STATE_COLORS = {"Low": "#4C78A8", "Medium": "#8A8A8A", "High": "#D65F2E"}
PLOTTED_STATES = ["Low", "High"]
CHANGE_TYPE_ORDER = ["substitution", "specification", "generalization"]
CHANGE_TYPE_LABELS = {
    "substitution": "Substitution",
    "specification": "Specification",
    "generalization": "Generalization",
}
CHANGE_TYPE_COLORS = {
    "substitution": "#D65F2E",
    "specification": "#4C78A8",
    "generalization": "#59A14F",
}


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def ensure_dirs() -> None:
    for path in (FIG_DIR, CACHE_DIR):
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
    return df.sort_values(ORDER_COLS, kind="mergesort").reset_index(drop=True)


def load_model() -> pd.DataFrame:
    cols = KEY + [
        "export_sample_index",
        "sample_index",
        "global_posterior_uncertainty",
        "global_intent_entropy",
        "global_belief_entropy_mean",
        "global_belief_confidence_mean",
        "src_pred_pos_score",
        "rec_history_length",
        "src_history_length",
    ]
    df = pd.read_csv(MODEL_PATH, usecols=cols)
    for col in [
        "export_sample_index",
        "user_id",
        "item_id",
        "timestamp",
        "global_posterior_uncertainty",
        "global_intent_entropy",
        "global_belief_entropy_mean",
        "global_belief_confidence_mean",
        "src_pred_pos_score",
        "rec_history_length",
        "src_history_length",
        "sample_index",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_future_position_depth(df: pd.DataFrame, window: int) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(["user_id", "channel"], sort=False):
        pos = pd.to_numeric(g["position"], errors="coerce").to_numpy(dtype=float)
        values = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            future = pos[i + 1 : i + 1 + window]
            future = future[np.isfinite(future)]
            if len(future) and np.isfinite(pos[i]):
                values[i] = float(np.mean(future) - pos[i])
        out.loc[g.index] = values
    return out


def compute_future_dwell_time(df: pd.DataFrame, window: int) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    rec = df[df["channel"] == "R"]
    search = df[df["channel"] == "S"]
    for _, g in rec.groupby(["user_id", "channel"], sort=False):
        page_time = pd.to_numeric(g["page_time"], errors="coerce").to_numpy(dtype=float)
        values = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            future = page_time[i + 1 : i + 1 + window]
            future = future[np.isfinite(future)]
            if len(future):
                values[i] = float(np.mean(future))
        out.loc[g.index] = values
    for _, g in search.groupby(["user_id", "channel", "search_session_id"], sort=False, dropna=False):
        ts = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(ts)
        session_span = float(np.max(ts[finite]) - np.min(ts[finite])) if np.any(finite) else np.nan
        values = np.full(len(g), session_span, dtype=float)
        out.loc[g.index] = values
    return out


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


def load_or_encode_queries(sessions: pd.DataFrame, batch_size: int) -> np.ndarray:
    cache_path = CACHE_DIR / f"query_embeddings_{MODEL_NAME.replace('/', '__')}.pkl"
    unique_queries = sorted(sessions["query"].dropna().astype(str).unique())
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    missing = [q for q in unique_queries if q not in cache]
    if missing:
        print(f"Encoding missing unique queries: {len(missing):,}")
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
        print(f"Saved query embedding cache: {cache_path}")
    else:
        print(f"Loaded complete query embedding cache: {cache_path}")

    return np.vstack([cache[q] for q in sessions["query"].astype(str)]).astype(np.float32)


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
            "next_query_reformulation": np.full(len(df), np.nan, dtype=float),
            "next_query_similarity": np.full(len(df), np.nan, dtype=float),
            "next_query_token_jaccard": np.full(len(df), np.nan, dtype=float),
            "next_query_length_delta": np.full(len(df), np.nan, dtype=float),
            "next_query_added_token_ratio": np.full(len(df), np.nan, dtype=float),
            "next_query_removed_token_ratio": np.full(len(df), np.nan, dtype=float),
            "next_query_lexical_reformulation": np.full(len(df), np.nan, dtype=float),
            "next_query_change_type": pd.Series(pd.NA, index=df.index, dtype="object"),
            "future_query_reformulation_length": np.full(len(df), np.nan, dtype=float),
            "future_query_max_adjacent_similarity": np.full(len(df), np.nan, dtype=float),
            "future_query_pairs_observed": np.zeros(len(df), dtype=np.int16),
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
            out.loc[row_idx, "next_query_similarity"] = float(sims[0])
            out.loc[row_idx, "next_query_reformulation"] = float(sims[0] >= threshold)
            (
                jaccard,
                length_delta,
                added_ratio,
                removed_ratio,
                lexical_reformulation,
                change_type,
            ) = lexical_query_change(q_text[start], q_text[start + 1])
            out.loc[row_idx, "next_query_token_jaccard"] = jaccard
            out.loc[row_idx, "next_query_length_delta"] = length_delta
            out.loc[row_idx, "next_query_added_token_ratio"] = added_ratio
            out.loc[row_idx, "next_query_removed_token_ratio"] = removed_ratio
            out.loc[row_idx, "next_query_lexical_reformulation"] = lexical_reformulation
            out.loc[row_idx, "next_query_change_type"] = change_type
            out.loc[row_idx, "future_query_max_adjacent_similarity"] = float(np.max(sims))
            out.loc[row_idx, "future_query_reformulation"] = float(np.max(sims) >= threshold)
            is_reformulation = sims >= threshold
            run_len = 0
            for value in is_reformulation:
                if not value:
                    break
                run_len += 1
            out.loc[row_idx, "future_query_reformulation_length"] = float(run_len)

    return out


def future_query_click_count_average(
    df: pd.DataFrame,
    sessions: pd.DataFrame,
    future_window: int,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "future_query_click_count_average": np.full(len(df), np.nan, dtype=float),
            "future_query_sessions_observed_for_click_count": np.zeros(len(df), dtype=np.int16),
        },
        index=df.index,
    )

    event_groups = {int(uid): g for uid, g in df.groupby("user_id", sort=False)}
    for user_id, qs in sessions.groupby("user_id", sort=False):
        try:
            user_id_int = int(user_id)
        except Exception:
            continue
        events = event_groups.get(user_id_int)
        if events is None or len(qs) == 0:
            continue

        q_ts = qs["timestamp"].to_numpy(dtype=float)
        click_counts = pd.to_numeric(qs["clicked_item_count"], errors="coerce").to_numpy(dtype=float)
        event_ts = events["timestamp"].to_numpy(dtype=float)
        starts = np.searchsorted(q_ts, event_ts, side="right")
        valid = starts < len(qs)
        if not np.any(valid):
            continue

        values = np.full(len(events), np.nan, dtype=float)
        observed = np.zeros(len(events), dtype=np.int16)
        for local_i, start in enumerate(starts):
            if start >= len(qs):
                continue
            future = click_counts[start : start + future_window]
            future = future[np.isfinite(future)]
            if len(future):
                values[local_i] = float(np.mean(future))
                observed[local_i] = len(future)

        idx = events.index.to_numpy()
        out.loc[idx[valid], "future_query_click_count_average"] = values[valid]
        out.loc[idx[valid], "future_query_sessions_observed_for_click_count"] = observed[valid]

    return out


def build_metrics(args: argparse.Namespace) -> pd.DataFrame:
    behavior = load_behavior()
    print(f"Loaded behavior rows: {len(behavior):,}")

    behavior["future_click_position_depth"] = compute_future_position_depth(behavior, args.position_window)
    behavior["future_dwell_time"] = compute_future_dwell_time(behavior, args.dwell_window)

    sessions = build_query_sessions(behavior)
    print(f"Unique search sessions for query reformulation: {len(sessions):,}")
    session_emb = load_or_encode_queries(sessions, args.batch_size)
    query_outcomes = future_query_reformulation(
        behavior,
        sessions,
        session_emb,
        future_window=args.query_window,
        threshold=args.reformulation_threshold,
        max_future_seconds=args.query_time_window_minutes * 60,
    )
    query_click_counts = future_query_click_count_average(
        behavior,
        sessions,
        future_window=args.query_click_window,
    )
    behavior = pd.concat([behavior, query_outcomes, query_click_counts], axis=1)

    model = load_model()
    behavior = add_key_occurrence(behavior)
    behavior["behavior_row_id"] = behavior["_row_id"].astype("int64")
    behavior["behavior_key_occurrence"] = behavior["_key_occ"].astype("int16")
    behavior["behavior_id"] = (
        behavior["channel"].astype(str)
        + ":u"
        + behavior["user_id"].astype("Int64").astype(str)
        + ":i"
        + behavior["item_id"].astype("Int64").astype(str)
        + ":t"
        + behavior["timestamp"].round(6).astype(str)
        + ":src"
        + behavior["source_row"].astype("Int64").astype(str)
        + ":pos"
        + behavior["detail_pos"].astype("Int64").astype(str)
        + ":occ"
        + behavior["behavior_key_occurrence"].astype(str)
    )
    model = add_key_occurrence(model)
    aligned = behavior.merge(
        model[
            KEY
            + [
                "_key_occ",
                "export_sample_index",
                "sample_index",
                "global_posterior_uncertainty",
                "global_intent_entropy",
                "global_belief_entropy_mean",
                "global_belief_confidence_mean",
                "src_pred_pos_score",
                "rec_history_length",
                "src_history_length",
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
    aligned = aligned.rename(columns={"global_intent_entropy": "preference_elaboration"})
    aligned["preference_elaboration_state"] = assign_preference_elaboration_state(aligned["preference_elaboration"])
    aligned["belief_entropy_position_depth_state"] = assign_preference_elaboration_state(
        aligned["global_belief_entropy_mean"]
    )
    aligned["belief_confidence_dwell_state"] = assign_preference_elaboration_state(
        aligned["global_belief_confidence_mean"]
    )
    aligned["search_duration_state_metric"] = -aligned["src_pred_pos_score"]
    aligned["channel_specific_dwell_state"] = pd.NA
    s_state = assign_preference_elaboration_state(
        aligned.loc[aligned["channel"] == "S", "search_duration_state_metric"]
    )
    aligned.loc[aligned["channel"] == "R", "channel_specific_dwell_state"] = aligned.loc[
        aligned["channel"] == "R", "belief_confidence_dwell_state"
    ]
    aligned.loc[s_state.index, "channel_specific_dwell_state"] = s_state
    aligned = aligned.drop(columns=["_channel_order", "_row_id", "_key_occ"])
    print(f"Aligned behavior-model rows: {len(aligned):,}")
    print(
        "Unique IDs: "
        f"behavior_id={aligned['behavior_id'].nunique():,}, "
        f"model_export_sample_index={aligned['model_export_sample_index'].nunique():,}"
    )
    return aligned


def assign_preference_elaboration_state(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = values[np.isfinite(values)].copy()
    if valid.empty:
        return out
    if valid.nunique(dropna=True) < 3:
        return out
    ranks = valid.rank(method="first", pct=True)
    out.loc[ranks.index[ranks <= 0.20]] = "Low"
    out.loc[ranks.index[(ranks > 0.20) & (ranks < 0.80)]] = "Medium"
    out.loc[ranks.index[ranks >= 0.80]] = "High"
    return out


def plot_state_histograms(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    out_path: Path,
    transform_y=None,
    bins: int | np.ndarray = 40,
    robust_range: tuple[float, float] | None = None,
    value_filter=None,
    state_col: str = "preference_elaboration_state",
    channels: list[str] | None = None,
    row_filter=None,
    xlim: tuple[float, float] | None = None,
    raw_label: str | None = None,
) -> None:
    if channels is None:
        channels = ["R", "S"]
    if row_filter is not None:
        df = df[row_filter(df)].copy()
    plot_df = df[[state_col, y_col, "channel"]].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df[state_col].isin(PLOTTED_STATES)].copy()
    plot_df = plot_df[plot_df["channel"].isin(channels)].copy()
    if value_filter is not None:
        keep = value_filter(pd.to_numeric(plot_df[y_col], errors="coerce").to_numpy(dtype=float))
        plot_df = plot_df[keep].copy()
    if transform_y is not None:
        plot_df[y_col] = transform_y(plot_df[y_col].to_numpy(dtype=float))

    if robust_range is not None:
        vals = pd.to_numeric(plot_df[y_col], errors="coerce")
        low = float(vals.quantile(robust_range[0]))
        high = float(vals.quantile(robust_range[1]))
        plot_df = plot_df[(plot_df[y_col] >= low) & (plot_df[y_col] <= high)].copy()
        hist_range = (low, high)
    else:
        hist_range = None

    if channels == ["S"]:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True, constrained_layout=True)
        axes = np.asarray(axes).reshape(1, 2)
        panels = [(0, state, "S") for state in PLOTTED_STATES]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True, constrained_layout=True)
        panels = [(row_i, state, channel) for row_i, state in enumerate(PLOTTED_STATES) for channel in channels]

    for panel_i, (row_i, state, channel) in enumerate(panels):
            col_i = panel_i if channels == ["S"] else channels.index(channel)
            ax = axes[row_i, col_i]
            panel_mask = (plot_df["channel"] == channel) & (plot_df[state_col] == state)
            vals = pd.to_numeric(plot_df.loc[panel_mask, y_col], errors="coerce").dropna()
            if vals.empty:
                ax.set_title(f"{state} - {'Recommendation' if channel == 'R' else 'Search'}")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.hist(
                vals,
                bins=bins,
                range=hist_range,
                density=True,
                alpha=0.72,
                color=STATE_COLORS[state],
                label=f"n={len(vals):,}",
            )
            if raw_label is not None:
                raw_vals = pd.to_numeric(
                    df.loc[
                        (df["channel"] == channel)
                        & (df[state_col] == state),
                        y_col,
                    ],
                    errors="coerce",
                )
                if value_filter is not None:
                    raw_keep = value_filter(raw_vals.to_numpy(dtype=float))
                    raw_vals = raw_vals[raw_keep]
                raw_vals = raw_vals.replace([np.inf, -np.inf], np.nan).dropna()
                mean_label = f"mean={raw_vals.mean():.1f}{raw_label}"
                median_label = f"median={raw_vals.median():.1f}{raw_label}"
                if transform_y is not None:
                    mean_x = float(transform_y(np.asarray([raw_vals.mean()], dtype=float))[0])
                    median_x = float(transform_y(np.asarray([raw_vals.median()], dtype=float))[0])
                else:
                    mean_x = float(raw_vals.mean())
                    median_x = float(raw_vals.median())
            else:
                mean_label = f"mean={vals.mean():.2f}"
                median_label = f"median={vals.median():.2f}"
                mean_x = float(vals.mean())
                median_x = float(vals.median())
            ax.axvline(mean_x, color="#111111", linewidth=2.0, label=mean_label)
            ax.axvline(
                median_x,
                color="#111111",
                linewidth=2.0,
                linestyle="--",
                label=median_label,
            )
            ax.set_title(f"{state} - {'Recommendation' if channel == 'R' else 'Search'}")
            ax.grid(axis="y", alpha=0.24)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            if channels == ["S"] or row_i == 1:
                ax.set_xlabel(y_label)
            if col_i == 0:
                ax.set_ylabel("Density")
            if xlim is not None:
                ax.set_xlim(*xlim)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_next_query_reformulation_probability(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float,
    time_window_minutes: float,
    outcome_col: str = "next_query_reformulation",
    outcome_label: str | None = None,
    row_filter=None,
    filter_label: str | None = None,
) -> None:
    if row_filter is not None:
        df = df[row_filter(df)].copy()
    plot_df = df[
        ["preference_elaboration_state", outcome_col, "channel"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["preference_elaboration_state"].isin(STATE_LABELS)].copy()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()

    summary = (
        plot_df.groupby(["channel", "preference_elaboration_state"], observed=True)
        .agg(
            probability=(outcome_col, "mean"),
            count=(outcome_col, "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(STATE_LABELS), dtype=float)
    width = 0.34
    offsets = {"R": -width / 2, "S": width / 2}
    channel_names = {"R": "Recommendation", "S": "Search"}

    for channel in ["R", "S"]:
        channel_summary = summary[summary["channel"] == channel].set_index("preference_elaboration_state")
        probs = [channel_summary.loc[state, "probability"] if state in channel_summary.index else np.nan for state in STATE_LABELS]
        counts = [channel_summary.loc[state, "count"] if state in channel_summary.index else 0 for state in STATE_LABELS]
        bars = ax.bar(
            x + offsets[channel],
            probs,
            width=width,
            color=CHANNEL_COLORS[channel],
            alpha=0.82,
            label=channel_names[channel],
        )
        for bar, prob, count in zip(bars, probs, counts):
            if not np.isfinite(prob):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{prob:.3f}\nn={int(count):,}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    title = "Preference Elaboration vs Future Query Reformulation"
    if filter_label:
        title = f"{title}\n{filter_label}"
    ax.set_title(title)
    ax.set_xlabel("Preference elaboration level")
    if outcome_label is None:
        time_label = f"within {time_window_minutes:g} min, " if time_window_minutes and time_window_minutes > 0 else ""
        outcome_label = f"next query reformulation {time_label}sim >= {threshold}"
    ax.set_ylabel(f"P({outcome_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS)
    ax.set_ylim(0, min(1.0, max(0.1, float(summary["probability"].max()) + 0.08)))
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_next_query_change_type_stacked(
    df: pd.DataFrame,
    out_path: Path,
    row_filter=None,
    filter_label: str | None = None,
) -> None:
    if row_filter is not None:
        df = df[row_filter(df)].copy()
    plot_df = df[
        ["preference_elaboration_state", "next_query_change_type", "channel"]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[plot_df["preference_elaboration_state"].isin(STATE_LABELS)].copy()
    plot_df = plot_df[plot_df["channel"].isin(["R", "S"])].copy()

    counts = (
        plot_df.groupby(
            ["channel", "preference_elaboration_state", "next_query_change_type"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        counts.groupby(["channel", "preference_elaboration_state"], observed=True)["count"]
        .sum()
        .rename("total")
        .reset_index()
    )
    summary = counts.merge(totals, on=["channel", "preference_elaboration_state"], how="left")
    summary["share"] = summary["count"] / summary["total"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True, constrained_layout=True)
    channel_names = {"R": "Recommendation", "S": "Search"}
    x = np.arange(len(STATE_LABELS), dtype=float)

    for ax, channel in zip(axes, ["R", "S"]):
        channel_summary = summary[summary["channel"] == channel]
        bottoms = np.zeros(len(STATE_LABELS), dtype=float)
        for change_type in CHANGE_TYPE_ORDER:
            typed = channel_summary[channel_summary["next_query_change_type"] == change_type]
            shares = (
                typed.set_index("preference_elaboration_state")["share"]
                .reindex(STATE_LABELS)
                .fillna(0)
                .to_numpy(dtype=float)
            )
            ax.bar(
                x,
                shares,
                bottom=bottoms,
                color=CHANGE_TYPE_COLORS[change_type],
                label=CHANGE_TYPE_LABELS[change_type],
                width=0.62,
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms += shares
        ax.set_title(channel_names[channel])
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_LABELS)
        ax.set_xlabel("Preference elaboration level")
        ax.set_ylim(0, min(1.0, max(0.1, float(bottoms.max()) * 1.18)))
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Share of next query reformulation type")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    title = "Preference Elaboration vs Future Query Reformulation"
    if filter_label:
        title = f"{title}\n{filter_label}"
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_all(metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_next_query_change_type_stacked(
        metrics,
        FIG_DIR / "preference_elaboration_future_query_reformulation.png",
        row_filter=lambda data: data["rec_history_length"].between(8, 9)
        & data["src_history_length"].between(1, 10),
        filter_label="History controlled: 8 <= R history <= 9, 1 <= S history <= 10",
    )
    plot_state_histograms(
        metrics,
        "future_dwell_time",
        "Preference Elaboration vs Future Dwell Time",
        f"Future time, log1p(seconds): R next {args.dwell_window} dwell; S session last-first",
        FIG_DIR / "preference_elaboration_dwell_time.png",
        transform_y=np.log1p,
        bins=45,
        value_filter=lambda values: values > 0,
        state_col="channel_specific_dwell_state",
        row_filter=lambda data: (data["channel"] == "R") | (data["rec_history_length"] >= 10),
        xlim=(0, 10),
        raw_label="s",
    )
    plot_state_histograms(
        metrics,
        "future_click_position_depth",
        "Preference Elaboration vs Future Click Position Depth",
        f"Mean next {args.position_window} same-channel positions - current position",
        FIG_DIR / "preference_elaboration_click_position_depth.png",
        bins=60,
        robust_range=(0.01, 0.99),
        state_col="belief_entropy_position_depth_state",
        channels=["S"],
    )
    plot_state_histograms(
        metrics,
        "future_query_click_count_average",
        "Preference Elaboration vs Future Query Click Count",
        f"Average clicked items per next {args.query_click_window} query sessions",
        FIG_DIR / "preference_elaboration_future_query_click_count_average.png",
        bins=40,
        state_col="belief_entropy_position_depth_state",
        channels=["S"],
        row_filter=lambda data: data["future_query_sessions_observed_for_click_count"] == args.query_click_window,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot behavioral outcomes against continuous preference elaboration."
    )
    parser.add_argument("--query-window", type=int, default=10)
    parser.add_argument("--query-click-window", type=int, default=5)
    parser.add_argument("--query-time-window-minutes", type=float, default=0.0)
    parser.add_argument("--position-window", type=int, default=5)
    parser.add_argument("--dwell-window", type=int, default=3)
    parser.add_argument("--reformulation-threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reuse-metrics", action="store_true", help="Read the cached metrics parquet if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.reuse_metrics and METRICS_PATH.exists():
        metrics = pd.read_parquet(METRICS_PATH)
        print(f"Loaded cached metrics: {METRICS_PATH} rows={len(metrics):,}")
    else:
        metrics = build_metrics(args)
        metrics.to_parquet(METRICS_PATH, index=False)
        print(f"Saved metrics: {METRICS_PATH} rows={len(metrics):,}")
    plot_all(metrics, args)


if __name__ == "__main__":
    main()
