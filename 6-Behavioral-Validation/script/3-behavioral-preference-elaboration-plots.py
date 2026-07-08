from __future__ import annotations

import argparse
import pickle
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
        page_time = pd.to_numeric(g["page_time"], errors="coerce").to_numpy(dtype=float)
        values = np.full(len(g), np.nan, dtype=float)
        finite = np.isfinite(page_time)
        total = float(np.sum(page_time[finite]))
        count = int(np.sum(finite))
        for i in range(len(g)):
            item_total = total - (float(page_time[i]) if finite[i] else 0.0)
            item_count = count - (1 if finite[i] else 0)
            if item_count > 0:
                values[i] = item_total / item_count
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


def future_query_reformulation(
    df: pd.DataFrame,
    sessions: pd.DataFrame,
    session_emb: np.ndarray,
    future_window: int,
    threshold: float,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "future_query_reformulation": np.full(len(df), np.nan, dtype=float),
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
        emb = session_emb[qs["_emb_row"].to_numpy(dtype=int)]
        pair_sim = np.sum(emb[:-1] * emb[1:], axis=1)

        n = len(qs)
        start_flags = np.full(n + 1, np.nan, dtype=float)
        start_lengths = np.full(n + 1, np.nan, dtype=float)
        start_max = np.full(n + 1, np.nan, dtype=float)
        start_pairs = np.zeros(n + 1, dtype=np.int16)
        for start in range(n):
            max_pair_end = min(n - 1, start + future_window - 1)
            if max_pair_end <= start:
                continue
            sims = pair_sim[start:max_pair_end]
            sims = sims[np.isfinite(sims)]
            if len(sims) == 0:
                continue
            start_pairs[start] = len(sims)
            start_max[start] = float(np.max(sims))
            start_flags[start] = float(np.max(sims) >= threshold)
            is_reformulation = sims >= threshold
            run_len = 0
            for value in is_reformulation:
                if not value:
                    break
                run_len += 1
            start_lengths[start] = float(run_len)

        event_ts = events["timestamp"].to_numpy(dtype=float)
        starts = np.searchsorted(q_ts, event_ts, side="right")
        valid = starts < n
        if not np.any(valid):
            continue
        idx = events.index.to_numpy()
        out.loc[idx[valid], "future_query_reformulation"] = start_flags[starts[valid]]
        out.loc[idx[valid], "future_query_reformulation_length"] = start_lengths[starts[valid]]
        out.loc[idx[valid], "future_query_max_adjacent_similarity"] = start_max[starts[valid]]
        out.loc[idx[valid], "future_query_pairs_observed"] = start_pairs[starts[valid]]

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
            vals = pd.to_numeric(
                plot_df.loc[
                    (plot_df["channel"] == channel)
                    & (plot_df[state_col] == state),
                    y_col,
                ],
                errors="coerce",
            ).dropna()
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
            ax.axvline(vals.mean(), color="#111111", linewidth=2.0, label=f"mean={vals.mean():.2f}")
            ax.set_title(f"{state} - {'Recommendation' if channel == 'R' else 'Search'}")
            ax.grid(axis="y", alpha=0.24)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            if channels == ["S"] or row_i == 1:
                ax.set_xlabel(y_label)
            if col_i == 0:
                ax.set_ylabel("Density")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_all(metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_state_histograms(
        metrics,
        "future_query_reformulation_length",
        "Preference Elaboration vs Future Query Reformulation",
        f"Future consecutive query reformulation length (sim >= {args.reformulation_threshold})",
        FIG_DIR / "preference_elaboration_future_query_reformulation.png",
        bins=np.arange(-0.5, args.query_window, 1.0),
        value_filter=lambda values: values > 0,
    )
    plot_state_histograms(
        metrics,
        "future_dwell_time",
        "Preference Elaboration vs Future Dwell Time",
        f"Future dwell time, log1p(page_time): R next {args.dwell_window}; S same session excluding current",
        FIG_DIR / "preference_elaboration_dwell_time.png",
        transform_y=np.log1p,
        bins=45,
        value_filter=lambda values: values > 0,
        state_col="belief_confidence_dwell_state",
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
    parser.add_argument("--position-window", type=int, default=5)
    parser.add_argument("--dwell-window", type=int, default=3)
    parser.add_argument("--reformulation-threshold", type=float, default=0.5)
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
