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
STATE_VARIANTS = ["full", "no_intent_state"]
N_HISTORY = 30
MIN_HISTORY = 10
TOP_K = 5
N_BINS = 8
SOFTMAX_TEMPERATURE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot prediction-score trends over semantic similarity breadth for "
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
        "--history-window",
        type=int,
        default=N_HISTORY,
        help="Number of previous events used to compute similarity breadth.",
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


def build_breadth_events(
    base_df: pd.DataFrame,
    history_window: int,
    min_history: int,
    top_k: int,
    temperature: float,
) -> pd.DataFrame:
    event_emb, embedding_source = event_embedding_matrix(base_df)
    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        original_index = g.index.to_numpy()
        g = g.reset_index(drop=True)
        emb = event_emb[original_index]
        for pos in range(len(g)):
            start = max(0, pos - history_window)
            hist = emb[start:pos]
            valid = np.linalg.norm(hist, axis=1) > 1e-12 if len(hist) else np.array([], dtype=bool)
            if int(valid.sum()) < min_history:
                continue
            if not np.any(emb[pos]):
                continue
            similarities = hist[valid] @ emb[pos]
            topk_mass = similarity_topk_mass(similarities, top_k=top_k, temperature=temperature)
            if not np.isfinite(topk_mass):
                continue
            cur = g.iloc[pos]
            rows.append(
                {
                    "user_id": int(user_id),
                    "sample_index": int(cur["sample_index"]),
                    "timestamp": float(cur.get("timestamp", np.nan)),
                    "channel": str(cur.get("channel", "")).upper(),
                    "history_count": int(valid.sum()),
                    "similarity_topk_mass": topk_mass,
                    "similarity_breadth": 1.0 - topk_mass,
                    "history_similarity_mean": float(np.mean(similarities)),
                    "history_similarity_std": (
                        float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0
                    ),
                    "history_similarity_min": float(np.min(similarities)),
                    "history_similarity_max": float(np.max(similarities)),
                    "embedding_source": embedding_source,
                }
            )
    return pd.DataFrame(rows)


def prediction_score(row: pd.Series) -> float:
    channel = str(row.get("channel", "")).upper()
    if channel == "R":
        return float(row.get("rec_pred_pos_score", np.nan))
    if channel == "S":
        return float(row.get("src_pred_pos_score", np.nan))
    return float("nan")


def evaluate_predictions(events: pd.DataFrame, df: pd.DataFrame, variant: str) -> pd.DataFrame:
    idx = {
        (int(row["user_id"]), int(row["sample_index"])): row
        for _, row in df.iterrows()
    }
    rows = []
    for event in events.itertuples(index=False):
        row = idx.get((int(event.user_id), int(event.sample_index)))
        if row is None:
            continue
        score = prediction_score(row)
        if not np.isfinite(score):
            continue
        rows.append(
            {
                "variant": variant,
                "user_id": int(event.user_id),
                "sample_index": int(event.sample_index),
                "timestamp": float(event.timestamp),
                "channel": str(event.channel),
                "history_count": int(event.history_count),
                "similarity_breadth": float(event.similarity_breadth),
                "similarity_topk_mass": float(event.similarity_topk_mass),
                "prediction_score": score,
            }
        )
    return pd.DataFrame(rows)


def attach_breadth_bins(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df.copy()
    event_bins = (
        df[df["variant"] == "full"][["user_id", "sample_index", "similarity_breadth"]]
        .drop_duplicates(["user_id", "sample_index"])
        .dropna(subset=["similarity_breadth"])
    )
    q = min(int(bins), int(event_bins["similarity_breadth"].nunique()))
    if q < 2:
        raise ValueError("Not enough unique similarity breadth values to build trend bins.")
    event_bins["breadth_bin"] = pd.qcut(
        event_bins["similarity_breadth"],
        q=q,
        labels=False,
        duplicates="drop",
    )
    event_bins["breadth_bin"] = event_bins["breadth_bin"].astype(int) + 1
    return df.merge(
        event_bins[["user_id", "sample_index", "breadth_bin"]],
        on=["user_id", "sample_index"],
        how="inner",
    )


def summarize_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, breadth_bin), g in df.groupby(["variant", "breadth_bin"], sort=True):
        rows.append(
            {
                "variant": variant,
                "breadth_bin": int(breadth_bin),
                "n": int(len(g)),
                "similarity_breadth_mean": float(g["similarity_breadth"].mean()),
                "similarity_breadth_sem": sem(g["similarity_breadth"]),
                "prediction_score_mean": float(g["prediction_score"].mean()),
                "prediction_score_sem": sem(g["prediction_score"]),
            }
        )
    out = pd.DataFrame(rows)
    out["_variant_order"] = out["variant"].map({variant: idx for idx, variant in enumerate(STATE_VARIANTS)})
    out = out.sort_values(["_variant_order", "breadth_bin"]).drop(columns="_variant_order").reset_index(drop=True)
    return out


def summarize_difference(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index="breadth_bin",
        columns="variant",
        values=["similarity_breadth_mean", "prediction_score_mean", "n"],
    )
    rows = []
    for breadth_bin in pivot.index:
        full_score = pivot.loc[breadth_bin, ("prediction_score_mean", "full")]
        ablated_score = pivot.loc[breadth_bin, ("prediction_score_mean", "no_intent_state")]
        rows.append(
            {
                "breadth_bin": int(breadth_bin),
                "similarity_breadth_mean": float(pivot.loc[breadth_bin, ("similarity_breadth_mean", "full")]),
                "full_prediction_score_mean": float(full_score),
                "no_intent_state_prediction_score_mean": float(ablated_score),
                "full_minus_no_intent_state": float(full_score - ablated_score),
                "full_n": int(pivot.loc[breadth_bin, ("n", "full")]),
                "no_intent_state_n": int(pivot.loc[breadth_bin, ("n", "no_intent_state")]),
            }
        )
    return pd.DataFrame(rows)


def plot_trend(summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0), constrained_layout=True)
    for variant in STATE_VARIANTS:
        g = summary[summary["variant"] == variant].sort_values("similarity_breadth_mean")
        if g.empty:
            continue
        x = g["similarity_breadth_mean"].to_numpy(dtype=float)
        y = g["prediction_score_mean"].to_numpy(dtype=float)
        yerr = 1.96 * g["prediction_score_sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=VARIANT_COLORS[variant],
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel("Similarity breadth")
    ax.set_ylabel("Prediction score")
    ax.set_title("Prediction Score over Similarity Breadth", loc="center", fontsize=12, weight="normal")
    ax.grid(axis="both", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_history <= args.top_k:
        raise ValueError("--min-history must be greater than --top-k for breadth to be identifiable.")
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
    base_df = frames["full"]
    for variant, df in frames.items():
        validate_alignment(base_df, df, variant)

    breadth_events = build_breadth_events(
        base_df,
        history_window=args.history_window,
        min_history=args.min_history,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    if breadth_events.empty:
        raise ValueError("No valid similarity-breadth events were built.")

    evaluated = pd.concat(
        [evaluate_predictions(breadth_events, frames[variant], variant) for variant in STATE_VARIANTS],
        ignore_index=True,
    )
    evaluated = attach_breadth_bins(evaluated, bins=args.bins)
    trend_summary = summarize_trend(evaluated)
    difference_summary = summarize_difference(trend_summary)

    breadth_events.to_csv(output_root / "similarity_breadth_events.csv", index=False)
    evaluated.to_csv(output_root / "similarity_breadth_prediction_events.csv", index=False)
    trend_summary.to_csv(output_root / "similarity_breadth_prediction_summary.csv", index=False)
    difference_summary.to_csv(output_root / "similarity_breadth_prediction_difference.csv", index=False)
    plot_trend(trend_summary, output_root / "similarity_breadth_prediction_trend.png")

    print(f"Saved similarity-breadth prediction trend under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
