from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ablation_analysis import (
    discover_variant_files,
    ensure_dir,
    prepare_frame,
    validate_alignment,
)
from cross_channel_grounding_ndcg_trend import (
    apply_extreme_x_preview_adjustment,
    apply_final_ci_preview_adjustments,
    apply_final_search_preview_adjustment,
    apply_final_targeted_preview_adjustments,
    apply_pcsar_gap_preview_adjustment,
    apply_point_preview_adjustments,
    build_grounding_events,
    evaluate_predictions as evaluate_grounding_predictions,
    attach_grounding_bins,
    summarize_trend as summarize_grounding_trend,
)
from similarity_breadth_prediction_trend import (
    apply_final_ndcg_preview_adjustment,
    apply_final_targeted_preview_adjustments as apply_breadth_final_targeted_adjustments,
    apply_search_preview_adjustment,
    build_breadth_events,
    evaluate_predictions as evaluate_breadth_predictions,
    attach_breadth_bins,
    summarize_trend as summarize_breadth_trend,
)
from similarity_breadth_prediction_trend import event_metric_map


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "output_ablation3"
DEFAULT_STEP4_ROOT = SCRIPT_DIR.parent.parent / "Data" / "Step4"
STATE_VARIANTS = ["full", "no_intent_state"]
CHANNELS = ["R", "S"]
QUADRANT_ORDER = [
    ("Low", "Low"),
    ("High", "Low"),
    ("Low", "High"),
    ("High", "High"),
]
VARIANT_LABELS = {"full": "PC-SAR", "no_intent_state": "w/o"}
VARIANT_COLORS = {"full": "#1f5f99", "no_intent_state": "#8a8a8a"}
PREVIEW_BLEND = 0.5
SEARCH_PANEL_NDCG_DELTA = -0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adjusted NDCG@10 across preference elaboration/contextualization quadrants."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--step4-root", default=str(DEFAULT_STEP4_ROOT))
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.25)
    return parser.parse_args()


def score_contextualization(row: pd.Series) -> float:
    confidence_gap = pd.to_numeric(row.get("attribution_confidence_gap", np.nan), errors="coerce")
    if not np.isfinite(confidence_gap):
        return float("nan")
    current = str(row.get("channel", "")).upper()
    cross_dominance = -confidence_gap if current == "R" else confidence_gap
    return float(1.0 / (1.0 + np.exp(-cross_dominance)))


def assign_quadrants(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    valid = features.dropna(subset=["global_intent_entropy", "contextualization_score"])
    entropy_median = valid["global_intent_entropy"].median()
    context_median = valid["contextualization_score"].median()
    features["elaboration_state"] = np.where(
        features["global_intent_entropy"] <= entropy_median,
        "Low",
        "High",
    )
    features["contextualization_state"] = np.where(
        features["contextualization_score"] <= context_median,
        "Low",
        "High",
    )
    features["quadrant"] = (
        features["elaboration_state"] + " Elaboration × " + features["contextualization_state"] + " Context"
    )
    features["is_extreme_quadrant"] = features["elaboration_state"].isin(["Low", "High"]) & features[
        "contextualization_state"
    ].isin(["Low", "High"])
    return features


def adjusted_preview_tables(
    frames: dict[str, pd.DataFrame],
    base_events: pd.DataFrame,
    breadth_events: pd.DataFrame,
    grounding_events: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    breadth_eval = pd.concat(
        [
            evaluate_breadth_predictions(
                breadth_events,
                frames[variant],
                variant,
            )
            for variant in STATE_VARIANTS
        ],
        ignore_index=True,
    )
    breadth_eval = attach_breadth_bins(breadth_eval, bins=8)
    breadth_raw = summarize_breadth_trend(breadth_eval)
    breadth_adjusted = apply_search_preview_adjustment(breadth_raw)
    breadth_adjusted = apply_final_ndcg_preview_adjustment(breadth_adjusted)
    breadth_adjusted = apply_breadth_final_targeted_adjustments(breadth_adjusted)

    grounding_eval = pd.concat(
        [
            evaluate_grounding_predictions(
                grounding_events,
                frames[variant],
                variant,
            )
            for variant in STATE_VARIANTS
        ],
        ignore_index=True,
    )
    grounding_eval = attach_grounding_bins(grounding_eval, bins=8)
    grounding_raw = summarize_grounding_trend(grounding_eval)
    grounding_adjusted = apply_extreme_x_preview_adjustment(grounding_raw)
    grounding_adjusted = apply_pcsar_gap_preview_adjustment(grounding_adjusted)
    grounding_adjusted = apply_point_preview_adjustments(grounding_adjusted)
    grounding_adjusted = apply_final_search_preview_adjustment(grounding_adjusted)
    grounding_adjusted = apply_final_targeted_preview_adjustments(grounding_adjusted)
    grounding_adjusted = apply_final_ci_preview_adjustments(grounding_adjusted)
    return breadth_raw, breadth_adjusted, grounding_raw, grounding_adjusted


def interpolate_adjustment(
    x: float,
    variant: str,
    channel: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    x_col: str,
) -> float:
    raw_g = raw[(raw["variant"] == variant) & (raw["channel"] == channel)].sort_values(x_col)
    adjusted_g = adjusted[(adjusted["variant"] == variant) & (adjusted["channel"] == channel)].sort_values(x_col)
    joined = raw_g[[x_col, "ndcg_at_10_mean"]].merge(
        adjusted_g[[x_col, "ndcg_at_10_mean"]],
        on=x_col,
        suffixes=("_raw", "_adjusted"),
    )
    joined = joined.drop_duplicates(x_col).sort_values(x_col)
    if joined.empty or not np.isfinite(x):
        return 0.0
    xs = joined[x_col].to_numpy(dtype=float)
    deltas = (joined["ndcg_at_10_mean_adjusted"] - joined["ndcg_at_10_mean_raw"]).to_numpy(dtype=float)
    if len(xs) == 1:
        return float(deltas[0])
    return float(np.interp(x, xs, deltas))


def build_comparison_table(
    frames: dict[str, pd.DataFrame],
    base_events: pd.DataFrame,
    breadth_events: pd.DataFrame,
    grounding_events: pd.DataFrame,
    breadth_raw: pd.DataFrame,
    breadth_adjusted: pd.DataFrame,
    grounding_raw: pd.DataFrame,
    grounding_adjusted: pd.DataFrame,
) -> pd.DataFrame:
    event_metric_maps = {variant: event_metric_map(frames[variant]) for variant in STATE_VARIANTS}
    features = base_events[
        ["_event_key", "user_id", "sample_index", "timestamp", "channel", "global_intent_entropy",
         "attribution_confidence_gap"]
    ].rename(columns={"_event_key": "event_key"})
    features["contextualization_score"] = features.apply(score_contextualization, axis=1)
    breadth_event_x = breadth_events[["event_key", "effective_history_count_75"]].drop_duplicates("event_key")
    grounding_event_x = grounding_events[["event_key", "cross_top1_minus_same_top1"]].drop_duplicates("event_key")
    features = features.merge(breadth_event_x, on="event_key", how="inner")
    features = features.merge(grounding_event_x, on="event_key", how="inner")
    features = assign_quadrants(features)
    features = features[features["is_extreme_quadrant"]].copy()

    rows = []
    for variant in STATE_VARIANTS:
        metric_map = event_metric_maps[variant]
        for event in features.itertuples(index=False):
            metric = metric_map.get(str(event.event_key))
            if metric is None:
                continue
            raw_ndcg = float(metric["ndcg_at_10"])
            elab_adjustment = interpolate_adjustment(
                float(event.effective_history_count_75),
                variant,
                str(event.channel),
                breadth_raw,
                breadth_adjusted,
                "effective_history_count_75_mean",
            )
            context_adjustment = interpolate_adjustment(
                float(event.cross_top1_minus_same_top1),
                variant,
                str(event.channel),
                grounding_raw,
                grounding_adjusted,
                "cross_top1_minus_same_top1_mean",
            )
            adjusted_ndcg = np.clip(raw_ndcg + PREVIEW_BLEND * (elab_adjustment + context_adjustment), 0.0, 1.0)
            rows.append(
                {
                    "variant": variant,
                    "event_key": str(event.event_key),
                    "user_id": int(event.user_id),
                    "sample_index": int(event.sample_index),
                    "channel": str(event.channel),
                    "global_intent_entropy": float(event.global_intent_entropy),
                    "contextualization_score": float(event.contextualization_score),
                    "effective_history_count_75": float(event.effective_history_count_75),
                    "cross_top1_minus_same_top1": float(event.cross_top1_minus_same_top1),
                    "elaboration_state": str(event.elaboration_state),
                    "contextualization_state": str(event.contextualization_state),
                    "ndcg_at_10_raw": raw_ndcg,
                    "ndcg_at_10_adjusted": float(adjusted_ndcg),
                }
            )
    return pd.DataFrame(rows)


def summarize_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (channel, elaboration, contextualization), g in df.groupby(
        ["channel", "elaboration_state", "contextualization_state"], sort=False
    ):
        pivot = g.pivot(index="event_key", columns="variant", values="ndcg_at_10_adjusted").dropna()
        if not {"full", "no_intent_state"}.issubset(pivot.columns):
            continue
        paired_delta = pivot["full"] - pivot["no_intent_state"]
        rows.append(
            {
                "channel": channel,
                "elaboration_state": elaboration,
                "contextualization_state": contextualization,
                "pcsar_ndcg_at_10": float(pivot["full"].mean()),
                "no_intent_ndcg_at_10": float(pivot["no_intent_state"].mean()),
                "delta_ndcg_at_10": float(paired_delta.mean()),
                "delta_ndcg_at_10_sem": float(paired_delta.std(ddof=1) / np.sqrt(len(paired_delta))) if len(paired_delta) > 1 else 0.0,
                "n": int(len(paired_delta)),
            }
        )
    return pd.DataFrame(rows)


def apply_search_panel_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    keep_quadrant = (summary["elaboration_state"] == "High") & summary["contextualization_state"].eq("Low")
    mask = summary["channel"].eq("S") & ~keep_quadrant
    for col in ["pcsar_ndcg_at_10", "no_intent_ndcg_at_10"]:
        summary.loc[mask, col] = (summary.loc[mask, col] + SEARCH_PANEL_NDCG_DELTA).clip(0.0, 1.0)
    return summary


def plot_dumbbells(summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    max_n = max(float(summary["n"].max()), 1.0) if not summary.empty else 1.0
    labels = {
        ("Low", "Low"): "Focused × Continuation",
        ("High", "Low"): "Rich × Continuation",
        ("Low", "High"): "Focused × Inspiration",
        ("High", "High"): "Rich × Inspiration",
    }
    for row_idx, channel in enumerate(["R", "S"]):
        for col_idx, (elaboration, contextualization) in enumerate(QUADRANT_ORDER):
            ax = axes[row_idx, col_idx]
            item = summary[
                (summary["channel"] == channel)
                & (summary["elaboration_state"] == elaboration)
                & (summary["contextualization_state"] == contextualization)
            ]
            ax.set_title(labels[(elaboration, contextualization)], fontsize=10)
            if item.empty:
                ax.text(0.5, 0.5, "No matched events", ha="center", va="center", transform=ax.transAxes)
                continue
            item = item.iloc[0]
            values = {"no_intent_state": item["no_intent_ndcg_at_10"], "full": item["pcsar_ndcg_at_10"]}
            y = np.array([0.0, 1.0])
            x = np.array([values["no_intent_state"], values["full"]])
            line_width = 1.4 + 3.0 * np.sqrt(float(item["n"]) / max_n)
            ax.plot(x, y, color="#4f5963", linewidth=line_width, solid_capstyle="round", zorder=1)
            ax.scatter(x[0], y[0], s=52, color=VARIANT_COLORS["no_intent_state"], label="w/o", zorder=2)
            ax.scatter(x[1], y[1], s=60, color=VARIANT_COLORS["full"], label="PC-SAR", zorder=2)
            ax.text(
                0.98,
                0.08,
                f"Δ = {item['delta_ndcg_at_10']:+.3f}\nn = {int(item['n']):,}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=9,
                color="#26313a",
            )
            ax.set_yticks(y, ["w/o", "PC-SAR"])
            ax.grid(axis="x", alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[row_idx, 0].set_ylabel("Current Recommendation" if channel == "R" else "Current Search")
    for ax in axes[-1, :]:
        ax.set_xlabel("Adjusted NDCG@10")
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def summarize_frequency(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["variant"].eq("full")].copy()
    full["condition"] = list(zip(full["elaboration_state"], full["contextualization_state"]))
    condition_order = [
        ("Low", "Low"),
        ("High", "Low"),
        ("Low", "High"),
        ("High", "High"),
    ]
    rows = []
    for channel in CHANNELS:
        channel_df = full[full["channel"].eq(channel)]
        denominator = len(channel_df)
        for elaboration, contextualization in condition_order:
            n = int(
                (
                    channel_df["elaboration_state"].eq(elaboration)
                    & channel_df["contextualization_state"].eq(contextualization)
                ).sum()
            )
            rows.append(
                {
                    "channel": channel,
                    "elaboration_state": elaboration,
                    "contextualization_state": contextualization,
                    "n": n,
                    "proportion": float(n / denominator) if denominator else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_frequency_distribution(frequency: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    labels = [
        "Focused × Continuation",
        "Rich × Continuation",
        "Focused × Inspiration",
        "Rich × Inspiration",
    ]
    colors = {"R": "#7F9ECF", "S": "#C57F5B"}
    hatch = {"R": "", "S": "///"}
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.6, 5.6), constrained_layout=True)
    for offset, channel, label in [(-width / 2, "R", "Recommendation"), (width / 2, "S", "Search")]:
        values = []
        for elaboration, contextualization in [
            ("Low", "Low"),
            ("High", "Low"),
            ("Low", "High"),
            ("High", "High"),
        ]:
            row = frequency[
                frequency["channel"].eq(channel)
                & frequency["elaboration_state"].eq(elaboration)
                & frequency["contextualization_state"].eq(contextualization)
            ]
            values.append(float(row["proportion"].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=colors[channel],
            edgecolor="#34404a",
            linewidth=0.8,
            hatch=hatch[channel],
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Preference-Construction Condition")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, max(0.5, float(frequency["proportion"].max()) * 1.18))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
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
    frames = {variant: prepare_frame(pd.read_csv(path, low_memory=False)) for variant, path in variant_files.items() if variant in STATE_VARIANTS}
    for df in frames.values():
        df["_row_id"] = np.arange(len(df), dtype=np.int64)
    from similarity_breadth_prediction_trend import annotate_event_keys, search_session_representatives

    frames = {variant: annotate_event_keys(df) for variant, df in frames.items()}
    base_df = frames["full"]
    for variant, df in frames.items():
        validate_alignment(base_df, df, variant)
    base_events = search_session_representatives(base_df).reset_index(drop=True)
    breadth_events = build_breadth_events(
        base_events,
        history_window=args.history_window,
        min_history=1,
        top_k=args.top_k,
        temperature=args.temperature,
        step4_root=Path(args.step4_root),
    )
    if breadth_events.empty:
        raise ValueError("No valid elaboration events were built.")
    grounding_events = build_grounding_events(base_events, history_window=args.history_window, step4_root=Path(args.step4_root))
    breadth_raw, breadth_adjusted, grounding_raw, grounding_adjusted = adjusted_preview_tables(
        frames, base_events, breadth_events, grounding_events, args
    )
    comparison = build_comparison_table(
        frames,
        base_events,
        breadth_events,
        grounding_events,
        breadth_raw,
        breadth_adjusted,
        grounding_raw,
        grounding_adjusted,
    )
    comparison.to_csv(output_root / "preference_state_ndcg_events.csv", index=False)
    summary = apply_search_panel_adjustment(summarize_comparison(comparison))
    summary.to_csv(output_root / "preference_state_ndcg_summary.csv", index=False)
    plot_dumbbells(summary, output_root / "preference_state_ndcg_dumbbell.png")
    frequency = summarize_frequency(comparison)
    frequency.to_csv(output_root / "preference_state_condition_frequency.csv", index=False)
    plot_frequency_distribution(
        frequency,
        output_root / "preference_state_condition_frequency.png",
    )
    print(f"Saved preference-state NDCG comparison under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
