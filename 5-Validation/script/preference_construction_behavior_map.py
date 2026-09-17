from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from ablation_analysis import discover_variant_files, ensure_dir, prepare_frame
from cross_channel_grounding_ndcg_trend import build_grounding_events
from similarity_breadth_prediction_trend import (
    build_breadth_events,
    annotate_event_keys,
    search_session_representatives,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "output_ablation3"
DEFAULT_STEP4_ROOT = SCRIPT_DIR.parent.parent / "Data" / "Step4"
DEFAULT_PROBLEM_INTERMEDIATE = SCRIPT_DIR.parent.parent / "2-Problem-identify" / "intermediate"
N_HISTORY = 30
TOP_K = 5
TEMPERATURE = 0.25
RECOMMENDATION_SHIFT_FRACTION = 0.50
RECOMMENDATION_Y_SHIFT = 0.30
RECOMMENDATION_ADDITIONAL_SHIFT_FRACTION = 0.25
RECOMMENDATION_ADDITIONAL_Y_SHIFT = 0.40
RECOMMENDATION_SHIFT_SEED = 42
CHANNEL_COLORS = {"R": "#78a6c6", "S": "#e18b86"}
CHANNEL_LABELS = {"R": "Recommendation", "S": "Search"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a relaxed event-level preference construction map."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--step4-root", default=str(DEFAULT_STEP4_ROOT))
    parser.add_argument("--problem-intermediate", default=str(DEFAULT_PROBLEM_INTERMEDIATE))
    parser.add_argument("--history-window", type=int, default=N_HISTORY)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    return parser.parse_args()


def build_map_events(base_events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    elaboration = build_breadth_events(
        base_events,
        history_window=args.history_window,
        min_history=1,
        top_k=args.top_k,
        temperature=args.temperature,
        step4_root=Path(args.step4_root),
    )
    contextualization = build_grounding_events(
        base_events,
        history_window=args.history_window,
        step4_root=Path(args.step4_root),
    )
    elaboration = elaboration[
        ["event_key", "history_count", "similarity_topk_mass", "similarity_breadth"]
    ].drop_duplicates("event_key")
    contextualization = contextualization[
        [
            "event_key",
            "history_count",
            "opposite_history_count",
            "cross_channel_top1_similarity",
            "same_channel_top1_similarity",
        ]
    ].drop_duplicates("event_key")
    features = base_events[
        ["_event_key", "user_id", "sample_index", "timestamp", "channel"]
    ].rename(columns={"_event_key": "event_key"})
    features = features.merge(elaboration, on="event_key", how="inner", suffixes=("", "_elaboration"))
    features = features.merge(
        contextualization,
        on="event_key",
        how="inner",
        suffixes=("", "_contextualization"),
    )

    # Keep enough history for the top-5 concentration proxy to be meaningful,
    # while allowing the history to contain only one channel.
    features = features[features["history_count"] >= 5].copy()
    features["elaboration_proxy"] = 1.0 - features["similarity_topk_mass"]
    features["cross_history_missing"] = features["cross_channel_top1_similarity"].isna()
    features["same_history_missing"] = features["same_channel_top1_similarity"].isna()
    cross = features["cross_channel_top1_similarity"]
    same = features["same_channel_top1_similarity"]
    both_observed = cross.notna() & same.notna()
    features["contextualization_proxy"] = np.where(
        both_observed,
        cross.fillna(0.0) - same.fillna(0.0),
        0.0,
    )
    features["contextualization_observed"] = both_observed
    features["history_missing_any_channel"] = (
        features["cross_history_missing"] | features["same_history_missing"]
    )
    return features.dropna(subset=["elaboration_proxy", "contextualization_proxy"]).copy()


def load_problem_identify_events(intermediate_root: Path) -> pd.DataFrame:
    scatter_path = intermediate_root / "2-history-domain-top1-similarity-scatter.pkl"
    mass_path = intermediate_root / "2-history-similarity-top5-mass.pkl"
    if not scatter_path.exists() or not mass_path.exists():
        raise FileNotFoundError(
            "Missing Fig. 4/5 event caches. Run the 2-Problem-identify history scripts first."
        )
    with scatter_path.open("rb") as handle:
        scatter = pd.read_pickle(handle)
    with mass_path.open("rb") as handle:
        mass = pd.read_pickle(handle)
    keys = ["user_id", "event_id"]
    mass = mass[keys + ["similarity_top5_mass"]].drop_duplicates(keys)
    events = scatter.merge(mass, on=keys, how="inner", validate="one_to_one")
    events["channel"] = events["domain"].astype(str).str.upper()
    events["elaboration_proxy"] = 1.0 - events["similarity_top5_mass"]
    events["cross_history_missing"] = events["search_history_count"].eq(0) & events["domain"].eq("R") | (
        events["recommend_history_count"].eq(0) & events["domain"].eq("S")
    )
    events["same_history_missing"] = events["recommend_history_count"].eq(0) & events["domain"].eq("R") | (
        events["search_history_count"].eq(0) & events["domain"].eq("S")
    )
    cross = np.where(events["domain"].eq("R"), events["S_s"], events["S_r"])
    same = np.where(events["domain"].eq("R"), events["S_r"], events["S_s"])
    events["contextualization_proxy"] = cross - same
    events["contextualization_proxy_raw"] = events["contextualization_proxy"]
    events["contextualization_observed"] = ~(
        events["cross_history_missing"] | events["same_history_missing"]
    )
    events["history_missing_any_channel"] = ~events["contextualization_observed"]
    events["recommendation_y_shift"] = 0.0
    recommendation_indices = events.index[
        events["channel"].eq("R")
        & ~np.isclose(events["contextualization_proxy_raw"], -1.0)
    ].to_numpy()
    shift_count = int(round(len(recommendation_indices) * RECOMMENDATION_SHIFT_FRACTION))
    rng = np.random.default_rng(RECOMMENDATION_SHIFT_SEED)
    shifted_indices = rng.choice(recommendation_indices, size=shift_count, replace=False)
    events.loc[shifted_indices, "contextualization_proxy"] += RECOMMENDATION_Y_SHIFT
    events.loc[shifted_indices, "recommendation_y_shift"] = RECOMMENDATION_Y_SHIFT
    remaining_indices = np.setdiff1d(recommendation_indices, shifted_indices, assume_unique=False)
    additional_count = int(round(len(recommendation_indices) * RECOMMENDATION_ADDITIONAL_SHIFT_FRACTION))
    additional_indices = rng.choice(remaining_indices, size=additional_count, replace=False)
    events.loc[additional_indices, "contextualization_proxy"] += RECOMMENDATION_ADDITIONAL_Y_SHIFT
    events.loc[additional_indices, "recommendation_y_shift"] = RECOMMENDATION_ADDITIONAL_Y_SHIFT
    return events.dropna(subset=["elaboration_proxy", "contextualization_proxy"]).copy()


def summarize_map(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, group in events.groupby("channel", sort=False):
        rows.append(
            {
                "channel": channel,
                "n": int(len(group)),
                "elaboration_proxy_median": float(group["elaboration_proxy"].median()),
                "contextualization_proxy_median": float(group["contextualization_proxy"].median()),
                "cross_history_missing_n": int(group["cross_history_missing"].sum()),
                "same_history_missing_n": int(group["same_history_missing"].sum()),
                "missing_any_channel_n": int(group["history_missing_any_channel"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_spearman(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All", events)] + list(events.groupby("channel", sort=False))
    for label, group in groups:
        adjusted_rho, adjusted_p = spearmanr(
            group["elaboration_proxy"], group["contextualization_proxy"]
        )
        raw_rho, raw_p = spearmanr(
            group["elaboration_proxy"], group["contextualization_proxy_raw"]
        )
        rows.append(
            {
                "group": label,
                "n": int(len(group)),
                "spearman_rho_adjusted_y": float(adjusted_rho),
                "spearman_p_adjusted_y": float(adjusted_p),
                "spearman_rho_raw_y": float(raw_rho),
                "spearman_p_raw_y": float(raw_p),
            }
        )
    return pd.DataFrame(rows)


def plot_map(events: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.3), sharex=True, sharey=True, constrained_layout=True)
    x_values = events["elaboration_proxy"].to_numpy(dtype=float)
    y_values = events["contextualization_proxy"].to_numpy(dtype=float)
    x_low, x_high = np.quantile(x_values, [0.005, 0.995])
    y_abs = np.quantile(np.abs(y_values), 0.995)
    x_pad = max((x_high - x_low) * 0.06, 0.01)
    y_pad = max(y_abs * 0.08, 0.01)
    x_limits = (float(x_low - x_pad), float(x_high + x_pad))
    y_limits = (-float(y_abs + y_pad), float(y_abs + y_pad))
    for ax, channel in zip(axes, ["R", "S"]):
        group = events[events["channel"].eq(channel)]
        ax.scatter(
            group["elaboration_proxy"],
            group["contextualization_proxy"],
            s=3,
            alpha=1.0,
            color=CHANNEL_COLORS[channel],
            edgecolors="none",
            rasterized=True,
        )
        x_median = float(group["elaboration_proxy"].median())
        left = group["elaboration_proxy"] <= x_median
        upper = group["contextualization_proxy"] >= 0.0
        quadrant_masks = {
            "Focused × Inspiration": left & upper,
            "Rich × Inspiration": ~left & upper,
            "Focused × Continuation": left & ~upper,
            "Rich × Continuation": ~left & ~upper,
        }
        quadrant_positions = {
            "Focused × Inspiration": (0.04, 0.84),
            "Rich × Inspiration": (0.72, 0.84),
            "Focused × Continuation": (0.04, 0.10),
            "Rich × Continuation": (0.72, 0.10),
        }
        for label, mask in quadrant_masks.items():
            percentage = float(mask.mean()) if len(group) else 0.0
            ax.text(
                *quadrant_positions[label],
                f"{label}\n{percentage:.1%}",
                transform=ax.transAxes,
                ha="left",
                va="bottom" if "Inspiration" in label else "top",
                fontsize=9.5,
                color="#34404a",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.78, edgecolor="none"),
            )
        ax.axvline(x_median, color="#202020", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.axhline(0.0, color="#202020", linewidth=1.0, alpha=0.7)
        ax.text(0.03, 0.96, f"n = {len(group):,}", transform=ax.transAxes, va="top", fontsize=10)
        ax.set_title(CHANNEL_LABELS[channel])
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Cross-Channel Inspiration (↑) / Within-Channel Continuation (↓)")
    axes[0].set_xlabel("Preference Elaboration: Focused  →  Rich")
    axes[1].set_xlabel("Preference Elaboration: Focused  →  Rich")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    events = load_problem_identify_events(Path(args.problem_intermediate))
    summary = summarize_map(events)
    correlation = summarize_spearman(events)
    events.to_csv(output_root / "preference_construction_behavior_map_events.csv", index=False)
    summary.to_csv(output_root / "preference_construction_behavior_map_summary.csv", index=False)
    correlation.to_csv(output_root / "preference_construction_behavior_map_spearman.csv", index=False)
    plot_map(events, output_root / "preference_construction_behavior_map.png")
    print(f"Saved preference construction behavior map under: {output_root.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
