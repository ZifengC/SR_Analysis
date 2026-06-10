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
DEFAULT_INPUT = SCRIPT_DIR.parent / "pcsar_intent_features_test_full.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "5"
STATE_LABELS = ["Low", "Medium", "High"]
STATE_COLORS = {
    "Low": "#2f6f4e",
    "Medium": "#c7862f",
    "High": "#a33a2f",
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
        help="Directory for state-validation outputs.",
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
        "history_rec_share",
        "history_src_share",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
    ]
    numeric_cols.extend(pi_columns(df))
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
            future_distances = [js_distance(cur, future_center) for cur in future]
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
                    "future_mean_consistency": float(1.0 - np.nanmean(current_future_distances)),
                    "future_nearest_continuation": float(1.0 - np.nanmin(current_future_distances)),
                    "future_dispersion": float(np.nanmean(future_distances)),
                    "current_to_future_js": float(current_to_future),
                    "current_to_future_mean_js": float(np.nanmean(current_future_distances)),
                    "current_to_future_min_js": float(np.nanmin(current_future_distances)),
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
                "future_mean_consistency_mean": float(np.nanmean(g["future_mean_consistency"])),
                "future_mean_consistency_sem": sem(g["future_mean_consistency"]),
                "future_nearest_continuation_mean": float(np.nanmean(g["future_nearest_continuation"])),
                "future_nearest_continuation_sem": sem(g["future_nearest_continuation"]),
                "future_dispersion_mean": float(np.nanmean(g["future_dispersion"])),
                "future_dispersion_sem": sem(g["future_dispersion"]),
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
                "high_minus_low_future_mean_consistency": float(
                    high["future_mean_consistency_mean"] - low["future_mean_consistency_mean"]
                ),
                "high_minus_low_future_nearest_continuation": float(
                    high["future_nearest_continuation_mean"] - low["future_nearest_continuation_mean"]
                ),
                "high_minus_low_future_dispersion": float(
                    high["future_dispersion_mean"] - low["future_dispersion_mean"]
                ),
                "high_minus_low_history_src_share": float(
                    high["history_src_share_mean"] - low["history_src_share_mean"]
                ),
                "low_n": int(low["n"]),
                "high_n": int(high["n"]),
            }
        )
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

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), constrained_layout=True)
    panels = [
        ("future_consistency_mean", "future_consistency_sem", "Future Consistency"),
        (
            "future_nearest_continuation_mean",
            "future_nearest_continuation_sem",
            "Future Nearest Continuation",
        ),
        ("future_dispersion_mean", "future_dispersion_sem", "Future Dispersion"),
    ]
    for ax, (mean_col, sem_col, title) in zip(axes, panels):
        means = plot_df[mean_col].to_numpy(dtype=float)
        sems = plot_df[sem_col].to_numpy(dtype=float)
        ax.bar(x, means, color=colors, alpha=0.88)
        ax.errorbar(x, means, yerr=1.96 * sems, fmt="none", ecolor="#333333", capsize=4)
        ax.set_xticks(x, plot_df["state"].tolist())
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel("Inferred Uncertainty State")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("1 - JS(current intent, future center)")
    axes[1].set_ylabel("Mean JS(future intents, future center)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = prepare_frame(pd.read_csv(input_path, low_memory=False))
    events = build_state_events(df, future_window=args.future_window)
    if events.empty:
        raise ValueError("No state-validation events could be built. Check user sequences and global_pi_* columns.")

    all_summaries = []
    enriched = events.copy()
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

    enriched.to_csv(output_dir / "state_validation_events.csv", index=False)
    summary.to_csv(output_dir / "state_validation_summary.csv", index=False)
    high_low.to_csv(output_dir / "state_validation_high_low_differences.csv", index=False)
    plot_uncertainty_summary(summary, output_dir / "state_validation_uncertainty.png")

    print(f"Saved state validation outputs under: {output_dir.resolve()}")
    print(summary.to_string(index=False))
    if not high_low.empty:
        print("\nHigh - Low differences:")
        print(high_low.to_string(index=False))


if __name__ == "__main__":
    main()
