from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt


TRANSITION_ORDER = ["R->R", "R->S", "S->R", "S->S"]
TRANSITION_COLORS = {
    "R->R": "#1f77b4",
    "R->S": "#ff7f0e",
    "S->R": "#2ca02c",
    "S->S": "#d62728",
}
SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
STACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MIN_RUN_LENGTH = 2
MIN_BIN_SAMPLES = 5
MAX_RUN_LENGTH_BIN = 10
TRANSITION_SCORE_BINS = 10
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "pcsar_intent_features_test_full.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate intent-pattern analyses from intermediate CSV exports."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to the exported intermediate CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for figures and tables.",
    )
    parser.add_argument(
        "--exploration-column",
        type=str,
        default="history_src_share",
        help="Column used as the exploration score proxy.",
    )
    parser.add_argument(
        "--user-key",
        type=str,
        default="user_id",
        help="Column used to build per-user sequences.",
    )
    parser.add_argument(
        "--run-jsd-threshold",
        type=float,
        default=0.50,
        help="JSD threshold used to extend a stable run.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required columns from candidates: {list(candidates)}")
    return None


def normalize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    min_v = np.nanmin(values[finite])
    max_v = np.nanmax(values[finite])
    if np.isclose(max_v, min_v):
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def safe_entropy(pi: np.ndarray) -> float:
    pi = np.asarray(pi, dtype=np.float64)
    pi = np.clip(pi, 1e-12, 1.0)
    pi = pi / np.sum(pi)
    k = max(pi.shape[-1], 1)
    denom = np.log(k) if k > 1 else 1.0
    return float(-(pi * np.log(pi)).sum() / denom)


def js_distance(pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    a = np.asarray(pi_a, dtype=np.float64)
    b = np.asarray(pi_b, dtype=np.float64)
    a = np.clip(a, 1e-12, 1.0)
    b = np.clip(b, 1e-12, 1.0)
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    kl_am = np.sum(a * (np.log(a) - np.log(m)))
    kl_bm = np.sum(b * (np.log(b) - np.log(m)))
    return float(np.sqrt(max(0.5 * (kl_am + kl_bm), 0.0)))


def sem(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values).dropna()
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def pi_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    def suffix_value(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(cols, key=suffix_value)


def prepare_frame(df: pd.DataFrame, exploration_column: str, user_key: str) -> pd.DataFrame:
    if exploration_column not in df.columns:
        fallback = resolve_column(
            df,
            ["history_src_share", "history_rec_share"],
            required=False,
        )
        if fallback is None:
            raise KeyError(
                f"Missing exploration column '{exploration_column}' and no fallback column found."
            )
        exploration_column = fallback

    df = df.copy()
    channel_col = resolve_column(df, ["channel", "domain"], required=False)
    if channel_col is None:
        df["channel"] = "R"
        channel_col = "channel"
    if channel_col != "channel":
        df["channel"] = df[channel_col]

    numeric_cols = [
        "timestamp",
        "sample_index",
        "global_dominant_intent",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
        "global_belief_uncertainty_mean",
        "belief_uncertainty_mean",
        "rec_src_intent_shift_js",
        "attribution_confidence_gap",
    ]
    df = coerce_numeric(df, numeric_cols)
    df["exploration_score"] = pd.to_numeric(df[exploration_column], errors="coerce")
    belief_col = resolve_column(
        df,
        [
            "global_posterior_uncertainty",
            "global_belief_uncertainty_mean",
            "belief_uncertainty_mean",
        ],
    )
    df["belief_uncertainty"] = pd.to_numeric(df[belief_col], errors="coerce")

    channel_order = df["channel"].map({"R": 0, "S": 1}).fillna(99).astype(int)
    sort_cols = [user_key]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    df["_channel_order"] = channel_order
    sort_cols.append("_channel_order")
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=["_channel_order"])
    return df


def contiguous_runs(values: np.ndarray) -> list[int]:
    values = np.asarray(values)
    if len(values) == 0:
        return []
    runs: list[int] = []
    run_len = 1
    for prev, cur in zip(values[:-1], values[1:]):
        if prev == cur:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return runs


def mean_consecutive_jsd(pi_matrix: np.ndarray) -> float:
    if len(pi_matrix) < 2:
        return 0.0
    vals = [js_distance(a, b) for a, b in zip(pi_matrix[:-1], pi_matrix[1:])]
    return float(np.mean(vals)) if vals else 0.0


def mean_abs_diff(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def consecutive_jsd(pi_matrix: np.ndarray) -> np.ndarray:
    if len(pi_matrix) < 2:
        return np.array([], dtype=float)
    return np.array([js_distance(a, b) for a, b in zip(pi_matrix[:-1], pi_matrix[1:])], dtype=float)


def build_user_summary(df: pd.DataFrame, user_key: str) -> pd.DataFrame:
    pi_cols = pi_columns(df, "global_pi_")
    if not pi_cols:
        raise KeyError("Could not find global_pi_* columns in the input CSV.")

    rows = []
    for user_id, g in df.groupby(user_key, sort=False):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue

        dom = pd.to_numeric(g["global_dominant_intent"], errors="coerce").fillna(-1).to_numpy()
        intent_entropy = pd.to_numeric(g["global_intent_entropy"], errors="coerce").to_numpy()
        dominant_prob = pd.to_numeric(g["global_dominant_intent_prob"], errors="coerce").to_numpy()
        uncertainty = pd.to_numeric(g["belief_uncertainty"], errors="coerce").to_numpy()
        attr_gap = pd.to_numeric(g["attribution_confidence_gap"], errors="coerce").to_numpy()
        pi = g[pi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        runs = contiguous_runs(dom)
        run_length = int(max(runs) if runs else 0)
        rows.append(
            {
                user_key: user_id,
                "sequence_length": int(len(g)),
                "run_length": run_length,
                "max_run_length": run_length,
                "switch_count": int(max(len(runs) - 1, 0)),
                "mean_intent_entropy": float(np.nanmean(intent_entropy)),
                "mean_dominant_intent_prob": float(np.nanmean(dominant_prob)),
                "posterior_dispersion": mean_consecutive_jsd(pi),
                "mean_uncertainty": float(np.nanmean(uncertainty)),
                "attribution_dispersion": mean_abs_diff(attr_gap),
            }
        )

    return pd.DataFrame(rows)


def build_stable_runs(df: pd.DataFrame, user_key: str, run_jsd_threshold: float) -> pd.DataFrame:
    pi_cols = pi_columns(df, "global_pi_")
    if not pi_cols:
        raise KeyError("Could not find global_pi_* columns in the input CSV.")

    belief_col = resolve_column(
        df,
        [
            "global_posterior_uncertainty",
            "global_belief_uncertainty_mean",
            "belief_uncertainty_mean",
        ],
    )

    rows = []
    for user_id, g in df.groupby(user_key, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < MIN_RUN_LENGTH:
            continue

        g = g.dropna(
            subset=[
                "global_dominant_intent",
                "global_dominant_intent_prob",
                "global_intent_entropy",
                belief_col,
                "attribution_confidence_gap",
            ]
        ).reset_index(drop=True)
        if len(g) < MIN_RUN_LENGTH:
            continue

        dom = pd.to_numeric(g["global_dominant_intent"], errors="coerce").fillna(-1).to_numpy()
        intent_entropy = pd.to_numeric(g["global_intent_entropy"], errors="coerce").to_numpy()
        dominant_prob = pd.to_numeric(g["global_dominant_intent_prob"], errors="coerce").to_numpy()
        uncertainty = pd.to_numeric(g[belief_col], errors="coerce").to_numpy()
        attr_gap = pd.to_numeric(g["attribution_confidence_gap"], errors="coerce").to_numpy()
        pi = g[pi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        adj_jsd = consecutive_jsd(pi)
        if len(adj_jsd) == 0:
            continue

        threshold = run_jsd_threshold
        start = 0
        while start < len(g):
            end = start
            while end < len(g) - 1 and adj_jsd[end] <= threshold:
                end += 1

            run_len = end - start + 1
            if run_len >= MIN_RUN_LENGTH:
                seg = slice(start, end + 1)
                seg_dom = dom[seg]
                seg_pi = pi[seg]
                seg_entropy = intent_entropy[seg]
                seg_prob = dominant_prob[seg]
                seg_uncertainty = uncertainty[seg]
                seg_attr = attr_gap[seg]
                switch_count = int(np.sum(seg_dom[1:] != seg_dom[:-1])) if run_len > 1 else 0
                rows.append(
                    {
                        user_key: user_id,
                        "run_start_pos": int(start),
                        "run_end_pos": int(end),
                        "run_length": int(run_len),
                        "run_length_cap": int(min(run_len, MAX_RUN_LENGTH_BIN)),
                        "mean_intent_entropy": float(np.nanmean(seg_entropy)),
                        "mean_dominant_intent_prob": float(np.nanmean(seg_prob)),
                        "switch_count": switch_count,
                        "posterior_dispersion": mean_consecutive_jsd(seg_pi),
                        "mean_uncertainty": float(np.nanmean(seg_uncertainty)),
                        "attribution_dispersion": mean_abs_diff(seg_attr),
                        "neighbor_threshold_user": threshold,
                    }
                )

            start = max(end + 1, start + 1)

    return pd.DataFrame(rows)


def build_transition_summary(df: pd.DataFrame, user_key: str) -> pd.DataFrame:
    pi_cols = pi_columns(df, "global_pi_")
    if not pi_cols:
        raise KeyError("Could not find global_pi_* columns in the input CSV.")

    proxy_col = resolve_column(
        df,
        ["attribution_source_proxy", "channel", "domain"],
        required=False,
    )
    if proxy_col is None:
        raise KeyError("Could not find an anchor proxy column.")

    rows = []
    for user_id, g in df.groupby(user_key, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        dom = pd.to_numeric(g["global_dominant_intent"], errors="coerce").fillna(-1).to_numpy()
        uncertainty = pd.to_numeric(g["belief_uncertainty"], errors="coerce").to_numpy()
        attr_gap = pd.to_numeric(g["attribution_confidence_gap"], errors="coerce").to_numpy()
        proxy = g[proxy_col].astype(str).replace("nan", np.nan).to_numpy()
        pi = g[pi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for i in range(1, len(g)):
            prev_proxy = proxy[i - 1]
            cur_proxy = proxy[i]
            if pd.isna(prev_proxy) or pd.isna(cur_proxy):
                continue
            rows.append(
                {
                    user_key: user_id,
                    "exploration_score": float(g["exploration_score"].iloc[i]),
                    "anchor_transition": f"{prev_proxy}->{cur_proxy}",
                    "intent_shift": js_distance(pi[i - 1], pi[i]),
                    "dominant_intent_change_rate": float(dom[i - 1] != dom[i]),
                    "uncertainty": float(uncertainty[i]),
                    "attribution_shift": float(abs(attr_gap[i] - attr_gap[i - 1])),
                }
            )

    return pd.DataFrame(rows)


def aggregate_transition_score_lines(
    summary: pd.DataFrame,
    score_col: str,
    metric_cols: list[str],
    n_bins: int = TRANSITION_SCORE_BINS,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    valid = summary[np.isfinite(summary[score_col])].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["score_bin"] = pd.qcut(
        valid[score_col].rank(method="first"),
        q=min(n_bins, valid[score_col].nunique()),
        labels=False,
        duplicates="drop",
    )
    if valid["score_bin"].isna().all():
        return pd.DataFrame()

    rows = []
    for transition in TRANSITION_ORDER:
        g_transition = valid[valid["anchor_transition"] == transition]
        if g_transition.empty:
            continue
        for bin_id, g_bin in g_transition.groupby("score_bin", sort=True):
            if len(g_bin) < MIN_BIN_SAMPLES:
                continue
            row = {
                "anchor_transition": transition,
                "score_bin": int(bin_id),
                "exploration_score_bin_center": float(np.nanmean(g_bin[score_col])),
                "n": int(len(g_bin)),
            }
            for metric in metric_cols:
                row[f"{metric}_mean"] = float(np.nanmean(g_bin[metric]))
                row[f"{metric}_sem"] = sem(g_bin[metric])
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["anchor_transition", "score_bin"])
        .reset_index(drop=True)
    )


def decile_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    valid = df[np.isfinite(df[score_col])].copy()
    if valid.empty:
        return pd.DataFrame()
    n_bins = min(10, valid[score_col].nunique())
    if n_bins < 1:
        return pd.DataFrame()
    valid["decile"] = (
        pd.qcut(
            valid[score_col].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    out = (
        valid.groupby("decile", as_index=False)
        .agg(
            exploration_score=(score_col, "mean"),
            intent_entropy_mean=("global_intent_entropy", "mean"),
            intent_entropy_sem=("global_intent_entropy", sem),
            belief_uncertainty_mean=("belief_uncertainty", "mean"),
            belief_uncertainty_sem=("belief_uncertainty", sem),
            dominant_intent_prob_mean=("global_dominant_intent_prob", "mean"),
            dominant_intent_prob_sem=("global_dominant_intent_prob", sem),
            n=("decile", "size"),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )
    return out


def aggregate_user_metrics(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for x_val, g in df.groupby(x_col, sort=True):
        if len(g) < MIN_BIN_SAMPLES:
            continue
        row = {x_col: x_val, "n": int(len(g))}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(np.nanmean(g[metric]))
            row[f"{metric}_sem"] = sem(g[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(x_col).reset_index(drop=True)


def plot_multi_series(
    summary: pd.DataFrame,
    x_col: str,
    series_specs: list[tuple[str, str, str, str]],
    out_path: Path,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2))
    for mean_col, sem_col, label, color in series_specs:
        curve = summary[[x_col, mean_col, sem_col]].dropna().sort_values(x_col)
        if curve.empty:
            continue
        x = curve[x_col].to_numpy(dtype=float)
        y = curve[mean_col].to_numpy(dtype=float)
        y_sem = curve[sem_col].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=1.96 * y_sem,
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_lines(
    summary: pd.DataFrame,
    x_col: str,
    series_specs: list[tuple[str, str, str, str]],
    out_path: Path,
    x_label: str,
    title: str | None,
) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(
        len(series_specs),
        1,
        figsize=(8.8, 3.0 * len(series_specs)),
        sharex=True,
        constrained_layout=True,
    )
    if len(series_specs) == 1:
        axes = [axes]

    for ax, (mean_col, sem_col, label, color) in zip(axes, series_specs):
        curve = summary[[x_col, mean_col, sem_col]].dropna().sort_values(x_col)
        if not curve.empty:
            x = curve[x_col].to_numpy(dtype=float)
            y = curve[mean_col].to_numpy(dtype=float)
            y_sem = curve[sem_col].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=1.96 * y_sem,
                marker="o",
                linewidth=2.2,
                color=color,
            )
        ax.set_title(label, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(x_label)
    if title:
        fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transition_score_lines(summary: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 6.0))

    for transition in TRANSITION_ORDER:
        g = summary[summary["anchor_transition"] == transition].sort_values("score_bin")
        if g.empty:
            continue
        x = g["score_bin"].to_numpy(dtype=float) + 1.0
        y = g["intent_shift_mean"].to_numpy(dtype=float)
        y_sem = g["intent_shift_sem"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=1.96 * y_sem,
            color=TRANSITION_COLORS[transition],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=transition,
        )

    ax.set_xlabel("Exploration score bin")
    ax.set_ylabel("Mean intent shift")
    ax.set_title("Transition Type -> Intent Shift", loc="left", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    ax.set_xticks(np.arange(1, TRANSITION_SCORE_BINS + 1))
    ax.set_xlim(0.8, TRANSITION_SCORE_BINS + 0.2)
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path, low_memory=False)
    df = prepare_frame(raw, args.exploration_column, args.user_key)

    fig1_dir = output_root / "1"
    fig1 = decile_summary(df, "exploration_score")
    save_table(fig1, fig1_dir / "exploration_intent_ambiguity.csv")
    plot_multi_series(
        fig1,
        "decile",
        [
            ("intent_entropy_mean", "intent_entropy_sem", "Intent entropy", SERIES_COLORS[0]),
            (
                "dominant_intent_prob_mean",
                "dominant_intent_prob_sem",
                "Dominant intent prob",
                SERIES_COLORS[1],
            ),
        ],
        fig1_dir / "exploration_intent_ambiguity.png",
        "Exploration Score Decile",
        "Mean value",
        None,
    )

    user_summary = build_user_summary(df, args.user_key)

    fig23_dir = output_root / "23"
    run_summary = build_stable_runs(df, args.user_key, args.run_jsd_threshold)
    save_table(run_summary, fig23_dir / "stable_runs.csv")
    fig23 = aggregate_user_metrics(
        run_summary,
        "run_length_cap",
        ["mean_intent_entropy", "mean_uncertainty", "posterior_dispersion"],
    )
    save_table(fig23, fig23_dir / "run_length_intent_consolidation_dispersion.csv")
    plot_stacked_lines(
        fig23,
        "run_length_cap",
        [
            ("mean_intent_entropy_mean", "mean_intent_entropy_sem", "Intent entropy", SERIES_COLORS[0]),
            ("mean_uncertainty_mean", "mean_uncertainty_sem", "Uncertainty", SERIES_COLORS[1]),
            ("posterior_dispersion_mean", "posterior_dispersion_sem", "Posterior dispersion", SERIES_COLORS[2]),
        ],
        fig23_dir / "run_length_intent_consolidation_dispersion.png",
        "Semantic neighbor run length",
        None,
    )

    fig4_dir = output_root / "4"
    transition_summary = build_transition_summary(df, args.user_key)
    transition_score_summary = aggregate_transition_score_lines(
        transition_summary,
        "exploration_score",
        ["intent_shift"],
        n_bins=TRANSITION_SCORE_BINS,
    )
    save_table(transition_score_summary, fig4_dir / "transition_type_intent_shift.csv")
    plot_transition_score_lines(
        transition_score_summary,
        fig4_dir / "transition_type_intent_shift.png",
    )

    print(f"Saved analysis tables and figures under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
