from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent.parent
    / "5-Validation"
    / "output_mechanism"
    / "event_level"
    / "model_events.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "Part3"
STATE_LABELS = ["Low", "Medium", "High"]
CHANNEL_MARKERS = {"R": "o", "S": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize event-level PC-SAR intent posterior structure with t-SNE."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to event-level model_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for t-SNE figures and tables.",
    )
    parser.add_argument(
        "--max-per-intent",
        type=int,
        default=2000,
        help="Maximum sampled events per dominant intent.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=20000,
        help="Maximum total sampled events after per-intent sampling.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=40.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="Random seed for sampling and t-SNE.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def suffix_value(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 10**9


def pi_columns(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if col.startswith("global_pi_")], key=suffix_value)


def assign_uncertainty_state(df: pd.DataFrame) -> pd.Series:
    uncertainty = pd.to_numeric(df["global_posterior_uncertainty"], errors="coerce")
    valid = uncertainty.dropna()
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if valid.nunique() < 2:
        return out
    out.loc[valid.index] = pd.qcut(
        valid.rank(method="first"),
        q=3,
        labels=STATE_LABELS,
        duplicates="drop",
    ).astype("object")
    return out


def load_event_sample(path: Path, max_per_intent: int, max_total: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    pi_cols = pi_columns(df)
    if not pi_cols:
        raise KeyError("Missing global_pi_* columns in event-level input.")

    keep_cols = [
        "event_index",
        "user_id",
        "channel",
        "event_type",
        "timestamp",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
    ] + pi_cols
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()
    for col in pi_cols + [
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=pi_cols).reset_index(drop=True)
    df["dominant_intent"] = df[pi_cols].to_numpy(dtype=float).argmax(axis=1)
    df["uncertainty_state"] = assign_uncertainty_state(df)
    df["channel"] = df.get("channel", "").astype(str).str.upper()

    sampled_parts = []
    for _, g in df.groupby("dominant_intent", sort=True):
        n = min(max_per_intent, len(g))
        sampled_parts.append(g.sample(n=n, random_state=random_state))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_total:
        sampled = sampled.sample(n=max_total, random_state=random_state).reset_index(drop=True)
    return sampled.reset_index(drop=True)


def run_tsne(sample: pd.DataFrame, perplexity: float, random_state: int) -> pd.DataFrame:
    pi_cols = pi_columns(sample)
    x = sample[pi_cols].to_numpy(dtype=np.float64)
    x = np.clip(x, 1e-12, 1.0)
    x = x / np.clip(x.sum(axis=1, keepdims=True), 1e-12, None)
    effective_perplexity = min(float(perplexity), max(5.0, (len(sample) - 1) / 3.0))
    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
    ).fit_transform(x)
    out = sample.copy()
    out["tsne_1"] = coords[:, 0]
    out["tsne_2"] = coords[:, 1]
    out["tsne_perplexity"] = effective_perplexity
    return out


def build_summary(embedded: pd.DataFrame) -> pd.DataFrame:
    pi_cols = pi_columns(embedded)
    x = embedded[pi_cols].to_numpy(dtype=np.float64)
    labels = embedded["dominant_intent"].to_numpy()
    if len(np.unique(labels)) > 1 and len(embedded) > len(np.unique(labels)):
        sil = float(silhouette_score(x, labels, metric="euclidean"))
    else:
        sil = float("nan")

    usage = embedded["dominant_intent"].value_counts(normalize=True).sort_index()
    usage_entropy = float(-(usage * np.log(np.clip(usage, 1e-12, 1.0))).sum() / np.log(len(usage)))
    rows = [
        {
            "metric": "sample_size",
            "value": float(len(embedded)),
        },
        {
            "metric": "dominant_intent_silhouette",
            "value": sil,
        },
        {
            "metric": "dominant_intent_usage_entropy",
            "value": usage_entropy,
        },
        {
            "metric": "mean_dominant_intent_prob",
            "value": float(embedded["global_dominant_intent_prob"].mean()),
        },
        {
            "metric": "mean_uncertainty",
            "value": float(embedded["global_posterior_uncertainty"].mean()),
        },
    ]
    return pd.DataFrame(rows)


def plot_by_dominant_intent(embedded: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    intents = sorted(embedded["dominant_intent"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for idx, intent in enumerate(intents):
        g_intent = embedded[embedded["dominant_intent"] == intent]
        for channel, marker in CHANNEL_MARKERS.items():
            g = g_intent[g_intent["channel"] == channel]
            if g.empty:
                continue
            ax.scatter(
                g["tsne_1"],
                g["tsne_2"],
                s=9,
                alpha=0.52,
                marker=marker,
                color=cmap(idx % 10),
                linewidths=0,
                label=f"Intent {intent} / {channel}",
            )
    ax.set_title("t-SNE of Global Intent Posteriors by Dominant Intent", loc="left", weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 20:
        ax.legend(handles, labels, frameon=False, markerscale=1.8, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_uncertainty_state(embedded: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    colors = {"Low": "#2f6f4e", "Medium": "#c7862f", "High": "#a33a2f"}

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for state in STATE_LABELS:
        g_state = embedded[embedded["uncertainty_state"] == state]
        for channel, marker in CHANNEL_MARKERS.items():
            g = g_state[g_state["channel"] == channel]
            if g.empty:
                continue
            ax.scatter(
                g["tsne_1"],
                g["tsne_2"],
                s=9,
                alpha=0.48,
                marker=marker,
                color=colors[state],
                linewidths=0,
                label=f"{state} uncertainty / {channel}",
            )
    ax.set_title("t-SNE of Global Intent Posteriors by Uncertainty State", loc="left", weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, markerscale=1.8, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    sample = load_event_sample(
        input_path,
        max_per_intent=args.max_per_intent,
        max_total=args.max_total,
        random_state=args.random_state,
    )
    embedded = run_tsne(sample, perplexity=args.perplexity, random_state=args.random_state)
    summary = build_summary(embedded)

    embedded.to_csv(output_dir / "intent_tsne_sample.csv", index=False)
    summary.to_csv(output_dir / "intent_tsne_summary.csv", index=False)
    plot_by_dominant_intent(embedded, output_dir / "intent_tsne_by_dominant_intent.png")
    plot_by_uncertainty_state(embedded, output_dir / "intent_tsne_by_uncertainty_state.png")

    print(f"Saved t-SNE outputs under: {output_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
