from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N_HISTORY = 10
K_FUTURE = 5
ANCHOR_SIM_THRESHOLD = 0.30
ANCHOR_ORDER = ['R->R', 'R->S', 'S->R', 'S->S']
ANCHOR_COLORS = {
    'R->R': '#1f77b4',
    'R->S': '#ff7f0e',
    'S->R': '#2ca02c',
    'S->S': '#d62728',
}
N_BINS = 10

SCRIPT_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SCRIPT_DIR.parent
INTERMEDIATE_DIR = PROBLEM_DIR / 'intermediate'
FIG_DIR = PROBLEM_DIR / 'figures'
from _shared import ensure_part1_artifacts


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def sem(series):
    series = pd.Series(series).dropna()
    return series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0


def summarize_anchor_curves(scores):
    plot_df = scores[
        scores['anchor_transition'].isin(ANCHOR_ORDER)
        & scores['nearest_anchor_similarity'].notna()
        & (scores['nearest_anchor_similarity'] >= ANCHOR_SIM_THRESHOLD)
        & scores['anchor_exploration_adoption'].notna()
    ].copy()
    if plot_df.empty:
        return pd.DataFrame(columns=[
            'anchor_transition', 'anchor_similarity_bin', 'anchor_similarity_mid',
            'adoption_score_mean', 'adoption_score_sem',
            'success_rate', 'success_rate_sem', 'events',
        ])

    plot_df['anchor_similarity_bin'] = pd.qcut(
        plot_df['nearest_anchor_similarity'],
        q=N_BINS,
        duplicates='drop',
    )
    plot_df['anchor_success'] = plot_df['anchor_exploration_adoption'] > 0
    summary = (
        plot_df.groupby(['anchor_transition', 'anchor_similarity_bin'], observed=True)
        .agg(
            anchor_similarity_mid=('nearest_anchor_similarity', 'mean'),
            adoption_score_mean=('anchor_exploration_adoption', 'mean'),
            adoption_score_sem=('anchor_exploration_adoption', sem),
            success_rate=('anchor_success', 'mean'),
            success_rate_sem=('anchor_success', sem),
            events=('event_id', 'size'),
        )
        .reset_index()
        .sort_values(['anchor_transition', 'anchor_similarity_mid'])
    )
    return summary


def plot_anchor_curves(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), sharex=True)
    panels = [
        (
            axes[0],
            'adoption_score_mean',
            'adoption_score_sem',
            'Mean anchor adoption score',
            'anchor_exploration_adoption = future_consistency - nearest_anchor_similarity',
        ),
        (
            axes[1],
            'success_rate',
            'success_rate_sem',
            'Successful anchor exploration rate',
            'P(anchor_exploration_adoption > 0)',
        ),
    ]

    for ax, mean_col, sem_col, title, ylabel in panels:
        for transition in ANCHOR_ORDER:
            curve = summary[summary['anchor_transition'] == transition].sort_values('anchor_similarity_mid')
            if curve.empty:
                continue
            color = ANCHOR_COLORS[transition]
            x = curve['anchor_similarity_mid'].to_numpy(dtype=float)
            y = curve[mean_col].to_numpy(dtype=float)
            sem_vals = curve[sem_col].to_numpy(dtype=float)
            ax.plot(x, y, marker='o', linewidth=2.2, color=color, label=transition)
            ax.fill_between(x, y - 1.96 * sem_vals, y + 1.96 * sem_vals, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel('Nearest historical anchor similarity')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, title='Anchor transition')
    axes[1].legend(frameon=False, title='Anchor transition')
    fig.suptitle(
        f'Part 2.2: Anchor-Conditioned Transition Adoption (fixed N_HISTORY={N_HISTORY}, K_FUTURE={K_FUTURE})',
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        f'Anchor filter: nearest_anchor_similarity >= {ANCHOR_SIM_THRESHOLD:.2f}',
        ha='center',
        va='bottom',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_anchor_conditioned_transition_adoption.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    scores_path = INTERMEDIATE_DIR / '1-exploration-scores.pkl'
    if not scores_path.exists():
        raise FileNotFoundError('Missing 1-exploration-scores.pkl. Run Part 1.1 first.')

    scores = load_pickle(scores_path)
    summary = summarize_anchor_curves(scores)
    print(summary)
    plot_anchor_curves(summary)


if __name__ == '__main__':
    main()
