from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

TRANSITION_ORDER = ['R->R', 'R->S', 'S->R', 'S->S']
TRANSITION_COLORS = {
    'R->R': '#1f77b4',
    'R->S': '#ff7f0e',
    'S->R': '#2ca02c',
    'S->S': '#d62728',
}
N_BINS = 10
N_HISTORY = 10
K_FUTURE = 5

SCRIPT_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SCRIPT_DIR.parent
INTERMEDIATE_DIR = PROBLEM_DIR / 'intermediate'
FIG_DIR = PROBLEM_DIR / 'figures'


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def sem(series):
    series = pd.Series(series).dropna()
    return series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0


def summarize_transition_curves(scores):
    plot_df = scores[
        scores['transition_into'].isin(TRANSITION_ORDER)
        & scores['exploration_pct_user'].notna()
        & scores['future_consistency'].notna()
    ].copy()
    plot_df['exploration_pct_bin'] = pd.cut(
        plot_df['exploration_pct_user'],
        bins=np.linspace(0, 1, N_BINS + 1),
        include_lowest=True,
    )
    summary = (
        plot_df.groupby(['transition_into', 'exploration_pct_bin'], observed=True)
        .agg(
            exploration_pct_mid=('exploration_pct_user', 'mean'),
            future_consistency_mean=('future_consistency', 'mean'),
            future_consistency_sem=('future_consistency', sem),
            events=('future_consistency', 'size'),
        )
        .reset_index()
        .sort_values(['transition_into', 'exploration_pct_mid'])
    )
    return summary


def plot_transition_curves(summary):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2))
    for transition in TRANSITION_ORDER:
        curve = summary[summary['transition_into'] == transition].sort_values('exploration_pct_mid')
        if curve.empty:
            continue
        color = TRANSITION_COLORS[transition]
        x = curve['exploration_pct_mid'].to_numpy(dtype=float)
        y = curve['future_consistency_mean'].to_numpy(dtype=float)
        sem_vals = curve['future_consistency_sem'].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=1.96 * sem_vals,
            marker='o',
            linewidth=2.2,
            color=color,
            label=transition,
        )

    ax.set_xlabel('Within-user Exploration Score Percentile')
    ax.set_ylabel(f'Future Consistency')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title='Transition')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_transition_level_exploration_asymmetry.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    scores_path = INTERMEDIATE_DIR / '1-exploration-scores.pkl'
    if not scores_path.exists():
        raise FileNotFoundError('Missing 1-exploration-scores.pkl. Run Part 1.1 first.')

    scores = load_pickle(scores_path)
    summary = summarize_transition_curves(scores)
    print(summary)
    plot_transition_curves(summary)


if __name__ == '__main__':
    main()
