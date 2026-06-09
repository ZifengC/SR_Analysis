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
from _shared import ensure_part1_artifacts


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def sem(series):
    series = pd.Series(series).dropna()
    return series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0


def l2_normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def mean_unit(vectors):
    center = vectors.mean(axis=0, keepdims=True)
    return l2_normalize(center)[0]


def build_transition_curves(scores, events, event_emb):
    merged = scores.merge(
        events[['event_id', 'user_id', 'event_pos', 'embedding_row']],
        on=['event_id', 'user_id', 'event_pos'],
        how='inner',
        suffixes=('', '_event'),
    )
    merged = merged[
        merged['transition_into'].isin(TRANSITION_ORDER)
        & merged['exploration_score'].notna()
    ].copy()

    rows = []
    for user_id, g in merged.groupby('user_id', sort=False):
        g = g.sort_values('event_pos')
        emb_rows = g['embedding_row'].to_numpy(dtype=int)
        exploration_scores = g['exploration_score'].to_numpy(dtype=float)
        transitions = g['transition_into'].to_numpy()
        m = len(g)
        if m <= K_FUTURE:
            continue

        for i in range(m - K_FUTURE):
            future = event_emb[emb_rows[i + 1:i + 1 + K_FUTURE]]
            if len(future) < K_FUTURE:
                continue
            current = event_emb[emb_rows[i]]
            future_center = mean_unit(future)
            future_consistency = float(np.dot(current, future_center))
            future_dispersion = float(np.mean(1.0 - (future @ future_center)))

            rows.append({
                'user_id': int(user_id),
                'event_pos': int(g.iloc[i]['event_pos']),
                'transition_into': transitions[i],
                'exploration_score': exploration_scores[i],
                'future_consistency': future_consistency,
                'future_dispersion': future_dispersion,
            })

    return pd.DataFrame(rows)


def summarize_transition_curves(curves):
    if curves.empty:
        return pd.DataFrame(columns=[
            'transition_into', 'exploration_score_bin', 'exploration_score_mid',
            'future_consistency_mean', 'future_consistency_sem',
            'future_dispersion_mean', 'future_dispersion_sem',
            'events',
        ])

    plot_df = curves[
        curves['transition_into'].isin(TRANSITION_ORDER)
        & curves['exploration_score'].notna()
    ].copy()
    plot_df['exploration_score_bin'] = pd.qcut(
        plot_df['exploration_score'],
        q=N_BINS,
        duplicates='drop',
    )
    summary = (
        plot_df.groupby(['transition_into', 'exploration_score_bin'], observed=True)
        .agg(
            exploration_score_mid=('exploration_score', 'mean'),
            future_consistency_mean=('future_consistency', 'mean'),
            future_consistency_sem=('future_consistency', sem),
            future_dispersion_mean=('future_dispersion', 'mean'),
            future_dispersion_sem=('future_dispersion', sem),
            events=('future_consistency', 'size'),
        )
        .reset_index()
        .sort_values(['transition_into', 'exploration_score_mid'])
    )
    return summary


def plot_transition_curves(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharex=True)
    panels = [
        (
            axes[0],
            'future_consistency_mean',
            'future_consistency_sem',
            'Future Consistency',
        ),
        (
            axes[1],
            'future_dispersion_mean',
            'future_dispersion_sem',
            'Future Expansion',
        ),
    ]

    for ax, mean_col, sem_col, ylabel in panels:
        for transition in TRANSITION_ORDER:
            curve = summary[summary['transition_into'] == transition].sort_values('exploration_score_mid')
            if curve.empty:
                continue
            color = TRANSITION_COLORS[transition]
            x = curve['exploration_score_mid'].to_numpy(dtype=float)
            y = curve[mean_col].to_numpy(dtype=float)
            y_sem = curve[sem_col].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=1.96 * y_sem,
                marker='o',
                linewidth=2.2,
                color=color,
                label=transition,
            )
        ax.set_xlabel('Exploration score')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, title='Transition')


    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_transition_stratified_exploration_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    scores_path = INTERMEDIATE_DIR / '1-exploration-scores.pkl'
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if not scores_path.exists() or not events_path.exists() or not emb_path.exists():
        raise FileNotFoundError('Missing Part 1 caches. Run Part 1 first.')

    scores = load_pickle(scores_path)
    events = load_pickle(events_path)
    event_emb = np.load(emb_path)
    curves = build_transition_curves(scores, events, event_emb)
    summary = summarize_transition_curves(curves)
    print(curves.head())
    print(summary)
    plot_transition_curves(summary)


if __name__ == '__main__':
    main()
