from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RUN_NEIGHBOR_Q = 0.50
K_FUTURE = 5
MIN_FUTURE = 1
TRANSITION_ORDER = ['R->R', 'R->S', 'S->R', 'S->S']
TRANSITION_COLORS = {
    'R->R': '#1f77b4',
    'R->S': '#ff7f0e',
    'S->R': '#2ca02c',
    'S->S': '#d62728',
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SCRIPT_DIR.parent
INTERMEDIATE_DIR = PROBLEM_DIR / 'intermediate'
FIG_DIR = PROBLEM_DIR / 'figures'
from _shared import ensure_part1_artifacts


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def l2_normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def mean_unit(vectors):
    center = vectors.mean(axis=0, keepdims=True)
    return l2_normalize(center)[0]


def run_transition_label(domains, start, end):
    transitions = [f'{domains[t]}->{domains[t + 1]}' for t in range(start, end)]
    if not transitions:
        return 'UNKNOWN'
    counts = pd.Series(transitions).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    for item in TRANSITION_ORDER:
        if item in top:
            return item
    return top[0]


def build_run_table(events, event_emb):
    rows = []

    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
        domains = g['domain'].to_numpy()
        m = len(g)
        if m < 2:
            continue

        adjacent_cos = np.sum(emb[:-1] * emb[1:], axis=1)
        neighbor_threshold = float(np.quantile(adjacent_cos, RUN_NEIGHBOR_Q))

        start = 0
        while start < m:
            end = start
            while end < m - 1 and adjacent_cos[end] >= neighbor_threshold:
                end += 1

            run_len = end - start + 1
            future = emb[end + 1:end + 1 + K_FUTURE]
            if run_len >= 2 and len(future) >= MIN_FUTURE:
                run_vecs = emb[start:end + 1]
                run_center = mean_unit(run_vecs)
                future_center = mean_unit(future)
                current_radius = float(np.mean(1.0 - (run_vecs @ run_center)))
                future_radius = float(np.mean(1.0 - (future @ future_center)))
                future_stability = float(np.dot(run_center, future_center))
                run_transition_type = run_transition_label(domains, start, end)
                run_scores = g.iloc[start:end + 1]['exploration_score'].to_numpy(dtype=float)
                if np.isfinite(run_scores).any():
                    run_exploration_score = float(np.nanmean(run_scores))
                else:
                    run_exploration_score = np.nan

                rows.append({
                    'user_id': int(user_id),
                    'run_start_pos': int(start),
                    'run_end_pos': int(end),
                    'run_length': int(run_len),
                    'current_radius': current_radius,
                    'future_stability': future_stability,
                    'future_radius': future_radius,
                    'run_exploration_score': run_exploration_score,
                    'run_transition_type': run_transition_type,
                    'neighbor_threshold_user': neighbor_threshold,
                })

            start = max(end + 1, start + 1)

    return pd.DataFrame(rows)


def plot_transition_quadrant(runs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    cmap = plt.get_cmap('viridis')
    x_thr = float(runs['future_stability'].median())
    y_thr = float(runs['future_radius'].median())
    x_min = float(runs['future_stability'].min())
    x_max = float(runs['future_stability'].max())
    y_min = float(runs['future_radius'].min())
    y_max = float(runs['future_radius'].max())

    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04
    label_box = dict(boxstyle='round,pad=0.22', facecolor='white', alpha=0.72, edgecolor='none')

    last_sc = None
    for ax, transition in zip(axes, TRANSITION_ORDER):
        sub = runs[runs['run_transition_type'] == transition]
        if sub.empty:
            ax.set_visible(False)
            continue

        last_sc = ax.scatter(
            sub['future_stability'],
            sub['future_radius'],
            c=sub['run_exploration_score'],
            s=14,
            alpha=0.35,
            cmap=cmap,
            edgecolors='none',
        )
        ax.axvline(x_thr, color='black', linestyle='--', linewidth=1)
        ax.axhline(y_thr, color='black', linestyle='--', linewidth=1)

        ax.text(x_min + x_pad, y_max - y_pad, 'unstable + diffuse', fontsize=10, va='top', ha='left', bbox=label_box)
        ax.text(x_thr + x_pad * 0.4, y_max - y_pad, 'stable + diffuse', fontsize=10, va='top', ha='left', bbox=label_box)
        ax.text(x_min + x_pad, y_min + y_pad * 0.8, 'convergent but not retained', fontsize=10, va='bottom', ha='left', bbox=label_box)
        ax.text(x_thr + x_pad * 0.4, y_min + y_pad * 0.8, 'strong consolidation', fontsize=10, va='bottom', ha='left', bbox=label_box)

        ax.set_title(transition, fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.18)

    for ax in axes[2:]:
        ax.set_xlabel('Future Consistency', fontsize=13)
    for ax in axes[::2]:
        ax.set_ylabel('Future Semantic Dispersion', fontsize=13)

    cbar = fig.colorbar(last_sc, ax=axes.tolist(), fraction=0.035, pad=0.02)
    cbar.set_label('Exploration Score', fontsize=14)

    fig.suptitle('Figure 2: Consolidation Quadrants by Transition', y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_consolidation_quadrants_by_transition.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    scores_path = INTERMEDIATE_DIR / '1-exploration-scores.pkl'
    if not events_path.exists() or not emb_path.exists() or not scores_path.exists():
        raise FileNotFoundError('Missing Part 1.1 caches. Run Part 1.1 first.')

    events = load_pickle(events_path)
    event_emb = np.load(emb_path)
    scores = load_pickle(scores_path)[['event_id', 'exploration_score']]
    events = events.merge(scores, on='event_id', how='left', validate='one_to_one')

    runs = build_run_table(events, event_emb)
    print(runs.head())
    plot_transition_quadrant(runs)


if __name__ == '__main__':
    main()
