from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RUN_NEIGHBOR_Q = 0.50
K_FUTURE = 5
MIN_FUTURE = 1

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


def build_run_table(events, event_emb):
    rows = []

    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
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
                    'neighbor_threshold_user': neighbor_threshold,
                })

            start = max(end + 1, start + 1)

    return pd.DataFrame(rows)


def plot_quadrant_figure(runs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    cmap = plt.get_cmap('viridis')

    sc = ax.scatter(
        runs['future_stability'],
        runs['future_radius'],
        c=runs['run_exploration_score'],
        s=14,
        alpha=0.35,
        cmap=cmap,
        edgecolors='none',
    )

    x_thr = float(runs['future_stability'].median())
    y_thr = float(runs['future_radius'].median())
    ax.axvline(x_thr, color='black', linestyle='--', linewidth=1)
    ax.axhline(y_thr, color='black', linestyle='--', linewidth=1)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04
    label_box = dict(boxstyle='round,pad=0.22', facecolor='white', alpha=0.72, edgecolor='none')

    ax.text(x_min + x_pad, y_max - y_pad, 'unstable + diffuse', fontsize=12, va='top', ha='left', bbox=label_box)
    ax.text(x_thr + x_pad * 0.4, y_max - y_pad, 'stable + diffuse', fontsize=12, va='top', ha='left', bbox=label_box)
    ax.text(x_min + x_pad, y_min + y_pad * 0.8, 'convergent but not retained', fontsize=12, va='bottom', ha='left', bbox=label_box)
    ax.text(x_thr + x_pad * 0.4, y_min + y_pad * 0.8, 'strong consolidation', fontsize=12, va='bottom', ha='left', bbox=label_box)

    ax.set_xlabel('Future Consistency', fontsize=14)
    ax.set_ylabel('Future Semantic Dispersion', fontsize=14)
    ax.grid(alpha=0.18)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Exploration Score', fontsize=14)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_consolidation_quadrants.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_quadrant_figure_3d(runs):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')

    z = runs['run_length'].to_numpy(dtype=float)
    sc = ax.scatter(
        runs['future_stability'],
        runs['future_radius'],
        z,
        c=runs['run_exploration_score'],
        s=16,
        alpha=0.38,
        cmap=cmap,
        edgecolors='none',
        depthshade=True,
    )

    x_thr = float(runs['future_stability'].median())
    y_thr = float(runs['future_radius'].median())
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))

    # Quadrant guides projected onto the base plane and extended vertically.
    ax.plot([x_thr, x_thr], [runs['future_radius'].min(), runs['future_radius'].max()], [z_min, z_min],
            color='black', linestyle='--', linewidth=1.1, alpha=0.85)
    ax.plot([runs['future_stability'].min(), runs['future_stability'].max()], [y_thr, y_thr], [z_min, z_min],
            color='black', linestyle='--', linewidth=1.1, alpha=0.85)
    ax.plot([x_thr, x_thr], [runs['future_radius'].min(), runs['future_radius'].min()], [z_min, z_max],
            color='black', linestyle=':', linewidth=0.9, alpha=0.55)
    ax.plot([x_thr, x_thr], [runs['future_radius'].max(), runs['future_radius'].max()], [z_min, z_max],
            color='black', linestyle=':', linewidth=0.9, alpha=0.55)
    ax.plot([runs['future_stability'].min(), runs['future_stability'].min()], [y_thr, y_thr], [z_min, z_max],
            color='black', linestyle=':', linewidth=0.9, alpha=0.55)
    ax.plot([runs['future_stability'].max(), runs['future_stability'].max()], [y_thr, y_thr], [z_min, z_max],
            color='black', linestyle=':', linewidth=0.9, alpha=0.55)

    ax.set_xlabel('Future Consistency', fontsize=13, labelpad=12)
    ax.set_ylabel('Future Semantic Dispersion', fontsize=13, labelpad=12)
    ax.set_zlabel('Run Length', fontsize=13, labelpad=10)

    ax.set_title('Figure 2: Consolidation Quadrants in 3D', pad=18, fontsize=15)
    ax.view_init(elev=24, azim=-58)

    try:
        ax.xaxis.pane.set_alpha(0.08)
        ax.yaxis.pane.set_alpha(0.08)
        ax.zaxis.pane.set_alpha(0.08)
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.08, shrink=0.88)
    cbar.set_label('Exploration Score', fontsize=13)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_consolidation_quadrants-3D.png', dpi=200, bbox_inches='tight')
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
    plot_quadrant_figure(runs)
    plot_quadrant_figure_3d(runs)


if __name__ == '__main__':
    main()
