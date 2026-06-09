from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RUN_NEIGHBOR_QS = [0.30, 0.50, 0.70]
K_FUTURE = 5
MIN_FUTURE = 1
RUN_LENGTH_CAP = 10

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


def sem(series):
    series = pd.Series(series).dropna()
    return series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0


def build_run_table(events, event_emb, neighbor_q):
    rows = []

    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
        m = len(g)
        if m < 2:
            continue

        adjacent_cos = np.sum(emb[:-1] * emb[1:], axis=1)
        neighbor_threshold = float(np.quantile(adjacent_cos, neighbor_q))

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
                future_consistency = float(np.dot(run_center, future_center))
                semantic_dispersion = float(np.mean(1.0 - (run_vecs @ run_center)))

                rows.append({
                    'user_id': int(user_id),
                    'run_length': int(run_len),
                    'run_length_cap': int(min(run_len, RUN_LENGTH_CAP)),
                    'future_consistency': future_consistency,
                    'semantic_dispersion': semantic_dispersion,
                    'neighbor_q': float(neighbor_q),
                    'neighbor_threshold_user': neighbor_threshold,
                })

            start = max(end + 1, start + 1)

    return pd.DataFrame(rows)


def summarize_runs(runs):
    if runs.empty:
        return pd.DataFrame(columns=[
            'neighbor_q', 'run_length_cap', 'run_length_mid',
            'future_consistency_mean', 'future_consistency_sem',
            'semantic_dispersion_mean', 'semantic_dispersion_sem',
            'runs',
        ])

    summary = (
        runs.groupby(['neighbor_q', 'run_length_cap'], observed=True)
        .agg(
            run_length_mid=('run_length_cap', 'first'),
            future_consistency_mean=('future_consistency', 'mean'),
            future_consistency_sem=('future_consistency', sem),
            semantic_dispersion_mean=('semantic_dispersion', 'mean'),
            semantic_dispersion_sem=('semantic_dispersion', sem),
            runs=('future_consistency', 'size'),
        )
        .reset_index()
        .sort_values(['neighbor_q', 'run_length_cap'])
    )
    return summary


def plot_runs(summary):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharex=True)
    colors = {
        0.30: '#1f77b4',
        0.50: '#ff7f0e',
        0.70: '#2ca02c',
    }
    panels = [
        (axes[0], 'future_consistency_mean', 'future_consistency_sem', 'Future Consistency'),
        (axes[1], 'semantic_dispersion_mean', 'semantic_dispersion_sem', 'Semantic Dispersion'),
    ]

    for ax, mean_col, sem_col, ylabel in panels:
        for neighbor_q in RUN_NEIGHBOR_QS:
            curve = summary[summary['neighbor_q'] == neighbor_q].sort_values('run_length_mid')
            if curve.empty:
                continue
            x = curve['run_length_mid'].to_numpy(dtype=float)
            y = curve[mean_col].to_numpy(dtype=float)
            y_sem = curve[sem_col].to_numpy(dtype=float)
            color = colors[neighbor_q]
            ax.errorbar(
                x,
                y,
                yerr=1.96 * y_sem,
                marker='o',
                linewidth=2.2,
                color=color,
                label=f'RUN_NEIGHBOR_Q={neighbor_q:.2f}',
            )
        ax.set_xlabel('Run length (10 = 10+)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle(
        f'Part 3: Run-Length Consolidation and Radius (K_FUTURE={K_FUTURE})',
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_run_length_vs_consolidation_and_radius.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if not events_path.exists() or not emb_path.exists():
        raise FileNotFoundError('Missing Part 1 caches. Run Part 1 first.')

    events = load_pickle(events_path)
    event_emb = np.load(emb_path)

    all_runs = []
    for neighbor_q in RUN_NEIGHBOR_QS:
        runs = build_run_table(events, event_emb, neighbor_q)
        all_runs.append(runs)
    runs = pd.concat(all_runs, ignore_index=True)
    summary = summarize_runs(runs)
    print(runs.head())
    print(summary)
    plot_runs(summary)


if __name__ == '__main__':
    main()
