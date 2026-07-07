from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N_HISTORY = 10
MIN_HISTORY = 5
TOP_K = 1
N_BINS = 42
DOMAIN_LABELS = {
    'R': 'Recommend',
    'S': 'Search',
}
DOMAIN_COLORS = {
    'R': '#2f6f9f',
    'S': '#d95f59',
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SCRIPT_DIR.parent
INTERMEDIATE_DIR = PROBLEM_DIR / 'intermediate'
FIG_DIR = PROBLEM_DIR / 'figures'

from _shared import ensure_part1_artifacts


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def dump_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def topk_mean(values, k=TOP_K):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    k = min(int(k), len(values))
    return float(np.partition(values, -k)[-k:].mean())


def build_domain_topk_delta(events, event_emb, n_history=N_HISTORY, min_history=MIN_HISTORY):
    rows = []

    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
        domains = g['domain'].to_numpy()
        event_ids = g['event_id'].to_numpy()
        event_pos = g['event_pos'].to_numpy() if 'event_pos' in g.columns else np.arange(len(g))
        timestamps = g['timestamp'].to_numpy()

        for t in range(len(g)):
            start = max(0, t - n_history)
            hist = emb[start:t]
            hist_domains = domains[start:t]
            if len(hist) < min_history:
                continue

            similarities = hist @ emb[t]
            search_sims = similarities[hist_domains == 'S']
            rec_sims = similarities[hist_domains == 'R']
            if len(search_sims) == 0 or len(rec_sims) == 0:
                continue

            s_s = topk_mean(search_sims)
            s_r = topk_mean(rec_sims)
            rows.append({
                'user_id': int(user_id),
                'event_id': event_ids[t],
                'event_pos': int(event_pos[t]),
                'timestamp': float(timestamps[t]),
                'domain': domains[t],
                'domain_label': DOMAIN_LABELS.get(domains[t], domains[t]),
                'history_count': int(len(hist)),
                'search_history_count': int(len(search_sims)),
                'recommend_history_count': int(len(rec_sims)),
                'S_s': s_s,
                'S_r': s_r,
                'delta_search_minus_recommend': float(s_s - s_r),
            })

    return pd.DataFrame(rows)


def summarize_delta(delta_df):
    if delta_df.empty:
        return pd.DataFrame(columns=[
            'domain', 'domain_label', 'events', 'mean_delta', 'median_delta',
            'std_delta', 'p25_delta', 'p75_delta', 'share_search_higher',
        ])

    work = delta_df.copy()
    work['search_higher'] = work['delta_search_minus_recommend'] > 0
    summary = (
        work.groupby(['domain', 'domain_label'], observed=True)
        .agg(
            events=('event_id', 'size'),
            mean_delta=('delta_search_minus_recommend', 'mean'),
            median_delta=('delta_search_minus_recommend', 'median'),
            std_delta=('delta_search_minus_recommend', 'std'),
            p25_delta=('delta_search_minus_recommend', lambda x: x.quantile(0.25)),
            p75_delta=('delta_search_minus_recommend', lambda x: x.quantile(0.75)),
            share_search_higher=('search_higher', 'mean'),
        )
        .reset_index()
        .sort_values('domain')
    )
    return summary


def plot_delta_distribution(delta_df):
    import matplotlib.pyplot as plt

    plot_df = delta_df.dropna(subset=['delta_search_minus_recommend']).copy()
    if plot_df.empty:
        raise ValueError('No delta rows to plot.')

    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'semibold',
        'font.size': 11,
    })

    values = plot_df['delta_search_minus_recommend'].to_numpy(dtype=float)
    x_low, x_high = np.quantile(values, [0.005, 0.995])
    x_abs = max(abs(float(x_low)), abs(float(x_high)), 0.2)
    x_abs = min(1.05, x_abs * 1.08)
    bins = np.linspace(-x_abs, x_abs, N_BINS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=True, sharey=True)
    for ax, domain in zip(axes, ['R', 'S']):
        sub = plot_df[plot_df['domain'] == domain]
        if sub.empty:
            ax.set_visible(False)
            continue

        color = DOMAIN_COLORS[domain]
        label = DOMAIN_LABELS[domain]
        vals = sub['delta_search_minus_recommend'].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        share_positive = float(np.mean(vals > 0))

        ax.hist(
            vals,
            bins=bins,
            density=True,
            color=color,
            alpha=0.78,
            edgecolor='white',
            linewidth=0.8,
        )
        ax.axvline(0, color='#202020', linewidth=1.8, alpha=0.85, label='No difference')
        ax.axvline(mean, color='#202020', linestyle='--', linewidth=1.6, alpha=0.78, label=f'Mean {mean:.3f}')
        ax.axvline(median, color='#202020', linewidth=2.0, label=f'Median {median:.3f}')
        ax.text(
            0.04,
            0.94,
            f'n = {len(sub):,}\nmean = {mean:.3f}\nmedian = {median:.3f}\nP(delta > 0) = {share_positive:.3f}',
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.86, edgecolor='none'),
        )
        ax.set_title(f'{label} current')
        ax.set_xlabel('Delta = top-1 similarity to search history - recommendation history')
        ax.set_xlim(-x_abs, x_abs)
        ax.grid(axis='y', alpha=0.22)
        ax.legend(frameon=False, loc='upper right')

    axes[0].set_ylabel('Density')
    fig.suptitle(
        (
            f'Figure 5b: Domain-Specific Historical Pull Delta\n'
            f'Last {N_HISTORY} history events, top-{TOP_K} raw cosine similarity, both history types present'
        ),
        y=1.03,
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.01,
        'Positive delta means the current interaction is closer to search history; negative delta means closer to recommendation history.',
        ha='center',
        va='top',
        fontsize=10,
        color='#555555',
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5b_history_domain_top1_delta_distribution.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if not events_path.exists() or not emb_path.exists():
        raise FileNotFoundError('Missing Part 1 caches. Run Part 1 first.')

    events = load_pickle(events_path)
    event_emb = np.load(emb_path)
    delta_df = build_domain_topk_delta(events, event_emb)
    summary = summarize_delta(delta_df)

    dump_pickle(delta_df, INTERMEDIATE_DIR / '2-history-domain-top1-delta.pkl')
    summary.to_csv(INTERMEDIATE_DIR / '2-history-domain-top1-delta-summary.csv', index=False)
    plot_delta_distribution(delta_df)

    print(summary)
    print('delta rows', delta_df.shape)
    print('saved', FIG_DIR / 'fig5b_history_domain_top1_delta_distribution.png')


if __name__ == '__main__':
    main()
