from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N_HISTORY = 10
MIN_HISTORY = 5
TOP_K = 1
SCATTER_SAMPLE_SIZE = 120000
DOMAIN_LABELS = {
    'R': 'Recommend',
    'S': 'Search',
}
DOMAIN_COLORS = {
    'R': '#2f6f9f',
    'S': '#d95f59',
}
FIT_COLORS = {
    'R': '#17435f',
    'S': '#9f2f2d',
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


def build_domain_topk_similarity(events, event_emb, n_history=N_HISTORY, min_history=MIN_HISTORY):
    rows = []

    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
        domains = g['domain'].to_numpy()
        event_ids = g['event_id'].to_numpy()
        event_pos = g['event_pos'].to_numpy() if 'event_pos' in g.columns else np.arange(len(g))
        timestamps = g['timestamp'].to_numpy()
        texts = g['text'].to_numpy()

        for t in range(len(g)):
            start = max(0, t - n_history)
            hist = emb[start:t]
            hist_domains = domains[start:t]
            if len(hist) < min_history:
                continue

            similarities = hist @ emb[t]
            search_sims = similarities[hist_domains == 'S']
            rec_sims = similarities[hist_domains == 'R']

            rows.append({
                'user_id': int(user_id),
                'event_id': event_ids[t],
                'event_pos': int(event_pos[t]),
                'timestamp': float(timestamps[t]),
                'domain': domains[t],
                'domain_label': DOMAIN_LABELS.get(domains[t], domains[t]),
                'text': texts[t],
                'history_count': int(len(hist)),
                'search_history_count': int(len(search_sims)),
                'recommend_history_count': int(len(rec_sims)),
                'S_s': topk_mean(search_sims),
                'S_r': topk_mean(rec_sims),
                'S_s_mean': float(np.mean(search_sims)) if len(search_sims) else np.nan,
                'S_r_mean': float(np.mean(rec_sims)) if len(rec_sims) else np.nan,
            })

    return pd.DataFrame(rows)


def filter_has_both_history(scatter_df):
    return scatter_df[
        (scatter_df['search_history_count'] > 0)
        & (scatter_df['recommend_history_count'] > 0)
    ].copy()


def summarize_domain_topk(scatter_df):
    filtered = filter_has_both_history(scatter_df)
    if filtered.empty:
        return pd.DataFrame(columns=[
            'domain', 'domain_label', 'events', 'S_s_mean', 'S_s_median', 'S_r_mean', 'S_r_median',
            'search_history_share', 'recommend_history_share',
        ])

    work = filtered.copy()
    work['has_search_history'] = work['search_history_count'] > 0
    work['has_recommend_history'] = work['recommend_history_count'] > 0
    summary = (
        work.groupby(['domain', 'domain_label'], observed=True)
        .agg(
            events=('event_id', 'size'),
            S_s_mean=('S_s', 'mean'),
            S_s_median=('S_s', 'median'),
            S_r_mean=('S_r', 'mean'),
            S_r_median=('S_r', 'median'),
            search_history_share=('has_search_history', 'mean'),
            recommend_history_share=('has_recommend_history', 'mean'),
        )
        .reset_index()
        .sort_values('domain')
    )
    return summary


def plot_domain_topk_scatter(scatter_df):
    import matplotlib.pyplot as plt

    filtered = filter_has_both_history(scatter_df)
    plot_df = filtered.dropna(subset=['S_s', 'S_r']).copy()
    if plot_df.empty:
        raise ValueError('No domain top-k rows to plot.')

    if len(plot_df) > SCATTER_SAMPLE_SIZE:
        plot_df = plot_df.sample(SCATTER_SAMPLE_SIZE, random_state=0)

    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'semibold',
        'font.size': 11,
    })

    x_max = float(filtered['S_r'].quantile(0.995))
    y_max = float(filtered['S_s'].quantile(0.995))
    axis_max = max(x_max, y_max, 0.25)
    axis_pad = max(axis_max * 0.06, 0.08)
    axis_max = axis_max + axis_pad

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharex=True, sharey=True)
    x_line = np.linspace(0.0, axis_max, 250)

    for ax, domain in zip(axes, ['R', 'S']):
        sub = plot_df[plot_df['domain'] == domain]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(
            sub['S_r'],
            sub['S_s'],
            s=8,
            alpha=0.16,
            color=DOMAIN_COLORS[domain],
            edgecolors='none',
            rasterized=True,
        )

        full_sub = filtered[filtered['domain'] == domain]
        x = full_sub['S_r'].to_numpy(dtype=float)
        y = full_sub['S_s'].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        fit_note = ''
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=FIT_COLORS[domain],
                linewidth=2.4,
                alpha=0.95,
            )
            fit_note = f'fit: S_s = {slope:.2f} S_r + {intercept:.2f}'

        ax.axline((0, 0), slope=1, color='#202020', linestyle='--', linewidth=1.1, alpha=0.55)
        ax.axvline(0, color='#202020', linewidth=0.9, alpha=0.35)
        ax.axhline(0, color='#202020', linewidth=0.9, alpha=0.35)
        ax.set_xlim(-0.05, axis_max)
        ax.set_ylim(-0.05, axis_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.22)
        ax.set_xlabel(f'S_r: top-{TOP_K} similarity to recommendation history')
        ax.set_title(DOMAIN_LABELS[domain], fontsize=13)

        note = f'n = {len(full_sub):,}'
        if fit_note:
            note = f'{note}\n{fit_note}'
        ax.text(
            0.04,
            0.96,
            note,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.86, edgecolor='none'),
        )

    axes[0].set_ylabel(f'S_s: top-{TOP_K} similarity to search history')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_history_domain_top1_similarity_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_part1_artifacts()
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if not events_path.exists() or not emb_path.exists():
        raise FileNotFoundError('Missing Part 1 caches. Run Part 1 first.')

    events = load_pickle(events_path)
    event_emb = np.load(emb_path)
    scatter_df = build_domain_topk_similarity(events, event_emb)
    summary = summarize_domain_topk(scatter_df)

    dump_pickle(scatter_df, INTERMEDIATE_DIR / '2-history-domain-top1-similarity-scatter.pkl')
    summary.to_csv(INTERMEDIATE_DIR / '2-history-domain-top1-similarity-scatter-summary.csv', index=False)
    plot_domain_topk_scatter(scatter_df)

    print(summary)
    print('scatter rows', scatter_df.shape)
    print('saved', FIG_DIR / 'fig5_history_domain_top1_similarity_scatter.png')


if __name__ == '__main__':
    main()
