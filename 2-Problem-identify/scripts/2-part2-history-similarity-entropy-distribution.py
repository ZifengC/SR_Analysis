from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N_HISTORY = 10
MIN_HISTORY = 5
TOP_K = 5
N_BINS = 34
SOFTMAX_TEMPERATURE = 0.25
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


def similarity_topk_mass(similarities, top_k=TOP_K):
    similarities = np.asarray(similarities, dtype=np.float64)
    logits = similarities / SOFTMAX_TEMPERATURE
    logits = logits - np.nanmax(logits)
    weights = np.exp(logits)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.nan

    top_k = min(int(top_k), len(weights))
    top_weights = np.partition(weights, -top_k)[-top_k:]
    return float(top_weights.sum() / weight_sum)


def build_history_similarity_topk_mass(events, event_emb, n_history=N_HISTORY, min_history=MIN_HISTORY):
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
            if len(hist) < min_history:
                continue

            similarities = hist @ emb[t]
            topk_mass = similarity_topk_mass(similarities)
            rows.append({
                'user_id': int(user_id),
                'event_id': event_ids[t],
                'event_pos': int(event_pos[t]),
                'timestamp': float(timestamps[t]),
                'domain': domains[t],
                'domain_label': DOMAIN_LABELS.get(domains[t], domains[t]),
                'text': texts[t],
                'history_count': int(len(hist)),
                'similarity_top5_mass': topk_mass,
                'history_similarity_mean': float(np.mean(similarities)),
                'history_similarity_std': float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0,
                'history_similarity_min': float(np.min(similarities)),
                'history_similarity_max': float(np.max(similarities)),
            })

    return pd.DataFrame(rows)


def summarize_topk_mass(topk_df):
    if topk_df.empty:
        return pd.DataFrame(columns=[
            'domain', 'domain_label', 'events', 'mean', 'median', 'std', 'p10', 'p25', 'p75', 'p90',
        ])

    summary = (
        topk_df.groupby(['domain', 'domain_label'], observed=True)['similarity_top5_mass']
        .agg(
            events='size',
            mean='mean',
            median='median',
            std='std',
            p10=lambda x: x.quantile(0.10),
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
        .sort_values('domain')
    )
    return summary


def plot_topk_mass_distribution(topk_df, summary):
    import matplotlib.pyplot as plt

    plot_df = topk_df.dropna(subset=['similarity_top5_mass']).copy()
    if plot_df.empty:
        raise ValueError('No top-k mass rows to plot.')

    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'semibold',
        'font.size': 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), sharex=True, sharey=True)
    all_values = plot_df['similarity_top5_mass'].to_numpy(dtype=float)
    x_low, x_high = np.quantile(all_values, [0.005, 0.995])
    x_pad = max((x_high - x_low) * 0.08, 0.001)
    x_low = max(0.0, float(x_low - x_pad))
    x_high = min(1.0, float(x_high + x_pad))
    if x_high - x_low < 0.01:
        center = (x_low + x_high) / 2.0
        x_low = max(0.0, center - 0.005)
        x_high = min(1.0, center + 0.005)
    bins = np.linspace(x_low, x_high, N_BINS + 1)

    for ax, domain in zip(axes, ['R', 'S']):
        sub = plot_df[plot_df['domain'] == domain]
        color = DOMAIN_COLORS[domain]
        label = DOMAIN_LABELS[domain]
        if sub.empty:
            ax.set_visible(False)
            continue

        values = sub['similarity_top5_mass'].to_numpy(dtype=float)
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=color,
            alpha=0.78,
            edgecolor='white',
            linewidth=0.8,
        )

        med = float(np.median(values))
        mean = float(np.mean(values))
        ax.axvline(med, color='#202020', linewidth=2.0, label=f'Median {med:.3f}')
        ax.axvline(mean, color='#202020', linestyle='--', linewidth=1.6, alpha=0.78, label=f'Mean {mean:.3f}')
        ax.text(
            0.97,
            0.94,
            f'n = {len(sub):,}\nmean = {mean:.3f}\nmedian = {med:.3f}',
            transform=ax.transAxes,
            va='top',
            ha='right',
            fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.86, edgecolor='none'),
        )
        ax.set_title(label)
        ax.set_xlabel('Top-5 mass of current-to-history similarities')
        ax.set_xlim(x_low, x_high)
        ax.grid(axis='y', alpha=0.22)
        ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.74))

    axes[0].set_ylabel('Density')
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.88, wspace=0.18)
    fig.savefig(FIG_DIR / 'fig4_history_similarity_top5_mass_distribution.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    summary.to_csv(INTERMEDIATE_DIR / '2-history-similarity-top5-mass-summary.csv', index=False)


def main():
    ensure_part1_artifacts()
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if not events_path.exists() or not emb_path.exists():
        raise FileNotFoundError('Missing Part 1 caches. Run Part 1 first.')

    events = load_pickle(events_path)
    event_emb = np.load(emb_path)
    topk_df = build_history_similarity_topk_mass(events, event_emb)
    summary = summarize_topk_mass(topk_df)

    dump_pickle(topk_df, INTERMEDIATE_DIR / '2-history-similarity-top5-mass.pkl')
    summary.to_csv(INTERMEDIATE_DIR / '2-history-similarity-top5-mass-summary.csv', index=False)
    plot_topk_mass_distribution(topk_df, summary)

    print(summary)
    print('top5 mass rows', topk_df.shape)
    print('saved', FIG_DIR / 'fig4_history_similarity_top5_mass_distribution.png')


if __name__ == '__main__':
    main()
