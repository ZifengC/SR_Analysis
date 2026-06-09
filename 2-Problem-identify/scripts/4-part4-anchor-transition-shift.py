from pathlib import Path
import ast
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
N_HISTORY = 10
K_FUTURE = 5
MIN_HISTORY = 3
MIN_FUTURE = 1
EXPLORE_Q = 0.80
EXPLOIT_Q = 0.20
RUN_NEIGHBOR_Q = 0.75
ANCHOR_SIM_THRESHOLD = 0.70
SENSITIVITY_ANCHOR_SIM_THRESHOLD = 0.20
BATCH_SIZE = 256

from _shared import CACHE_DIR, FIG_DIR, INTERMEDIATE_DIR, STEP1_DIR, STEP4_DIR, ensure_part1_artifacts


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def dump_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    try:
        return [int(value)]
    except Exception:
        return []


def decode_tokens(tokens, rev_vocab):
    return ''.join(rev_vocab.get(int(t), '') for t in tokens if int(t) != 0)


def l2_normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def mean_unit(vectors):
    center = vectors.mean(axis=0, keepdims=True)
    return l2_normalize(center)[0]

import matplotlib.pyplot as plt

ANCHOR_ORDER = ['R->R', 'R->S', 'S->R', 'S->S']


def load_scores():
    return load_pickle(INTERMEDIATE_DIR / '1-exploration-scores.pkl')


def run_variant(scores, threshold, suffix, write_default=False):
    anchor_kept = scores['nearest_anchor_similarity'] >= threshold
    plot_anchor = scores[
        anchor_kept
        & scores['anchor_transition'].isin(ANCHOR_ORDER)
        & scores['nearest_anchor_similarity'].notna()
    ].copy()

    fig4 = (
        plot_anchor.groupby('anchor_transition')
        .agg(
            events=('anchor_shift', 'size'),
            users=('user_id', 'nunique'),
            anchor_shift_mean=('anchor_shift', 'mean'),
            anchor_shift_median=('anchor_shift', 'median'),
            nearest_anchor_similarity_mean=('nearest_anchor_similarity', 'mean'),
            exploration_rate=('episode_type', lambda x: (x == 'exploration').mean()),
            top10_exploration_rate=('episode_type_top10', lambda x: (x == 'exploration_top10').mean()),
        )
        .reindex(ANCHOR_ORDER)
        .reset_index()
    )
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(fig4['anchor_transition'], fig4['exploration_rate'])
    plt.ylabel('Exploration rate')
    plt.xlabel('Anchor transition type')
    plt.title(f'Exploration Rate by Anchor Transition (sim >= {threshold})')
    plt.ylim(0, max(0.05, float(fig4['exploration_rate'].max()) * 1.2))
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig4_exploration_rate_by_anchor_transition{suffix}.png', dpi=200)
    if write_default:
        plt.savefig(FIG_DIR / 'fig4_exploration_rate_by_anchor_transition.png', dpi=200)
    plt.close()

    print(f'\nPart4 variant: ANCHOR_SIM_THRESHOLD={threshold}')
    print(fig4)
    return fig4


def main():
    ensure_part1_artifacts()
    scores = load_scores()
    variants = [
        (ANCHOR_SIM_THRESHOLD, f'_ANCHOR_SIM_THRESHOLD{ANCHOR_SIM_THRESHOLD:.2f}', True),
        (SENSITIVITY_ANCHOR_SIM_THRESHOLD, f'_ANCHOR_SIM_THRESHOLD{SENSITIVITY_ANCHOR_SIM_THRESHOLD:.2f}', False),
    ]
    baseline_summary = None
    for threshold, suffix, is_default in variants:
        fig4 = run_variant(scores, threshold, suffix, write_default=is_default)
        if is_default:
            baseline_summary = fig4

    episode_summary = (
        scores.dropna(subset=['exploration_score'])
        .groupby(['domain', 'episode_type'])
        .agg(
            events=('event_id', 'size'),
            users=('user_id', 'nunique'),
            exploration_score_mean=('exploration_score', 'mean'),
            future_consistency_mean=('future_consistency', 'mean'),
            adoption_score_mean=('anchor_exploration_adoption', 'mean'),
            successful_anchor_exploration_rate=('successful_anchor_exploration', 'mean'),
        )
        .reset_index()
    )
    print('\nCore CSV outputs:')
    print(' - none; figures only, plus Part1 pickle inputs')
    print('\nSensitivity PNG outputs include parameter suffixes such as _ANCHOR_SIM_THRESHOLD0.20.')
    print('\nEpisode summary:')
    print(episode_summary)


if __name__ == '__main__':
    main()
