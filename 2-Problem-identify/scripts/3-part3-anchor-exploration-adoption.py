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
ANCHOR_SIM_THRESHOLD = 0.30
SENSITIVITY_ANCHOR_SIM_THRESHOLD = 0.50
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
    anchor_transitions = scores[
        anchor_kept
        & scores['anchor_transition'].isin(ANCHOR_ORDER)
        & scores['nearest_anchor_similarity'].notna()
    ].copy()
    anchor_transitions['anchor_kept'] = True
    anchor_transitions['anchor_transition_kept'] = anchor_transitions['anchor_transition']
    anchor_transitions = anchor_transitions[[
        'user_id', 'event_id', 'event_pos', 'timestamp', 'domain', 'anchor_transition',
        'nearest_anchor_domain', 'nearest_anchor_pos', 'nearest_anchor_similarity', 'anchor_kept', 'anchor_shift',
        'history_similarity', 'future_consistency', 'exploration_score', 'exploration_pct_user',
        'episode_type', 'episode_type_top10', 'anchor_exploration_adoption',
        'successful_anchor_exploration', 'successful_anchor_exploration_top10', 'text'
    ]]
    explore20 = anchor_transitions[(anchor_transitions['episode_type'] == 'exploration') & anchor_transitions['anchor_exploration_adoption'].notna()].copy()
    explore10 = anchor_transitions[(anchor_transitions['episode_type_top10'] == 'exploration_top10') & anchor_transitions['anchor_exploration_adoption'].notna()].copy()
    adoption20 = (
        explore20.groupby('anchor_transition')
        .agg(
            exploration_events=('event_id', 'size'),
            users=('user_id', 'nunique'),
            adoption_rate=('successful_anchor_exploration', 'mean'),
            adoption_score_mean=('anchor_exploration_adoption', 'mean'),
            future_consistency_mean=('future_consistency', 'mean'),
            nearest_anchor_similarity_mean=('nearest_anchor_similarity', 'mean'),
        )
        .reindex(ANCHOR_ORDER)
    )
    adoption10 = (
        explore10.groupby('anchor_transition')
        .agg(
            exploration_events_top10=('event_id', 'size'),
            adoption_rate_top10=('successful_anchor_exploration_top10', 'mean'),
            adoption_score_mean_top10=('anchor_exploration_adoption', 'mean'),
        )
        .reindex(ANCHOR_ORDER)
    )
    adoption_stats = pd.concat([adoption20, adoption10], axis=1).reset_index()
    x = np.arange(len(adoption_stats))
    width = 0.36
    plt.figure(figsize=(8, 4.6))
    plt.bar(x - width / 2, adoption_stats['adoption_rate'], width, label='Top 20% exploration')
    plt.bar(x + width / 2, adoption_stats['adoption_rate_top10'], width, label='Top 10% exploration')
    plt.xticks(x, adoption_stats['anchor_transition'])
    plt.ylabel('Successful anchor exploration rate')
    plt.xlabel('Nearest historical semantic anchor -> current domain')
    plt.title(f'Anchor-Based Exploration Adoption (sim >= {threshold})')
    plt.ylim(0, max(0.05, float(adoption_stats[['adoption_rate', 'adoption_rate_top10']].max().max()) * 1.2))
    plt.legend(frameon=False)
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig4_exploration_adoption_by_transition{suffix}.png', dpi=200)
    if write_default:
        plt.savefig(FIG_DIR / 'fig4_exploration_adoption_by_transition.png', dpi=200)
    plt.close()

    print(f'\nPart3 variant: ANCHOR_SIM_THRESHOLD={threshold}')
    print('anchor_transitions', anchor_transitions.shape)
    print(adoption_stats)
    return anchor_transitions, adoption_stats


def main():
    ensure_part1_artifacts()
    scores = load_scores()
    variants = [
        (ANCHOR_SIM_THRESHOLD, f'_ANCHOR_SIM_THRESHOLD{ANCHOR_SIM_THRESHOLD:.2f}', True),
        (SENSITIVITY_ANCHOR_SIM_THRESHOLD, f'_ANCHOR_SIM_THRESHOLD{SENSITIVITY_ANCHOR_SIM_THRESHOLD:.2f}', False),
    ]
    for threshold, suffix, is_default in variants:
        run_variant(scores, threshold, suffix, write_default=is_default)


if __name__ == '__main__':
    main()
