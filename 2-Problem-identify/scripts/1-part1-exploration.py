from pathlib import Path
import ast
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
N_HISTORY = 10
N_FUTURE = 10
MIN_HISTORY = 3
MIN_FUTURE = 1
EXPLORE_Q = 0.80
EXPLOIT_Q = 0.20
RUN_NEIGHBOR_Q = 0.75
ANCHOR_SIM_THRESHOLD = 0.30
BATCH_SIZE = 256
SENSITIVITY_N_HISTORY = [5, 10, 20]
SENSITIVITY_N_FUTURE_VALUES = [5, 10, 20]

from _shared import CACHE_DIR, FIG_DIR, INTERMEDIATE_DIR, STEP1_DIR, STEP4_DIR


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

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError('Missing dependency: sentence-transformers.') from exc


def query_tokens_from_row(row):
    if 'keyword_tokens' in row and row['keyword_tokens'] is not None:
        tokens = parse_int_list(row['keyword_tokens'])
        if tokens:
            return tokens
    return parse_int_list(row['keyword'])


def build_events():
    vocab = load_pickle(STEP1_DIR / 'vocab_dict.pkl')
    rev_vocab = {v: k for k, v in vocab.items()}
    note_feat = load_pickle(STEP1_DIR / 'note_feat.pkl')
    caption_map = dict(zip(note_feat['note_id'].astype(int), note_feat['caption']))
    src_all = load_pickle(STEP4_DIR / 'src_all.pkl')
    rec_all = load_pickle(STEP4_DIR / 'rec_all.pkl')
    print('src_all', src_all.shape)
    print('rec_all', rec_all.shape)

    src_tmp = src_all.copy()
    src_tmp['_query_tokens'] = [query_tokens_from_row(row) for _, row in src_tmp.iterrows()]
    src_tmp['query_text'] = src_tmp['_query_tokens'].apply(lambda x: decode_tokens(x, rev_vocab))
    src_events = (
        src_tmp.sort_values(['user_id', 'timestamp'])
        .groupby(['user_id', 'search_session_id'], as_index=False)
        .agg(
            timestamp=('timestamp', 'min'),
            text=('query_text', 'first'),
            item_ids=('item_id', lambda x: list(map(int, x))),
            num_clicked_items=('item_id', 'size'),
        )
    )
    src_events['domain'] = 'S'
    src_events['event_id'] = 'S:' + src_events['user_id'].astype(str) + ':' + src_events['search_session_id'].astype(str)

    rec_tmp = rec_all.reset_index(drop=False).rename(columns={'index': 'row_id'}).copy()
    rec_tmp['text'] = rec_tmp['item_id'].apply(lambda i: decode_tokens(caption_map.get(int(i), []), rev_vocab))
    rec_events = rec_tmp[['user_id', 'timestamp', 'text', 'item_id', 'row_id']].copy()
    rec_events['domain'] = 'R'
    rec_events['event_id'] = 'R:' + rec_events['user_id'].astype(str) + ':' + rec_events['row_id'].astype(str)
    rec_events['search_session_id'] = np.nan
    rec_events['item_ids'] = rec_events['item_id'].apply(lambda x: [int(x)])
    rec_events['num_clicked_items'] = 1

    events = pd.concat(
        [
            src_events[['event_id', 'user_id', 'timestamp', 'domain', 'text', 'search_session_id', 'item_ids', 'num_clicked_items']],
            rec_events[['event_id', 'user_id', 'timestamp', 'domain', 'text', 'search_session_id', 'item_ids', 'num_clicked_items']],
        ],
        ignore_index=True,
    )
    events['text'] = events['text'].fillna('').astype(str).str.strip()
    events = events[events['text'] != ''].copy()
    events['_domain_order'] = events['domain'].map({'R': 0, 'S': 1}).fillna(2)
    events = events.sort_values(['user_id', 'timestamp', '_domain_order', 'event_id'], kind='mergesort').reset_index(drop=True)
    events['event_pos'] = events.groupby('user_id').cumcount()
    events = events.drop(columns=['_domain_order'])
    return events


def load_or_build_events():
    events_path = INTERMEDIATE_DIR / '1-events.pkl'
    if events_path.exists():
        events = load_pickle(events_path)
        print('Loaded cached events:', events_path, events.shape)
        return events
    events = build_events()
    dump_pickle(events, events_path)
    print('Saved cached events:', events_path)
    return events


def build_embeddings(events):
    model_slug = MODEL_NAME.replace('/', '__')
    emb_cache_path = CACHE_DIR / f'text_embedding_cache_{model_slug}.pkl'
    if emb_cache_path.exists():
        text_to_emb = load_pickle(emb_cache_path)
        print('Loaded cache:', emb_cache_path, 'texts:', len(text_to_emb))
    else:
        unique_texts = sorted(events['text'].unique())
        print('Encoding unique texts:', len(unique_texts))
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            unique_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        text_to_emb = dict(zip(unique_texts, embeddings))
        dump_pickle(text_to_emb, emb_cache_path)
        print('Saved cache:', emb_cache_path)
    event_emb = np.vstack([text_to_emb[t] for t in events['text']]).astype(np.float32)
    return l2_normalize(event_emb)


def load_or_build_embeddings(events):
    emb_path = INTERMEDIATE_DIR / '1-event-embeddings.npy'
    if emb_path.exists():
        event_emb = np.load(emb_path)
        if len(event_emb) == len(events):
            print('Loaded cached embeddings:', emb_path, event_emb.shape)
            return event_emb
        print('Embedding cache size mismatch; rebuilding:', emb_path, event_emb.shape, 'vs events', len(events))
    event_emb = build_embeddings(events)
    np.save(emb_path, event_emb)
    print('Saved cached embeddings:', emb_path)
    return event_emb


def build_scores(events, event_emb, n_history=N_HISTORY, n_future=N_FUTURE):
    rows = []
    for user_id, g in events.groupby('user_id', sort=False):
        idx = g.index.to_numpy()
        emb = event_emb[idx]
        domains = g['domain'].to_numpy()
        event_ids = g['event_id'].to_numpy()
        timestamps = g['timestamp'].to_numpy()
        texts = g['text'].to_numpy()
        m = len(g)
        for t in range(m):
            hist = emb[max(0, t - n_history):t]
            fut = emb[t + 1:t + 1 + n_future]
            history_similarity = np.nan
            exploration_score = np.nan
            if len(hist) >= MIN_HISTORY:
                hist_center = mean_unit(hist)
                history_similarity = float(np.dot(emb[t], hist_center))
                exploration_score = float(1.0 - history_similarity)
            future_consistency = np.nan
            if len(fut) >= MIN_FUTURE:
                fut_center = mean_unit(fut)
                future_consistency = float(np.dot(emb[t], fut_center))
            prev_domain = domains[t - 1] if t > 0 else 'START'
            rows.append({
                'user_id': int(user_id),
                'event_id': event_ids[t],
                'event_pos': int(t),
                'timestamp': float(timestamps[t]),
                'domain': domains[t],
                'prev_domain': prev_domain,
                'transition_into': f'{prev_domain}->{domains[t]}' if t > 0 else 'START',
                'text': texts[t],
                'history_count': int(len(hist)),
                'future_count': int(len(fut)),
                'history_similarity': history_similarity,
                'exploration_score': exploration_score,
                'future_consistency': future_consistency,
            })
    scores = pd.DataFrame(rows)

    anchor_cols = {
        'best_R_similarity': np.full(len(scores), np.nan, dtype=np.float32),
        'best_S_similarity': np.full(len(scores), np.nan, dtype=np.float32),
        'nearest_anchor_similarity': np.full(len(scores), np.nan, dtype=np.float32),
        'nearest_anchor_pos': np.full(len(scores), -1, dtype=np.int32),
        'nearest_anchor_domain': np.array([''] * len(scores), dtype=object),
    }
    for user_id, g in events.groupby('user_id', sort=False):
        event_idx = g.index.to_numpy()
        score_idx = scores.index[scores['user_id'].to_numpy() == user_id].to_numpy()
        emb = event_emb[event_idx]
        domains = g['domain'].to_numpy()
        m = len(g)
        if m < 2:
            continue
        sim = emb @ emb.T
        for t in range(1, m):
            prior = np.arange(t)
            r_prior = prior[domains[:t] == 'R']
            s_prior = prior[domains[:t] == 'S']
            best_r = np.nan
            best_s = np.nan
            best_r_pos = -1
            best_s_pos = -1
            if len(r_prior):
                vals = sim[t, r_prior]
                j = int(np.argmax(vals))
                best_r = float(vals[j])
                best_r_pos = int(r_prior[j])
            if len(s_prior):
                vals = sim[t, s_prior]
                j = int(np.argmax(vals))
                best_s = float(vals[j])
                best_s_pos = int(s_prior[j])
            row_i = score_idx[t]
            anchor_cols['best_R_similarity'][row_i] = best_r
            anchor_cols['best_S_similarity'][row_i] = best_s
            if np.isnan(best_r) and np.isnan(best_s):
                continue
            if np.isnan(best_s) or (not np.isnan(best_r) and best_r >= best_s):
                anchor_cols['nearest_anchor_similarity'][row_i] = best_r
                anchor_cols['nearest_anchor_pos'][row_i] = best_r_pos
                anchor_cols['nearest_anchor_domain'][row_i] = 'R'
            else:
                anchor_cols['nearest_anchor_similarity'][row_i] = best_s
                anchor_cols['nearest_anchor_pos'][row_i] = best_s_pos
                anchor_cols['nearest_anchor_domain'][row_i] = 'S'
    for col, values in anchor_cols.items():
        scores[col] = values

    scores['anchor_transition'] = np.where(
        scores['nearest_anchor_domain'].isin(['R', 'S']),
        scores['nearest_anchor_domain'] + '->' + scores['domain'],
        'NO_ANCHOR',
    )
    scores['anchor_shift'] = 1.0 - scores['nearest_anchor_similarity']
    scores['anchor_kept'] = scores['nearest_anchor_similarity'] >= ANCHOR_SIM_THRESHOLD
    scores['anchor_transition_kept'] = np.where(scores['anchor_kept'], scores['anchor_transition'], 'LOW_ANCHOR')

    valid = scores['exploration_score'].notna()
    scores.loc[valid, 'exploration_pct_user'] = scores[valid].groupby('user_id')['exploration_score'].rank(pct=True, method='average')
    scores['episode_type'] = 'neutral'
    scores.loc[scores['exploration_pct_user'] >= EXPLORE_Q, 'episode_type'] = 'exploration'
    scores.loc[scores['exploration_pct_user'] <= EXPLOIT_Q, 'episode_type'] = 'exploitation'
    scores['episode_type_top10'] = 'neutral'
    scores.loc[scores['exploration_pct_user'] >= 0.90, 'episode_type_top10'] = 'exploration_top10'
    scores.loc[scores['exploration_pct_user'] <= 0.10, 'episode_type_top10'] = 'exploitation_top10'

    scores['center_exploration_adoption'] = scores['future_consistency'] - scores['history_similarity']
    scores['anchor_exploration_adoption'] = scores['future_consistency'] - scores['nearest_anchor_similarity']
    scores['successful_anchor_exploration'] = (
        (scores['episode_type'] == 'exploration')
        & scores['anchor_kept']
        & scores['anchor_exploration_adoption'].notna()
        & (scores['anchor_exploration_adoption'] > 0)
    )
    scores['successful_anchor_exploration_top10'] = (
        (scores['episode_type_top10'] == 'exploration_top10')
        & scores['anchor_kept']
        & scores['anchor_exploration_adoption'].notna()
        & (scores['anchor_exploration_adoption'] > 0)
    )
    valid_adoption = scores['anchor_exploration_adoption'].notna()
    scores.loc[valid_adoption, 'anchor_adoption_pct_user'] = scores[valid_adoption].groupby('user_id')['anchor_exploration_adoption'].rank(pct=True, method='average')
    return scores


def summarize_part1(scores):
    plot_df = scores.dropna(subset=['exploration_pct_user', 'future_consistency']).copy()
    plot_df['exploration_pct_bin'] = pd.cut(plot_df['exploration_pct_user'], bins=np.linspace(0, 1, 11), include_lowest=True)
    return (
        plot_df.groupby('exploration_pct_bin', observed=True)
        .agg(
            exploration_pct_mid=('exploration_pct_user', 'mean'),
            future_consistency_mean=('future_consistency', 'mean'),
            future_consistency_sem=('future_consistency', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
            events=('future_consistency', 'size'),
        )
        .reset_index()
    )


def plot_part1_sensitivity(history_curves, future_curves):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    palette = plt.get_cmap('tab10')

    def draw_panel(ax, curves, title, ylabel):
        for i, curve in enumerate(curves):
            color = palette(i % 10)
            ax.errorbar(
                curve['summary']['exploration_pct_mid'],
                curve['summary']['future_consistency_mean'],
                yerr=1.96 * curve['summary']['future_consistency_sem'],
                marker='o',
                linewidth=2,
                color=color,
                label=curve['label'],
            )
        ax.set_title(title)
        ax.set_xlabel('Within-user Exploration Score Percentile')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, title='Variant')

    draw_panel(
        axes[0],
        history_curves,
        f'Fixed N_FUTURE={N_FUTURE}, vary N_HISTORY',
        f'Future Consistency',
    )
    draw_panel(
        axes[1],
        future_curves,
        f'Fixed N_HISTORY={N_HISTORY}, vary N_FUTURE',
        f'Future Consistency',
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_exploration_vs_future_consistency_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    events = load_or_build_events()
    event_emb = load_or_build_embeddings(events)
    if 'embedding_row' not in events.columns:
        events['embedding_row'] = np.arange(len(events))

    print(f'\nPart1 default variant: N_HISTORY={N_HISTORY}, N_FUTURE={N_FUTURE}')
    default_scores = build_scores(events, event_emb, n_history=N_HISTORY, n_future=N_FUTURE)
    dump_pickle(default_scores, INTERMEDIATE_DIR / '1-exploration-scores.pkl')
    print('scores', default_scores.shape)
    print(default_scores['episode_type'].value_counts(dropna=False))

    history_summaries = {N_HISTORY: summarize_part1(default_scores)}
    for n_history in SENSITIVITY_N_HISTORY:
        print(f'\nPart1 sensitivity variant: N_HISTORY={n_history}, N_FUTURE={N_FUTURE}')
        scores = default_scores if n_history == N_HISTORY else build_scores(events, event_emb, n_history=n_history, n_future=N_FUTURE)
        history_summaries[n_history] = summarize_part1(scores)
        print('scores', scores.shape)
        print(scores['episode_type'].value_counts(dropna=False))
    history_curves = [{'label': f'N={n}', 'summary': history_summaries[n]} for n in [5, 10, 20]]

    future_summaries = {N_FUTURE: summarize_part1(default_scores)}
    for n_future in SENSITIVITY_N_FUTURE_VALUES:
        print(f'\nPart1 sensitivity variant: N_HISTORY={N_HISTORY}, N_FUTURE={n_future}')
        scores = default_scores if n_future == N_FUTURE else build_scores(events, event_emb, n_history=N_HISTORY, n_future=n_future)
        future_summaries[n_future] = summarize_part1(scores)
        print('scores', scores.shape)
        print(scores['episode_type'].value_counts(dropna=False))
    future_curves = [{'label': f'N={n}', 'summary': future_summaries[n]} for n in [5, 10, 20]]

    plot_part1_sensitivity(history_curves, future_curves)

    print('\nevents', events.shape)
    print(events['domain'].value_counts())
    print('event_emb', event_emb.shape)
    print('saved Part1 intermediates in', INTERMEDIATE_DIR)


if __name__ == '__main__':
    main()
