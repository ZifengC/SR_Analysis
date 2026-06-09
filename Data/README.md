# Data README

This `Data/` directory contains the processed Qilin search-recommendation user
behavior dataset used by the analysis pipeline. The data is not raw logs. It is
already re-indexed, tokenized, split, and converted into model-ready pickle
files.

## Directory Layout

```text
Data/
  Step1/
    exposure_src.pkl
    exposure_rec.pkl
    note_feat.pkl
    user_feat.pkl
    vocab_dict.pkl
    note_dict.pkl
    user_dict.pkl

  dataset/
    src_train.pkl
    src_val.pkl
    src_test.pkl
    rec_train.pkl
    rec_val.pkl
    rec_test.pkl

  vocab/
    query_vocab.pkl
    src_session_vocab_np.pkl
    item_vocab_np.pkl
    user_vocab.pkl
    user_vocab_np.pkl
```

## Core Concept

The dataset has two behavior domains:

- `src` / source / search: user behavior generated from search sessions.
- `rec` / recommendation: user behavior generated from recommendation feed
  exposure.

Keep two levels separate:

- Session/exposure level: one row represents a search session or recommendation
  exposure containing multiple exposed items.
- Sample level: one row represents one clicked positive item with negative
  candidate items for model training/evaluation.

This distinction matters. A search session/query can contain many item-level
click samples.

## Step1: Session / Exposure Level

`Step1/exposure_src.pkl` is search session level.

Columns:

```text
user_idx
query
sample_type
note_set
label
timestamp
src_session_id
query_token
```

Meaning:

- `sample_type = 0`: search/source domain.
- `user_idx`: re-indexed user id.
- `query`: raw search query text.
- `query_token`: tokenized query.
- `src_session_id`: global search session id. This is a session id, not an item
  id and not a query id.
- `note_set`: list of item/note ids exposed in this search session.
- `label`: click labels aligned with `note_set`; `1` means clicked and `0`
  means not clicked.
- `timestamp`: item-level timestamps aligned with `note_set`.

`Step1/exposure_rec.pkl` is recommendation exposure level.

Columns:

```text
user_idx
query
sample_type
note_set
label
timestamp
```

Meaning:

- `sample_type = 1`: recommendation domain.
- `note_set`, `label`, and `timestamp` are aligned item-level lists inside one
  recommendation exposure.
- `query` is recommendation context text/history text, not a search query in the
  source-domain sense.

`Step1/note_feat.pkl` stores item features:

```text
note_id, caption, note_type, like_num
```

`caption` is tokenized note title/caption. Use `vocab_dict.pkl` to decode token
ids back to text.

`Step1/user_feat.pkl` stores user features:

```text
user_idx, gender, age, fans_num, follows_num
```

`note_dict.pkl` and `user_dict.pkl` store id re-indexing maps.

## dataset/: Model Sample Level

`dataset/src_*.pkl` contains source/search-domain clicked item samples.

Columns:

```text
user_id
item_id
timestamp
rec_his
src_session_his
keyword
search_session_id
neg_items
```

Meaning:

- One row is one clicked positive search item.
- `item_id`: clicked positive item.
- `keyword`: tokenized current search query, stored as a stringified token list
  in the current files.
- `search_session_id`: search session id for the current query/session.
- `neg_items`: negative candidate item ids for this positive item.
- `rec_his`: number of recommendation clicks before this search item timestamp.
- `src_session_his`: number of previous search sessions before the current
  search session. This varies across sessions for the same user.

`dataset/rec_*.pkl` contains recommendation-domain clicked item samples.

Columns:

```text
user_id
item_id
timestamp
rec_his
src_session_his
neg_items
```

Meaning in the current `Data/` version:

- One row is one clicked positive recommendation item.
- `item_id`: clicked positive recommendation item.
- `neg_items`: negative candidate item ids for this positive item.
- `rec_his`: number of recommendation clicks before this recommendation click
  timestamp.
- `src_session_his`: total number of source/search sessions for this user. In
  the current `Data/` files, this value is constant within each user in
  `rec_train.pkl`, `rec_val.pkl`, and `rec_test.pkl`.

Important: `src_session_his` is time-aware in `src_*`, but not time-aware in the
current `rec_*` files in this directory.

## vocab/: Lookup Tables

`vocab/query_vocab.pkl`

- List of query token lists.
- Used to decode `query_id` in ranking result files.

`vocab/src_session_vocab_np.pkl`

Dictionary with:

```text
keyword:   padded query tokens, shape (num_sessions + 1, 50)
pos_items: padded clicked item ids per search session, shape (num_sessions + 1, 5)
```

Index `0` is a zero/padding row. Real `src_session_id` values index into this
table.

`vocab/item_vocab_np.pkl`

Dictionary with:

```text
caption
first_level_category_id
second_level_category_id
```

In this processed dataset:

- `caption` is padded item caption/title tokens.
- `first_level_category_id` is derived from `note_type`.
- `second_level_category_id` is derived from log-transformed `like_num`.
- Index `0` is a zero/padding item.

`vocab/user_vocab.pkl`

Dictionary keyed by `user_id`. Each value contains:

```text
user_id
rec_his
rec_his_ts
src_session_his
src_session_his_ts
```

Meaning:

- `rec_his`: full list of clicked recommendation item ids for the user.
- `rec_his_ts`: timestamps for `rec_his`.
- `src_session_his`: full list of source/search session ids for the user.
- `src_session_his_ts`: first timestamp for each source/search session.

This file stores full histories. The integer `rec_his` and `src_session_his`
columns in `dataset/*.pkl` are history counts at sample-construction time, not
the full history lists.

`vocab/user_vocab_np.pkl`

Dictionary with numeric user feature arrays:

```text
onehot_feat1        -> encoded gender
onehot_feat2        -> encoded age
search_active_level -> log-transformed fans_num
rec_active_level    -> log-transformed follows_num
```

## Current Data Version Note

The notebooks under `Data/Qilin-data-session-split-creating/` are the
generation workflow used to create the processed artifacts in this directory.
The runnable data for the analysis lives in:

- `Data/Step1/`
- `Data/Step4/`
- `Data/dataset/`
- `Data/vocab/`

For the current analysis code, the most important files are:

- `Data/Step4/src_all.pkl`
- `Data/Step4/rec_all.pkl`
- `Data/Step1/vocab_dict.pkl`
- `Data/Step1/note_feat.pkl`

The `dataset/rec_*.pkl` files still use the sample-construction history counts in
their `src_session_his` column. Do not treat that column as a full timeline.
