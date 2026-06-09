# Analysis Project Overview

This directory contains the analysis layer for the SR project. It does not train the model itself. Instead, it consumes processed Qilin search-recommendation data and exported outputs from the upstream `RelSAR` codebase to study ranking behavior, user exploration patterns, and PC-SAR ablations.

## Relationship to `RelSAR`

`RelSAR/` is the upstream implementation of the model and export pipeline.

- `RelSAR/test.py` is the source of the PC-SAR validation CSV exports.
- The analysis folders in this directory consume those exports and the processed dataset artifacts under `Data/`.
- In other words:
  - `RelSAR` produces model outputs and intermediate features.
  - `Analysis/` interprets those outputs and turns them into tables, plots, and comparison summaries.

## Directory Map

```text
Analysis/
  Data/
    Step1/
    Step4/
    dataset/
    vocab/
  1-Compare-baseline/
  2-Problem-identify/
  3-PC-SAR-Validation/
  4-SelectSample/
```

## `Data/`

### What it is

`Data/` contains the processed Qilin search-recommendation dataset used by both `RelSAR` and the analysis scripts. The data is already re-indexed, tokenized, split, and converted into model-ready pickle files. It is not raw log data.

### Main contents

```text
Data/
  Step1/
  Step4/
  dataset/
  vocab/
```

### Core data structure

The dataset has two behavior domains:

- `src`: search/source behavior
- `rec`: recommendation behavior

It also has two abstraction levels:

- Session/exposure level: one row contains a search session or a recommendation exposure with multiple items.
- Sample level: one row contains one clicked positive item plus negative candidates for training/evaluation.

### Inputs

`Data/` is the source input for:

- `RelSAR` model training and evaluation
- the baseline comparison pipeline
- the analysis scripts in `2-Problem-identify/`

### Outputs

`Data/` itself is mostly an input package. The processed files under it are consumed by other modules rather than produced by the analysis pipeline.

### Key files

- `Data/Step1/exposure_src.pkl`: search-session exposure data
- `Data/Step1/exposure_rec.pkl`: recommendation exposure data
- `Data/Step1/note_feat.pkl`: item/note features
- `Data/Step1/user_feat.pkl`: user features
- `Data/Step4/src_all.pkl`: full search event table for `2-Problem-identify`
- `Data/Step4/rec_all.pkl`: full recommendation event table for `2-Problem-identify`
- `Data/dataset/src_{train,val,test}.pkl`: search-domain clicked-item samples
- `Data/dataset/rec_{train,val,test}.pkl`: recommendation-domain clicked-item samples
- `Data/vocab/*`: lookup tables for decoding ids and tokens

## `1-Compare-baseline/`

### What it does

This module compares TrustSAR ranking outputs against a fixed UNISAR baseline.

### Inputs

- `1-results-UNISAR.csv`
- `1-results-TrustSAR.csv`
- `Data/Step1/vocab_dict.pkl`
- `Data/Step1/note_feat.pkl`
- `Data/vocab/query_vocab.pkl`

### Outputs

- translated ranking tables
- rank-difference table
- readable user timelines
- similarity-bucket summaries

### Relation to `RelSAR`

This folder is downstream of the model outputs. It does not train `RelSAR`; it only analyzes ranking results produced by upstream systems.

## `2-Problem-identify/`

### What it does

This is the behavioral analysis layer. It converts user search queries and clicked recommendation captions into embeddings, then measures how exploratory, stable, or future-consistent each event is.

The analysis focuses on questions such as:

- Is the current behavior far from the user's recent history?
- Does the current behavior match the user's near future?
- Does repeated nearby behavior consolidate into stable runs?
- How does exploration differ by transition type?
- How do anchors and future adoption interact?

### Main ideas

- Search events are represented by decoded search queries.
- Recommendation events are represented by clicked item captions.
- Events are sorted by user and time.
- The pipeline computes embedding-based measures such as:
  - exploration score
  - future consistency
  - transition-level asymmetry
  - semantic-neighbor run length
  - anchor-conditioned adoption

### Inputs

The scripts primarily consume processed data from `Data/`, especially:

- `Data/Step1/vocab_dict.pkl`
- `Data/Step1/note_feat.pkl`
- `Data/Step4/src_all.pkl`
- `Data/Step4/rec_all.pkl`

When available, cached intermediate artifacts are reused:

- `2-Problem-identify/intermediate/1-events.pkl`
- `2-Problem-identify/intermediate/1-event-embeddings.npy`
- `2-Problem-identify/intermediate/1-exploration-scores.pkl`

### Outputs

Typical outputs include:

- `2-Problem-identify/figures/*.png`
- `2-Problem-identify/intermediate/*.pkl` or `.npy`

### Relation to `RelSAR`

This folder does not depend on the PC-SAR model internals directly. It is mainly a data-behavior analysis layer over the same Qilin dataset that `RelSAR` uses. The goal is to characterize user behavior patterns that help interpret model behavior later.

## `3-PC-SAR-Validation/`

### What it does

This is the PC-SAR validation and ablation layer. It reads the model-exported feature table produced by `RelSAR/test.py` and compares different PC-SAR variants.

The core validation questions are:

- What changes when intent-state machinery is removed?
- What changes when counterfactual attribution is removed?
- How do intent distributions evolve across search and recommendation events?
- How strong is the shift between `rec` and `src` intent states?

### Main input files

These are the three exported test-set variants:

- `pcsar_intent_features_test_full.csv`
- `pcsar_intent_features_test_no_intent_state.csv`
- `pcsar_intent_features_test_no_counterfactual.csv`

These CSVs have the same schema but different model flags:

- `full`: all PC-SAR components enabled
- `no_intent_state`: intent-state bias and uncertainty attention disabled
- `no_counterfactual`: counterfactual attribution disabled

### Where the CSVs come from

They are exported from `RelSAR/test.py`, which runs the trained model on the test split and writes out:

- model configuration flags
- intent posteriors
- belief / uncertainty diagnostics
- attribution proxies
- rec/src prediction scores
- embedding vectors
- item-level and session-level metadata

### Scripts

#### `script/ablation_analysis.py`

Compares the `full`, `no_intent_state`, and `no_counterfactual` variants.

**Inputs**

- the three CSV variants listed above
- optional raw trajectory CSV for relabeling transitions

**Outputs**

- `output_ablation/state/state_shock_events.csv`
- `output_ablation/state/state_summary.csv`
- `output_ablation/state/state_shock_summary.png`
- `output_ablation/attribution/transition_events.csv`
- `output_ablation/attribution/transition_summary.csv`
- `output_ablation/attribution/transition_exploration_summary.csv`
- `output_ablation/attribution/transition_gap.csv`
- `output_ablation/attribution/transition_summary.png`

#### `script/intent_pattern_analysis.py`

Aggregates intent-pattern statistics from the full PC-SAR export.

**Inputs**

- `pcsar_intent_features_test_full.csv`

**Outputs**

- `output/1/exploration_intent_ambiguity.csv`
- `output/1/exploration_intent_ambiguity.png`
- `output/23/stable_runs.csv`
- `output/23/run_length_intent_consolidation_dispersion.csv`
- `output/23/run_length_intent_consolidation_dispersion.png`
- `output/4/transition_type_intent_shift.csv`
- `output/4/transition_type_intent_shift.png`

### Relation to `RelSAR`

This folder is the direct downstream consumer of `RelSAR` exports. It does not re-run the model training loop. Instead, it interprets the exported latent intent structure and compares ablated variants against the full PC-SAR model.

## Overall Data Flow

```text
Raw Qilin behavior data
  -> processed dataset in Data/
  -> RelSAR training / evaluation
  -> RelSAR/test.py export of PC-SAR features
  -> 3-PC-SAR-Validation analyses
  -> figures and CSV summaries for interpretation
```

```text
Data/ + ranking outputs
  -> 1-Compare-baseline/
```

```text
Data/
  -> 2-Problem-identify/
```

## Quick Summary

- `Data/` provides the processed search-recommendation dataset.
- `2-Problem-identify/` explains user behavior patterns with embedding-based metrics.
- `3-PC-SAR-Validation/` evaluates PC-SAR internals and ablations using exported test-set features.
- `4-SelectSample/` exports readable alternating trajectories for manual inspection.
- `RelSAR/` is the upstream model code that generates the PC-SAR validation exports consumed here.
