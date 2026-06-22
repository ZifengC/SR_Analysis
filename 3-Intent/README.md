# PC-SAR Intent Analysis

This folder analyzes the latent intent structure exported by PC-SAR. It is a
downstream analysis module: model training and feature export happen in the
upstream `RelSAR` codebase, while this folder produces interpretation tables
and figures.

## Scripts

### `script/intent_pattern_analysis.py`

Uses the full PC-SAR feature export to analyze exploration, transition-level
intent shift, future consistency, and stable intent runs.

Default input:

```text
3-Intent/pcsar_intent_features_test_full.csv
```

### `script/intent_tsne.py`

Visualizes event-level global intent posteriors with t-SNE.

Default input:

```text
5-Validation/output_mechanism/event_level/model_events.csv
```

## Result Organization

The main results are organized as three numbered parts:

```text
3-Intent/output/
  Part1/
    exploration_intent_ambiguity.csv
    exploration_intent_ambiguity.png
  Part2/
    transition_type_intent_shift.csv
    transition_type_intent_shift.png
  Part3/
    intent_tsne_sample.csv
    intent_tsne_summary.csv
    intent_tsne_by_dominant_intent.png
    intent_tsne_by_uncertainty_state.png
```

### Part 1: Exploration and Intent Ambiguity

Uses `history_src_share` as the exploration proxy and summarizes how exploration
relates to intent entropy and dominant-intent probability.

### Part 2: Transition-Type Intent Shift

Compares intent-posterior movement across `R->R`, `R->S`, `S->R`, and `S->S`
transitions.

### Part 3: Intent Posterior t-SNE

Projects event-level global intent posteriors into two dimensions and visualizes
their structure by dominant intent and uncertainty state.

## Auxiliary Analyses

Supporting results are stored separately from the three main parts:

```text
3-Intent/output/AuxiliaryAnalysis/
  Part1/
    intent_future_consistency_and_expansion.csv
    intent_future_consistency_and_expansion_curve.csv
    intent_future_consistency_and_expansion.png
  Part2/
    stable_runs.csv
    run_length_intent_consolidation_dispersion.csv
    run_length_intent_consolidation_dispersion.png
```

`AuxiliaryAnalysis/Part1` studies intent future consistency and expansion.
`AuxiliaryAnalysis/Part2` studies stable-run length, consolidation, and posterior
dispersion.

## Run

Run from the `Analysis/` root with Python 3.11:

```bash
/Users/Brodie/miniconda3/bin/python3 3-Intent/script/intent_pattern_analysis.py
/Users/Brodie/miniconda3/bin/python3 3-Intent/script/intent_tsne.py
```

Both scripts accept command-line output overrides. By default,
`intent_pattern_analysis.py` writes to `3-Intent/output/`, and `intent_tsne.py`
writes to `3-Intent/output/Part3/`.
