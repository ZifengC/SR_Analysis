# PC-SAR Validation

This folder contains the validation and ablation analysis for PC-SAR.

Its role is downstream of the `RelSAR` codebase:

- `RelSAR/test.py` exports the model feature table for the test split.
- The scripts in this folder read those exports and summarize how the PC-SAR internals behave.
- The outputs are CSV tables and plots used for interpretation, ablation comparison, and reporting.

## Relationship to `RelSAR`

`RelSAR/` is the upstream implementation that generates the exported validation data.

In particular:

- `RelSAR/test.py` runs the trained model on a dataset split and writes a flattened feature table.
- The exported table contains intent posteriors, belief/uncertainty diagnostics, attribution proxies, prediction scores, embeddings, and metadata.
- This folder does not train the model. It only analyzes the exported PC-SAR features.

## Main Input Files

The core inputs are three test-set exports with the same schema:

- `pcsar_intent_features_test_full.csv`
- `pcsar_intent_features_test_no_intent_state.csv`
- `pcsar_intent_features_test_no_counterfactual.csv`

These files are generated from the same test split but with different model settings:

- `full`: all PC-SAR components enabled
- `no_intent_state`: intent-state bias and uncertainty-attention components disabled
- `no_counterfactual`: counterfactual attribution disabled

The three files are used to compare how PC-SAR behavior changes when one component is removed.

## What Is Inside the CSVs

The CSVs are sample-level exports. Each row corresponds to one test example and includes:

- model configuration flags
- user and sample identifiers
- search/recommendation channel information
- history lengths and history shares
- global/rec/src intent distributions
- uncertainty and belief statistics
- attribution proxies
- prediction scores and ranks
- embedding vectors and latent feature vectors

The important derived fields include:

- `history_rec_share`
- `history_src_share`
- `global_pi_*`, `rec_pi_*`, `src_pi_*`
- `attribution_source_proxy`
- `attribution_confidence_gap`
- `rec_src_intent_shift_js`

These are computed inside `RelSAR/test.py`, not inside the analysis scripts.

## Scripts

### `script/ablation_analysis.py`

Compares the three PC-SAR variants.

#### Inputs

- `pcsar_intent_features_test_full.csv`
- `pcsar_intent_features_test_no_intent_state.csv`
- `pcsar_intent_features_test_no_counterfactual.csv`
- optional raw trajectory CSV for transition relabeling

#### What it does

- compares the `full` variant against the `no_intent_state` variant
- compares the `full` variant against the `no_counterfactual` variant
- aggregates state-level and transition-level summaries
- produces ablation tables and figures

#### Outputs

- `output_ablation/state/state_shock_events.csv`
- `output_ablation/state/state_summary.csv`
- `output_ablation/state/state_shock_summary.png`
- `output_ablation/attribution/transition_events.csv`
- `output_ablation/attribution/transition_summary.csv`
- `output_ablation/attribution/transition_exploration_summary.csv`
- `output_ablation/attribution/transition_gap.csv`
- `output_ablation/attribution/transition_summary.png`

### `script/intent_pattern_analysis.py`

Runs intent-pattern analysis on the full PC-SAR export.

#### Input

- `pcsar_intent_features_test_full.csv`

#### What it does

- treats `history_src_share` as the exploration proxy
- summarizes how exploration relates to intent ambiguity
- measures stable run length and semantic dispersion
- summarizes how intent shift varies across transition types

#### Outputs

- `output/1/exploration_intent_ambiguity.csv`
- `output/1/exploration_intent_ambiguity.png`
- `output/23/stable_runs.csv`
- `output/23/run_length_intent_consolidation_dispersion.csv`
- `output/23/run_length_intent_consolidation_dispersion.png`
- `output/4/transition_type_intent_shift.csv`
- `output/4/transition_type_intent_shift.png`

## Recommended Run Order

Run the scripts from `Analysis/` with your Python 3.11 environment:

```bash
cd "/Users/Brodie/Documents/Code-RelSAR/Results&Analysis/Analysis"
/Users/Brodie/miniconda3/bin/python3 3-PC-SAR-Validation/script/intent_pattern_analysis.py
/Users/Brodie/miniconda3/bin/python3 3-PC-SAR-Validation/script/ablation_analysis.py
```

If you want to regenerate the CSV exports first, run `RelSAR/test.py` with the appropriate checkpoint and split. The analysis scripts in this folder already point at the CSV exports stored in `3-PC-SAR-Validation/`.

## High-Level Interpretation

This folder answers questions like:

- Does PC-SAR become less stable when intent-state machinery is removed?
- Does counterfactual attribution affect transition behavior?
- How does latent intent shift differ between search and recommendation events?
- Which event sequences look more exploratory or more consolidated?

## Summary

- `RelSAR/` produces the exported PC-SAR feature tables.
- `3-PC-SAR-Validation/` analyzes those exports.
- The results are used to interpret PC-SAR behavior and compare ablated variants against the full model.
