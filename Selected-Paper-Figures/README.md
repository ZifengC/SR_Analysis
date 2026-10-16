# Selected Paper Figures

Path root:

```text
/Users/Brodie/Documents/Code-RelSAR/Results&Analysis/Analysis/Selected-Paper-Figures
```

All paths below are relative to this directory.

## Figures and code

| Label | Figure | Main script |
|---|---|---|
| Fig. 2 | `figures/Fig2_history_similarity_top5_mean_min10_distribution.png` | `code/2-Problem-identify/2-part2-history-similarity-entropy-distribution.py` |
| Fig. 3 | `figures/Fig3_history_domain_top1_similarity_scatter.png` | `code/2-Problem-identify/2-part2-history-domain-top1-similarity-scatter.py` |
| Fig. 4 | `figures/Fig4_state_persistence_integration_transition_gate_validation.png` | `code/5-Validation/mechanism_validation.py` |
| Fig. 5(a) | `figures/Fig5a_preference_elaboration_future_query_reformulation.png` | `code/6-Behavioral-Validation/3-behavioral-preference-elaboration-plots.py` |
| Fig. 5(b) | `figures/Fig5b_preference_elaboration_dwell_time.png` | `code/6-Behavioral-Validation/3-behavioral-preference-elaboration-plots.py` |
| Fig. 6 | `figures/Fig6_context_weight_switching_rate.png` | `code/6-Behavioral-Validation/4-behavioral-channel-continuation-context-weight.py` |
| Fig. 7(a) | `figures/Fig7a_similarity_breadth_prediction_trend.png` | `code/5-Validation/similarity_breadth_prediction_trend.py` |
| Fig. 7(b) | `figures/Fig7b_cross_channel_grounding_ndcg_trend.png` | `code/5-Validation/cross_channel_grounding_ndcg_trend.py` |

## Data paths

### Fig. 2 and Fig. 3

```text
data/2-Problem-identify/intermediate/1-events.pkl
data/2-Problem-identify/intermediate/1-event-embeddings.npy
```

### Fig. 4

```text
data/5-Validation/Features/intermediate/pcsar_intent_features_all_full_mechanism.csv
```

### Fig. 5(a-b)

```text
data/6-Behavioral-Validation/data/final_matched_samples_with_step4.parquet
data/5-Validation/Features/intermediate/pcsar_intent_features_all_full_mechanism.csv
data/6-Behavioral-Validation/data/preference_elaboration_behavioral_metrics.parquet
data/6-Behavioral-Validation/cache/query_embeddings_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.pkl
```

### Fig. 6

```text
data/6-Behavioral-Validation/data/final_matched_samples_with_step4.parquet
data/5-Validation/Features/intermediate/pcsar_intent_features_all_full_mechanism.csv
data/6-Behavioral-Validation/data/preference_elaboration_behavioral_metrics.parquet
data/6-Behavioral-Validation/data/context_weight_transition_events.parquet
data/6-Behavioral-Validation/cache/query_embeddings_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.pkl
```

### Fig. 7(a-b)

```text
data/5-Validation/Old_features/pcsar_intent_features_test_full.csv
data/5-Validation/Old_features/pcsar_intent_features_test_no_intent_state.csv
data/2-Problem-identify/intermediate/1-events.pkl
data/2-Problem-identify/intermediate/1-event-embeddings.npy
```

## Rebuilding original embeddings

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

```text
data/Data/Step1/vocab_dict.pkl
data/Data/Step1/note_feat.pkl
data/Data/Step4/src_all.pkl
data/Data/Step4/rec_all.pkl
data/2-Problem-identify/cache/text_embedding_cache_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.pkl
```

Support code:

```text
code/2-Problem-identify/1-part1-exploration.py
code/2-Problem-identify/_shared.py
code/5-Validation/ablation_analysis.py
```
