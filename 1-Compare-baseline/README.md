# Compare Baseline Pipeline

This folder contains the current baseline-comparison analysis pipeline for
comparing TrustSAR ranking outputs against a fixed UNISAR baseline.

Run the commands from `Analysis/` with your Python 3.11 environment.

## Folder Layout

```text
Analysis/
  Data/
    Step1/
    dataset/
    vocab/

  1-Compare-baseline/
    Code/
      1-sample.ipynb
      2-compare_ranks.py
      3-export_timelines_txt.py
      4-similarity_bucket_analysis.py

    New_results/
      1-results-UNISAR.csv
      1-results-TrustSAR.csv              # optional if translated file exists
      2-results-UNISAR-translated.csv
      2-results-TrustSAR-translated.csv
      3-rank_diff_vs_UNISAR_sorted.csv
      4-extreme_timelines_global.txt
      5-sim_bucket_user_level_char.csv
      5-sim_bucket_stats_char.csv

    Old_results/
      CompareResults0130/
      CompareResults0202/
```

`1-Compare-baseline/Old_results/` is archival. It is not part of the current
pipeline.

## Current Pipeline

Run all commands from the `Analysis/` root.

### 1. Compare Ranking Results

```bash
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/2-compare_ranks.py
```

Inputs:

```text
1-Compare-baseline/New_results/1-results-UNISAR.csv
1-Compare-baseline/New_results/1-results-TrustSAR.csv
Data/Step1/vocab_dict.pkl
Data/Step1/note_feat.pkl
Data/vocab/query_vocab.pkl
```

If `1-results-TrustSAR.csv` or `1-results-UNISAR.csv` is missing, the script can
use the corresponding existing translated file instead.

Outputs:

```text
1-Compare-baseline/New_results/2-results-UNISAR-translated.csv
1-Compare-baseline/New_results/2-results-TrustSAR-translated.csv
1-Compare-baseline/New_results/3-rank_diff_vs_UNISAR_sorted.csv
```

Main metric:

```text
rank_diff = rank_trust - rank_unisar
```

Interpretation:

```text
rank_diff < 0: TrustSAR ranks the item higher than UNISAR
rank_diff > 0: TrustSAR ranks the item lower than UNISAR
rank_diff = 0: both models assign the same rank
```

### 2. Export User Timelines

```bash
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/3-export_timelines_txt.py
```

Inputs:

```text
1-Compare-baseline/New_results/3-rank_diff_vs_UNISAR_sorted.csv
Data/dataset/src_train.pkl
Data/dataset/src_val.pkl
Data/dataset/src_test.pkl
Data/dataset/rec_train.pkl
Data/dataset/rec_val.pkl
Data/dataset/rec_test.pkl
Data/Step1/vocab_dict.pkl
Data/Step1/note_feat.pkl
```

Output:

```text
1-Compare-baseline/New_results/4-extreme_timelines_global.txt
```

This file contains readable per-user timelines for search-domain users where
TrustSAR moved at least one item up or down relative to UNISAR.

### 3. Analyze Similarity Buckets

```bash
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/4-similarity_bucket_analysis.py
```

Input:

```text
1-Compare-baseline/New_results/4-extreme_timelines_global.txt
```

Outputs:

```text
1-Compare-baseline/New_results/5-sim_bucket_user_level_char.csv
1-Compare-baseline/New_results/5-sim_bucket_stats_char.csv
```

The script compares each user's final search query with the latest previous
different event text, computes character-level TF-IDF cosine similarity, assigns
a similarity bucket, and summarizes rank differences within each bucket.

## Minimal Run Order

```bash
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/2-compare_ranks.py
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/3-export_timelines_txt.py
/Users/Brodie/miniconda3/bin/python3 1-Compare-baseline/Code/4-similarity_bucket_analysis.py
```

## Result File Sequence

```text
1-*  model ranking result inputs
2-*  decoded ranking result files
3-*  TrustSAR vs UNISAR rank-difference comparison
4-*  readable user timeline export
5-*  similarity-bucket analysis outputs
```
