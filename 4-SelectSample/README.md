# Select Sample Timeline Export

This folder contains a small utility for selecting user trajectories that switch
repeatedly between recommendation (`rec`) and search/source (`src`) behavior.

The goal is to produce readable Chinese timelines from the processed `Data/`
artifacts, with each search session collapsed into a single line.

## What It Does

The script:

1. Loads `Data/dataset/rec_{train,val,test}.pkl`
2. Loads `Data/dataset/src_{train,val,test}.pkl`
3. Decodes query tokens and item captions back to Chinese text
4. Collapses all rows that belong to the same search session into one line
5. Sorts all events chronologically by timestamp
6. Selects users whose timelines switch repeatedly between `rec` and `src`
7. Writes a human-readable timeline text file and a CSV summary

## Input Data

- `Data/dataset/rec_train.pkl`
- `Data/dataset/rec_val.pkl`
- `Data/dataset/rec_test.pkl`
- `Data/dataset/src_train.pkl`
- `Data/dataset/src_val.pkl`
- `Data/dataset/src_test.pkl`
- `Data/Step1/vocab_dict.pkl`
- `Data/Step1/note_feat.pkl`

## Output

The default output directory is:

```text
4-SelectSample/output/
```

It contains:

- `selected_samples.txt`: readable timelines
- `selected_samples.csv`: per-user summary and selection score

## Timeline Format

Recommendation events are written like this:

```text
[2024-11-26 14:10:17] (rec) 软软糯糯的黄豆打糕！
```

Search events are collapsed by `user_id + search_session_id` and written like
this:

```text
[2024-11-26 22:10:23] (search) 婴儿弯睫毛 | items: 不翻车！新手挑选婴儿弯睫毛指南🔆; 清纯可人小白花😝有没有人来怜惜一下‼️🔥; 美睫的尽头不就是C➕婴儿弯嘛！
```

If multiple clicked items belong to the same search session, they are kept on
one line instead of being emitted repeatedly.

## Selection Logic

By default, the script keeps users with:

- at least a minimum number of `rec`/`src` switches
- at least a minimum total event count
- at least one event from both channels

The selected users are ranked by a switch-based score so that the most
alternating trajectories appear first.

## Run

From `Analysis/` with your Python 3.11 environment:

```bash
cd "/Users/Brodie/Documents/Code-RelSAR/Results&Analysis/Analysis"
/Users/Brodie/miniconda3/bin/python3 4-SelectSample/select_sample.py
```

You can also adjust the filters:

```bash
/Users/Brodie/miniconda3/bin/python3 4-SelectSample/select_sample.py --min-switches 6 --min-events 12 --top-n 50
```
