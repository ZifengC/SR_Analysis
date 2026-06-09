"""
Decode query_id/item_id back to text and compare TrustSAR vs UNISAR ranks.

Run from Results-Analysis/:
    python 1-Compare-baseline/Code/2-compare_ranks.py

Inputs:
    1-Compare-baseline/New_results/1-results-TrustSAR.csv
      or 1-Compare-baseline/New_results/2-results-TrustSAR-translated.csv
    1-Compare-baseline/New_results/1-results-UNISAR.csv
      or 1-Compare-baseline/New_results/2-results-UNISAR-translated.csv
    Data/Step1/vocab_dict.pkl
    Data/Step1/note_feat.pkl
    Data/vocab/query_vocab.pkl

Outputs:
    1-Compare-baseline/New_results/2-results-TrustSAR-translated.csv
    1-Compare-baseline/New_results/2-results-UNISAR-translated.csv
    1-Compare-baseline/New_results/3-rank_diff_vs_UNISAR_sorted.csv
"""

import pickle
from pathlib import Path

import pandas as pd


# Results-Analysis/ contains shared Data/ and this pipeline folder.
BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parent
DATA = PROJECT_ROOT / "Data"
NEW_RESULTS = BASELINE_ROOT / "New_results"

TRUSTSAR_RAW = NEW_RESULTS / "1-results-TrustSAR.csv"
UNISAR_RAW = NEW_RESULTS / "1-results-UNISAR.csv"
TRUSTSAR_TRANSLATED = NEW_RESULTS / "2-results-TrustSAR-translated.csv"
UNISAR_TRANSLATED = NEW_RESULTS / "2-results-UNISAR-translated.csv"
RANK_DIFF_OUTPUT = NEW_RESULTS / "3-rank_diff_vs_UNISAR_sorted.csv"


def load_vocab():
    vocab = pickle.load(open(DATA / "Step1" / "vocab_dict.pkl", "rb"))
    rev_vocab = {v: k for k, v in vocab.items()}
    query_vocab = pickle.load(open(DATA / "vocab" / "query_vocab.pkl", "rb"))
    note_feat = pickle.load(open(DATA / "Step1" / "note_feat.pkl", "rb"))
    caption_map = dict(zip(note_feat["note_id"], note_feat["caption"]))
    return rev_vocab, query_vocab, caption_map


def decode(tokens, rev_vocab, unk="<UNK>"):
    # Token 0 is padding; skip it.
    return "".join(rev_vocab.get(int(t), unk) for t in tokens if int(t) != 0)


def translate(in_path: Path, out_path: Path, rev_vocab, query_vocab, caption_map):
    df = pd.read_csv(in_path)
    df["query_text"] = df["query_id"].apply(
        lambda q: decode(query_vocab[int(q)], rev_vocab)
        if pd.notna(q) and 0 <= int(q) < len(query_vocab)
        else ""
    )
    df["item_caption"] = df["item"].apply(
        lambda i: decode(caption_map.get(int(i), []), rev_vocab)
    )
    df.to_csv(out_path, index=False)


def ensure_translated(raw_path: Path, translated_path: Path, rev_vocab, query_vocab, caption_map):
    if raw_path.exists():
        translate(raw_path, translated_path, rev_vocab, query_vocab, caption_map)
        return
    if translated_path.exists():
        print(f"Using existing translated file: {translated_path}")
        return
    raise FileNotFoundError(
        f"Missing both {raw_path} and {translated_path}. "
        "Provide either the raw model output or the translated output."
    )


def compare():
    base = pd.read_csv(UNISAR_TRANSLATED)
    trust = pd.read_csv(TRUSTSAR_TRANSLATED)
    keys = ["domain", "user", "query_id", "item"]

    merged = base.merge(
        trust[keys + ["rank"]],
        on=keys,
        how="inner",
        suffixes=("_unisar", "_trust"),
    )
    merged["rank_diff"] = merged["rank_trust"] - merged["rank_unisar"]
    merged["direction"] = merged["rank_diff"].apply(
        lambda x: "up" if x < 0 else ("down" if x > 0 else "same")
    )
    merged = merged.sort_values(["rank_diff", "rank_unisar", "rank_trust"])
    merged.to_csv(RANK_DIFF_OUTPUT, index=False)
    return merged


def main():
    rev_vocab, query_vocab, caption_map = load_vocab()

    ensure_translated(
        TRUSTSAR_RAW,
        TRUSTSAR_TRANSLATED,
        rev_vocab,
        query_vocab,
        caption_map,
    )
    ensure_translated(
        UNISAR_RAW,
        UNISAR_TRANSLATED,
        rev_vocab,
        query_vocab,
        caption_map,
    )

    merged = compare()
    print(merged["direction"].value_counts())


if __name__ == "__main__":
    main()
