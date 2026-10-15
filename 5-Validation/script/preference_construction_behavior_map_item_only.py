from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent.parent
PROBLEM_ROOT = ANALYSIS_ROOT / "2-Problem-identify"
DATA_ROOT = ANALYSIS_ROOT / "Data"
OUTPUT_ROOT = SCRIPT_DIR.parent / "output_ablation3"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_HISTORY = 10
TOP_K = 5

sys.path.insert(0, str(SCRIPT_DIR))
from preference_construction_behavior_map import (  # noqa: E402
    plot_map,
    summarize_map,
    summarize_spearman,
)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def decode_caption(tokens, reverse_vocab: dict[int, str]) -> str:
    return "".join(reverse_vocab.get(int(token), "") for token in tokens if int(token) != 0).strip()


def build_item_events() -> pd.DataFrame:
    base = pd.read_pickle(PROBLEM_ROOT / "intermediate" / "1-events.pkl")
    items = base[
        ["event_id", "user_id", "timestamp", "domain", "event_pos", "item_ids"]
    ].explode("item_ids", ignore_index=False)
    items = items.rename(columns={"domain": "channel", "item_ids": "item_id"}).copy()
    items["item_id"] = items["item_id"].astype(int)
    items["item_position"] = items.groupby("event_id", sort=False).cumcount()
    items["item_event_id"] = (
        items["event_id"].astype(str) + ":item:" + items["item_position"].astype(str)
    )

    vocab = load_pickle(DATA_ROOT / "Step1" / "vocab_dict.pkl")
    reverse_vocab = {int(value): key for key, value in vocab.items()}
    needed_item_ids = set(items["item_id"].unique())
    note_feat = pd.read_pickle(DATA_ROOT / "Step1" / "note_feat.pkl")
    captions = note_feat[note_feat["note_id"].astype(int).isin(needed_item_ids)][
        ["note_id", "caption"]
    ].copy()
    captions["item_id"] = captions["note_id"].astype(int)
    captions["text"] = captions["caption"].map(
        lambda value: decode_caption(value, reverse_vocab)
    )
    items = items.merge(captions[["item_id", "text"]], on="item_id", how="left")
    items["text"] = items["text"].fillna("").astype(str).str.strip()
    items = items[items["text"].ne("")].copy()
    return items.sort_values(
        ["user_id", "timestamp", "event_pos", "item_position"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_or_encode_texts(texts: list[str]) -> dict[str, np.ndarray]:
    base_cache_path = (
        PROBLEM_ROOT
        / "cache"
        / "text_embedding_cache_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.pkl"
    )
    item_cache_path = (
        SCRIPT_DIR.parent
        / "cache"
        / "item_caption_embeddings_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.pkl"
    )
    needed = set(texts)
    base_cache = load_pickle(base_cache_path)
    embeddings = {text: base_cache[text] for text in needed if text in base_cache}
    del base_cache

    item_cache = load_pickle(item_cache_path) if item_cache_path.exists() else {}
    embeddings.update({text: item_cache[text] for text in needed if text in item_cache})
    missing = sorted(needed - embeddings.keys())
    if missing:
        from sentence_transformers import SentenceTransformer

        print(f"Encoding {len(missing):,} missing item captions...")
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        encoded = model.encode(
            missing,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        item_cache.update(dict(zip(missing, encoded)))
        dump_pickle(item_cache, item_cache_path)
        embeddings.update(dict(zip(missing, encoded)))
    return embeddings


def build_fixed10_item_only_metrics(item_events: pd.DataFrame) -> pd.DataFrame:
    unique_texts = item_events["text"].drop_duplicates().tolist()
    text_to_embedding = load_or_encode_texts(unique_texts)
    embedding_matrix = np.vstack([text_to_embedding[text] for text in unique_texts]).astype(
        np.float32
    )
    text_to_row = {text: row for row, text in enumerate(unique_texts)}
    item_events = item_events.copy()
    item_events["embedding_row"] = item_events["text"].map(text_to_row).astype(int)

    rows = []
    for _, user_events in item_events.groupby("user_id", sort=False):
        history_embedding_rows: list[int] = []
        history_channels: list[str] = []
        for _, time_block in user_events.groupby("timestamp", sort=False):
            if len(history_embedding_rows) >= N_HISTORY:
                history_rows = np.asarray(history_embedding_rows[-N_HISTORY:], dtype=int)
                history_channel_array = np.asarray(history_channels[-N_HISTORY:])
                current_rows = time_block["embedding_row"].to_numpy(dtype=int)
                similarities = embedding_matrix[current_rows] @ embedding_matrix[history_rows].T
                top5_mean = np.partition(similarities, -TOP_K, axis=1)[:, -TOP_K:].mean(axis=1)

                for row_position, (_, event) in enumerate(time_block.iterrows()):
                    channel = str(event["channel"])
                    same_mask = history_channel_array == channel
                    cross_mask = ~same_mask
                    same_top1 = (
                        float(similarities[row_position, same_mask].max())
                        if same_mask.any()
                        else 0.0
                    )
                    cross_top1 = (
                        float(similarities[row_position, cross_mask].max())
                        if cross_mask.any()
                        else 0.0
                    )
                    contextualization = cross_top1 - same_top1
                    rows.append(
                        {
                            "user_id": int(event["user_id"]),
                            "event_id": event["item_event_id"],
                            "parent_event_id": event["event_id"],
                            "item_id": int(event["item_id"]),
                            "timestamp": float(event["timestamp"]),
                            "channel": channel,
                            "fixed_history_count": N_HISTORY,
                            "raw_top5_similarity_mean": float(top5_mean[row_position]),
                            "elaboration_proxy": 1.0 - float(top5_mean[row_position]),
                            "cross_channel_top1_similarity": cross_top1,
                            "same_channel_top1_similarity": same_top1,
                            "contextualization_proxy": contextualization,
                            "contextualization_proxy_raw": contextualization,
                            "cross_history_missing": not cross_mask.any(),
                            "same_history_missing": not same_mask.any(),
                            "history_missing_any_channel": not (cross_mask.any() and same_mask.any()),
                            "contextualization_observed": bool(cross_mask.any() and same_mask.any()),
                            "recommendation_y_shift": 0.0,
                        }
                    )

            history_embedding_rows.extend(time_block["embedding_row"].astype(int).tolist())
            history_channels.extend(time_block["channel"].astype(str).tolist())

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    item_events = build_item_events()
    metrics = build_fixed10_item_only_metrics(item_events)
    metrics = metrics[metrics["contextualization_observed"]].copy()
    stem = "preference_construction_behavior_map_fixed10_raw_top5_item_only"
    metrics.to_csv(OUTPUT_ROOT / f"{stem}_events.csv", index=False)
    summary = summarize_map(metrics)
    summary.to_csv(OUTPUT_ROOT / f"{stem}_summary.csv", index=False)
    summarize_spearman(metrics).to_csv(
        OUTPUT_ROOT / f"{stem}_spearman.csv",
        index=False,
    )
    plot_map(
        metrics,
        OUTPUT_ROOT / f"{stem}.png",
        x_label="Preference Elaboration (1 − Raw Top-5 Item Similarity Mean): Focused  →  Rich",
        scatter_alpha=0.28,
    )
    print(summary.to_string(index=False))
    print(f"Saved item-only behavior map: {(OUTPUT_ROOT / f'{stem}.png').resolve()}")


if __name__ == "__main__":
    main()
