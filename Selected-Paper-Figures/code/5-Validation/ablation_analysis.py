from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VARIANT_COLORS = {
    "full": "#1f5f99",
    "no_intent_state": "#6fa8dc",
    "no_counterfactual": "#8c6bb1",
}
VARIANT_LABELS = {
    "full": "PC-SAR",
    "no_intent_state": "w/o Preference Elaboration",
    "no_counterfactual": "w/o Counterfactual",
}

# Fig. 7 only needs stable event identifiers and positive-item ranks. Semantic
# vectors always come from the original MiniLM artifacts, never model exports.
FIG7_REQUIRED_COLUMNS = {
    "sample_index",
    "user_id",
    "timestamp",
    "channel",
    "item_id",
    "search_session_id",
    "rec_pred_pos_rank",
    "src_pred_pos_rank",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sem(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values).dropna()
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def resolve_variant(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("old"):
        return None
    if stem.endswith("full") or stem.endswith("full_mechanism"):
        return "full"
    if stem.endswith("no_intent_state") or stem.endswith("no_intent"):
        return "no_intent_state"
    if stem.endswith("no_counterfactual") or stem.endswith("full_mechanism_no_counterfactual"):
        return "no_counterfactual"
    return None


def supports_ablation_schema(path: Path) -> bool:
    columns = set(pd.read_csv(path, nrows=0).columns)
    return FIG7_REQUIRED_COLUMNS.issubset(columns)


def discover_variant_files(root: Path) -> dict[str, Path]:
    variant_files: dict[str, Path] = {}
    patterns = [
        "pcsar_intent_features_test*.csv",
        "Old_features/*.csv",
        "Features/**/*.csv",
    ]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            variant = resolve_variant(path)
            if variant is not None and variant not in variant_files and supports_ablation_schema(path):
                variant_files[variant] = path
    return variant_files


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = coerce_numeric(
        df,
        [
            "sample_index",
            "user_id",
            "timestamp",
            "item_id",
            "search_session_id",
            "rec_pred_pos_rank",
            "src_pred_pos_rank",
        ],
    )
    return df.sort_values(["user_id", "timestamp", "sample_index"], kind="mergesort").reset_index(drop=True)


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = np.isfinite(values).all(axis=1, keepdims=True) & (norms > 1e-12)
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values, norms, out=out, where=valid)
    return out


def load_native_event_embeddings(native_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    events_path = native_root / "1-events.pkl"
    embeddings_path = native_root / "1-event-embeddings.npy"
    if not events_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing native embedding artifacts: {events_path} or {embeddings_path}"
        )
    events = pd.read_pickle(events_path).reset_index(drop=True)
    embeddings = normalize_matrix(np.load(embeddings_path))
    if len(events) != len(embeddings):
        raise ValueError(f"Native event/embedding size mismatch: {len(events)} vs {len(embeddings)}")
    return events, embeddings


def native_event_embedding_matrix(
    df: pd.DataFrame,
    native_root: Path,
) -> tuple[np.ndarray, str]:
    events, embeddings = load_native_event_embeddings(native_root)
    native = events.copy()
    native["native_embedding_row"] = np.arange(len(native), dtype=np.int64)

    search = native[native["domain"].eq("S")].dropna(subset=["search_session_id"])
    search_map = {
        (int(row.user_id), int(row.search_session_id)): embeddings[int(row.native_embedding_row)]
        for row in search.itertuples(index=False)
    }
    recommendation = native[native["domain"].eq("R")].copy()
    recommendation["item_id"] = recommendation["item_ids"].map(
        lambda value: int(value[0]) if len(value) else -1
    )
    recommendation_map = {
        (int(row.user_id), int(row.item_id), float(row.timestamp)): embeddings[int(row.native_embedding_row)]
        for row in recommendation.itertuples(index=False)
    }

    out = np.zeros((len(df), embeddings.shape[1]), dtype=np.float64)
    missing = []
    for pos, row in enumerate(df.itertuples(index=False)):
        channel = str(row.channel).upper()
        if channel == "S" and pd.notna(row.search_session_id):
            key = (int(row.user_id), int(row.search_session_id))
            vector = search_map.get(key)
        elif channel == "R" and pd.notna(row.item_id) and pd.notna(row.timestamp):
            key = (int(row.user_id), int(row.item_id), float(row.timestamp))
            vector = recommendation_map.get(key)
        else:
            key = None
            vector = None
        if vector is None:
            missing.append((channel, key))
        else:
            out[pos] = vector
    if missing:
        raise ValueError(f"Could not map {len(missing)} evaluation events to native embeddings.")
    return out, "native_multilingual_minilm_query_for_S_caption_for_R"


def build_native_anchor_pool(native_root: Path, users: set[int] | None = None) -> pd.DataFrame:
    events, embeddings = load_native_event_embeddings(native_root)
    if users is not None:
        mask = events["user_id"].astype(int).isin(users).to_numpy()
        events = events.loc[mask].copy()
        embeddings = embeddings[mask]
    else:
        events = events.copy()
    events["channel"] = events["domain"].astype(str).str.upper()
    events["embedding"] = list(embeddings)
    events["anchor_embedding_source"] = "native_multilingual_minilm_query_for_S_caption_for_R"
    return events[
        ["user_id", "timestamp", "channel", "search_session_id", "embedding", "anchor_embedding_source"]
    ].sort_values(["user_id", "timestamp", "channel"], kind="mergesort").reset_index(drop=True)


def validate_alignment(base_df: pd.DataFrame, other_df: pd.DataFrame, variant: str) -> None:
    base_keys = set(zip(base_df["user_id"].tolist(), base_df["sample_index"].tolist()))
    other_keys = set(zip(other_df["user_id"].tolist(), other_df["sample_index"].tolist()))
    if base_keys != other_keys:
        missing = len(base_keys - other_keys)
        extra = len(other_keys - base_keys)
        raise ValueError(f"Alignment mismatch for {variant}: missing={missing}, extra={extra}")
