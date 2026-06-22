from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "full": "Full",
    "no_intent_state": "w/o construction state",
    "no_counterfactual": "w/o construction source",
}
VARIANT_COLORS = {
    "full": "#1f77b4",
    "no_intent_state": "#ff7f0e",
    "no_counterfactual": "#ff7f0e",
}
STATE_VARIANTS = ["full", "no_intent_state"]
ATTR_VARIANTS = ["full", "no_counterfactual"]
TRANSITION_ORDER = ["R->R", "R->S", "S->R", "S->S"]
TRANSITION_METRICS = [
    "intent_shift",
    "adoption_rate",
    "future_consistency",
    "cross_channel_gain",
    "source_prediction_score_gap",
    "anchor_similarity",
]
CONTRIBUTION_GROUP_ORDER = ["Same-channel", "Cross-channel"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "output_ablation"
DEFAULT_STEP4_ROOT = SCRIPT_DIR.parent.parent / "Data" / "Step4"
REQUIRED_ABLATION_COLUMNS = {
    "sample_index",
    "user_id",
    "timestamp",
    "channel",
    "item_id",
    "search_session_id",
    "history_src_share",
    "global_intent_entropy",
    "global_posterior_uncertainty",
    "rec_src_intent_shift_js",
    "attribution_source_proxy",
    "rec_pred_pos_score",
    "src_pred_pos_score",
}
REQUIRED_ABLATION_PREFIXES = (
    "pos_item_emb_",
    "rec_history_mean_emb_",
    "shared_user_feat_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation analysis for PC-SAR exports.")
    parser.add_argument(
        "--input-root",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Directory containing the exported CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to write ablation tables and figures.",
    )
    parser.add_argument(
        "--step4-root",
        type=str,
        default=str(DEFAULT_STEP4_ROOT),
        help="Directory containing Data/Step4 rec_all.pkl and src_all.pkl for full-history anchors.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def sem(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values).dropna()
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def mean_or_nan(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values).dropna()
    return float(series.mean()) if len(series) > 0 else float("nan")


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
    if not REQUIRED_ABLATION_COLUMNS.issubset(columns):
        return False
    if not all(any(col.startswith(prefix) for col in columns) for prefix in REQUIRED_ABLATION_PREFIXES):
        return False
    return any(
        col.startswith("rec_topk_mean_emb_") or col.startswith("rec_user_feat_")
        for col in columns
    )


def discover_variant_files(root: Path) -> dict[str, Path]:
    variant_files: dict[str, Path] = {}
    patterns = [
        "pcsar_intent_features_test*.csv",
        "Features/**/*.csv",
        "Old_features/*.csv",
    ]
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        for path in paths:
            variant = resolve_variant(path)
            if (
                variant is not None
                and variant not in variant_files
                and supports_ablation_schema(path)
            ):
                variant_files[variant] = path
    return variant_files


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "sample_index",
        "user_id",
        "timestamp",
        "history_length",
        "global_history_length",
        "rec_history_length",
        "src_history_length",
        "history_rec_share",
        "history_src_share",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
        "global_belief_uncertainty_mean",
        "global_belief_confidence_mean",
        "global_attention_temp_mean",
        "rec_dominant_intent_prob",
        "rec_intent_entropy",
        "rec_posterior_uncertainty",
        "rec_belief_uncertainty_mean",
        "rec_belief_confidence_mean",
        "rec_attention_temp_mean",
        "src_dominant_intent_prob",
        "src_intent_entropy",
        "src_posterior_uncertainty",
        "src_belief_uncertainty_mean",
        "src_belief_confidence_mean",
        "src_attention_temp_mean",
        "attribution_confidence_gap",
        "attribution_entropy_gap",
        "rec_src_intent_shift_dot",
        "rec_src_intent_shift_js",
        "rec_pred_top1_score",
        "rec_pred_pos_score",
        "rec_pred_pos_rank",
        "src_pred_top1_score",
        "src_pred_pos_score",
        "src_pred_pos_rank",
        "use_counterfactual",
        "use_intent_logit_bias",
        "use_uncertainty_attention",
    ]
    df = df.copy()
    df = coerce_numeric(df, numeric_cols)
    sort_cols = ["user_id"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def vector_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    def suffix_value(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(cols, key=suffix_value)


def row_vector(row: pd.Series, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.array([], dtype=np.float64)
    return row[cols].to_numpy(dtype=np.float64)


def row_cos(row_a: pd.Series, row_b: pd.Series, cols_a: list[str], cols_b: list[str] | None = None) -> float:
    if cols_b is None:
        cols_b = cols_a
    return safe_cosine(row_vector(row_a, cols_a), row_vector(row_b, cols_b))


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = np.isfinite(values).all(axis=1, keepdims=True) & (norms > 1e-12)
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values, norms, out=out, where=valid)
    return out


def event_embedding_matrix(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return one comparable embedding per sample.

    R events use the clicked item embedding. S events prefer the query embedding
    when it is exported; otherwise they fall back to the clicked item embedding.
    """
    pos_cols = vector_columns(df, "pos_item_emb_")
    query_cols = vector_columns(df, "query_emb_")
    if not pos_cols:
        raise KeyError("Missing pos_item_emb_* columns required for semantic anchors.")

    pos = df[pos_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    emb = pos.copy()
    source = "pos_item_emb"

    if query_cols:
        query = df[query_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        query_valid = np.linalg.norm(query, axis=1) > 1e-12
        is_search = df["channel"].astype(str).str.upper().to_numpy() == "S"
        use_query = is_search & query_valid
        emb[use_query] = query[use_query]
        source = "query_emb_for_S_pos_item_emb_for_R"

    return normalize_matrix(emb), source


def embedding_maps(df: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], str]:
    emb, source = event_embedding_matrix(df)
    item_map: dict[int, np.ndarray] = {}
    session_map: dict[int, np.ndarray] = {}

    item_ids = pd.to_numeric(df.get("item_id", pd.Series(dtype=float)), errors="coerce")
    for pos, item_id in enumerate(item_ids):
        if pd.notna(item_id) and np.any(emb[pos]):
            item_map.setdefault(int(item_id), emb[pos])

    query_cols = vector_columns(df, "query_emb_")
    if query_cols and "search_session_id" in df.columns:
        query = df[query_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        query = normalize_matrix(query)
        session_ids = pd.to_numeric(df["search_session_id"], errors="coerce")
        is_search = df["channel"].astype(str).str.upper().to_numpy() == "S"
        for pos, session_id in enumerate(session_ids):
            if is_search[pos] and pd.notna(session_id) and np.any(query[pos]):
                session_map.setdefault(int(session_id), query[pos])

    return item_map, session_map, source


def build_step4_anchor_pool(base_df: pd.DataFrame, step4_root: Path) -> pd.DataFrame:
    rec_path = step4_root / "rec_all.pkl"
    src_path = step4_root / "src_all.pkl"
    if not rec_path.exists() or not src_path.exists():
        return pd.DataFrame()

    item_map, session_map, embedding_source = embedding_maps(base_df)
    users = set(pd.to_numeric(base_df["user_id"], errors="coerce").dropna().astype(int))
    rows = []

    rec = pd.read_pickle(rec_path)[["user_id", "item_id", "timestamp"]]
    rec = rec[rec["user_id"].isin(users)]
    for row in rec.itertuples(index=False):
        item_id = int(row.item_id)
        emb = item_map.get(item_id)
        if emb is None:
            continue
        rows.append(
            {
                "user_id": int(row.user_id),
                "timestamp": float(row.timestamp),
                "channel": "R",
                "item_id": item_id,
                "search_session_id": np.nan,
                "embedding": emb,
                "anchor_embedding_source": f"step4_item_lookup:{embedding_source}",
            }
        )

    src = pd.read_pickle(src_path)[["user_id", "item_id", "timestamp", "search_session_id"]]
    src = src[src["user_id"].isin(users)]
    for row in src.itertuples(index=False):
        item_id = int(row.item_id)
        session_id = int(row.search_session_id)
        emb = session_map.get(session_id)
        source = f"step4_query_lookup:{embedding_source}"
        if emb is None:
            emb = item_map.get(item_id)
            source = f"step4_item_fallback:{embedding_source}"
        if emb is None:
            continue
        rows.append(
            {
                "user_id": int(row.user_id),
                "timestamp": float(row.timestamp),
                "channel": "S",
                "item_id": item_id,
                "search_session_id": session_id,
                "embedding": emb,
                "anchor_embedding_source": source,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["user_id", "timestamp", "channel"], kind="mergesort").reset_index(drop=True)


def build_state_events(base_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        if "channel" in g.columns and (g["channel"].astype(str).str.upper() == "S").any():
            candidates = g[g["channel"].astype(str).str.upper() == "S"].copy()
        else:
            candidates = g.copy()

        score = (
            candidates["history_src_share"].fillna(0.0)
            + 0.25 * candidates["global_intent_entropy"].fillna(0.0)
            + 0.25 * candidates["global_posterior_uncertainty"].fillna(0.0)
        )
        shock_pos = int(score.idxmax())
        post_candidates = g.iloc[shock_pos + 1 :]
        post_candidates = post_candidates[post_candidates["channel"].astype(str).str.upper() == "R"]
        if post_candidates.empty:
            continue
        post_pos = int(post_candidates.index[0])

        shock = g.iloc[shock_pos]
        post = g.iloc[post_pos]
        rows.append(
            {
                "user_id": int(user_id),
                "shock_sample_index": int(shock["sample_index"]),
                "post_sample_index": int(post["sample_index"]),
                "shock_channel": str(shock["channel"]),
                "post_channel": str(post["channel"]),
                "shock_score": float(score.loc[shock_pos]),
            }
        )
    return pd.DataFrame(rows)


def build_transition_events(base_df: pd.DataFrame, step4_root: Path | None = None) -> pd.DataFrame:
    event_emb, embedding_source = event_embedding_matrix(base_df)
    anchor_pool = build_step4_anchor_pool(base_df, step4_root) if step4_root is not None else pd.DataFrame()
    use_step4_pool = not anchor_pool.empty
    rows = []
    for user_id, g in base_df.groupby("user_id", sort=False):
        original_index = g.index.to_numpy()
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        emb = event_emb[original_index]
        channels = g["channel"].astype(str).str.upper().to_numpy()
        user_anchor_pool = pd.DataFrame()
        if use_step4_pool:
            user_anchor_pool = anchor_pool[anchor_pool["user_id"] == int(user_id)].reset_index(drop=True)
        for i in range(len(g)):
            cur = g.iloc[i]
            cur_channel = channels[i]
            if cur_channel not in {"R", "S"}:
                continue
            if not np.any(emb[i]):
                continue

            future_index = int(g.iloc[i + 1]["sample_index"]) if i + 1 < len(g) else np.nan
            exploration_score = float(
                cur.get("history_src_share", 0.0)
                + 0.25 * cur.get("global_intent_entropy", 0.0)
                + 0.25 * cur.get("global_posterior_uncertainty", 0.0)
            )

            for anchor_channel in ("R", "S"):
                if use_step4_pool and not user_anchor_pool.empty:
                    candidates = user_anchor_pool[
                        (user_anchor_pool["channel"] == anchor_channel)
                        & (user_anchor_pool["timestamp"] < float(cur.get("timestamp", np.nan)))
                    ]
                    if candidates.empty:
                        continue
                    candidate_emb = np.vstack(candidates["embedding"].to_numpy())
                    similarities = candidate_emb @ emb[i]
                    best_pos = int(np.argmax(similarities))
                    anchor_row = candidates.iloc[best_pos]
                    anchor_sample_index = np.nan
                    anchor_item_id = anchor_row.get("item_id", np.nan)
                    anchor_search_session_id = anchor_row.get("search_session_id", np.nan)
                    anchor_timestamp = float(anchor_row.get("timestamp", np.nan))
                    anchor_similarity = float(similarities[best_pos])
                    anchor_gap = int(
                        (
                            (user_anchor_pool["timestamp"] > anchor_timestamp)
                            & (user_anchor_pool["timestamp"] < float(cur.get("timestamp", np.nan)))
                        ).sum()
                        + 1
                    )
                    anchor_source = str(anchor_row.get("anchor_embedding_source", embedding_source))
                else:
                    prior = np.arange(i)
                    anchor_candidates = prior[channels[:i] == anchor_channel]
                    if len(anchor_candidates) == 0:
                        continue
                    candidate_emb = emb[anchor_candidates]
                    valid = np.linalg.norm(candidate_emb, axis=1) > 1e-12
                    if not valid.any():
                        continue
                    valid_candidates = anchor_candidates[valid]
                    similarities = candidate_emb[valid] @ emb[i]
                    best_pos = int(np.argmax(similarities))
                    anchor_index = int(valid_candidates[best_pos])
                    anchor_row = g.iloc[int(anchor_index)]
                    anchor_sample_index = int(anchor_row["sample_index"])
                    anchor_item_id = anchor_row.get("item_id", np.nan)
                    anchor_search_session_id = anchor_row.get("search_session_id", np.nan)
                    anchor_timestamp = float(anchor_row.get("timestamp", np.nan))
                    anchor_similarity = float(similarities[best_pos])
                    anchor_gap = int(i - int(anchor_index))
                    anchor_source = embedding_source
                rows.append(
                    {
                        "user_id": int(user_id),
                        "anchor_sample_index": anchor_sample_index,
                        "anchor_item_id": anchor_item_id,
                        "anchor_search_session_id": anchor_search_session_id,
                        "anchor_timestamp": anchor_timestamp,
                        "sample_index": int(cur["sample_index"]),
                        "future_sample_index": future_index,
                        "anchor_channel": anchor_channel,
                        "current_channel": cur_channel,
                        "transition_type": f"{anchor_channel}->{cur_channel}",
                        "exploration_score": exploration_score,
                        "anchor_gap": anchor_gap,
                        "anchor_similarity": anchor_similarity,
                        "anchor_embedding_source": anchor_source,
                    }
                )
    return pd.DataFrame(rows)


def get_idx(df: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    return {
        (int(row["user_id"]), int(row["sample_index"])): row
        for _, row in df.iterrows()
    }


def get_row(idx: dict[tuple[int, int], pd.Series], user_id: int, sample_index: int) -> pd.Series | None:
    key = (int(user_id), int(sample_index))
    return idx.get(key)


def evaluate_state_events(events: pd.DataFrame, df: pd.DataFrame, variant: str) -> pd.DataFrame:
    idx = get_idx(df)
    rec_feat_cols = vector_columns(df, "rec_user_feat_")
    hist_cols = vector_columns(df, "rec_history_mean_emb_")
    topk_cols = vector_columns(df, "rec_topk_mean_emb_") or rec_feat_cols
    shock_item_cols = vector_columns(df, "pos_item_emb_")

    rows = []
    for _, event in events.iterrows():
        shock = get_row(idx, int(event["user_id"]), int(event["shock_sample_index"]))
        post = get_row(idx, int(event["user_id"]), int(event["post_sample_index"]))
        if shock is None or post is None:
            continue

        post_rec = row_vector(post, topk_cols)
        shock_rec = row_vector(shock, topk_cols)
        shock_item = row_vector(shock, shock_item_cols)
        shock_history = row_vector(shock, hist_cols)
        post_to_shock = safe_cosine(post_rec, shock_item)
        post_to_history = safe_cosine(post_rec, shock_history)
        shock_resistance = post_to_history - post_to_shock
        preference_margin = post_to_history - safe_cosine(shock_rec, shock_item)
        rows.append(
            {
                "variant": variant,
                "user_id": int(event["user_id"]),
                "shock_sample_index": int(event["shock_sample_index"]),
                "post_sample_index": int(event["post_sample_index"]),
                "shock_channel": event["shock_channel"],
                "post_channel": event["post_channel"],
                "shock_score": float(event["shock_score"]),
                "post_to_shock": post_to_shock,
                "post_to_history": post_to_history,
                "shock_resistance": shock_resistance,
                "preference_margin": preference_margin,
            }
        )
    return pd.DataFrame(rows)


def evaluate_transition_events(events: pd.DataFrame, df: pd.DataFrame, variant: str) -> pd.DataFrame:
    idx = get_idx(df)
    shared_cols = vector_columns(df, "shared_user_feat_")

    rows = []
    for _, event in events.iterrows():
        cur = get_row(idx, int(event["user_id"]), int(event["sample_index"]))
        if cur is None:
            continue
        fut = None
        if not pd.isna(event["future_sample_index"]):
            fut = get_row(idx, int(event["user_id"]), int(event["future_sample_index"]))

        shared_cur = row_vector(cur, shared_cols)
        future_consistency = (
            safe_cosine(shared_cur, row_vector(fut, shared_cols)) if fut is not None else np.nan
        )

        adoption_rate = float(
            str(cur.get("attribution_source_proxy", "")).upper()
            == str(cur.get("channel", "")).upper()
        )
        channel = str(cur.get("channel", "")).upper()
        rec_pos_score = float(cur.get("rec_pred_pos_score", np.nan))
        src_pos_score = float(cur.get("src_pred_pos_score", np.nan))
        cross_channel_gain = rec_pos_score - src_pos_score
        if channel == "R":
            same_source_score = rec_pos_score
            cross_source_score = src_pos_score
        elif channel == "S":
            same_source_score = src_pos_score
            cross_source_score = rec_pos_score
        else:
            same_source_score = np.nan
            cross_source_score = np.nan
        same_minus_cross_contribution = same_source_score - cross_source_score
        is_cross_transition = "->" in str(event["transition_type"]) and (
            str(event["transition_type"]).split("->")[0] != str(event["transition_type"]).split("->")[1]
        )
        if is_cross_transition:
            relevant_source_contribution = cross_source_score
            irrelevant_source_contribution = same_source_score
        else:
            relevant_source_contribution = same_source_score
            irrelevant_source_contribution = cross_source_score
        source_prediction_score_gap = abs(relevant_source_contribution - irrelevant_source_contribution)

        rows.append(
            {
                "variant": variant,
                "user_id": int(event["user_id"]),
                "sample_index": int(event["sample_index"]),
                "timestamp": float(cur.get("timestamp", np.nan)),
                "item_id": int(cur.get("item_id", -1)) if pd.notna(cur.get("item_id", np.nan)) else np.nan,
                "search_session_id": int(cur.get("search_session_id", -1)) if pd.notna(cur.get("search_session_id", np.nan)) else np.nan,
                "channel": channel,
                "transition_type": event["transition_type"],
                "transition_group": "Cross-channel" if is_cross_transition else "Same-channel",
                "exploration_score": float(event["exploration_score"]),
                "anchor_sample_index": (
                    int(event["anchor_sample_index"]) if pd.notna(event["anchor_sample_index"]) else np.nan
                ),
                "anchor_item_id": event.get("anchor_item_id", np.nan),
                "anchor_search_session_id": event.get("anchor_search_session_id", np.nan),
                "anchor_timestamp": event.get("anchor_timestamp", np.nan),
                "anchor_channel": str(event["anchor_channel"]),
                "anchor_gap": int(event["anchor_gap"]),
                "anchor_similarity": float(event["anchor_similarity"]),
                "anchor_embedding_source": str(event.get("anchor_embedding_source", "")),
                "intent_shift": float(cur.get("rec_src_intent_shift_js", np.nan)),
                "adoption_rate": adoption_rate,
                "future_consistency": future_consistency,
                "cross_channel_gain": cross_channel_gain,
                "same_source_score": same_source_score,
                "cross_source_score": cross_source_score,
                "same_minus_cross_contribution": same_minus_cross_contribution,
                "relevant_source_contribution": relevant_source_contribution,
                "irrelevant_source_contribution": irrelevant_source_contribution,
                "source_prediction_score_gap": source_prediction_score_gap,
            }
        )

    return pd.DataFrame(rows)


def attach_raw_transition_labels(transition_df: pd.DataFrame, raw_trajectory_path: Path) -> pd.DataFrame:
    """Legacy helper for adjacent-transition labels.

    Semantic-anchor transitions should not call this by default because raw
    adjacent labels would overwrite the embedding-nearest anchor definition.
    """
    if not raw_trajectory_path.exists() or transition_df.empty:
        transition_df = transition_df.copy()
        transition_df["transition_type_plot"] = transition_df.get("transition_type", pd.Series(dtype=str))
        return transition_df

    raw = pd.read_csv(raw_trajectory_path, low_memory=False)
    required = {"user_id", "timestamp", "channel"}
    if not required.issubset(raw.columns):
        transition_df = transition_df.copy()
        transition_df["transition_type_plot"] = transition_df.get("transition_type", pd.Series(dtype=str))
        return transition_df

    raw = raw.copy()
    raw["channel"] = raw["channel"].astype(str).str.upper()
    raw["channel_order"] = raw["channel"].map({"R": 0, "S": 1}).fillna(99).astype(int)
    raw["item_id_filled"] = pd.to_numeric(raw.get("item_id", np.nan), errors="coerce").fillna(-1).astype(int)
    raw["search_session_id_filled"] = pd.to_numeric(
        raw.get("search_session_id", np.nan), errors="coerce"
    ).fillna(-1).astype(int)
    raw = raw.sort_values(
        ["user_id", "timestamp", "channel_order", "item_id_filled", "search_session_id_filled"],
        kind="mergesort",
    ).reset_index(drop=True)
    raw["prev_channel"] = raw.groupby("user_id")["channel"].shift(1)
    raw["raw_transition_type"] = raw["prev_channel"].fillna("") + "->" + raw["channel"]
    raw = raw[raw["prev_channel"].notna()].copy()
    raw = raw.rename(
        columns={
            "timestamp": "timestamp_raw",
            "item_id_filled": "item_id_merge",
            "search_session_id_filled": "search_session_id_merge",
        }
    )

    merged = transition_df.copy()
    merged["item_id_merge"] = pd.to_numeric(merged.get("item_id", np.nan), errors="coerce").fillna(-1).astype(int)
    merged["search_session_id_merge"] = pd.to_numeric(
        merged.get("search_session_id", np.nan), errors="coerce"
    ).fillna(-1).astype(int)
    merged["timestamp_merge"] = pd.to_numeric(merged.get("timestamp", np.nan), errors="coerce")

    raw = raw[[
        "user_id",
        "timestamp_raw",
        "item_id_merge",
        "search_session_id_merge",
        "channel",
        "raw_transition_type",
    ]]
    merged = merged.merge(
        raw,
        on=["user_id", "item_id_merge", "search_session_id_merge", "channel"],
        how="left",
    )

    matched = int(merged["raw_transition_type"].notna().sum())
    total = int(len(merged))
    if total > 0 and matched / total < 0.5:
        print(f"[warn] raw transition relabel match rate low: {matched}/{total}")

    merged["transition_type_plot"] = merged["raw_transition_type"].fillna(merged["transition_type"])
    return merged


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, g in df.groupby(grouper, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for metric in metrics:
            row[f"{metric}_mean"] = mean_or_nan(g[metric])
            row[f"{metric}_sem"] = sem(g[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def plot_state_resistance_panel(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(STATE_VARIANTS))
    labels = [VARIANT_LABELS[v] for v in STATE_VARIANTS]
    metric = "shock_resistance"
    title = "Exploration Resistance"
    means = []
    sems = []
    for variant in STATE_VARIANTS:
        cur = df[df["variant"] == variant]
        means.append(float(np.nanmean(cur[metric])) if not cur.empty else np.nan)
        sems.append(sem(cur[metric]) if not cur.empty else np.nan)
    ax.bar(x, means, color=[VARIANT_COLORS[v] for v in STATE_VARIANTS], alpha=0.88)
    ax.errorbar(x, means, yerr=1.96 * np.asarray(sems), fmt="none", ecolor="#333333", capsize=4)
    ax.scatter(x, means, s=24, color="#111111", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_title(title, loc="center", fontsize=11, weight="normal")
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_source_prediction_panel(ax, df: pd.DataFrame) -> None:
    plot_df = summarize_prediction_score_gap(df)
    if plot_df.empty:
        ax.set_visible(False)
        return
    groups = [group for group in CONTRIBUTION_GROUP_ORDER if group in set(plot_df["transition_group"])]
    x = np.arange(len(groups))
    width = 0.36
    for idx, variant in enumerate([variant for variant in ATTR_VARIANTS if variant in set(plot_df["variant"])]):
        g = plot_df[plot_df["variant"] == variant].set_index("transition_group").reindex(groups)
        means = g["source_prediction_score_gap_mean"].to_numpy(dtype=float)
        sems = g["source_prediction_score_gap_sem"].to_numpy(dtype=float)
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            means,
            width=width,
            color=VARIANT_COLORS[variant],
            alpha=0.88,
            label=VARIANT_LABELS[variant],
        )
        ax.errorbar(xpos, means, yerr=1.96 * sems, fmt="none", ecolor="#333333", capsize=4)
        ax.scatter(xpos, means, s=24, color="#111111", zorder=3)
    ax.set_xticks(x, groups)
    ax.set_title("Channel Prediction Score Gap", loc="center", fontsize=11, weight="normal")
    ax.set_ylabel("Channel Prediction Score Gap")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)


def plot_state_summary(df: pd.DataFrame, out_path: Path, transition_df: pd.DataFrame | None = None) -> None:
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    plot_state_resistance_panel(axes[0], df)
    if transition_df is None:
        metric = "preference_margin"
        title = "Post-exploration preference margin"
        x = np.arange(len(STATE_VARIANTS))
        labels = [VARIANT_LABELS[v] for v in STATE_VARIANTS]
        means = []
        sems = []
        for variant in STATE_VARIANTS:
            cur = df[df["variant"] == variant]
            means.append(float(np.nanmean(cur[metric])) if not cur.empty else np.nan)
            sems.append(sem(cur[metric]) if not cur.empty else np.nan)
        axes[1].bar(x, means, color=[VARIANT_COLORS[v] for v in STATE_VARIANTS], alpha=0.88)
        axes[1].errorbar(x, means, yerr=1.96 * np.asarray(sems), fmt="none", ecolor="#333333", capsize=4)
        axes[1].scatter(x, means, s=24, color="#111111", zorder=3)
        axes[1].set_xticks(x, labels)
        axes[1].set_title(title, loc="center", fontsize=11, weight="normal")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
    else:
        plot_source_prediction_panel(axes[1], transition_df)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_prediction_score_gap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df.dropna(subset=["source_prediction_score_gap", "transition_group"]).copy()
    for (variant, transition_group), g in valid.groupby(["variant", "transition_group"], sort=False):
        rows.append(
            {
                "variant": variant,
                "transition_group": transition_group,
                "n": int(len(g)),
                "source_prediction_score_gap_mean": mean_or_nan(g["source_prediction_score_gap"]),
                "source_prediction_score_gap_sem": sem(g["source_prediction_score_gap"]),
                "relevant_source_contribution_mean": mean_or_nan(g["relevant_source_contribution"]),
                "relevant_source_contribution_sem": sem(g["relevant_source_contribution"]),
                "irrelevant_source_contribution_mean": mean_or_nan(g["irrelevant_source_contribution"]),
                "irrelevant_source_contribution_sem": sem(g["irrelevant_source_contribution"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_variant_order"] = out["variant"].map({variant: idx for idx, variant in enumerate(ATTR_VARIANTS)}).fillna(99)
        out["_group_order"] = out["transition_group"].map(
            {group: idx for idx, group in enumerate(CONTRIBUTION_GROUP_ORDER)}
        ).fillna(99)
        out = out.sort_values(["_variant_order", "_group_order"]).drop(
            columns=["_variant_order", "_group_order"]
        ).reset_index(drop=True)
    return out


def plot_transition_summary(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.9), constrained_layout=True)
    plot_source_prediction_panel(ax, df)
    ax.set_xlabel("Transition Group")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_transition_gap(df: pd.DataFrame, transition_col: str = "transition_type") -> pd.DataFrame:
    rows = []
    for variant, g in df.groupby("variant", sort=False):
        if variant not in VARIANT_LABELS:
            continue
        pivot = summarize(g, [transition_col], TRANSITION_METRICS).set_index(transition_col)
        for metric in TRANSITION_METRICS:
            rs = float(pivot.loc["R->S", f"{metric}_mean"]) if "R->S" in pivot.index else np.nan
            sr = float(pivot.loc["S->R", f"{metric}_mean"]) if "S->R" in pivot.index else np.nan
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "r_to_s_mean": rs,
                    "s_to_r_mean": sr,
                    "transition_gap": rs - sr if np.isfinite(rs) and np.isfinite(sr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_alignment(base_df: pd.DataFrame, other_df: pd.DataFrame, variant: str) -> None:
    base_keys = set(zip(base_df["user_id"].tolist(), base_df["sample_index"].tolist()))
    other_keys = set(zip(other_df["user_id"].tolist(), other_df["sample_index"].tolist()))
    if base_keys != other_keys:
        missing = len(base_keys - other_keys)
        extra = len(other_keys - base_keys)
        print(f"[warn] alignment mismatch for {variant}: missing={missing}, extra={extra}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    variant_files = discover_variant_files(input_root)
    required = {"full", "no_intent_state", "no_counterfactual"}
    missing = sorted(required - set(variant_files))
    if missing:
        raise FileNotFoundError(f"Missing variant CSV(s): {missing}")

    frames = {
        variant: prepare_frame(pd.read_csv(path, low_memory=False))
        for variant, path in variant_files.items()
    }
    base_df = frames["full"]
    for variant, df in frames.items():
        validate_alignment(base_df, df, variant)

    state_events = build_state_events(base_df)
    transition_events = build_transition_events(base_df, Path(args.step4_root))

    state_tables = []
    for variant in STATE_VARIANTS:
        state_tables.append(evaluate_state_events(state_events, frames[variant], variant))
    state_df = pd.concat(state_tables, ignore_index=True)
    state_summary = summarize(
        state_df,
        ["variant"],
        [
            "post_to_shock",
            "post_to_history",
            "shock_resistance",
            "preference_margin",
        ],
    )
    save_table(state_summary, output_root / "state_summary.csv")

    transition_tables = []
    for variant in ATTR_VARIANTS:
        transition_tables.append(evaluate_transition_events(transition_events, frames[variant], variant))
    transition_df = pd.concat(transition_tables, ignore_index=True)
    transition_df["transition_type_plot"] = transition_df["transition_type"]
    transition_group_col = "transition_type_plot" if "transition_type_plot" in transition_df.columns else "transition_type"
    transition_summary = summarize(transition_df, ["variant", transition_group_col], TRANSITION_METRICS)
    transition_summary = transition_summary.rename(columns={transition_group_col: "transition_type"})
    save_table(transition_summary, output_root / "transition_summary.csv")
    transition_prediction_score_gap_summary = summarize_prediction_score_gap(transition_df)
    save_table(
        transition_prediction_score_gap_summary,
        output_root / "transition_prediction_score_gap_summary.csv",
    )
    transition_gap = build_transition_gap(transition_df, transition_col=transition_group_col)
    save_table(transition_gap, output_root / "transition_gap.csv")
    plot_state_summary(
        state_df,
        output_root / "state_attribution_summary.png",
        transition_df=transition_df,
    )

    print(f"Saved ablation analysis under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
