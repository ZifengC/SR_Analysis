from __future__ import annotations

import argparse
import ast
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
DATASET_ROOT = DATA_ROOT / "dataset"
STEP1_ROOT = DATA_ROOT / "Step1"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select trajectories that switch repeatedly between rec and src."
    )
    parser.add_argument(
        "--min-switches",
        type=int,
        default=6,
        help="Minimum number of rec/src switches required to keep a user.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=12,
        help="Minimum number of collapsed timeline events required to keep a user.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Maximum number of users to export after ranking by alternation score.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Directory where the text timeline and CSV summary will be written.",
    )
    parser.add_argument(
        "--timeline-name",
        type=str,
        default="selected_samples.txt",
        help="Filename for the readable timeline export.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="selected_samples.csv",
        help="Filename for the per-user summary CSV.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frames(names: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_pickle(DATASET_ROOT / name) for name in names]
    return pd.concat(frames, ignore_index=True)


def build_decoders() -> tuple[dict[int, str], dict[int, list[int]]]:
    vocab = load_pickle(STEP1_ROOT / "vocab_dict.pkl")
    rev_vocab = {int(v): str(k) for k, v in vocab.items()}

    note_feat = load_pickle(STEP1_ROOT / "note_feat.pkl")
    caption_map: dict[int, list[int]] = {}
    for _, row in note_feat.iterrows():
        note_id = int(row["note_id"])
        caption = row["caption"]
        if caption is None:
            caption = []
        elif isinstance(caption, float) and pd.isna(caption):
            caption = []
        elif isinstance(caption, str):
            try:
                caption = ast.literal_eval(caption)
            except Exception:
                caption = []
        elif not isinstance(caption, (list, tuple)):
            try:
                caption = list(caption)
            except Exception:
                caption = []
        caption_map[note_id] = [int(x) for x in caption]
    return rev_vocab, caption_map


def decode_tokens(tokens: Iterable[int], rev_vocab: dict[int, str], unk: str = "<UNK>") -> str:
    parts: list[str] = []
    for token in tokens:
        token_id = int(token)
        if token_id == 0:
            continue
        parts.append(rev_vocab.get(token_id, unk))
    return "".join(parts)


def decode_query(raw, rev_vocab: dict[int, str]) -> str:
    if isinstance(raw, str):
        try:
            tokens = ast.literal_eval(raw)
        except Exception:
            tokens = []
    else:
        tokens = raw or []
    return decode_tokens(tokens, rev_vocab)


def decode_caption(note_id: int, caption_map: dict[int, list[int]], rev_vocab: dict[int, str]) -> str:
    tokens = caption_map.get(int(note_id), [])
    if not tokens:
        return ""
    return decode_tokens(tokens, rev_vocab)


def ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def build_rec_events(df: pd.DataFrame, rev_vocab: dict[int, str], caption_map: dict[int, list[int]]) -> pd.DataFrame:
    out = df.copy()
    out["channel"] = "rec"
    out["text"] = out["item_id"].apply(lambda x: decode_caption(int(x), caption_map, rev_vocab))
    out["session_id"] = -1
    out["search_query"] = ""
    return out[["user_id", "timestamp", "channel", "search_query", "text", "item_id", "session_id"]]


def build_src_session_events(df: pd.DataFrame, rev_vocab: dict[int, str], caption_map: dict[int, list[int]]) -> pd.DataFrame:
    src = df.copy()
    src["query_text"] = src["keyword"].apply(lambda x: decode_query(x, rev_vocab))
    src["item_text"] = src["item_id"].apply(lambda x: decode_caption(int(x), caption_map, rev_vocab))

    rows = []
    for (user_id, session_id), g in src.groupby(["user_id", "search_session_id"], sort=False):
        g = g.sort_values(["timestamp", "item_id"], kind="mergesort")
        query_text = next((q for q in g["query_text"].tolist() if q), "")
        item_texts = [t for t in g["item_text"].tolist() if t]
        rows.append(
            {
                "user_id": int(user_id),
                "timestamp": float(g["timestamp"].min()),
                "channel": "search",
                "search_query": query_text,
                "text": "; ".join(item_texts),
                "item_id": int(g["item_id"].iloc[0]) if len(g) else -1,
                "session_id": int(session_id),
                "item_count": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def build_timeline(rec_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
    rec_events = rec_df.copy()
    rec_events["item_count"] = 1

    src_events = src_df.copy()
    if "item_count" not in src_events.columns:
        src_events["item_count"] = 1

    events = pd.concat([rec_events, src_events], ignore_index=True, sort=False)
    events["channel_order"] = events["channel"].map({"rec": 0, "search": 1}).fillna(99).astype(int)
    events = events.sort_values(
        ["user_id", "timestamp", "channel_order", "session_id", "item_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return events


def count_switches(channels: list[str]) -> int:
    if len(channels) < 2:
        return 0
    return sum(prev != cur for prev, cur in zip(channels[:-1], channels[1:]))


def select_users(events: pd.DataFrame, min_switches: int, min_events: int, top_n: int) -> pd.DataFrame:
    rows = []
    for user_id, g in events.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        channels = g["channel"].tolist()
        unique_channels = set(channels)
        if len(g) < min_events or len(unique_channels) < 2:
            continue
        switches = count_switches(channels)
        if switches < min_switches:
            continue

        rec_count = int((g["channel"] == "rec").sum())
        search_count = int((g["channel"] == "search").sum())
        alternation_rate = switches / max(len(g) - 1, 1)
        balance = min(rec_count, search_count) / max(max(rec_count, search_count), 1)
        score = switches * alternation_rate * (0.5 + 0.5 * balance)
        rows.append(
            {
                "user_id": int(user_id),
                "event_count": int(len(g)),
                "rec_count": rec_count,
                "search_count": search_count,
                "switch_count": int(switches),
                "alternation_rate": float(alternation_rate),
                "balance": float(balance),
                "score": float(score),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        ["score", "switch_count", "event_count", "user_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if top_n > 0:
        summary = summary.head(top_n)
    return summary


def format_search_line(row: pd.Series) -> str:
    query = str(row.get("search_query", "")).strip()
    items = str(row.get("text", "")).strip()
    if not items:
        items = "<EMPTY>"
    if query and items:
        return f"[{ts_to_str(row['timestamp'])}] (search) {query} | items: {items}"
    if query:
        return f"[{ts_to_str(row['timestamp'])}] (search) {query}"
    return f"[{ts_to_str(row['timestamp'])}] (search) {items}"


def format_rec_line(row: pd.Series) -> str:
    text = str(row.get("text", "")).strip() or "<EMPTY>"
    return f"[{ts_to_str(row['timestamp'])}] (rec) {text}"


def write_outputs(events: pd.DataFrame, summary: pd.DataFrame, out_root: Path, timeline_name: str, summary_name: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    timeline_path = out_root / timeline_name
    summary_path = out_root / summary_name

    selected_users = set(summary["user_id"].tolist()) if not summary.empty else set()
    selected_events = events[events["user_id"].isin(selected_users)].copy()

    lines: list[str] = []
    for _, user_summary in summary.iterrows():
        user_id = int(user_summary["user_id"])
        g = selected_events[selected_events["user_id"] == user_id].sort_values(
            ["timestamp", "channel_order", "session_id", "item_id"],
            kind="mergesort",
        )
        lines.append(
            "=== user {uid} | score={score:.4f} | switches={switches} | events={events} | rec={rec} | search={search} ===".format(
                uid=user_id,
                score=float(user_summary["score"]),
                switches=int(user_summary["switch_count"]),
                events=int(user_summary["event_count"]),
                rec=int(user_summary["rec_count"]),
                search=int(user_summary["search_count"]),
            )
        )
        for _, row in g.iterrows():
            if row["channel"] == "search":
                lines.append(format_search_line(row))
            else:
                lines.append(format_rec_line(row))
        lines.append("")

    timeline_path.write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Wrote timeline: {timeline_path}")
    print(f"Wrote summary:   {summary_path}")
    print(f"Selected users:  {len(summary)}")


def main() -> None:
    args = parse_args()
    rev_vocab, caption_map = build_decoders()

    rec = load_frames(["rec_train.pkl", "rec_val.pkl", "rec_test.pkl"])
    src = load_frames(["src_train.pkl", "src_val.pkl", "src_test.pkl"])

    rec_events = build_rec_events(rec, rev_vocab, caption_map)
    src_events = build_src_session_events(src, rev_vocab, caption_map)
    events = build_timeline(rec_events, src_events)

    summary = select_users(events, args.min_switches, args.min_events, args.top_n)
    write_outputs(events, summary, Path(args.output_root), args.timeline_name, args.summary_name)


if __name__ == "__main__":
    main()
