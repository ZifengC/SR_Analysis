"""
Export user timelines for UP (rank_diff <= -1) and DOWN (rank_diff >= 1)
cases, in a readable chronological format.

Run from Results-Analysis/:
    python 1-Compare-baseline/Code/3-export_timelines_txt.py

Inputs:
    1-Compare-baseline/New_results/3-rank_diff_vs_UNISAR_sorted.csv
    Data/dataset/src_train.pkl
    Data/dataset/src_val.pkl
    Data/dataset/src_test.pkl
    Data/dataset/rec_train.pkl
    Data/dataset/rec_val.pkl
    Data/dataset/rec_test.pkl
    Data/Step1/vocab_dict.pkl
    Data/Step1/note_feat.pkl

Output:
    1-Compare-baseline/New_results/4-extreme_timelines_global.txt
"""

import ast
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Results-Analysis/ contains shared Data/ and this pipeline folder.
BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parent
DATA = PROJECT_ROOT / "Data"
NEW_RESULTS = BASELINE_ROOT / "New_results"
RANK_DIFF_INPUT = NEW_RESULTS / "3-rank_diff_vs_UNISAR_sorted.csv"
TIMELINE_OUTPUT = NEW_RESULTS / "4-extreme_timelines_global.txt"


def load_decoders():
    vocab = pickle.load(open(DATA / "Step1" / "vocab_dict.pkl", "rb"))
    rev_vocab = {v: k for k, v in vocab.items()}
    note_feat = pickle.load(open(DATA / "Step1" / "note_feat.pkl", "rb"))
    caption_map = dict(zip(note_feat["note_id"], note_feat["caption"]))
    return rev_vocab, caption_map


def decode(tokens, rev_vocab, unk="<UNK>"):
    return "".join(rev_vocab.get(int(t), unk) for t in tokens if int(t) != 0)


def decode_keyword(raw, rev_vocab):
    if isinstance(raw, str):
        try:
            tokens = ast.literal_eval(raw)
        except Exception:
            tokens = []
    else:
        tokens = raw
    return decode(tokens or [], rev_vocab)


def ts_to_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def load_split(name):
    return pickle.load(open(DATA / "dataset" / name, "rb"))


def build_search_events(df, rev_vocab, caption_map):
    df = df.copy()
    df["query_text"] = df["keyword"].apply(lambda k: decode_keyword(k, rev_vocab))
    df["item_caption"] = df["item_id"].apply(
        lambda i: decode(caption_map.get(int(i), []), rev_vocab)
    )
    df["event"] = "search"
    return df[["user_id", "timestamp", "query_text", "item_caption", "event"]]


def build_rec_events(df, rev_vocab, caption_map):
    df = df.copy()
    df["query_text"] = ""
    df["item_caption"] = df["item_id"].apply(
        lambda i: decode(caption_map.get(int(i), []), rev_vocab)
    )
    df["event"] = "rec"
    return df[["user_id", "timestamp", "query_text", "item_caption", "event"]]


def main():
    rev_vocab, caption_map = load_decoders()

    diff = pd.read_csv(RANK_DIFF_INPUT)
    # 只用搜索域(src)的 UNISAR vs TrustSAR 差异来判定 UP/DOWN
    diff = diff[diff["domain"] == "src"]
    idx_min = diff.groupby("user")["rank_diff"].idxmin()
    idx_max = diff.groupby("user")["rank_diff"].idxmax()
    min_rows = diff.loc[idx_min]
    max_rows = diff.loc[idx_max]

    # “extreme” 阈值：|diff| >= 1
    up_users = set(min_rows[min_rows["rank_diff"] <= -1]["user"])
    down_users = set(max_rows[max_rows["rank_diff"] >= 1]["user"])
    target_users = up_users | down_users
    if not target_users:
        print("No users meeting thresholds.")
        return

    # user info dict
    user_info = {}
    for _, r in min_rows.iterrows():
        user_info.setdefault(int(r.user), {})["min"] = {
            "diff": float(r.rank_diff),
            "unisar": int(r.rank_unisar),
            "trust": int(r.rank_trust),
        }
    for _, r in max_rows.iterrows():
        user_info.setdefault(int(r.user), {})["max"] = {
            "diff": float(r.rank_diff),
            "unisar": int(r.rank_unisar),
            "trust": int(r.rank_trust),
        }

    # 事件流：搜索 + 推荐全部保留
    src_events = pd.concat(
        [
            build_search_events(load_split("src_train.pkl"), rev_vocab, caption_map),
            build_search_events(load_split("src_val.pkl"), rev_vocab, caption_map),
            build_search_events(load_split("src_test.pkl"), rev_vocab, caption_map),
        ],
        ignore_index=True,
    )
    rec_events = pd.concat(
        [
            build_rec_events(load_split("rec_train.pkl"), rev_vocab, caption_map),
            build_rec_events(load_split("rec_val.pkl"), rev_vocab, caption_map),
            build_rec_events(load_split("rec_test.pkl"), rev_vocab, caption_map),
        ],
        ignore_index=True,
    )
    events = pd.concat([src_events, rec_events], ignore_index=True)
    events = events[events["user_id"].isin(target_users)]

    # Keep each user's timeline only up to (and including) their last search event,
    # so the final entry is always a search rather than a post-search recommendation.
    last_search_ts = (
        events[events["event"] == "search"]
        .groupby("user_id")["timestamp"]
        .max()
        .rename("last_search_ts")
    )
    events = events.merge(last_search_ts, on="user_id", how="left")
    events = events[events["timestamp"] <= events["last_search_ts"]]

    # Drop users that never had a search event (last_search_ts is NaN)
    events = events.dropna(subset=["last_search_ts"])

    # Ensure search appears after rec when timestamps tie, so the last line is search
    events["_order"] = events["event"].map({"rec": 0, "search": 1}).fillna(0)
    events = events.sort_values(["user_id", "timestamp", "_order"], ascending=[True, True, False])

    lines = []
    for uid, group in events.groupby("user_id"):
        info = user_info.get(int(uid), {})
        min_part = info.get("min", {})
        max_part = info.get("max", {})
        tags = []
        if int(uid) in up_users:
            tags.append("UP")
        if int(uid) in down_users:
            tags.append("DOWN")
        lines.append(
            f"=== 用户 {uid} [{'/'.join(tags)}] "
            f"(min_diff {min_part.get('diff')}, UNISAR {min_part.get('unisar')}, TrustSAR {min_part.get('trust')}; "
            f"max_diff {max_part.get('diff')}, UNISAR {max_part.get('unisar')}, TrustSAR {max_part.get('trust')}) ==="
        )
        for _, row in group.iterrows():
            tstr = ts_to_str(float(row["timestamp"]))
            if row["event"] == "search":
                lines.append(
                    f"[{tstr}] (search) {row['query_text']} | item: {row['item_caption']}"
                )
            else:
                lines.append(f"[{tstr}] (rec) {row['item_caption']}")
        lines.append("")

    out_path = TIMELINE_OUTPUT
    out_path.write_text("\n".join(lines))
    print(
        f"Wrote {events['user_id'].nunique()} users "
        f"({len(up_users)} UP, {len(down_users)} DOWN, overlap {len(up_users & down_users)}) "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
