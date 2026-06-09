"""
Analyze how min_diff is distributed across similarity buckets.

For each user in 4-extreme_timelines_global.txt:
1) Take the last search query in time as the anchor query.
2) Find the latest previous event whose text is different from the anchor query.
   - search event text: query text
   - rec event text: item caption
3) Compute TF-IDF cosine similarity between the two texts.
4) Bucket similarity by thresholds: 0.1, 0.5, 1.0.
5) Summarize min_diff distribution and test significance of UNISAR vs TrustSAR rank
   differences (paired tests) within each bucket.

Run from Results-Analysis/:
    python 1-Compare-baseline/Code/4-similarity_bucket_analysis.py

Inputs:
    1-Compare-baseline/New_results/4-extreme_timelines_global.txt

Outputs:
    1-Compare-baseline/New_results/5-sim_bucket_user_level_char.csv
    1-Compare-baseline/New_results/5-sim_bucket_stats_char.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer


BASELINE_ROOT = Path(__file__).resolve().parents[1]
NEW_RESULTS = BASELINE_ROOT / "New_results"
DEFAULT_TIMELINE = NEW_RESULTS / "4-extreme_timelines_global.txt"
USER_LEVEL_OUTPUT = "5-sim_bucket_user_level_char.csv"
STATS_OUTPUT = "5-sim_bucket_stats_char.csv"

HEADER_RE = re.compile(
    r"^=== 用户 (\d+) .*?\(min_diff ([^,]+), UNISAR (\d+), TrustSAR (\d+);"
)
EVENT_RE = re.compile(r"^\[(.*?)\] \((search|rec)\) (.*)$")


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if s.lower() == "nan":
        return ""
    return s


def parse_timeline(path: Path):
    users = []
    current = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue

            hm = HEADER_RE.match(line)
            if hm:
                if current is not None:
                    users.append(current)
                current = {
                    "user_id": int(hm.group(1)),
                    "min_diff": float(hm.group(2)),
                    "rank_unisar": int(hm.group(3)),
                    "rank_trust": int(hm.group(4)),
                    "events": [],
                }
                continue

            em = EVENT_RE.match(line)
            if em and current is not None:
                event_type = em.group(2)
                payload = em.group(3)
                if event_type == "search":
                    query_text = payload.split(" | item:", 1)[0].strip()
                    text_for_diff = query_text
                else:
                    text_for_diff = payload.strip()
                current["events"].append(
                    {
                        "event": event_type,
                        "text_for_diff": normalize_text(text_for_diff),
                    }
                )

    if current is not None:
        users.append(current)
    return users


def cosine_tfidf_char(a: str, b: str, cache: dict):
    if not a or not b:
        return np.nan
    key = (a, b) if a <= b else (b, a)
    if key in cache:
        return cache[key]

    vect = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    mat = vect.fit_transform([a, b])
    sim = float((mat[0] @ mat[1].T).toarray()[0, 0])
    cache[key] = sim
    return sim


def bucketize(sim: float) -> str:
    if pd.isna(sim):
        return "no_prev_diff"
    sim = max(0.0, min(1.0, float(sim)))
    if sim < 0.1:
        return "[0.0,0.1)"
    if sim < 0.5:
        return "[0.1,0.5)"
    if sim < 1.0:
        return "[0.5,1.0)"
    return "[1.0]"


def build_user_level(users):
    rows = []
    cache = {}

    for u in users:
        events = u["events"]
        last_query = ""
        for ev in reversed(events):
            if ev["event"] == "search" and ev["text_for_diff"]:
                last_query = ev["text_for_diff"]
                break

        prev_diff_text = ""
        if last_query:
            for ev in reversed(events):
                candidate = ev["text_for_diff"]
                if candidate and candidate != last_query:
                    prev_diff_text = candidate
                    break

        sim = cosine_tfidf_char(last_query, prev_diff_text, cache)
        row = {
            "user_id": u["user_id"],
            "min_diff": u["min_diff"],
            "rank_unisar": u["rank_unisar"],
            "rank_trust": u["rank_trust"],
            "rank_gap_trust_minus_unisar": u["rank_trust"] - u["rank_unisar"],
            "last_query": last_query,
            "latest_different_text": prev_diff_text,
            "similarity": sim,
            "sim_bucket": bucketize(sim),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_bucket_stats(df: pd.DataFrame):
    out = []
    grouped = {k: v for k, v in df.groupby("sim_bucket", dropna=False)}
    order = ["[0.0,0.1)", "[0.1,0.5)", "[0.5,1.0)", "[1.0]", "no_prev_diff"]
    for bucket in order:
        g = grouped.get(bucket)
        if g is None or g.empty:
            out.append(
                {
                    "sim_bucket": bucket,
                    "users": 0,
                    "min_diff_mean": np.nan,
                    "min_diff_median": np.nan,
                    "rank_gap_mean": np.nan,
                    "rank_gap_median": np.nan,
                    "ttest_stat": np.nan,
                    "ttest_p_two_sided": np.nan,
                    "wilcoxon_stat": np.nan,
                    "wilcoxon_p_two_sided": np.nan,
                }
            )
            continue

        x = g["rank_unisar"].to_numpy()
        y = g["rank_trust"].to_numpy()
        d = y - x
        rg = g["rank_gap_trust_minus_unisar"]
        row = {
            "sim_bucket": bucket,
            "users": int(len(g)),
            "min_diff_mean": float(g["min_diff"].mean()),
            "min_diff_median": float(g["min_diff"].median()),
            "rank_gap_mean": float(rg.mean()),
            "rank_gap_median": float(rg.median()),
        }

        # Paired t-test
        if len(g) >= 2 and np.std(d) > 0:
            t_res = ttest_rel(y, x)
            row["ttest_stat"] = float(t_res.statistic)
            row["ttest_p_two_sided"] = float(t_res.pvalue)
        else:
            row["ttest_stat"] = np.nan
            row["ttest_p_two_sided"] = np.nan

        # Wilcoxon signed-rank test
        if len(g) >= 2 and np.any(d != 0):
            try:
                w_res = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
                row["wilcoxon_stat"] = float(w_res.statistic)
                row["wilcoxon_p_two_sided"] = float(w_res.pvalue)
            except Exception:
                row["wilcoxon_stat"] = np.nan
                row["wilcoxon_p_two_sided"] = np.nan
        else:
            row["wilcoxon_stat"] = np.nan
            row["wilcoxon_p_two_sided"] = np.nan

        out.append(row)

    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--out_dir", type=Path, default=NEW_RESULTS)
    args = parser.parse_args()

    users = parse_timeline(args.timeline)
    df = build_user_level(users)
    stats_df = build_bucket_stats(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / USER_LEVEL_OUTPUT, index=False)
    stats_df.to_csv(args.out_dir / STATS_OUTPUT, index=False)

    print(f"users parsed: {len(df)}")
    for bucket in ["[0.0,0.1)", "[0.1,0.5)", "[0.5,1.0)", "[1.0]"]:
        row = stats_df[stats_df["sim_bucket"] == bucket]
        if row.empty:
            print(f"{bucket}: 0 users")
            continue
        r = row.iloc[0]
        print(
            f"{bucket}: {int(r['users'])} users, "
            f"min_diff_mean={r['min_diff_mean']:.2f}, "
            f"t-test p={r['ttest_p_two_sided']:.2e}, "
            f"Wilcoxon p={r['wilcoxon_p_two_sided']:.2e}"
        )


if __name__ == "__main__":
    main()
