from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SCRIPT_DIR.parent
ANALYSIS_ROOT = PROBLEM_DIR.parent


def resolve_data_root() -> Path:
    candidates = [
        ANALYSIS_ROOT / "Data",
        ANALYSIS_ROOT / "Data" / "Qilin-data-session-split-creating",
    ]
    required_pairs = [
        ("Step1", "vocab_dict.pkl"),
        ("Step4", "src_all.pkl"),
    ]
    for root in candidates:
        if all((root / subdir / filename).exists() for subdir, filename in required_pairs):
            return root
    return candidates[0]


DATA_ROOT = resolve_data_root()
STEP1_DIR = DATA_ROOT / "Step1"
STEP4_DIR = DATA_ROOT / "Step4"
FIG_DIR = PROBLEM_DIR / "figures"
CACHE_DIR = PROBLEM_DIR / "cache"
INTERMEDIATE_DIR = PROBLEM_DIR / "intermediate"

for path in (FIG_DIR, CACHE_DIR, INTERMEDIATE_DIR):
    path.mkdir(parents=True, exist_ok=True)


def part1_artifacts_ready() -> bool:
    return all(
        (INTERMEDIATE_DIR / name).exists()
        for name in ("1-events.pkl", "1-event-embeddings.npy", "1-exploration-scores.pkl")
    )


def ensure_part1_artifacts() -> None:
    if part1_artifacts_ready():
        return
    script_path = SCRIPT_DIR / "1-part1-exploration.py"
    sys_path_was = list(sys.path)
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.path[:] = sys_path_was
