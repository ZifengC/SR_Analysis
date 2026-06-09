# Embedding Behavior Signatures Python Pipeline

Run from `Analysis/` with a Python 3.11 environment that has the analysis dependencies installed.

Recommended command sequence:

```bash
cd "/Users/Brodie/Documents/Code-RelSAR/Results&Analysis/Analysis"
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/1-part1-exploration.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-transition-level.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-transition-exploration-curves.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-consolidation-quadrant.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-consolidation-transition-colors.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/3-part3-consolidation-runs.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/3-part3-anchor-exploration-adoption.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/4-part4-anchor-transition-shift.py
```

Notes:

- `1-part1-exploration.py` rebuilds `intermediate/1-events.pkl`, `intermediate/1-event-embeddings.npy`, and `intermediate/1-exploration-scores.pkl` if they are missing.
- The later scripts reuse those intermediates automatically.
- If you prefer, you can replace `/Users/Brodie/miniconda3/bin/python3` with the `python3` from your active environment.
