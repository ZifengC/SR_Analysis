# Embedding Behavior Signatures

This analysis studies search-recommendation user behavior with text embeddings. It does not require explicit exploration labels. The core idea is simple: convert every search query or clicked recommendation caption into an embedding, then measure whether a behavior is close to the user's past, close to the user's future, or far from both.

## Current Paths

The scripts now expect the processed data in the current repository layout:

- `Data/Step1/`
- `Data/Step4/`

The main reusable intermediate files live under:

- `2-Problem-identify/intermediate/`
- `2-Problem-identify/cache/`

If those intermediates are missing, the scripts rebuild them automatically from `Data/Step1/` and `Data/Step4/`.

## Data Unit

Each user behavior event is one of two channels:

- `S`: search/source event. The text is the decoded search query.
- `R`: recommendation event. The text is the decoded clicked item caption.

Each event text is encoded as a normalized embedding:

```text
e_t = normalize(Encoder(text_t))
```

The current encoder is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

All similarity scores are cosine similarities. Since embeddings are normalized, cosine similarity is just a dot product:

```text
cos(e_i, e_j) = e_i dot e_j
```

Events are sorted within each user by timestamp. If search and recommendation events share the same timestamp, recommendation is ordered first.

## Part 1.1: Exploration Score and Future Consistency

### What It Measures

Part 1 asks two basic questions:

1. Is the current behavior new relative to the user's recent history?
2. Does the current behavior predict the user's near future?

For each user, we first build a recent preference center from the previous `N_HISTORY` events:

```text
recent_preference_center = average embedding of previous N events
```

Then we measure how far the current event is from that center:

```text
exploration_score = 1 - similarity(current_event, recent_preference_center)
```

Plain meaning:

- High `exploration_score`: the current behavior is unlike recent history, so it is more exploratory.
- Low `exploration_score`: the current behavior is close to recent history, so it is more exploitative.

Then we measure whether the current behavior matches the next `N_FUTURE` events:

```text
future_consistency = similarity(current_event, average embedding of next N events)
```

Plain meaning:

- High `future_consistency`: the current behavior is aligned with what the user does next.
- Low `future_consistency`: the current behavior does not continue into the near future.

Exploration and exploitation are assigned within each user:

```text
exploration  = top 20% exploration_score within the same user
exploitation = bottom 20% exploration_score within the same user
```

### Current Conclusion

Part 1 shows that the farther an event is from the recent history center, the lower its future consistency. This gives a continuous `explore`-to-`exploit` axis.

### Event Definition

One `event` is one ordered behavioral unit on a user's timeline.

- `S` event: one search session, collapsed from all rows that share the same `user_id` and `search_session_id`
- `R` event: one recommendation click row from `rec_all`

For each event, the script keeps:

- `user_id`
- `timestamp`
- `domain` (`S` or `R`)
- `text`
- `event_id`
- `search_session_id` for `S` events, `NaN` for `R` events
- `item_ids`
- `num_clicked_items`

The event builder reads:

- `Data/Step1/vocab_dict.pkl`
- `Data/Step1/note_feat.pkl`
- `Data/Step4/src_all.pkl`
- `Data/Step4/rec_all.pkl`

Events are then sorted within each user by:

- `timestamp`
- `domain` order, with `R` before `S` when timestamps tie
- `event_id` as a final tie-breaker

That sorted event list is the input to Part 1.

### Figure Structure

- `Figure 1` shows the main relationship between exploration score percentile and future consistency.
- The sensitivity version uses two panels: left varies `N_HISTORY` with fixed `N_FUTURE`, right varies `N_FUTURE` with fixed `N_HISTORY`.

## Part 3: Run-Length Consolidation and Radius

### What It Measures

Part 3 asks whether users become more stable after staying in the same semantic region for multiple consecutive events.

First, we compare each pair of adjacent events:

```text
adjacent_similarity = similarity(current_event, next_event)
```

For each user, we define a high-similarity threshold using the user's own adjacent similarities:

```text
semantic_neighbor = adjacent_similarity >= user-level 75th percentile
```

A run is a consecutive sequence connected by these high-similarity adjacent edges:

```text
run = a sequence of events that keep staying in a similar semantic region
```

For each run, we compare the run's average embedding with the user's next `K_FUTURE` events:

```text
future_stability = similarity(average embedding of run, average embedding of next K events)
```

Plain meaning:

- Longer run: the user stayed in the same semantic area for more steps.
- Higher `future_stability`: that semantic area continues into the near future.

### Current Conclusion

Part 3 shows that longer semantic-neighbor runs lead to higher future stability, while the semantic radius shrinks as run length grows. This means repeated nearby behavior consolidates preference and narrows the active semantic range.

### Optional Adjustment

If the long-run sample size is too small, compare different semantic-neighbor thresholds:

```text
RUN_NEIGHBOR_Q = 0.30
RUN_NEIGHBOR_Q = 0.50
RUN_NEIGHBOR_Q = 0.70
```

Lowering this threshold relaxes the semantic-neighbor definition and usually increases coverage, but it also weakens semantic purity.

### Figure Structure

- `Figure 3` uses two side-by-side panels.
- The left panel shows future stability by semantic-neighbor run length.
- The right panel shows semantic radius by semantic-neighbor run length.
- Each panel overlays multiple `RUN_NEIGHBOR_Q` values to compare how strictness changes coverage and consolidation strength.

## Part 2.2: Anchor-Conditioned Transition Adoption

### What It Measures

Part 2.2 asks how strongly an event's nearest historical anchor is absorbed into the future, and whether that absorption differs by transition type.

For each event with a valid nearest anchor, we measure:

```text
anchor_exploration_adoption = future_consistency - nearest_anchor_similarity
```

Then we stratify the events by `anchor_transition` and by anchor strength.

### Figure Structure

- `Figure 2.2` uses two side-by-side panels.
- Each panel overlays four transition-specific trend lines.
- The left panel shows mean anchor adoption score as a function of nearest anchor similarity.
- The right panel shows successful anchor exploration rate as a function of nearest anchor similarity.
- A shaded band shows the 95% confidence interval around the mean in each similarity bin.
- The four transition types are ordered as `R->R`, `R->S`, `S->R`, `S->S`.
- Only anchors with `nearest_anchor_similarity >= 0.50` are included.
- This figure is fixed at `N_HISTORY=10` and `K_FUTURE=5`.

## Part 2.2.2: Transition-Stratified Exploration Curves

### What It Measures

Part 2.2.2 uses the exploration score from Part 1 as the x-axis, then asks whether more exploratory behavior is followed by stronger future coherence or broader future dispersion. Each event is grouped by `transition_into`.

For each transition type, it measures how exploration score relates to:

- `future_consistency`
- `future_dispersion`

Here, `future_dispersion` is the mean semantic distance of the future window from its own center:

```text
future_dispersion = mean(1 - cos(future_event, future_center))
```

### Figure Structure

- `Figure 2.2.2` uses two side-by-side panels.
- The left panel shows future consistency as a function of exploration score.
- The right panel shows future semantic dispersion as a function of exploration score.
- Each panel overlays the four transition types with fixed colors.
- Exploration score is binned into quantiles for stable curves.
- This figure is fixed at `N_HISTORY=10` and `K_FUTURE=5`.

## Part 2.3: Consolidation Quadrants

### What It Measures

Part 2.3 combines three run-level quantities:

```text
current_radius = mean(1 - cos(event, current run center))
future_stability = similarity(current run center, future behavior)
future_radius = mean(1 - cos(event, future center))
```

This lets us separate four cases:

- high `future_stability` + low `future_radius`: strongest consolidation
- high `future_stability` + high `future_radius`: future direction is retained but still diffuse
- low `future_stability` + low `future_radius`: future becomes more focused but does not follow the current run
- low `future_stability` + high `future_radius`: neither retained nor concentrated

### Figure Structure

- `Figure 2.3` uses one panel.
- The panel is a quadrant scatter of `future stability` vs `future radius`, colored by mean exploration score over the run.
- The four quadrant labels summarize the state space.
- This figure is fixed at `RUN_NEIGHBOR_Q=0.50` and `K_FUTURE=5`.

## Part 2.4: Transition-Colored Consolidation Quadrants

### What It Measures

Part 2.4 keeps the same quadrant geometry as Part 2.3, but colors each run by its dominant semantic-neighbor transition type:

```text
R->R, R->S, S->R, S->S
```

This makes it easier to compare whether different transition types occupy different consolidation states.

### Figure Structure

- `Figure 2.4` uses one panel.
- The panel is the same quadrant scatter of `future stability` vs `future radius`.
- Point color indicates the dominant run transition type.
- The four quadrant labels summarize the state space.
- This figure is fixed at `RUN_NEIGHBOR_Q=0.50` and `K_FUTURE=5`.

## How To Run Now

Run from `Analysis/` with the Python 3.11 environment you want to use for analysis.

Recommended order:

```bash
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/1-part1-exploration.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-transition-level.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-transition-exploration-curves.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-consolidation-quadrant.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/2-part2-consolidation-transition-colors.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/3-part3-consolidation-runs.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/3-part3-anchor-exploration-adoption.py
/Users/Brodie/miniconda3/bin/python3 2-Problem-identify/scripts/4-part4-anchor-transition-shift.py
```

The Part 1 cache files are rebuilt automatically if they are absent, so you do not need to manage cache state manually.

## Part 4: Anchor-Based Exploration Adoption

### What It Measures

Part 3 asks whether an exploratory behavior is adopted by the user's future behavior.

For each event, we find the most similar historical event from either channel:

```text
nearest_anchor = historical R or S event most similar to the current event
nearest_anchor_similarity = similarity(current_event, nearest_anchor)
```

Then we compare two similarities:

```text
future_consistency = similarity(current_event, future behavior)
nearest_anchor_similarity = similarity(current_event, nearest historical anchor)
```

The adoption score is:

```text
anchor_exploration_adoption = future_consistency - nearest_anchor_similarity
```

Plain meaning:

- Positive value: the current event is closer to the future than to its nearest historical anchor. This suggests the exploration was adopted or became a preference shift.
- Non-positive value: the current event is still closer to past behavior than future behavior. This suggests transient exploration.

Part 3 only evaluates retained anchors:

```text
nearest_anchor_similarity >= ANCHOR_SIM_THRESHOLD
```

Current default:

```text
ANCHOR_SIM_THRESHOLD = 0.30
```

### Current Conclusion

Among top exploration events, successful anchor exploration rates differ strongly by transition type:

```text
R->R: 0.120
R->S: 0.341
S->R: 0.168
S->S: 0.317
```

Interpretation:

Not all exploration becomes future preference. Search-related current behavior is much more likely to be adopted by the future, especially:

- `R->S`: the nearest historical anchor is recommendation, but the current behavior is search.
- `S->S`: both the anchor and current behavior are search.

This suggests search is a stronger active signal of preference shift. Recommendation-side exploration is more often transient, likely because recommendation includes passive exposure.

### Figure Structure

- `Figure 4` groups exploration-adoption rate by anchor transition type.
- It compares how often exploration is actually adopted when the nearest historical anchor comes from `R` or `S`.

## Part 4.2: Anchor-Based Cross-Channel Transition Shift

### What It Measures

Part 4 asks which anchor-transition types are more likely to be exploration episodes. It uses the same nearest historical anchor idea, but the main plotted y-axis is now `exploration_rate`.

The primary metric is:

```text
exploration_rate = P(episode_type == exploration | anchor_transition)
```

Plain meaning:

- High `exploration_rate`: this transition type often corresponds to behavior far from the recent preference center.
- Low `exploration_rate`: this transition type is usually more local or exploitative.

The CSV also keeps semantic movement size for reference:

```text
anchor_shift = 1 - nearest_anchor_similarity
```

The transition type is not the immediately previous event. It is:

```text
nearest historical anchor channel -> current channel
```

For example:

```text
R->S: the nearest historical anchor is recommendation, current event is search
S->R: the nearest historical anchor is search, current event is recommendation
```

### Current Conclusion

Current exploration rate by anchor transition:

```text
S->S: 0.119
R->R: 0.179
S->R: 0.189
R->S: 0.253
```

Interpretation:

`S->S` is the most local and stable transition. Search-to-search behavior tends to refine the same semantic region.

`S->R` has the largest semantic shift, suggesting that when recommendation follows a search anchor, it often expands the semantic region.

`R->S` has the highest exploration rate, suggesting that when users move from a recommendation-like historical anchor into search, the search action often expresses a new or shifted intent.

Overall, cross-channel transitions are asymmetric. Search and recommendation do not play the same behavioral role.

### Figure Structure

- `Figure 4` groups exploration rate by anchor transition type.
- It shows which cross-channel transitions are more local, more exploratory, or more likely to express a shifted intent.

## Overall Conclusion

The four parts support one coherent behavioral story:

1. Users have measurable exploration and exploitation patterns in embedding space.
2. Behaviors far from recent preference history are less future-consistent.
3. Consecutive semantically similar behavior leads to stronger future stability, showing preference consolidation.
4. Search and recommendation transitions are asymmetric: search is a stronger active preference signal, while recommendation more often creates semantic expansion or transient exploration.

In short:

```text
Exploration exists.
Exploitation is more future-consistent.
Repeated semantic staying consolidates preference.
Cross-channel transition direction changes behavioral meaning.
```

## Tunable Parameters

The main parameters are defined near the top of each script.

```text
MODEL_NAME
```

Embedding model. Changing it requires rerunning all parts.

```text
N_HISTORY
```

Number of previous events used to build the recent preference center in Part 1.

```text
K_FUTURE
```

Number of future events used for future consistency and future stability.

```text
EXPLORE_Q / EXPLOIT_Q
```

Within-user percentile thresholds for exploration and exploitation labels.

```text
RUN_NEIGHBOR_Q
```

User-level threshold for defining semantic-neighbor runs in Part 2. Lower values create more long runs but weaken the semantic-stay definition.

```text
ANCHOR_SIM_THRESHOLD
```

Minimum historical-anchor similarity. Part 4 applies its own threshold filters from Part 1 scores, so Part 4 threshold changes require rerunning Part 4.

## How to Run

Use the Python 3.11 environment, then run the analyses in order.

1. Part 1.1 builds event-level exploration scores and Figure 1.
2. Part 2.1 builds event-level semantic-neighbor runs and Figure 2.
3. Part 2.2 builds anchor-conditioned transition adoption and Figure 2.2.
4. Part 2.2.2 builds transition-stratified exploration curves and Figure 2.2.2.
5. Part 2.3 builds consolidation quadrants and Figure 2.3.
6. Part 2.4 builds transition-colored consolidation quadrants and Figure 2.4.
7. Part 3 builds run-length consolidation and radius curves and Figure 3.
8. Part 4 builds anchor-based adoption statistics and Figure 4.
9. Part 4.2 builds anchor-transition exploration rates and Figure 4.

Part 1.1 must run first because it creates the event timeline, embeddings, exploration scores, and anchors. Parts 2.1, 2.2, 3, and 4 reuse Part 1.1 state and can then be rerun independently for their own sensitivity settings.


## Environment Pinning

Use a Python 3.11 environment with the project dependencies installed.
