# Rolling Player Features — Implementation Plan

> ⚠️ DEPRECATED — superseded by docs/team_features_migration_plan.md
> Player-level rolling features were removed due to 55–72% structural missingness.

How to get lagged weekly player stats (QB EPA, RB yards per carry, WR target share, etc.)
onto individual PBP rows for use as model features.

---

## Why This Is Non-Trivial

`pbp_clean` and `players_clean` are at different granularities and currently share no join key:

| Dataset | Granularity | Player ID? |
|---|---|---|
| `pbp_clean` | One row per **play** | ❌ Not currently |
| `players_clean` | One row per **player × week** | ✅ `player_id` |

Three separate player roles exist on every play (passer, rusher, receiver), each requiring
its own join. Rolling windows must be **lagged by 1 week** so that week N features only
use data from weeks 1 through N-1 — no leakage.

---

## Step 1 — Add Player IDs to `pbp_clean`

**File:** `src/preprocessing/clean_pbp.py`

Add three columns to the `cols` list inside `select_columns()`:

```python
"passer_player_id",    # non-null on pass plays
"rusher_player_id",    # non-null on run plays
"receiver_player_id",  # non-null on pass plays (targeted receiver)
```

Then re-run:
```bash
python3 -m src.preprocessing.clean_pbp
```

These IDs are `00-XXXXXXX` format strings that match `player_id` in `players_clean`.
They are `NaN` when that role was not involved in the play (e.g. `rusher_player_id` is
null on pass plays).

---

## Step 2 — Build Rolling Player Stats

**File:** `src/features/player_features.py`

Takes `players_clean.parquet` as input. For each player, sorts by `(season, week)` and
computes rolling 3-week averages using a **shift-then-roll** pattern to prevent leakage:

```
week N feature = mean of weeks [N-3, N-2, N-1]   ← never includes week N itself
```

### Rolling stats to compute

**QB**
| Output column | Source | Window |
|---|---|---|
| `qb_pass_yds_L3` | `passing_yards` | 3-week rolling mean |
| `qb_pass_epa_L3` | `passing_epa` | 3-week rolling mean |
| `qb_air_yards_L3` | `passing_air_yards` | 3-week rolling mean |
| `qb_int_rate_L3` | `interceptions / attempts` | 3-week rolling mean |
| `qb_season_epa` | `passing_epa` | Season-to-date expanding mean |

**RB**
| Output column | Source | Window |
|---|---|---|
| `rb_rush_yds_L3` | `rushing_yards` | 3-week rolling mean |
| `rb_rush_epa_L3` | `rushing_epa` | 3-week rolling mean |
| `rb_ypc_L3` | `rushing_yards / carries` | 3-week rolling mean |

**WR / TE**
| Output column | Source | Window |
|---|---|---|
| `rec_yards_L3` | `receiving_yards` | 3-week rolling mean |
| `rec_epa_L3` | `receiving_epa` | 3-week rolling mean |
| `target_share_L3` | `target_share` | 3-week rolling mean |
| `air_yards_share_L3` | `air_yards_share` | 3-week rolling mean |

### Output

Three separate DataFrames keyed by `(player_id, season, week)`:

```
qb_rolling    → (player_id, season, week) + qb_* columns
rb_rolling    → (player_id, season, week) + rb_* columns
rec_rolling   → (player_id, season, week) + rec_* columns
```

Saved to `data/interim/`:
```
players_rolling_qb.parquet
players_rolling_rb.parquet
players_rolling_rec.parquet
```

---

## Step 3 — Join onto PBP Plays

**File:** `src/features/feature_pipeline.py`

Load `pbp_clean` (now with player IDs) and the three rolling parquets.
Perform three left joins, each on `(player_id, season, week)`:

```
pbp["passer_player_id"]   → join qb_rolling    → adds qb_* columns
pbp["rusher_player_id"]   → join rb_rolling    → adds rb_* columns
pbp["receiver_player_id"] → join rec_rolling   → adds rec_* columns
```

Rows where the player ID is null (e.g. run plays have no `passer_player_id`) will
naturally get `NaN` for the QB columns — handle at training time with imputation or
a model that supports missing values natively (XGBoost, LightGBM).

### Output

`data/processed/model_dataset.parquet` — the fully-featured PBP dataset ready for model training.

---

## Leakage Checklist

Before training, verify:

- [ ] All rolling features use `shift(1)` before `.rolling(3).mean()` — week N never sees week N data
- [ ] Week 1 of each season has no prior-season carryover (rolling window resets at season boundary)
- [ ] `passer_player_id`, `rusher_player_id`, `receiver_player_id` are **dropped** from the final feature matrix — they are join keys, not model inputs
- [ ] `yards_gained` is never used in any rolling calculation

---

## File Summary

| Step | File | Status |
|---|---|---|
| 1 | `src/preprocessing/clean_pbp.py` | 🔧 Add 3 player ID columns to `select_columns` |
| 2 | `src/features/player_features.py` | 🔧 Build rolling stats from `players_clean` |
| 3 | `src/features/feature_pipeline.py` | 🔧 Join rolling stats onto PBP by player ID + week |
