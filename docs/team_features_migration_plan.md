# Team Features Migration Plan

Migrate from player-level rolling stats (which suffer from 55–72% structural missingness)
to team-level rolling stats computed directly from `pbp_clean`. Team stats are non-missing
on every play row and better capture scheme tendencies that drive play-call decisions.

> **Tracking:** Mark each step header with `~~` strikethrough and append `✅ DONE` when complete.
> Example: `## ~~Step N — ...~~ ✅ DONE`

---

## Background

Player rolling features (`qb_*`, `rb_*`, `rec_*`) were built on a join keyed to
`passer_player_id`, `rusher_player_id`, `receiver_player_id` — columns that are
`null` on the wrong play type by construction:

| Column | Null on |
|---|---|
| `passer_player_id` | all run plays (~50%) |
| `rusher_player_id` | all pass plays (~50%) |
| `receiver_player_id` | run plays + untargeted pass plays (~65%) |

Team stats (`posteam`, `defteam`) are populated on **every** offensive play row.
Rolling team stats require no cross-table join and no position-roster lookup.

---

## ~~Step 1 — Deprecate `clean_pbp.py` Player ID Columns~~ ✅ DONE

**File:** `src/preprocessing/clean_pbp.py`

Remove the three player ID columns from the `cols` list inside `select_columns()`:

```python
# Remove these:
"passer_player_id",
"rusher_player_id",
"receiver_player_id",
```

Then re-run `clean_pbp.py` to regenerate `pbp_clean.parquet` without them:

```bash
python3 -m src.preprocessing.clean_pbp
```

Final shape will drop from 35 → 32 columns.

---

## ~~Step 2 — Delete Player Rolling Artifacts~~ ✅ DONE

**Files to delete:**

```
src/features/player_features.py
data/interim/players_rolling_qb.parquet
data/interim/players_rolling_rb.parquet
data/interim/players_rolling_rec.parquet
```

**Docs to mark deprecated:**

Add a deprecation notice at the top of `docs/rolling_player_features_plan.md`:

```
> ⚠️ DEPRECATED — superseded by docs/team_features_migration_plan.md
> Player-level rolling features were removed due to 55–72% structural missingness.
```

---

## ~~Step 3 — Implement `src/features/team_features.py`~~ ✅ DONE

**Key design:** compute weekly team aggregates from `pbp_clean` itself, then apply
shift-then-roll within `(team, season)` groups, then join back onto play rows.
No intermediate parquet files needed — everything happens in memory.

### Offense rolling features

Aggregate by `(posteam, season, week)`:

| Feature | Source | Logic |
|---|---|---|
| `off_pass_rate_L3` | `play_type` | `mean(play_type == "pass")` → rolling 3-week mean |
| `off_rush_yds_L3` | `yards_gained` on run plays | mean yards per run play → rolling 3-week mean |
| `off_pass_yds_L3` | `yards_gained` on pass plays | mean yards per pass play → rolling 3-week mean |

### Defense rolling features

Aggregate by `(defteam, season, week)`:

| Feature | Source | Logic |
|---|---|---|
| `def_rush_yds_allowed_L3` | `yards_gained` on run plays | mean yards allowed per run → rolling 3-week mean |
| `def_pass_yds_allowed_L3` | `yards_gained` on pass plays | mean yards allowed per pass → rolling 3-week mean |

### Rolling pattern (same as player_features.py)

```
weekly_stats
  .sort_values(["team", "season", "week"])
  .groupby(["team", "season"])["col"]
  .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
```

`shift(1)` ensures week N only sees data from weeks 1…N-1. Rolling resets at
every season boundary because the group key includes `season`.

### Function signatures

```python
def build_offense_rolling(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (posteam, season, week) + off_pass_rate_L3, off_rush_yds_L3, off_pass_yds_L3
    """

def build_defense_rolling(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (defteam, season, week) + def_rush_yds_allowed_L3, def_pass_yds_allowed_L3
    """

def build_team_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Master function. Calls both builders and left-joins results onto pbp rows.
    Join keys: offense on (posteam, season, week), defense on (defteam, season, week).
    Returns pbp with 5 new columns, zero additional missingness.
    """
```

---

## ~~Step 4 — Rewrite `src/features/feature_pipeline.py`~~ ✅ DONE

Replace the three player-ID join functions with a single team features call.

**Before (current):**
```python
# Inputs: pbp_clean + 3 rolling player parquets
pbp = _join_player_rolling(pbp, qb_rolling, "passer_player_id")
pbp = _join_player_rolling(pbp, rb_rolling, "rusher_player_id")
pbp = _join_player_rolling(pbp, rec_rolling, "receiver_player_id")
```

**After:**
```python
# Input: pbp_clean only — team features computed inline
from src.features.team_features import build_team_features
pbp = build_team_features(pbp)
```

Update the module docstring prerequisite block:

```
Prerequisite:
    python3 -m src.preprocessing.clean_pbp    # produces pbp_clean.parquet
    (no other prerequisites)
```

---

## ~~Step 5 — Re-run the Pipeline~~ ✅ DONE

```bash
python3 -m src.preprocessing.clean_pbp      # regenerate without player IDs
python3 -m src.features.feature_pipeline    # rebuild model_dataset.parquet
```

Expected output shape: **279,917 rows × ~37 columns** (32 from clean_pbp + 5 team features).
All 5 new feature columns should have 0% missingness — verify this after the run.

---

## ~~Step 6 — EDA: Team Rolling Features~~ ✅ DONE

**File:** `notebooks/03_eda_team_features.ipynb`

A focused notebook validating the new features before they enter the model.

### Sections

**6.1 Missingness audit**
- Confirm all 5 new columns are 0% null
- Confirm week 1 of each season is NaN (expected — no prior data to roll)

**6.2 Distributions**
- Histograms of `off_pass_rate_L3`, `off_rush_yds_L3`, `off_pass_yds_L3`
- Histograms of `def_rush_yds_allowed_L3`, `def_pass_yds_allowed_L3`
- Check for outliers or unexpected ranges

**6.3 Correlation with target**
- Box plots: `yards_gained` by quintile of each team feature
- Correlation matrix: team features vs. `yards_gained` vs. `play_type`
- Split by play type (run vs. pass) — offense rush yards should correlate more on runs, etc.

**6.4 Leakage sanity check**
- For a sample team-season, show the raw weekly stat alongside the lagged rolling value
- Confirm week N's feature only reflects weeks 1…N-1

**6.5 Team trends**
- Line chart: rolling pass rate over a season for 2–3 example teams
- Validates the rolling window is behaving correctly across weeks

---

## File Summary

| Step | File | Action |
|---|---|---|
| ✅ 1 | `src/preprocessing/clean_pbp.py` | Remove 3 player ID columns; re-run |
| ✅ 2 | `src/features/player_features.py` | Delete |
| ✅ 2 | `data/interim/players_rolling_*.parquet` | Delete (3 files) |
| ✅ 2 | `docs/rolling_player_features_plan.md` | Add deprecation notice |
| ✅ 3 | `src/features/team_features.py` | Create |
| ✅ 4 | `src/features/feature_pipeline.py` | Replace player joins with team feature call |
| ✅ 5 | _(run commands)_ | Regenerate `pbp_clean` and `pbp_features` |
| ✅ 6 | `notebooks/03_eda_team_features.ipynb` | Create |
