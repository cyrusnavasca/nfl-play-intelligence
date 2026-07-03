# Core Features Integration Plan

Wire a curated, high-signal subset of engineered features into `feature_pipeline.py`
so they land in `model_dataset.parquet`. Trim the feature builder modules down to
only the features we actually want, deprecating everything unused, and fix one
piece of stale intent left over from the team-features migration.

---

## Background

Three feature builder modules were authored during the feature-engineering phase but
never wired into the pipeline (see `docs/feature_engineering.md`):

| Module | Currently outputs | Wired into pipeline? |
|---|---|---|
| `src/features/formation_features.py` | 6 features | ❌ no |
| `src/features/situational_features.py` | 22 features | ❌ no |
| `src/features/encoding.py` | 4 features | ❌ no |

The pipeline was rebuilt around the team-features migration
(`docs/team_features_migration_plan.md`, Step 4) and only ever calls
`build_team_features()`. This plan integrates a **curated subset** of the above,
deprecating the rest.

### Target feature set (12 features to add)

| Group | Feature | Source module |
|---|---|---|
| Formation | `is_heavy_formation` | `formation_features.py` |
| Formation | `is_spread_formation` | `formation_features.py` |
| Formation | `box_advantage` | `formation_features.py` |
| Situational | `score_differential` | `situational_features.py` |
| Situational | `time_adjusted_score_diff` | `situational_features.py` |
| Situational | `red_zone` | `situational_features.py` |
| Situational | `backed_up` | `situational_features.py` |
| Situational | `two_minute_drill` | `situational_features.py` |
| Encoding | `is_turf` | `encoding.py` |
| Encoding | `is_indoor` | `encoding.py` |
| Encoding | `is_playoffs` | `encoding.py` |
| Encoding | `is_home` | `encoding.py` |

Everything else produced by these modules is **deprecated** (see Step 2).

### Dependency safety

A grep confirms **no code or notebook currently imports these modules or references
their output columns** — the `build_*()` functions are defined but never called. That
means:

- Wiring the modules into the pipeline breaks nothing.
- Deprecating the unmentioned features breaks nothing.
- The only live consumer of the pipeline output is `model_dataset.parquet`, which is
  regenerated in Step 5.

Do **not** remove raw columns that surviving features derive from (`offense_personnel`,
`defenders_in_box`, `posteam_score`, `defteam_score`, `game_seconds_remaining`,
`half_seconds_remaining`, `yardline_100`, `surface`, `roof`, `season_type`,
`posteam_type`). They must remain in `pbp_clean` for the builders to run.

---

## Step 1 — Trim `formation_features.py`

**File:** `src/features/formation_features.py`

**Keep:** `is_heavy_formation`, `is_spread_formation`, `box_advantage`.

**Deprecate as output columns:** `off_rb_count`, `off_te_count`, `off_wr_count`.

These three counts are still needed **internally** — `is_heavy_formation`,
`is_spread_formation`, and `box_advantage` all derive from them. Keep the parsing
logic (`parse_offense_personnel`, `_extract_position_count`) as internal helpers that
produce temporary columns, then **drop the count columns before returning** so they do
not appear in the final dataset.

Rewrite `build_formation_features()` so it:

1. Parses `offense_personnel` into temporary `off_rb_count`, `off_te_count`,
   `off_wr_count`.
2. Computes `is_heavy_formation = (off_rb_count + off_te_count) >= 3`.
3. Computes `is_spread_formation = off_wr_count >= 3`.
4. Computes `box_advantage = off_rb_count + off_te_count - defenders_in_box`.
5. Drops the three temporary count columns.
6. Returns the frame with only the 3 new feature columns added.

> Note: `defenders_in_box` now has zero nulls (null rows are dropped in `clean_pbp.py`),
> so `box_advantage` will be fully populated.

---

## Step 2 — Trim `situational_features.py`

**File:** `src/features/situational_features.py`

**Keep exactly these 5 features:**
`score_differential`, `time_adjusted_score_diff`, `red_zone`, `backed_up`,
`two_minute_drill`.

**Deprecate (remove) all of these:**
`total_score`, `score_diff_abs`, `is_blowout`, `is_close`, `game_script`,
`game_progress`, `half_progress`, `is_overtime`, `two_minute_game`,
`two_minute_half`, `distance_bucket`, `ydstogo_normalized`, `yards_available`,
`ydstogo_ratio`, `is_short_yardage`, `is_third_and_long`, `is_fourth_down`.

### ⚠️ `two_minute_drill` correctness — critical

The current implementation depends on two columns that are being deprecated:

```python
# CURRENT — depends on two_minute_game / two_minute_half columns
df["two_minute_game"] = (df["game_seconds_remaining"] <= 120).astype("Int8")
df["two_minute_half"] = (df["half_seconds_remaining"] <= 120).astype("Int8")
df["two_minute_drill"] = (
    (df["two_minute_game"] == 1) | (df["two_minute_half"] == 1)
).astype("Int8")
```

Since `two_minute_game` and `two_minute_half` are being removed, `two_minute_drill`
**must be recomputed inline** directly from the raw time columns, with no dependency on
the deprecated intermediates:

```python
# NEW — self-contained, no deprecated dependencies
df["two_minute_drill"] = (
    (df["game_seconds_remaining"] <= 120) | (df["half_seconds_remaining"] <= 120)
).astype("Int8")
```

### `time_adjusted_score_diff` — verify self-contained

`time_adjusted_score_diff` depends on `score_differential` (kept) and computes game
progress **inline** (`1 - game_seconds_remaining / 3600`), not from the deprecated
`game_progress` column. Confirm it stays inline so deprecating `game_progress` does not
break it.

### Rewrite `build_situational_features()`

Reduce it to only the calls that produce the 5 surviving features:

```python
def build_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_score_features(df)             # score_differential (drop total_score)
    df = add_time_adjusted_score_diff(df)   # time_adjusted_score_diff
    df = add_field_position_features(df)     # red_zone, backed_up
    df = add_two_minute_drill_features(df)  # two_minute_drill (inline)
    return df
```

Remove the now-unused helper functions (`add_score_context_features`,
`add_game_progress_features`, `add_down_distance_features`) and trim
`add_score_features` so it only emits `score_differential`.

---

## Step 3 — Fix stale intent in `encoding.py`

**File:** `src/features/encoding.py`

**Keep all 4 encodings:** `is_turf`, `is_indoor`, `is_playoffs`, `is_home`.

The `encode_roof()` docstring currently references a module that does not exist and
describes behavior that has since moved:

```
Indoor games have structural temp/wind missingness, which is handled
downstream by weather_features.build_weather_features().
```

`weather_features.py` was never created, and temp/wind imputation is now done directly
in `clean_pbp.py` (dome/closed → 0). **Update the docstring** to reflect reality:

```
Indoor games (dome/closed) have no recorded temp/wind; those nulls are
imputed to 0 upstream in clean_pbp.py. This flag preserves the indoor
signal that the imputation would otherwise hide.
```

No functional change to `encoding.py` — docstring only.

---

## Step 4 — Wire builders into `feature_pipeline.py`

**File:** `src/features/feature_pipeline.py`

Add the three builders after the existing team-features call. Order does not matter
between them (they read disjoint raw columns), but run them after
`build_team_features` for consistency.

```python
from src.features.team_features import build_team_features
from src.features.formation_features import build_formation_features
from src.features.situational_features import build_situational_features
from src.features.encoding import build_encodings

def build_feature_dataset(pbp_path: Path = PBP_PATH) -> pd.DataFrame:
    pbp = pd.read_parquet(pbp_path)
    print(f"[INFO] Loaded pbp_clean: {pbp.shape}")

    pbp = build_team_features(pbp)
    pbp = build_formation_features(pbp)
    pbp = build_situational_features(pbp)
    pbp = build_encodings(pbp)

    print(f"[INFO] Final feature dataset: {pbp.shape}")
    return pbp
```

Update the module docstring's pipeline description to list the newly added feature
groups.

---

## Step 5 — Re-run the Pipeline

```bash
python3 -m src.preprocessing.clean_pbp      # unchanged; regenerate pbp_clean
python3 -m src.features.feature_pipeline     # rebuild model_dataset.parquet
```

Expected: current **43 columns → 55 columns** (43 + 12 new features).

Verify after the run:
- All 12 new columns are present.
- `is_heavy_formation`, `is_spread_formation`, `box_advantage` are fully populated
  (0 nulls — `offense_personnel` and `defenders_in_box` have no nulls).
- `score_differential`, `time_adjusted_score_diff`, `red_zone`, `backed_up`,
  `two_minute_drill` are fully populated.
- `is_turf`, `is_indoor`, `is_playoffs`, `is_home` are populated (a small number of
  `is_turf` nulls are acceptable for unrecognized surface strings).
- Spot-check `two_minute_drill`: it should equal 1 whenever
  `game_seconds_remaining <= 120` OR `half_seconds_remaining <= 120`.

---

## Step 6 — Update docs

**File:** `docs/feature_engineering.md`

Mark the integrated features as implemented and note that the unmentioned features from
`formation_features.py` / `situational_features.py` were deprecated in this pass. Remove
or annotate the references to the never-built `weather_features.py` and
`interaction_features.py` so the doc no longer implies work that isn't planned.

---

## File Summary

| Step | File | Action |
|---|---|---|
| 1 | `src/features/formation_features.py` | Keep 3 features; deprecate count outputs (keep as internal helpers) |
| 2 | `src/features/situational_features.py` | Keep 5 features; deprecate 17; recompute `two_minute_drill` inline |
| 3 | `src/features/encoding.py` | Keep 4 encodings; fix stale `weather_features` docstring |
| 4 | `src/features/feature_pipeline.py` | Wire in the three builders |
| 5 | _(run commands)_ | Regenerate `model_dataset.parquet` (→ 55 cols) |
| 6 | `docs/feature_engineering.md` | Update status; remove stale module references |
