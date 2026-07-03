# Feature Engineering Plan

Comprehensive list of features to engineer from `pbp_clean.parquet` and `players_clean.parquet`.
Features are grouped by source and type. Implementation status tracked against `src/features/`.

---

## Status Key
- ✅ Implemented and wired into `feature_pipeline.py` (lands in `model_dataset.parquet`)
- 🔧 Needs implementation
- 🗑️ Deprecated — built once but intentionally not integrated (see Core Features Integration, `docs/core_features_integration_plan.md`)
- ⚠️ Requires joining player data back to PBP first

> **Core features integration (implemented):** A curated 12-feature subset from
> `formation_features.py`, `situational_features.py`, and `encoding.py` is now wired into
> the pipeline. Everything from those modules not listed as ✅ below was deprecated in that
> pass. `weather_features.py` and `interaction_features.py` were never built and are not
> currently planned — their sections below are kept as future ideas only.

---

## 1. Category Simplifications / Encodings

✅ All four encodings below are implemented in `encoding.py` and wired into the pipeline
(`is_turf`, `is_indoor`, `is_playoffs`, `is_home`).

These raw string columns need to be collapsed or encoded before modeling.

### 1.1 Surface → `is_turf`
```
grass                       → 0
fieldturf, matrixturf,
sportturf, astroturf, a_turf → 1
blank / unknown             → NaN
```
> Binary is sufficient. Specific turf brand adds noise with no proven signal difference.

### 1.2 Roof → `is_indoor`
```
dome, closed  → 1  (fully enclosed, no weather effects)
outdoors, open → 0  (exposed to elements)
```

### 1.3 Season Type → `is_playoffs`
```
REG  → 0
POST → 1
```

### 1.4 Possession Team Side → `is_home`
```
posteam_type: home → 1
posteam_type: away → 0
```

---

## 2. Game State Features

### Implemented & wired ✅
| Feature | Logic | Notes |
|---|---|---|
| `score_differential` | `posteam_score - defteam_score` | Positive = leading |
| `time_adjusted_score_diff` | `score_differential × game_progress` | Urgency-weighted; game progress computed inline |
| `red_zone` | `yardline_100 ≤ 20` | Binary |
| `backed_up` | `yardline_100 ≥ 80` | Binary |
| `two_minute_drill` | `game_seconds_remaining ≤ 120` OR `half_seconds_remaining ≤ 120` | Combined binary, computed inline |

### Deprecated 🗑️ (built once, not integrated)
The intermediate two-minute flags and the following context features were removed from
`situational_features.py` during the core-features integration — they were never wired in
and are not part of the curated set:

| Feature | Reason |
|---|---|
| `two_minute_game`, `two_minute_half` | Folded inline into `two_minute_drill` |
| `total_score` | Not in curated set |
| `score_diff_abs`, `is_blowout`, `is_close`, `game_script` | Not in curated set |
| `game_progress`, `half_progress`, `is_overtime` | Not in curated set |
| `distance_bucket`, `ydstogo_normalized`, `yards_available`, `ydstogo_ratio` | Not in curated set |
| `is_short_yardage`, `is_third_and_long`, `is_fourth_down` | Not in curated set |

---

## 3. Personnel & Formation Features

### 3.1 Offensive Personnel Parsing
Parse `offense_personnel` (e.g. `"1 RB, 1 TE, 3 WR"`) into numeric counts.

| Feature | Logic | Status |
|---|---|---|
| `off_rb_count` | Extract RB number from string | 🗑️ internal-only (temp, dropped before output) |
| `off_te_count` | Extract TE number | 🗑️ internal-only (temp, dropped before output) |
| `off_wr_count` | Extract WR number | 🗑️ internal-only (temp, dropped before output) |
| `personnel_package` | Label: `"11"`, `"12"`, `"21"`, `"22"`, etc. | 🔧 not built |
| `is_heavy_formation` | `off_rb_count + off_te_count ≥ 3` — run-leaning package | ✅ implemented & wired |
| `is_spread_formation` | `off_wr_count ≥ 3` — pass-leaning package | ✅ implemented & wired |

> Standard NFL shorthand: first digit = RBs, second = TEs (e.g. "11" = 1 RB, 1 TE, 3 WR).
> The three position counts are kept as internal temporaries — the surviving features derive
> from them — but are dropped before the frame is returned, so they never reach
> `model_dataset.parquet`.

### 3.2 Defensive Personnel Parsing 🔧 (not built)
Parse `defense_personnel` (e.g. `"4 DL, 2 LB, 5 DB"`) into numeric counts.

| Feature | Logic |
|---|---|
| `def_dl_count` | Extract DL number |
| `def_lb_count` | Extract LB number |
| `def_db_count` | Extract DB number |
| `defense_package` | Label: `"base"` (4 DB), `"nickel"` (5 DB), `"dime"` (6 DB), `"prevent"` (7+ DB) |
| `is_nickel_or_dime` | `def_db_count ≥ 5` — light box, pass defense |
| `box_db_ratio` | `defenders_in_box / def_db_count` — run vs. pass alignment signal |

### 3.3 Defenders in Box
| Feature | Logic | Intuition | Status |
|---|---|---|---|
| `box_advantage` | `off_rb_count + off_te_count - defenders_in_box` | Positive = numbers advantage in run game | ✅ implemented & wired |
| `is_stacked_box` | `defenders_in_box ≥ 8` | Defense loaded for run stop | 🔧 not built |
| `is_light_box` | `defenders_in_box ≤ 5` | Defense expecting pass | 🔧 not built |
| `pass_rusher_ratio` | `number_of_pass_rushers / def_dl_count` | Blitz indicator | 🔧 not built |
| `is_blitz` | `number_of_pass_rushers ≥ 5` | Extra pressure coming | 🔧 not built |

---

## 4. Interaction Features 🔧 (not built — future idea only)

> There is no `interaction_features.py` module. None of the features below are built or
> planned; this section is retained purely as a backlog of ideas.

Combinations that are more informative than either feature alone.

| Feature | Components | Intuition |
|---|---|---|
| `down_x_distance` | `down × ydstogo` | 3rd-and-1 vs. 3rd-and-10 are completely different plays |
| `down_x_yardline` | `down × yardline_100` | 1st-and-10 at own 20 vs. opponent 20 |
| `score_diff_x_time` | `score_differential × game_seconds_remaining` | Blowout with 5 min left vs. blowout with 2 min left |
| `box_x_distance` | `defenders_in_box × ydstogo` | Stacked box on 3rd-and-1 vs. 3rd-and-10 |
| `formation_x_down` | `offense_formation × down` | Shotgun on 1st down is different than shotgun on 3rd |
| `personnel_x_ydstogo` | `personnel_package × distance_bucket` | 12 personnel on short yardage vs. 11 personnel |

---

## 5. Environmental Features

`is_turf` and `is_indoor` are implemented in `encoding.py` (see §1) and wired into the
pipeline. The weather buckets below are 🔧 **not built** — there is no `weather_features.py`
module, and they are not currently planned. Note that temp/wind nulls (indoor games) are
imputed to 0 upstream in `clean_pbp.py`, so `is_indoor` preserves that signal.

| Feature | Logic | Status |
|---|---|---|
| `is_turf` | See §1.1 | ✅ implemented & wired |
| `is_indoor` | See §1.2; also explains temp/wind nulls | ✅ implemented & wired |
| `temp_bucket` | `≤32, 33–45, 46–60, 61–75, >75` | 🔧 not built |
| `wind_bucket` | `0–5, 6–10, 11–15, 16–20, >20` | 🔧 not built |
| `is_freezing` | `temp ≤ 32` | 🔧 not built |
| `is_high_wind` | `wind ≥ 20` | 🔧 not built |
| `is_bad_weather` | `is_freezing OR is_high_wind` | 🔧 not built |

---

## 6. Rolling / Lagged Player Features

> ⚠️ Requires: (1) add `passer_player_id`, `rusher_player_id`, `receiver_player_id` back to `pbp_clean`, then (2) join `players_clean` on `(player_id, season, week - 1)` or a rolling window.
> These features are pre-snap observable — always use the **prior week** to avoid leakage.

### 6.1 QB Features (join on passer)
| Feature | Window | Source column |
|---|---|---|
| `qb_pass_yds_L3` | Rolling 3-week avg | `passing_yards` |
| `qb_pass_epa_L3` | Rolling 3-week avg | `passing_epa` |
| `qb_air_yards_L3` | Rolling 3-week avg | `passing_air_yards` |
| `qb_int_rate_L3` | `interceptions / attempts`, 3-week | `interceptions`, `attempts` |
| `qb_season_epa` | Season-to-date avg | `passing_epa` |

### 6.2 RB Features (join on rusher)
| Feature | Window | Source column |
|---|---|---|
| `rb_rush_yds_L3` | Rolling 3-week avg | `rushing_yards` |
| `rb_rush_epa_L3` | Rolling 3-week avg | `rushing_epa` |
| `rb_ypc_L3` | `rushing_yards / carries`, 3-week | `rushing_yards`, `carries` |
| `rb_season_carries` | Season-to-date total | `carries` |

### 6.3 WR / TE Features (join on receiver)
| Feature | Window | Source column |
|---|---|---|
| `rec_yards_L3` | Rolling 3-week avg | `receiving_yards` |
| `rec_epa_L3` | Rolling 3-week avg | `receiving_epa` |
| `target_share_L3` | Rolling 3-week avg | `target_share` |
| `air_yards_share_L3` | Rolling 3-week avg | `air_yards_share` |

---

## 7. Team-Level Rolling Features

Aggregated from PBP by possession/defense team per rolling window.

| Feature | Logic | Window |
|---|---|---|
| `posteam_pass_rate_L3` | `% pass plays` for posteam | Last 3 games |
| `posteam_epa_per_play_L3` | Mean EPA per play for posteam | Last 3 games |
| `posteam_rush_yds_L3` | Mean rushing yards per carry | Last 3 games |
| `defteam_epa_allowed_L3` | Mean EPA allowed per play | Last 3 games |
| `defteam_pass_rate_faced_L3` | `% pass plays` faced by defteam | Last 3 games |
| `home_advantage` | `is_home` (already exists) | — |

---

## 8. Target Encoding Candidates

High-cardinality categorical columns that shouldn't be one-hot encoded.

| Column | Cardinality | Suggested encoding |
|---|---|---|
| `posteam` | 32 teams | Mean target encoding (cross-validated) |
| `defteam` | 32 teams | Mean target encoding |
| `offense_formation` | ~8 values | One-hot or ordinal |
| `personnel_package` | ~6 values | One-hot |
| `defense_package` | ~4 values | One-hot |

---

## 9. Features to Drop / Exclude at Training Time

These columns exist in `pbp_clean` but must never enter the model as features.

| Column | Reason |
|---|---|
| `yards_gained` | Target variable |
| `game_id`, `play_id` | Identifiers only |
| `game_date` | Leaks season/week info — use `season` + `week` instead |
| `posteam_score`, `defteam_score` | Replaced by engineered score features |
| `home_team`, `away_team` | Encoded into `is_home` |
| `game_half` | Covered by `qtr` + time features |
| `offense_personnel`, `defense_personnel` | Raw strings — replaced by parsed features |
| `roof`, `surface` | Raw strings — replaced by `is_indoor`, `is_turf` |
| `temp`, `wind` | Raw numerics — may keep alongside bucket flags |

---

## Implementation Map

| Category | Target file | Status |
|---|---|---|
| Category simplifications | `src/features/encoding.py` | ✅ wired |
| Game state (curated subset) | `src/features/situational_features.py` | ✅ wired |
| Personnel / formation parsing | `src/features/formation_features.py` | ✅ wired |
| Interaction features | _(none — never built)_ | 🔧 not planned |
| Environmental features | _(none — never built)_ | 🔧 not planned |
| Rolling player features | `src/features/player_features.py` | 🔧 not built |
| Rolling team features | `src/features/team_features.py` | ✅ wired |
| Full pipeline | `src/features/feature_pipeline.py` | ✅ |
