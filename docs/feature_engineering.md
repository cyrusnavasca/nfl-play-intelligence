# Feature Engineering Plan

Comprehensive list of features to engineer from `pbp_clean.parquet` and `players_clean.parquet`.
Features are grouped by source and type. Implementation status tracked against `src/features/`.

---

## Status Key
- ✅ Already implemented (`src/features/situational_features.py`)
- 🔧 Needs implementation
- ⚠️ Requires joining player data back to PBP first

---

## 1. Category Simplifications / Encodings

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

### Already Implemented ✅
| Feature | Logic | Notes |
|---|---|---|
| `score_differential` | `posteam_score - defteam_score` | Positive = leading |
| `time_adjusted_score_diff` | `score_differential × game_progress` | Urgency-weighted |
| `red_zone` | `yardline_100 ≤ 20` | Binary |
| `backed_up` | `yardline_100 ≥ 80` | Binary |
| `two_minute_game` | `game_seconds_remaining ≤ 120` | Binary |
| `two_minute_half` | `half_seconds_remaining ≤ 120` | Binary |
| `two_minute_drill` | Either two-minute flag is 1 | Combined binary |

### To Engineer 🔧

**Game progress**
| Feature | Logic | Intuition |
|---|---|---|
| `game_progress` | `1 - (game_seconds_remaining / 3600)` | 0.0 at kickoff → 1.0 at final whistle |
| `half_progress` | `1 - (half_seconds_remaining / 1800)` | Urgency within a half |
| `is_overtime` | `qtr == 5` | OT plays have distinct tendencies |

**Score context**
| Feature | Logic | Intuition |
|---|---|---|
| `score_diff_abs` | `abs(score_differential)` | Blowout vs. close game |
| `is_blowout` | `score_diff_abs ≥ 17` | Team likely runs more to protect lead |
| `is_close` | `score_diff_abs ≤ 3` | Pass-heavy when every drive matters |
| `game_script` | `score_differential × game_progress` | Combines deficit + urgency in one signal |
| `total_score` | `posteam_score + defteam_score` | High-scoring games may open up play calls |

**Down & distance context**
| Feature | Logic | Intuition |
|---|---|---|
| `distance_bucket` | `short (1–3), medium (4–6), long (7+)` | Categorical compression of ydstogo |
| `ydstogo_normalized` | `ydstogo / 10` | Scales distance to 0–1-ish range |
| `yards_available` | `100 - yardline_100` | Yards between LOS and end zone |
| `ydstogo_ratio` | `ydstogo / yards_available` | How much of remaining field is needed |
| `is_short_yardage` | `down ≥ 2 & ydstogo ≤ 2` | Classic power run situation |
| `is_third_and_long` | `down == 3 & ydstogo ≥ 7` | Pass-heavy; defense knows it too |
| `is_fourth_down` | `down == 4` | Distinct decision point |

---

## 3. Personnel & Formation Features

### 3.1 Offensive Personnel Parsing
Parse `offense_personnel` (e.g. `"1 RB, 1 TE, 3 WR"`) into numeric counts.

| Feature | Logic |
|---|---|
| `off_rb_count` | Extract RB number from string |
| `off_te_count` | Extract TE number |
| `off_wr_count` | Extract WR number |
| `personnel_package` | Label: `"11"` (1RB/1TE/3WR), `"12"` (1RB/2TE/2WR), `"21"`, `"22"`, etc. |
| `is_heavy_formation` | `off_rb_count + off_te_count ≥ 3` — run-leaning package |
| `is_spread_formation` | `off_wr_count ≥ 3` — pass-leaning package |

> Standard NFL shorthand: first digit = RBs, second = TEs (e.g. "11" = 1 RB, 1 TE, 3 WR).

### 3.2 Defensive Personnel Parsing
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
| Feature | Logic | Intuition |
|---|---|---|
| `box_advantage` | `off_rb_count + off_te_count - defenders_in_box` | Positive = numbers advantage in run game |
| `is_stacked_box` | `defenders_in_box ≥ 8` | Defense loaded for run stop |
| `is_light_box` | `defenders_in_box ≤ 5` | Defense expecting pass |
| `pass_rusher_ratio` | `number_of_pass_rushers / def_dl_count` | Blitz indicator |
| `is_blitz` | `number_of_pass_rushers ≥ 5` | Extra pressure coming |

---

## 4. Interaction Features

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

| Feature | Logic | Notes |
|---|---|---|
| `is_turf` | See §1.1 | Binary |
| `is_indoor` | See §1.2 | Binary; also explains temp/wind nulls |
| `temp_bucket` | `≤32, 33–45, 46–60, 61–75, >75` | Ordinal or one-hot |
| `wind_bucket` | `0–5, 6–10, 11–15, 16–20, >20` | Ordinal or one-hot |
| `is_freezing` | `temp ≤ 32` | Extreme cold flag |
| `is_high_wind` | `wind ≥ 20` | Extreme wind flag; strongly suppresses passing |
| `is_bad_weather` | `is_freezing OR is_high_wind` | Combined adverse weather flag |

> For `is_indoor` rows, fill `temp` and `wind` with their respective means (or 0) before bucketing, since missing-ness is structural not random.

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

| Category | Target file |
|---|---|
| Category simplifications | `src/features/encoding.py` |
| Game state (additions) | `src/features/situational_features.py` |
| Personnel / formation parsing | `src/features/formation_features.py` |
| Interaction features | `src/features/interaction_features.py` |
| Environmental features | `src/features/weather_features.py` |
| Rolling player features | `src/features/player_features.py` |
| Rolling team features | `src/features/team_features.py` |
| Full pipeline | `src/features/feature_pipeline.py` |
