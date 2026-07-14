# NFL Play Intelligence

## Overview

Predict whether an NFL offensive play will be a **run** or a **pass** from the
pre-snap game state — down, distance, field position, score, formation, and
lagged team tendencies. Binary classification (`pass=1 / run=0`), evaluated with
5-fold cross-validation on `roc_auc`. An interactive Streamlit dashboard scores
hypothetical plays and explores experiments.

## Data source

Play-by-play from [`nfl_data_py`](https://github.com/nflverse/nfl_data_py)
(`import_pbp_data`), **seasons 2018–2025** → 276,286 modeled plays after cleaning
and leakage filtering. Ingest:

```bash
python3 -m src.ingestion.ingest_data --start-year 2018 --end-year 2025
```

Leakage guardrails: `epa`, `yards_gained`, IDs, dates, and raw score columns are
never model inputs (see `DROP_ALWAYS` in `src/selection/shared/feature_schema.py`).

## Pipeline

```
ingestion  ->  preprocessing  ->  features  ->  selection  ->  modeling
(nfl_data_py)  (clean pbp)        (engineer)    (manual+embedded)  (CV + persist)
```

1. **Ingest** raw play-by-play → `data/raw/`.
2. **Clean** into `data/interim/pbp_clean.parquet`.
3. **Engineer** situational, formation, rolling-team, encoded features.
4. **Select** — a **manual** candidate list (`configs/features/*.yaml`) feeds an
   embedded (LightGBM gain) pruning stage → `data/processed/play_type_modeling.parquet`.
5. **Model** — train/compare `baseline`, `random_forest`, `xgboost` under 5-fold
   CV; persist the winner to `artifacts/modeling/`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard

The persisted best model (`artifacts/modeling/best_model/`) powers an interactive
Streamlit app:

```bash
streamlit run app/streamlit_app.py          # opens http://localhost:8501
```

Tabs: **Overview** (metrics, class balance) · **Feature Importance** (top-N bar) ·
**Experiments** (leaderboard + learning-rate sweep) · **Play Predictor** (down /
distance / formation / shotgun inputs → live P(pass) gauge + a what-if sweep).

## Run experiments

```bash
# feature selection (edit the manual list first)
python3 -m src.selection.run_selection --features configs/features/default.yaml

# validate the modeling parquet against the schema
python3 -m src.data.schema

# modeling experiment (profile-driven)
python3 -m src.pipelines.run --config configs/models/default.yaml
python3 -m src.pipelines.run --config configs/models/xgboost_tuned.yaml --tune --persist-best
```

Tune by copying a profile and editing values under `models.<model_key>` (never
Python literals); compare runs via `model_comparison.csv` in the new experiment dir.
See `docs/running_experiments.md`.

## Results

Best model: **XGBoost**, tuned via Optuna TPE (experiment `exp_007`), persisted to
`artifacts/modeling/best_model/`.

| Metric | Value |
|--------|-------|
| ROC-AUC (5-fold CV) | **0.8178** ± 0.0012 |
| ROC-AUC (holdout test) | 0.8165 |
| Plays / features | 276,286 / 65 |
| Pass rate (base) | 58.2% |

Most important features (gain):

1. `is_qb_in_gun` — shotgun is the single strongest run/pass tell
2. `offense_formation_EMPTY`
3. `two_minute_drill`
4. `defenders_in_box`
5. `down`, `ydstogo`

Reproduce the eval: set `EXPERIMENT_ID` at the top of
`notebooks/03_model_evaluation.ipynb` (or leave `None` for the active experiment).

## File structure

```
app/
└── streamlit_app.py     interactive dashboard (loads best_model/)
src/
├── ingestion/           download raw pbp + players (nfl_data_py)
├── preprocessing/       clean_pbp
├── features/            situational, formation, team-rolling, encoding
├── selection/           feature_config (manual), embedded -> modeling parquet
├── data/                schema.py (single source of truth: paths, target, registry)
├── models/              baseline, random_forest, xgboost builders + registry
├── evaluation/          cross_validation, metrics, feature_importance, model_selection
├── pipelines/           run/train/predict (thin wiring over evaluation/)
└── utils/               experiment profiles + artifact resolution
configs/
├── features/            manual feature lists -> selection
└── models/              experiment profiles (default.yaml, xgboost_tuned.yaml, ...)
notebooks/               01_data_exploration, 02_feature_selection, 03_model_evaluation
artifacts/modeling/      experiments/<id>/, best_model/, active.json
docs/                    design + handoff notes (start with modeling_plan.md)
tests/                   suite incl. test_architecture.py (dependency rules)
```

## Architecture rules

Enforced by `tests/test_architecture.py`:

```
pipelines/  -> models/, evaluation/, data/, preprocessing/, utils/
evaluation/ -> data/, utils/    (never imports pipelines/)
models/     -> data/            (never imports evaluation/)
```

All CV/fold logic lives in `src/evaluation/`; `src/pipelines/` is thin wiring.
Column names, paths, target, and registry keys come only from `src/data/schema.py`
and `src/selection/shared/feature_schema.py`.

## Tests

```bash
pytest                 # full suite (testpaths=tests)
```

## Docs

- `docs/modeling_plan.md` — modeling pipeline design, phase handoffs, target layout.
- `docs/feature_selection_plan.md` — upstream selection producing the modeling parquet.
- `docs/running_experiments.md` — profiles, starter configs, comparing runs.
- `docs/handoff_metrics_and_hyperparams.md` — metrics/config snapshot schema.
```
