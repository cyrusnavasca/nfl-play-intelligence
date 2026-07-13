# NFL Play Intelligence

Predict whether an NFL offensive play will be a **run** or a **pass** from the
pre-snap game state — down, distance, field position, score, formation, and
lagged team tendencies. Binary classification, evaluated with 5-fold
cross-validation on `roc_auc`.

## Pipeline

```
ingestion  ->  preprocessing  ->  features  ->  selection  ->  modeling
(nfl_data_py)  (clean pbp)        (engineer)    (screen 34)    (CV + persist)
```

1. **Ingest** raw play-by-play + player data from `nfl_data_py` → `data/raw/`.
2. **Clean** play-by-play into `data/interim/pbp_clean.parquet`.
3. **Engineer** situational, formation, rolling-team, and encoded features →
   `data/interim/features_full.parquet`.
4. **Select** — a **manual** candidate feature list (`configs/features/*.yaml`)
   feeds an embedded (LightGBM gain) pruning stage → the modeling set at
   `data/processed/play_type_modeling.parquet`. No automated filter screening.
5. **Model** — train/compare `baseline`, `random_forest`, `xgboost` under 5-fold
   CV; persist the winner to `artifacts/modeling/`.

Leakage guardrails: `epa`, `yards_gained`, IDs, dates, and raw score columns are
never model inputs (see `DROP_ALWAYS` in `src/selection/shared/feature_schema.py`).
The target is stored as `pass=1 / run=0`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Ingest raw data (play-by-play + players)
python3 -m src.ingestion.ingest_data --start-year 2018 --end-year 2025

# 2. Edit the manual feature list, then run selection (produces the modeling parquet)
#    configs/features/default.yaml  — numeric:/categorical: lists + embedded threshold
python3 -m src.selection.run_selection
python3 -m src.selection.run_selection --features configs/features/my_trial.yaml

# 3. Validate the modeling parquet matches the schema contract
python3 -m src.data.schema

# 4. Run a modeling experiment (profile-driven)
python3 -m src.pipelines.run --config configs/models/default.yaml

# Fixed experiment id / persist the winning model / reuse existing artifacts
python3 -m src.pipelines.run --config configs/models/xgboost_tuned.yaml --experiment exp_003
python3 -m src.pipelines.run --config configs/models/default.yaml --persist-best
python3 -m src.pipelines.run --config configs/models/default.yaml --skip-training
```

**Choosing features** — edit `configs/features/default.yaml` by hand (guided by
`notebooks/02_feature_selection.ipynb`: VIF, MI, ANOVA, chi-square). Delete a line
to drop a feature. Survivors go through embedded LightGBM pruning at
`embedded_importance_threshold` (set `0.0` to keep the manual list as-is).

**Tuning hyperparameters** — copy a model profile, edit values under
`models.<model_key>` (never Python literals):

```bash
cp configs/models/default.yaml configs/models/my_trial.yaml
python3 -m src.pipelines.run --config configs/models/my_trial.yaml
# compare runs via model_comparison.csv in the new experiment dir
```

## Layout

```
src/
├── ingestion/      download raw pbp + players (nfl_data_py)
├── preprocessing/  clean_pbp
├── features/       situational, formation, team-rolling, encoding pipelines
├── selection/      feature_config (manual), embedded, threshold -> modeling parquet
├── data/           schema.py (single source of truth: paths, target, registry)
├── models/         baseline, random_forest, xgboost builders + registry
├── evaluation/     cross_validation, metrics, feature_importance, model_selection
├── pipelines/      run/train/predict (thin wiring over evaluation/)
└── utils/          experiment profiles + artifact resolution

configs/
├── features/       manual feature lists (default.yaml) -> selection
└── models/         experiment profiles (default.yaml, xgboost_tuned.yaml)
notebooks/          01_data_exploration, 02_feature_selection, 03_model_evaluation
artifacts/modeling/ experiments/<id>/, best_model/, active.json
docs/               design + handoff notes (see modeling_plan.md first)
```

## Architecture rules

Enforced by `tests/test_architecture.py`:

```
pipelines/     -> models/, evaluation/, data/, preprocessing/, utils/
evaluation/    -> data/, utils/          (never imports pipelines/)
models/        -> data/                  (never imports evaluation/)
```

All CV/fold logic lives in `src/evaluation/`; `src/pipelines/` is thin wiring.
Column names, paths, target, and model-registry keys come only from
`src/data/schema.py` and `src/selection/shared/feature_schema.py`.

## Tests

```bash
pytest                 # full suite (testpaths=tests)
pytest tests/pipelines/test_play_type_train.py -q
```

## Docs

- `docs/modeling_plan.md` — modeling pipeline design, phase handoffs, target layout.
- `docs/feature_selection_plan.md` — upstream selection producing the modeling parquet.
- `docs/running_experiments.md` — profiles, starter configs, comparing runs.
- `docs/handoff_metrics_and_hyperparams.md` — metrics/config snapshot schema.
