# Project Structure

```
nfl-play-intelligence/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── inference.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baselines.ipynb
│   └── 04_error_analysis.ipynb
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── download_data.py
│   │   ├── update_data.py
│   │   └── validate_schema.py
│   │
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── missing_values.py
│   │   ├── split_data.py
│   │   └── encoding.py
│   │
│   ├── features/
│   │   ├── rolling_features.py
│   │   ├── player_features.py
│   │   ├── team_features.py
│   │   ├── situational_features.py
│   │   ├── weather_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── neural_network.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   ├── calibration.py
│   │   └── error_analysis.py
│   │
│   ├── explainability/
│   │   ├── shap_values.py
│   │   ├── feature_importance.py
│   │   └── partial_dependence.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   ├── model_loader.py
│   │   └── preprocessing.py
│   │
│   ├── visualization/
│   │   ├── plots.py
│   │   ├── dashboards.py
│   │   └── game_reports.py
│   │
│   ├── utils/
│   │   ├── constants.py
│   │   ├── logging.py
│   │   ├── helpers.py
│   │   └── paths.py
│   │
│   └── pipeline.py
│
├── models/
│   ├── checkpoints/
│   ├── trained/
│   └── metadata/
│
├── artifacts/
│   ├── feature_importance/
│   ├── shap/
│   ├── predictions/
│   └── reports/
│
├── api/
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── dependencies.py
│
├── dashboard/
│   ├── Home.py
│   ├── pages/
│   │   ├── Team_Explorer.py
│   │   ├── Player_Explorer.py
│   │   ├── Play_Predictor.py
│   │   └── Model_Insights.py
│   └── assets/
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── scripts/
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   └── update_dataset.py
│
└── docs/
    ├── project_structure.md
    ├── architecture.md
    ├── feature_dictionary.md
    └── model_card.md
```
