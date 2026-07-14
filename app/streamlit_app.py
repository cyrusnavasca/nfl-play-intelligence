"""
NFL Play Intelligence — interactive Streamlit dashboard.

Run:
    streamlit run app/streamlit_app.py

Loads the persisted best model (artifacts/modeling/best_model/) and lets you
explore experiments, feature importance, and score hypothetical play situations.
All artifact paths are resolved through src.utils helpers — never hardcoded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Run from repo root so repo-relative artifact/parquet paths resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_play_type_dataset
from src.utils.experiments import (
    best_model_dir,
    experiment_root,
    list_experiments,
    read_experiment_config,
)
from src.utils.io import MODEL_FILENAME, load_model

st.set_page_config(
    page_title="NFL Play Intelligence",
    page_icon="🏈",
    layout="wide",
)

CLASS_NAMES = {0: "run", 1: "pass"}
FORMATIONS = [
    "SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY",
    "PISTOL", "UNDER CENTER", "JUMBO", "WILDCAT",
]


# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading best model…")
def get_model():
    return load_model(best_model_dir() / MODEL_FILENAME)


@st.cache_data(show_spinner="Loading dataset…")
def get_dataset_summary():
    X, y = load_play_type_dataset()
    medians = X.median()
    return {
        "columns": list(X.columns),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "medians": medians,
        "class_counts": {
            "pass": int((y == 1).sum()),
            "run": int((y == 0).sum()),
        },
    }


@st.cache_data
def get_feature_importance():
    fi_dir = best_model_dir() / "feature_importance"
    files = list(fi_dir.glob("*.csv")) if fi_dir.exists() else []
    if not files:
        return None
    return pd.read_csv(files[0])


def _roc_auc_for(exp_id, cfg, best):
    """roc_auc from config, falling back to model_comparison.csv (source of truth)."""
    if cfg.get("roc_auc") is not None:
        return cfg["roc_auc"]
    comp = experiment_root(exp_id) / "model_comparison.csv"
    if best and comp.exists():
        df = pd.read_csv(comp)
        hit = df[df["model"] == best]
        if not hit.empty:
            return float(hit["roc_auc_mean"].iloc[0])
    return None


@st.cache_data
def get_experiments_table():
    rows = []
    for exp_id in list_experiments():
        try:
            cfg = read_experiment_config(exp_id)
        except Exception:
            continue
        best = cfg.get("best_model")
        hp = (cfg.get("models", {}).get(best, {}) or {}).get("hyperparameters", {}) if best else {}
        rows.append({
            "experiment": exp_id,
            "name": cfg.get("name"),
            "best_model": best,
            "roc_auc": _roc_auc_for(exp_id, cfg, best),
            "learning_rate": hp.get("learning_rate"),
            "n_estimators": hp.get("n_estimators"),
            "max_depth": hp.get("max_depth"),
            "persisted": cfg.get("persisted_best", False),
        })
    return pd.DataFrame(rows)


@st.cache_data
def get_best_metadata():
    import json
    meta_path = best_model_dir() / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


# ── Feature vector builder ──────────────────────────────────────────────────
def set_onehot(vec: pd.Series, prefix: str, value: str) -> None:
    """Zero every column in a one-hot group, then set the chosen member to 1."""
    members = [c for c in vec.index if c.startswith(prefix)]
    for m in members:
        vec[m] = 0.0
    key = f"{prefix}{value}"
    if key in vec.index:
        vec[key] = 1.0


def build_feature_row(summary, inputs) -> pd.DataFrame:
    """Start from median baseline, override with the user's play situation."""
    vec = summary["medians"].copy().astype(float)

    # Raw numerics.
    vec["down"] = float(inputs["down"])
    vec["ydstogo"] = float(inputs["ydstogo"])
    vec["yardline_100"] = float(inputs["yardline_100"])
    vec["score_differential"] = float(inputs["score_diff"])
    vec["time_adjusted_score_diff"] = float(inputs["score_diff"])
    vec["game_seconds_remaining"] = float(inputs["secs_remaining"])
    vec["qtr"] = float(inputs["qtr"])
    vec["defenders_in_box"] = float(inputs["box"])

    # One-hot groups.
    set_onehot(vec, "is_qb_in_gun_", "1" if inputs["shotgun"] else "0")
    set_onehot(vec, "two_minute_drill_", "1" if inputs["two_min"] else "0")
    set_onehot(vec, "red_zone_", "1" if inputs["yardline_100"] <= 20 else "0")
    set_onehot(vec, "offense_formation_", inputs["formation"])
    half = "Overtime" if inputs["qtr"] >= 5 else ("Half1" if inputs["qtr"] <= 2 else "Half2")
    set_onehot(vec, "game_half_", half)

    return vec.reindex(summary["columns"]).to_frame().T


def predict_pass_prob(model, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row)[:, 1][0])


# ── App ─────────────────────────────────────────────────────────────────────
model = get_model()
summary = get_dataset_summary()
meta = get_best_metadata()

st.title("🏈 NFL Play Intelligence")
st.caption(
    f"Run vs. pass prediction · best model **{meta.get('model_key', '?')}** "
    f"from experiment **{meta.get('experiment_id', '?')}** · "
    f"{summary['n_rows']:,} plays × {summary['n_features']} features"
)

tab_overview, tab_fi, tab_exp, tab_predict = st.tabs(
    ["📋 Overview", "📊 Feature Importance", "🧪 Experiments", "🎯 Play Predictor"]
)

# ── Overview ──
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    exp_df = get_experiments_table()
    best_auc = None
    if not exp_df.empty and exp_df["roc_auc"].notna().any():
        best_auc = exp_df.loc[exp_df["roc_auc"].idxmax(), "roc_auc"]
    c1.metric("Best ROC-AUC", f"{best_auc:.4f}" if best_auc else "—")
    c2.metric("Plays", f"{summary['n_rows']:,}")
    c3.metric("Features", summary["n_features"])
    pass_rate = summary["class_counts"]["pass"] / summary["n_rows"]
    c4.metric("Pass rate", f"{pass_rate:.1%}")

    st.subheader("Class balance")
    cc = summary["class_counts"]
    pie = px.pie(
        names=["pass", "run"], values=[cc["pass"], cc["run"]],
        color=["pass", "run"],
        color_discrete_map={"pass": "#1f77b4", "run": "#ff7f0e"},
        hole=0.45,
    )
    pie.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(pie, use_container_width=True)

    with st.expander("Model card (metadata.json)"):
        st.json(meta)

# ── Feature importance ──
with tab_fi:
    fi = get_feature_importance()
    if fi is None:
        st.warning("No feature_importance CSV found in best_model/.")
    else:
        st.subheader("Which features drive the prediction?")
        top_n = st.slider("Show top N features", 5, min(40, len(fi)), 15)
        top = fi.nlargest(top_n, "importance").sort_values("importance")
        bar = px.bar(
            top, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues",
        )
        bar.update_layout(height=28 * top_n + 80, margin=dict(t=10, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(bar, use_container_width=True)
        st.caption(
            "`is_qb_in_gun` dominates — shotgun formation is the single strongest "
            "run/pass tell in the data."
        )

# ── Experiments ──
with tab_exp:
    exp_df = get_experiments_table()
    if exp_df.empty:
        st.info("No experiments found.")
    else:
        st.subheader("Experiment leaderboard")
        show = exp_df.sort_values("roc_auc", ascending=False, na_position="last")
        st.dataframe(
            show.style.format({"roc_auc": "{:.4f}", "learning_rate": "{:.4g}"}),
            use_container_width=True, hide_index=True,
        )

        lr_df = exp_df.dropna(subset=["roc_auc", "learning_rate"])
        if len(lr_df) > 1 and lr_df["learning_rate"].nunique() > 1:
            st.subheader("Learning-rate sweep")
            sweep = lr_df.sort_values("learning_rate")
            line = px.line(
                sweep, x="learning_rate", y="roc_auc", markers=True,
                text="experiment", log_x=True,
            )
            line.update_traces(textposition="top center")
            line.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(line, use_container_width=True)
            st.caption("Lower learning rate wins at fixed n_estimators=800.")

# ── Play predictor ──
with tab_predict:
    st.subheader("Score a hypothetical play")
    left, right = st.columns([1, 1])

    with left:
        down = st.selectbox("Down", [1, 2, 3, 4], index=0)
        ydstogo = st.slider("Yards to go", 1, 30, 10)
        yardline_100 = st.slider("Yards from end zone", 1, 99, 50)
        qtr = st.selectbox("Quarter", [1, 2, 3, 4, 5], index=0,
                           format_func=lambda q: "OT" if q == 5 else f"Q{q}")
        secs_remaining = st.slider("Seconds left in game", 0, 3600, 1800, step=30)
        score_diff = st.slider("Score differential (offense − defense)", -35, 35, 0)
        box = st.slider("Defenders in box", 1, 11, 6)
        formation = st.selectbox("Offense formation", FORMATIONS)
        cola, colb = st.columns(2)
        shotgun = cola.toggle("Shotgun (QB in gun)", value=True)
        two_min = colb.toggle("Two-minute drill", value=False)

    inputs = dict(
        down=down, ydstogo=ydstogo, yardline_100=yardline_100, qtr=qtr,
        secs_remaining=secs_remaining, score_diff=score_diff, box=box,
        formation=formation, shotgun=shotgun, two_min=two_min,
    )
    row = build_feature_row(summary, inputs)
    p_pass = predict_pass_prob(model, row)

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_pass * 100,
            number={"suffix": "%"},
            title={"text": "P(pass)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#ffe8d6"},
                    {"range": [50, 100], "color": "#d6e6ff"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.8, "value": 50},
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        call = "PASS" if p_pass >= 0.5 else "RUN"
        emoji = "🎯" if p_pass >= 0.5 else "🏃"
        st.markdown(f"### {emoji} Predicted call: **{call}**")
        st.progress(p_pass, text=f"pass {p_pass:.1%} · run {1 - p_pass:.1%}")

    st.divider()
    st.subheader("What-if: sweep one variable")
    sweep_feat = st.selectbox(
        "Vary", ["ydstogo", "yardline_100", "score_diff", "secs_remaining", "box"]
    )
    ranges = {
        "ydstogo": range(1, 31),
        "yardline_100": range(1, 100, 2),
        "score_diff": range(-35, 36, 2),
        "secs_remaining": range(0, 3601, 120),
        "box": range(1, 12),
    }
    xs, ys = [], []
    for v in ranges[sweep_feat]:
        swept = dict(inputs)
        swept[sweep_feat] = v
        ys.append(predict_pass_prob(model, build_feature_row(summary, swept)))
        xs.append(v)
    pdp = px.line(x=xs, y=ys, markers=True,
                  labels={"x": sweep_feat, "y": "P(pass)"})
    pdp.add_hline(y=0.5, line_dash="dash", line_color="gray")
    pdp.update_layout(height=360, margin=dict(t=10, b=10), yaxis_range=[0, 1])
    st.plotly_chart(pdp, use_container_width=True)
    st.caption("Holds all other inputs at their current values above.")
