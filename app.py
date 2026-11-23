"""Streamlit dashboard: genuine, held-out Sri Lankan next-day rainfall comparison."""
# AI-assisted implementation: OpenAI ChatGPT (GPT-6), 2026-11-22. See AI_DISCLOSURE.md.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve
from data_utils import (
    DATA_URL, FEATURES, download_data, load_raw, prepare_data, chronological_split
)
from modeling import run_experiment

st.set_page_config(
    page_title="Sri Lanka | Ensemble Rainfall Lab", page_icon="🌧️", layout="wide"
)
st.markdown("""<style>
.block-container {padding-top:1.6rem; max-width:1260px}
[data-testid="stMetric"] {background:#f3f7fb; border-radius:12px; padding:14px}
[data-testid="stMetricLabel"] {font-size:.85rem}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def data_for_dashboard():
    download_data()
    return prepare_data(load_raw())

@st.cache_resource(show_spinner=False)
def models_for_dashboard(_train, _validation, _test):
    # Cached for each Streamlit session; data arguments are excluded from hashing.
    # Historical CSV is fixed, so this matches the exact loaded dataset.
    return run_experiment(_train, _validation, _test, save=False, verbose=False)

st.caption("ENSEMBLE LEARNING • SRI LANKAN WEATHER • REPRODUCIBLE EVALUATION")
st.title("🌧️ Sri Lanka Rainfall Prediction Lab")
st.write("Compare **Bagging, Boosting and Stacking** when predicting whether a Sri Lankan district will receive **at least 1 mm of rain tomorrow**.")

try:
    with st.spinner("Loading historical Sri Lankan weather data…"):
        data = data_for_dashboard()
        train, validation, test = chronological_split(data)
except Exception as exc:
    st.error(f"Could not load real weather data: {exc}")
    st.info("Download the original dataset from the source link in README.md and place it at data/sri_lanka_weather_2020_2025.csv. No invented sample data are substituted.")
    st.stop()

with st.spinner("Training Random Forest, Gradient Boosting and Stacking — initial load only…"):
    fitted, metrics, predictions = models_for_dashboard(train, validation, test)

st.sidebar.header("Analysis controls")
split_name = st.sidebar.selectbox("Evaluation period", ["test_2025", "validation_2024"],
                                 format_func=lambda v: "2025 — final unseen test" if v == "test_2025" else "2024 — validation")
selected_metric = st.sidebar.selectbox("Compare by metric", ["f1", "roc_auc", "recall", "precision", "accuracy"],
                                      format_func=lambda s: s.upper() if s == "f1" else s.replace("_", " ").title())
ml = metrics[(metrics["split"] == split_name)].copy()
model_names = [n for n in ml["model"] if not n.startswith("Baseline")]
valid = ml[~ml["model"].str.startswith("Baseline")].sort_values(selected_metric, ascending=False)
leader = valid.iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Real observation rows", f"{len(data):,}")
m2.metric("Sri Lankan districts", str(data.district.nunique()))
m3.metric("Selected period records", f"{len(test if split_name == 'test_2025' else validation):,}")
m4.metric("Top model by " + selected_metric.replace("_", " ").upper(), str(leader["model"]).split(" — ")[0])

summary, comparison, samples, methods = st.tabs([
    "🌏 Sri Lanka overview", "📊 Model comparison", "🔎 Explore test cases", "📚 Method and attribution"
])
with summary:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        district_rain = data.assign(rain_indicator=data.rain_tomorrow).groupby("district")["rain_indicator"].mean().sort_values(ascending=False).reset_index()
        district_rain["rain_pct"] = (100 * district_rain.rain_indicator).round(1)
        fig = px.bar(district_rain, x="district", y="rain_pct", title="Rainy next-day observations by district (%)", labels={"rain_pct":"Rainy days (%)", "district":"District"})
        fig.update_layout(xaxis_tickangle=-50, height=420, margin=dict(t=45,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        yr = data.assign(year=data.target_date.dt.year).groupby("year", as_index=False).agg(observations=("rain_tomorrow", "size"), rainy=("rain_tomorrow", "sum"))
        fig = px.bar(yr, x="year", y="observations", title="Dataset coverage by year", text="observations")
        fig.update_layout(height=420, margin=dict(t=45,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.info("Source measurements are from a public third-party Sri Lankan weather compilation. The dataset publisher does not document upstream station/reanalysis provenance, so these are treated as historical weather records, not verified meteorological-station observations.")

with comparison:
    st.subheader("Three ensemble models compared on the same historical split")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        fig = px.bar(ml, x="model", y=selected_metric, color="model", text=selected_metric,
                     title=f"{selected_metric.replace('_',' ').upper()} — {split_name.replace('_',' ')}")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1.08], showlegend=False, xaxis_title="", xaxis_tickangle=-20, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        view = ml[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].rename(
            columns={"model":"Model", "roc_auc":"ROC-AUC", "f1":"F1"})
        st.markdown("**Measured evaluation metrics**")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Precision and recall refer to the rainy-day class (≥ 1 mm), not the dry class.")
    st.subheader("ROC curves")
    fig = go.Figure()
    period_preds = predictions[predictions["split"].eq(split_name)]
    for model_name, group in period_preds.groupby("model"):
        fp, tp, _ = roc_curve(group.actual, group.probability)
        roc = ml.loc[ml.model.eq(model_name), "roc_auc"].iloc[0]
        fig.add_trace(go.Scatter(x=fp, y=tp, mode="lines", name=f"{model_name.split(' — ')[0]} (AUC {roc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random ranking"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=380, margin=dict(t=25,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Confusion matrix for a selected model")
    selected_name = st.selectbox("Select model", list(ml["model"]), key="conf_model")
    entry = ml.loc[ml.model.eq(selected_name)].iloc[0]
    tn, fp, fn, tp = entry.tn_fp_fn_tp
    cm = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    x=["Predicted dry", "Predicted rainy"], y=["Actual dry", "Actual rainy"])
    fig.update_layout(width=580, height=330, coloraxis_showscale=False, margin=dict(t=20,b=15))
    st.plotly_chart(fig, use_container_width=False)
    st.download_button("Download evaluation metrics", metrics.to_csv(index=False), "ensemble_metrics.csv", "text/csv")

with samples:
    st.subheader("Try a historical case from the held-out test set")
    st.write("The selected row contains weather values **observed the day before** the label date; the outcome is hidden from every training model.")
    cities = sorted(test.district.unique().tolist())
    city = st.selectbox("Sri Lankan district", cities, index=cities.index("Colombo") if "Colombo" in cities else 0)
    district_rows = test[test.district.eq(city)].sort_values("target_date")
    chosen_date = st.select_slider("Choose the date being predicted", options=district_rows.target_date.dt.strftime("%Y-%m-%d").tolist(), value=district_rows.target_date.dt.strftime("%Y-%m-%d").iloc[0])
    record = district_rows.loc[district_rows.target_date.dt.strftime("%Y-%m-%d").eq(chosen_date)].iloc[[0]]
    st.caption(f"Input weather observation: {record.date.iloc[0].date()} • Target date: {chosen_date}")
    st.write(f"Temperature: **{record.temp.iloc[0]:.1f} °C** · Humidity: **{record.humidity.iloc[0]:.1f}%** · Precipitation: **{record.precip.iloc[0]:.1f} mm** · Wind speed: **{record.windspeed.iloc[0]:.1f}** (dataset units)")
    scores = pd.DataFrame([{
        "Technique": name.split(" — ")[0],
        "Predicted rain probability": float(model.predict_proba(record[FEATURES])[0, 1]),
        "Classification": "Rain" if model.predict_proba(record[FEATURES])[0, 1] >= 0.5 else "Dry",
    } for name, model in fitted.items() if not name.startswith("Baseline")])
    st.dataframe(scores.style.format({"Predicted rain probability":"{:.1%}"}), use_container_width=True, hide_index=True)
    st.success(f"Actual outcome: {'RAIN' if int(record.rain_tomorrow.iloc[0]) == 1 else 'DRY'} (next-day rainfall: {record.next_precip.iloc[0]:.2f} mm)")
    st.caption("This is a retrospective demonstration, not a live weather forecast. Probabilities have not been separately calibrated.")

with methods:
    st.markdown("""
**Goal:** Binary classification of tomorrow's rainfall (≥ 1 mm) for a Sri Lankan district using today's observations and rainfall history.

**Inputs:** district, temperature, relative humidity, today's precipitation, wind speed, 3-day and 7-day rainfall totals, number of rainy days in the past week, and cyclical day-of-year features.

**Leakage control:** For date *T*, input features use observations no later than *T*. The label is rainfall at *T+1*. We train on target dates in **2020–2023**, validate on **2024**, and reserve **2025** as the untouched final test.

**Bagging — Random Forest:** Averages votes across decision trees trained on bootstrap samples, reducing variance.

**Boosting — Histogram Gradient Boosting:** Adds trees sequentially to reduce predictive errors.

**Stacking:** Combines three base learners (Random Forest, Gradient Boosting, Logistic Regression); a logistic meta-learner learns from their out-of-fold training predictions. The internal stacking folds are not time-purged, which is a methodology limitation despite the chronological external validation/test.

**Metrics:** Accuracy is overall correctness; Precision is the share of rainy predictions that are actually rainy; Recall is the share of rainy days detected; F1 balances precision and recall; ROC-AUC measures probability ranking across thresholds.

**Limitations:** The public dataset's original measurement method and licensing are not specified by its GitHub author. Verify provenance and reuse permissions for publication. Possible district imbalance, weather uncertainty and temporal drift limit real-world forecasting. Do not use this educational model for safety-critical warnings.

**AI usage declaration:** Generative AI assisted with implementation, documentation, Streamlit layout, and code review. The learner must execute the notebook, verify data and metrics, interpret differences, explain the methods in their own words, and cite this assistance in the submitted material.
""")
    st.markdown(f"**Dataset publisher:** [Lahirulakshan129 / SriLanka-Weather-Dataset]({DATA_URL})  ")
    st.markdown("**Software references:** [scikit-learn ensemble documentation](https://scikit-learn.org/stable/modules/ensemble.html) · [Streamlit docs](https://docs.streamlit.io/)")
    st.caption("For academic demonstration only. © respective dataset and library authors.")
