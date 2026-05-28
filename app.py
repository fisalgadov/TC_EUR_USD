import pandas as pd
import streamlit as st

from eurusd_model import BASE_FEATURES, FEATURE_COLUMNS, scenario_prediction, train_pipeline


st.set_page_config(page_title="EURUSD Scenario Regressor", layout="wide")
st.title("EURUSD Regression Scenario App")


@st.cache_resource
def load_model():
    return train_pipeline(start="2012-01-01")


artifacts = load_model()
latest_x = artifacts.X.iloc[-1].copy()
latest_actual = float(artifacts.y.iloc[-1])
baseline_pred = float(artifacts.model.predict(pd.DataFrame([latest_x]))[0])

if artifacts.data_source == "synthetic_fallback":
    st.warning("Live market download was unavailable; using synthetic fallback data for this session.")

st.subheader("Model quality (time-series cross validation)")
col1, col2, col3 = st.columns(3)
col1.metric("Selected model", artifacts.model_name)
col2.metric("Mean R²", f"{artifacts.metrics['r2_mean']:.4f}")
col3.metric("Mean RMSE", f"{artifacts.metrics['rmse_mean']:.6f}")

st.write("Features used:", ", ".join(FEATURE_COLUMNS))

st.subheader("Scenario builder")
st.write("Move feature shocks (%) to simulate market scenarios.")

shocks = {}
left, right = st.columns(2)
for i, feature in enumerate(BASE_FEATURES):
    target_col = left if i % 2 == 0 else right
    with target_col:
        shocks[feature] = st.slider(f"{feature} shock (%)", min_value=-30, max_value=30, value=0)

scenario_pred = scenario_prediction(artifacts.model, latest_x, shocks)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Latest EURUSD", f"{latest_actual:.6f}")
col_b.metric("Baseline prediction", f"{baseline_pred:.6f}")
col_c.metric(
    "Scenario prediction",
    f"{scenario_pred:.6f}",
    delta=f"{scenario_pred - baseline_pred:+.6f}",
)

st.caption("App retrains on startup and prefers Yahoo Finance data when available.")
