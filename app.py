"""
EUR/USD Monthly Forecast — Streamlit App
Scenario analysis for EUR/USD monthly returns using raw feature inputs

Usage:
    streamlit run app.py
"""

import streamlit as st
st.set_page_config(page_title="EUR/USD Forecast", layout="wide")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Config
ARTIFACT_PATH = Path(__file__).parent / "model_eurusd_monthly.pkl"
N_MONTHS = 12

# Load artifact
@st.cache_resource
def load_artifact():
    try:
        return joblib.load(ARTIFACT_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model artifact not found at {ARTIFACT_PATH}")
        st.info("Run: `python eurusd_model.py --save-artifact` to generate it")
        st.stop()

art = load_artifact()
model = art["model"]
selector = art["selector"]
features = art["features"]
metrics = art["metrics"]
raw_cols = art["raw_cols"]
last_raw = art["last_raw"]
last_eurusd = art["last_eurusd"]
last_date = pd.Timestamp(art["last_date"])
hist_data = art["hist_data"]

# Labels for display
LABELS = {
    'vix': 'VIX', 'us10y': 'US 10Y (%)', 'us2y': 'US 2Y (%)', 'us5y': 'US 5Y (%)',
    'gold': 'Gold ($/oz)', 'oil': 'Oil ($/bbl)', 'sp500': 'S&P 500', 'stoxx50': 'STOXX 50',
    'dax': 'DAX', 'de10y': 'DE 10Y (%)', 'de2y': 'DE 2Y (%)', 'us_cpi': 'US CPI',
    'eur_cpi': 'EUR CPI', 'us_unemp': 'US Unemp (%)', 'eur_unemp': 'EUR Unemp (%)',
    'fed_rate': 'Fed Rate (%)', 'ecb_rate': 'ECB Rate (%)', 'us_gdp': 'US GDP',
    'eur_gdp': 'EUR GDP', 'us_ppi': 'US PPI',
}

# Feature engineering from raw data (matching training notebook)
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly features from raw data (matches training logic)."""
    feat = df.copy()
    
    # CPI differential and momentum
    if 'eur_cpi' in df.columns and 'us_cpi' in df.columns:
        feat['cpi_diff'] = df['eur_cpi'] - df['us_cpi']
        feat['cpi_diff_mom3'] = feat['cpi_diff'] - feat['cpi_diff'].shift(3)
    
    # Oil features
    if 'oil' in df.columns:
        feat['oil_logret'] = np.log(df['oil'] / df['oil'].shift(1))
        feat['oil_logret_lag2'] = feat['oil_logret'].shift(2)
        feat['oil_mom6'] = df['oil'] - df['oil'].shift(6)
    
    # Other differentials
    if 'ecb_rate' in df.columns and 'fed_rate' in df.columns:
        feat['rate_diff'] = df['ecb_rate'] - df['fed_rate']
    if 'stoxx50' in df.columns and 'sp500' in df.columns:
        feat['eur_us_equity_diff'] = df['stoxx50'] - df['sp500']
    
    # Interactions
    if 'rate_diff' in feat.columns and 'vix' in df.columns:
        feat['eur_us_diff_x_vix'] = feat['rate_diff'] * df['vix']
    if 'cpi_diff' in feat.columns and 'eur_us_equity_diff' in feat.columns:
        feat['cpi_diff_x_equity_diff'] = feat['cpi_diff'] * feat['eur_us_equity_diff']
    
    return feat

# Forecast function
def run_forecast(scenario_values: list) -> list:
    """Multi-month forecast using historical + scenario data."""
    forecast_df = hist_data[raw_cols].copy()
    fx_path = []
    prev_fx = last_eurusd
    
    for month_idx, curr_raw in enumerate(scenario_values):
        # Add scenario month
        new_date = last_date + pd.DateOffset(months=month_idx+1)
        new_row = pd.DataFrame([curr_raw], index=[new_date])
        forecast_df = pd.concat([forecast_df, new_row])
        
        # Compute features
        features_df = create_features(forecast_df)
        latest_features = features_df[features].iloc[-1:].values
        latest_features = np.nan_to_num(latest_features, nan=0.0)
        
        # Predict
        log_ret = model.predict(latest_features)[0]
        fx = prev_fx * np.exp(log_ret)
        fx_path.append(fx)
        prev_fx = fx
    
    return fx_path

# Slider params
def _slider_params(rc: str):
    s = hist_data[rc].dropna()
    if len(s) == 0:
        return 0.0, 100.0, 0.01, "%.2f"
    mu, sd = s.mean(), s.std()
    lo = min(float(s.min()), mu - 4 * sd)
    hi = max(float(s.max()), mu + 4 * sd)
    step = max((hi - lo) / 500, 1e-6)
    fmt = "%.0f" if hi > 50 else ("%.2f" if hi > 1 else "%.4f")
    return lo, hi, step, fmt

slider_cfg = {rc: _slider_params(rc) for rc in raw_cols}

# Session state initialization
for rc in raw_cols:
    for m in range(1, N_MONTHS + 1):
        key = f"{rc}_m{m}"
        if key not in st.session_state:
            st.session_state[key] = last_raw.get(rc, 0.0)

# UI
st.title("🌍 EUR/USD Monthly Forecast")
st.markdown(f"**Model:** Ridge(α=10) | **Features:** {len(features)} | **CV R²:** {metrics['cv_r2_mean']:.3f} | "
            f"**Dir Acc:** {metrics.get('cv_dir_mean', 0):.1%}")

# Tabs
tab_scenario, tab_forecast, tab_info = st.tabs(["📊 Scenario Editor", "📈 Forecast", "ℹ️ Info"])

with tab_scenario:
    st.subheader("Define Monthly Scenarios (12 months)")
    
    for rc in raw_cols:
        with st.expander(f"**{LABELS.get(rc, rc)}**", expanded=False):
            lo, hi, step, fmt = slider_cfg[rc]
            
            # Historical chart
            hist = hist_data[rc].dropna().iloc[-36:]
            mu, sd = hist.mean(), hist.std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name='Historical', mode='lines'))
            fig.add_hline(y=mu, line_dash="dash", annotation_text="Mean", line_color="gray")
            fig.add_hrect(y0=mu-sd, y1=mu+sd, fillcolor="lightblue", opacity=0.2, annotation_text="±1σ")
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Sliders
            cols = st.columns(2)
            for m in range(1, N_MONTHS + 1):
                col = cols[(m-1) % 2]
                key = f"{rc}_m{m}"
                val = col.slider(
                    f"Month {m}",
                    min_value=lo,
                    max_value=hi,
                    value=st.session_state[key],
                    step=step,
                    format=fmt,
                    key=key
                )

with tab_forecast:
    st.subheader("12-Month EUR/USD Forecast")
    
    # Gather scenario values
    scenario_vals = []
    for m in range(1, N_MONTHS + 1):
        row = {rc: st.session_state[f"{rc}_m{m}"] for rc in raw_cols}
        scenario_vals.append(row)
    
    # Run forecast
    fx_pred = run_forecast(scenario_vals)
    
    # Table
    forecast_df = pd.DataFrame({
        'Month': range(1, N_MONTHS + 1),
        'EUR/USD': fx_pred,
        'Change (%)': [0] + [(fx_pred[i] / fx_pred[i-1] - 1) * 100 for i in range(1, N_MONTHS)]
    })
    st.dataframe(forecast_df.style.format({'EUR/USD': '{:.4f}', 'Change (%)': '{:+.2f}'}), use_container_width=True)
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, N_MONTHS + 1)),
        y=fx_pred,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_hline(y=last_eurusd, line_dash="dash", annotation_text=f"Current: {last_eurusd:.4f}", line_color="gray")
    fig.update_layout(
        title="EUR/USD Forecast Path",
        xaxis_title="Month",
        yaxis_title="EUR/USD",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_info:
    st.subheader("Model Information")
    st.markdown(f"""
    **Selected Features ({len(features)}):**  
    {', '.join(f'`{f}`' for f in features)}
    
    **Performance Metrics:**
    - CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}
    - Test R²: {metrics.get('test_r2', 0):.4f}
    - Direction Accuracy: {metrics.get('test_dir_acc', 0):.1%}
    
    **Data:**
    - Historical observations: {len(hist_data)}
    - Last EUR/USD: {last_eurusd:.4f} ({last_date.strftime('%Y-%m-%d')})
    
    **Raw Features ({len(raw_cols)}):**  
    {', '.join(f'{LABELS.get(rc, rc)}' for rc in raw_cols)}
    """)
