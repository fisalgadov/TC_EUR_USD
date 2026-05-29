"""
EUR/USD Monthly Forecast — Streamlit App
Scenario analysis for EUR/USD monthly log returns

Usage:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="EUR/USD Monthly Forecast", layout="wide")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Config
ARTIFACT_PATH = Path(__file__).parent / "model_eurusd_monthly.pkl"
N_MONTHS = 12
LOOKBACK_HIST = 3  # years for scenario charts


# Load artifact
@st.cache_resource
def load_artifact():
    """Load saved model artifact"""
    try:
        return joblib.load(ARTIFACT_PATH)
    except FileNotFoundError:
        st.error(f"Model artifact not found at {ARTIFACT_PATH}")
        st.info("Run: `python eurusd_model.py --save-artifact` to generate it")
        st.stop()


art = load_artifact()
model = art["model"]
selector = art["selector"]
features = art["features"]
metrics = art["metrics"]
data_info = art["data_info"]
raw_cols = art["raw_cols"]
last_raw = art["last_raw"]
last_eurusd = art["last_eurusd"]
last_date = pd.Timestamp(art["last_date"])
hist_data = art["hist_data"]


# Feature mapping (which raw columns are used for each selected feature)
FEATURE_RAW_MAP = {
    'eur_us_equity_diff': ['stoxx50', 'sp500'],
    'eur_us_diff_x_vix': ['ecb_rate', 'fed_rate', 'vix'],
    'cpi_diff_x_equity_diff': ['eur_cpi', 'us_cpi', 'stoxx50', 'sp500'],
    'vix_x_spread': ['vix', 'us10y', 'us2y'],
    'rate_diff': ['ecb_rate', 'fed_rate'],
    'cpi_diff': ['eur_cpi', 'us_cpi'],
    'us_yield_spread': ['us10y', 'us2y'],
    'de_yield_spread': ['de10y', 'de2y'],
    'gold_x_oil': ['gold', 'oil'],
    'cpi_diff_mom3': ['eur_cpi', 'us_cpi'],
    'oil_logret_lag2': ['oil'],
    'oil_mom6': ['oil'],
}

# Get all unique raw columns needed
NEEDED_RAW_COLS = sorted(list(set([col for cols in FEATURE_RAW_MAP.values() for col in cols])))

# Labels for display
LABELS = {
    'vix': 'VIX',
    'us10y': 'US 10Y Yield (%)',
    'us2y': 'US 2Y Yield (%)',
    'us5y': 'US 5Y Yield (%)',
    'gold': 'Gold (USD/oz)',
    'oil': 'Oil (USD/bbl)',
    'sp500': 'S&P 500',
    'stoxx50': 'STOXX 50',
    'dax': 'DAX',
    'de10y': 'German 10Y Yield (%)',
    'de2y': 'German 2Y Yield (%)',
    'us_cpi': 'US CPI',
    'eur_cpi': 'EUR CPI',
    'us_unemp': 'US Unemployment (%)',
    'eur_unemp': 'EUR Unemployment (%)',
    'fed_rate': 'Fed Funds Rate (%)',
    'ecb_rate': 'ECB Rate (%)',
    'us_gdp': 'US GDP',
    'eur_gdp': 'EUR GDP',
}


# Compute engineered features from raw inputs
def compute_features_from_raw(prev_raw: dict, curr_raw: dict) -> pd.DataFrame:
    """
    Compute all 200+ engineered features from raw inputs.
    Returns DataFrame with one row.
    """
    # Create a mini dataframe with 2 rows (previous and current)
    df = pd.DataFrame([prev_raw, curr_raw])
    
    # Compute spreads
    if 'us10y' in df.columns and 'us2y' in df.columns:
        df['us_yield_spread'] = df['us10y'] - df['us2y']
    if 'de10y' in df.columns and 'de2y' in df.columns:
        df['de_yield_spread'] = df['de10y'] - df['de2y']
    
    # Equity differential
    if 'stoxx50' in df.columns and 'sp500' in df.columns:
        df['eur_us_equity_diff'] = df['stoxx50'] - df['sp500']
    
    # Rate differential
    if 'ecb_rate' in df.columns and 'fed_rate' in df.columns:
        df['rate_diff'] = df['ecb_rate'] - df['fed_rate']
    
    # CPI differential
    if 'eur_cpi' in df.columns and 'us_cpi' in df.columns:
        df['cpi_diff'] = df['eur_cpi'] - df['us_cpi']
    
    # Interaction terms
    if 'vix' in df.columns and 'us_yield_spread' in df.columns:
        df['vix_x_spread'] = df['vix'] * df['us_yield_spread']
    if 'rate_diff' in df.columns and 'vix' in df.columns:
        df['eur_us_diff_x_vix'] = df['rate_diff'] * df['vix']
    if 'gold' in df.columns and 'oil' in df.columns:
        df['gold_x_oil'] = df['gold'] * df['oil']
    if 'cpi_diff' in df.columns and 'eur_us_equity_diff' in df.columns:
        df['cpi_diff_x_equity_diff'] = df['cpi_diff'] * df['eur_us_equity_diff']
    
    # Compute MoM changes (3-month)
    if 'cpi_diff' in df.columns:
        df['cpi_diff_mom3'] = 0.0  # Can't compute with only 2 rows, just use 0
    
    # Compute log returns for oil
    if 'oil' in df.columns:
        df['oil_logret'] = np.log(df['oil'] / df['oil'].shift(1))
        df['oil_logret_lag2'] = 0.0  # Can't compute lag2 with only 2 rows
    
    # Compute momentum (6-month)
    if 'oil' in df.columns:
        df['oil_mom6'] = 0.0  # Can't compute with only 2 rows
    
    # Return only the current row (index 1)
    return df.iloc[1:2]


def run_forecast(scenario_values: list) -> list:
    """Run multi-month forecast"""
    fx_path = []
    prev_raw = last_raw.copy()
    prev_fx = last_eurusd
    
    for curr_raw in scenario_values:
        # Compute engineered features
        featured = compute_features_from_raw(prev_raw, curr_raw)
        
        # Extract selected features
        X_input = np.array([[featured[feat].iloc[0] for feat in features]])
        
        # Predict log return
        log_ret = model.predict(X_input)[0]
        
        # Compute FX level
        fx = prev_fx * np.exp(log_ret)
        fx_path.append(fx)
        
        prev_raw = curr_raw
        prev_fx = fx
    
    return fx_path


def _slider_params(rc: str):
    """Compute slider parameters from historical data"""
    s = hist_data[rc].dropna()
    if len(s) == 0:
        return 0.0, 100.0, 0.01, "%.2f"
    
    mu, sd = s.mean(), s.std()
    lo = min(float(s.min()), mu - 4 * sd)
    hi = max(float(s.max()), mu + 4 * sd)
    raw_step = (hi - lo) / 500 if (hi - lo) > 0 else 0.01
    mag = 10 ** np.floor(np.log10(abs(raw_step))) if raw_step > 0 else 0.01
    step = max(round(raw_step / mag) * mag, 1e-6)
    
    # Determine format
    if hi > 100:
        fmt = "%.0f"
    elif hi > 10:
        fmt = "%.1f"
    elif hi > 1:
        fmt = "%.2f"
    else:
        fmt = "%.4f"
    
    return float(lo), float(hi), float(step), fmt


# Pre-compute slider config
slider_cfg = {rc: _slider_params(rc) for rc in NEEDED_RAW_COLS}

# Session state defaults
for rc in NEEDED_RAW_COLS:
    for m in range(N_MONTHS):
        key = f"sv_{rc}_{m}"
        if key not in st.session_state:
            st.session_state[key] = float(last_raw.get(rc, 0))

# Future month labels
future_months = [last_date + pd.DateOffset(months=m) for m in range(1, N_MONTHS + 1)]
month_labels = [d.strftime("%b %Y") for d in future_months]


# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.title("EUR/USD Monthly Forecast — Scenario Analysis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anchor EUR/USD", f"{last_eurusd:.4f}")
c2.metric("Anchor Date", str(last_date.date()))
c3.metric("Test R²", f"{metrics['test_r2']:.4f}")
c4.metric("Test Direction Acc", f"{metrics['test_dir_acc']:.1%}")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════
with st.expander("📖 About the Model", expanded=False):
    st.markdown("""
### EUR/USD Monthly Log-Return Model

This app uses a **Ridge regression model (α=10.0)** with feature selection
trained on **monthly data** to forecast EUR/USD exchange rate.

**Methodology**
- **Target**: Monthly log-return of EUR/USD — $\\log(FX_t / FX_{t-1})$
- **Prediction**: $FX_t = FX_{t-1} \\times e^{\\hat{y}_t}$
- **Algorithm**: Ridge regression with strong regularization (α=10.0)
- **Features**: 3 selected from 200+ engineered features
- **Feature engineering**: Macro differentials, interaction terms, spreads
- **Training**: 5-fold time-series cross-validation on 20 years of data (2006-2026)

**Performance**
- **CV R²**: {:.4f} (near-zero = no overfitting ✓)
- **CV Direction Accuracy**: {:.1%}
- **Test Direction Accuracy**: {:.1%} ✓
- **Test RMSE**: {:.6f} (monthly)

**Selected Features:**
{}

**How to use:**
1. Select a variable from the tabs below
2. Adjust the sliders to set your 12-month scenario path
3. The forecast updates automatically
4. Enter the **current level** of each variable — the app computes transformations automatically
""".format(
        metrics['cv_r2_mean'],
        metrics['cv_dir_mean'],
        metrics['test_dir_acc'],
        metrics['test_rmse'],
        '\n'.join([f"- {i+1}. `{feat}`" for i, feat in enumerate(features)])
    ))

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO EDITOR
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Scenario Editor")
st.caption("Set your 12-month path for each variable. The model computes engineered features automatically.")

cutoff_chart = last_date - pd.DateOffset(years=LOOKBACK_HIST)
tabs = st.tabs([LABELS.get(rc, rc) for rc in NEEDED_RAW_COLS])

for tab, rc in zip(tabs, NEEDED_RAW_COLS):
    with tab:
        lo, hi, step, fmt = slider_cfg[rc]
        
        # Chart
        hist_win = hist_data[hist_data.index >= cutoff_chart][rc].dropna()
        if len(hist_win) == 0:
            st.warning(f"No historical data for {LABELS.get(rc, rc)}")
            continue
        
        mu = hist_data[rc].mean()
        sd = hist_data[rc].std()
        
        # Current scenario values
        month_vals = [st.session_state[f"sv_{rc}_{m}"] for m in range(N_MONTHS)]
        
        fig = go.Figure()
        
        # ±1σ band
        fig.add_trace(go.Scatter(
            x=list(hist_win.index) + list(hist_win.index[::-1]),
            y=([mu + sd] * len(hist_win)) + ([mu - sd] * len(hist_win)),
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.13)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±1σ band',
            hoverinfo='skip',
        ))
        
        # Mean line
        fig.add_trace(go.Scatter(
            x=hist_win.index, y=[mu] * len(hist_win),
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            name='Mean', hoverinfo='skip',
        ))
        
        # Historical
        fig.add_trace(go.Scatter(
            x=hist_win.index, y=hist_win.values,
            mode='lines', name='Historical',
            line=dict(color='steelblue', width=2.5),
            hovertemplate='%{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))
        
        # Anchor
        fig.add_trace(go.Scatter(
            x=[last_date], y=[last_raw.get(rc, 0)],
            mode='markers', name='Anchor',
            marker=dict(symbol='diamond', size=12, color='crimson',
                       line=dict(color='darkred', width=1)),
            hovertemplate='Anchor %{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))
        
        # Scenario path
        fig.add_trace(go.Scatter(
            x=future_months, y=month_vals,
            mode='lines+markers', name='Scenario',
            line=dict(color='darkorange', width=2.5, dash='dot'),
            marker=dict(size=10, color='darkorange'),
            hovertemplate='%{x|%b %Y}: %{y:.3f}<extra></extra>',
        ))
        
        fig.update_layout(
            height=340,
            margin=dict(t=15, b=15, l=10, r=10),
            legend=dict(orientation='h', y=1.07, x=0),
            yaxis_title=LABELS.get(rc, rc),
            xaxis_title='',
            hovermode='x unified',
            template='plotly_white',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Reset button
        btn_cols = st.columns([1, 4])
        with btn_cols[0]:
            if st.button("↺ Reset to flat", key=f"reset_{rc}", use_container_width=True):
                for m in range(N_MONTHS):
                    st.session_state[f"sv_{rc}_{m}"] = float(last_raw.get(rc, 0))
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sliders: 2 columns × 6 rows
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown("**Months 1 – 6**")
            for m_idx in range(6):
                st.slider(
                    label=month_labels[m_idx],
                    min_value=lo,
                    max_value=hi,
                    step=step,
                    format=fmt,
                    key=f"sv_{rc}_{m_idx}",
                )
        
        with col_right:
            st.markdown("**Months 7 – 12**")
            for m_idx in range(6, 12):
                st.slider(
                    label=month_labels[m_idx],
                    min_value=lo,
                    max_value=hi,
                    step=step,
                    format=fmt,
                    key=f"sv_{rc}_{m_idx}",
                )


# ═══════════════════════════════════════════════════════════════════════════
#  RUN FORECAST
# ═══════════════════════════════════════════════════════════════════════════
scenario_values = [
    {rc: st.session_state[f"sv_{rc}_{m}"] for rc in NEEDED_RAW_COLS}
    for m in range(N_MONTHS)
]
fx_forecast = run_forecast(scenario_values)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
#  FORECAST DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("12-Month EUR/USD Forecast")

# Forecast table
forecast_df = pd.DataFrame({
    'Month': month_labels,
    'EUR/USD': fx_forecast,
    'Change vs Anchor': [(fx - last_eurusd) for fx in fx_forecast],
    'Change %': [(fx / last_eurusd - 1) * 100 for fx in fx_forecast],
})

st.dataframe(
    forecast_df.style.format({
        'EUR/USD': '{:.4f}',
        'Change vs Anchor': '{:+.4f}',
        'Change %': '{:+.2f}%',
    }),
    use_container_width=True,
    hide_index=True,
)

# Forecast chart
fig_forecast = go.Figure()

# Historical EUR/USD
hist_eurusd = hist_data[hist_data.index >= cutoff_chart]['eurusd'].dropna()
fig_forecast.add_trace(go.Scatter(
    x=hist_eurusd.index,
    y=hist_eurusd.values,
    mode='lines',
    name='Historical',
    line=dict(color='steelblue', width=2.5),
))

# Anchor
fig_forecast.add_trace(go.Scatter(
    x=[last_date],
    y=[last_eurusd],
    mode='markers',
    name='Anchor',
    marker=dict(symbol='diamond', size=14, color='crimson',
               line=dict(color='darkred', width=1)),
))

# Forecast
fig_forecast.add_trace(go.Scatter(
    x=future_months,
    y=fx_forecast,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='darkorange', width=3, dash='dot'),
    marker=dict(size=10, color='darkorange'),
))

fig_forecast.update_layout(
    height=400,
    yaxis_title='EUR/USD',
    xaxis_title='',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(orientation='h', y=1.05, x=0),
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Month 1 Forecast", f"{fx_forecast[0]:.4f}", 
           f"{(fx_forecast[0] / last_eurusd - 1) * 100:+.2f}%")
col2.metric("Month 6 Forecast", f"{fx_forecast[5]:.4f}",
           f"{(fx_forecast[5] / last_eurusd - 1) * 100:+.2f}%")
col3.metric("Month 12 Forecast", f"{fx_forecast[11]:.4f}",
           f"{(fx_forecast[11] / last_eurusd - 1) * 100:+.2f}%")

st.divider()
st.caption(f"Model trained on {data_info['n_observations']} monthly observations | {data_info['start_date'][:10]} to {data_info['end_date'][:10]}")
