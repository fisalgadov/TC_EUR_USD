# TC_EUR_USD

EURUSD regression modeling project with an interactive scenario app.

## What is included
- `eurusd_model.py`: downloads market/macroeconomic proxies (EURUSD, VIX, DXY, US rates, gold), engineers features, runs time-series cross-validation, and trains the best regression model.
- `app.py`: Streamlit scenario app to shock explanatory variables and inspect the EURUSD prediction impact.
- `tests/test_eurusd_model.py`: focused regression-model tests for feature engineering, CV metrics, and scenario predictions.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and view model quality
```bash
python eurusd_model.py --start 2012-01-01
```

This prints the selected model and time-series CV metrics (mean R² and RMSE).

## Run tests
```bash
python -m unittest discover -s tests
```

## Run scenario app
```bash
streamlit run app.py
```

Use the sliders to move explanatory variables (shock %), then compare baseline and scenario EURUSD predictions.

> If live Yahoo Finance access is blocked in your environment, the app/model automatically fall back to synthetic data so the workflow remains runnable.
