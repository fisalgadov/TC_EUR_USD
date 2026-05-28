from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


TICKERS = {
    "eurusd": "EURUSD=X",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "us2y": "^IRX",
    "gold": "GC=F",
}

BASE_FEATURES = ["vix", "dxy", "us10y", "us2y", "gold"]
FEATURE_COLUMNS = BASE_FEATURES + [
    "rate_spread",
    "vix_chg_5d",
    "dxy_chg_5d",
    "eurusd_lag1",
    "eurusd_lag5",
]


@dataclass
class ModelArtifacts:
    model_name: str
    model: object
    metrics: Dict[str, float]
    X: pd.DataFrame
    y: pd.Series
    data_source: str



def generate_fallback_data(start: str = "2012-01-01", periods: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=periods, freq="B")

    vix = 18 + rng.normal(0, 1.4, periods)
    dxy = 101 + rng.normal(0, 1.1, periods)
    us10y = 2.3 + rng.normal(0, 0.15, periods)
    us2y = 2.0 + rng.normal(0, 0.14, periods)
    gold = 1750 + rng.normal(0, 35, periods)

    eurusd = np.zeros(periods)
    eurusd[0] = 1.12
    for i in range(1, periods):
        structural = (
            1.11
            - 0.0013 * (vix[i] - 18)
            - 0.0024 * (dxy[i] - 101)
            + 0.0100 * (us10y[i] - us2y[i])
            + 0.00003 * (gold[i] - 1750)
        )
        eurusd[i] = 0.7 * eurusd[i - 1] + 0.3 * structural + rng.normal(0, 0.0015)

    return pd.DataFrame(
        {
            "eurusd": eurusd,
            "vix": vix,
            "dxy": dxy,
            "us10y": us10y,
            "us2y": us2y,
            "gold": gold,
        },
        index=dates,
    )



def fetch_market_data(start: str = "2012-01-01") -> Tuple[pd.DataFrame, str]:
    symbols = list(TICKERS.values())
    try:
        raw = yf.download(symbols, start=start, auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("No market data returned from Yahoo Finance.")

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw.copy()

        reverse_map = {v: k for k, v in TICKERS.items()}
        close = close.rename(columns=reverse_map)
        missing = [k for k in TICKERS if k not in close.columns]
        if missing:
            raise RuntimeError(f"Missing downloaded series: {missing}")

        return close[list(TICKERS.keys())].dropna(how="all"), "yahoo_finance"
    except Exception:
        return generate_fallback_data(start=start), "synthetic_fallback"



def build_feature_frame(close: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = close.copy()
    data["rate_spread"] = data["us10y"] - data["us2y"]
    data["vix_chg_5d"] = data["vix"].pct_change(5)
    data["dxy_chg_5d"] = data["dxy"].pct_change(5)
    data["eurusd_lag1"] = data["eurusd"].shift(1)
    data["eurusd_lag5"] = data["eurusd"].shift(5)

    modeled = data.dropna().copy()
    X = modeled[FEATURE_COLUMNS]
    y = modeled["eurusd"]
    return X, y



def time_series_cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_r2 = []
    fold_rmse = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_r2.append(r2_score(y_test, preds))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, preds))))

    return {
        "r2_mean": float(np.mean(fold_r2)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "r2_std": float(np.std(fold_r2)),
        "rmse_std": float(np.std(fold_rmse)),
    }



def train_best_model(X: pd.DataFrame, y: pd.Series) -> ModelArtifacts:
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
        ),
    }

    scored = []
    for name, model in candidates.items():
        metrics = time_series_cv_score(model, X, y, n_splits=5)
        scored.append((name, model, metrics))

    scored.sort(key=lambda item: (item[2]["r2_mean"], -item[2]["rmse_mean"]), reverse=True)
    best_name, best_model, best_metrics = scored[0]
    best_model.fit(X, y)

    return ModelArtifacts(
        model_name=best_name,
        model=best_model,
        metrics=best_metrics,
        X=X,
        y=y,
        data_source="unknown",
    )



def train_pipeline(start: str = "2012-01-01") -> ModelArtifacts:
    close, source = fetch_market_data(start=start)
    X, y = build_feature_frame(close)
    artifacts = train_best_model(X, y)
    artifacts.data_source = source
    return artifacts



def scenario_prediction(model: object, base_features: pd.Series, feature_shocks: Dict[str, float]) -> float:
    scenario = base_features.copy()
    for feature, pct in feature_shocks.items():
        if feature not in scenario.index:
            continue
        scenario[feature] = scenario[feature] * (1 + pct / 100.0)

    return float(model.predict(pd.DataFrame([scenario]))[0])



def main() -> None:
    parser = argparse.ArgumentParser(description="Train EURUSD regression model with time-series CV")
    parser.add_argument("--start", default="2012-01-01", help="Training start date (YYYY-MM-DD)")
    args = parser.parse_args()

    artifacts = train_pipeline(start=args.start)
    latest_x = artifacts.X.iloc[-1]
    baseline_pred = float(artifacts.model.predict(pd.DataFrame([latest_x]))[0])

    print(f"Data source: {artifacts.data_source}")
    print(f"Model selected: {artifacts.model_name}")
    print(
        "Time-series CV metrics: "
        f"R2={artifacts.metrics['r2_mean']:.4f} +/- {artifacts.metrics['r2_std']:.4f}, "
        f"RMSE={artifacts.metrics['rmse_mean']:.6f} +/- {artifacts.metrics['rmse_std']:.6f}"
    )
    print(f"Latest observed EURUSD: {artifacts.y.iloc[-1]:.6f}")
    print(f"Baseline model prediction: {baseline_pred:.6f}")


if __name__ == "__main__":
    main()
