import unittest

from sklearn.linear_model import Ridge

from eurusd_model import (
    build_feature_frame,
    generate_fallback_data,
    scenario_prediction,
    train_best_model,
)


class EurUsdModelTests(unittest.TestCase):
    def test_build_feature_frame_has_expected_columns(self):
        close = generate_fallback_data(periods=80)
        X, y = build_feature_frame(close)

        self.assertEqual(len(X), len(y))
        self.assertIn("rate_spread", X.columns)
        self.assertIn("eurusd_lag5", X.columns)
        self.assertFalse(X.isna().any().any())

    def test_time_series_cv_score_produces_reasonable_metrics(self):
        close = generate_fallback_data(periods=600)
        X, y = build_feature_frame(close)
        metrics = train_best_model(X, y).metrics

        self.assertGreater(metrics["r2_mean"], 0.5)
        self.assertGreater(metrics["rmse_mean"], 0)

    def test_scenario_prediction_changes_output_with_shock(self):
        close = generate_fallback_data(periods=400)
        X, y = build_feature_frame(close)
        model = Ridge(alpha=1.0).fit(X, y)

        base = X.iloc[-1]
        baseline = scenario_prediction(model, base, {})
        shocked = scenario_prediction(model, base, {"dxy": 10})

        self.assertNotEqual(round(baseline, 8), round(shocked, 8))


if __name__ == "__main__":
    unittest.main()
