import os
import pickle
import pytest


class TestScoring:
    """scoring.py must compute F1 and persist it."""

    def test_latestscore_txt_created(self, workspace_with_trained_model, reload_module):
        """After scoring, latestscore.txt must exist."""
        mod = reload_module("scoring")
        mod.score_model()

        # Check in output_model_path (where scoring writes it)
        score_file = workspace_with_trained_model["paths"]["model"] / "latestscore.txt"
        assert score_file.exists(), "latestscore.txt not found in output_model_path"

    def test_latestscore_contains_numeric(self, workspace_with_trained_model, reload_module):
        """latestscore.txt must contain a parseable number."""
        mod = reload_module("scoring")
        mod.score_model()

        score_file = workspace_with_trained_model["paths"]["model"] / "latestscore.txt"
        content = score_file.read_text().strip()
        score = float(content)
        assert 0.0 <= score <= 1.0, f"F1 score out of range: {score}"

    def test_scoring_uses_f1_metric(self, workspace_with_trained_model, reload_module):
        """Verify the score matches a manually computed F1."""
        import pandas as pd
        from sklearn.metrics import f1_score

        mod = reload_module("scoring")
        mod.score_model()

        # Load model and test data manually
        model_file = workspace_with_trained_model["paths"]["prod"] / "trainedmodel.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(
            workspace_with_trained_model["paths"]["test_data"] / "testdata.csv"
        )
        feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
        X_test = test_df[feature_cols]
        y_test = test_df["exited"]

        predictions = model.predict(X_test)
        expected_f1 = f1_score(y_test, predictions)

        score_file = workspace_with_trained_model["paths"]["model"] / "latestscore.txt"
        actual_score = float(score_file.read_text().strip())

        assert abs(actual_score - expected_f1) < 1e-6, (
            f"Score mismatch: got {actual_score}, expected F1={expected_f1}"
        )
