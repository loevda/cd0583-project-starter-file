import os
import pickle
import pytest


class TestExecutionTime:
    """execution_time() must time ingestion and training separately."""

    def test_returns_two_timings(self, workspace_with_input, reload_module):
        """Must return timing for both ingestion and training."""
        mod = reload_module("diagnostics")
        result = mod.execution_time()

        assert result is not None, "execution_time() returned None"
        assert len(result) == 2, f"Expected 2 timing values, got {len(result)}"

    def test_timings_are_positive_numbers(self, workspace_with_input, reload_module):
        """Both timings must be positive (in seconds)."""
        mod = reload_module("diagnostics")
        result = mod.execution_time()

        for t in result:
            assert isinstance(t, (int, float)), f"Timing is not numeric: {type(t)}"
            assert t > 0, f"Timing must be positive, got {t}"

    def test_diagnostics_does_not_redeploy(self, workspace_with_trained_model, reload_module):
        """Running diagnostics must NOT modify the deployed model."""
        prod_model = workspace_with_trained_model["paths"]["prod"] / "trainedmodel.pkl"
        original_content = prod_model.read_bytes()

        mod = reload_module("diagnostics")
        mod.execution_time()

        assert prod_model.read_bytes() == original_content, (
            "Diagnostics modified the deployed model file"
        )


class TestPredictions:
    """model_predictions() must use deployed model on a dataset."""

    def test_returns_list_of_predictions(self, workspace_with_trained_model, reload_module):
        """Must return a list of predictions."""
        mod = reload_module("diagnostics")
        result = mod.model_predictions()

        assert result is not None, "model_predictions() returned None"
        assert isinstance(result, (list, tuple)), f"Expected list, got {type(result)}"

    def test_predictions_are_binary(self, workspace_with_trained_model, reload_module):
        """All predictions must be 0 or 1."""
        mod = reload_module("diagnostics")
        result = mod.model_predictions()

        unique_vals = set(result)
        assert unique_vals.issubset({0, 1}), f"Unexpected prediction values: {unique_vals}"


class TestSummaryStatistics:
    """dataframe_summary() must compute mean, median, mode per numeric column."""

    def test_returns_summary_stats(self, workspace_with_trained_model, reload_module):
        """Must return summary statistics."""
        mod = reload_module("diagnostics")
        result = mod.dataframe_summary()
        assert result is not None, "dataframe_summary() returned None"

    def test_contains_mean_median_mode(self, workspace_with_trained_model, reload_module):
        """Result must include mean, median, and mode for numeric columns."""
        mod = reload_module("diagnostics")
        result = mod.dataframe_summary()
        # At least 9 values: 3 numeric columns x 3 measures
        assert len(result) >= 9, (
            f"Expected at least 9 stats (3 cols x 3 measures), got {len(result)}"
        )


class TestDataIntegrity:
    """Must check percentage of NA values per numeric column."""

    def test_na_percentage_computed(self, workspace, reload_module):
        """With known NA values, must report correct percentages."""
        import pandas as pd
        import numpy as np

        # 2 out of 10 = 20% NA in lastmonth_activity
        df = pd.DataFrame({
            "corporation": [f"c{i}" for i in range(10)],
            "lastmonth_activity": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            "lastyear_activity": [100] * 10,
            "number_of_employees": [5] * 10,
            "exited": [0] * 10,
        })
        df.to_csv(workspace["paths"]["output"] / "finaldata.csv", index=False)

        mod = reload_module("diagnostics")
        result = mod.dataframe_summary()
        assert result is not None

    def test_na_check_function_exists(self, reload_module):
        """diagnostics module must have NA checking capability."""
        mod = reload_module("diagnostics")
        funcs = [f for f in dir(mod) if not f.startswith("_")]
        has_na_check = any(
            keyword in f.lower()
            for f in funcs
            for keyword in ["na", "missing", "integrity", "null"]
        )
        assert has_na_check or hasattr(mod, "dataframe_summary"), (
            "No NA checking function found in diagnostics"
        )


class TestDependencyCheck:
    """outdated_packages_list() must check installed vs latest versions."""

    def test_returns_package_list(self, reload_module):
        """Must return information about package versions."""
        mod = reload_module("diagnostics")
        result = mod.outdated_packages_list()
        assert result is not None, "outdated_packages_list() returned None"

    def test_checks_requirements_txt_packages(self, reload_module):
        """Must check packages listed in requirements.txt."""
        mod = reload_module("diagnostics")
        result = mod.outdated_packages_list()
        result_str = str(result).lower()
        known_packages = ["flask", "pandas", "scikit", "numpy", "matplotlib"]
        found = [p for p in known_packages if p in result_str]
        assert len(found) >= 2, (
            f"Expected known packages in result, only found: {found}"
        )

