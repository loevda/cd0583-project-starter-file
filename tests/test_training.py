import os
import pickle
import pytest


class TestTraining:
    """training.py must train and persist a model."""

    def test_model_file_created(self, workspace_with_input, reload_module):
        """After training, a .pkl file must exist in output_model_path."""
        mod = reload_module("training")
        mod.train_model()

        model_dir = workspace_with_input["paths"]["model"]
        pkl_files = list(model_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1, "No .pkl file found in output_model_path"

    def test_model_is_pickle_format(self, workspace_with_input, reload_module):
        """The saved file must be a valid pickle."""
        mod = reload_module("training")
        mod.train_model()

        model_dir = workspace_with_input["paths"]["model"]
        pkl_files = list(model_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1

        with open(pkl_files[0], "rb") as f:
            loaded = pickle.load(f)

        assert hasattr(loaded, "predict"), "Loaded object is not a sklearn model"
        assert hasattr(loaded, "fit"), "Loaded object is not a sklearn model"

    def test_model_can_predict(self, workspace_with_input, reload_module):
        """The trained model must produce predictions."""
        import pandas as pd
        import numpy as np

        mod = reload_module("training")
        mod.train_model()

        model_dir = workspace_with_input["paths"]["model"]
        pkl_files = list(model_dir.glob("*.pkl"))

        with open(pkl_files[0], "rb") as f:
            model = pickle.load(f)

        # Predict on the ingested data
        df = pd.read_csv(workspace_with_input["paths"]["output"] / "finaldata.csv")
        feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
        X = df[feature_cols]
        predictions = model.predict(X)
        assert len(predictions) == len(df)
        assert set(predictions).issubset({0, 1})

    def test_model_saved_as_trainedmodel_pkl(self, workspace_with_input, reload_module):
        """Rubric specifies the file should be named trainedmodel.pkl."""
        mod = reload_module("training")
        mod.train_model()

        expected = workspace_with_input["paths"]["model"] / "trainedmodel.pkl"
        assert expected.exists(), "trainedmodel.pkl not found in output_model_path"
