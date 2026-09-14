import os
import pickle
import pytest


class TestDeployment:
    """deployment.py must copy model, score, and file record to prod."""

    def test_model_copied_to_prod(self, workspace_with_trained_model, reload_module):
        """trainedmodel.pkl must appear in prod_deployment_path after deployment."""
        # Remove existing prod model to verify it gets copied
        prod_model = workspace_with_trained_model["paths"]["prod"] / "trainedmodel.pkl"
        if prod_model.exists():
            prod_model.unlink()

        mod = reload_module("deployment")
        mod.store_model_into_pickle(None)  # model arg may not be needed if it reads from disk

        assert prod_model.exists(), "trainedmodel.pkl not found in prod_deployment_path"

    def test_score_copied_to_prod(self, workspace_with_trained_model, reload_module):
        """latestscore.txt must appear in prod_deployment_path."""
        # Remove existing to verify copy
        prod_score = workspace_with_trained_model["paths"]["prod"] / "latestscore.txt"
        if prod_score.exists():
            prod_score.unlink()

        # Ensure source exists in model dir
        src_score = workspace_with_trained_model["paths"]["model"] / "latestscore.txt"
        src_score.write_text("0.92")

        mod = reload_module("deployment")
        mod.store_model_into_pickle(None)

        assert prod_score.exists(), "latestscore.txt not found in prod_deployment_path"
        assert prod_score.read_text().strip() == "0.92"

    def test_ingestedfiles_copied_to_prod(self, workspace_with_trained_model, reload_module):
        """ingestedfiles.txt must appear in prod_deployment_path."""
        prod_record = workspace_with_trained_model["paths"]["prod"] / "ingestedfiles.txt"
        if prod_record.exists():
            prod_record.unlink()

        # Ensure source exists in output dir
        src_record = workspace_with_trained_model["paths"]["output"] / "ingestedfiles.txt"
        src_record.write_text(str(["dataset1.csv", "dataset2.csv"]))

        mod = reload_module("deployment")
        mod.store_model_into_pickle(None)

        assert prod_record.exists(), "ingestedfiles.txt not found in prod_deployment_path"

    def test_deployed_model_is_valid_pickle(self, workspace_with_trained_model, reload_module):
        """The deployed model must be loadable and functional."""
        mod = reload_module("deployment")
        mod.store_model_into_pickle(None)

        prod_model = workspace_with_trained_model["paths"]["prod"] / "trainedmodel.pkl"
        with open(prod_model, "rb") as f:
            model = pickle.load(f)

        assert hasattr(model, "predict")
