import os
import pytest


class TestConfusionMatrix:
    """reporting.py must produce confusionmatrix.png."""

    def test_confusion_matrix_created(self, workspace_with_trained_model, reload_module):
        """confusionmatrix.png must exist after running reporting."""
        mod = reload_module("reporting")
        mod.score_model()

        # Check in output_model_path
        png_file = workspace_with_trained_model["paths"]["model"] / "confusionmatrix.png"
        assert png_file.exists(), "confusionmatrix.png not found in output_model_path"

    def test_confusion_matrix_is_valid_png(self, workspace_with_trained_model, reload_module):
        """The output file must be a valid PNG image."""
        mod = reload_module("reporting")
        mod.score_model()

        png_file = workspace_with_trained_model["paths"]["model"] / "confusionmatrix.png"
        # PNG magic bytes
        with open(png_file, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG", "File is not a valid PNG"

    def test_confusion_matrix_not_empty(self, workspace_with_trained_model, reload_module):
        """The PNG must have meaningful content (not a blank file)."""
        mod = reload_module("reporting")
        mod.score_model()

        png_file = workspace_with_trained_model["paths"]["model"] / "confusionmatrix.png"
        size = png_file.stat().st_size
        assert size > 1000, f"PNG file too small ({size} bytes), likely empty"
