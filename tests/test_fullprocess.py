import os
import pickle
import pytest


class TestNewDataDetection:
    """fullprocess must detect new files not in ingestedfiles.txt."""

    def test_no_new_data_stops_process(self, workspace_with_trained_model, reload_module):
        """If input dir has no new files, process must not retrain."""
        # Record already lists dataset1.csv and dataset2.csv
        # Input dir has exactly those files — no new data
        mod = reload_module("fullprocess")

        # Capture that training is NOT called
        import training
        original_train = training.train_model
        train_called = []
        training.train_model = lambda *a, **kw: train_called.append(True)

        try:
            mod.run()
        except SystemExit:
            pass
        finally:
            training.train_model = original_train

        assert len(train_called) == 0, (
            "Training was called even though no new data exists"
        )

    def test_new_data_triggers_ingestion(self, workspace_with_trained_model, reload_module):
        """If a new CSV appears in input dir, ingestion must run."""
        from tests.conftest import make_sample_df

        # Add a new file not in ingestedfiles.txt
        new_df = make_sample_df(n=5, seed=500)
        new_df.to_csv(
            workspace_with_trained_model["paths"]["input"] / "dataset_new.csv",
            index=False,
        )

        mod = reload_module("fullprocess")

        # Track if ingestion was called
        import ingestion
        original_ingest = ingestion.merge_multiple_dataframe
        ingest_called = []
        ingestion.merge_multiple_dataframe = (
            lambda *a, **kw: (ingest_called.append(True), original_ingest(*a, **kw))
        )

        try:
            mod.run()
        except SystemExit:
            pass
        finally:
            ingestion.merge_multiple_dataframe = original_ingest

        assert len(ingest_called) > 0, (
            "Ingestion was not triggered despite new data being present"
        )


class TestDriftDetection:
    """Model drift = new F1 is lower than deployed score."""

    def test_no_drift_no_retrain(self, workspace_with_trained_model, reload_module):
        """If new score >= deployed score, no retraining should occur."""
        from tests.conftest import make_sample_df

        # Add new data that won't cause drift
        new_df = make_sample_df(n=5, seed=1)  # similar to existing data
        new_df.to_csv(
            workspace_with_trained_model["paths"]["input"] / "dataset_extra.csv",
            index=False,
        )

        # Set deployed score very low so new score won't be lower
        (workspace_with_trained_model["paths"]["prod"] / "latestscore.txt").write_text("0.1")

        mod = reload_module("fullprocess")

        import training
        original_train = training.train_model
        train_called = []
        training.train_model = lambda *a, **kw: train_called.append(True)

        try:
            mod.run()
        except SystemExit:
            pass
        finally:
            training.train_model = original_train

        assert len(train_called) == 0, (
            "Retraining occurred despite no drift (deployed score was already low)"
        )

    def test_drift_triggers_retrain(self, workspace_with_trained_model, reload_module):
        """If new F1 < deployed F1, retraining must occur."""
        from tests.conftest import make_sample_df

        # Add very different new data to cause drift
        new_df = make_sample_df(n=50, seed=999)
        new_df.to_csv(
            workspace_with_trained_model["paths"]["input"] / "dataset_drift.csv",
            index=False,
        )

        # Set deployed score high so new score will likely be lower
        (workspace_with_trained_model["paths"]["prod"] / "latestscore.txt").write_text("0.99")

        mod = reload_module("fullprocess")

        import training
        original_train = training.train_model
        train_called = []
        training.train_model = lambda *a, **kw: train_called.append(True)

        try:
            mod.run()
        except SystemExit:
            pass
        finally:
            training.train_model = original_train

        assert len(train_called) > 0, (
            "Retraining was NOT triggered despite drift (new score < 0.99)"
        )


class TestCronJob:
    """Rubric requires a cronjob.txt with 10-minute schedule."""

    def test_cronjob_file_exists(self):
        """cronjob.txt must be present in the project root."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cron_file = os.path.join(project_root, "cronjob.txt")
        assert os.path.exists(cron_file), "cronjob.txt not found in project root"

    def test_cronjob_runs_every_10_minutes(self):
        """Cron expression must specify every 10 minutes."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cron_file = os.path.join(project_root, "cronjob.txt")
        if not os.path.exists(cron_file):
            pytest.skip("cronjob.txt not yet created")

        content = open(cron_file).read().strip()
        # Should contain */10 in the minute field
        assert "*/10" in content, f"Expected '*/10' in cron, got: {content}"
        assert "fullprocess.py" in content, f"Expected fullprocess.py in cron, got: {content}"
