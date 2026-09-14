"""
Shared fixtures for the CD0583 MLOps project test suite.

Strategy:
- All modules load config.json at import time (top-level code).
- We create an isolated temp workspace with its own config.json,
  then use importlib.reload() to re-import modules against that config.
- Each test gets a clean set of temp directories.
"""

import json
import os
import sys
import importlib
import pytest
import pandas as pd
import numpy as np


# ─── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_COLUMNS = [
    "corporation",
    "lastmonth_activity",
    "lastyear_activity",
    "number_of_employees",
    "exited",
]


def make_sample_df(n=10, seed=42):
    """Create a sample DataFrame matching the project schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "corporation": [f"corp_{i}" for i in range(n)],
            "lastmonth_activity": rng.integers(10, 500, size=n),
            "lastyear_activity": rng.integers(100, 10000, size=n),
            "number_of_employees": rng.integers(1, 200, size=n),
            "exited": rng.integers(0, 2, size=n),
        }
    )


# ─── Workspace Fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """
    Create an isolated temp workspace with all required directories
    and a config.json pointing to them. Changes CWD to tmp_path.
    """
    paths = {
        "input": tmp_path / "input",
        "output": tmp_path / "ingesteddata",
        "test_data": tmp_path / "testdata",
        "model": tmp_path / "practicemodels",
        "prod": tmp_path / "production_deployment",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    config = {
        "input_folder_path": str(paths["input"]),
        "output_folder_path": str(paths["output"]),
        "test_data_path": str(paths["test_data"]),
        "output_model_path": str(paths["model"]),
        "prod_deployment_path": str(paths["prod"]),
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)

    return {
        "paths": paths,
        "config": config,
        "config_path": config_path,
        "tmp_path": tmp_path,
    }


# ─── Module Reload Helper ──────────────────────────────────────────────────────


@pytest.fixture
def reload_module():
    """Reload a module so it picks up the current CWD's config.json."""

    def _reload(module_name):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

    return _reload


# ─── Pre-populated Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def workspace_with_input(workspace):
    """Workspace with two sample CSV files already in the input directory."""
    df1 = make_sample_df(n=10, seed=1)
    df2 = make_sample_df(n=8, seed=2)
    df1.to_csv(workspace["paths"]["input"] / "dataset1.csv", index=False)
    df2.to_csv(workspace["paths"]["input"] / "dataset2.csv", index=False)
    return workspace


@pytest.fixture
def workspace_with_duplicates(workspace):
    """Workspace where input files share duplicate rows."""
    df1 = make_sample_df(n=10, seed=1)
    df2 = pd.concat([df1.iloc[:5], make_sample_df(n=5, seed=99)], ignore_index=True)
    df1.to_csv(workspace["paths"]["input"] / "dataset1.csv", index=False)
    df2.to_csv(workspace["paths"]["input"] / "dataset2.csv", index=False)
    return workspace


@pytest.fixture
def workspace_with_trained_model(workspace_with_input):
    """Workspace with a pre-trained model in model and prod directories."""
    import pickle
    from sklearn.linear_model import LogisticRegression

    df1 = pd.read_csv(workspace_with_input["paths"]["input"] / "dataset1.csv")
    df2 = pd.read_csv(workspace_with_input["paths"]["input"] / "dataset2.csv")
    combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates()

    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = combined[feature_cols]
    y = combined["exited"]

    model = LogisticRegression(C=1.0, max_iter=200, solver="liblinear", random_state=0)
    model.fit(X, y)

    model_file = workspace_with_input["paths"]["model"] / "trainedmodel.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    prod_model_file = workspace_with_input["paths"]["prod"] / "trainedmodel.pkl"
    with open(prod_model_file, "wb") as f:
        pickle.dump(model, f)

    (workspace_with_input["paths"]["prod"] / "latestscore.txt").write_text("0.85")
    (workspace_with_input["paths"]["prod"] / "ingestedfiles.txt").write_text(
        str(["dataset1.csv", "dataset2.csv"])
    )

    test_df = make_sample_df(n=20, seed=77)
    test_df.to_csv(
        workspace_with_input["paths"]["test_data"] / "testdata.csv", index=False
    )

    return workspace_with_input


# ─── Flask App Fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def flask_client(workspace_with_trained_model, reload_module):
    """
    Provide a Flask test client configured against the temp workspace.
    """
    app_module = reload_module("app")
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    return client
