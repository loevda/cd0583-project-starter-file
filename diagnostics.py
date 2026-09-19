import pandas as pd
import numpy as np
import pickle
import subprocess
import sys
import time
import os
import re
import json

################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

################## Function to get model predictions
def model_predictions():
    """Read the deployed model and test dataset, return a list of binary predictions."""
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = test_data[feature_cols]
    predictions = model.predict(X)
    return list(predictions)

################## Function to get summary statistics
def dataframe_summary():
    """Calculate mean, median, and mode for each numeric column in the ingested dataset."""
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_cols = data.select_dtypes(include=[np.number])

    summary = []
    for col in numeric_cols.columns:
        summary.append(float(numeric_cols[col].mean()))
        summary.append(float(numeric_cols[col].median()))
        mode_val = numeric_cols[col].mode()
        summary.append(float(mode_val.iloc[0]) if len(mode_val) > 0 else 0.0)

    return summary

################## Function to check NA/missing data integrity
def check_na_percentage():
    """Calculate the percentage of NA values per column in the ingested dataset."""
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    na_percentages = []
    for col in data.columns:
        na_pct = float(data[col].isna().sum()) / len(data) * 100.0
        na_percentages.append(na_pct)
    return na_percentages

################## Function to get timings
def execution_time():
    """Time the ingestion and training processes without modifying the deployed model."""
    import ingestion
    import training

    start = time.time()
    ingestion.merge_multiple_dataframe()
    ingest_time = time.time() - start

    start = time.time()
    training.train_model()
    train_time = time.time() - start

    return [ingest_time, train_time]

################## Function to check dependencies
def outdated_packages_list():
    """Check installed vs latest versions of each package listed in requirements.txt."""
    import importlib.metadata

    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    with open(req_path, "r") as f:
        lines = f.readlines()

    # Extract package names (strip version specifiers like ==, >=, ~=, etc.)
    packages = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[=><!~\[;]", line)[0].strip()
        if pkg_name:
            packages.append(pkg_name)

    # Get all outdated packages in one call, against THIS project's own environment.
    # Plain "pip" resolves via PATH and can silently hit an unrelated global/conda
    # install instead of this venv. `python -m pip` is the portable fix and works
    # in any standard venv. It only fails here because
    # this local dev environment is managed by uv, which doesn't install a pip
    # module into .venv -- in that one case, fall back to uv's pip-compatible CLI.
    outdated_map = {}
    for cmd in (
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
        ["uv", "pip", "list", "--outdated", "--format=json"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                continue
            outdated_data = json.loads(result.stdout)
            # Map lowercase name -> latest_version
            outdated_map = {item["name"].lower(): item["latest_version"] for item in outdated_data}
            break
        except Exception:
            continue

    results = []
    for pkg in packages:
        # Get installed version
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"

        # Get latest version: from outdated map, or same as installed if not outdated
        latest = outdated_map.get(pkg.lower(), installed)

        results.append({
            "package": pkg,
            "installed": installed,
            "latest": latest
        })

    return results


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
