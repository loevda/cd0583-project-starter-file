import pandas as pd
import numpy as np
import pickle
import subprocess
import time
import os
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
    """Check installed vs latest versions of packages from requirements.txt."""
    result = subprocess.run(
        ["pip", "list", "--outdated"],
        capture_output=True, text=True
    )
    output = result.stdout
    # If no outdated packages found, fall back to full package list
    if not output or len(output.strip().splitlines()) <= 1:
        result = subprocess.run(
            ["pip", "list"],
            capture_output=True, text=True
        )
        output = result.stdout
    return output


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
