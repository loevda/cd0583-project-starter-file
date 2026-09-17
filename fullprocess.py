import os
import sys
import json
import glob
import ast
import shutil
import importlib
import subprocess

import ingestion
import training
import scoring
import deployment
import reporting

# Absolute directory of this script, used to launch sibling scripts (apicalls.py)
# regardless of the current working directory.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

################## Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']


def _read_ingested_files():
    """Read the deployed ``ingestedfiles.txt`` as a list of filenames.

    Returns an empty list when the file is missing or empty (a fresh deployment),
    so the no-new-data path can short-circuit cleanly instead of raising.
    """
    ingested_file = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    if not os.path.exists(ingested_file):
        return []
    with open(ingested_file, "r") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        return [str(name) for name in ast.literal_eval(content)]
    except (ValueError, SyntaxError):
        # Fall back to treating each non-empty line as a filename.
        return [line.strip() for line in content.splitlines() if line.strip()]


def _read_deployed_score():
    """Read the deployed ``latestscore.txt`` as a float.

    Returns ``None`` when the file is missing or unparseable (a fresh deployment),
    so callers can guard instead of raising ``FileNotFoundError``.
    """
    score_file = os.path.join(prod_deployment_path, "latestscore.txt")
    if not os.path.exists(score_file):
        return None
    try:
        with open(score_file, "r") as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return None


def _copy_submission_artifact(src, dst):
    """Copy a freshly generated artifact to its ``_2`` submission name if present."""
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Saved submission artifact: {dst}")


def run():
    """Run the complete model scoring and monitoring process.

    Flow: check for new data -> ingest -> check drift -> retrain/redeploy -> report.
    Retraining and redeployment happen only when BOTH new data and model drift exist.
    """
    ################## Check and read new data
    # First, read ingestedfiles.txt from the production deployment directory.
    ingested_files = _read_ingested_files()

    # Second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt.
    input_csvs = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(input_folder_path, "*.csv"))
    ]
    new_files = [f for f in input_csvs if f not in ingested_files]

    ################## Deciding whether to proceed, part 1
    # If we found new data, proceed. Otherwise, end the process here without
    # retraining or redeploying the model.
    if not new_files:
        print("No new data found. Stopping process without retraining or redeploying.")
        return

    print(f"New data detected: {new_files}. Running ingestion.")
    ingestion.merge_multiple_dataframe()

    ################## Checking for model drift
    # Read the deployed model's score, then score the deployed model on the
    # latest ingested data via scoring.py (which writes latestscore.txt).
    #
    # Reload scoring first so it re-reads the same config.json this process is
    # using. fullprocess may be re-imported in isolation (e.g. by the test suite)
    # without its dependencies being reloaded, which would otherwise leave scoring
    # pointing at a stale workspace and writing latestscore.txt to the wrong path.
    deployed_score = _read_deployed_score()
    importlib.reload(scoring)
    scoring.score_model()
    new_score_file = os.path.join(output_model_path, "latestscore.txt")
    with open(new_score_file, "r") as f:
        new_score = float(f.read().strip())

    # Drift = the new F1 score is lower than the deployed score.
    if deployed_score is None:
        drift_detected = True
        print(
            "No deployed score found; treating as drift to establish a baseline "
            f"(new score={new_score})."
        )
    else:
        drift_detected = new_score < deployed_score
        print(
            f"Deployed score={deployed_score}, new score={new_score}, "
            f"drift_detected={drift_detected}."
        )

    ################## Deciding whether to proceed, part 2
    # If we found model drift, proceed. Otherwise, end the process here.
    if not drift_detected:
        print("No model drift detected. Stopping without retraining or redeploying.")
        return

    ################## Re-deployment
    # Evidence of drift: re-run training on the latest ingested data and redeploy.
    print("Model drift detected. Retraining and redeploying the model.")
    training.train_model()
    deployment.store_model_into_pickle(None)

    ################## Diagnostics and reporting
    # Run reporting and the API-call step for the redeployed model, producing the
    # final submission artifacts confusionmatrix2.png and apireturns2.txt.
    print("Running reporting and API calls for the redeployed model.")
    try:
        reporting.score_model()
        _copy_submission_artifact(
            os.path.join(output_model_path, "confusionmatrix.png"),
            os.path.join(output_model_path, "confusionmatrix2.png"),
        )
    except Exception as exc:  # noqa: BLE001 - reporting must not crash the pipeline
        print(f"Reporting step failed: {exc}")

    try:
        subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "apicalls.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        _copy_submission_artifact(
            os.path.join(output_model_path, "apireturns.txt"),
            os.path.join(output_model_path, "apireturns2.txt"),
        )
    except Exception as exc:  # noqa: BLE001 - the API may not be running
        print(f"API-call step failed: {exc}")


if __name__ == "__main__":
    run()
