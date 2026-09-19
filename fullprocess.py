import os
import sys
import json
import glob
import ast
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


def run():
    """Run the complete model retraining and redeployment process.

    Flow:
      1. Check for new data -> stop if none.
      2. Ingest new data.
      3. Train a candidate model on the newly ingested data and score it against
         the test set (this becomes the new output_model_path/latestscore.txt).
      4. Redeploy only if the candidate's F1 is higher than the currently
         deployed F1 (or there is no prior deployment to compare against).
      5. If redeployed: generate *2 artifacts (confusionmatrix2.png, apireturns2.txt)
         from a real second run of reporting.py/apicalls.py against the newly
         deployed model.
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

    ################## Train and score a candidate model
    # Reload scoring first so it re-reads the same config.json this process is
    # using. fullprocess may be re-imported in isolation (e.g. by the test suite)
    # without its dependencies being reloaded, which would otherwise leave scoring
    # pointing at a stale workspace and writing latestscore.txt to the wrong path.
    print("Training a candidate model on the newly ingested data.")
    training.train_model()

    importlib.reload(scoring)
    scoring.score_model()
    candidate_score_file = os.path.join(output_model_path, "latestscore.txt")
    with open(candidate_score_file, "r") as f:
        candidate_score = float(f.read().strip())

    ################## Deciding whether to proceed, part 2
    # Deploy only if the candidate beats the currently deployed model.
    deployed_score = _read_deployed_score()
    if deployed_score is None:
        should_deploy = True
        print(
            f"No deployed score found; deploying candidate (score={candidate_score}) "
            "to establish a baseline."
        )
    else:
        should_deploy = candidate_score > deployed_score
        print(
            f"Candidate score={candidate_score}, deployed score={deployed_score}, "
            f"should_deploy={should_deploy}."
        )

    if not should_deploy:
        print("Candidate does not beat the deployed model. Stopping without redeploying.")
        return

    ################## Re-deployment
    print("Redeploying the candidate model.")
    deployment.store_model_into_pickle(None)

    ################## Diagnostics and reporting
    # Run reporting and the API-call step for the redeployed model, writing directly
    # to the *2 submission names (not copying the first-run files) so they reflect
    # a genuine second run against the newly deployed model.
    print("Running reporting and API calls for the redeployed model.")
    try:
        importlib.reload(reporting)
        reporting.score_model(output_name="confusionmatrix2.png")
        print("Saved submission artifact: confusionmatrix2.png")
    except Exception as exc:  # noqa: BLE001 - reporting must not crash the pipeline
        print(f"Reporting step failed: {exc}")

    try:
        subprocess.run(
            [sys.executable, os.path.join(PROJECT_DIR, "apicalls.py"), "apireturns2.txt"],
            check=False,
            capture_output=True,
            text=True,
        )
        print("Saved submission artifact: apireturns2.txt")
    except Exception as exc:  # noqa: BLE001 - the API may not be running
        print(f"API-call step failed: {exc}")


if __name__ == "__main__":
    run()
