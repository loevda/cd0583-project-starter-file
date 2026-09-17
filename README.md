# Dynamic Risk Assessment Pipeline

A config-driven MLOps pipeline that ingests corporate attrition data, trains a logistic-regression model, deploys it to a production directory, exposes diagnostics over a Flask API, and automates the full retrain/redeploy loop on a cron schedule.

The model predicts whether a company has **exited** (churned) based on three numeric features: `lastmonth_activity`, `lastyear_activity`, and `number_of_employees`.

## Getting Started

**Python 3.13** is required (pinned via `.python-version` and `requires-python = ">=3.13"` in `pyproject.toml`).

### Option A — `uv` (recommended)

`uv` reads `pyproject.toml` / `uv.lock` and provisions a fully pinned, reproducible environment including the correct Python version.

```bash
uv sync
uv run python fullprocess.py
```

Or activate the virtualenv directly:

```bash
source .venv/bin/activate
python fullprocess.py
```

### Option B — `pip`

```bash
pip install -r requirements.txt
python fullprocess.py
```

## Configuration

All paths are resolved from `config.json` at module import time. No paths are hard-coded in any script.

| Key | Purpose |
|-----|---------|
| `input_folder_path` | Directory containing raw CSV files to ingest |
| `output_folder_path` | Where merged/deduplicated data is written |
| `test_data_path` | Held-out data used for scoring |
| `output_model_path` | Where trained model artifacts are saved |
| `prod_deployment_path` | Production deployment directory |

## Project Structure

```
├── ingestion.py            # Discover, merge, and deduplicate raw CSVs
├── training.py             # Train logistic regression on ingested data
├── scoring.py              # Evaluate model F1 on test data
├── deployment.py           # Copy artifacts to production directory
├── diagnostics.py          # Predictions, summary stats, timing, dependency checks
├── reporting.py            # Confusion-matrix plot generation
├── fullprocess.py          # Orchestrator: new-data detection → drift → redeploy
├── app.py                  # Flask API (prediction, scoring, stats, diagnostics)
├── wsgi.py                 # WSGI entry point for production serving (Gunicorn/uWSGI)
├── apicalls.py             # HTTP client that calls all API endpoints
├── config.json             # Central path configuration
├── cronjob.txt             # Cron entry for scheduled automation
├── requirements.txt        # Runtime + dev dependencies
├── pyproject.toml          # Project metadata, Python version pin
├── uv.lock                 # Locked dependency resolution
├── sourcedata/             # Production input CSVs
├── practicedata/            # Development/sample input CSVs
├── testdata/               # Held-out test data for scoring
├── models/                 # Trained model artifacts and reports
├── production_deployment/  # Deployed model and metadata (runtime)
├── ingesteddata/           # Merged dataset output (runtime)
└── tests/                  # Pytest suite (executable specification)
```

## Running Individual Components

### Ingestion

Discovers all CSVs in `input_folder_path`, concatenates them, removes duplicate rows, and writes the result.

```bash
python ingestion.py
```

### Training

Trains a `LogisticRegression` on the ingested data (`finaldata.csv`) and saves the model.

```bash
python training.py
```

### Scoring

Loads the trained model, evaluates F1 score against the test dataset, and writes the result.

```bash
python scoring.py
```

### Deployment

Copies the trained model, latest score, and ingested-files record into the production directory.

```bash
python deployment.py
```

### Diagnostics

Measures execution time of ingestion and training, computes summary statistics, checks NA percentages, and reports outdated packages.

```bash
python diagnostics.py
```

### Reporting

Generates a confusion-matrix heatmap from the deployed model and test data.

```bash
python reporting.py
```

### Flask API

Starts the API server on port 8000 with four endpoints:

```bash
python app.py
```

For production serving (behind a proper WSGI server):

```bash
gunicorn wsgi:app
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/prediction` | POST | Accepts JSON with feature arrays, returns model predictions |
| `/scoring` | GET | Runs scoring and returns the F1 score |
| `/summarystats` | GET | Returns mean, median, and mode per numeric column |
| `/diagnostics` | GET | Returns execution timing, NA percentages, and outdated packages |

### API Client

Calls all endpoints and writes combined responses to a file. Requires the API to be running.

```bash
python app.py &
python apicalls.py
```

### Full Process (Orchestrator)

Runs the complete cycle: detect new data → ingest → check drift → retrain/redeploy → report.

```bash
python fullprocess.py
```

## Concepts

### Config-Driven Paths

Every module reads `config.json` at import time. Changing a path in one place redirects the entire pipeline. No script contains hard-coded directory names or input filenames.

### Dynamic File Discovery

Ingestion uses `glob` to find all CSVs in the input directory. Adding or removing files requires no code changes.

### Drift Detection

Drift is defined as: the F1 score of the deployed model on newly ingested data is **lower** than the score recorded in the deployed `latestscore.txt`. Retraining and redeployment only occur when **both** new data and drift are present.

### Read-Only Diagnostics

Running diagnostics never modifies or redeploys the production model. It reads from the deployed artifacts without side effects.

### Short-Circuit Logic

`fullprocess.py` exits early when there is no new data, and guards against a missing `latestscore.txt` on a fresh deployment (no `FileNotFoundError`).

## Artifacts

| File | Location | Produced By | Description |
|------|----------|-------------|-------------|
| `finaldata.csv` | `output_folder_path` | `ingestion.py` | Merged, deduplicated dataset |
| `ingestedfiles.txt` | `output_folder_path` | `ingestion.py` | List of ingested CSV filenames |
| `trainedmodel.pkl` | `output_model_path` | `training.py` | Pickled logistic regression model |
| `latestscore.txt` | `output_model_path` | `scoring.py` | F1 score as a float |
| `confusionmatrix.png` | `output_model_path` | `reporting.py` | Confusion-matrix heatmap |
| `apireturns.txt` | `output_model_path` | `apicalls.py` | Combined JSON from all API endpoints |
| `confusionmatrix2.png` | `output_model_path` | `fullprocess.py` | Post-redeploy confusion matrix |
| `apireturns2.txt` | `output_model_path` | `fullprocess.py` | Post-redeploy API output |
| `trainedmodel.pkl` | `prod_deployment_path` | `deployment.py` | Deployed model copy |
| `latestscore.txt` | `prod_deployment_path` | `deployment.py` | Deployed score copy |
| `ingestedfiles.txt` | `prod_deployment_path` | `deployment.py` | Deployed ingestion record |

## Automation

A cron job runs the full pipeline every 10 minutes. The entry is saved in `cronjob.txt`:

```
*/10 * * * * cd /path/to/project && /path/to/.venv/bin/python fullprocess.py >> /tmp/cd0583.log 2>&1
```

To install:

```bash
crontab cronjob.txt
```

## Testing & CI

The test suite in `tests/` is the executable specification of the pipeline. Tests use isolated temp workspaces via pytest fixtures — they never touch the real project directories.

### Running Tests

```bash
# Full suite
python -m pytest -v

# By component
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_training.py tests/test_scoring.py tests/test_deployment.py -v
python -m pytest tests/test_diagnostics.py -v
python -m pytest tests/test_reporting.py tests/test_api.py -v
python -m pytest tests/test_fullprocess.py -v
```

### CI (GitHub Actions)

Every push triggers a CI workflow (`.github/workflows/ci.yml`) that runs all test groups in parallel on Python 3.13. Each job installs dependencies from `requirements.txt` and runs its assigned test file(s).

