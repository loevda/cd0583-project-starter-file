# CD0583 – ML Pipeline Project

A starter project for building an end-to-end machine learning pipeline with monitoring, diagnostics, and a Flask API.

## Overview

You will implement a binary classification pipeline that predicts whether a company has **exited** (churned) based on activity and employee metrics. The pipeline covers the full MLOps lifecycle: data ingestion, model training, scoring, deployment, and ongoing monitoring.

## Project Structure

```
├── ingestion.py        # Step 1: Load and merge raw CSV data
├── training.py         # Step 2: Train a logistic regression model
├── scoring.py          # Step 3: Score the model (F1) against test data
├── deployment.py       # Step 4: Deploy model artifacts to production
├── diagnostics.py      # Step 5: Diagnostics (predictions, stats, timing, deps)
├── reporting.py        # Step 6: Generate confusion matrix report
├── fullprocess.py      # Orchestration: detect new data, check drift, redeploy
├── app.py              # Flask API with prediction/scoring/diagnostics endpoints
├── apicalls.py         # Script to call the API endpoints
├── config.json         # Path configuration
├── requirements.txt    # Python dependencies
├── sourcedata/         # Raw input data (your task data)
├── practicedata/        # Sample data for development/testing
└── testdata/           # Held-out test data for scoring
```

## Tasks

Each script is a **skeleton** — you need to implement the logic:

1. **Ingestion** – Read CSVs from `input_folder_path`, concatenate, deduplicate, write to `output_folder_path`.
2. **Training** – Train a `LogisticRegression` on the ingested data, save as `trainedmodel.pkl`.
3. **Scoring** – Load the model, evaluate on test data, write F1 score to `latestscore.txt`.
4. **Deployment** – Copy model, score, and ingested-files list into the production directory.
5. **Diagnostics** – Return predictions, summary statistics, execution timings, and outdated packages.
6. **Reporting** – Generate and save a confusion matrix plot.
7. **Full Process** – Orchestrate: detect new data → check drift → redeploy if needed → run diagnostics/reporting.
8. **Flask API** – Implement endpoints: `/prediction`, `/scoring`, `/summarystats`, `/diagnostics`.

## Getting Started

> **Runtime:** This project has been developed and run on **Python 3.13** (see `.python-version` and `requires-python = ">=3.13"` in `pyproject.toml`).

You can set up and run the project with either [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`.

### Option A — with `uv` (recommended)

`uv` reads `pyproject.toml` / `uv.lock` and provisions a pinned, reproducible environment (including Python 3.13 if needed).

```bash
# Create/sync the virtual environment from the lockfile
uv sync

# Run any script through the managed environment
uv run ingestion.py

# Run the test suite
uv run pytest -v
```

If you prefer to activate the environment instead of prefixing each command:

```bash
source .venv/bin/activate   # uv creates .venv/ during `uv sync`
python ingestion.py
```

### Option B — with `pip`

```bash
pip install -r requirements.txt
python ingestion.py
```

## Configuration

All paths are defined in `config.json`:

| Key | Purpose |
|-----|---------|
| `input_folder_path` | Where raw CSVs are read from |
| `output_folder_path` | Where ingested/merged data is written |
| `test_data_path` | Held-out data for scoring |
| `output_model_path` | Where trained model artifacts are saved |
| `prod_deployment_path` | Production deployment directory |
