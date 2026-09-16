from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import importlib
import json
import os

import diagnostics
import scoring

# Reload modules so they pick up the current CWD's config.json
# (important when app itself is reloaded in tests)
diagnostics = importlib.reload(diagnostics)
scoring = importlib.reload(scoring)


###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################### Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """Accept a JSON body with feature columns as lists, return predictions."""
    data = request.get_json()

    # Build a DataFrame from the JSON payload
    df = pd.DataFrame({
        'lastmonth_activity': data['lastmonth_activity'],
        'lastyear_activity': data['lastyear_activity'],
        'number_of_employees': data['number_of_employees'],
    })

    # Load the deployed model
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Predict
    predictions = model.predict(df)

    return jsonify({"predictions": [int(p) for p in predictions]})


####################### Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def get_score():
    """Run scoring and return the F1 score."""
    scoring.score_model()

    # Read the score that was just written
    score_file = os.path.join(config['output_model_path'], "latestscore.txt")
    with open(score_file, 'r') as f:
        score = float(f.read().strip())

    return jsonify({"score": score})


####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def get_summary_stats():
    """Return mean, median, and mode for each numeric column."""
    summary = diagnostics.dataframe_summary()
    return jsonify(summary)


####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def get_diagnostics():
    """Return execution timing, dependency info, and NA percentages."""
    timing = diagnostics.execution_time()
    packages = diagnostics.outdated_packages_list()
    na_pct = diagnostics.check_na_percentage()

    return jsonify({
        "execution_time": {
            "ingestion_seconds": timing[0],
            "training_seconds": timing[1]
        },
        "outdated_packages": packages,
        "na_percentages": na_pct
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
