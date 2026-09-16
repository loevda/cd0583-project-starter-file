import requests
import json
import os

# Load config to get the output path
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

# Specify the API base URL
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response_prediction = requests.post(
    f"{URL}/prediction",
    json={
        "lastmonth_activity": [100, 200],
        "lastyear_activity": [1000, 2000],
        "number_of_employees": [10, 20],
    }
)

response_scoring = requests.get(f"{URL}/scoring")
response_summarystats = requests.get(f"{URL}/summarystats")
response_diagnostics = requests.get(f"{URL}/diagnostics")

# Combine all API responses
responses = {
    "prediction": response_prediction.json(),
    "scoring": response_scoring.json(),
    "summarystats": response_summarystats.json(),
    "diagnostics": response_diagnostics.json(),
}

# Write the combined responses to the workspace
os.makedirs(model_path, exist_ok=True)
with open(os.path.join(model_path, "apireturns.txt"), "w") as f:
    json.dump(responses, f, indent=2)

print("API calls complete. Results written to apireturns.txt")
