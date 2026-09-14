from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



################# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


################# Function for model scoring
def score_model():
    # Load the trained model
    with open(os.path.join(model_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)
    
    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    # Define features and target
    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_test = test_data[feature_cols]
    y_test = test_data["exited"]
    
    # Make predictions and calculate F1 score
    predictions = model.predict(X_test)
    f1 = metrics.f1_score(y_test, predictions)
    
    # Write the result to the latestscore.txt file
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "latestscore.txt"), "w") as f:
        f.write(str(f1))


if __name__ == '__main__':
    score_model()

