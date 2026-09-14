import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

################### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


################# Function for training the model
def train_model():
    
    finaldata_path = os.path.join(dataset_csv_path, "finaldata.csv")
    if not os.path.exists(finaldata_path):
        raise FileNotFoundError(
            f"finaldata.csv not found at '{finaldata_path}'. "
            "Run ingestion (merge_multiple_dataframe) before training."
        )
    
    # Read the ingested data
    data = pd.read_csv(finaldata_path)
    
    # Define features and target
    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = data[feature_cols]
    y = data["exited"]
    
    # Use this logistic regression for training
    clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       n_jobs=None, penalty='l2', random_state=0, 
                       solver='liblinear', tol=0.0001, verbose=0, 
                       warm_start=False)
    
    # Fit the logistic regression to the data
    clf.fit(X, y)
    
    # Write the trained model to your workspace in a file called trainedmodel.pkl
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "trainedmodel.pkl"), "wb") as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    train_model()

