from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



################## Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


#################### Function for deployment
def store_model_into_pickle(model):
    # Copy the latest pickle file, the latestscore.txt value, 
    # and the ingestedfiles.txt file into the deployment directory
    import shutil
    
    os.makedirs(prod_deployment_path, exist_ok=True)
    
    # Copy trained model from model_path to prod
    src_model = os.path.join(model_path, "trainedmodel.pkl")
    if os.path.exists(src_model):
        shutil.copy2(src_model, os.path.join(prod_deployment_path, "trainedmodel.pkl"))
    
    # Copy latest score from model_path to prod
    src_score = os.path.join(model_path, "latestscore.txt")
    if os.path.exists(src_score):
        shutil.copy2(src_score, os.path.join(prod_deployment_path, "latestscore.txt"))
    
    # Copy ingested files record from output dir to prod
    src_ingested = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    if os.path.exists(src_ingested):
        shutil.copy2(src_ingested, os.path.join(prod_deployment_path, "ingestedfiles.txt"))


if __name__ == '__main__':
    store_model_into_pickle(None)
        

