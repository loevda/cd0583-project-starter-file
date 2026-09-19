import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


############## Function for reporting
def score_model(output_name="confusionmatrix.png"):
    """Generate a confusion matrix plot using the deployed model and test data.

    Loads the deployed model from prod_deployment_path, predicts on test data,
    and saves the confusion matrix as output_name in output_model_path.
    """
    # Load the deployed model
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    # Define features and target
    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X_test = test_data[feature_cols]
    y_test = test_data["exited"]

    # Make predictions
    predictions = model.predict(X_test)

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_test, predictions)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Exited'],
                yticklabels=['Stayed', 'Exited'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    # Save the plot
    os.makedirs(model_path, exist_ok=True)
    plt.savefig(os.path.join(model_path, output_name), dpi=100)
    plt.close()


if __name__ == '__main__':
    score_model()
