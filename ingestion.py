import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime




############# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



############# Function for data ingestion
def merge_multiple_dataframe():
    # Check for datasets, compile them together, and write to an output file
    csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

    frames = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()

    os.makedirs(output_folder_path, exist_ok=True)
    combined.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    filenames = [os.path.basename(f) for f in csv_files]
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        f.write(str(filenames))



if __name__ == '__main__':
    merge_multiple_dataframe()
