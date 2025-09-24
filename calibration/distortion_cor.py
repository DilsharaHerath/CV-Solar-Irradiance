# Preprocessing Images and Detect Sun Positions
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import io
import textwrap
import csv

image_path = './../dataset/13/20191001130000.jpg'
timeseries_path = './PSA_timeSeries_data.csv'

image1 = cv2.imread(image_path)
df = pd.read_csv(timeseries_path)

DNI_threshold = 700
# print(df.head())

# === USER SETTINGS ===
file_path = "PSA_timeSeries_Metas.csv"   # path to your file
value_column = "col3"                    # choose from col2, col3, col4, col5, col6
threshold = 700                          # set your threshold
# ======================

def save_filtered_data(df, threshold, output_file):
    # List to store filtered data
    filtered_data = []
    
    # Iterate through DataFrame rows
    for num in range(len(df)):
        if df['DNI'][num] > threshold:
            filtered_data.append([df['Date-Time'][num], df['DNI'][num]])
    
    # Write filtered data to CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['Date-Time', 'DNI'])
        # Write data rows
        writer.writerows(filtered_data)

output_filename = 'filtered_DNI_data.csv'
save_filtered_data(df, threshold, output_filename)