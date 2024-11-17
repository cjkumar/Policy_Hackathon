#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:12:40 2024

@author: calebkumar
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Feature engineering: Calculate severity categories
data['severity'] = pd.cut(
    data['SaO2'],
    bins=[-float('inf'), 88, 94, float('inf')],
    labels=['<88', '88-94', '>=94']
)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to filter data and return counts
@app.route('/filter', methods=['POST'])
def filter_data():
    # Get the selected criteria from the request
    ethnicity = request.json.get('ethnicity')
    severity = request.json.get('severity')
    
    # Filter the data based on the criteria
    filtered_data = data
    if ethnicity != 'All':
        filtered_data = filtered_data[filtered_data['race_ethnicity'] == ethnicity]
    if severity != 'All':
        filtered_data = filtered_data[filtered_data['severity'] == severity]
    
    # Count the number of matching patients
    count = len(filtered_data)
    return jsonify({'count': count})

if __name__ == '__main__':
    app.run(debug=True)