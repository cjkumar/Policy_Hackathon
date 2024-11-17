#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:52:33 2024

@author: calebkumar
"""

import pandas as pd
import numpy as np

# Load the dataset with debugging enabled
file_path = 'bold_dataset.csv'

# Load the dataset without specifying dtype to check for mixed types
data = pd.read_csv(file_path, low_memory=False)

# Step 1: Identify data types for all columns
print("Column Data Types:")
print(data.dtypes)

# Step 2: Check for mixed types in columns
for col in data.columns:
    unique_types = data[col].apply(type).unique()
    if len(unique_types) > 1:
        print(f"Mixed types found in column '{col}': {unique_types}")

# Step 3: Example fix for columns with mixed types
# Replace "column_name" with the problematic column identified in the output
if 'column_name' in data.columns:
    data['column_name'] = pd.to_numeric(data['column_name'], errors='coerce')

# Step 4: Re-check data types after conversion
print("\nUpdated Column Data Types:")
print(data.dtypes)

# Step 5: Count missing values (optional for cleaning)
print("\nMissing Values per Column:")
print(data.isnull().sum())

# Step 6: Save cleaned data for verification (optional)
cleaned_file_path = 'bold_dataset_cleaned.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to: {cleaned_file_path}")


# Step 1: Inspect column 3
print("\nInspecting column 3:")
print(data.iloc[:, 3].head())  # Replace 3 with the actual column index if necessary

# Step 2: Check unique types in column 3
unique_types = data.iloc[:, 3].apply(type).unique()
print(f"Unique types in column 3: {unique_types}")

# Step 3: Attempt conversion to a consistent type (e.g., numeric)
data.iloc[:, 3] = pd.to_numeric(data.iloc[:, 3], errors='coerce')  # Convert to numeric, invalid entries become NaN

# Step 4: Re-check for mixed types
updated_unique_types = data.iloc[:, 3].apply(type).unique()
print(f"Updated unique types in column 3: {updated_unique_types}")

# Step 5: Check for remaining NaN values in column 3
missing_values = data.iloc[:, 3].isnull().sum()
print(f"Number of missing values in column 3 after conversion: {missing_values}")