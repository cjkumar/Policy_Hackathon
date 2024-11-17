#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:02:56 2024

@author: calebkumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Categorize SpO2 and SaO2 into ranges
def categorize_oxygen(value):
    if value < 88:
        return "<88"
    elif 88 <= value <= 94:
        return "88-94"
    else:
        return "94+"

data['SpO2_Category'] = data['SpO2'].apply(categorize_oxygen)
data['SaO2_Category'] = data['SaO2'].apply(categorize_oxygen)

# Categorize patients into age groups (65 and over, under 65)
data['Age_Group'] = np.where(data['admission_age'] >= 65, "65 and Over", "Under 65")

# Function to generate normalized heatmap for each age group
def generate_normalized_heatmap(data, age_group):
    # Filter data for the specific age group
    age_group_data = data[data['Age_Group'] == age_group]
    
    # Create a 3x3 pivot table for counts of SpO2 and SaO2 categories
    heatmap_data = age_group_data.pivot_table(
        index='SaO2_Category', 
        columns='SpO2_Category', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Normalize the heatmap by the total count of rows for the age group
    total_count = age_group_data.shape[0]
    heatmap_data_normalized = heatmap_data / total_count
    
    # Reorder both axes to follow <88, 88-94, 94+
    order = ["<88", "88-94", "94+"]
    heatmap_data_normalized = heatmap_data_normalized.reindex(index=order, columns=order, fill_value=0)
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='YlGnBu', 
        cbar=True
    )
    plt.title(f"Normalized Heatmap of SaO2 vs SpO2 for Age Group: {age_group}")
    plt.xlabel("SpO2 Category")
    plt.ylabel("SaO2 Category")
    plt.tight_layout()
    plt.show()

# Generate normalized heatmaps for each age group
unique_age_groups = data['Age_Group'].unique()
for age_group in unique_age_groups:
    generate_normalized_heatmap(data, age_group)