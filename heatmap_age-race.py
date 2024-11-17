#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:06:44 2024

@author: calebkumar
"""

import os
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

# Filter relevant ethnicities
relevant_ethnicities = [
    "White", "Black", "Asian", "Hispanic OR Latino"
]
data = data[data['race_ethnicity'].isin(relevant_ethnicities)]

# Directory for saving heatmaps
output_dir = "plots/heatmaps"
os.makedirs(output_dir, exist_ok=True)

# Function to generate normalized heatmap for each age group and ethnicity
def generate_normalized_heatmap(data, age_group, ethnicity, output_dir):
    # Filter data for the specific age group and ethnicity
    subset_data = data[(data['Age_Group'] == age_group) & (data['race_ethnicity'] == ethnicity)]
    
    # Create a 3x3 pivot table for counts of SpO2 and SaO2 categories
    heatmap_data = subset_data.pivot_table(
        index='SaO2_Category', 
        columns='SpO2_Category', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Normalize the heatmap by the total count of rows for this subgroup
    total_count = subset_data.shape[0]
    heatmap_data_normalized = heatmap_data / total_count if total_count > 0 else heatmap_data
    
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
    plt.title(f"Heatmap: {ethnicity} ({age_group})")
    plt.xlabel("SpO2 Category")
    plt.ylabel("SaO2 Category")
    plt.tight_layout()
    
    # Save the heatmap as a PNG file
    filename = f"{ethnicity.replace(' ', '_')}_{age_group.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

# Generate and save heatmaps for each combination of age group and ethnicity
age_groups = ["Under 65", "65 and Over"]
for age_group in age_groups:
    for ethnicity in relevant_ethnicities:
        generate_normalized_heatmap(data, age_group, ethnicity, output_dir)