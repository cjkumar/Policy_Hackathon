#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:17:51 2024

@author: calebkumar
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Condition 1: SaO2 < 88 and SpO2 > 94
condition1 = (data['SaO2'] < 88) & (data['SpO2'] > 94)

# Condition 2: 88 <= SaO2 < 94
condition2 = (data['SaO2'] >= 88) & (data['SaO2'] < 94)

# Total number of rows per ethnicity in the entire dataset
total_ethnicity_counts = data['race_ethnicity'].value_counts()

# Calculate counts for each condition
ethnicity_counts_condition1 = data.loc[condition1, 'race_ethnicity'].value_counts()
ethnicity_counts_condition2 = data.loc[condition2, 'race_ethnicity'].value_counts()

# Normalize each condition by the total number of rows per ethnicity
normalized_condition1 = (ethnicity_counts_condition1 / total_ethnicity_counts).dropna()
normalized_condition2 = (ethnicity_counts_condition2 / total_ethnicity_counts).dropna()

# Ensure both series have the same ethnicities
all_ethnicities = normalized_condition1.index.union(normalized_condition2.index)
normalized_condition1 = normalized_condition1.reindex(all_ethnicities, fill_value=0)
normalized_condition2 = normalized_condition2.reindex(all_ethnicities, fill_value=0)

# Remove "Unknown" ethnicity if it exists
if "Unknown" in all_ethnicities:
    normalized_condition1 = normalized_condition1.drop("Unknown")
    normalized_condition2 = normalized_condition2.drop("Unknown")

# Sort ethnicities by the total proportion of both conditions combined
total_proportions = normalized_condition1 + normalized_condition2
sorted_ethnicities = total_proportions.sort_values(ascending=True).index
normalized_condition1 = normalized_condition1.reindex(sorted_ethnicities)
normalized_condition2 = normalized_condition2.reindex(sorted_ethnicities)

# Create a stacked bar graph
plt.figure(figsize=(12, 8))
bar1 = plt.bar(normalized_condition1.index, normalized_condition1.values, color='skyblue', label='SaO2 < 88 & SpO2 > 94')
bar2 = plt.bar(normalized_condition1.index, normalized_condition2.values, bottom=normalized_condition1.values, color='salmon', label='88 ≤ SaO2 < 94')

# Add labels and title
plt.title("Stacked Bar Graph by Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.legend()

# Add proportion text above each stack
for i, (val1, val2) in enumerate(zip(normalized_condition1.values, normalized_condition2.values)):
    total = val1 + val2
    plt.text(i, total, f'{total:.2%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()