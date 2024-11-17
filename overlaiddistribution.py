#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 00:09:49 2024

@author: calebkumar
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load bootstrap data and plot distributions
def overlay_distributions(file1, file2, label1, label2, output_file="overlay_distribution.png"):
    """
    Overlay two distribution plots from bootstrap data files.

    Args:
        file1 (str): Path to the first bootstrap CSV file.
        file2 (str): Path to the second bootstrap CSV file.
        label1 (str): Label for the first distribution.
        label2 (str): Label for the second distribution.
        output_file (str): Path to save the output plot.

    Returns:
        None
    """
    # Load the bootstrap data
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    
    # Check if the files have the expected column
    if 'Bootstrap Values' not in data1.columns or 'Bootstrap Values' not in data2.columns:
        raise ValueError("Bootstrap CSV files must contain a column named 'Bootstrap Values'.")
    
    # Extract bootstrap values
    bootstrap_values1 = data1['Bootstrap Values']
    bootstrap_values2 = data2['Bootstrap Values']
    
    # Plot the distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_values1, kde=True, bins=30, color="blue", alpha=0.6, label=label1)
    sns.histplot(bootstrap_values2, kde=True, bins=30, color="orange", alpha=0.6, label=label2)
    
    # Add legend, title, and labels
    plt.axvline(bootstrap_values1.mean(), color='blue', linestyle='--', label=f"{label1} Mean")
    plt.axvline(bootstrap_values2.mean(), color='orange', linestyle='--', label=f"{label2} Mean")
    plt.title("Overlay of Bootstrap Distributions")
    plt.xlabel("Bootstrap Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"Overlay distribution plot saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Paths to the bootstrap data files (update these paths as necessary)
    file1 = "plots/distributiondata/Black_65 and Over_Female_bootstrap.csv"
    file2 = "plots/distributiondata/White_Under 65_Male_bootstrap.csv"
    
    # Labels for the distributions
    label1 = "Black (65 and Over, Female)"
    label2 = "White (Under 65, Male)"
    
    # Output file for the overlay plot
    output_file = "overlay_distribution.png"
    
    # Plot the distributions
    overlay_distributions(file1, file2, label1, label2, output_file)