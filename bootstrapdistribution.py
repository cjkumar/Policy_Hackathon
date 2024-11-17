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

# Map gender for clarity
data['Gender'] = np.where(data['sex_female'] == 1, "Female", "Male")

# Directories for saving plots and data
output_dir = "plots/heatmaps_bs"
distribution_dir = "plots/distributions"
distribution_data_dir = "plots/distributiondata"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(distribution_dir, exist_ok=True)
os.makedirs(distribution_data_dir, exist_ok=True)

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['Grouping', 'Mean', 'Lower CI', 'Upper CI'])

# Function to bootstrap data and calculate the single value for mean and CI
def bootstrap_single_value(data, n_bootstrap=1000):
    order = ["<88", "88-94", "94+"]
    bootstrap_ratios = []
    
    for _ in range(n_bootstrap):
        # Resample data with replacement
        resample = data.sample(n=len(data), replace=True)
        
        # Create pivot table
        heatmap_data = resample.pivot_table(
            index='SaO2_Category', 
            columns='SpO2_Category', 
            aggfunc='size', 
            fill_value=0
        )
        # Normalize
        total_count = resample.shape[0]
        heatmap_data_normalized = heatmap_data / total_count if total_count > 0 else heatmap_data
        
        # Reorder axes
        heatmap_data_normalized = heatmap_data_normalized.reindex(index=order, columns=order, fill_value=0)
        
        # Extract the top right cell and its two adjacent cells
        top_right = heatmap_data_normalized.iloc[0, 2]  # Top-right cell
        left_of_top_right = heatmap_data_normalized.iloc[0, 1]  # Cell left of top-right
        below_top_right = heatmap_data_normalized.iloc[1, 2]  # Cell below top-right
        
        # Calculate the ratio for this bootstrap sample
        total_sum = heatmap_data_normalized.values.sum()
        ratio = (top_right + left_of_top_right + below_top_right) / total_sum
        bootstrap_ratios.append(ratio)
    
    # Calculate mean and confidence intervals
    mean_value = np.mean(bootstrap_ratios)
    lower_ci = np.percentile(bootstrap_ratios, 2.5)
    upper_ci = np.percentile(bootstrap_ratios, 97.5)
    
    return mean_value, lower_ci, upper_ci, bootstrap_ratios

# Function to process each grouping and store statistics
def process_grouping(data, age_group, ethnicity, gender, results_df, n_bootstrap=1000):
    # Filter data for the specific age group, ethnicity, and gender
    subset_data = data[
        (data['Age_Group'] == age_group) & 
        (data['race_ethnicity'] == ethnicity) & 
        (data['Gender'] == gender)
    ]
    
    if subset_data.empty:
        return results_df
    
    # Bootstrap and calculate statistics
    mean_value, lower_ci, upper_ci, bootstrap_ratios = bootstrap_single_value(subset_data, n_bootstrap)
    
    # Add results to the DataFrame
    group_name = f"{ethnicity}_{age_group}_{gender}"
    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Grouping': [group_name],
            'Mean': [mean_value],
            'Lower CI': [lower_ci],
            'Upper CI': [upper_ci]
        })
    ], ignore_index=True)
    
    # Save bootstrap values to a CSV file
    bootstrap_data_file = os.path.join(distribution_data_dir, f"{group_name}_bootstrap.csv")
    pd.DataFrame({'Bootstrap Values': bootstrap_ratios}).to_csv(bootstrap_data_file, index=False)
    
    # Plot and save the distribution of bootstrap values
    plt.figure(figsize=(8, 6))
    sns.histplot(bootstrap_ratios, kde=True, bins=30, color="blue", alpha=0.7)
    plt.axvline(mean_value, color='red', linestyle='--', label='Mean')
    plt.axvline(lower_ci, color='green', linestyle='--', label='2.5% CI')
    plt.axvline(upper_ci, color='green', linestyle='--', label='97.5% CI')
    plt.title(f"Distribution of Values: {group_name}")
    plt.xlabel("Bootstrap Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(distribution_dir, f"{group_name}_distribution.png"))
    plt.close()
    
    return results_df

# Process all groupings
age_groups = ["Under 65", "65 and Over"]
genders = ["Male", "Female"]
for age_group in age_groups:
    for ethnicity in relevant_ethnicities:
        for gender in genders:
            results_df = process_grouping(data, age_group, ethnicity, gender, results_df)

# Save results to a CSV
results_df.to_csv("heatmap_summary_statistics.csv", index=False)

print("Summary statistics saved to heatmap_summary_statistics.csv")
print("Distribution plots saved in 'plots/distributions'")
print("Bootstrap data saved in 'plots/distributiondata'")