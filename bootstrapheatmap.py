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

# Directory for saving heatmaps
output_dir = "plots/heatmaps_bs"
os.makedirs(output_dir, exist_ok=True)

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['Grouping', 'Mean', 'Lower CI', 'Upper CI'])

# Function to bootstrap data and calculate confidence intervals
def bootstrap_heatmap(data, n_bootstrap=1000):
    order = ["<88", "88-94", "94+"]
    bootstrap_results = []
    
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
        
        # Add to bootstrap results
        bootstrap_results.append(heatmap_data_normalized.values)
    
    # Convert to array
    bootstrap_array = np.array(bootstrap_results)  # Shape: (n_bootstrap, 3, 3)
    
    # Calculate mean and confidence intervals
    heatmap_mean = bootstrap_array.mean(axis=0)
    heatmap_lower = np.percentile(bootstrap_array, 2.5, axis=0)
    heatmap_upper = np.percentile(bootstrap_array, 97.5, axis=0)
    
    return heatmap_mean, heatmap_lower, heatmap_upper, order

def generate_heatmap_with_ci(data, age_group, ethnicity, gender, output_dir, results_df, n_bootstrap=1000):
    # Filter data for the specific age group, ethnicity, and gender
    subset_data = data[
        (data['Age_Group'] == age_group) & 
        (data['race_ethnicity'] == ethnicity) & 
        (data['Gender'] == gender)
    ]
    
    if subset_data.empty:
        return results_df
    
    # Bootstrap data and calculate statistics
    heatmap_mean, heatmap_lower, heatmap_upper, order = bootstrap_heatmap(subset_data, n_bootstrap)
    
    # Flatten the heatmap arrays and store the results
    group_name = f"{ethnicity}_{age_group}_{gender}"
    new_row = pd.DataFrame({
        'Grouping': [group_name],
        'Mean': [heatmap_mean.flatten().tolist()],
        'Lower CI': [heatmap_lower.flatten().tolist()],
        'Upper CI': [heatmap_upper.flatten().tolist()]
    })
    
    # Concatenate the new row to the results DataFrame
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Save heatmaps (mean, lower CI, upper CI)
    mean_filename = f"{group_name}_mean.png"
    lower_filename = f"{group_name}_lower.png"
    upper_filename = f"{group_name}_upper.png"

    # Plot and save mean heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_mean, 
        annot=True, 
        fmt='.2%', 
        cmap='YlGnBu', 
        cbar=True,
        xticklabels=order,
        yticklabels=order
    )
    plt.title(f"Heatmap: {ethnicity} ({age_group}, {gender})\nMean Values")
    plt.xlabel("SpO2 Category")
    plt.ylabel("SaO2 Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, mean_filename))
    plt.close()

    # Plot and save lower CI heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_lower, 
        annot=True, 
        fmt='.2%', 
        cmap='YlGnBu', 
        cbar=True,
        xticklabels=order,
        yticklabels=order
    )
    plt.title(f"Heatmap: {ethnicity} ({age_group}, {gender})\nLower 2.5% CI")
    plt.xlabel("SpO2 Category")
    plt.ylabel("SaO2 Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, lower_filename))
    plt.close()

    # Plot and save upper CI heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_upper, 
        annot=True, 
        fmt='.2%', 
        cmap='YlGnBu', 
        cbar=True,
        xticklabels=order,
        yticklabels=order
    )
    plt.title(f"Heatmap: {ethnicity} ({age_group}, {gender})\nUpper 97.5% CI")
    plt.xlabel("SpO2 Category")
    plt.ylabel("SaO2 Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, upper_filename))
    plt.close()
    
    return results_df


# Generate heatmaps and store statistics
age_groups = ["Under 65", "65 and Over"]
genders = ["Male", "Female"]
for age_group in age_groups:
    for ethnicity in relevant_ethnicities:
        for gender in genders:
            results_df = generate_heatmap_with_ci(data, age_group, ethnicity, gender, output_dir, results_df)

# Save results to a CSV
results_df.to_csv("heatmap_statistics.csv", index=False)