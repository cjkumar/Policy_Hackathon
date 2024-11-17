import pandas as pd
import numpy as np

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)
 
# Feature engineering: Calculate Delta_SpO2_SaO2
data['Delta_SpO2_SaO2'] = data['SpO2'] - data['SaO2']

# Filter relevant ethnicities
relevant_ethnicities = [
    "White", "Black", "Asian", 
    "American Indian / Alaska Native", "Hispanic OR Latino"
]
data = data[data['race_ethnicity'].isin(relevant_ethnicities)]

# Assume binary outcomes for a diagnostic test
# Add a binary column indicating "hypoxic" for example purposes
data['hypoxic'] = np.where(data['SaO2'] < 88, 1, 0)

# Assume a diagnostic test prediction (e.g., SpO2 threshold)
# Add a binary column indicating test prediction (1=hypoxic, 0=non-hypoxic)
# Adjust the threshold (94) based on your test criteria
data['test_prediction'] = np.where(data['SpO2'] < 94, 1, 0)

# Function to calculate sensitivity, specificity, and Youden's J
def calculate_youden(data_subset):
    # True positives, false negatives, false positives, true negatives
    tp = ((data_subset['test_prediction'] == 1) & (data_subset['hypoxic'] == 1)).sum()
    fn = ((data_subset['test_prediction'] == 0) & (data_subset['hypoxic'] == 1)).sum()
    fp = ((data_subset['test_prediction'] == 1) & (data_subset['hypoxic'] == 0)).sum()
    tn = ((data_subset['test_prediction'] == 0) & (data_subset['hypoxic'] == 0)).sum()
    
    # Sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Youden's J statistic
    youden_j = sensitivity + specificity - 1
    return sensitivity, specificity, youden_j

# Calculate overall Youden's J statistic
overall_sensitivity, overall_specificity, overall_youden_j = calculate_youden(data)
print(f"Overall Sensitivity: {overall_sensitivity:.2f}")
print(f"Overall Specificity: {overall_specificity:.2f}")
print(f"Overall Youden's J: {overall_youden_j:.2f}")

# Calculate Youden's J statistic for each race/ethnicity group
youden_results = {}
for ethnicity in relevant_ethnicities:
    subset = data[data['race_ethnicity'] == ethnicity]
    sensitivity, specificity, youden_j = calculate_youden(subset)
    youden_results[ethnicity] = {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Youden's J": youden_j
    }

# Print results for each ethnicity
print("\nYouden's J Statistic by Ethnicity:")
for ethnicity, metrics in youden_results.items():
    print(f"{ethnicity}:")
    print(f"  Sensitivity: {metrics['Sensitivity']:.2f}")
    print(f"  Specificity: {metrics['Specificity']:.2f}")
    print(f"  Youden's J: {metrics['Youden\'s J']:.2f}")  # Fixed