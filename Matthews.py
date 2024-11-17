import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

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

# Function to calculate MCC
def calculate_mcc(data_subset):
    # Extract true labels and predicted labels
    y_true = data_subset['hypoxic']
    y_pred = data_subset['test_prediction']
    
    # Compute Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc

# Calculate overall Youden's J statistic
overall_sensitivity, overall_specificity, overall_youden_j = calculate_youden(data)
print(f"Overall Sensitivity: {overall_sensitivity:.2f}")
print(f"Overall Specificity: {overall_specificity:.2f}")
print(f"Overall Youden's J: {overall_youden_j:.2f}")

# Calculate overall MCC
overall_mcc = calculate_mcc(data)
print(f"Overall Matthews Correlation Coefficient: {overall_mcc:.2f}")

# Calculate metrics for each race/ethnicity group
results = {}
for ethnicity in relevant_ethnicities:
    subset = data[data['race_ethnicity'] == ethnicity]
    if len(subset) > 0:  # Ensure subset is not empty
        sensitivity, specificity, youden_j = calculate_youden(subset)
        mcc = calculate_mcc(subset)
        results[ethnicity] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Youden's J": youden_j,
            "MCC": mcc
        }

# Print results for each ethnicity
print("\nMetrics by Ethnicity:")
for ethnicity, metrics in results.items():
    print(f"{ethnicity}:")
    print(f"  Sensitivity: {metrics['Sensitivity']:.2f}")
    print(f"  Specificity: {metrics['Specificity']:.2f}")
    print(f"  Youden's J: {metrics['Youden\'s J']:.2f}")
    print(f"  MCC: {metrics['MCC']:.2f}")