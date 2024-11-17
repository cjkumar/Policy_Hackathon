import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'bold_dataset_cleaned.csv'  # Update with the correct path
data = pd.read_csv(file_path)
# Convert subject_id (column 3) to string
data['subject_id'] = data['subject_id'].astype(str)

# Verify conversion
print("\nSample of subject_id after conversion:")
print(data['subject_id'].head())

# Check unique types to confirm the fix
unique_types_subject_id = data['subject_id'].apply(type).unique()
print(f"Unique types in subject_id: {unique_types_subject_id}")


# Feature engineering: Calculate delta between SaO2 and SpO2
data['Delta_SaO2_SpO2'] = data['SaO2'] - data['SpO2']

# Check data availability
print("Unique values in race_ethnicity:", data['race_ethnicity'].unique())
print("Number of rows with Delta_SaO2_SpO2:", data['Delta_SaO2_SpO2'].notnull().sum())

# Filter rows with relevant data
data = data.dropna(subset=['SaO2', 'SpO2', 'race_ethnicity'])

# Define two groups: White vs. others
data['Race_Comparison'] = np.where(data['race_ethnicity'] == 'White', 'White', 'Other')

# Debug: Check the group sizes
print("Number of White rows:", data[data['Race_Comparison'] == 'White'].shape[0])
print("Number of Other rows:", data[data['Race_Comparison'] == 'Other'].shape[0])

# Ensure datetime columns are parsed correctly
data['datetime_icu_admit'] = pd.to_datetime(data['datetime_icu_admit'], errors='coerce')
data['datetime_hospital_admit'] = pd.to_datetime(data['datetime_hospital_admit'], errors='coerce')

# Calculate time difference in hours between hospital admit and ICU admit
data['time_diff_hours'] = (data['datetime_icu_admit'] - data['datetime_hospital_admit']).dt.total_seconds() / 3600

# Define time windows
time_window_threshold = 1  # Threshold in hours to define ICU vs Non-ICU
data_time_window_1 = data[data['time_diff_hours'] <= time_window_threshold]  # ICU: <= 1 hour
data_time_window_2 = data[data['time_diff_hours'] > time_window_threshold]  # Non-ICU: > 1 hour

# Debug: Check time window sizes
print("Number of rows in Time Window 1 (ICU):", data_time_window_1.shape[0])
print("Number of rows in Time Window 2 (Non-ICU):", data_time_window_2.shape[0])

# Function to create Bland-Altman plots
def bland_altman_plot(group1, group2, title):
    """
    Create a Bland-Altman plot for the given groups.
    """
    if group1.empty or group2.empty:
        print(f"No data for {title}. Skipping plot.")
        return

    mean_values = (group1 + group2) / 2
    diff_values = group1 - group2
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_values, diff_values, alpha=0.6, label='Differences')
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(mean_diff + 1.96 * std_diff, color='blue', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', label='-1.96 SD')
    plt.title(title)
    plt.xlabel('Mean of SaO2 and SpO2')
    plt.ylabel('Difference (SaO2 - SpO2)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Bland-Altman: White vs. Other Groups
group_white = data[data['Race_Comparison'] == 'White']['Delta_SaO2_SpO2']
group_other = data[data['Race_Comparison'] == 'Other']['Delta_SaO2_SpO2']
if not group_white.empty and not group_other.empty:
    bland_altman_plot(group_white, group_other, 'Bland-Altman: White vs Other Groups')
else:
    print("Insufficient data for White vs Other Bland-Altman plot.")

# Bland-Altman: Time Window 1 (ICU) vs Time Window 2 (Non-ICU)
group_time_1 = data_time_window_1['Delta_SaO2_SpO2']
group_time_2 = data_time_window_2['Delta_SaO2_SpO2']
if not group_time_1.empty and not group_time_2.empty:
    bland_altman_plot(group_time_1, group_time_2, 'Bland-Altman: ICU vs Non-ICU Time Windows')
else:
    print("Insufficient data for ICU vs Non-ICU Bland-Altman plot.")