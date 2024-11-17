import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Calculate delta between SpO2 and SaO2
data['Delta_SpO2_SaO2'] = data['SpO2'] - data['SaO2']

# Define BMI ranges
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight (<18.5)"
    elif 18.5 <= bmi < 25:
        return "Normal (18.5-24.9)"
    elif 25 <= bmi < 30:
        return "Overweight (25-29.9)"
    else:
        return "Obese (>=30)"

# Apply BMI categorization
data['BMI_Category'] = data['BMI_admission'].apply(categorize_bmi)

# Function to calculate mean, standard deviation, and count
def calculate_stats(grouped_data):
    means = grouped_data.mean()
    stds = grouped_data.std()
    counts = grouped_data.count()
    return means, stds, counts

# Group data by BMI category
bmi_means, bmi_stds, bmi_counts = calculate_stats(data.groupby('BMI_Category')['Delta_SpO2_SaO2'])
bmi_means = bmi_means.reindex(["Underweight (<18.5)", "Normal (18.5-24.9)", 
                                "Overweight (25-29.9)", "Obese (>=30)"])
bmi_stds = bmi_stds.reindex(bmi_means.index)
bmi_counts = bmi_counts.reindex(bmi_means.index)

# Group data by sex
sex_means, sex_stds, sex_counts = calculate_stats(data.groupby('sex_female')['Delta_SpO2_SaO2'])

# Group data by race/ethnicity
race_means, race_stds, race_counts = calculate_stats(data.groupby('race_ethnicity')['Delta_SpO2_SaO2'])

# Function to plot bar graph with n and error bars
def plot_bar_with_error(means, stds, counts, title, xlabel, ylabel, xticks_rotation=0, color='blue'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.index, means, yerr=stds, capsize=5, color=color, alpha=0.7)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=xticks_rotation)
    # Add n above each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'n={count}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

# Plot BMI category vs Delta
plot_bar_with_error(
    bmi_means, bmi_stds, bmi_counts,
    title='Average Delta (SpO2 - SaO2) by BMI Category',
    xlabel='BMI Category',
    ylabel='Average Delta (SpO2 - SaO2)',
    xticks_rotation=45,
    color='skyblue'
)

# Plot Sex vs Delta
plot_bar_with_error(
    sex_means, sex_stds, sex_counts,
    title='Average Delta (SpO2 - SaO2) by Sex',
    xlabel='Sex (0=Male, 1=Female)',
    ylabel='Average Delta (SpO2 - SaO2)',
    xticks_rotation=0,
    color='salmon'
)

# Plot Race/Ethnicity vs Delta
plot_bar_with_error(
    race_means, race_stds, race_counts,
    title='Average Delta (SpO2 - SaO2) by Race/Ethnicity',
    xlabel='Race/Ethnicity',
    ylabel='Average Delta (SpO2 - SaO2)',
    xticks_rotation=45,
    color='lightgreen'
)


