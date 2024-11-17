import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Filter rows where SaO2 < 88 and SpO2 > 94
sa02_spo2_condition = data[(data['SaO2'] < 88) & (data['SpO2'] > 94)]

# Count the total number of rows matching the condition
count_sa02_spo2 = sa02_spo2_condition.shape[0]

# Print the total count
print(f"Number of rows where SaO2 < 88 and SpO2 > 94: {count_sa02_spo2}")

# Ethnicity breakdown for rows matching the condition
ethnicity_counts_condition = sa02_spo2_condition['race_ethnicity'].value_counts()

# Total number of rows per ethnicity in the entire dataset
total_ethnicity_counts = data['race_ethnicity'].value_counts()

# Normalize by dividing the counts of the condition by the total number of rows per ethnicity
normalized_ethnicity_counts = (ethnicity_counts_condition / total_ethnicity_counts).dropna()

# Check if "Unknown" exists and remove it
if "Unknown" in normalized_ethnicity_counts.index:
    normalized_ethnicity_counts = normalized_ethnicity_counts.drop("Unknown")

# Remove ethnicities with insufficient data (e.g., fewer than a threshold count of rows)
threshold = 5  # Set your threshold here
valid_ethnicities = ethnicity_counts_condition[ethnicity_counts_condition >= threshold].index
normalized_ethnicity_counts = normalized_ethnicity_counts.loc[normalized_ethnicity_counts.index.intersection(valid_ethnicities)]

# Sort the ethnicities based on the magnitude of the normalized values (low to high)
normalized_ethnicity_counts = normalized_ethnicity_counts.sort_values(ascending=True)

# Print normalized ethnicity breakdown
print("\Race/Ethnicity Breakdown for SaO2 < 88 and SpO2 > 94 (Ordered):")
print(normalized_ethnicity_counts)

# Plot normalized ethnicity breakdown as a bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(normalized_ethnicity_counts.index, normalized_ethnicity_counts.values, color='skyblue', alpha=0.8)
plt.title("Proportion of Patients with SaO2 < 88 and SpO2 >= 94 by Patient Race/Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Proportion")
plt.xticks(rotation=45)

# Add proportions above the bars
for bar, proportion in zip(bars, normalized_ethnicity_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{proportion:.2%}', ha='center', va='bottom')

plt.tight_layout()
plt.show()