import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Drop rows with missing Delta_SpO2_SaO2 or race_ethnicity
data = data.dropna(subset=['Delta_SpO2_SaO2', 'race_ethnicity'])

# Set up the boxplot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the boxplot without outliers
sns.boxplot(
    data=data,
    x='race_ethnicity',
    y='Delta_SpO2_SaO2',
    order=relevant_ethnicities,
    palette='tab10',
    showfliers=False  # Removes the circles representing outliers
)

# Add plot titles and labels
plt.title("Boxplot of Delta_SpO2_SaO2 by Ethnicity (Without Outliers)", fontsize=16)
plt.xlabel("Ethnicity", fontsize=14)
plt.ylabel("Delta_SpO2_SaO2 (SpO2 - SaO2)", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()