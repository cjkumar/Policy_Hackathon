import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "bold_dataset.csv"  # Replace with the correct file path
data = pd.read_csv(file_path)

# Drop rows with missing values for SaO2, SpO2, and ethnicity
data = data.dropna(subset=['SaO2', 'SpO2', 'race_ethnicity'])

# Set up the figure and scatterplot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the scatterplot with a best-fit line and confidence interval
sns.lmplot(
    data=data,
    x='SpO2', y='SaO2', hue='race_ethnicity', 
    palette='tab10', height=6, aspect=1.5, 
    ci=95, scatter_kws={'alpha': 0.7}, line_kws={'linewidth': 2}
)

# Add plot titles and labels
plt.title("Scatterplot of SaO2 vs SpO2 by Ethnicity", fontsize=16)
plt.xlabel("SpO2", fontsize=14)
plt.ylabel("SaO2", fontsize=14)
plt.ylim([0,100])
plt.tight_layout()
plt.show()