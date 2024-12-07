import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_with_fills(data_folder, output_file="all_distributions_kde_filled.png"):
    """
    Plot KDEs for all bootstrap CSV files in a folder with filled curves

    Args:
        data_folder (str): Path to the folder containing bootstrap CSV files.
        output_file (str): Path to save the resulting KDE plot.

    Returns:
        None
    """
    # Initialize a dictionary to store bootstrap values and labels
    bootstrap_data = {}
    
    # Iterate over all CSV files in the folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_folder, file_name)
            
            # Extract group name from the file name
            group_name = file_name.replace("_bootstrap.csv", "").replace("_", " ")
            
            # Load the bootstrap data
            data = pd.read_csv(file_path)
            if "Bootstrap Values" in data.columns:
                bootstrap_data[group_name] = data["Bootstrap Values"]
    
    # Plot KDEs for all groups with fills
    plt.figure(figsize=(12, 8))
    for group, values in bootstrap_data.items():
        sns.kdeplot(values, label=group, linewidth=1.5, fill=True, alpha=0.5)

    # Customize plot
    plt.title("KDE Plot of Bootstrap Distributions (Filled)", fontsize=16)
    plt.xlabel("Bootstrap Values", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Move the legend to the side
    plt.legend(title="Groups", fontsize=10, title_fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"KDE plot with filled curves saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Folder containing bootstrap data CSVs
    data_folder = "plots/distributiondata"  # Update to the correct folder path
    
    # Output file for the KDE plot
    output_file = "all_distributions_kde_filled.png"
    
    # Generate the plot
    plot_kde_with_fills(data_folder, output_file)
