import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_selected_kde_with_fills(data_folder, output_file="selected_distributions_kde_filled.png"):
    """
    Plot KDEs for specific bootstrap CSV files with filled curves and the legend moved to the side.

    Args:
        data_folder (str): Path to the folder containing bootstrap CSV files.
        output_file (str): Path to save the resulting KDE plot.

    Returns:
        None
    """
    # Specify the groups to include in the plot
    selected_groups = [
        "Asian_Under 65_Female",
        "White_Under 65_Male",
        "Black_65 and Over_Female"
    ]
    
    # Initialize a dictionary to store bootstrap values and labels
    bootstrap_data = {}
    
    # Iterate over selected groups
    for group_name in selected_groups:
        file_name = f"{group_name}_bootstrap.csv"
        file_path = os.path.join(data_folder, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File not found for group {group_name}. Skipping.")
            continue
        
        # Load the bootstrap data
        data = pd.read_csv(file_path)
        if "Bootstrap Values" in data.columns:
            bootstrap_data[group_name.replace("_", " ")] = data["Bootstrap Values"]
        else:
            print(f"Warning: No 'Bootstrap Values' column found in {file_name}. Skipping.")
    
    # Ensure there is data to plot
    if not bootstrap_data:
        print("Error: No valid data found for the selected groups.")
        return
    
    # Plot KDEs for selected groups with fills
    plt.figure(figsize=(12, 8))
    for group, values in bootstrap_data.items():
        sns.kdeplot(values, label=group, linewidth=1.5, fill=True, alpha=0.5)

    # Customize plot
    plt.title("KDE Plot of Selected Bootstrap Distributions (Filled)", fontsize=16)
    plt.xlabel("Problematic Misalignment Proportion", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    # Move the legend to the side
    plt.legend(title="Groups", fontsize=10, title_fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"KDE plot for selected groups saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Folder containing bootstrap data CSVs
    data_folder = "plots/distributiondata"  # Update to the correct folder path
    
    # Output file for the KDE plot
    output_file = "selected_distributions_kde_filled.png"
    
    # Generate the plot
    plot_selected_kde_with_fills(data_folder, output_file)