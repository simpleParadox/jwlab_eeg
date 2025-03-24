import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from path_mappings import w2v_path_mapping_multiple_seeds
from plot_significance_same_time_window import scores_and_error

def extract_layer_number(filename):
    """Extract layer number from filename."""
    match = re.search(r'layer_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def plot_layer_wise_decoding(model_name, age_group, start_idx=50, end_idx=100, save_fig=True, plot_w2v_results=False):
    """
    Plot the mean accuracy for each window across different layers.
    
    Args:
        model_name (str): Model name to filter files (e.g., 'gpt2-large')
        age_group (int): Age group to filter files (e.g., 9 or 12)
        start_idx (int): Start index for accuracy values
        end_idx (int): End index for accuracy values
        save_fig (bool): Whether to save the figure
    """
    # Define directory path
    directory = f'/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors'
    
    # Search pattern to filter files
    pattern = f"*{age_group}m-{model_name}*svd_vectors_*{age_group}m_all_data.npz"
    file_paths = glob.glob(os.path.join(directory, pattern))
    
    
    if not file_paths:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Dictionary to store accuracies by layer
    layer_accuracies = {}
    
    # Process each file
    for file_path in file_paths:
        layer = extract_layer_number(file_path)
        if layer is None:
            print(f"Could not extract layer number from {file_path}")
            continue
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)['arr_0'].tolist()
        
        # Initialize list for this layer if it doesn't exist
        if layer not in layer_accuracies:
            layer_accuracies[layer] = []
        
        # Extract and store mean accuracies for each window
        window_means = []
        for window_idx in range(111):  # Windows from 0 to 110
            if window_idx in data[0]:
                window_data = data[0][window_idx]
                if window_data and len(window_data) > end_idx:
                    # Calculate mean for specified range
                    mean_acc = np.mean(window_data[start_idx:end_idx])
                    window_means.append(mean_acc)
                else:
                    window_means.append(np.nan)
            else:
                window_means.append(np.nan)
        
        layer_accuracies[layer] = window_means
    
    if not layer_accuracies:
        print("No valid layer data found")
        return
    
    # Sorting layers
    sorted_layers = sorted(layer_accuracies.keys())
    
    # Define x-axis values (time points)
    x_graph = np.arange(-200, 910, 10)  # Assuming 10ms step size for 111 windows
    
    # Create figure
    sns.set_theme(style='whitegrid', context='paper')
    plt.figure(figsize=(12, 10))
    
    # Create a custom color map for different layers
    num_layers = len(sorted_layers)
    colors = sns.color_palette("viridis", num_layers)
    
    # Plot each layer
    x_graph += 100
    for i, layer in enumerate(sorted_layers):
        plt.plot(x_graph, layer_accuracies[layer], label=f"Layer {layer}", 
                 color=colors[i], linewidth=1.5)
    
    if plot_w2v_results:
        # Load the w2v results for the same age group.
        w2v_path_mapping = f'{age_group}m_{start_idx}_{end_idx}'
        non_permuted_w2v = np.load(w2v_path_mapping_multiple_seeds[w2v_path_mapping], allow_pickle=True)['arr_0'].tolist()
        x_graph_w2v, y_graph_w2v, error_w2v = scores_and_error(non_permuted_w2v) # Uses default seed_range because w2v results are stored separately for each seed range.
        color = 'green' if age_group == 9 else 'purple'
        plt.plot(x_graph_w2v, y_graph_w2v, label=f'{age_group}m w2v', color=color, linewidth=1.5, linestyle='solid')
        plt.fill_between(x_graph_w2v, y_graph_w2v - error_w2v, y_graph_w2v + error_w2v, color=color, alpha=0.2)
        
    
    # Add horizontal line at chance level (0.5)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.8)
    
    # Add vertical line at time 0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    # Customize plot
    plt.title(f"{model_name} ({age_group}m) Layer-wise Decoding Accuracy SVD", fontsize=16)
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("2 vs 2 Accuracy", fontsize=16)
    plt.ylim(0.35, 0.65)  # Typical range for 2v2 accuracy
    plt.xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'], fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add legend, adjusted to not overlap with plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig:
        output_dir = os.path.join(os.getcwd(), 'layer_wise_plots')
        os.makedirs(output_dir, exist_ok=True)
        if not plot_w2v_results:
            plt.savefig(os.path.join(output_dir, f"{model_name}_{age_group}m_layer_wise_decoding_{start_idx}_{end_idx}.png"), dpi=300, bbox_inches='tight')
        else:
            # Add the w2v results to the filename.
            plt.savefig(os.path.join(output_dir, f"{model_name}_and_w2v_{age_group}m_layer_wise_decoding_{start_idx}_{end_idx}.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return layer_accuracies

# Example usage:
if __name__ == "__main__":
    # Plot for gpt2-large at 9 months
    plot_layer_wise_decoding("gpt2-large", 9, start_idx=50, end_idx=100)
    
    # # Plot for gpt2-large at 12 months
    plot_layer_wise_decoding("gpt2-large", 12, start_idx=50, end_idx=100)
    
    # # Also for gpt2-large-mean for both groups.
    # plot_layer_wise_decoding('gpt2-large-mean', 9, start_idx=50, end_idx=100)

    # plot_layer_wise_decoding('gpt2-large-mean', 12, start_idx=50, end_idx=100)
    
    
    
    # plot_layer_wise_decoding('gpt2-large-mean', 9, start_idx=50, end_idx=100, plot_w2v_results=True)
    
    # plot_layer_wise_decoding('gpt2-large-mean', 12, start_idx=50, end_idx=100, plot_w2v_results=True)

