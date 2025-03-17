from curses.ascii import alt
from re import I
from turtle import color
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from mne.stats import fdr_correction, permutation_cluster_test
import glob
from tqdm import tqdm
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, Span, Label
from bokeh.io import export_png
from bokeh.io.export import get_svg
from selenium import webdriver
import chromedriver_binary
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import get_significance
from path_mappings import w2v_path_mapping_multiple_seeds, perm_w2v_path_mapping, ph_path_mapping_multiple_seeds, perm_ph_path_mapping

def plot_image_same_time_with_cluster(results_1_dict,
                                      results_2_dict,
                                      results_1, 
                                      results_2,
                                      results_1_dots, 
                                      results_2_dots,
                                      error_1,
                                      error_2, 
                                      time_window=None, 
                                      alpha=0.05, 
                                      n_permutations=1000,
                                      linestyle_1='solid',
                                      color_1='green',
                                      linestyle_2='solid', 
                                      color_2='purple',
                                      sig_1_options={'color': 'green', 'alpha': 0.3, 'markerfacecolor':None, 'marker_shape':'o', 'markeredgecolor': 'green', 'markeredgewidth': 1},
                                      sig_2_options={'color': 'purple', 'alpha': 0.3, 'markerfacecolor':None, 'marker_shape':'o', 'markeredgecolor': 'purple', 'markeredgewidth': 1},
                                      **kwargs):
    """
    Plot two result curves with significance dots and shaded areas for significant clusters.
    
    Parameters:
    -----------
    results_1 : array-like
        First results array to plot (should be the first element of the results array.)
    results_2 : array-like
        Second results array to plot (should be the first element of the results array.)
    time_window : array-like, optional
        Time points corresponding to results data
    significant_times : array-like, optional
        Boolean array indicating significant time points from user
    alpha : float, optional
        Significance level for the cluster permutation test
    n_permutations : int, optional
        Number of permutations for the cluster permutation test
    **kwargs : dict
        Additional keyword arguments for customizing the plot
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    ax : matplotlib Axes
        The axes containing the plot
    """
    
    
    sns.set_theme(style="whitegrid", context="paper")
    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    # Set default time window if not provided
    if time_window is None:
        time_window = np.arange(len(results_1))
    
    # Plot both results
    line1, = ax.plot(time_window, results_1, label=kwargs.get('label_1', 'Results 1'), color=color_1, linestyle=linestyle_1)
    line2, = ax.plot(time_window, results_2, label=kwargs.get('label_2', 'Results 2'), color=color_2, linestyle=linestyle_2)
    
    windows = sorted(results_1_dict[0].keys())
    print("Windows: ", windows)
    n_times = len(windows) 
    
    # Check that we have sample data for first window to determine n_subjects
    if len(results_1_dict[0][windows[0]]) == 0:
        raise ValueError("No sample data found in results")
    
    n_subjects = 49 #len(results1[0][windows[0]])
    print("Number of subjects: ", n_subjects)
    if len(results_1_dict[0]) == 0:
        raise ValueError("No sample data found in results")
    
    n_subjects = 49 #len(results1[0][windows[0]])
    print("Number of subjects: ", n_subjects)
    
    # Initialize arrays
    data1 = np.zeros((n_subjects, n_times))
    data2 = np.zeros((n_subjects, n_times))
    print("Shape of data1: ", data1.shape)
    print("Shape of data2: ", data2.shape)
    # Fill arrays with data
    for i, t in enumerate(windows):
        # if len(results1[0][t]) != n_subjects or len(results2[0][t]) != n_subjects:
            # raise ValueError(f"Inconsistent number of samples in window {t}")
        data1[:, i] = np.array(results_1_dict[0][t][:n_subjects]) # First n_subjects values.
        data2[:, i] = np.array(results_2_dict[0][t][:n_subjects])
    # Perform cluster permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [data1, data2], n_permutations=n_permutations, threshold=None, tail=0, n_jobs=1, seed=kwargs.get('seed', None)
    )
    
    # Plot significant clusters.
    times = np.array(time_window)
    shifted_times = times
    labels = []
    significant_windows = []
    clusters_after_onset = []
    labels_after_onset = []
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= alpha:
            # Get indices of cluster timepoints
            print("Cluster: ", c)
            cluster_inds = [c[0][0], c[0][-1]] 
            
            if len(cluster_inds) > 0:
                significant_windows.append([times[i] for i in cluster_inds])
                label = f'Significant (p = {cluster_p_values[i_c]:.3f})'
                labels.append(label)
                
                # Check if the cluster is after the onset
                # if shifted_times[cluster_inds[0]] >= 0:
                clusters_after_onset.append(c[0])
                labels_after_onset.append(label)
                # Apply the shift to the highlight spans too
                print(f"Significant cluster found: windows {shifted_times[cluster_inds[0]]} to {shifted_times[cluster_inds[-1]]}, p={cluster_p_values[i_c]:.4f}")
    
    # Calculate the largest cluster
    # Find out the largest cluster for all the clusters after onset.
    largest_cluster = None
    largest_cluster_size = 0
    largest_cluster_idx = None
    for i, cluster in enumerate(clusters_after_onset):
        if len(cluster) > largest_cluster_size:
            largest_cluster = cluster
            largest_cluster_idx = i
            largest_cluster_size = len(cluster)
    
    if largest_cluster is not None:
        print(f"Largest cluster found: windows {shifted_times[largest_cluster[0]]} to {shifted_times[largest_cluster[-1]]}")        
        ax.axvspan(shifted_times[largest_cluster[0]], shifted_times[cluster_inds[-1]], 
                            color='pink', alpha=0.3)
        
        # Add a label for the largest cluster
        ax.text(shifted_times[largest_cluster[0]], 0.70, f"{labels_after_onset[largest_cluster_idx]}", fontsize=12, ha='left')
    
    
                   
    # Plot user-provided significance dots if provided
    ax.plot(results_1_dots, [0.35] * len(results_1_dots), sig_1_options['marker_shape'], 
            color=sig_1_options['color'], alpha=sig_1_options['alpha'], markerfacecolor=sig_1_options['markerfacecolor'],
            markeredgecolor=sig_1_options['markeredgecolor'], markeredgewidth=sig_1_options['markeredgewidth'])
    ax.plot(results_2_dots, [0.37] * len(results_2_dots), sig_2_options['marker_shape'], 
            color=sig_2_options['color'], alpha=sig_2_options['alpha'], markerfacecolor=sig_2_options['markerfacecolor'],
            markeredgecolor=sig_2_options['markeredgecolor'], markeredgewidth=sig_2_options['markeredgewidth'])
    
    # Plot the error regions using fill_between
    ax.fill_between(time_window, results_1 - error_1, results_1 + error_1, color=color_1, alpha=0.3)
    ax.fill_between(time_window, results_2 - error_2, results_2 + error_2, color=color_2, alpha=0.3)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
    
    # Set x-ticks
    ax.set_xticks(np.arange(-200, 1001, 200))
    ax.set_xticklabels(['-200', '0', '200', '400', '600', '800', '1000'])
    
    # if significant_times is not None and np.any(significant_times):
    #     sig_times = time_window[significant_times]
    #     y_max = max(np.max(results_1), np.max(results_2))
    #     y_dot_pos = kwargs.get('y_dot_position', y_max * 1.05)
    #     ax.plot(sig_times, [y_dot_pos] * len(sig_times), 'k.', markersize=kwargs.get('dot_size', 10))
    
    # Add accuracy at 0ms text.
    # if 0 in x_graph:
    #     acc_at_zero = y_graph[np.where(x_graph == 0)[0][0]]
    #     ax.text(700, 0.35, f"Acc at 0ms: {acc_at_zero:.3f}", fontsize=12, ha='right')
    
    # Configure legend
    ax.legend(loc='upper right', fontsize=14)
    ax.set_xlabel(kwargs.get('xlabel', 'Time (s)'), fontsize=16)
    ax.set_ylabel(kwargs.get('ylabel', '2 vs 2 Accuracy'), fontsize=16)
    ax.set_title(kwargs.get('title', '2 vs 2 Accuracy Over Time'), fontsize=18)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.set_ylim(0.3, 0.8)
    
    # Add borders to the ax.
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
   
    return fig, ax


def plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name, linestyle='solid'):
    """
    This function is used to plot the image with the significance dots using Seaborn.
    Exports to both PNG and SVG formats.
    """
    # Create a Seaborn figure
    sns.set(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot the main line
    line_style = '-' if linestyle == 'solid' else '--'
    ax.plot(x_graph, y_graph, color='black', linestyle=line_style, linewidth=2, label='Accuracy')
    
    # Add shaded error region
    ax.fill_between(x_graph, y_graph - error, y_graph + error, color='#58c0fc', alpha=0.7)
    
    # Add significance dots
    ax.plot(x_dots, [0.35] * len(x_dots), 'ro', label='Significance')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
    
    # Set axis labels
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy')
    
    # Set x-ticks
    ax.set_xticks(np.arange(-200, 1001, 200))
    ax.set_xticklabels(['-200', '0', '200', '400', '600', '800', '1000'])
    
    # Add text for accuracy at 0ms
    if 0 in x_graph:
        acc_at_zero = y_graph[np.where(x_graph == 0)[0][0]]
        ax.text(700, 0.35, f"Acc at 0ms: {acc_at_zero:.3f}", fontsize=12, ha='right')
    
    # Configure legend
    ax.legend(loc='upper left')
    
    # Save as PNG
    plt.savefig(f'{file_name}.png', bbox_inches='tight')
    
    # Save as SVG
    plt.savefig(f'{file_name}.svg', format='svg', bbox_inches='tight')
    
    return fig
    

def scores_and_error(results):
    scoreMean = []
    stdev = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            scoreMean.append(float(round(np.mean(results[i][j]), 4)))
            stdev.append(round(stats.sem(results[i][j]), 4)) # This is actually std error of the mean (and not std dev).

    length_per_window_plt = 10  # 1200 / len(scoreMean)

    x_graph = np.arange(-200, 910, length_per_window_plt)
    x_graph += 100  # NOTE: Change this based on the window size.
    y_graph = scoreMean

    stdevplt = np.array(stdev)
    error = stdevplt

    return x_graph, y_graph, error


# The fixed seed path should contain the results stored using the store_fixed_seed_results.py script inside regression/fixed_seed_results.
# null_h_ph_kde_path_9m = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuted_npz_processed/kde/eeg_to_ph/9m_eeg_to_ph_null.npz" 
# non_permuted_results_9m_ph_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/ph_9m_fixed_seed.npz'
# results_9m_ph_fixed_seed = np.load(non_permuted_results_9m_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
# x_graph, y_graph, error = scores_and_error(results_9m_ph_fixed_seed)
# print(y_graph)
# significance_dots = get_significance_dots_from_kde(y_graph, null_h_ph_kde_path_9m)
# significance_dots += 100 # Increase by 100 to match the shift.
# print("Significance dots with shift:")
# print(significance_dots.tolist())
# plot_image_same_time_window(x_graph, y_graph, error, significance_dots, file_name='eeg_to_ph_9m_dots_statsmodels.png')


# ===========================================
# Getting significance dots using the raw permuted results.


# Phoneme 9m.
# non_permuted_results_9m_ph_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/ph_9m_fixed_seed.npz'
# results_9m_ph_fixed_seed = np.load(non_permuted_results_9m_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
# x_graph, y_graph, error = scores_and_error(results_9m_ph_fixed_seed)
# print(y_graph)

# all_acc_waves = []
# all_acc_waves_no_mean = {}
# path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/eeg_to_ph/9m"
# for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
#     perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
#     for window, acc in perm_accs.items():
#         all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
#         all_acc_waves.append(np.mean(acc))
# reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
# print("Significance dots without shift:")
# print(reject_fdr)
# print("Length of reject_fdr: ", len(reject_fdr))
# # Get the x_dots.
# x_dots = x_graph[reject_fdr]
# print("Significance dots with shift:")
# print(x_dots.tolist())
# plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='2025_eeg_to_ph_9m_dots_from_raw_using_fdr_kde_all_over.png',
#                             linestyle='dashed')







# # # ===========================================
# # # Phoneme for 12m.
# non_permuted_results_12m_ph_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/ph_12m_fixed_seed.npz'
# results_12m_ph_fixed_seed = np.load(non_permuted_results_12m_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
# x_graph, y_graph, error = scores_and_error(results_12m_ph_fixed_seed)
# print(y_graph)

# all_acc_waves = []
# all_acc_waves_no_mean = {}
# path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/eeg_to_ph/12m"
# for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
#     perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
#     for window, acc in perm_accs.items():
#         all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
#         all_acc_waves.append(np.mean(acc))
# # import pdb; pdb.set_trace()
# reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
# print("Significance dots without shift:")
# print(reject_fdr)
# print("Length of reject_fdr: ", len(reject_fdr))
# # Get the x_dots.
# x_dots = x_graph[reject_fdr]
# print("Significance dots with shift:")
# print(x_dots.tolist())
# plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='eeg_to_ph_12m_dots_from_raw_using_fdr_kde_all_over.png',
#                             linestyle='dashed')







# ===========================================
# w2v for 9m.
# non_permuted_results_9m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/w2v_9m_fixed_seed.npz'
# non_permuted_results_9m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250315-005846_2025_mar_14_w2v-9m-eeg_to_vectors_fixed_seed_50_iters_range_50_100_1_iterations_range(50, 100)_9m_all_data.npz'
# non_permuted_results_9m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250315-010731_2025_mar_14_w2v-9m-eeg_to_vectors_fixed_seed_50_iters_range_100_150_1_iterations_range(100, 150)_9m_all_data.npz'
# non_permuted_results_9m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250315-010424_2025_mar_14_w2v-9m-eeg_to_vectors_fixed_seed_50_iters_range_150_200_1_iterations_range(150, 200)_9m_all_data.npz'
# results_9m_w2v_fixed_seed = np.load(non_permuted_results_9m_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
# x_graph, y_graph, error = scores_and_error(results_9m_w2v_fixed_seed)
# print(y_graph)

# all_acc_waves = []
# all_acc_waves_no_mean = {}
# path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/same_time_results/permutation/9m_mod_2v2"
# for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
#     perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
#     for window, acc in perm_accs.items():
#         all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
#         all_acc_waves.append(np.mean(acc))
# # import pdb; pdb.set_trace()
# reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
# print("Significance dots without shift:")
# print(reject_fdr)
# print("Length of reject_fdr: ", len(reject_fdr))
# # Get the x_dots.
# x_dots = x_graph[reject_fdr]
# print("Significance dots with shift:")
# print(x_dots.tolist())
# plot_image_same_time_window(x_graph, y_graph, error, x_dots, 
#                             file_name='9m_seaborn_fixed_seed_range_50_100_Mar_15_eeg_to_w2v_dots_from_raw_using_fdr_kde_all_over',
#                             linestyle='solid')





# # ===========================================
# w2v for 12m.
# non_permuted_results_12m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/w2v_12m_fixed_seed.npz'
# non_permuted_results_12m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250313-144653_2025_mar_13_w2v-12m-eeg_to_vectors_no_fixed_seed_50_iters_trial_1_1_iterations_50_12m_all_data.npz'
# non_permuted_results_12m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250313-144424_2025_mar_13_w2v-12m-eeg_to_vectors_no_fixed_seed_50_iters_trial_2_1_iterations_50_12m_all_data.npz'
# non_permuted_results_12m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/same_time_results/observed/vectors/20250313-144451_2025_mar_13_w2v-12m-eeg_to_vectors_no_fixed_seed_50_iters_trial_3_1_iterations_50_12m_all_data.npz'
# results_12m_w2v_fixed_seed = np.load(non_permuted_results_12m_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
# x_graph, y_graph, error = scores_and_error(results_12m_w2v_fixed_seed)
# print(y_graph)

# all_acc_waves = []
# all_acc_waves_no_mean = {}
# path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/same_time_results/permutation/12m_mod_2v2"
# # Only select the first 100 files in this case because we have around 527 perm tests.
# count = 0
# for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
#     perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
#     for window, acc in perm_accs.items():
#         all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
#         all_acc_waves.append(np.mean(acc))
#     count += 1
#     if count == 99:
#         break
# # import pdb; pdb.set_trace()
# reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
# print("Significance dots without shift:")
# print(reject_fdr)
# print("Length of reject_fdr: ", len(reject_fdr))
# # Get the x_dots.
# x_dots = x_graph[reject_fdr]
# print("Significance dots with shift:")
# print(x_dots.tolist())
# plot_image_same_time_window(x_graph, y_graph, error, x_dots, 
#                             file_name='eeg_to_w2v_12m_dots_trial_3_from_raw_using_fdr_kde_all_over.png',
                            # linestyle='solid')
                            
                            
                            
"""
Plotting both 9m and 12m with clusters in the same graph.
"""

# NOTE: I doubled checked the graphs by copy pasting the results array from the .out file and they seem to give the same graph.
# Load the non-permuted results.




age_group_1 = '9m'
age_group_2 = '12m'
ALL_SEED_RANGES = ['0_50', '50_100', '100_150', '150_200']
for seed_range in ALL_SEED_RANGES:
    
    non_permuted_mapping_key_group_1 = f'{age_group_1}_{seed_range}'
    non_permuted_mapping_key_group_2 = f'{age_group_2}_{seed_range}'
    w2v_perm_mapping_key_group_1 = f'w2v_perm_{age_group_1}'
    w2v_perm_mapping_key_group_2 = f'w2v_perm_{age_group_2}'
    ph_perm_mapping_key_group_1 = f'ph_perm_{age_group_1}'
    ph_perm_mapping_key_group_2 = f'ph_perm_{age_group_2}'


    # Plotting the ph decoding results for both 9m and 12m.
    # First get the paths.
    non_permuted_results_group_1_ph_fixed_seed_path = ph_path_mapping_multiple_seeds[non_permuted_mapping_key_group_1]
    non_permuted_results_group_2_ph_fixed_seed_path = ph_path_mapping_multiple_seeds[non_permuted_mapping_key_group_2]
    path_ph_perm_group_1 = perm_ph_path_mapping[ph_perm_mapping_key_group_1]
    path_ph_perm_group_2 = perm_ph_path_mapping[ph_perm_mapping_key_group_2]

    # Now get the paths for the w2v decoding results.
    non_permuted_results_group_1_w2v_fixed_seed_path = w2v_path_mapping_multiple_seeds[non_permuted_mapping_key_group_1]
    non_permuted_results_group_2_w2v_fixed_seed_path = w2v_path_mapping_multiple_seeds[non_permuted_mapping_key_group_2]
    path_w2v_perm_group_1 = perm_w2v_path_mapping[w2v_perm_mapping_key_group_1]
    path_w2v_perm_group_2 = perm_w2v_path_mapping[w2v_perm_mapping_key_group_2]



    # Load the non-permuted results for phoneme.
    results_group_1_ph_fixed_seed = np.load(non_permuted_results_group_1_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
    results_group_2_ph_fixed_seed = np.load(non_permuted_results_group_2_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()

    # Load the non-permuted results for w2v.
    results_group_1_w2v_fixed_seed = np.load(non_permuted_results_group_1_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
    results_group_2_w2v_fixed_seed = np.load(non_permuted_results_group_2_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()

    # Get the scores and errors for phoneme.
    x_graph_9m_ph, y_graph_9m_ph, error_9m_ph = scores_and_error(results_group_1_ph_fixed_seed)
    x_graph_12m_ph, y_graph_12m_ph, error_12m_ph = scores_and_error(results_group_2_ph_fixed_seed)
    time_window_ph = x_graph_9m_ph

    # Get the scores and errors for w2v.
    x_graph_9m_w2v, y_graph_9m_w2v, error_9m_w2v = scores_and_error(results_group_1_w2v_fixed_seed)
    x_graph_12m_w2v, y_graph_12m_w2v, error_12m_w2v = scores_and_error(results_group_2_w2v_fixed_seed)
    time_window_w2v = x_graph_9m_w2v


    # There will be three cases.
    # 1. 9m vs 12m for phoneme.
    # 2. 9m phoneme vs 9m w2v.
    # 3. 12m phoneme vs 12m w2v.


    # Get significance dots for case 1.
    # Get significance dots for phoneme.
    x_dots_9m_ph, reject_fdr_9m_ph, threshold_fdr_9m_ph, p_values_fdr_9m_ph, p_values_9m_ph = get_significance(path_ph_perm_group_1, y_graph_9m_ph, x_graph_9m_ph, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
    x_dots_12m_ph, reject_fdr_12m_ph, threshold_fdr_12m_ph, p_values_fdr_12m_ph, p_values_12m_ph = get_significance(path_ph_perm_group_2, y_graph_12m_ph, x_graph_12m_ph, p_val_thresh=0.01, kde_per_window=False, do_kde=True)

    # Get significance dots for case 2.
    # Get significance dots for w2v.
    x_dots_9m_w2v, reject_fdr_9m_w2v, threshold_fdr_9m_w2v, p_values_fdr_9m_w2v, p_values_9m_w2v = get_significance(path_w2v_perm_group_1, y_graph_9m_w2v, x_graph_9m_w2v, p_val_thresh=0.01, kde_per_window=False, do_kde=True)


    # Get significance dots for case 3.
    # Get significance dots for w2v.
    x_dots_12m_w2v, reject_fdr_12m_w2v, threshold_fdr_12m_w2v, p_values_fdr_12m_w2v, p_values_12m_w2v = get_significance(path_w2v_perm_group_2, y_graph_12m_w2v, x_graph_12m_w2v, p_val_thresh=0.01, kde_per_window=False, do_kde=True)

    # Plot case 1.

    fig_title = f'9m vs 12m ph fixed seed {seed_range}'
    fig, ax = plot_image_same_time_with_cluster(results_group_1_ph_fixed_seed,
                                                results_group_2_ph_fixed_seed,
                                                y_graph_9m_ph,
                                                y_graph_12m_ph,
                                                x_dots_9m_ph,
                                                x_dots_12m_ph,
                                                error_9m_ph,
                                                error_12m_ph,
                                                time_window_ph,
                                                alpha=0.05,
                                                n_permutations=10000,
                                                color_1='green',
                                                color_2='purple',
                                                linestyle_1='dashed',
                                                linestyle_2='dashed',
                                                sig_1_options={'color': 'green', 'alpha': 0.8, 'markerfacecolor':'None', 'marker_shape':'d', 'markeredgecolor':'green', 'markeredgewidth':1},
                                                sig_2_options={'color': 'purple', 'alpha': 0.8, 'markerfacecolor':'None', 'marker_shape':'d', 'markeredgecolor':'purple', 'markeredgewidth':1},
                                                figsize=(10, 10),
                                                label_1=f'9m ph fixed seed {seed_range}',
                                                label_2=f'12m ph fixed seed {seed_range}',
                                                xlabel='Time (s)',
                                                ylabel='2 vs 2 Accuracy',
                                                title=f'{fig_title}',
                                                seed=42)
    file_name = f'from_npz_9m_12m_ph_only_fixed_seed_range_{seed_range}_with_clusters_ph_dots_from_raw_using_fdr_kde_all_over'
    plt.savefig(f'{file_name}.png', bbox_inches='tight')
    # plt.savefig(f'{file_name}.svg', format='svg', bbox_inches='tight')

    # # Plot case 2.
    fig_title = f'9m ph vs 9m w2v fixed seed {seed_range}'
    fig, ax = plot_image_same_time_with_cluster(results_group_1_ph_fixed_seed,
                                                results_group_1_w2v_fixed_seed,
                                                y_graph_9m_ph,
                                                y_graph_9m_w2v,
                                                x_dots_9m_ph,
                                                x_dots_9m_w2v,
                                                error_9m_ph,
                                                error_9m_w2v,
                                                time_window_ph,
                                                alpha=0.05,
                                                n_permutations=10000,
                                                color_1='green',
                                                color_2='green',
                                                linestyle_1='dashed',
                                                linestyle_2='solid',
                                                sig_1_options={'color': 'green', 'alpha': 0.8, 'markerfacecolor':'None', 'marker_shape':'d', 'markeredgecolor':'green', 'markeredgewidth':1},
                                                sig_2_options={'color': 'green', 'alpha': 1, 'markerfacecolor':None, 'marker_shape':'o', 'markeredgecolor':'green', 'markeredgewidth':None},
                                                figsize=(10, 10),
                                                label_1=f'9m ph fixed seed {seed_range}',
                                                label_2=f'9m w2v fixed seed {seed_range}',
                                                xlabel='Time (s)',
                                                ylabel='2 vs 2 Accuracy',
                                                title=f'{fig_title}',
                                                seed=42)
    file_name = f'from_npz_9m_ph_vs_w2v_fixed_seed_range_{seed_range}_with_clusters_ph_w2v_dots_from_raw_using_fdr_kde_all_over'
    plt.savefig(f'{file_name}.png', bbox_inches='tight')
    # # plt.savefig(f'{file_name}.svg', format='svg', bbox_inches='tight')

    # # Plot case 3.
    fig_title = f'12m ph vs 12m w2v fixed seed {seed_range}'
    fig, ax = plot_image_same_time_with_cluster(results_group_2_ph_fixed_seed,
                                                results_group_2_w2v_fixed_seed,
                                                y_graph_12m_ph,
                                                y_graph_12m_w2v,
                                                x_dots_12m_ph,
                                                x_dots_12m_w2v,
                                                error_12m_ph,
                                                error_12m_w2v,
                                                time_window_ph,
                                                alpha=0.05,
                                                n_permutations=10000,
                                                color_1='purple',
                                                color_2='purple',
                                                linestyle_1='dashed',
                                                linestyle_2='solid',
                                                sig_1_options={'color': 'purple', 'alpha': 0.8, 'markerfacecolor':'None', 'marker_shape':'d', 'markeredgecolor':'purple', 'markeredgewidth':1},
                                                sig_2_options={'color': 'purple', 'alpha': 1, 'markerfacecolor':None, 'marker_shape':'o', 'markeredgecolor':'purple', 'markeredgewidth':None},
                                                figsize=(10, 10),
                                                label_1=f'12m ph fixed seed {seed_range}',
                                                label_2=f'12m w2v fixed seed {seed_range}',
                                                xlabel='Time (s)',
                                                ylabel='2 vs 2 Accuracy',
                                                title=f'{fig_title}',
                                                seed=42)
    file_name = f'from_npz_12m_ph_vs_w2v_fixed_seed_range_{seed_range}_with_clusters_ph_w2v_dots_from_raw_using_fdr_kde_all_over'
    plt.savefig(f'{file_name}.png', bbox_inches='tight')
    # # plt.savefig(f'{file_name}.svg', format='svg', bbox_inches='tight')