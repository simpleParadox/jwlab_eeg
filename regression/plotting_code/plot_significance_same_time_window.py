from curses.ascii import alt
from turtle import color
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
import glob
from tqdm import tqdm



def plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name):
    """
    This function is used to plot the image with the significance dots.
    """
    # The x_graph is the x-axis values.
    # The y_graph is the y-axis values.
    # The error is the error bars.
    # The x_dots are the significance dots that should be plotted at y-value = 0.35.

    fig, ax = plt.subplots()
    # Set ylim.
    ax.set_ylim(0.3, 0.8)
    ax.plot(x_graph, y_graph,'k-', label='Accuracy')
    ax.fill_between(x_graph, y_graph - error, y_graph + error, color='#58c0fc')
    ax.plot(x_dots, [0.35] * len(x_dots), 'ro', label='Significance')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.axhline(y=0.5, color='black', linestyle='--')
    # Set xticks.
    ax.set_xticks(np.arange(-200, 1001, 200), ['-200', '0', '200', '400', '600', '800', '1000'])
    acc_at_zero = y_graph[np.where(x_graph == 0)[0][0]]
    ax.text(700, 0.35, f"Acc at 0ms: {acc_at_zero}", fontsize=12, ha='right')
    plt.savefig(f'{file_name}.png')
    

def get_significance_dots_from_raw(scoreMean, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True):

    """
    This function is used to get the significance dots from the raw results.
    :param results: The raw results from the model.
    :param null_h_raw_path: The path to the null distribution.
    :return: The significance dots.
    """
    print("This function applies KDE on the null distribution and then calculates the p-values.")
    if type(scoreMean) == list:
        scoreMean = np.array(scoreMean)
    pvalues = np.ones(scoreMean.shape)  # This is 1-d array of p-values.
    print("pvalues shape: ", pvalues.shape)
    if all_acc_waves == []:
        return
    alpha_level = p_val_thresh
    ptests = all_acc_waves
    # ptests = np.ndarray.flatten(np.array(ptests))
    if do_kde:
        if kde_per_window:
            # Do the kde per window.
            for window, acc in all_acc_waves_no_mean.items():
                kde = stats.gaussian_kde(acc)
                pv = kde.integrate_box_1d(scoreMean[window], 1)
                if pv == 0:
                    pv = 1 / len(acc)
                pvalues[window] = pv
            # FDR Correction.
            reject_fdr, pvalues_fdr = fdr_correction(np.squeeze(pvalues), alpha=alpha_level, method='negcorr')
            if np.sum(reject_fdr) == 0:  # no significant acc
                threshold_fdr = 100
            else:
                threshold_fdr = np.min(np.abs(scoreMean)[reject_fdr])

            # import pdb; pdb.set_trace()
            
            return reject_fdr, threshold_fdr, pvalues_fdr, pvalues


        else:
            # NOTE: You should be using this one - kde_per_window=False.
            # This is because when computation is a limiting factor, less values are present.
            kde = stats.gaussian_kde(ptests) # Get the KDE over all the values for all the windows together. This is sort of like a hack given computational requirements.
            for trainTime in range(pvalues.shape[0]):
                print(trainTime)
                pv = kde.integrate_box_1d(scoreMean[trainTime], 1)

                #             print(pv)
                # if pv == 0:
                #     pv = 1 / ptests.shape[0]
                    # xs = np.linspace(0, 100, num=100)
                #                 plt.plot(xs, kde(xs))
                #                 plt.show()
                # if X[trainTime, testTime] > 59:
                #     pass
                #                 print(pv, X[trainTime, testTime], trainTime, testTime)
                pvalues[trainTime] = pv
            # print("pvalues: ", pvalues)

            reject_fdr, pvalues_fdr = fdr_correction(pvalues, alpha=alpha_level, method='negcorr')
            if np.sum(reject_fdr) == 0:  # no significant acc
                threshold_fdr = 100
            else:
                threshold_fdr = np.min(np.abs(scoreMean)[reject_fdr])

            return reject_fdr, threshold_fdr, pvalues_fdr, pvalues
    else:
        # NOTE: You should not get p-values without using KDE. That means kde should generally be used.
        # Do not do KDE.
        # Calculate the p-values using the all_acc_waves_no_mean values.

        for window, acc in all_acc_waves_no_mean.items():
            pv = np.sum(np.array(acc) > scoreMean[window]) / len(acc) # Count the number of times the permuted accuracies are greater than the observed accuracy.
            pvalues[window] = pv

        reject_fdr, pvalues_fdr = fdr_correction(pvalues, alpha=alpha_level, method='negcorr')
        if np.sum(reject_fdr) == 0:  # no significant acc
            threshold_fdr = 100
        else:
            threshold_fdr = np.min(np.abs(scoreMean)[reject_fdr])

        return reject_fdr, threshold_fdr, pvalues_fdr, pvalues


def average_fold_accs(data):
    result = {}
    for key, val in data.items():
        result[key] = np.mean(val)

    return result

def get_significance_dots_from_kde(scoreMean, null_h_kde_path):
    null_h =  np.load(null_h_kde_path,allow_pickle=True)['arr_0'].tolist()

    # alt_h_avg = average_fold_accs(alt_h)
    alt_h_avg = scoreMean

    # Only considering the positive windows.

    # """
    # Kernel Density Estimation for smoothing the null_distribution values.
    # """
    # temp = stats.gaussian_kde(h0[0])
    #
    # # Now we perform the significance testing between the 'h1_avg' and 'h0'.
    """
    The p-value is calculated by finding the number of times the permuted accuracies
    are above the observed value (true value).
    The process is done for each window.
    """
    #
    denom = len(null_h[0])  # np.arange(15,len(alt_h_avg)))#len(null_h[0])
    p_values_list = []  # Stores the window based p-values against the null distribution.
    for window in range(len(alt_h_avg)):
        obs_score = alt_h_avg[window]
        permute_scores = null_h[window]
        count = 0
        # Now count how many of the permute scores are >= obs_score.
        for j in range(len(permute_scores)):
            if permute_scores[j] > obs_score:
                count += 1

        p_value = count / denom
        p_values_list.append(p_value)

    # Implementing the Benjamini-Hochberg/Yekutieli correction.
    # First have an index_array just in case.
    idxs = [i for i in range(len(alt_h_avg))]
    
    # Sort the p_values and idxs in ascending order.
    p_vals_list_asc, p_vals_idx_sort = (list(t) for t in zip(*sorted(zip(p_values_list, idxs))))
    p_vals_asc_rank = [i for i in range(len(alt_h_avg))]

    reject, pvals_corrected, alph_sidak, alph_bonf = multipletests(p_vals_list_asc, is_sorted=True, method='fdr_by', alpha=0.01)  # You can change the alpha=0.05 to 0.01 for a stricter condition. Reject is 'rejecting' the null hypothesis - i.e., keeping above chance.

    p_vals_idx_sort = np.array(p_vals_idx_sort)

    length_per_window_plt = 10  # 1200 / len(scoreMean)

    x_graph = np.arange(-200, 910, length_per_window_plt)
    shift = 0
    x_graph += shift  # NOTE: Change this based on the window size.

    dot_idxs = p_vals_idx_sort[reject]
    x_dots = x_graph[dot_idxs]
    print("Significance dots without shift:")
    print(x_dots.tolist())
    return x_dots




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
non_permuted_results_9m_ph_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/ph_9m_fixed_seed.npz'
results_9m_ph_fixed_seed = np.load(non_permuted_results_9m_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
x_graph, y_graph, error = scores_and_error(results_9m_ph_fixed_seed)
print(y_graph)

all_acc_waves = []
all_acc_waves_no_mean = {}
path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/eeg_to_ph/9m"
for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
    perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
    for window, acc in perm_accs.items():
        all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
        all_acc_waves.append(np.mean(acc))
reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
print("Significance dots without shift:")
print(reject_fdr)
print("Length of reject_fdr: ", len(reject_fdr))
# Get the x_dots.
x_dots = x_graph[reject_fdr]
print("Significance dots with shift:")
print(x_dots.tolist())
plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='eeg_to_ph_9m_dots_from_raw_using_fdr_kde_all_over.png')







# ===========================================
# Phoneme for 12m.
non_permuted_results_12m_ph_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/ph_12m_fixed_seed.npz'
results_12m_ph_fixed_seed = np.load(non_permuted_results_12m_ph_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
x_graph, y_graph, error = scores_and_error(results_12m_ph_fixed_seed)
print(y_graph)

all_acc_waves = []
all_acc_waves_no_mean = {}
path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/eeg_to_ph/12m"
for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
    perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
    for window, acc in perm_accs.items():
        all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
        all_acc_waves.append(np.mean(acc))
# import pdb; pdb.set_trace()
reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
print("Significance dots without shift:")
print(reject_fdr)
print("Length of reject_fdr: ", len(reject_fdr))
# Get the x_dots.
x_dots = x_graph[reject_fdr]
print("Significance dots with shift:")
print(x_dots.tolist())
plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='eeg_to_ph_12m_dots_from_raw_using_fdr_kde_all_over.png')







# ===========================================
# w2v for 9m.
non_permuted_results_9m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/w2v_9m_fixed_seed.npz'
results_9m_w2v_fixed_seed = np.load(non_permuted_results_9m_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
x_graph, y_graph, error = scores_and_error(results_9m_w2v_fixed_seed)
print(y_graph)

all_acc_waves = []
all_acc_waves_no_mean = {}
path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/same_time_results/permutation/9m_mod_2v2"
for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
    perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
    for window, acc in perm_accs.items():
        all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
        all_acc_waves.append(np.mean(acc))
# import pdb; pdb.set_trace()
reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
print("Significance dots without shift:")
print(reject_fdr)
print("Length of reject_fdr: ", len(reject_fdr))
# Get the x_dots.
x_dots = x_graph[reject_fdr]
print("Significance dots with shift:")
print(x_dots.tolist())
plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='eeg_to_w2v_9m_dots_from_raw_using_fdr_kde_all_over.png')





# ===========================================
# w2v for 12m.
non_permuted_results_12m_w2v_fixed_seed_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/fixed_seed_results/w2v_12m_fixed_seed.npz'
results_12m_w2v_fixed_seed = np.load(non_permuted_results_12m_w2v_fixed_seed_path, allow_pickle=True)['arr_0'].tolist()
x_graph, y_graph, error = scores_and_error(results_12m_w2v_fixed_seed)
print(y_graph)

all_acc_waves = []
all_acc_waves_no_mean = {}
path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/permuation_test_results/same_time_results/permutation/12m_mod_2v2"
# Only select the first 100 files in this case because we have around 527 perm tests.
count = 0
for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
    perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
    for window, acc in perm_accs.items():
        all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
        all_acc_waves.append(np.mean(acc))
    count += 1
    if count == 99:
        break
# import pdb; pdb.set_trace()
reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=0.01, kde_per_window=False, do_kde=True)
print("Significance dots without shift:")
print(reject_fdr)
print("Length of reject_fdr: ", len(reject_fdr))
# Get the x_dots.
x_dots = x_graph[reject_fdr]
print("Significance dots with shift:")
print(x_dots.tolist())
plot_image_same_time_window(x_graph, y_graph, error, x_dots, file_name='eeg_to_w2v_12m_dots_from_raw_using_fdr_kde_all_over.png')