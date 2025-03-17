from mne.stats import fdr_correction
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde
import glob
from tqdm import tqdm


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





def average_fold_accs(data):
    result = {}
    for key, val in data.items():
        result[key] = np.mean(val)

    return result


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

def get_significance(path, y_graph, x_graph, p_val_thresh=0.01, kde_per_window=False, do_kde=True):
    # Get significance dots for 9m.
    all_acc_waves = []
    all_acc_waves_no_mean = {}
    for file_index, f in enumerate(tqdm(glob.glob(path + "/*.npz"))):
        perm_accs = np.load(f, allow_pickle=True)['arr_0'].tolist()[0] # Get the first element because the dictionary only contains one element.
        for window, acc in perm_accs.items():
            all_acc_waves_no_mean[window] = acc if window not in all_acc_waves_no_mean else all_acc_waves_no_mean[window] + acc
            all_acc_waves.append(np.mean(acc))
    reject_fdr, threshold_fdr, p_values_fdr, p_values  = get_significance_dots_from_raw(y_graph, all_acc_waves, all_acc_waves_no_mean, p_val_thresh=p_val_thresh, kde_per_window=kde_per_window, do_kde=do_kde)
    print("Reject fdr: ", reject_fdr)
    print(reject_fdr)
    x_dots = x_graph[reject_fdr]
    return x_dots, reject_fdr, threshold_fdr, p_values_fdr, p_values
